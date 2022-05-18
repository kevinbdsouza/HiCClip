import os
import math
import argparse
import numpy as np
import time
from tqdm import tqdm
from config import Config
import wandb
import torch
from torch import nn
from embedding_reader import EmbeddingReader
from dalle2.dalle2_pytorch.dalle2_pytorch import CLIP
from dalle2.dalle2_pytorch.train import load_clip_model, save_clip_model, print_ribbon
from dalle2.dalle2_pytorch.optimizer import get_optimizer
from torch.cuda.amp import autocast, GradScaler

os.environ["WANDB_SILENT"] = "true"


def train_test_eval_split(pairpos_batched, maps_batched):
    total_batches = len(maps_batched)
    train_batches = int(0.7 * total_batches)
    eval_batches = int(0.2 * total_batches)

    train_pos = pairpos_batched[:train_batches]
    train_maps = maps_batched[:train_batches]

    eval_pos = pairpos_batched[train_batches:train_batches + eval_batches]
    eval_maps = maps_batched[train_batches:train_batches + eval_batches]

    test_pos = pairpos_batched[train_batches + eval_batches:]
    test_maps = maps_batched[train_batches + eval_batches:]
    return train_pos, train_maps, eval_pos, eval_maps, test_pos, test_maps


def eval_model(model, device, maps_batched, pos_batched, phase="Validation"):
    model.eval()
    with torch.no_grad():
        total_loss = 0.
        total_samples = 0.

        for maps, pos in zip(maps_batched, pos_batched):
            maps_tensor = torch.tensor(maps).to(device)
            pos_tensor = torch.tensor(pos).to(device)

            batch_samples = maps_tensor.shape[0]

            loss = model(text_embed=pos_tensor, image_embed=maps_tensor)

            total_loss += loss.item() * batch_samples
            total_samples += batch_samples

        avg_loss = (total_loss / total_samples)
        wandb.log({f'{phase}': avg_loss})


def train_clip(device, resume, cfg):
    """Train Clip model"""

    clip = CLIP(
        dim_text=cfg.dim_text,
        dim_image=cfg.dim_image,
        dim_latent=cfg.dim_latent,
        num_text_tokens=cfg.num_text_tokens,
        text_enc_depth=cfg.text_enc_depth,
        text_seq_len=cfg.text_seq_len,
        text_heads=cfg.text_heads,
        visual_enc_depth=cfg.visual_enc_depth,
        visual_image_size=cfg.visual_image_size,
        visual_patch_size=cfg.visual_patch_size,
        visual_heads=cfg.visual_heads,
        use_all_token_embeds=cfg.use_all_token_embeds,
        decoupled_contrastive_learning=cfg.decoupled_contrastive_learning,
        extra_latent_projection=cfg.extra_latent_projection,
        use_visual_ssl=cfg.use_visual_ssl,
        visual_ssl_type=cfg.visual_ssl_type,
        use_mlm=cfg.use_mlm,
        text_ssl_loss_weight=cfg.text_ssl_loss_weight,
        image_ssl_loss_weight=cfg.image_ssl_loss_weight).to(device)

    # Load pre-trained model from DPRIOR_PATH
    if resume:
        clip = load_clip_model(cfg.pretrained_clip_model_path, device)
        wandb.init(entity=cfg.wandb_clip_entity, project=cfg.wandb_clip_project, config=cfg.wandb_clip_config)

    # Create save_path if it doesn't exist
    if not os.path.exists(cfg.save_path_clip):
        os.makedirs(cfg.save_path_clip)

    ### Training code ###
    scaler = GradScaler(enabled=cfg.amp)
    optimizer = get_optimizer(clip.parameters(), wd=cfg.wandb_clip_config.weight_decay,
                              lr=cfg.wandb_clip_config.learning_rate)
    epochs = cfg.wandb_clip_config.epochs

    step = 0
    t = time.time()

    for _ in range(epochs):

        for chr in cfg.chr_train_list:

            pairpos_batched = np.load(cfg.batched_hic_path + "embed_%s.npy" % chr)
            maps_batched = np.load(cfg.batched_hic_path + "hic_%s.npy" % chr)

            train_pos, train_maps, eval_pos, eval_maps, test_pos, test_maps = train_test_eval_split(pairpos_batched,
                                                                                                    maps_batched)

            for pairpos, maps in zip(train_pos, train_maps):

                clip.train()

                pairpos_tensor = torch.tensor(pairpos).to(device)
                maps_tensor = torch.tensor(maps).to(device)

                with autocast(enabled=cfg.amp):
                    loss = clip(text_embed=pairpos_tensor, image_embed=maps_tensor)
                    scaler.scale(loss).backward()

                "samples per second"
                step += 1
                samples_per_sec = cfg.wandb_clip_config.batch_size * step / (time.time() - t)

                "save checkpoint every save_interval minutes"
                if (int(time.time() - t) >= 60 * cfg.save_interval):
                    t = time.time()

                    save_clip_model(cfg.save_path_clip, clip, optimizer, scaler, cfg.wandb_clip_config)

                "log to wandb"
                wandb.log({"Training loss": loss.item(),
                           "Steps": step,
                           "Samples per second": samples_per_sec})

                if (step % cfg.report_metrics_every) == 0:
                    eval_model(clip, device, eval_maps, eval_pos, phase="Validation")

                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(clip.parameters(), cfg.wandb_clip_config.max_gradient_clipping_norm)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            "Test run"
            eval_model(clip, device, test_maps, test_pos, phase="Test")
    return clip


def train_clip_call():
    cfg = Config()

    resume = False

    if (cfg.pretrained_clip_model_path is not None):
        resume = True
    else:
        wandb.init(
            entity=cfg.wandb_clip_entity,
            project=cfg.wandb_clip_project,
            config=cfg.wandb_clip_config)

    device = "cpu"
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    # Training loop
    clip = train_clip(device, resume, cfg)
    return clip


if __name__ == "__main__":
    train_clip_call()
