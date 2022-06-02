import os
import numpy as np
import time
from config import Config
import random
import torch
from torch import nn
from x_clip import CLIP
from dalle.train import load_clip_model, save_clip_model
from dalle.optimizer import get_optimizer
from torch.cuda.amp import autocast, GradScaler

os.environ["WANDB_SILENT"] = "true"
os.chdir("/home/kevindsouza/Documents/projects/PhD/HiCFold/src")


def eval_model(model, device, maps_batched, pos_batched):
    model.eval()
    with torch.no_grad():
        total_loss = 0.
        total_samples = 0.

        for maps, pos in zip(maps_batched, pos_batched):
            pos_tensor = torch.tensor(np.array(pos)).to(device)
            maps_tensor = torch.tensor(np.array(maps)).unsqueeze(1).to(device)

            batch_samples = maps_tensor.shape[0]

            loss = model(pos_tensor, maps_tensor, freeze_image_encoder=False, return_loss=True)

            total_loss += loss.item() * batch_samples
            total_samples += batch_samples

        avg_loss = (total_loss / total_samples)
    return avg_loss


def train_clip(device, cfg):
    """Train Clip model"""

    "load pre-trained model from DPRIOR_PATH"
    if cfg.exp_resume:
        clip, cfg = load_clip_model(cfg, device)
    else:
        clip = CLIP(
            dim_text=cfg.clip_config["dim_text"],
            dim_image=cfg.clip_config["dim_image"],
            dim_latent=cfg.clip_config["dim_latent"],
            num_text_tokens=cfg.clip_config["num_text_tokens"],
            text_enc_depth=cfg.clip_config["text_enc_depth"],
            text_seq_len=cfg.clip_config["text_seq_len"],
            text_heads=cfg.clip_config["text_heads"],
            visual_enc_depth=cfg.clip_config["visual_enc_depth"],
            visual_image_size=cfg.clip_config["visual_image_size"],
            visual_patch_size=cfg.clip_config["visual_patch_size"],
            visual_heads=cfg.clip_config["visual_heads"],
            use_all_token_embeds=cfg.clip_config["use_all_token_embeds"],
            decoupled_contrastive_learning=cfg.clip_config["decoupled_contrastive_learning"],
            extra_latent_projection=cfg.clip_config["extra_latent_projection"],
            use_visual_ssl=cfg.clip_config["use_visual_ssl"],
            visual_ssl_type=cfg.clip_config["visual_ssl_type"],
            use_mlm=cfg.clip_config["use_mlm"],
            text_ssl_loss_weight=cfg.clip_config["text_ssl_loss_weight"],
            image_ssl_loss_weight=cfg.clip_config["image_ssl_loss_weight"]).to(device)

    batch_indices = np.arange(1, 4533)
    scaler = GradScaler(enabled=cfg.amp)
    optimizer = get_optimizer(clip.parameters(), wd=cfg.optim_config["weight_decay"],
                              lr=cfg.optim_config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.optim_config["scheduler_lr"],
                                                    epochs=cfg.optim_config["epochs"],
                                                    steps_per_epoch=len(batch_indices))

    "create save_path if it doesn't exist"
    if not os.path.exists(cfg.save_path_clip):
        os.makedirs(cfg.save_path_clip)

    epochs = cfg.optim_config["epochs"]
    step = 0

    clip.train()
    for ep in range(epochs):
        epoch_loss = []
        random.shuffle(batch_indices)
        for batch in batch_indices:
            pairpos_batched = np.load(cfg.cross_chromosome_batches + "cross_chr_ind_%s.npy" % batch, allow_pickle=True)
            maps_batched = np.load(cfg.cross_chromosome_batches + "cross_chr_hic_%s.npy" % batch, allow_pickle=True)

            pairpos = np.array(pairpos_batched)
            maps = np.array(maps_batched)

            sample_indices = np.random.permutation(pairpos.shape[0])
            pairpos, maps = pairpos[sample_indices, :], maps[sample_indices, :, :]

            pairpos_tensor = torch.tensor(pairpos).to(device)
            maps_tensor = torch.tensor(maps).unsqueeze(1).to(device)

            with autocast(enabled=cfg.amp):
                loss = clip(pairpos_tensor, maps_tensor, freeze_image_encoder=False, return_loss=True)
                if torch.isnan(loss) or torch.isinf(loss):
                    clip, cfg = load_clip_model(cfg, device)
                    continue
                else:
                    save_clip_model(clip, cfg, cfg.clip_config)

                scaler.scale(loss).backward()

            "runnning loss"
            epoch_loss.append(loss.item())

            step += 1
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(clip.parameters(), cfg.optim_config["max_gradient_clipping_norm"])

            scaler.step(optimizer)
            scale = scaler.get_scale()
            skip_lr_sched = (scale > scaler.get_scale())
            if not skip_lr_sched:
                scheduler.step()

            scaler.update()
            optimizer.zero_grad()

            epoch_loss.append(np.mean(epoch_loss))

        print("############################")
        print("epoch: %s loss: %s" % (ep, np.mean(epoch_loss)))
        print("############################")
    return clip


def train_clip_call():
    cfg = Config()

    device = "cpu"
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    # Training loop
    clip = train_clip(device, cfg)
    return clip


if __name__ == "__main__":
    train_clip_call()
