import os
import numpy as np
import time
from config import Config
import wandb
import torch
from torch import nn
from embedding_reader import EmbeddingReader
from dalle.dalle2_pytorch import DiffusionPrior, DiffusionPriorNetwork
from dalle.train import load_diffusion_model, save_diffusion_model, print_ribbon
from dalle.optimizer import get_optimizer
from torch.cuda.amp import autocast, GradScaler

os.environ["WANDB_SILENT"] = "true"
NUM_TEST_EMBEDDINGS = 100  # for cosine similarity reporting during training
REPORT_METRICS_EVERY = 100  # for cosine similarity and other metric reporting during training


def eval_model(model, device, image_reader, text_reader, start, end, batch_size, loss_type, phase="Validation"):
    model.eval()
    with torch.no_grad():
        total_loss = 0.
        total_samples = 0.

        for emb_images, emb_text in zip(image_reader(batch_size=batch_size, start=start, end=end),
                                        text_reader(batch_size=batch_size, start=start, end=end)):
            emb_images_tensor = torch.tensor(emb_images[0]).to(device)
            emb_text_tensor = torch.tensor(emb_text[0]).to(device)

            batches = emb_images_tensor.shape[0]

            loss = model(text_embed=emb_text_tensor, image_embed=emb_images_tensor)

            total_loss += loss.item() * batches
            total_samples += batches

        avg_loss = (total_loss / total_samples)
        wandb.log({f'{phase} {loss_type}': avg_loss})


def report_cosine_sims(diffusion_prior, image_reader, text_reader, train_set_size, NUM_TEST_EMBEDDINGS, device):
    diffusion_prior.eval()

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    tstart = train_set_size
    tend = train_set_size + NUM_TEST_EMBEDDINGS

    for embt, embi in zip(text_reader(batch_size=NUM_TEST_EMBEDDINGS, start=tstart, end=tend),
                          image_reader(batch_size=NUM_TEST_EMBEDDINGS, start=tstart, end=tend)):
        # make a copy of the text embeddings for shuffling
        text_embed = torch.tensor(embt[0]).to(device)
        text_embed_shuffled = text_embed.clone()
        # roll the text embeddings to simulate "unrelated" captions
        rolled_idx = torch.roll(torch.arange(NUM_TEST_EMBEDDINGS), 1)
        text_embed_shuffled = text_embed_shuffled[rolled_idx]
        text_embed_shuffled = text_embed_shuffled / \
                              text_embed_shuffled.norm(dim=1, keepdim=True)
        test_text_shuffled_cond = dict(text_embed=text_embed_shuffled)
        # prepare the text embedding
        text_embed = text_embed / text_embed.norm(dim=1, keepdim=True)
        test_text_cond = dict(text_embed=text_embed)
        # prepare image embeddings
        test_image_embeddings = torch.tensor(embi[0]).to(device)
        test_image_embeddings = test_image_embeddings / \
                                test_image_embeddings.norm(dim=1, keepdim=True)
        # predict on the unshuffled text embeddings
        predicted_image_embeddings = diffusion_prior.p_sample_loop(
            (NUM_TEST_EMBEDDINGS, 768), text_cond=test_text_cond)
        predicted_image_embeddings = predicted_image_embeddings / \
                                     predicted_image_embeddings.norm(dim=1, keepdim=True)
        # predict on the shuffled embeddings
        predicted_unrelated_embeddings = diffusion_prior.p_sample_loop(
            (NUM_TEST_EMBEDDINGS, 768), text_cond=test_text_shuffled_cond)
        predicted_unrelated_embeddings = predicted_unrelated_embeddings / \
                                         predicted_unrelated_embeddings.norm(dim=1, keepdim=True)
        # calculate similarities
        original_similarity = cos(
            text_embed, test_image_embeddings).cpu().numpy()
        predicted_similarity = cos(
            text_embed, predicted_image_embeddings).cpu().numpy()
        unrelated_similarity = cos(
            text_embed, predicted_unrelated_embeddings).cpu().numpy()
        predicted_img_similarity = cos(
            test_image_embeddings, predicted_image_embeddings).cpu().numpy()
        wandb.log({"CosineSimilarity(text_embed,image_embed)": np.mean(original_similarity),
                   "CosineSimilarity(text_embed,predicted_image_embed)": np.mean(predicted_similarity),
                   "CosineSimilarity(orig_image_embed,predicted_image_embed)": np.mean(predicted_img_similarity),
                   "CosineSimilarity(text_embed,predicted_unrelated_embed)": np.mean(unrelated_similarity),
                   "Cosine similarity difference": np.mean(predicted_similarity - original_similarity)})


def train_diffusion_prior(image_embed_dim,
                          image_embed_url,
                          text_embed_url,
                          batch_size,
                          train_percent,
                          val_percent,
                          test_percent,
                          num_epochs,
                          dp_loss_type,
                          clip,
                          dp_condition_on_text_encodings,
                          dp_timesteps,
                          dp_normformer,
                          dp_cond_drop_prob,
                          dpn_depth,
                          dpn_dim_head,
                          dpn_heads,
                          save_interval,
                          save_path,
                          device,
                          RESUME,
                          DPRIOR_PATH,
                          config,
                          wandb_entity,
                          wandb_project,
                          learning_rate=0.001,
                          max_grad_norm=0.5,
                          weight_decay=0.01,
                          dropout=0.05,
                          amp=False):
    # DiffusionPriorNetwork
    prior_network = DiffusionPriorNetwork(
        dim=image_embed_dim,
        depth=dpn_depth,
        dim_head=dpn_dim_head,
        heads=dpn_heads,
        attn_dropout=dropout,
        ff_dropout=dropout,
        normformer=dp_normformer).to(device)

    # DiffusionPrior with text embeddings and image embeddings pre-computed
    diffusion_prior = DiffusionPrior(
        net=prior_network,
        clip=clip,
        image_embed_dim=image_embed_dim,
        timesteps=dp_timesteps,
        cond_drop_prob=dp_cond_drop_prob,
        loss_type=dp_loss_type,
        condition_on_text_encodings=dp_condition_on_text_encodings).to(device)

    # Load pre-trained model from DPRIOR_PATH
    if RESUME:
        diffusion_prior = load_diffusion_model(DPRIOR_PATH, device)
        wandb.init(entity=wandb_entity, project=wandb_project, config=config)

        # Create save_path if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Get image and text embeddings from the servers
    print_ribbon("Downloading embeddings - image and text")
    image_reader = EmbeddingReader(embeddings_folder=image_embed_url, file_format="npy")
    text_reader = EmbeddingReader(embeddings_folder=text_embed_url, file_format="npy")
    num_data_points = text_reader.count

    ### Training code ###
    scaler = GradScaler(enabled=amp)
    optimizer = get_optimizer(diffusion_prior.net.parameters(), wd=weight_decay, lr=learning_rate)
    epochs = num_epochs

    step = 0
    t = time.time()

    train_set_size = int(train_percent * num_data_points)
    val_set_size = int(val_percent * num_data_points)
    eval_start = train_set_size

    for _ in range(epochs):

        for emb_images, emb_text in zip(image_reader(batch_size=batch_size, start=0, end=train_set_size),
                                        text_reader(batch_size=batch_size, start=0, end=train_set_size)):

            diffusion_prior.train()

            emb_images_tensor = torch.tensor(emb_images[0]).to(device)
            emb_text_tensor = torch.tensor(emb_text[0]).to(device)

            with autocast(enabled=amp):
                loss = diffusion_prior(text_embed=emb_text_tensor, image_embed=emb_images_tensor)
                scaler.scale(loss).backward()

            # Samples per second
            step += 1
            samples_per_sec = batch_size * step / (time.time() - t)
            # Save checkpoint every save_interval minutes
            if (int(time.time() - t) >= 60 * save_interval):
                t = time.time()

                save_diffusion_model(
                    save_path,
                    diffusion_prior,
                    optimizer,
                    scaler,
                    config,
                    image_embed_dim)

            # Log to wandb
            wandb.log({"Training loss": loss.item(),
                       "Steps": step,
                       "Samples per second": samples_per_sec})
            # Log cosineSim(text_embed,predicted_image_embed) - cosineSim(text_embed,image_embed)
            # Use NUM_TEST_EMBEDDINGS samples from the test set each time
            # Get embeddings from the most recently saved model
            if (step % REPORT_METRICS_EVERY) == 0:
                report_cosine_sims(diffusion_prior,
                                   image_reader,
                                   text_reader,
                                   train_set_size,
                                   NUM_TEST_EMBEDDINGS,
                                   device)
                ### Evaluate model(validation run) ###
                eval_model(diffusion_prior,
                           device,
                           image_reader,
                           text_reader,
                           eval_start,
                           eval_start + NUM_TEST_EMBEDDINGS,
                           NUM_TEST_EMBEDDINGS,
                           dp_loss_type,
                           phase="Validation")

            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(diffusion_prior.parameters(), max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

    ### Test run ###
    test_set_size = int(test_percent * train_set_size)
    start = train_set_size + val_set_size
    end = num_data_points
    eval_model(diffusion_prior, device, image_reader, text_reader, start, end, batch_size, dp_loss_type, phase="Test")
    return diffusion_prior


def train_diffusion_call(clip, clip_image_embeds, clip_text_embeds):
    cfg = Config()

    RESUME = False

    if (cfg.pretrained_model_path is not None):
        RESUME = True
    else:
        wandb.init(
            entity=cfg.wandb_diffusion_entity,
            project=cfg.wandb_diffusion_project,
            config=cfg.wandb_diffusion_config)

    device = "cpu"
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    # Training loop
    diffusion_prior = train_diffusion_prior(cfg.image_embed_dim,
                                            cfg.image_embed_url,
                                            cfg.text_embed_url,
                                            cfg.batch_size,
                                            cfg.train_percent,
                                            cfg.val_percent,
                                            cfg.test_percent,
                                            cfg.num_epochs,
                                            cfg.dp_loss_type,
                                            cfg.clip,
                                            cfg.dp_condition_on_text_encodings,
                                            cfg.dp_timesteps,
                                            cfg.dp_normformer,
                                            cfg.dp_cond_drop_prob,
                                            cfg.dpn_depth,
                                            cfg.dpn_dim_head,
                                            cfg.dpn_heads,
                                            cfg.save_interval,
                                            cfg.save_path,
                                            device,
                                            RESUME,
                                            cfg.pretrained_model_path,
                                            cfg.wandb_config,
                                            cfg.wandb_entity,
                                            cfg.wandb_project,
                                            cfg.learning_rate,
                                            cfg.max_grad_norm,
                                            cfg.weight_decay,
                                            cfg.dropout,
                                            cfg.amp)

    return diffusion_prior


if __name__ == "__main__":
    clip, clip_image_embeds, clip_text_embeds = None, None, None
    train_diffusion_call(clip, clip_image_embeds, clip_text_embeds)
