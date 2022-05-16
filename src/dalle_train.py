from config import Config
import torch
from dalle2.train_diffusion_prior import train_diffusion_call
from dalle2.train_clip import train_clip_call
from dalle2.dalle2_pytorch.dalle2_pytorch import DALLE2, Unet, Decoder


def train_clip():
    clip = train_clip_call()
    return clip


def get_precomputed_embeddings():
    clip_image_embeds, clip_text_embeds = None, None

    return clip_image_embeds, clip_text_embeds


def train_diffusion_prior(clip, clip_image_embeds, clip_text_embeds):
    diffusion_prior = train_diffusion_call(clip, clip_image_embeds, clip_text_embeds)
    return diffusion_prior


def train_decoder(clip):
    decoder = None
    return decoder


def train_full():
    clip = train_clip()
    clip_image_embeds, clip_text_embeds = get_precomputed_embeddings()
    diffusion_prior = train_diffusion_prior(clip, clip_image_embeds, clip_text_embeds)
    decoder = train_decoder(clip)
    return diffusion_prior, decoder


def get_heatmap(diffusion_prior, decoder):
    heatmap = None

    return heatmap


if __name__ == "__main__":
    cfg = Config()
    diffusion_prior, decoder = train_full()
    heatmap = get_heatmap(diffusion_prior, decoder)
