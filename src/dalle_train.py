from config import Config
import torch
from dalle2.dalle2_pytorch import DALLE2, DiffusionPriorNetwork, DiffusionPrior, Unet, Decoder, CLIP


def train_clip():
    pass


def get_precomputed_embeddings():
    pass


def train_diffusion_prior():
    diffusion_prior = None
    return diffusion_prior


def train_decoder():
    decoder = None
    return decoder


def train_full():
    train_clip()
    get_precomputed_embeddings()
    diffusion_prior = train_diffusion_prior()
    decoder = train_decoder()
    return diffusion_prior, decoder


def get_heatmap(diffusion_prior, decoder):
    heatmap = None

    return heatmap


if __name__ == "__main__":
    cfg = Config()
    diffusion_prior, decoder = train_full()
    heatmap = get_heatmap(diffusion_prior, decoder)
