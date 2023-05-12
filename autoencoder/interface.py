from autoencoder.autoencoder import AutoencoderKL
from autoencoder.configs import VAE_CONFIG
from autoencoder.utils import load_weights_from_ckpt

import torch

def load_vae(path_to_ckpt):
    vae = AutoencoderKL(VAE_CONFIG, embed_dim = 8)
    ckpt = torch.load(path_to_ckpt)
    load_weights_from_ckpt(ckpt, vae, 'first_stage_model.', '')
    return vae