import torch
from autoencoder.vocoder import Generator
from autoencoder.configs import AttrDict, HIFIGAN_16K_64

def load_weights_from_ckpt(ckpt, model, old_model_name, new_model_name):
    new_state_dict = dict()
    stdict = ckpt['state_dict']
    for key in stdict.keys():
        delta_key_name = key[len(old_model_name):]
        if old_model_name == key[:len(old_model_name)]:
            new_state_dict[new_model_name + delta_key_name] = stdict[key]
    model.load_state_dict(new_state_dict)
    return new_state_dict


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param

def get_vocoder(config, device):
    config = AttrDict(HIFIGAN_16K_64)
    vocoder = Generator(config)
    vocoder.eval()
    vocoder.remove_weight_norm()
    vocoder.to(device)
    return vocoder


def vocoder_infer(mels, vocoder, lengths=None):
    with torch.no_grad():
        wavs = vocoder(mels).squeeze(1)

    wavs = (wavs.cpu().numpy() * 32768).astype("int16")

    if lengths is not None:
        wavs = wavs[:, :lengths]

    return wavs