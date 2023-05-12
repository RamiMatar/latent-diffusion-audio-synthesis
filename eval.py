import os
import torch
import torchvision
import torchaudio
from diffusion import LatentDiffusion, args
from diffusers import DDPMScheduler
from diffusers import UNet2DConditionModel
from autoencoder.utils import load_weights_from_ckpt
from diffusion import disabled_train
from autoencoder.interface import load_vae
import json
from diffusion import args

class SampleGenerator:
    def __init__(self, ckpt_path, output_dir, device='cuda:0'):
        self.ldm = LatentDiffusion(args).to(device)
        self.device = device
        self.init_models(ckpt_path)
        self.output_dir = output_dir
        

    def init_models(self, ckpt_path):
        load_weights_from_ckpt(torch.load(ckpt_path), self.ldm.model, 'model.', '')
        load_weights_from_ckpt(torch.load(ckpt_path), self.ldm.ema_model, 'ema_model.', '')
        self.ldm.vae = load_vae(args.vae_ckpt_path)
        self.ldm.vae = self.ldm.vae.to(self.device)
        self.ldm.vae.eval()
        for param in self.ldm.vae.parameters():
            param.requires_grad = False
        self.ldm.vae.train = disabled_train

    def generate_samples(self, n, prompts):
        self.ldm.model = self.ldm.model.to(self.device)
        self.ldm.ema_model = self.ldm.ema_model.to(self.device)
        
        for prompt in prompts:
            print("prompt is: ", prompt)
            sampled_mels = self.ldm.sample_n_ddim(self.ldm.model, n, guidance = 12.0, text = prompt, random_noise=True)
            ema_sampled_mels = self.ldm.sample_n_ddim(self.ldm.ema_model, n, guidance = 12.0, text = prompt, random_noise=True)

            for idx in range(n):
                sample = sampled_mels[idx]
                ema_sample = ema_sampled_mels[idx]

                sample = sample.squeeze(1).transpose(1, 2)
                ema_sample = ema_sample.squeeze(1).transpose(1, 2)

                waveform = self.ldm.vae.vocoder(sample).squeeze(1)
                ema_waveform = self.ldm.vae.vocoder(ema_sample).squeeze(1)
                mels = self.ldm.mel(waveform)
                ema_mels = self.ldm.mel(ema_waveform)

                output_dir = os.path.join(self.output_dir, f"prompt_{prompt}")

                output_dir_ema = os.path.join(output_dir, f"ema_sample_{idx + 1}")
                os.makedirs(output_dir_ema, exist_ok=True)

                output_dir_normal = os.path.join(output_dir, f"normal_sample_{idx + 1}")
                os.makedirs(output_dir_normal, exist_ok=True)

                torchvision.utils.save_image(ema_mels[0], os.path.join(output_dir_ema, f"EMA.png"))
                torchvision.utils.save_image(mels[0], os.path.join(output_dir_normal, f"Normal.png"))
                torchaudio.save(os.path.join(output_dir_normal, f"Waveform_Normal.wav"), waveform.detach().cpu(), 16000)
                torchaudio.save(os.path.join(output_dir_ema, f"Waveform_EMA.wav"), ema_waveform.detach().cpu(), 16000)


def main():
    generator = SampleGenerator('epoch=49-step=100000.ckpt', 'prompt samples')
    prompts = ['woman singing and clapping in the background',
               'epic piano song',
               'acoustic guitar']
    prompts_2 = ["high quality happy piano song with a lot of reverb and no background noise", 
               "acoustic guitar playing fast melody", 
               "electric guitar and drums playing a rock song"]
    musical_prompts = [
    "Piano playing high, tinkling notes.",
    "Drums beating a fast, energetic rhythm.",
    "Flute playing a slow, tranquil tune.",
    "Violin producing a somber, melancholic sound.",
    "Electric guitar playing a loud, driving riff.",
    "Cello playing deep, resonating notes.",
    "Harp playing light, delicate arpeggios.",
    "Trumpet playing a jazzy, swing tune.",
    "Orchestra playing a powerful, swelling theme." 
    ]
    generator.generate_samples(3, musical_prompts)


if __name__ == "__main__":
    main()
