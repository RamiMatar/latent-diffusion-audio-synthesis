from typing import List
import torch
import torchaudio
import torchaudio.functional as F_audio
import torch.nn.functional as F
import os
import copy
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import torchvision
from tqdm.notebook import tqdm
import pytorch_lightning as pl
from torchvision.utils import make_grid
from argparse import Namespace
from ema import EMA
from diffusers import DDIMScheduler, DDPMScheduler
from unet import UNet
from autoencoder.interface import load_vae
from autoencoder.utils import vocoder_infer
import numpy as np
from data import load_data
from diffusers import AutoencoderKL, UNet2DConditionModel
import json 
from clap import CLAPAudioEmbeddingClassifierFreev2
from util import disabled_train
from preprocess import *
from autoencoder.utils import load_weights_from_ckpt

class LatentDiffusion(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_train_steps = args.num_train_steps
        self.num_inference_steps = args.num_inference_steps
        self.init_diffusion(args)
        self.init_clap_aldm(args)
        self.init_vae(args)
        self.height, self.width = args.mel_size
        self.vae_ratio = args.vae_compression_ratio if args.latent else 1
        self.h, self.w = self.height // self.vae_ratio, self.width // self.vae_ratio
        self.test_noise = torch.randn((args.n_samples_after_epoch, args.latent_dim, self.w, self.h))
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate = args.sample_rate, hop_length = args.hop_length, n_mels = 64, n_fft = args.n_fft, power = 2.0, norm = 'slaney', onesided = True)
        self.losses = []
        if args.ckpt_path:
            self.init_unet_from_ckpt(args.ckpt_path)

    def init_unet_from_ckpt(self, ckpt_path):
        load_weights_from_ckpt(torch.load(ckpt_path), self.model, 'model.', '')
        print("loaded weights from ckpt unet")

    def init_clap_aldm(self, args):
        self.conditional = args.conditional
        self.guidance = args.guidance
        self.p_unconditional = args.p_unconditional
        self.clap = CLAPAudioEmbeddingClassifierFreev2(
            key="waveform",
            sampling_rate=16000,
            embed_mode="audio",
            unconditional_prob=0.1,
        )
        ckpt = torch.load(args.clap_ckpt_path)
        load_weights_from_ckpt(ckpt, self.clap, 'cond_stage_model.', '')
        self.clap.eval()
        self.clap.train = disabled_train
        for param in self.clap.parameters():
            param.requires_grad = False

    def init_vae(self, args):
        self.vae = load_vae(args.vae_ckpt_path)
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False
        self.vae.train = disabled_train
        
    def init_diffusion(self, args):
        self.diffuser = DDPMScheduler(
            num_train_timesteps = self.num_train_steps,
            beta_start = 0.0015,
            beta_end = 0.0195,
            clip_sample = True,
            prediction_type = 'epsilon',
        )
        self.latent = args.latent
        self.ema = EMA(0.995)
        with open("unet_config.json", "r") as f:
            unet_config = json.load(f)
        self.model = UNet2DConditionModel(**unet_config)
        self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)
        

    def forward(self, x, t):
        return None
    
  
    def sample_n_ddim(self, model, n, guidance = 3.0, text = "Smooth violin", random_noise = False, train = True):
        model.eval().to(self.device)
        if random_noise:
            noise = torch.randn((n, 8, self.w, self.h)).to(self.device)
        else:
            noise = self.test_noise.to(self.device)

        self.clap.embed_mode = "text"
        uncond_embedding = self.clap([text]).squeeze(0).repeat(n,1)
        if text != "":
            embedding = self.clap(text).squeeze(0).repeat(n,1)
            embedding = torch.cat([uncond_embedding, embedding], dim = 0)
            
        else:
            embedding = uncond_embedding
        self.clap.embed_mode = "audio"
        
        input = noise
        for t in self.diffuser.timesteps:
            with torch.no_grad():
                if text != "":
                    input = torch.cat([noise, noise], dim = 0) 
                    noise_prediction = model(input, t, encoder_hidden_states = None, class_labels = embedding).sample
                    noise_pred_uncond, noise_pred_text = noise_prediction.chunk(2)
                    noise_prediction = noise_pred_uncond + guidance * (noise_pred_text - noise_pred_uncond)
                else:
                    input = noise
                    noise_prediction = model(input, t, encoder_hidden_states = None, class_labels = embedding).sample
                prev_noisy_sample = self.diffuser.step(noise_prediction, t, noise).prev_sample
                
                noise = prev_noisy_sample
        if self.latent:
            output = self.vae.decode(input)
        else:
            output = input
        if train:
            model.train()
        return output
    
    def configure_optimizers(self):
        return optim.AdamW(self.model.parameters(), lr=self.args.lr)

    def training_step(self, batch, batch_idx):
        waveforms, mels = batch
        t = torch.randint(low=1, high=self.num_train_steps, size=(waveforms.shape[0],)).to(self.device)
        if not self.conditional or np.random.random() < self.p_unconditional:
            c = None
        else:
            c = self.clap(waveforms).squeeze(1)
        mels = mels.unsqueeze(1)
        if self.latent:
            encoding = self.vae.encode(mels).sample()
            if torch.max(torch.abs(encoding)) > 10:
                encoding = torch.clip(encoding, -10, 10)
        else:
            encoding = mels
        noise = torch.randn_like(encoding)
        x_t = self.diffuser.add_noise(encoding, noise, t)
        predicted_noise = self.model(x_t, 
                                     t, 
                                     encoder_hidden_states = None,
                                    class_labels = c
                                    ).sample
        loss = F.mse_loss(noise, predicted_noise)

        # Log
        if batch_idx % self.args.log_interval == 0:
            denoised = self.diffuser.step(predicted_noise[0], t[0].to('cpu'), x_t[0])
            mels_noised = self.vae.decode(x_t).squeeze(1).transpose(1,2)
            mels_x0_recon = self.vae.decode(denoised.pred_original_sample.unsqueeze(0)).squeeze(1).transpose(1,2)
            mels_xt_1_recon = self.vae.decode(denoised.prev_sample.unsqueeze(0)).squeeze(1).transpose(1,2)
            mels = mels.squeeze(1).transpose(1,2)
            waveform_noised = self.vae.vocoder(mels_noised[0]).squeeze(1)
            waveform = self.vae.vocoder(mels[0]).squeeze(1)
            x_t_minus_one_waveform = self.vae.vocoder(mels_xt_1_recon[0]).squeeze(1)
            denoised_waveform = self.vae.vocoder(mels_x0_recon[0]).squeeze(1)
            self.logger.experiment.add_audio("Train/Waveform_Noised", waveform_noised[0], self.trainer.global_step, sample_rate = self.args.sample_rate)
            self.logger.experiment.add_audio("Train/Waveform_Original", waveform[0], self.trainer.global_step, sample_rate = self.args.sample_rate)
            self.logger.experiment.add_audio("Train/Waveform_x0_Recon", denoised_waveform[0], self.trainer.global_step, sample_rate = self.args.sample_rate)
            self.logger.experiment.add_audio("Train/Waveform_xt-1_Recon", x_t_minus_one_waveform[0], self.trainer.global_step, sample_rate = self.args.sample_rate)

            mels_noised = self.mel(waveform_noised[0].unsqueeze(0))
            mels_x0_recon = self.mel(denoised_waveform[0].unsqueeze(0))
            mels_xt_1_recon = self.mel(x_t_minus_one_waveform[0].unsqueeze(0))
            mels = self.mel(waveform[0].unsqueeze(0))
            # Create subdirectory for current epoch

            self.logger.experiment.add_image("Train/Mels_Noised", mels_noised[0], self.trainer.global_step, dataformats = 'HW')
            self.logger.experiment.add_image("Train/Mels_Original", mels[0], self.trainer.global_step, dataformats = 'HW')
            self.logger.experiment.add_image("Train/Mels_x_0_Recon", mels_x0_recon[0], self.trainer.global_step, dataformats = 'HW')
            self.logger.experiment.add_image("Train/Mels_x_t-1_Recon", mels_xt_1_recon[0], self.trainer.global_step, dataformats = 'HW')
            
            output_dir = os.path.join(self.args.output_dir, f"epoch_{self.current_epoch}", f"step_{batch_idx}")
            os.makedirs(output_dir, exist_ok=True)
            # Save MELs and Waveforms
            torchvision.utils.save_image(mels_noised[0], os.path.join(output_dir, f"Mels_Noised.png"))
            torchvision.utils.save_image(mels[0], os.path.join(output_dir, f"Mels_Original.png"))
            torchvision.utils.save_image(mels_x0_recon[0], os.path.join(output_dir, f"Mels_x0_Recon.png"))
            torchvision.utils.save_image(mels_xt_1_recon[0], os.path.join(output_dir, f"Mels_xt-1_Recon.png"))
            torchaudio.save(os.path.join(output_dir, f"Waveform_Noised.wav"), waveform_noised[0].unsqueeze(0).detach().cpu(), 16000)
            torchaudio.save(os.path.join(output_dir, f"Waveform_Original.wav"), waveform[0].unsqueeze(0).detach().cpu() , 16000)
            torchaudio.save(os.path.join(output_dir, f"Waveform_x0_Recon.wav"), denoised_waveform[0].unsqueeze(0).detach().cpu(), 16000)
            torchaudio.save(os.path.join(output_dir, f"Waveform_xt-1_Recon.wav"), x_t_minus_one_waveform[0].unsqueeze(0).detach().cpu(), 16000)
        
        self.log("Train/MSE", loss)
        self.log("mse", loss, prog_bar=True, logger=False)
        self.losses.append(loss)
        return {"loss": loss}

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.ema.step_ema(self.model, self.ema_model)



    def on_train_epoch_end(self):
        n = 3
        print(sum(self.losses)/len(self.losses))
    #  self.diffuser.set_timesteps(self.num_inference_steps)
        self.losses = []
        if self.device == torch.device('cuda:0'):
            with torch.no_grad():
                sampled_mels = self.sample_n_ddim(self.model, n)
                ema_sampled_mels = self.sample_n_ddim(self.ema_model, n, train=False)
                # Save images of mel spectrograms and waveforms to disk
            for idx in range(n):
                sample = sampled_mels[idx]
                ema_sample = ema_sampled_mels[idx]
                sample = sample.squeeze(1).transpose(1,2)
                print(sample.shape)
                ema_sample = ema_sample.squeeze(1).transpose(1,2)

                # Convert to waveform and back to mel
                waveform = self.vae.vocoder(sample).squeeze(1)
                ema_waveform = self.vae.vocoder(ema_sample).squeeze(1)
                mels = self.mel(waveform)
                ema_mels = self.mel(ema_waveform)

                output_dir = os.path.join(self.args.output_dir, f"epoch_{self.current_epoch}")
                os.makedirs(output_dir, exist_ok=True)

                output_dir_ema = os.path.join(output_dir, f"ema_sample_{idx+1}")
                os.makedirs(output_dir_ema, exist_ok=True)

                output_dir_normal = os.path.join(output_dir, f"normal_sample_{idx+1}")
                os.makedirs(output_dir_normal, exist_ok=True)

                torchvision.utils.save_image(ema_mels[0], os.path.join(output_dir_ema, f"EMA.png"))
                torchvision.utils.save_image(mels[0], os.path.join(output_dir_normal, f"Normal.png"))
                torchaudio.save(os.path.join(output_dir_normal, f"Waveform_Normal.wav"), waveform.detach().cpu() , 16000)
                torchaudio.save(os.path.join(output_dir_ema, f"Waveform_EMA.wav"), ema_waveform.detach().cpu(), 16000)



    def validation_step(self, batch, batch_idx):
        waveforms, metadata = batch
        t = torch.randint(low=1, high=self.num_train_steps, size=(waveforms.shape[0],)).to(self.device)
        mels = self.mel(waveforms)
        if not self.conditional or np.random.random() < self.p_uncoditional:
            c = None
        else:
            c = self.clap.get_audio_embedding_from_data(x = waveforms, use_tensor=True)
        if self.latent:
            encoding = self.vae.encode(mels)
        else:
            encoding = mels
        x_t, noise = self.forward(encoding, t)
        predicted_noise = self.model(x_t, t, c)
        loss = F.mse_loss(noise, predicted_noise)

        # Log images
        if batch_idx % self.args.log_interval == 0:
            denoised = self.denoise(x_t, t, predicted_noise)
            self.logger.experiment.add_image("Val/Noised", x_t[0], self.trainer.global_step, dataformats = 'CHW')
            self.logger.experiment.add_image("Val/Denoised", denoised[0], self.trainer.global_step, dataformats = 'CHW')
            self.logger.experiment.add_image("Val/Original", mels[0], self.trainer.global_step, dataformats = 'CHW')
            self.logger.experiment.add_image("Val/Predicted Noise", predicted_noise[0], self.trainer.global_step, dataformats = 'CHW')

        self.log("Val/MSE", loss)
        self.log("val_mse", loss, prog_bar=True, logger=False)

    def on_fit_start(self, *args, **kwargs):
        super().on_fit_start(*args, **kwargs)
      #  self.diffuser.set_timesteps(self.num_train_steps)
     #   self.alpha = self.alpha.to(self.device)
     #   self.beta = self.beta.to(self.device)
     #   self.alpha_hat = self.alpha_hat.to(self.device)
        
    def train_dataloader(self):
        return load_data(self.args.dataset_path, batch_size = self.args.batch_size, shuffle = True)

    def val_dataloader(self):
        return load_data(self.args.dataset_path, batch_size = self.args.batch_size, shuffle = True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Latent Diffusion Model for text-to-audio generation")
    parser.add_argument("--run_name", default="ldm", type=str, help="Run name")
    parser.add_argument("--epochs", default=400, type=int, help="Number of epochs")
    parser.add_argument("--batch_size", default=4, type=int, help="Batch size")
    parser.add_argument("--cond_embed_sample_rate", default=48000, type=int, help="Sample rate for conditioning embeddings")
    parser.add_argument("--dataset_path", default="fma_large", type=str, help="Dataset path")
    parser.add_argument("--val_dataset_path", default="bigdataset", type=str, help="Validation dataset path")
    parser.add_argument("--device", default="cuda", type=str, help="Device to use (e.g., 'cuda' or 'cpu')")
    parser.add_argument("--ckpt_path", default="tta_checkpoint.ckpt", type=str, help="Checkpoint path")
    parser.add_argument("--vae_ckpt_path", default="tta_checkpoint.ckpt", type=str, help="VAE checkpoint path")
    parser.add_argument("--clap_ckpt_path", default="tta_checkpoint.ckpt", type=str, help="CLAP checkpoint path")
    parser.add_argument("--vae_name", default="first_stage_model.", type=str, help="Placeholder description")
    parser.add_argument("--lr", default=1e-5, type=float, help="Learning rate")
    parser.add_argument("--val_batch_per_epoch", default=0, type=int, help="Number of validation batches per epoch")
    parser.add_argument("--save_interval", default=100, type=int, help="Save interval")
    parser.add_argument("--log_interval", default=50, type=int, help="Log interval")
    parser.add_argument("--num_workers", default=4, type=int, help="Number of workers")
    parser.add_argument("--output_dir", default="outputssss", type=str, help="Output directory")
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Checkpoint path")
    parser.add_argument("--train_batches_per_epoch", default=2000, type=int, help="Number of training batches per epoch")
    parser.add_argument("--cosine_scheduler", default=False, type=bool, help="Use cosine scheduler")
    parser.add_argument("--conditional", default=True, type=bool, help="Conditional training")
    parser.add_argument("--sample_rate", default=16000, type=int, help="Sample rate")
    parser.add_argument("--hop_length", default=160, type=int, help="Hop length")
    parser.add_argument("--n_fft", default=1024, type=int, help="Number of FFTs")
    parser.add_argument("--n_mels", default=64, type=int, help="Number of Mel frequency bands")
    parser.add_argument("--guidance", default=3.0, type=float, help="Placeholder description")
    parser.add_argument("--n_samples_after_epoch", default=3, type=int, help="Number of samples after epoch")
    parser.add_argument("--p_unconditional", default=0, type=int, help="Placeholder description")
    parser.add_argument("--cosine_schedule", default=False, type=bool, help="Use cosine schedule")
    parser.add_argument("--mel_size", default=(64, 1024), type=tuple, help="Size of Mel spectrograms")
    parser.add_argument("--num_train_steps", default=1000, type=int, help="Number of training steps")
    parser.add_argument("--num_inference_steps", default=250, type=int, help="Number of inference steps")
    parser.add_argument("--latent_dim", default=8, type=int, help="Latent dimension of the VAE")
    parser.add_argument("--beta_start", default=1e-4, type=float, help="Start value for beta")
    parser.add_argument("--beta_end", default=0.02, type=float, help="End value for beta")
    parser.add_argument("--latent", default=True, type=bool, help="Placeholder description")
    parser.add_argument("--vae_compression_ratio", default=4, type=int, help="VAE compression ratio")

    args = parser.parse_args()
    
    os.makedirs('images', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('waveforms', exist_ok=True)
        # Example of how to train the DiffusionTrainer using PyTorch Lightning
    trainer = pl.Trainer(limit_train_batches = args.train_batches_per_epoch, max_epochs=args.epochs, accelerator = 'gpu', limit_val_batches=args.val_batch_per_epoch, log_every_n_steps=1)
    diffusion = LatentDiffusion(args)
    #compiled_diffusion = torch.compile(diffusion)
    trainer.fit(diffusion, ckpt_path=args.checkpoint_path)
