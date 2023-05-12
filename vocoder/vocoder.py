import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as F_audio
import torchaudio
import numpy as np
from util import Conv1dBlock

    
class Vocoder(nn.Module):
    def __init__(self, mel_bins = 64, stft_bins = 2048, hop_length = 256, sampling_rate = 16000, channels = 1536, n_conv = 7, loss_weights = [0,1, 1,0]):
        super().__init__()
        self.stft_bins = stft_bins
        self.R = hop_length
        self.sample_rate = sampling_rate
        self.mel_bins = mel_bins
        self.proj_in = nn.Conv1d(mel_bins, channels, kernel_size = 3, padding = 1, bias = True)
        self.convs = nn.Sequential(
            *[Conv1dBlock(channels) for _ in range(n_conv)]
        )
        self.proj_out = nn.Conv1d(channels, stft_bins * 3, kernel_size = 3, padding = 1, bias = True)
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate = sampling_rate, n_fft = stft_bins, hop_length = hop_length, n_mels = mel_bins)
        self.dct = torchaudio.transforms.LFCC(sample_rate = sampling_rate, n_lfcc=20)
        self.weights = loss_weights
        
    def forward(self, x):
        x = torch.abs(x)
        m2l = self.mel_to_linear(x)
        x = F.relu(self.proj_in(x))
        x = self.convs(x)
        x = self.proj_out(x)
        M = 5 * F.tanh((x[:,:self.stft_bins,:] + m2l)/5)
        delta_m = F.relu(x[:,self.stft_bins:2*self.stft_bins,:])
        delta_n = F.relu(x[:,2 * self.stft_bins:,:])
        return M, delta_m, delta_n
    
    def compute_lambda(self, phase_gradient_time, phase_gradient_freq, N = 2048, R = 256):
        B, C, T = phase_gradient_time.shape
        n = torch.arange(T, device = phase_gradient_freq.device).expand(B,C,T)
        m = torch.arange(C, device = phase_gradient_freq.device).expand(B,T,C).transpose(1,2)
        delta_m = phase_gradient_time * N / (2 * torch.pi) - m
        delta_m = torch.clamp(phase_gradient_time, -4.0, 4.0)
        m_dot = m + delta_m
        delta_n = -1 * phase_gradient_freq * N / (2 * torch.pi * R)
        n_dot = n + delta_n

        mdotdm = torch.gradient(m_dot, dim = 1)[0]
        ndotdn = torch.gradient(n_dot, dim = 2)[0]
        lambda_val = torch.exp(-1 * (mdotdm / ndotdn) ** 2)
        return lambda_val

    def integrate(self, phase_gradient_time, phase_gradient_freq, lambda_val, R=256, lambda_I=0.4, lambda_S=0.5):
        """
        Integrate the phase gradients based on the given lambda values.

        Parameters:
        phase_gradient_time (tensor): Phase gradient with respect to time
        phase_gradient_freq (tensor): Phase gradient with respect to frequency
        lambda_val (tensor): Lambda values used to determine horizontal and vertical phase updates
        R (int): Hop size between consecutive frames in the STFT (default 256)
        lambda_I (float): Impulsive Lambda value (default 0.4)
        lambda_S (float): Sinusoidal Lambda value (default 0.5)

        Returns:
        phase (tensor): Integrated phase
        """
        
        # Create masks for active time gradient components and active frequency gradient components
        vertical_mask = lambda_val > lambda_S
        horizontal_mask = lambda_val < lambda_I
        random_mask = (lambda_val >= lambda_I) & (lambda_val <= lambda_S)

        horizontal_gradient = phase_gradient_time * horizontal_mask * R
        vertical_gradient = phase_gradient_freq * vertical_mask
        rolled_horizontal_gradient = torch.roll(horizontal_gradient, dims=1, shifts=1)
        rolled_vertical_gradient = torch.roll(vertical_gradient, dims=0, shifts=1)

        # Initialize the phase tensor
        phase = torch.zeros_like(phase_gradient_time)

        # Set random phase values where lambda is between 0.4 and 0.5
        random_values = torch.rand_like(phase) * random_mask
        phase = phase + random_values

        # Vertical phase updates
        for m in range(1, phase.shape[0]):
            phase[m] = phase[m - 1] + rolled_vertical_gradient[m]

        # Horizontal phase updates
        for n in range(1, phase.shape[1]):
            phase[:, n] += phase[:, n - 1] + rolled_horizontal_gradient[:, n]

        return phase

    def mel_to_linear(self, mel_spectrogram):
        # Create the linear filterbank
        linear_filterbank = F_audio.linear_fbanks(n_freqs=self.stft_bins, f_min=0, f_max=self.sample_rate / 2, n_filter=self.mel_bins, sample_rate=self.sample_rate).to(mel_spectrogram)
        linear_filterbank = linear_filterbank.t()  # Transpose the filterbank

        # Compute the pseudo-inverse of the linear filterbank
        linear_pseudo_inv = torch.pinverse(linear_filterbank)

        # Transform the mel spectrogram back to the linear-frequency scale
        linear_spectrogram = torch.matmul(linear_pseudo_inv, mel_spectrogram)

        return linear_spectrogram

    def compute_mel(self, x):
        return self.mel(x)

    def loss(self, M_pred, delta_m_pred, delta_n_pred, M_target, delta_m_target, delta_n_target):
        # L1 - Magnitude MSE
        L1 = F.mse_loss(M_pred, M_target)

        # L2 - LFCC MSE
        M_pred_dct = self.dct(M_pred)
        M_target_dct = self.dct(M_target)
        L2 = F.mse_loss(M_pred_dct, M_target_dct)

        # L3 - Phase gradient MSE
        M2_target = M_target**2
        lambda_mask = self.compute_lambda(delta_m_target, delta_n_target)  # Compute lambda mask here
        L3 = torch.where(lambda_mask > 0.5,
                         M2_target * (delta_m_pred - delta_m_target)**2,
                         M2_target * (delta_n_pred - delta_n_target)**2).mean()

        # L4 - Lambda MSE
        lambda_pred = self.compute_lambda(delta_m_pred, delta_n_pred)  # Compute lambda from phase gradient estimates
        L4 = F.mse_loss(M2_target * lambda_pred, M2_target * lambda_mask)

        loss = self.weights[0] * L1 + self.weights[1] * L2 + self.weights[2] * L3 + self.weights[3] * L4
        return loss
    
    def evaluate_and_compute_loss(self, x):
        # x: target signal
        mel_spec, M_target, delta_m_target, delta_n_target, _, _ = self.get_targets(x)
        M_pred, delta_m_pred, delta_n_pred = self.forward(mel_spec)
        loss = self.loss(M_pred, delta_m_pred, delta_n_pred, M_target, delta_m_target, delta_n_target)
        reconstruction = self.reconstruct(M_pred, delta_m_pred, delta_n_pred)
        return reconstruction, loss, mel_spec, M_target, M_pred
    
    def evaluate(self, x, with_loss = True):
        # x: target signal
        # with_loss: whether to compute loss or not
        # Compute the mel spectrogram
        mel_spec = self.compute_mel(x)
        loss = None
        # Compute the loss
        if with_loss:
            reconstruction, loss = self.evaluate_and_compute_loss(mel_spec, x)
        else:
            M_pred, delta_m_pred, delta_n_pred = self.forward(mel_spec)
            reconstruction = self.reconstruct(M_pred, delta_m_pred, delta_n_pred)
        return reconstruction, loss
        
    
    def get_targets(self, x):
        # x: target signal
        # Compute the phase gradients
        stft = torch.stft(x, self.stft_bins, self.R, onesided=False, return_complex=True)
        B,C,T = stft.shape
        M_target = torch.abs(stft)
        mel = self.compute_mel(x)
        phase = torch.angle(stft)
        phase_gradient_time = torch.gradient(phase, dim = 2)[0]
        phase_gradient_freq = torch.gradient(phase, dim = 1)[0]
        m = torch.arange(0, C).expand(B,T,C).permute(0,2,1).to(x)

        # Compute the delta_m and delta_n values
       # delta_m_target = phase_gradient_time * self.stft_bins / (2 * np.pi)  - m
        delta_n_target = -1 * phase_gradient_freq * self.stft_bins / (2 * torch.pi * self.R)

        delta_m_target = torch.clamp(phase_gradient_time, min = -4.0, max = 4.0)
        delta_n_target = torch.clamp(delta_n_target, min = - self.stft_bins / (2 * self.R), max = self.stft_bins / (2 * self.R))
        return mel, M_target, delta_m_target, delta_n_target, phase_gradient_time, phase_gradient_freq

    def reconstruct(self, M, phase_gradient_time, phase_gradient_freq):
        phase = self.integrate(M, phase_gradient_time, phase_gradient_freq)
        stft = M * torch.exp(1j * phase)
        x = torch.istft(stft, self.stft_bins, self.R, onesided=False)
        return x
    

if __name__ == '__main__':
    x, sr = torchaudio.load('data/vocal_synthetic_001-075-025.wav')
    vocoder = Vocoder()
    mel, M_target, delta_m_target, delta_n_target, phase_gradient_time, phase_gradient_freq = vocoder.get_targets(x)
    mel = torch.log(mel)
    lambda_val = vocoder.compute_lambda(phase_gradient_time=phase_gradient_time, phase_gradient_freq = phase_gradient_freq)
    phase_recon = vocoder.integrate(phase_gradient_time, phase_gradient_freq, lambda_val)
    stft = M_target * torch.exp(1j * phase_recon)
    stft_rand = M_target * torch.exp(1j * torch.rand_like(phase_recon))
    x_recon = torch.istft(stft, vocoder.stft_bins, vocoder.R, onesided=False)
    x_recon_rand = torch.istft(stft_rand, vocoder.stft_bins, vocoder.R, onesided=False)
    torchaudio.save('data/recon_rand.wav', x_recon_rand, sr)
    torchaudio.save('data/recon.wav', x_recon, sr)
