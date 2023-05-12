import pytorch_lightning as pl
import torchaudio
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from vocoder import Vocoder
from data import NSynthDataset, collate, SameSongDataset
from torch.utils.data import DataLoader

class VocoderLightning(pl.LightningModule):
    def __init__(self, vocoder, learning_rate=1e-3):
        super().__init__()
        self.vocoder = vocoder
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.vocoder(x)

    def training_step(self, batch, batch_idx):
        waveform = batch # waveform, _ = batch
        reconstruction, loss, mel_spec, M_target, M_pred = self.vocoder.evaluate_and_compute_loss(waveform)

        # Log the loss
        self.log("train_loss", loss, prog_bar = True)

        # Log audio, mel spectrogram, and spectrograms to TensorBoard
        self.logger.experiment.add_audio("reconstructed_audio", reconstruction[0], self.global_step, self.vocoder.sample_rate)
        self.logger.experiment.add_audio("real_audio", waveform[0], self.global_step, self.vocoder.sample_rate)
        self.logger.experiment.add_image("mel_spectrogram", mel_spec[0], self.global_step, dataformats="HW")
        self.logger.experiment.add_image("real_spectrogram", M_target[0], self.global_step, dataformats="HW")
        self.logger.experiment.add_image("reconstructed_spectrogram", M_pred[0], self.global_step, dataformats="HW")

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

if __name__ == '__main__':
    root_dir = 'data'
    metadata_file = 'examples.json'
    dataset = SameSongDataset(song_file = 'data/vocal_synthetic_001-075-025.wav', num_repeats=16)
    train_loader = DataLoader(dataset, batch_size = 1, collate_fn = collate)
    # Create a Vocoder instance
    vocoder = Vocoder()

    # Create the LightningModule
    vocoder_lightning = VocoderLightning(vocoder)

    # Configure the logger and checkpoint callbacks
    logger = TensorBoardLogger(save_dir="logs/")
    checkpoint_callback = ModelCheckpoint(monitor="train_loss", save_top_k=3, mode="min")

    # Train the model
    trainer = pl.Trainer(logger=logger, callbacks=[checkpoint_callback], accelerator='cpu', max_epochs=50)
    trainer.fit(vocoder_lightning, train_loader)