import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchaudio.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

class VAE(nn.Module):
    def __init__(self, timesteps, features, latent_dim, device, hidden_size):
        super(VAE, self).__init__()
        self.timesteps = timesteps
        self.features = features
        self.latent_dim = latent_dim
        self.device = device
        self.hidden_size = hidden_size

        # Encoder
        self.encoder_lstm = nn.LSTM(input_size=features, hidden_size=self.hidden_size, batch_first=True)
        self.fc_mean = nn.Linear(self.hidden_size, latent_dim)
        self.fc_log_var = nn.Linear(self.hidden_size, latent_dim)

        # Decoder
        self.decoder_lstm = nn.LSTM(input_size=latent_dim, hidden_size=self.hidden_size, batch_first=True)
        self.time_distributed_dense = nn.Linear(self.hidden_size, features)

    def encode(self, x):
        h, _ = self.encoder_lstm(x)
        h = h[:, -1, :] 
        z_mean = self.fc_mean(h)
        z_log_var = self.fc_log_var(h)
        return z_mean, z_log_var

    def reparameterize(self, mean, log_var):
        epsilon = torch.randn_like(mean)
        return mean + torch.exp(0.5 * log_var) * epsilon

    def decode(self, z):
        z_repeated = z.unsqueeze(1).repeat(1, self.timesteps, 1)
        h_decoded, _ = self.decoder_lstm(z_repeated)
        outputs = self.time_distributed_dense(h_decoded)
        return outputs

    def forward(self, x):
        z_mean, z_log_var = self.encode(x)
        z = self.reparameterize(z_mean, z_log_var)
        x_reconstructed = self.decode(z)
        return x_reconstructed, z_mean, z_log_var

    def compute_loss(self, x, x_reconstructed, z_mean, z_log_var):
        reconstruction_loss = nn.MSELoss()(x, x_reconstructed)
        kl_loss = -0.5 * torch.mean(z_log_var - torch.square(z_mean) - torch.exp(z_log_var) + 1)
        return reconstruction_loss + kl_loss

    def training_step(self, x):
        self.optimizer.zero_grad()
        x_reconstructed, z_mean, z_log_var = self(x)
        loss = self.compute_loss(x, x_reconstructed, z_mean, z_log_var)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def generate_sequences(self, num_samples):
        with torch.no_grad():
            latent_samples = torch.randn(num_samples, self.latent_dim).to(self.device)
            generated_sequences = self.decode(latent_samples)
            return generated_sequences.cpu().numpy()
        
def audio_array_to_melspectrogram(audio_array, sample_rate, timesteps):
    transform = transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=32)
    waveform = torch.tensor(audio_array, dtype=torch.float32)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    mel_spectrogram = transform(waveform).squeeze(0) 
    mel_spectrogram = mel_spectrogram.T 

    if mel_spectrogram.shape[0] > timesteps:
        mel_spectrogram = mel_spectrogram[:timesteps, :]
    else:
        padding = torch.zeros((timesteps - mel_spectrogram.shape[0], mel_spectrogram.shape[1]))
        mel_spectrogram = torch.cat((mel_spectrogram, padding), dim=0)
    
    return mel_spectrogram

def create_dataloader(audio_arrays, sample_rate, batch_size, timesteps):
    mel_spectrograms = [audio_array_to_melspectrogram(audio_array, sample_rate, timesteps) for audio_array in audio_arrays]
    mel_spectrograms = torch.stack(mel_spectrograms)
    dataset = TensorDataset(mel_spectrograms, mel_spectrograms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

if __name__ == "__main__":
    timesteps = 100
    features = 8 
    latent_dim = 64
    hidden_size = 256
    batch_size = 1
    num_epochs = 10
    sample_rate = 22050  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    audio_arrays = [
        np.random.randn(sample_rate * 5),  
        np.random.randn(sample_rate * 5),
    ]
    
    print(audio_arrays)
    
    dataloader = create_dataloader(audio_arrays, sample_rate, batch_size, timesteps)
    
    vae = VAE(timesteps, features, latent_dim, device, hidden_size).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)
    vae.optimizer = optimizer
    
    # Training loop
    epochs = 1000
    for epoch in range(epochs):
        for batch_x, _ in dataloader:
            batch_x = batch_x.to(device)
            loss = vae.training_step(batch_x)
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')