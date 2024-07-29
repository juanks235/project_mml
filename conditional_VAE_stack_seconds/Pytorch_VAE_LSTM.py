import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
import numpy as np

class LSTMVAE(nn.Module):
    def __init__(self, timesteps, features, hidden_dim, latent_dim, device):
        super(LSTMVAE, self).__init__()
        self.features = features # Add genre conditioning
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.timesteps = timesteps

        # Encoder
        self.encoder_lstm = nn.LSTM(input_size=self.features, hidden_size=self.hidden_dim, batch_first=True)
        self.encoder_latent_mean = nn.Linear(self.hidden_dim, self.latent_dim)
        self.encoder_latent_log_var = nn.Linear(self.hidden_dim, self.latent_dim)

        # Decoder
        self.decoder_expand = nn.Linear(self.latent_dim + 1, self.hidden_dim * self.timesteps)
        self.decoder_lstm = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.features - 1, batch_first=True)

    def encode(self, x):
        h, _ = self.encoder_lstm(x)
        h = h[:, -1, :]
        z_mean = self.encoder_latent_mean(h)
        z_log_var = self.encoder_latent_log_var(h)
        return z_mean, z_log_var

    def reparameterize(self, mean, log_var):
        epsilon = torch.randn_like(mean)
        return mean + torch.exp(0.5 * log_var) * epsilon

    def decode(self, z):
        z = z
        z = self.decoder_expand(z)
        z = torch.reshape(z, (-1, self.timesteps, self.hidden_dim))
        z, _ = self.decoder_lstm(z)
        return z

    def forward(self, x):
        label = x[:,-1,-1].reshape(-1,1)
        z_mean, z_log_var = self.encode(x)
        # Conditioning
        z_mean = torch.concat((z_mean, label), dim=1)
        z_log_var = torch.concat((z_log_var, label), dim=1)
        z = self.reparameterize(z_mean, z_log_var)
        z = z
        z = self.decode(z)
        return z, z_mean, z_log_var

    def compute_loss(self, x, x_reconstructed, z_mean, z_log_var):
        reconstruction_loss = nn.MSELoss()(x[:,:,:-1], x_reconstructed)
        kl_loss = -0.5 * torch.mean(z_log_var - torch.square(z_mean) - torch.exp(z_log_var) + 1)
        return reconstruction_loss + kl_loss

    def training_step(self, x):
        self.optimizer.zero_grad()
        x_reconstructed, z_mean, z_log_var = self(x)
        loss = self.compute_loss(x, x_reconstructed, z_mean, z_log_var)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def generate_sequences(self, sample_classes):
        with torch.no_grad():
            latent_samples = torch.randn(len(sample_classes), self.latent_dim).to(self.device)
            labels = torch.tensor([
                [0] if c == 0
                else [1]
                for c in sample_classes
            ], dtype=torch.float32).to(self.device) # Conditioning on desired label
            latent_samples = torch.concat((latent_samples, labels), dim=1)
            generated_sequences = self.decode(latent_samples)
            return generated_sequences.cpu().numpy()
    
    def plot_sequences(self, training_sequences, generated_sequences):
        plt.figure(figsize=(12, 6))
    
        for i in range(min(5, training_sequences.shape[0])):
            plt.plot(training_sequences[i, :, 0], color='blue', alpha=0.5, label='Training Sequence' if i == 0 else "")
        
        for i in range(min(5, generated_sequences.shape[0])):
            plt.plot(generated_sequences[i, :, 0], color='red', alpha=0.5, label='Generated Sequence' if i == 0 else "")
        
        plt.xlabel('Timestep')
        plt.ylabel('Feature Value')
        plt.legend()
        plt.title('Training vs Generated Sequences')
        plt.savefig('sequences_plot.png')  
        plt.close()
