import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
import numpy as np

class VAE(nn.Module):
    def __init__(self, timesteps, features, latent_dim):
        super(VAE, self).__init__()
        self.timesteps = timesteps
        self.features = features
        self.latent_dim = latent_dim

        # Encoder
        self.encoder_lstm = nn.LSTM(input_size=features, hidden_size=100, batch_first=True)
        self.fc_mean = nn.Linear(100, latent_dim)
        self.fc_log_var = nn.Linear(100, latent_dim)

        # Decoder
        self.decoder_lstm = nn.LSTM(input_size=latent_dim, hidden_size=100, batch_first=True)
        self.time_distributed_dense = nn.Linear(100, features)

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
            latent_samples = torch.randn(num_samples, self.latent_dim)
            generated_sequences = self.decode(latent_samples)
            return generated_sequences.numpy()
    
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

if __name__ == "__main__":
    batch_size = 16
    n_in = 9
    num_sequences = 1000

    sequences = np.random.laplace(loc=3, scale=5, size=(num_sequences, n_in, 1))
    sequences = torch.tensor(sequences, dtype=torch.float32)
    
    dataset = TensorDataset(sequences, sequences) 
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    vae = VAE(timesteps=n_in, features=1, latent_dim=2)
    vae.optimizer = optim.Adam(vae.parameters(), lr=1e-3)

    # Training loop
    epochs = 300
    for epoch in range(epochs):
        for batch_x, _ in dataloader:
            loss = vae.training_step(batch_x)
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')
    
    num_samples = 10
    generated_sequences = vae.generate_sequences(num_samples)
    
    vae.plot_sequences(sequences.numpy(), generated_sequences)
    
