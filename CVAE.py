import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
import numpy as np

def calculate_output_shape_convtranspose(input_dim, stride, padding, dilation, kernel_size, output_padding):
        return (input_dim - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1

class Resnet1DBlock(nn.Module):
    def __init__(self, kernel_size, input_dim, batch_size=None, type='encode'):
        super(Resnet1DBlock, self).__init__()
        self.type = type
        if type=='encode':
            self.conv1a = nn.Conv1d(input_dim, 2, kernel_size, 1, padding="same")
            self.norm1b = nn.InstanceNorm1d(2)

            self.conv1b = nn.Conv1d(2, 1, kernel_size, 1, padding="same")
            self.norm1a = nn.InstanceNorm1d(1)
        elif type=='decode':
            self.conv1a = nn.ConvTranspose1d(input_dim, batch_size, kernel_size, 1, padding=95, dilation=3, output_padding=1)
            conv1a_output_len = calculate_output_shape_convtranspose(batch_size, 1, 95, 3, kernel_size, 1)
            self.norm1a = nn.BatchNorm1d(conv1a_output_len)
            self.conv1b = nn.ConvTranspose1d(conv1a_output_len, batch_size, kernel_size, 1, padding=95, dilation=3, output_padding=1)
            conv1b_output_len = calculate_output_shape_convtranspose(batch_size, 1, 95, 3, kernel_size, 1)
            self.norm1b = nn.BatchNorm1d(conv1b_output_len)
        else:
            return None

    def forward(self, x):
        z = nn.ReLU()(x)
        z = self.conv1a(z)
        z = self.norm1a(z)
        z = nn.LeakyReLU(0.4)(z)

        if self.type == "decode":
            z = z.transpose(0, 1)

        z = self.conv1b(z)
        z = self.norm1b(z)
        z = nn.LeakyReLU(0.4)(z)

        if self.type == "encode":
            z = z + x
        elif self.type == "decode":
            z = z + torch.mean(x, dim=0).reshape((z.shape[0],-1))

        z = nn.ReLU()(z)
        
        return z

class CVAE(nn.Module):
    def __init__(self, latent_dim, device, n_timesteps, batch_size):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.device = device
        self.n_timesteps = n_timesteps
        self.batch_size = batch_size
        self.encoder = [
            nn.Conv1d(1, 64, 1, 1, padding="same"),
            Resnet1DBlock(64, 64, type='encode'),
            nn.Conv1d(64, 128, 1, 1, padding="same"),
            Resnet1DBlock(128, 128, type='encode'),
            nn.Conv1d(128, 128, 1, 1, padding="same"),
            Resnet1DBlock(128, 128, type='encode'),
            nn.Conv1d(128, 256, 1, 1, padding="same"),
            Resnet1DBlock(256, 256, type='encode'),
            torch.flatten            
        ]

        size = calculate_output_shape_convtranspose(self.batch_size,  1, 95, 3, 512, 1)

        # Decoder
        self.d1 = Resnet1DBlock(512, self.latent_dim, self.batch_size, "decode")
        size = calculate_output_shape_convtranspose(self.batch_size,  1, 95, 3, 512, 1)
        self.d2 = nn.ConvTranspose1d(size, self.batch_size, 511)
        
        self.d3 = Resnet1DBlock(256, 512, self.batch_size, "decode")
        size = calculate_output_shape_convtranspose(self.batch_size,  1, 95, 3, 256, 1)
        self.d4 = nn.ConvTranspose1d(size, self.batch_size, 255)
        
        self.d5 = Resnet1DBlock(128, 256, self.batch_size, "decode")
        size = calculate_output_shape_convtranspose(self.batch_size,  1, 95, 3, 128, 1)
        self.d6 = nn.ConvTranspose1d(size, self.batch_size, 127)

        
        self.d7 = Resnet1DBlock(64, 128, self.batch_size, "decode")
        size = calculate_output_shape_convtranspose(self.batch_size,  1, 95, 3, 64, 1)
        self.d8 = nn.ConvTranspose1d(size, self.batch_size, 63)

        self.d9 = Resnet1DBlock(n_timesteps, 64, self.batch_size, "decode")
        size = calculate_output_shape_convtranspose(self.batch_size,  1, 95, 3, n_timesteps, 1)
        self.d10 = nn.ConvTranspose1d(size, self.batch_size, n_timesteps - 1)

    def reparameterize(self, mean, log_var):
        epsilon = torch.randn_like(mean)
        return mean + torch.exp(0.5 * log_var) * epsilon

    def compute_loss(self, x, x_reconstructed, z_mean, z_log_var):
        reconstruction_loss = nn.MSELoss()(x, x_reconstructed)
        kl_loss = -0.5 * torch.mean(z_log_var - torch.square(z_mean) - torch.exp(z_log_var) + 1)
        return reconstruction_loss + kl_loss

    def training_step(self, x):
        self.optimizer.zero_grad()
        x_reconstructed, z_mean, z_log_var = self(x)
        loss = self.compute_loss(x.reshape(self.batch_size, -1), x_reconstructed.reshape(self.batch_size, -1), z_mean, z_log_var)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def generate_sequences(self, num_samples):
        with torch.no_grad():
            latent_samples = torch.randn(num_samples, self.latent_dim).to(self.device)
            z = self.d1(latent_samples.T)
            z = self.d2(z.T)
            z = self.d3(z.T)
            z = self.d4(z.T)
            z = self.d5(z.T)
            z = self.d6(z.T)
            z = self.d7(z.T)
            z = self.d8(z.T)
            z = self.d9(z.T)
            z = self.d10(z.T)
            return z.cpu().numpy()

    def forward(self, z):
        n_examples = z.shape[0]
        # Encoder
        for encode_layer in self.encoder:
            z = encode_layer(z)
        z = torch.reshape(z, (n_examples, -1))
        features_per_example = z.shape[1]
        z_mean = nn.Linear(features_per_example, self.latent_dim)(z)
        z_log_var = nn.Linear(features_per_example, self.latent_dim)(z)
        # Reparametrization
        z = self.reparameterize(z_mean, z_log_var)

        # Decoding
        z = self.d1(z.T)
        z = self.d2(z.T)
        z = self.d3(z.T)
        z = self.d4(z.T)
        z = self.d5(z.T)
        z = self.d6(z.T)
        z = self.d7(z.T)
        z = self.d8(z.T)
        z = self.d9(z.T)
        z = self.d10(z.T)
        return z, z_mean, z_log_var
