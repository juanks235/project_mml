import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
import numpy as np

class ConvVAE(nn.Module):
    def __init__(self, in_channels, latent_dim, img_size, device):
        super(ConvVAE, self).__init__()
        self.latent_dim = latent_dim
        self.image_size = img_size
        self.device = device
        
        # Encoder
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1) 
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1) 

        self.fc_mean = nn.Linear(128 * (self.image_size // 8) * (self.image_size // 8), latent_dim)
        self.fc_log_var = nn.Linear(128 * (self.image_size // 8) * (self.image_size // 8), latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(latent_dim, 128 * (self.image_size // 8) * (self.image_size // 8))
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1) 
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1) 
        self.deconv3 = nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1) 

    def encode(self, x):
        e = nn.functional.relu(self.conv1(x))
        e = nn.functional.relu(self.conv2(e))
        e = nn.functional.relu(self.conv3(e))
        e = e.view(e.size(0), -1)
        z_mean = self.fc_mean(e)
        z_log_var = self.fc_log_var(e)
        return z_mean, z_log_var

    def reparameterize(self, mean, log_var):
        epsilon = torch.randn_like(mean)
        return mean + torch.exp(0.5 * log_var) * epsilon

    def decode(self, z):
        d = self.fc_decode(z)
        d = d.view(d.size(0), 128, self.image_size // 8, self.image_size // 8)
        d = nn.functional.relu(self.deconv1(d))
        d = nn.functional.relu(self.deconv2(d))
        outputs = nn.functional.sigmoid(self.deconv3(d))
        return outputs

    def forward(self, x):
        z_mean, z_log_var = self.encode(x)
        z = self.reparameterize(z_mean, z_log_var)
        x_reconstructed = self.decode(z)
        return x_reconstructed, z_mean, z_log_var

    def compute_loss(self, x, x_reconstructed, z_mean, z_log_var):
        reconstruction_loss = nn.functional.binary_cross_entropy(x_reconstructed, x, reduction='sum')
        kl_loss = -0.5 * torch.mean(z_log_var - torch.square(z_mean) - torch.exp(z_log_var) + 1)
        return reconstruction_loss + kl_loss

    def training_step(self, x):
        self.optimizer.zero_grad()
        x_reconstructed, z_mean, z_log_var = self(x)
        loss = self.compute_loss(x, x_reconstructed, z_mean, z_log_var)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def generate_images(self, num_samples):
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(self.device)
            return self.decode(z).cpu().numpy()
    
    def plot_images(self, training_images, generated_images):
        num_images = min(5, training_images.shape[0], generated_images.shape[0])
        fig, axs = plt.subplots(2, num_images, figsize=(15, 6))
        
        for i in range(num_images):
            # Plot training images
            axs[0, i].imshow(training_images[i].transpose(1, 2, 0)) 
            axs[0, i].set_title('Training Image')
            axs[0, i].axis('off')
            
            # Plot generated images
            axs[1, i].imshow(generated_images[i].transpose(1, 2, 0)) 
            axs[1, i].set_title('Generated Image')
            axs[1, i].axis('off')
        
        plt.suptitle('Training vs Generated Images')
        plt.savefig('images_plot.png')
        plt.close()

if __name__ == "__main__":
    batch_size = 16
    in_channels = 1
    img_size = 64
    num_images = 1000

    images = np.random.rand(num_images, in_channels, img_size, img_size).astype(np.float32) 
    images = torch.tensor(images)

    dataset = TensorDataset(images, images)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vae = ConvVAE(in_channels=in_channels, img_size=img_size, latent_dim=20, device=device).to(device)
    vae.optimizer = optim.Adam(vae.parameters(), lr=1e-3)

    # Training loop
    epochs = 1000
    for epoch in range(epochs):
        for batch_x, _ in dataloader:
            batch_x = batch_x.to(device)
            loss = vae.training_step(batch_x)
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')

    num_samples = 10
    generated_images = vae.generate_images(num_samples)
    vae.plot_images(images[:num_samples].numpy(), generated_images)