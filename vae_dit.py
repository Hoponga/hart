# (todo) diffusion transformer trained on CIFAR-10 in latent space of VAE 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

class VAE(nn.Module):
    def __init__(self, latent_dim: int = 4):
        super().__init__()
        self.latent_dim = latent_dim

  
        self.encoder_core = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),   # 32 to 16
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 16 to 8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), # keep 8x8
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        enc_feat_dim = 256 * 8 * 8
        proj_dim = latent_dim * 8 * 8  
 
        self.enc_mu_fc = nn.Linear(enc_feat_dim, proj_dim)
        self.enc_logvar_fc = nn.Linear(enc_feat_dim, proj_dim)


        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 128, kernel_size=4, stride=2, padding=1),  # 8 -> 16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 16 -> 32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1),  # keep 32
            nn.Sigmoid(),  # outputs in [0,1]
        )

    # z, mu, logvar shapes (B, latent_dim, 8, 8) from cs294 hw spec 
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
 
        h = self.encoder_core(x)  # (B, 256, 8, 8)
        h_flat = h.view(h.size(0), -1)  # (B, 16384)
        mu_flat = self.enc_mu_fc(h_flat)      # (B, proj_dim)
        logvar_flat = self.enc_logvar_fc(h_flat)  # (B, proj_dim)

        # Reshape back to spatial maps
        mu = mu_flat.view(h.size(0), self.latent_dim, 8, 8)
        logvar = logvar_flat.view(h.size(0), self.latent_dim, 8, 8)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
   
        return self.decoder(z)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        z, mu, logvar = self.encode(x)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, logvar
    
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    



