# Training latent diffusion model on CIFAR-10 

import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset 
import torchvision.transforms as T 
import torchvision.datasets as datasets 
from torchvision.utils import save_image
import numpy as np 
import matplotlib.pyplot as plt 
from tqdm import tqdm 
from vae import VAE
from dit import DIT 
from diffusion_cifar10 import get_ddpm_schedule, get_alphas_sigmas




@torch.no_grad()
def latent_sample(model, vae, num_samples, device, steps, eta, classes, input_img = None): 
    model.eval() 
    vae.eval() 
    dit_input = None 
    if input_img is not None: 
        # of shape (C, H, W)

        dit_input, _, _ = vae.encode(input_img.to(device))
    else: 
        dit_input = torch.randn(num_samples, vae.latent_dim, 8, 8).to(device)

    latent_out = sample_eps(model, dit_input, steps, eta, classes)
    img_out = vae.decode(latent_out)

    return img_out 


# eta = 0: fully DDIM 
# eta = 1: fully DDPM 
@torch.no_grad()
def sample_eps(model, x, steps, eta, classes):

    ts = x.new_ones([x.shape[0]])

    t  = torch.linspace(1, 0, steps + 1)[:-1]
    log_snrs  = get_ddpm_schedule(t)

    # \sigma = \sqrt{1 - \prod_{i=1}^t \alpha_i}
    alphas, sigmas = get_alphas_sigmas(log_snrs)

    for i in tqdm(range(steps)):
   
        with torch.autocast(device_type="mps", dtype=torch.bfloat16):
            eps = model(x, ts * log_snrs[i], classes).float()

       
        x0_pred = (x - sigmas[i] * eps) / alphas[i]

        if i < steps - 1:        
            ddim_sigma     = eta * (sigmas[i + 1]**2 / sigmas[i]**2).sqrt() * \
                                   (1 - alphas[i]**2 / alphas[i + 1]**2).sqrt()
            adjusted_sigma = (sigmas[i + 1]**2 - ddim_sigma**2).sqrt()

            # trick: Taking the mean of our distribution as a weighted sum of our x0 prediction and noise prediction 
            # and variance as the ddim_sigma is equivalent to sampling 
            # x_t-1 | x_t ~ N(mu_\theta(x_t, t), \sigma_\theta), where mu_theta is 1/\sqrt{\alpha_t} (x_t - \sigma_t / \sqrt{\alpha_t} * \epsilon_\theta(x_t, t))
            x = x0_pred * alphas[i + 1] + eps * adjusted_sigma
            if eta:                                     
                x += torch.randn_like(x) * ddim_sigma

    return x0_pred                                  


def sample_ddpm_eps(model, x, steps, classes):  # eta = 1
    return sample_eps(model, x, steps, eta=1.0, classes=classes)

def sample_ddim_eps(model, x, steps, classes):  # eta = 0
    return sample_eps(model, x, steps, eta=0.0, classes=classes)     
            



if __name__ == "__main__": 
    # Load the CIFAR-10 dataset 
    transform = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = datasets.CIFAR10(root = "data", train = True, download = True, transform = transform)
    test_dataset = datasets.CIFAR10(root = "data", train = False, download = True, transform = transform)

    train_loader = DataLoader(train_dataset, batch_size = 128, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = 128, shuffle = False)

    # Load the VAE and DIT models 
    vae = VAE()

    config = {
        "image_height": 8, 
        "image_width": 8, 
        "patch_height": 4, 
        "patch_width": 4, 
        "num_channels": vae.latent_dim, 
        "hidden_size": 1024,
        "num_heads": 16, 
        "num_layers": 24, 
        "dropout": 0.1, 
        "timestep_embedding_size": 16, 
        "conditioning_size": 10, 
        "training": True, 
        "num_classes": 10,
        "intermediate_size": 4096, 
    }
    dit = DIT(config)

    # Load the optimizer 
    optimizer = torch.optim.Adam(dit.parameters(), lr = 1e-4)

    # Load the device 
    device = torch.device("mps")
    
    
    # just try sampling for now 
    dit.to(device)
    vae.to(device)

    dit.eval()
    vae.eval()

    # sample 100 images 
    for i in range(5): 
        out = latent_sample(dit, vae, 1, device, 1000, torch.tensor([0.0]).to(device), torch.tensor([[0]*10]).to(device))
        save_image(out, f"samples/dit_sample_{i}.png")
        
