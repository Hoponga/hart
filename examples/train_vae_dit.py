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
import torch.nn.functional as F
import os



def eval_loss(model, rng, reals, classes):
    """Compute the denoising score matching loss when the network predicts ε_t.

    Args:
        model: the DIT denoiser that outputs an estimate of the injected noise.
        rng:   SobolEngine (or any sampler) used to draw random timesteps.
        reals: clean latent images, shape (B, C, H, W).
        classes: class-conditioning information to pass to the model.

    Returns:
        A scalar MSE loss averaged over batch and spatial dimensions.
    """

    device = reals.device
    #       print(classes.shape)


    # 1. Draw uniformly-distributed continuous timesteps t ∈ (0,1).
    t = rng.draw(reals.shape[0])[:, 0].to(device)

    # 2. Convert to log-SNR schedule parameters.
    log_snrs = get_ddpm_schedule(t)                    # (B,)
    alphas, sigmas = get_alphas_sigmas(log_snrs)       # (B,), (B,)

    # 3. Mix clean image with random Gaussian noise.
    alphas = alphas[:, None, None, None]               # (B,1,1,1)
    sigmas = sigmas[:, None, None, None]
    noise   = torch.randn_like(reals)                  # ε ~ N(0, I)
    noised_reals = reals * alphas + noise * sigmas     # x_t

    # 4. Predict ε̂_t and compute simple MSE.
    # convert to one hot 
    classes = F.one_hot(classes, num_classes=10)

    with torch.autocast(device_type="mps", dtype=torch.bfloat16):
        # add a dimension to log_snrs so it's (B, 1)
        log_snrs = log_snrs.unsqueeze(1)

        eps_pred = model(noised_reals, log_snrs, classes)

    loss = (eps_pred - noise).pow(2).mean([1, 2, 3]).mean()  # average over batch & pixels
    return loss

def train(model, vae, dataloader, optimizer, device, epochs): 
    model.train() 
    vae.train() 

    for epoch in range(epochs): 
        for i, (reals, classes) in enumerate(tqdm(train_dl)):
            opt.zero_grad()
            reals = reals.to(device)
            classes = classes.to(device)

            # Evaluate the loss
            loss = eval_loss(model, rng, reals, classes)

            # Do the optimizer step and EMA update
            scaler.scale(loss).backward()
            scaler.step(opt)
            ema_update(model, model_ema, 0.95 if epoch < 20 else ema_decay)
            scaler.update()

            if i % 50 == 0:
                tqdm.write(f'Epoch: {epoch}, iteration: {i}, loss: {loss.item():g}')


@torch.no_grad()
def latent_sample(model, vae, num_samples, device, steps, eta, classes, input_img = None): 
    model.eval() 
    vae.eval() 

    dit_input = None 
    if input_img is not None: 
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
            t_step = (ts * log_snrs[i]).unsqueeze(1)  # shape (B, 1)
            eps = model(x, t_step, classes).float()

       
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

    # ------------------------------------------------------------------
    #  Load pretrained VAE if checkpoint exists
    # ------------------------------------------------------------------
    vae_ckpt_path = os.path.join("checkpoints", "vae_cifar10.pt")
    if os.path.exists(vae_ckpt_path):
        ckpt = torch.load(vae_ckpt_path, map_location="cpu")
        if "model_state" in ckpt:
            vae.load_state_dict(ckpt["model_state"], strict=False)
            print(f"Loaded pretrained VAE from {vae_ckpt_path}")
        else:
            print(f"Checkpoint {vae_ckpt_path} does not contain model_state. Skipping preload.")
    else:
        print("Warning: Pretrained VAE checkpoint not found. Using randomly initialised VAE.")

    config = {
        "image_height": 8, 
        "image_width": 8, 
        "patch_height": 4, 
        "patch_width": 4, 
        "num_channels": vae.latent_dim, 
        "hidden_size": 128,
        "num_heads": 4, 
        "num_layers": 5, 
        "dropout": 0.1, 
        "timestep_embedding_size": 16, 
        "conditioning_size": 10, 
        "training": True, 
        "num_classes": 10,
        "intermediate_size": 512, 
    }
    dit = DIT(config)

    # Load the optimizer 
    optimizer = torch.optim.Adam(dit.parameters(), lr = 1e-4)

    # Load the device 
    device = torch.device("mps")

    # total number of parameters
    total_params = sum(p.numel() for p in dit.parameters())
    print(f"Total number of parameters: {total_params}")
    
    
    # ------------------------------------------------------------------
    #  TRAIN / VALIDATE / SAMPLE LOOP
    # ------------------------------------------------------------------

    dit.to(device)
    vae.to(device)

    epochs          = 50          # total number of training epochs
    log_every       = 100         # iterations between console logs
    sample_every    = 5           # epochs between sample image dumps
    checkpoint_every= 5           # epochs between checkpoint writes
    samples_dir     = "samples"
    ckpt_dir        = "checkpoints"

    import os, pathlib
    pathlib.Path(samples_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    scaler = torch.amp.GradScaler()
    rng    = torch.quasirandom.SobolEngine(1, scramble=True)

    for epoch in range(1, epochs + 1):
        dit.train()
        running_loss = 0.0

        for it, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")):
            images = images.to(device)
            labels = labels.to(device)

            # --- encode images to VAE latent space (no grad) -----------------
            with torch.no_grad():

                latents, _, _ = vae.encode(images)   # (B, latent_dim, 8, 8)

            # --- forward & backward ----------------------------------------
            loss = eval_loss(dit, rng, latents, labels)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            if (it + 1) % log_every == 0:
                avg = running_loss / log_every
                running_loss = 0.0
                print(f"[epoch {epoch} iter {it+1}] loss={avg:.4f}")

        # ------------------ validation ------------------------------------
        dit.eval()
        with torch.no_grad():
            val_loss = 0.0
            n_val    = 0
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                latents, _, _  = vae.encode(images)
                loss           = eval_loss(dit, rng, latents, labels)
                val_loss      += loss.item() * images.size(0)
                n_val         += images.size(0)
            val_loss /= n_val
        print(f"[epoch {epoch}] validation loss = {val_loss:.4f}")

        # ------------------ sample & save ----------------------------------
        if epoch % sample_every == 0 or epoch == 1:
            dit.eval()
            class_sample = torch.arange(10, device=device).repeat_interleave(8, 0)  # 80 samples
            class_sample = F.one_hot(class_sample, num_classes = 10)

            print(class_sample.shape)
            imgs = latent_sample(
                dit,
                vae,
                num_samples=class_sample.size(0),
                device=device,
                steps=250,
                eta=torch.tensor([0.0], device=device),
                classes=class_sample,
            )
            grid_name = os.path.join(samples_dir, f"epoch_{epoch:04d}.png")
            save_image(imgs, grid_name, nrow=10, normalize=True, value_range=(-1, 1))
            print(f"wrote samples to {grid_name}")

        # ------------------ checkpoint -------------------------------------
        if epoch % checkpoint_every == 0 or epoch == epochs:
            ckpt_path = os.path.join(ckpt_dir, f"dit_epoch_{epoch:04d}.pth")
            torch.save({
                "epoch": epoch,
                "dit": dit.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
            }, ckpt_path)
            print(f"saved checkpoint to {ckpt_path}")

    print("Training complete.")
        
