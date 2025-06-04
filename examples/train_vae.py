import argparse
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm

from examples.vae_dit import VAE


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=[1, 2, 3])


def loss_function(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
):
    """Compute VAE loss with adjustable KL weight.

    Reconstruction loss is binary cross-entropy (outputs in [0,1]); KL is averaged per-sample.
    """
    bce = nn.BCELoss(size_average=False)(recon_x, x)

    #bce = nn.functional.binary_cross_entropy(recon_x, x, reduction="mean")
    kl = kl_divergence(mu, logvar).mean()
    total = bce + beta * kl
    return total, bce, kl


def train(args):

    
    if torch.cuda.is_available() and not args.cpu:
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and not args.cpu:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}") # mps 

    # Dataset & DataLoader
    transform = transforms.Compose([
        transforms.ToTensor(), 
    ])

    # cifar datasets 

    train_ds = datasets.CIFAR10(root=args.data_root, train=True, download=True, transform=transform)
    test_ds = datasets.CIFAR10(root=args.data_root, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model
    model = VAE(latent_dim=args.latent_dim).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)

    best_test_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        # annealing destroys loss for some reason 
        kl_weight = args.beta * min(1.0, epoch / args.kl_anneal_epochs) if args.kl_anneal_epochs > 0 else args.beta

        model.train()
        running_total, running_bce, running_kl = 0.0, 0.0, 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch_idx, (data, _) in enumerate(pbar):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss, bce_loss, kl = loss_function(recon_batch, data, mu, logvar, beta=kl_weight)
            loss.backward()
            optimizer.step()

            running_total += loss.item() * data.size(0)
            running_bce += bce_loss.item() * data.size(0)
            running_kl += kl.item() * data.size(0)

            pbar.set_postfix({"loss": f"{loss.item():.3f}", "bce": f"{bce_loss.item():.3f}", "kl": f"{kl.item():.3f}"})

        train_size = len(train_loader.dataset)
        print(
            f"Train Epoch {epoch}: avg total {running_total/train_size:.4f}, "
            f"bce {running_bce/train_size:.4f}, kl {running_kl/train_size:.4f}, kl_w {kl_weight:.3f}"
        )


        model.eval()
        test_total = 0.0
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(device)
                recon, mu, logvar = model(data)
                loss, _, _ = loss_function(recon, data, mu, logvar, beta=kl_weight)
                test_total += loss.item() * data.size(0)
        test_total /= len(test_loader.dataset)
        print(f"Validation loss: {test_total:.4f}")

   
        # if test_total < best_test_loss:
        #     best_test_loss = test_total
        #     Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        #     ckpt_path = Path(args.save_dir) / "vae_cifar10.pt"
        #     torch.save(
        #         {
        #             "epoch": epoch,
        #             "model_state": model.state_dict(),
        #             "optimizer_state": optimizer.state_dict(),
        #         },
        #         ckpt_path,
        #     )
        #     print(f"Saved checkpoint to {ckpt_path}")

        # Save reconstruction samples every N epochs
        if epoch % args.sample_interval == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                sample_batch, _ = next(iter(test_loader))
                sample_batch = sample_batch.to(device)[: args.sample_n]
                recon, _, _ = model(sample_batch)

                grid = torch.cat([sample_batch.cpu(), recon.cpu()], dim=0)
                Path(args.sample_dir).mkdir(parents=True, exist_ok=True)
                img_path = Path(args.sample_dir) / f"epoch_{epoch:03d}.png"
                save_image(grid, img_path, nrow=args.sample_n, normalize=False)
                print(f"Saved samples to {img_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VAE on CIFAR-10")
    parser.add_argument("--data_root", type=str, default="./data", help="Path to store CIFAR-10")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=0.5, help="Weight on KL term")
    parser.add_argument("--latent_dim", type=int, default=4, help="Number of latent channels")
    parser.add_argument("--kl_anneal_epochs", type=int, default=0, help="Epochs over which to linearly anneal KL weight to beta")
    parser.add_argument("--sample_interval", type=int, default=5, help="Save reconstructions every N epochs")
    parser.add_argument("--sample_n", type=int, default=8, help="Number of images to reconstruct for sampling")
    parser.add_argument("--sample_dir", type=str, default="./samples", help="Directory to save sample grids")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if GPU is available")

    args = parser.parse_args()
    train(args) 