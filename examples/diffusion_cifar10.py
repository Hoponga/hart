# trying to get metal on my laptop working with a simple diffusion model for CIFAR-10 


from contextlib import contextmanager
from copy import deepcopy
import math


from matplotlib import pyplot as plt
import torch
from torch import optim, nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import datasets, transforms, utils
from torchvision.transforms import functional as TF
import torch.multiprocessing as mp 
from tqdm import tqdm, trange


assert(torch.mps.is_available())

device = torch.device("mps")

@contextmanager
def train_mode(model, mode=True):
    """A context manager that places a model into training mode and restores
    the previous mode on exit."""
    modes = [module.training for module in model.modules()]
    try:
        yield model.train(mode)
    finally:
        for i, module in enumerate(model.modules()):
            module.training = modes[i]


def eval_mode(model):
    """A context manager that places a model into evaluation mode and restores
    the previous mode on exit."""
    return train_mode(model, False)


@torch.no_grad()
def ema_update(model, averaged_model, decay):
    """Incorporates updated model parameters into an exponential moving averaged
    version of a model. It should be called after each optimizer step."""
    model_params = dict(model.named_parameters())
    averaged_params = dict(averaged_model.named_parameters())
    assert model_params.keys() == averaged_params.keys()

    for name, param in model_params.items():
        averaged_params[name].mul_(decay).add_(param, alpha=1 - decay)

    model_buffers = dict(model.named_buffers())
    averaged_buffers = dict(averaged_model.named_buffers())
    assert model_buffers.keys() == averaged_buffers.keys()

    for name, buf in model_buffers.items():
        averaged_buffers[name].copy_(buf)



# residual block that returns actual block + skip connection (defined by some block)

class ResidualBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return self.main(input) + self.skip(input)


# resnet block of 2 conv2d layers + skip connection 
class ResConvBlock(ResidualBlock):
    def __init__(self, c_in, c_mid, c_out, dropout_last=True):
        skip = None if c_in == c_out else nn.Conv2d(c_in, c_out, 1, bias=False)
        super().__init__([
            nn.Conv2d(c_in, c_mid, 3, padding=1),
            nn.Dropout2d(0.1, inplace=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_mid, c_out, 3, padding=1),
            nn.Dropout2d(0.1, inplace=True) if dropout_last else nn.Identity(),
            nn.ReLU(inplace=True),
        ], skip)

# residual block that concatenates skip connection output rather than adding elementwise 
class SkipBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = nn.Identity()

    def forward(self, input):
        return torch.cat([self.main(input), self.skip(input)], dim=1)


# we can learn temporal embeddings as "fourier features" rather than raw cos/sin embeddings 
class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = nn.Parameter(torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


def expand_to_planes(input, shape):
    return input[..., None, None].repeat([1, 1, shape[2], shape[3]])


class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        c = 64  # The base channel count

        # The inputs to timestep_embed will approximately fall into the range
        # -10 to 10, so use std 0.2 for the Fourier Features.
        self.timestep_embed = FourierFeatures(1, 16, std=0.2)
        self.class_embed = nn.Embedding(10, 4)

        # Unet model 

        self.net = nn.Sequential(   # 32x32
            ResConvBlock(3 + 16 + 4, c, c),
            ResConvBlock(c, c, c),
            SkipBlock([
                nn.AvgPool2d(2),  # 32x32 -> 16x16
                ResConvBlock(c, c * 2, c * 2),
                ResConvBlock(c * 2, c * 2, c * 2),
                SkipBlock([
                    nn.AvgPool2d(2),  # 16x16 -> 8x8
                    ResConvBlock(c * 2, c * 4, c * 4),
                    ResConvBlock(c * 4, c * 4, c * 4),
                    SkipBlock([
                        nn.AvgPool2d(2),  # 8x8 -> 4x4
                        ResConvBlock(c * 4, c * 8, c * 8),
                        ResConvBlock(c * 8, c * 8, c * 8),
                        ResConvBlock(c * 8, c * 8, c * 8),
                        ResConvBlock(c * 8, c * 8, c * 4),
                        nn.Upsample(scale_factor=2),
                    ]),  # 4x4 -> 8x8
                    ResConvBlock(c * 8, c * 4, c * 4),
                    ResConvBlock(c * 4, c * 4, c * 2),
                    nn.Upsample(scale_factor=2),
                ]),  # 8x8 -> 16x16
                ResConvBlock(c * 4, c * 2, c * 2),
                ResConvBlock(c * 2, c * 2, c),
                nn.Upsample(scale_factor=2),
            ]),  # 16x16 -> 32x32
            ResConvBlock(c * 2, c, c),
            ResConvBlock(c, c, 3, dropout_last=False),
        )

    def forward(self, input, log_snrs, cond):
        timestep_embed = expand_to_planes(self.timestep_embed(log_snrs[:, None]), input.shape)
        class_embed = expand_to_planes(self.class_embed(cond), input.shape)
        return self.net(torch.cat([input, class_embed, timestep_embed], dim=1))


def get_alphas_sigmas(log_snrs):
    """Returns the scaling factors for the clean image (alpha) and for the
    noise (sigma), given the log SNR for a timestep."""
    return log_snrs.sigmoid().sqrt(), log_snrs.neg().sigmoid().sqrt()


def get_ddpm_schedule(t):
    """Returns log SNRs for the noise schedule from the DDPM paper."""
    return -torch.special.expm1(1e-4 + 10 * t**2).log()


@torch.no_grad()
def sample(model, x, steps, eta, classes):
    """Draws samples from a model given starting noise."""
    ts = x.new_ones([x.shape[0]])

    # Create the noise schedule
    t = torch.linspace(1, 0, steps + 1)[:-1]
    log_snrs = get_ddpm_schedule(t)
    alphas, sigmas = get_alphas_sigmas(log_snrs)

    # The sampling loop
    for i in trange(steps):

        # autocast works on metal? 
        with torch.autocast(device_type = "mps", dtype = torch.bfloat16):
            v = model(x, ts * log_snrs[i], classes).float()

  
        pred = x * alphas[i] - v * sigmas[i]
        eps = x * sigmas[i] + v * alphas[i]

  
        if i < steps - 1:
            ddim_sigma = eta * (sigmas[i + 1]**2 / sigmas[i]**2).sqrt() * \
                (1 - alphas[i]**2 / alphas[i + 1]**2).sqrt()
            adjusted_sigma = (sigmas[i + 1]**2 - ddim_sigma**2).sqrt()

  
            x = pred * alphas[i + 1] + eps * adjusted_sigma

         
            if eta:
                x += torch.randn_like(x) * ddim_sigma

  
    return pred


t_vis = torch.linspace(0, 1, 1000)
log_snrs_vis = get_ddpm_schedule(t_vis)
alphas_vis, sigmas_vis = get_alphas_sigmas(log_snrs_vis)



if __name__ == "__main__": 
    mp.set_start_method('spawn', force = True)


    batch_size = 100

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    train_set = datasets.CIFAR10('data', train=True, download=True, transform=tf)
    train_dl = data.DataLoader(train_set, batch_size, shuffle=True,
                            num_workers=4, persistent_workers=True, pin_memory=True)
    val_set = datasets.CIFAR10('data', train=False, download=True, transform=tf)
    val_dl = data.DataLoader(val_set, batch_size,
                            num_workers=4, persistent_workers=True, pin_memory=True)



    # Create the model and optimizer

    seed = 0

    print('Using device:', device)
    torch.manual_seed(0)

    model = Diffusion().to(device)
    model_ema = deepcopy(model)
    print('Model parameters:', sum(p.numel() for p in model.parameters()))

    opt = optim.Adam(model.parameters(), lr=2e-4)
    scaler = torch.amp.GradScaler()
    epoch = 0

    # Use a low discrepancy quasi-random sequence to sample uniformly distributed
    # timesteps. This considerably reduces the between-batch variance of the loss.
    rng = torch.quasirandom.SobolEngine(1, scramble=True)


    # Actually train the model

    ema_decay = 0.998

    # The number of timesteps to use when sampling
    steps = 500

    # The amount of noise to add each timestep when sampling
    # 0 = no noise (DDIM)
    # 1 = full noise (DDPM)
    eta = 1.


    def eval_loss(model, rng, reals, classes):
        # Draw uniformly distributed continuous timesteps
        t = rng.draw(reals.shape[0])[:, 0].to(device)

        # Calculate the noise schedule parameters for those timesteps
        log_snrs = get_ddpm_schedule(t)
        alphas, sigmas = get_alphas_sigmas(log_snrs)
        weights = log_snrs.exp() / log_snrs.exp().add(1)

        # Combine the ground truth images and the noise
        alphas = alphas[:, None, None, None]
        sigmas = sigmas[:, None, None, None]
        noise = torch.randn_like(reals)
        noised_reals = reals * alphas + noise * sigmas
        targets = noise * alphas - reals * sigmas

        # Compute the model output and the loss.
        with torch.autocast(device_type = "mps", dtype = torch.bfloat16):
            v = model(noised_reals, log_snrs, classes)
            return (v - targets).pow(2).mean([1, 2, 3]).mul(weights).mean()


    def train():
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
    @torch.random.fork_rng()
    @eval_mode(model_ema)
    def val():
        tqdm.write('\nValidating...')
        torch.manual_seed(seed)
        rng = torch.quasirandom.SobolEngine(1, scramble=True)
        total_loss = 0
        count = 0
        for i, (reals, classes) in enumerate(tqdm(val_dl)):
            reals = reals.to(device)
            classes = classes.to(device)

            loss = eval_loss(model_ema, rng, reals, classes)

            total_loss += loss.item() * len(reals)
            count += len(reals)
        loss = total_loss / count
        tqdm.write(f'Validation: Epoch: {epoch}, loss: {loss:g}')


    @torch.no_grad()
    @torch.random.fork_rng()
    @eval_mode(model_ema)
    def demo():
        tqdm.write('\nSampling...')
        torch.manual_seed(seed)

        noise = torch.randn([100, 3, 32, 32], device=device)
        fakes_classes = torch.arange(10, device=device).repeat_interleave(10, 0)
        fakes = sample(model_ema, noise, steps, eta, fakes_classes)

        grid = utils.make_grid(fakes, 10).cpu()
        filename = f'demo_{epoch:05}.png'
        TF.to_pil_image(grid.add(1).div(2).clamp(0, 1)).save(filename)
        tqdm.write('')


    def save():
        filename = 'cifar_diffusion.pth'
        obj = {
            'model': model.state_dict(),
            'model_ema': model_ema.state_dict(),
            'opt': opt.state_dict(),
            'scaler': scaler.state_dict(),
            'epoch': epoch,
        }
        torch.save(obj, filename)


    try:
        val()
        demo()
        while True:
            print('Epoch', epoch)
            train()
            epoch += 1
            if epoch % 5 == 0:
                val()
                demo()
            save()
    except KeyboardInterrupt:
        pass


@torch.no_grad()
def sample_eps(model, x, steps, eta, classes):
    """Draw samples from a model trained to predict \hat{\varepsilon}_t (noise).

    The parameter `eta` interpolates between deterministic DDIM (eta=0) and
    fully-stochastic DDPM (eta=1).  Values in-between produce a blend of the
    two.  The function mirrors the logic of `sample` above but adapts it to a
    network that outputs the noise directly instead of the velocity.
    """

    ts = x.new_ones([x.shape[0]])

    # Create the noise schedule (same as DDPM)
    t = torch.linspace(1, 0, steps + 1)[:-1]
    log_snrs = get_ddpm_schedule(t)
    alphas, sigmas = get_alphas_sigmas(log_snrs)

    for i in trange(steps):
        # Predict \hat{epsilon}_t
        with torch.autocast(device_type="mps", dtype=torch.bfloat16):
            eps = model(x, ts * log_snrs[i], classes).float()

        # Reconstruct \hat{x}_0 from the predicted noise
        x0_pred = (x - sigmas[i] * eps) / alphas[i]

        if i < steps - 1:
            # DDIM/DDPM variance splitting
            ddim_sigma = eta * (sigmas[i + 1]**2 / sigmas[i]**2).sqrt() * \
                (1 - alphas[i]**2 / alphas[i + 1]**2).sqrt()
            adjusted_sigma = (sigmas[i + 1]**2 - ddim_sigma**2).sqrt()

            # Deterministic component of the step
            x = x0_pred * alphas[i + 1] + eps * adjusted_sigma

            # Stochastic component (only if eta>0)
            if eta:
                x += torch.randn_like(x) * ddim_sigma

    # After the final step `x0_pred` is the denoised image.
    return x0_pred


# Convenience wrappers ---------------------------------------------------------

def sample_ddpm_eps(model, x, steps, classes):
    """DDPM sampling for an epsilon-prediction network (eta = 1)."""
    return sample_eps(model, x, steps, eta=1.0, classes=classes)


def sample_ddim_eps(model, x, steps, classes):
    """DDIM sampling for an epsilon-prediction network (eta = 0)."""
    return sample_eps(model, x, steps, eta=0.0, classes=classes)


