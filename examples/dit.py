import torch 
import torch.nn as nn 
from einops import rearrange, repeat
import numpy as np 
from vit import AttentionHead, MHA, FFN 
from diffusion_cifar10 import FourierFeatures

# Helper functions for 2d positional encoding -- taken from CS294 notebook  
def get_2d_sincos_pos_embed_from_grid(embed_dim, grid_shape):
    """Return 2-D sinusoidal positional embeddings as a torch tensor.

    Args:
        embed_dim (int): total embedding dimension (must be even).
        grid_shape (tuple or sequence of tensors): either a tuple (H, W) or
            a tuple of coordinate tensors produced by torch.meshgrid.

    Returns:
        torch.Tensor with shape (H*W, embed_dim).
    """
    assert embed_dim % 2 == 0, "embed_dim must be even."

    # Build coordinate grids if integer sizes are provided
    if isinstance(grid_shape, tuple) and isinstance(grid_shape[0], int):
        H, W = grid_shape
        if 'indexing' in torch.meshgrid.__code__.co_varnames:
            grid_y, grid_x = torch.meshgrid(
                torch.arange(H, dtype=torch.float32),
                torch.arange(W, dtype=torch.float32),
                indexing='ij'
            )
        else:
            grid_y, grid_x = torch.meshgrid(
                torch.arange(H, dtype=torch.float32),
                torch.arange(W, dtype=torch.float32)
            )
    else:
        # grid_shape already contains coordinate tensors (y, x)
        grid_y, grid_x = grid_shape

    emb_y = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_y)
    emb_x = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_x)

    emb = torch.cat([emb_y, emb_x], dim=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """Compute 1-D sinusoidal positional embeddings using torch.

    Args:
        embed_dim (int): total embedding dimension (must be even).
        pos (torch.Tensor): positions (any shape) on **current device**.

    Returns:
        torch.Tensor of shape (*pos.shape, embed_dim) flattened as (M, D).
    """
    assert embed_dim % 2 == 0, "embed_dim must be even."
    print("embed_dim: ", embed_dim)

    # Ensure tensor type / device are inherited from `pos`.
    dtype  = pos.dtype if torch.is_floating_point(pos) else torch.float32
    device = pos.device
    print("dtype: ", dtype)

    omega = torch.arange(embed_dim // 2, dtype=dtype, device=device)
    omega = 1.0 / (float(10000) ** (omega / (embed_dim / 2)))  # (D/2,)
    print("omega: ", omega)
    pos = pos.reshape(-1).to(dtype)  # (M,)
    out = torch.einsum('m,d->md', pos, omega)  # (M, D/2)

    emb = torch.cat([out.sin(), out.cos()], dim=1)  # (M, D)
    return emb

def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """Convenience wrapper that builds a square H=W=grid_size embedding."""
    # Use torch operations so the result is a tensor.
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_w, grid_h)  # note: width, height order matches original comment
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)




class DITPatchEmbeddings(nn.Module):
    """
    Convert the image into patches and then project them into a vector space.
    """

    def __init__(self, image_height, image_width, patch_height, patch_width, num_channels, hidden_size):
        super().__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.num_channels = num_channels
        self.hidden_size = hidden_size


        # convert a single patch into a transformer token 
        self.patch_to_token = nn.Linear(self.patch_height * self.patch_width * self.num_channels, self.hidden_size)

        self.patch_to_token.weight.data.normal_(mean=0.0, std=0.02)
        self.patch_to_token.bias.data.zero_()


    # convert 
    # x comes in as (B, C, H, W)

    def forward(self, x):
        num_along_h = self.image_height // self.patch_height
        num_along_w = self.image_width // self.patch_width 

        # conv chunkify 
        patches = rearrange(x, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=self.patch_height, p2=self.patch_width)

        # embed these patches 
        patches = self.patch_to_token(patches) # so now we have (B, num_patches, hidden_size) tokens, like an "image sentence"

        # 2d sinusoidal posiitonal encoding 
        # this gets us (num_patches, hidden_size) positional encodings 
        print("hi")

        pos_enc = get_2d_sincos_pos_embed_from_grid(self.hidden_size, (num_along_h, num_along_w)
        ).to(device=patches.device, dtype=patches.dtype)  # (L, D)

        patches = patches + pos_enc.unsqueeze(0)  # broadcast over batch

        return patches





class DITLayer(nn.Module): 
    def __init__(self, hidden_size, intermediate_size, num_heads, dropout = 0.3): 
        super().__init__()
        self.attention = MHA(hidden_size, num_heads, dropout)
        self.ffn = FFN(hidden_size, intermediate_size, dropout)
        self.num_heads = num_heads 

        self.norm1 = nn.LayerNorm(hidden_size, eps = 1e-6, elementwise_affine = False)
        self.norm2 = nn.LayerNorm(hidden_size, eps = 1e-6, elementwise_affine = False)

        # take conditioning input and use it to predict 6 things for adaptive normalization: 
        # shift, scale, gate (for attention), shift, scale, gate (for ffn) 
        # gate is the weight of the attention output in the residual connection 

        self.cond_proj = nn.Sequential(
            nn.SiLU(), 
            nn.Linear(hidden_size, 6 * hidden_size)
        )

        # self.cond_proj.weight.data.normal_(mean=0.0, std=0.02)
        # self.cond_proj.bias.data.zero_()
    def forward(self, x, cond): 
        # x is (B, num_patches, hidden_size)
        # cond is (B, hidden_size)

        # project cond into 6 * hidden_size 
        cond_out = self.cond_proj(cond).chunk(6, dim = 1) 
        scale_attn_input, shift_attn_input, gate_attn, scale_ffn_input, shift_ffn_output, gate_ffn = cond_out
        
        '''
        h = LayerNorm(hidden_size, elementwise_affine=False)(x)
        h = modulate(h, shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * Attention(hidden_size, num_heads)(h)
        
        h = LayerNorm(hidden_size, elementwise_affine=False)(x)
        h = modulate(h, shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * MLP(hidden_size)(h)'''

        res = x 
        x = self.norm1(x) 
        x = modulate(x, shift_attn_input, scale_attn_input)
        attn_output = res + gate_attn.unsqueeze(1) * self.attention(x)

        res = modulate(self.norm2(attn_output), shift_ffn_output, scale_ffn_output)
        res = attn_output + gate_ffn.unsqueeze(1) * self.ffn(res)


        return res 
        # split x into num_heads groups 

# Convert (B, num_patches, hidden_size) to output noise prediction of size (B, num_patches, patch_height*patch_width*num_channels)
class DITDecoder(nn.Module): 
    def __init__(self, hidden_size, patch_height, patch_width, num_channels): 
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.num_channels = num_channels

        self.cond_proj = nn.Sequential(
            nn.SiLU(), 
            nn.Linear(hidden_size, 2 * hidden_size)
        )
        # self.cond_proj.weight.data.normal_(mean=0.0, std=0.02)
        # self.cond_proj.bias.data.zero_()
        
        self.output_proj = nn.Linear(hidden_size, patch_height * patch_width * num_channels)
        self.output_proj.weight.data.normal_(mean=0.0, std=0.02)
        self.output_proj.bias.data.zero_()


    def forward(self, x, cond): 
        '''Given x (B x L x D), c (B x D)
        c = SiLU()(c)
        c = Linear(hidden_size, 2 * hidden_size)(c)
        shift, scale = c.chunk(2, dim=1)
        x = LayerNorm(hidden_size, elementwise_affine=False)(x)
        x = modulate(x, shift, scale)
        x = Linear(hidden_size, patch_size * patch_size * out_channels)(x)
        return x'''
        cond_out = self.cond_proj(cond).chunk(2, dim = 1)
        shift, scale = cond_out

        x = self.norm1(x)
        x = modulate(x, shift, scale)
        x = self.output_proj(x)
        return x 

def unpatchify(x, patch_height, patch_width, num_channels): 
    B, L, D = x.shape 

# denoiser model 
class DIT(nn.Module): 
    def __init__(self, config): 
        super().__init__()
        self.config = config 

        self.patch_embeddings = DITPatchEmbeddings(config["image_height"], config["image_width"], config["patch_height"], config["patch_width"], config["num_channels"], config["hidden_size"])
        self.layers = nn.ModuleList([DITLayer(config["hidden_size"], config["intermediate_size"], config["num_heads"], config["dropout"]) for _ in range(config["num_layers"])])

        self.output = nn.Linear(config["hidden_size"], config["num_classes"])

        self.timestep_embed = FourierFeatures(1, config["timestep_embedding_size"], std=0.2)


        # project timestep 
        self.timestep_proj = nn.Sequential(
            nn.SiLU(), 
            nn.Linear(config["timestep_embedding_size"], config["hidden_size"])
        )

        self.cond_proj = nn.Sequential(
            nn.SiLU(), 
            nn.Linear(config["conditioning_size"], config["hidden_size"])
        )

        # self.cond_proj.weight.data.normal_(mean=0.0, std=0.02)
        # self.cond_proj.bias.data.zero_()

        # self.timestep_proj.weight.data.normal_(mean=0.0, std=0.02)
        # self.timestep_proj.bias.data.zero_()

        self.layers = nn.ModuleList([DITLayer(config["hidden_size"], config["intermediate_size"], config["num_heads"], config["dropout"]) for _ in range(config["num_layers"])])
        self.decoder = DITDecoder(config["hidden_size"], config["patch_height"], config["patch_width"], config["num_channels"])

        if config["training"]: 
            self.classifier = nn.Linear(config["hidden_size"], config["num_classes"])
            self.classifier.weight.data.normal_(mean=0.0, std=0.02)
            self.classifier.bias.data.zero_()
        else: 
            self.classifier = None 

    # Remember t includes the class embedding too! 
    def forward(self, x, t, cond): 
        
        t = self.timestep_embed(t)
        print("x shape: ", x.shape)
        print("t shape: ", t.shape)
        print("cond shape: ", cond.shape)

        x = self.patch_embeddings(x)
        print("patch embeddings okay") 

        t = self.timestep_proj(t)
        print("timestep proj okay")

        cond = self.cond_proj(cond.to(torch.float32))
        print("cond proj okay")

        print("after x shape: ", x.shape)
        print("after t shape: ", t.shape)
        print("after cond shape: ", cond.shape)

        t = t + cond 

        for layer in self.layers: 
            x = layer(x, t)

        x = self.decoder(x, t)
        # now x is of shape (B, num_patches, patch_height*patch_width*num_channels)
        # reconstruct the image by rearranging from (num_patches, patch_data) to (c, h, w) 

        x = rearrange(x, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)", p1=self.config["patch_height"], p2=self.config["patch_width"])

        return x 
        




