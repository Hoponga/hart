# Simple vision transformer for image classification 
import torch 
import torch.nn as nn 


# (B, C, H, W) -> (B, num_patches, hidden_size) 
class PatchEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.image_size = config["image_size"]
        self.patch_size = config["patch_size"]
        self.num_channels = config["num_channels"]
        self.hidden_size = config["hidden_size"]
        
        # Calculate the number of patches from the image size and patch size

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.projection = nn.Conv2d(self.num_channels, self.hidden_size, kernel_size=self.patch_size, stride=self.patch_size)

    # convert 
    def forward(self, x):
        # (batch_size, num_channels, image_size, image_size) -> (batch_size, num_patches, hidden_size)
        #(x.shape)
        x = self.projection(x)
        #print(x.shape)
        x = x.flatten(2).transpose(1, 2)
        #print(x.shape)
        return x

class Embeddings(nn.Module): 
    # Combine the patch embeddings (learned) with [CLS] and position embeddings 
    def __init__(self, config): 
        super().__init__()
        self.config = config 
        self.patch_embeddings = PatchEmbeddings(config)
        self.cls_token = nn.Parameter(torch.randn(1, 1, config["hidden_size"]))
        self.num_patches = (config["image_size"] // config["patch_size"]) ** 2
        self.position_embeddings = nn.Parameter(torch.randn(1, self.num_patches + 1, config["hidden_size"]))
        self.dropout = nn.Dropout(config["dropout"])

    def forward(self, x): 
        x = self.patch_embeddings(x) # (batch_size, num_patches, hidden_size)
        batch_size, _, _ = x.shape 
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        #print(x.shape, self.position_embeddings.shape)
        x = x + self.position_embeddings
        return self.dropout(x)
    
class AttentionHead(nn.Module):
    def __init__(self, hidden_size, attention_head_size, dropout, bias=True):
        super().__init__()
        self.hidden_size = hidden_size 
        self.attention_head_size = attention_head_size 
        self.dropout = nn.Dropout(dropout) 
        
        self.query = nn.Linear(self.hidden_size, self.attention_head_size, bias=bias)
        self.key = nn.Linear(self.hidden_size, self.attention_head_size, bias=bias)
        self.value = nn.Linear(self.hidden_size, self.attention_head_size, bias=bias)


    def forward(self, x): 
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        # normalize by d_k 
        attention_scores =nn.Softmax(dim = -1)(torch.matmul(query, key.transpose(-2, -1)) / (self.attention_head_size ** 0.5))
        attention_scores = self.dropout(attention_scores)
        return (torch.matmul(attention_scores, value), attention_scores)


# todo: implement MLA or flash? 
class MHA(nn.Module): 
    def __init__(self, hidden_size, num_heads, dropout, bias=True): 
        super().__init__()
        self.hidden_size = hidden_size 
        self.num_heads = num_heads 
        self.dropout = nn.Dropout(dropout) 
        self.head_dim = hidden_size // num_heads 

        self.all_head_size = self.num_heads * self.head_dim 
        self.heads = nn.ModuleList([AttentionHead(hidden_size, self.head_dim, dropout, bias) for _ in range(num_heads)])
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(self.all_head_size, hidden_size)

    def forward(self, x): 
        # (batch_size, num_patches, hidden_size)
        attention_outputs = [head(x) for head in self.heads] # head returns output and scores 
        attention_output = torch.cat([output for output, _ in attention_outputs], dim=-1)
        attention_probs = torch.stack([scores for _, scores in attention_outputs], dim=1)
        return self.dropout(self.output(attention_output)), attention_probs

class FFN(nn.Module): 
    def __init__(self, hidden_size, intermediate_size, dropout = 0.3): 
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size), 
            nn.GELU(), 
            nn.Linear(intermediate_size, hidden_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x): 
        return self.dropout(self.ffn(x))

class TransformerBlock(nn.Module): 
    def __init__(self, hidden_size, num_heads, intermediate_size, dropout = 0.3): 
        super().__init__()
        self.attention = MHA(hidden_size, num_heads, dropout)
        self.ffn = FFN(hidden_size, intermediate_size, dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
    def forward(self, x): 
        attention_outputs, attention_probs = self.attention(self.norm1(x))
        x = x + attention_outputs
        x = x + self.ffn(self.norm2(x))
        return x, attention_probs

class Encoder(nn.Module): 
    # In the encoder, we have a series of transformer blocks 
    def __init__(self, hidden_size, num_heads, intermediate_size, num_layers, dropout = 0.3):   
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock(hidden_size, num_heads, intermediate_size, dropout) for _ in range(num_layers)])

    def forward(self, x):
        total_attention_probs = []
        for layer in self.layers: 
            x, attention_probs = layer(x)
            total_attention_probs.append(attention_probs)
        return x, total_attention_probs
    
class ViT(nn.Module): 
    def __init__(self, config): 
        super().__init__()
        self.embeddings = Embeddings(config)
        self.encoder = Encoder(config["hidden_size"], config["num_heads"], config["intermediate_size"], config["num_layers"], config["dropout"])
        self.classifier = nn.Linear(config["hidden_size"], config["num_classes"])

    def forward(self, x): 
        #self.embeddings returns (batch_size, num_patches + 1, hidden_size)
        x = self.embeddings(x)

        # The output of the encoder is (batch_size, num_patches + 1, hidden_size)
        # Why? Because we have a [CLS] token in the embeddings. Then, every transformer block returns one value for every token. 
        # So, we have one value for the [CLS] token and one value for every other token. 
        
        
        x, attention_probs = self.encoder(x)
        x = self.classifier(x[:, 0])
        return x, attention_probs



        

        
