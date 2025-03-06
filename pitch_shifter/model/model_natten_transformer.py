from pitch_shifter.model import pixelshuffle1d
import torch
import torch.nn as nn
import natten

class FF(nn.Module):
    def __init__(self, channels, expansion, act=nn.GELU):
        super().__init__()
        
        self.ff1 = nn.Linear(channels, channels * expansion)
        self.act = act()
        self.ff2 = nn.Linear(channels * expansion, channels)

    def forward(self, x):
        x = self.ff1(x)
        x = self.act(x)
        x = self.ff2(x)
        return x
    

class TransformerLayer(nn.Module):
    def __init__(self, dim, dim_head=64, mlp_expansion=4, kernel_size=127):
        super().__init__()
        
        self.attn = natten.NeighborhoodAttention1D(dim, dim // dim_head, kernel_size=kernel_size, dilation=1, qkv_bias=True, rel_pos_bias=True)
        self.norm1 = nn.LayerNorm(dim)
        
        self.mlp = FF(dim, mlp_expansion)
        self.norm2 = nn.LayerNorm(dim)
            
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
    
class AudioTransformer(nn.Module):
    def __init__(self, dim=512, num_layers=8, patching=32):
        super().__init__()

        self.patch_down = pixelshuffle1d.PixelUnshuffle1D(patching)
        self.patch_up = pixelshuffle1d.PixelShuffle1D(patching)

        self.input_proj = nn.Linear(patching, dim)
        self.output_proj = nn.Linear(dim, patching)
        self.output_act = nn.Tanh()
        
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(TransformerLayer(dim))
        
    def forward(self, x):
        # patchify
        x = self.patch_down(x)
        # channels last
        x = x.permute(0, 2, 1)
        x = self.input_proj(x)

        for layer in self.layers:
            x = layer(x)

        # output projection
        x = self.output_proj(x)
        x = self.output_act(x)
        # channels first
        x = x.permute(0, 2, 1)
        # unpatchify
        x = self.patch_up(x)
        return x
    

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AudioTransformer().to(device)
    opt_model = torch.compile(model)

    torch.backends.cuda.matmul.allow_tf32 = True
    # print # model params
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"# params: {params/1e6:.2f}M")
    x = torch.randn(32, 1, 16384*2).to("cuda")
    print(x.shape)
    from time import time
    from tqdm import trange
    # warmup
    with torch.no_grad():
        for _ in trange(50):
            x = torch.randn(32, 1, 16384*2).to("cuda")
            y = opt_model(x)

    with torch.no_grad():
        start = time()
        for _ in trange(200):
            x = torch.randn(32, 1, 16384*2).to("cuda")
            y = opt_model(x)
        end = time()
    print("Input shape", x.shape, "Output shape", y.shape)
    print("Took", end - start, "seconds")
    print("Nans?", y.isnan().any())