import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

class ResBlock(nn.Module):
    def __init__(self, channels, dilation, padding):
        super().__init__()

        self.pad = nn.ReplicationPad1d(padding)
        self.conv1 = weight_norm(nn.Conv1d(channels, channels, 3, padding=0, dilation=dilation))
        self.conv2 = weight_norm(nn.Conv1d(channels, channels, 1))

        self.conv_skip = weight_norm(nn.Conv1d(channels, channels, 1))
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        res = x
        x = self.act(x)
        x = self.pad(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        res = self.conv_skip(res)
        return x + res
    
class ResStack(nn.Module):
    def __init__(self, channels, num_blocks=4):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(ResBlock(channels, 3**i, 3**i))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
    
class Encoder(nn.Module):
    def __init__(self, channels, factors, blocks):
        super().__init__()

        self.blocks = nn.ModuleList()
        for i in range(1, len(channels)):
            if i > 1:
                self.blocks.append(nn.LeakyReLU(0.2))
            self.blocks.append(
                nn.Conv1d(channels[i - 1], channels[i], factors[i-1] * 2, stride = factors[i-1], padding = factors[i-1] // 2)
            )
            for _ in range(blocks[i - 1]):
                self.blocks.append(ResStack(channels[i]))

    def forward(self, x):
        residuals = []
        for block in self.blocks:
            x = block(x)
            if isinstance(block, nn.Conv1d):
                residuals.append(x)
        return x, residuals
    

class Decoder(nn.Module):
    def __init__(self, channels, factors, blocks):
        super().__init__()

        self.blocks = nn.ModuleList()
        for i in range(1, len(channels)):
            for _ in range(blocks[i - 1]):
                self.blocks.append(ResStack(channels[i - 1]))

            self.blocks.append(nn.LeakyReLU(0.2))
            self.blocks.append(
                nn.ConvTranspose1d(channels[i - 1] * 2, channels[i], factors[i-1] * 2, stride = factors[i-1], padding = factors[i-1] // 2)
            )

    def forward(self, x, residuals):
        for block in self.blocks:
            if isinstance(block, nn.ConvTranspose1d):
                x = torch.cat([x, residuals.pop()], dim=1)
            x = block(x)
        return x
    
class MelGANUNet(nn.Module):
    def __init__(self):
        super().__init__()
        channels = [1, 8, 64, 256, 512]
        factors = [8, 8, 4, 2]
        blocks = [1, 1, 1, 1]

        self.encoder = Encoder(channels, factors, blocks)
        self.decoder = Decoder(channels[::-1], factors[::-1], blocks[::-1])

        self.out_act = nn.Tanh()

    def forward(self, x):
        x, residuals = self.encoder(x)
        x = self.decoder(x, residuals)
        x = self.out_act(x)
        return x
    
if __name__ == "__main__":
    model = MelGANUNet().to("cuda")
    # print # model params
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"# params: {params/1e6:.2f}M")
    x = torch.randn(64, 1, 16384*3).to("cuda")
    print(x.shape)
    from time import time
    from tqdm import trange

    with torch.no_grad():
        start = time()
        for _ in trange(100):
            y = model(x)
        end = time()
    print("Input shape", x.shape, "Output shape", y.shape)
    print("Took", end - start, "seconds")
