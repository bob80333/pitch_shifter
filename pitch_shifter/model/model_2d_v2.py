import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from k_diffusion.layers import FourierFeatures


class DownsampleWithSkip(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        factor=2,
        skip="pixel_shuffle",
        average_channels=2,
    ):
        super().__init__()
        # calculate kernel size
        # factor = 2 -> kernel_size = 3, padding = 1
        # factor = 1 -> kernel_size = 1, padding = 0
        if factor == 4:
            kernel = 5
        elif factor == 2:
            kernel = 3
        else:
            kernel = 1
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel,
            stride=factor,
            padding=factor // 2,
        )

        if skip == "pixel_shuffle":
            self.skip = nn.PixelUnshuffle(factor)
        else:
            self.skip = nn.Identity()

        self.avg_channels = average_channels

    def forward(self, x):
        res = x
        x = self.conv(x)
        res = self.skip(res)
        if self.avg_channels > 1:
            # average avg_channels factor of channels
            res = res.view(
                res.size(0),
                res.size(1) // self.avg_channels,
                self.avg_channels,
                res.size(2),
                res.size(3),
            )
            res = res.mean(dim=2)

        return x + res


class UpsampleWithSkip(nn.Module):
    def __init__(
        self, in_channels, out_channels, factor=2, skip="pixel_shuffle", dup_channels=2
    ):
        super().__init__()
        # calculate kernel size
        # factor = 2 -> kernel_size = 3, padding = 1
        # factor = 1 -> kernel_size = 1, padding = 0
        if factor == 4:
            kernel = 8
        elif factor == 2:
            kernel = 4
        else:
            kernel = 1
        self.conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel,
            stride=factor,
            padding=factor // 2,
        )

        if skip == "pixel_shuffle":
            self.skip = nn.PixelShuffle(factor)
        else:
            self.skip = nn.Identity()

        self.dup_channels = dup_channels

    def forward(self, x):
        res = x
        x = self.conv(x)
        res = self.skip(res)
        if self.dup_channels > 1:
            # duplicate dup_channels factor of channels
            res = res.view(res.size(0), res.size(1), 1, res.size(2), res.size(3))
            res = res.expand(-1, -1, self.dup_channels, -1, -1)
            res = res.contiguous().view(
                res.size(0), res.size(1) * self.dup_channels, res.size(3), res.size(4)
            )

        return x + res


class ConvNextBlock(nn.Module):
    def __init__(self, channels, expansion=4, layer_scale_init=1e-6):
        super().__init__()

        self.dw_conv = nn.Conv2d(
            channels, channels, kernel_size=7, padding=3, groups=channels
        )

        self.norm = nn.LayerNorm(channels)

        self.pw_conv1 = nn.Linear(channels, channels * expansion)
        self.pw_conv2 = nn.Linear(channels * expansion, channels)

        self.act = nn.GELU()

        self.gamma = (
            nn.Parameter(torch.ones(channels) * layer_scale_init, requires_grad=True)
            if layer_scale_init > 0
            else None
        )

    def forward(self, x):
        res = x
        x = self.dw_conv(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        x = self.pw_conv1(x)
        x = self.act(x)
        x = self.pw_conv2(x)

        if self.gamma is not None:
            x = x * self.gamma

        x = x.permute(0, 3, 1, 2).contiguous()

        return x + res

class Encoder(nn.Module):
    def __init__(self, channels, blocks, factors, scale_vs_channels):
        super().__init__()

        self.blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            for _ in range(blocks[i]):
                self.blocks.append(ConvNextBlock(channels[i]))
            self.blocks.append(
                DownsampleWithSkip(
                    channels[i],
                    channels[i + 1],
                    average_channels=scale_vs_channels[i],
                    factor=factors[i],
                )
            )


    def forward(self, x):
        residuals = [x]
        for block in self.blocks:
            x = block(x)
            if isinstance(block, DownsampleWithSkip):
                residuals.append(x)

        residuals.pop() # remove last residual

        return x, residuals


class Decoder(nn.Module):
    def __init__(self, channels, blocks, factors, scale_vs_channels):
        super().__init__()

        self.blocks = nn.ModuleList()
        for i in range(1, len(channels)):
            self.blocks.append(
                UpsampleWithSkip(
                    channels[i - 1],
                    channels[i],
                    dup_channels=scale_vs_channels[i - 1],
                    factor=factors[i - 1],
                )
            )
            for _ in range(blocks[i - 1]):
                self.blocks.append(ConvNextBlock(channels[i]))

            self.blocks.append(nn.Conv2d(channels[i]*2, channels[i], kernel_size=1))


    def forward(self, x, residuals):
        for block in self.blocks:
            # skip conv
            if isinstance(block, nn.Conv2d):
                x = torch.cat([x, residuals.pop()], dim=1)                
            x = block(x)

        return x


class UNet(nn.Module):
    def __init__(self, channels=None, blocks=None, factors=None, scale_vs_channels=None, bottleneck_blocks=4, input_channels=2):
        super().__init__()

        if channels is None:
            channels = [4, 16, 128, 256, 512]
            blocks = [1, 3, 4, 4]
            factors = [4, 4, 2, 2]
            scale_vs_channels = [4, 2, 2, 2]

            bottleneck_blocks = 4
        else:
            assert len(channels)-1 == len(blocks) == len(factors) == len(scale_vs_channels)

        self.encoder = Encoder(channels, blocks, factors, scale_vs_channels)
        self.decoder = Decoder(
            channels[::-1], blocks[::-1], factors[::-1], scale_vs_channels[::-1]
        )
        self.bottleneck = nn.Sequential(*[ConvNextBlock(channels[-1]) for _ in range(bottleneck_blocks)])

        self.conv_in = nn.Conv2d(input_channels, channels[0], kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(channels[0], input_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv_in(x)
        x, residuals = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, residuals)
        x = self.conv_out(x)

        return x


class AudioUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.autoenc = UNet()
        self.to_spec = T.Spectrogram(1022, 1022, 256, power=None)
        self.to_wav = T.InverseSpectrogram(1022, 1022, 256)

    def forward(self, x):
        # calculate padding
        pad = 0
        if (x.shape[-1] + 256) % 16384 != 0:
            pad = 16384 - (x.shape[-1] + 256) % 16384
        x = F.pad(x, (0, pad))
        x = self.to_spec(x)

        x = torch.view_as_real(x)
        x = x.permute(0, 3, 1, 2)

        y = self.autoenc(x)

        y = y.permute(0, 2, 3, 1).contiguous()
        y = torch.view_as_complex(y)
        y = self.to_wav(y)
        if pad > 0:
            y = y[:, :-pad]

        return y


if __name__ == "__main__":
    model = AudioUNet().to("cuda")
    opt_model = torch.compile(model)
    # print # model params
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"# params: {params/1e6:.2f}M")
    
    from time import time
    from tqdm import trange

    # warmup
    with torch.no_grad():
        for _ in trange(50):
            # 191*256, the spectrogram pads by 256 to 49152, giving exactly 512x192 2D shape
            x = torch.randn(32, 16384*2 - 256).to("cuda")
            y = opt_model(x)

    with torch.no_grad():
        start = time()
        for _ in trange(200):
            x = torch.randn(32, 16384*2 - 256).to("cuda")
            y = opt_model(x)
        end = time()
    print("Input shape", x.shape, "Output shape", y.shape)
    print("Took", end - start, "seconds")
