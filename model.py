import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T


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
        for i in range(1, len(channels)):
            self.blocks.append(
                DownsampleWithSkip(
                    channels[i - 1],
                    channels[i],
                    average_channels=scale_vs_channels[i - 1],
                    factor=factors[i - 1],
                )
            )
            for _ in range(blocks[i - 1]):
                self.blocks.append(ConvNextBlock(channels[i]))

    def forward(self, x):
        residuals = []
        for block in self.blocks:
            if isinstance(block, DownsampleWithSkip):
                residuals.append(x)
            x = block(x)

        return x, residuals


class Decoder(nn.Module):
    def __init__(self, channels, blocks, factors, scale_vs_channels):
        super().__init__()

        self.blocks = nn.ModuleList()
        for i in range(1, len(channels)):
            for _ in range(blocks[i - 1]):
                self.blocks.append(ConvNextBlock(channels[i - 1]))
            self.blocks.append(
                UpsampleWithSkip(
                    channels[i - 1],
                    channels[i],
                    dup_channels=scale_vs_channels[i - 1],
                    factor=factors[i - 1],
                )
            )

    def forward(self, x, residuals):
        for block in self.blocks:
            x = block(x)
            if isinstance(block, UpsampleWithSkip):
                residual = residuals.pop()
                print(x.shape, residual.shape)
                x = x + residual

        return x


class AutoEncoder(nn.Module):
    def __init__(self, channels=None, blocks=None):
        super().__init__()

        if channels is None:
            channels = [2, 32, 128, 256]
            blocks = [2, 2, 2]
            factors = [4, 2, 2]
            scale_vs_channels = [1, 1, 2]

        self.encoder = Encoder(channels, blocks, factors, scale_vs_channels)
        self.decoder = Decoder(
            channels[::-1], blocks[::-1], factors[::-1], scale_vs_channels[::-1]
        )

        # self.in_conv = nn.Conv2d(3, channels[0], kernel_size=1, padding=0)

        # self.out_conv = nn.Conv2d(channels[0], 3, kernel_size=1, padding=0)

        self.out_act = nn.Tanh()

    def forward(self, x):
        x, residuals = self.encode(x)
        x = self.decode(x, residuals)

        return x

    def encode(self, x):
        # x = self.in_conv(x)
        x, residuals = self.encoder(x)

        return x, residuals

    def decode(self, x, residuals):
        x = self.decoder(x, residuals)
        # x = self.out_conv(x)
        x = self.out_act(x)

        return x


if __name__ == "__main__":
    model = AutoEncoder()
    # print # model params
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"# params: {params/1e6:.2f}M")
    x = torch.randn(8, 1, 65024)

    # convert to spectrogram
    spec = T.Spectrogram(2046, 2046, 512, power=None)

    x = spec(x)
     # Split complex input into real and imaginary channels
    x_real = x.real
    x_imag = x.imag
    x = torch.cat([x_real, x_imag], dim=1)  # Stack along channel dimension
    print(x.shape)
    from time import time

    with torch.no_grad():
        start = time()
        y = model(x)
        end = time()
    print("Input shape", x.shape, "Output shape", y.shape)
    print("Took", end - start, "seconds")

