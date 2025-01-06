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


class ConvNextAdaLNZeroBlock(nn.Module):
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

        self.adaln_modulation = nn.Sequential(nn.SiLU(), nn.Linear(128, channels * 6))

    def forward(self, x, pitch_shift):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaln_modulation(pitch_shift).chunk(6, dim=1)
        )
        res = x
        x = x * scale_msa[:, :, None, None] + shift_msa[:, :, None, None]
        x = self.dw_conv(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x * gate_msa[:, None, None, :]
        x = self.norm(x)
        x = x * scale_mlp[:, None, None, :] + shift_mlp[:, None, None, :]
        x = self.pw_conv1(x)
        x = self.act(x)
        x = self.pw_conv2(x)
        x = x * gate_mlp[:, None, None, :]

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
                self.blocks.append(ConvNextAdaLNZeroBlock(channels[i]))

    def forward(self, x, pitch_shift):
        residuals = []
        for block in self.blocks:
            if isinstance(block, DownsampleWithSkip):
                residuals.append(x)
            if isinstance(block, ConvNextAdaLNZeroBlock):
                x = block(x, pitch_shift)
            else:
                x = block(x)

        return x, residuals


class Decoder(nn.Module):
    def __init__(self, channels, blocks, factors, scale_vs_channels):
        super().__init__()

        self.blocks = nn.ModuleList()
        for i in range(1, len(channels)):
            for _ in range(blocks[i - 1]):
                self.blocks.append(ConvNextAdaLNZeroBlock(channels[i - 1]))
            self.blocks.append(
                UpsampleWithSkip(
                    channels[i - 1],
                    channels[i],
                    dup_channels=scale_vs_channels[i - 1],
                    factor=factors[i - 1],
                )
            )

    def forward(self, x, residuals, pitch_shift):
        for block in self.blocks:
            if isinstance(block, ConvNextAdaLNZeroBlock):
                x = block(x, pitch_shift)
            else:
                x = block(x)
            if isinstance(block, UpsampleWithSkip):
                residual = residuals.pop()
                x = x + residual

        return x


class UNet(nn.Module):
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

        self.timestep_embed = FourierFeatures(1, 128)

    def forward(self, x, pitch_shift=0):
        pitch_embed = self.timestep_embed(pitch_shift)
        x, residuals = self.encoder(x, pitch_embed)
        x = self.decoder(x, residuals, pitch_embed)

        return x


class AudioUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.autoenc = UNet()
        self.to_spec = T.Spectrogram(2046, 2046, 512, power=None)
        self.to_wav = T.InverseSpectrogram(2046, 2046, 512)

    def forward(self, x, pitch_shift=0):
        # calculate padding
        pad = 0
        if (x.shape[-1] + 512) % 8192 != 0:
            pad = 8192 - (x.shape[-1] + 512) % 8192
        x = F.pad(x, (0, pad))
        x = self.to_spec(x)

        x = torch.view_as_real(x)
        x = x.permute(0, 3, 1, 2)

        y = self.autoenc(x, pitch_shift)

        y = y.permute(0, 2, 3, 1).contiguous()
        y = torch.view_as_complex(y)
        y = self.to_wav(y)
        if pad > 0:
            y = y[:, :-pad]

        return y


if __name__ == "__main__":
    model = AudioUNet()
    # print # model params
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"# params: {params/1e6:.2f}M")
    x = torch.randn(8, 65024)
    # 127*512, the spectrogram pads by 512 to 65536, giving exactly 1024x128 2D shape
    print(x.shape)
    shift = torch.zeros(8, 1)
    from time import time

    with torch.no_grad():
        start = time()
        y = model(x, shift)
        end = time()
    print("Input shape", x.shape, "Output shape", y.shape)
    print("Took", end - start, "seconds")
