import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from pitch_shifter.model.pixelshuffle1d import PixelUnshuffle1D, PixelShuffle1D
from k_diffusion.layers import FourierFeatures

# flatten weights for muon optimizer to work correctly

class MyConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = nn.Parameter(self.weight.data.flatten(1, 2))

    def forward(self, input: torch.Tensor):
        return self._conv_forward(input, self.weight.view(self.out_channels, self.in_channels // self.groups, -1), self.bias)
    

class MyConvTranspose1d(nn.ConvTranspose1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = nn.Parameter(self.weight.data.flatten(1, 2))

    def forward(self, input: torch.Tensor, output_size=None):
        if self.padding_mode != "zeros":
            raise ValueError(
                "Only `zeros` padding mode is supported for ConvTranspose1d"
            )

        assert isinstance(self.padding, tuple)
        # One cannot replace List by Tuple or Sequence in "_output_padding" because
        # TorchScript does not support `Sequence[T]` or `Tuple[T, ...]`.
        num_spatial_dims = 1
        output_padding = self._output_padding(
            input,
            output_size,
            self.stride,  # type: ignore[arg-type]
            self.padding,  # type: ignore[arg-type]
            self.kernel_size,  # type: ignore[arg-type]
            num_spatial_dims,
            self.dilation,  # type: ignore[arg-type]
        )
        return F.conv_transpose1d(
            input,
            self.weight.view(self.in_channels, self.out_channels // self.groups, -1),
            self.bias,
            self.stride,
            self.padding,
            output_padding,
            self.groups,
            self.dilation,
        )

def snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # channels last
        self.alpha = nn.Parameter(torch.ones(1, 1, channels))

    def forward(self, x):
        return snake(x, self.alpha)


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
        if factor == 8:
            kernel = 16
        elif factor == 4:
            kernel = 8
        elif factor == 2:
            kernel = 4
        else:
            kernel = 1
        self.conv = MyConv1d(
            in_channels,
            out_channels,
            kernel_size=kernel,
            stride=factor,
            padding=factor // 2,
        )

        if skip == "pixel_shuffle":
            self.skip = PixelUnshuffle1D(factor)
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
        if factor == 8:
            kernel = 16
        elif factor == 4:
            kernel = 8
        elif factor == 2:
            kernel = 4
        else:
            kernel = 1
        self.conv = MyConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=kernel,
            stride=factor,
            padding=factor // 2,
        )

        if skip == "pixel_shuffle":
            self.skip = PixelShuffle1D(factor)
        else:
            self.skip = nn.Identity()

        self.dup_channels = dup_channels

    def forward(self, x):
        res = x
        x = self.conv(x)
        res = self.skip(res)
        if self.dup_channels > 1:
            # duplicate dup_channels factor of channels
            res = res.view(res.size(0), res.size(1), 1, res.size(2))
            res = res.expand(-1, -1, self.dup_channels, -1)
            res = res.contiguous().view(
                res.size(0), res.size(1) * self.dup_channels, res.size(3)
            )

        return x + res


class ConvNextBlock(nn.Module):
    def __init__(self, channels, expansion=4, layer_scale_init=1e-6, kernel=7):
        super().__init__()

        self.dw_conv = MyConv1d(
            channels, channels, kernel_size=kernel, padding=(kernel-1)//2, groups=channels
        )

        #self.norm = nn.LayerNorm(channels)

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
        x = x.permute(0, 2, 1).contiguous()
        #x = self.norm(x)
        x = self.pw_conv1(x)
        x = self.act(x)
        x = self.pw_conv2(x)

        if self.gamma is not None:
            x = x * self.gamma

        x = x.permute(0, 2, 1).contiguous()

        return x + res

class Encoder(nn.Module):
    def __init__(self, channels, blocks, factors, scale_vs_channels):
        super().__init__()

        kernels = [7, 23, 41]

        self.blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            for j in range(blocks[i]):
                self.blocks.append(ConvNextBlock(channels[i], kernel=kernels[j]))
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

        # remove last residual
        residuals.pop()

        return x, residuals


class Decoder(nn.Module):
    def __init__(self, channels, blocks, factors, scale_vs_channels):
        super().__init__()

        kernels = [7, 23, 41]

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
            for j in range(blocks[i - 1]):
                self.blocks.append(ConvNextBlock(channels[i], kernel=kernels[j]))

            self.blocks.append(MyConv1d(channels[i] * 2, channels[i], 1))


    def forward(self, x, residuals):
        for block in self.blocks:
            # skip conv
            if isinstance(block, MyConv1d):
                x = torch.cat([x, residuals.pop()], dim=1)
                
            x = block(x)

        return x


class WavUNet(nn.Module):
    def __init__(self, channels=None, blocks=None):
        super().__init__()

        if channels is None:
            channels = [16, 32, 128, 256, 512]
            blocks = [3, 3, 3, 3]
            factors = [2, 4, 4, 8]
            scale_vs_channels = [1, 1, 2, 4]

            bottleneck_blocks = 3

            patching = 2

        self.encoder = Encoder(channels, blocks, factors, scale_vs_channels)
        self.decoder = Decoder(
            channels[::-1], blocks[::-1], factors[::-1], scale_vs_channels[::-1]
        )

        self.bottleneck = nn.Sequential(*[ConvNextBlock(channels[-1]) for _ in range(bottleneck_blocks)])

        self.in_patch = PixelUnshuffle1D(patching)
        self.out_patch = PixelShuffle1D(patching)

        # take in 2 channels, output 1 channel
        # 2 inputs: conditioning audio (shifted) and x_t audio
        self.conv_in = MyConv1d(patching * 2, channels[0], 5, padding=2)
        self.conv_out = MyConv1d(channels[0], patching, 5, padding=2)
        
        self.timestamp_mlp = nn.Sequential(
            FourierFeatures(1, 128),
            nn.Linear(128, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, channels[0])
        )

    def forward(self, x, timestamp):
        ts_embed = self.timestamp_mlp(timestamp).unsqueeze(-1) # (B, C, T)
        x = self.in_patch(x)
        x = self.conv_in(x)
        x = x + ts_embed
        
        x, residuals = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, residuals)
        x = self.conv_out(x)
        x = self.out_patch(x)

        return x
    
    def apply_weightnorm(self):
        # replace each conv1d and linear with weight normalized version
        for module in self.modules():
            if isinstance(module, (MyConv1d, nn.Linear)):
                nn.utils.weight_norm(module)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WavUNet().to(device)
    print(model)
    opt_model = torch.compile(model)
    # tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    # print # model params
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"# params: {params/1e6:.2f}M")
    x = torch.randn(32, 2, 16384*2).to(device) # 2 channels because conditioning audio and noised audio
    print(x.shape)
    t = torch.rand(32, 1).to(device)
    from time import time
    from tqdm import trange
    # warmup
    with torch.no_grad():
        for _ in trange(50):
            x = torch.randn(32, 2, 16384*2).to(device)
            y = opt_model(x, t)

    with torch.no_grad():
        start = time()
        for _ in trange(200):
            x = torch.randn(32, 2, 16384*2).to(device)
            y = opt_model(x, t)
        end = time()
    print("Input shape", x.shape, "Output shape", y.shape)
    print("Took", end - start, "seconds")
