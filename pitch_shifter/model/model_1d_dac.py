import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from pitch_shifter.model.pixelshuffle1d import PixelUnshuffle1D, PixelShuffle1D

# flatten weights for muon optimizer to work correctly

class MyConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = nn.Parameter(self.weight.data.flatten(1, 2))

    def forward(self, input: torch.Tensor):
        return self._conv_forward(input, self.weight.view(self.out_channels, self.in_channels // self.groups, -1), self.bias)
    

# class MyConvTranspose1d(nn.ConvTranspose1d):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.weight = nn.Parameter(self.weight.data.flatten(1, 2))

#     def forward(self, input: torch.Tensor, output_size=None):
#         if self.padding_mode != "zeros":
#             raise ValueError(
#                 "Only `zeros` padding mode is supported for ConvTranspose1d"
#             )

#         assert isinstance(self.padding, tuple)
#         # One cannot replace List by Tuple or Sequence in "_output_padding" because
#         # TorchScript does not support `Sequence[T]` or `Tuple[T, ...]`.
#         num_spatial_dims = 1
#         output_padding = self._output_padding(
#             input,
#             output_size,
#             self.stride,  # type: ignore[arg-type]
#             self.padding,  # type: ignore[arg-type]
#             self.kernel_size,  # type: ignore[arg-type]
#             num_spatial_dims,
#             self.dilation,  # type: ignore[arg-type]
#         )
#         return F.conv_transpose1d(
#             input,
#             self.weight.view(self.in_channels, self.out_channels // self.groups, -1),
#             self.bias,
#             self.stride,
#             self.padding,
#             output_padding,
#             self.groups,
#             self.dilation,
#         )

# weight norm conv1d

# class MyConv1d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
#         super().__init__()

#         self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
#         # apply weight norm
#         self.conv = nn.utils.weight_norm(self.conv)

#     def forward(self, x):
#         return self.conv(x)

def snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # channels first
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return snake(x, self.alpha)



class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation=1, kernel=7):
        super().__init__()

        self.conv1 = MyConv1d(channels, channels, kernel, padding=(kernel - 1) // 2 * dilation, dilation=dilation)
        self.conv2 = MyConv1d(channels, channels, 1)

        self.act1 = Snake1d(channels)
        self.act2 = Snake1d(channels)

    def forward(self, x):
        res = x
        x = self.act1(x)
        x = self.conv1(x)
        x = self.act2(x)
        x = self.conv2(x)
        x = x + res
        return x
    

class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, dilations: list[int], kernel=7):
        super().__init__()

        self.dilations = dilations
        
        self.blocks = nn.ModuleList()
        for dilation in self.dilations:
            self.blocks.append(ResidualBlock(in_channels, dilation, kernel))

        self.blocks.append(Snake1d(in_channels))
        self.blocks.append(MyConv1d(in_channels, out_channels, kernel_size=2*stride, stride=stride, padding=stride // 2))

    def forward(self, x):
        res = x
        for block in self.blocks:
            x = block(x)
        return x, res
    

class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, dilations: list[int], kernel=7):
        super().__init__()

        self.dilations = dilations
        
        self.blocks = nn.ModuleList()
        # nearest neighbor upsample
        self.blocks.append(nn.Upsample(scale_factor=stride))
        # conv to reduce channels
        self.blocks.append(MyConv1d(in_channels, out_channels, 1))

        for dilation in self.dilations:
            self.blocks.append(ResidualBlock(out_channels, dilation, kernel))

        self.blocks.append(Snake1d(out_channels))
        self.blocks.append(MyConv1d(out_channels * 2, out_channels, kernel_size=1))

    def forward(self, x, res):
        for block in self.blocks:
            x = block(x)

            if isinstance(block, Snake1d):
                x = torch.cat([x, res], dim=1) # skip connection after snake
        return x

        

class Encoder(nn.Module):
    def __init__(self, channels: list[int], strides: list[int], dilations: list[int]):
        super().__init__()

        self.blocks = nn.ModuleList()

        for i in range(1, len(channels)):
            self.blocks.append(EncoderBlock(channels[i - 1], channels[i], strides[i - 1], dilations))

    def forward(self, x):
        residuals = []
        for block in self.blocks:
            x, res = block(x)
            residuals.append(res)
        return x, residuals
    

class Decoder(nn.Module):
    def __init__(self, channels: list[int], strides: list[int], dilations: list[int]):
        super().__init__()

        self.blocks = nn.ModuleList()

        for i in range(1, len(channels)):
            self.blocks.append(DecoderBlock(channels[i - 1], channels[i], strides[i - 1], dilations))

    def forward(self, x, residuals):
        for block, res in zip(self.blocks, residuals[::-1]):
            x = block(x, res)
        return x


class WavUNetDAC(nn.Module):
    def __init__(self, channels=None, strides=None, dilations=None):
        super().__init__()

        if channels is None:
            initial_stride = 4
            channels = [32, 64, 128, 256, 512]
            strides = [2, 4, 8, 8]
            dilations = [1, 3, 9]

        self.encoder = Encoder(channels, strides, dilations)
        self.decoder = Decoder(channels[::-1], strides[::-1], dilations)

        self.conv_in = MyConv1d(1, channels[0], kernel_size=initial_stride*2, stride = initial_stride, padding=initial_stride // 2)

        self.conv_out = nn.Sequential(
            nn.Upsample(scale_factor=initial_stride), # nearest neighbor upsample
            Snake1d(channels[0]), # act
            MyConv1d(channels[0], 1, kernel_size=7, padding=3), # conv to reduce channels\
            nn.Tanh()
        )

        self.bottleneck = nn.ModuleList()
        for dilation in dilations:
            self.bottleneck.append(ResidualBlock(channels[-1], dilation))


    def forward(self, x):
        x = self.conv_in(x)
        x, residuals = self.encoder(x)
        for block in self.bottleneck:
            x = block(x)
        x = self.decoder(x, residuals)
        x = self.conv_out(x)
        return x


if __name__ == "__main__":
    model = WavUNetDAC().to("cuda")
    print(model)
    opt_model = torch.compile(model)
    # tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    # print # model params
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"# params: {params/1e6:.2f}M")
    x = torch.randn(4, 1, 16384*4).to("cuda")
    print(x.shape)
    from time import time
    from tqdm import trange
    # warmup
    with torch.no_grad():
        for _ in trange(100):
            x = torch.randn(4, 1, 16384*4).to("cuda")
            y = opt_model(x)

    with torch.no_grad():
        start = time()
        for _ in trange(500):
            x = torch.randn(4, 1, 16384*4).to("cuda")
            y = opt_model(x)
        end = time()
    print("Input shape", x.shape, "Output shape", y.shape)
    print("Took", end - start, "seconds")
