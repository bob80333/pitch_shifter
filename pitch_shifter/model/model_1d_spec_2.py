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
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return snake(x, self.alpha)




class ConvBlock(nn.Module):
    def __init__(self, channels, kernel=3):
        super().__init__()

        self.act = nn.LeakyReLU(0.1)

        self.conv1 = nn.Conv1d(channels, channels, kernel, padding=kernel // 2)

    def forward(self, x):
        res = x

        x = self.act(x)
        x = self.conv1(x)

        return x + res


class Spec1dNet2(nn.Module):
    def __init__(self, channels = 1024, blocks = 8):
        super().__init__()

        self.bottleneck_ampl = nn.Sequential(*[ConvBlock(channels) for _ in range(blocks)])

        self.conv_in_ampl = MyConv1d(513, channels, 1)
        self.conv_out_ampl = MyConv1d(channels, 513, 1)

        self.to_spec = T.Spectrogram(1024, 1024, 256, power=None)
        self.to_wav = T.InverseSpectrogram(1024, 1024, 256)

    def forward(self, x):
        batch, channels, _ = x.shape
        # combine channels and batch
        x = x.view(batch * channels, -1)
        x = self.to_spec(x)
        x = torch.view_as_real(x)
        x_ampl = x[..., 0]

        x_phase = x[..., 1]

        x_ampl = self.conv_in_ampl(x_ampl)
        x_ampl = self.bottleneck_ampl(x_ampl)
        x_ampl = self.conv_out_ampl(x_ampl)


        # replace phase with 0s for now
        x_phase = torch.zeros_like(x_phase)

        x = torch.stack([x_ampl, x_phase], dim=-1)
        x = torch.view_as_complex(x)
        x = self.to_wav(x)

        # split channels and batch
        x = x.view(batch, channels, -1)

        return x, x_ampl
    
    def apply_weightnorm(self):
        # replace each conv1d and linear with weight normalized version
        for module in self.modules():
            if isinstance(module, (MyConv1d, nn.Linear)):
                nn.utils.weight_norm(module)


if __name__ == "__main__":
    model = Spec1dNet2().to("cuda")
    print(model)
    opt_model = torch.compile(model)
    # tf32
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
            y, _ = opt_model(x)
        end = time()
    print("Input shape", x.shape, "Output shape", y.shape)
    print("Took", end - start, "seconds")
