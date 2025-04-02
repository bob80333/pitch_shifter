import torch
import torch.nn as nn

# use output head from vocos
from vocos.heads import ISTFTHead

import torchaudio.transforms as T

from einops import rearrange


# flatten weights for muon optimizer to work correctly


class MyConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = nn.Parameter(self.weight.data.flatten(1, 2))

    def forward(self, input: torch.Tensor):
        return self._conv_forward(
            input,
            self.weight.view(self.out_channels, self.in_channels // self.groups, -1),
            self.bias,
        )


# vocos uses expansion=3
class ConvNextBlock(nn.Module):
    def __init__(self, channels, expansion=3, layer_scale_init=1e-6):
        super().__init__()

        self.dw_conv = MyConv1d(
            channels, channels, kernel_size=7, padding=3, groups=channels
        )

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
        x = self.pw_conv1(x)
        x = self.act(x)
        x = self.pw_conv2(x)

        if self.gamma is not None:
            x = x * self.gamma

        x = x.permute(0, 2, 1).contiguous()

        return x + res


class VocosModel(nn.Module):
    def __init__(self, model_width=512, num_layers = 8, n_fft=1024, hop_length=256):
        super().__init__()
        
        # fft part
        self.to_spec = T.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=None, center=True)

        self.head = ISTFTHead(dim=model_width, n_fft=n_fft, hop_length=hop_length, padding="center")
        
        # model part
        self.in_conv = MyConv1d(n_fft + 2, model_width, kernel_size=1)
        
        self.blocks = nn.ModuleList(
            [ConvNextBlock(model_width) for _ in range(num_layers)]
        )

    def forward(self, x):
        x = self.to_spec(x)
        
        # make real:
        x = torch.view_as_real(x)
        
        # move added last dim into channel (fft bins) dim:
        x = rearrange(x, 'b f t c -> b (fc) t')
        
        x = self.in_conv(x)

        for block in self.blocks:
            x = block(x)

        return self.head(x)