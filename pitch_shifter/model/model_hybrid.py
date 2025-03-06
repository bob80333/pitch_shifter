import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from pitch_shifter.model.pixelshuffle1d import PixelUnshuffle1D, PixelShuffle1D
from pitch_shifter.model.model_2d_v2 import UNet # doing my own spectrogram settings
from pitch_shifter.model.model_1d_v2 import WavUNet

class HybridUnet(nn.Module):
    def __init__(self):
        super().__init__()
        # spectrogram model settings
        spec_channels = [8, 16, 32, 64, 512]
        spec_blocks = [1, 3, 3, 3]
        spec_factors = [2, 2, 2, 4]
        spec_scale_vs_channels = [2, 2, 2, 2]
        spec_bottleneck = 3
         # the phase will skip the spectrogram model, since the wav model will take care of it
        spec_in_out_ch=1
        self.spec_model = UNet(
            channels=spec_channels,
            blocks=spec_blocks,
            factors=spec_factors,
            scale_vs_channels=spec_scale_vs_channels,
            input_channels=spec_in_out_ch,
            bottleneck_blocks=spec_bottleneck
        )

        wav_channels = [16, 32, 128, 256, 512]
        wav_blocks = [3, 3, 3, 3]
        wav_factors = [2, 4, 4, 8]
        wav_scale_vs_channels = [1, 1, 2, 4]
        wav_bottleneck = 3
        wav_patching = 2
        # self.wav_model = WavUNet(
        #     channels=wav_channels,
        #     blocks=wav_blocks,
        #     factors=wav_factors,
        #     scale_vs_channels=wav_scale_vs_channels,
        #     bottleneck_blocks=wav_bottleneck,
        #     patching=wav_patching
        # )

        # spectrogram conversion
        self.to_spec = T.Spectrogram(4096, 4096, 1024, power=None)
        self.to_wav = T.InverseSpectrogram(4096, 4096, 1024)

        # mel conversion
        self.to_mel = T.MelScale(n_mels = 224, sample_rate=48_000, n_stft=2049)
        self.to_hz = T.InverseMelScale(n_stft=2049, n_mels=224, sample_rate=48_000)


    def forward(self, x):
        # sequence length must be a multiple of 32768, except the last multiple should be 1024 less, one hop less (hop length * model downsampling = 1024 * 32 = 32768)
        padding = 32768 - ((x.shape[-1] + 1024) % 32768)
        x_padded = F.pad(x, (0, padding))

        # convert to spec
        spec_full = self.to_spec(x_padded)

        # split off the phase
        spec_full = torch.view_as_real(spec_full)
        ampl = spec_full[..., 0]
        phase = spec_full[..., 1]

        # to mel
        ampl_mel = self.to_mel(ampl)
        # run model on mel amplitudes
        ampl_mel = self.spec_model(ampl_mel)

        # convert back to hz
        ampl = self.to_hz(ampl_mel)

        # replace the phase with 0s, the original phase sounds terrible
        phase = torch.zeros_like(phase)

        # add the phase back
        spec_full = torch.stack([ampl, phase], dim=-1)
        spec_full = torch.view_as_complex(spec_full)

        # convert back to wav
        x = self.to_wav(spec_full)
        # remove padding
        x = x[..., :-padding]

        # x = self.wav_model(x)
        return x
    

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HybridUnet().to(device)
    #opt_model = torch.compile(model)
    model.spec_model = torch.compile(model.spec_model)
    #model.wav_model = torch.compile(model.wav_model)
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
            y = model(x)

    with torch.no_grad():
        start = time()
        for _ in trange(200):
            x = torch.randn(32, 1, 16384*2).to("cuda")
            y = model(x)
        end = time()
    print("Input shape", x.shape, "Output shape", y.shape)
    print("Took", end - start, "seconds")
    print("Nans?", y.isnan().any())
