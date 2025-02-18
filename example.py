import soundfile as sf
import python_stretch as ps
import numpy as np
import torch
from torchaudio.functional import resample
audio, sr = sf.read("p250_003_mic2.flac")

print(audio.shape, sr)

sf.write("original.wav", audio, sr)

stretch = ps.Signalsmith.Stretch()
stretch.preset(1, sr*2)

resampled_audio = resample(torch.from_numpy(audio).unsqueeze(0), sr, sr*2).squeeze().numpy()

stretch.setTransposeSemitones(12)
shifted_up = stretch.process(resampled_audio[None, :])
stretch.setTransposeSemitones(-12)
shifted_up = stretch.process(shifted_up)

shifted = resample(torch.from_numpy(shifted_up).unsqueeze(0), sr*2, sr).squeeze().numpy()

sf.write("shifted.wav", shifted, sr)

import torch
from model_1d_v2 import WavUNet

model = WavUNet().to("cuda")
model.load_state_dict(torch.load("outputs/output81/model_100000.pt"))


shifted_up = torch.tensor(shifted)
shifted_up = shifted_up.unsqueeze(0).unsqueeze(0).to("cuda")
print(shifted_up.shape)

# pad audio to multiple of 16384
pad = 16384 - shifted_up.shape[2] % 16384
shifted_up = torch.nn.functional.pad(shifted_up, (0, pad))

unshifted_audio = model(shifted_up)

# remove padding
unshifted_audio = unshifted_audio[:, :, :-pad]

unshifted_audio = unshifted_audio.squeeze().detach().cpu().numpy()
sf.write("model_restored.wav", unshifted_audio, sr)

# calculate SI-SDR, SNR for both model output and shifted audio

from auraloss.time import SISDRLoss
si_sdr_zero_mean = SISDRLoss(zero_mean=True)

unshifted_audio = torch.tensor(unshifted_audio).unsqueeze(0).to("cuda")
audio = torch.tensor(audio).unsqueeze(0).to("cuda")
# remove padding
shifted_up = shifted_up[:, :, :-pad].to("cuda")


si_sdr_model = -si_sdr_zero_mean(unshifted_audio, audio)
si_sdr_shifted = -si_sdr_zero_mean(shifted_up, audio)

print(f"SI-SDR for model output: {si_sdr_model}")
print(f"SI-SDR for shifted audio: {si_sdr_shifted}")

# calculate SNR
snr_model = 10 * torch.log10(torch.sum(audio**2) / torch.sum((audio - unshifted_audio)**2))
snr_shifted = 10 * torch.log10(torch.sum(audio**2) / torch.sum((audio - shifted_up)**2))

print(f"SNR for model output: {snr_model}")
print(f"SNR for shifted audio: {snr_shifted}")
