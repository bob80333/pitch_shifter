import soundfile as sf
import python_stretch as ps
import numpy as np
import torch
from torchaudio.functional import resample
audio, sr = sf.read("p250_003_mic2.flac")
#audio, sr = sf.read("ado_singing.wav")

print(audio.shape, sr)

sf.write("original.wav", audio, sr)

stretch = ps.Signalsmith.Stretch()
stretch.preset(1, sr*2)

resampled_audio = resample(torch.from_numpy(audio).unsqueeze(0), sr, sr*2).squeeze().numpy()

stretch.setTransposeSemitones(12)
shifted_up = stretch.process(resampled_audio[None, :])

up_octave = resample(torch.from_numpy(shifted_up).unsqueeze(0), sr*2, sr).squeeze().numpy()
sf.write("up_octave.wav", up_octave, sr)

stretch.setTransposeSemitones(-12)
shifted_up = stretch.process(shifted_up)

shifted = resample(torch.from_numpy(shifted_up).unsqueeze(0), sr*2, sr).squeeze().numpy()

sf.write("shifted.wav", shifted, sr)

import torch
from pitch_shifter.model.model_1d import WavUNet

model = WavUNet().to("cuda")
model.load_state_dict(torch.load("outputs/output81/model_100000.pt"))


shifted_up = torch.tensor(shifted)
shifted_up = shifted_up.unsqueeze(0).unsqueeze(0).to("cuda")
print(shifted_up.shape)

# pad audio to multiple of 16384
pad = 16384 - shifted_up.shape[2] % 16384
shifted_up = torch.nn.functional.pad(shifted_up, (0, pad))

with torch.no_grad():
    unshifted_audio = model(shifted_up)

# remove padding
unshifted_audio = unshifted_audio[:, :, :-pad]

unshifted_audio = unshifted_audio.squeeze().detach().cpu().numpy()
sf.write("model_restored.wav", unshifted_audio, sr)

# try removing artifacts from audio with pitch shifting

up_octave = torch.tensor(up_octave).unsqueeze(0).unsqueeze(0).to("cuda")

# pad audio to multiple of 16384
pad = 16384 - up_octave.shape[2] % 16384
up_octave = torch.nn.functional.pad(up_octave, (0, pad))

with torch.no_grad():
    up_octave_restored = model(up_octave)

# remove padding
up_octave_restored = up_octave_restored[:, :, :-pad]

sf.write("up_octave_restored.wav", up_octave_restored.squeeze().detach().cpu().numpy(), sr)

# re-shift the up_octave_restored back down to original pitch to judge how well the model removed the artifacts
up_octave_restored_down = up_octave_restored.squeeze().detach().cpu().numpy()
stretch.setTransposeSemitones(-12)
up_octave_restored_down = stretch.process(up_octave_restored_down[None, :])[0]

sf.write("up_octave_restored_down.wav", up_octave_restored_down, sr)


# now run model again on the up_octave_restored_down to see if restoring at both up and down works better
up_octave_restored_down_restored = torch.tensor(up_octave_restored_down).unsqueeze(0).unsqueeze(0).to("cuda")

# pad audio to multiple of 16384
pad = 16384 - up_octave_restored_down_restored.shape[2] % 16384

up_octave_restored_down_restored = torch.nn.functional.pad(up_octave_restored_down_restored, (0, pad))

with torch.no_grad():
    up_octave_restored_down_restored = model(up_octave_restored_down_restored)

# remove padding
up_octave_restored_down_restored = up_octave_restored_down_restored[:, :, :-pad]

sf.write("up_octave_restored_down_restored.wav", up_octave_restored_down_restored.squeeze().detach().cpu().numpy(), sr)



# calculate SI-SDR, SNR for both model output and shifted audio

from auraloss.time import SISDRLoss
si_sdr_zero_mean = SISDRLoss(zero_mean=True)

unshifted_audio = torch.tensor(unshifted_audio).unsqueeze(0).to("cuda")
audio = torch.tensor(audio).unsqueeze(0).to("cuda")
# remove padding
shifted_up = shifted_up[:, :, :-pad].to("cuda")

# add very small noise to audio to see how it affects the metrics
noised_audio = audio + 1e-4 * torch.randn_like(audio)
sf.write("noised_audio.wav", noised_audio.squeeze().cpu().numpy(), sr)

si_sdr_self = -si_sdr_zero_mean(audio, audio)
si_sdr_self_noised = -si_sdr_zero_mean(noised_audio, audio)
si_sdr_model = -si_sdr_zero_mean(unshifted_audio, audio)
si_sdr_shifted = -si_sdr_zero_mean(shifted_up, audio)
up_octave_restored_down = torch.tensor(up_octave_restored_down).to("cuda")
si_sdr_restored_up_down = -si_sdr_zero_mean(up_octave_restored_down, audio)
up_octave_restored_down_restored = up_octave_restored_down_restored.squeeze().detach().cpu().numpy()
sf.write("up_octave_restored_down_restored.wav", up_octave_restored_down_restored, sr)
up_octave_restored_down_restored = torch.tensor(up_octave_restored_down_restored).to("cuda")
si_sdr_restored_up_down_restored = -si_sdr_zero_mean(up_octave_restored_down_restored, audio)

print(f"SI-SDR for original audio: {si_sdr_self}")
print(f"SI-SDR for noised audio: {si_sdr_self_noised}")
print(f"SI-SDR for model output: {si_sdr_model}")
print(f"SI-SDR for shifted audio: {si_sdr_shifted}")
print(f"SI-SDR for up octave restored down: {si_sdr_restored_up_down}")
print(f"SI-SDR for up octave restored down restored: {si_sdr_restored_up_down_restored}")

# calculate SNR
snr_model = 10 * torch.log10(torch.sum(audio**2) / torch.sum((audio - unshifted_audio)**2))
snr_shifted = 10 * torch.log10(torch.sum(audio**2) / torch.sum((audio - shifted_up)**2))

print(f"SNR for model output: {snr_model}")
print(f"SNR for shifted audio: {snr_shifted}")
