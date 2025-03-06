import soundfile as sf
import python_stretch as ps
import numpy as np
import matplotlib.pyplot as plt
import torchaudio.transforms as T
import torch

audio, sr = sf.read("example/p250_003_mic2.flac")
#audio, sr = sf.read("example/ado_singing.wav")

# pad audio to multiple of 8192
print("before", audio.shape)

padding = 8192 - ((audio.shape[-1] + 256) % 8192)
audio = np.pad(audio, (0, padding))
print("after", audio.shape)

shifter = ps.Signalsmith.Stretch()
shifter.preset(1, sr)
shifter.setTransposeSemitones(12)
shifted_up = shifter.process(audio[None, :])

to_spec = T.Spectrogram(1022, 1022, 256, power=None)

# get phase of original
spec = to_spec(torch.tensor(audio).unsqueeze(0))
spec = torch.view_as_real(spec)
phase = spec[..., 1]

# # log phase, shift up minimum to 1
# phase = torch.log1p(phase - phase.min())
# # normalize to 0-1
# phase = (phase - phase.min()) / (phase.max() - phase.min())

print(phase.shape)
print("Original: Max, min, median:", phase.max(), phase.min(), phase.median())

# plot it
plt.figure()
plt.imshow(phase[0].numpy())
plt.title("Phase of original audio")
#plt.show()
# save it
plt.imsave("example/original_phase.png", phase[0].numpy())

# get phase of shifted up
spec = to_spec(torch.tensor(shifted_up))
spec = torch.view_as_real(spec)
phase = spec[..., 1]

# # log phase, shift up minimum to 1
# phase = torch.log1p(phase - phase.min())
# # normalize to 0-1
# phase = (phase - phase.min()) / (phase.max() - phase.min())

print(phase.shape)
print("Shifted: Max, min, median:", phase.max(), phase.min(), phase.median())

# plot it
plt.figure()
plt.imshow(phase[0].numpy())
plt.title("Phase of shifted up audio")
#plt.show()

# save it
plt.imsave("example/shifted_up_phase.png", phase[0].numpy())

# combine original magnitude with shifted phase, see what the errors are
spec = to_spec(torch.tensor(audio).unsqueeze(0))
spec = torch.view_as_real(spec)
magnitude = spec[..., 0]

random_phase = torch.randn_like(phase) * 1e-1

# combine magnitude with random phase
random_phase_spec = torch.stack([magnitude, random_phase], dim=-1)
random_phase_spec = torch.view_as_complex(random_phase_spec)

spec = torch.stack([magnitude, phase], dim=-1)
spec = torch.view_as_complex(spec)

to_wav = T.InverseSpectrogram(1022, 1022, 256)
new_audio = to_wav(spec)

random_phase_audio = to_wav(random_phase_spec)

# remove padding
new_audio = new_audio[..., :-padding]
audio = audio[:-padding]
random_phase_audio = random_phase_audio[..., :-padding]

print(audio.shape, new_audio.shape)

# pad new audio to be the same length as original
pad = audio.shape[0] - new_audio.shape[1]
new_audio = torch.nn.functional.pad(new_audio, (0, pad))

# calculate SISDR between original and new audio

from auraloss.time import SISDRLoss
sisdr_loss = SISDRLoss()

print("SISDR between original and new audio:", -sisdr_loss(new_audio, torch.tensor(audio)))

# calculate SISDR between original and random phase audio
print("SISDR between original and random phase audio:", -sisdr_loss(random_phase_audio, torch.tensor(audio)))


new_audio = new_audio.squeeze().detach().numpy()

sf.write("example/combined_audio.wav", new_audio, sr)

sf.write("example/random_phase_audio.wav", random_phase_audio.squeeze().detach().numpy(), sr)

# now try replacing phase with random noise


