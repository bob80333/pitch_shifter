import soundfile as sf
import python_stretch as ps
import numpy as np

audio, sr = sf.read("p250_003_mic2.flac")
print(audio.shape, sr)

stretch = ps.Signalsmith.Stretch()
stretch.preset(1, sr)

stretch.setTransposeSemitones(12)
shifted_up = stretch.process(audio[None, :])
stretch.setTransposeSemitones(-12)
shifted_up = stretch.process(shifted_up)

sf.write("original.wav", audio, sr)

sf.write("shifted.wav", shifted_up[0], sr)

import torch
from model_1d_v2 import WavUNet

model = WavUNet().to("cuda")
model.load_state_dict(torch.load("outputs/output77/model_50000.pt"))


shifted_up = torch.tensor(shifted_up)
shifted_up = shifted_up.unsqueeze(0).to("cuda")
print(shifted_up.shape)

# pad audio to multiple of 1024
pad = 1024 - shifted_up.shape[2] % 1024
shifted_up = torch.nn.functional.pad(shifted_up, (0, pad))

unshifted_audio = model(shifted_up)

# remove padding
unshifted_audio = unshifted_audio[:, :, :-pad]

unshifted_audio = unshifted_audio.squeeze().detach().cpu().numpy()
sf.write("model_restored.wav", unshifted_audio, sr)