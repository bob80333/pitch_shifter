import soundfile
from model import AudioUNet
import torch
import numpy as np
import python_stretch as ps

# load audio
audio, sr = soundfile.read("p243_014_mic2.flac")
print(audio.shape, sr)

pad = 12463

# pad audio
audio = np.pad(audio, (0, pad))

# shift audio by 1/2 octave up (only up for testing)
stretch = ps.Signalsmith.Stretch()
stretch.preset(1, 48_000)
stretch.setTransposeSemitones(6)

shifted_audio = stretch.process(audio[None, :])

soundfile.write("shifted_audio.wav", shifted_audio[0][:-pad], 48_000)

model = AudioUNet()
model.eval()
# load model
model.load_state_dict(torch.load("outputs/output15/model_12000.pt"))

with torch.inference_mode():
    # run model
    shifted_audio = torch.tensor(shifted_audio).float()
    unshifted_audio = model(shifted_audio)
    unshifted_audio = unshifted_audio.squeeze(0).numpy()[:-pad]

soundfile.write("unshifted_audio.wav", unshifted_audio, 48_000)