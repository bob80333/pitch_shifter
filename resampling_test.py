import soundfile as sf
import python_stretch as ps
from torchaudio.functional import resample
import torch

audio, sr = sf.read("p250_003_mic2.flac")
print(audio.shape)

stretch = ps.Signalsmith.Stretch()
stretch.preset(1, sr)

resampled_audio = resample(torch.from_numpy(audio).unsqueeze(0), sr, sr*2)
resampled_audio = resampled_audio.squeeze().numpy()

stretch.setTransposeSemitones(12)
shifted_up = stretch.process(audio[None, :])

stretch.setTransposeSemitones(-12)
shifted = stretch.process(shifted_up)

stretch.preset(1, sr*2)
stretch.setTransposeSemitones(12)
resampled_up = stretch.process(resampled_audio[None, :])
stretch.setTransposeSemitones(-12)
resampled_shifted = stretch.process(resampled_up)

resampled = resample(torch.from_numpy(resampled_shifted).unsqueeze(0), sr*2, sr).squeeze().numpy()

print(shifted.shape, resampled.shape)

sf.write("original.wav", audio, sr)
sf.write("shifted.wav", shifted[0], sr)
sf.write("resampled_shifted.wav", resampled, sr)
