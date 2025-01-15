from torch.utils.data import Dataset
import soundfile
import numpy as np
import torch
import python_stretch as ps
from torchaudio import functional as F

class AudioDataset(Dataset):
    def __init__(self, paths, samples=65024, test=False):
        self.paths = paths
        self.samples = samples
        self.test = test
        self.stretch = None # mono audio, 48 kHz

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        audio, sr = soundfile.read(path)

        # can't pickle Stretch object, so create it on first use
        if self.stretch is None:
            self.stretch = ps.Signalsmith.Stretch()
            self.stretch.preset(1, 48_000)

        # pad audio if necessary
        if len(audio) < self.samples:
            audio = np.pad(audio, (0, self.samples - len(audio) + 1))

        if self.test:
            # return middle 65024 samples
            start = len(audio) // 2 - self.samples // 2
            audio = audio[start : start + self.samples]
        else:
            # return random 65024 samples
            start = np.random.randint(0, len(audio) - self.samples)
            audio = audio[start : start + self.samples]

        audio = torch.tensor(audio).float()

        # do pitch shift augmentation
        # random pitch shift between -12 and 12 semitones (-1 octave to +1 octave)
        if self.test:
            shift = 12 # always shift up 1 octave for testing
        else:
            shift = np.random.randint(-12, 13)
        # shift audio up and back down to keep same pitch but introduce pitch shifting artifacts
        self.stretch.setTransposeSemitones(shift)
        shifted_audio = self.stretch.process(audio[None, :])
        self.stretch.setTransposeSemitones(-shift)
        shifted_audio = self.stretch.process(shifted_audio)[0]

        if shift > 0:
            # calculate lowpass filter cutoff frequency
            # 1 octave is a factor of 2 in frequency, 12 semitones is a factor of 2^(12/12) = 2
            # since we want 1/2 of the original frequency if we shift by 12, we use 2^(-shift/12)
            cutoff_freq = 2 ** (-shift / 12) * 24000 # 24 kHz is half of 48 kHz, so it's the Nyquist frequency
            # lowpass filter
            shifted_audio = F.lowpass_biquad(shifted_audio, sample_rate=48_000, cutoff_freq=cutoff_freq)
            audio = F.lowpass_biquad(audio, sample_rate=48_000, cutoff_freq=cutoff_freq)

        return audio, shifted_audio