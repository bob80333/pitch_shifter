from torch.utils.data import Dataset
import soundfile
import numpy as np
import torch
import python_stretch as ps

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

        return audio, shifted_audio