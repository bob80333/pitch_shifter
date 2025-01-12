from torch.utils.data import Dataset
import soundfile
import numpy as np

class AudioDataset(Dataset):
    def __init__(self, paths, samples=97792, test=False):
        self.paths = paths
        self.samples = samples
        self.test = test

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        audio, _ = soundfile.read(path)

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

        return audio