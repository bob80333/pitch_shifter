from torch.utils.data import Dataset
import soundfile
import numpy as np
import torch
import python_stretch as ps
from torchaudio import functional as F

class AudioDataset(Dataset):
    def __init__(self, paths, samples=16384*3, test=False):
        self.paths = paths
        self.samples = samples
        self.test = test
        self.stretch = None # mono audio, 48 kHz

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        audio, sr = soundfile.read(path)

        audio = audio.astype(np.float32) # convert to float32

        # can't pickle Stretch object, so create it on first use
        if self.stretch is None:
            self.stretch = ps.Signalsmith.Stretch()
            self.stretch.preset(1, 48_000)

        # pad audio if necessary
        if len(audio) < self.samples:
            audio = np.pad(audio, (0, self.samples - len(audio) + 1))

        if self.test:
            # return middle group of samples
            start = len(audio) // 2 - self.samples // 2
            audio = audio[start : start + self.samples]
        else:
            # return random group of samples
            start = np.random.randint(0, len(audio) - self.samples)
            audio = audio[start : start + self.samples]

        # do pitch shift augmentation
        # random pitch shift between -12 and 12 semitones (-1 octave to +1 octave)
        if self.test:
            shift = 12 # always shift up 1 octave for testing
        else:
            shift = np.random.randint(-12, 13)
        # shift audio up and back down to keep same pitch but introduce pitch shifting artifacts
        #shifted_audio = audio.copy()
        self.stretch.setTransposeSemitones(shift)
        shifted_audio = self.stretch.process(audio[None, :])
        self.stretch.setTransposeSemitones(-shift)
        shifted_audio = self.stretch.process(shifted_audio)[0]

        audio = torch.from_numpy(audio)
        audio = audio.float()
        shifted_audio = torch.from_numpy(shifted_audio)
        shifted_audio = shifted_audio.float()

        if shift > 0:
            # calculate lowpass filter cutoff frequency
            # 1 octave is a factor of 2 in frequency, 12 semitones is a factor of 2^(12/12) = 2
            # since we want 1/2 of the original frequency if we shift by 12, we use 2^(-shift/12)
            cutoff_freq = 2 ** (-shift / 12) * 24000 # 24 kHz is half of 48 kHz, so it's the Nyquist frequency
            # lowpass filter
            for _ in range(10):
                shifted_audio = F.lowpass_biquad(shifted_audio, sample_rate=48_000, cutoff_freq=cutoff_freq)
                audio = F.lowpass_biquad(audio, sample_rate=48_000, cutoff_freq=cutoff_freq)

        return audio, shifted_audio
    
class PreShiftedAudioDataset(Dataset):
    def __init__(self, preprocessed_files, samples=16384*3, test=False):
        self.preprocessed_files = preprocessed_files

        if test:
            # only use files that were shifted up 1 octave for testing
            self.preprocessed_files = [str(x) for x in self.preprocessed_files if "shifted_12" in str(x)]
        self.samples = samples
        self.test = test
        self.stretch = None # mono audio, 48 kHz

    def __len__(self):
        return len(self.preprocessed_files)

    def __getitem__(self, idx):
        stretched = self.preprocessed_files[idx]
        original = stretched[:-5] + "0.wav"

        audio, sr = soundfile.read(original)
        shifted_audio, sr = soundfile.read(stretched)

        audio = audio.astype(np.float32) # convert to float32
        shifted_audio = shifted_audio.astype(np.float32) # convert to float32

        if self.test:
            # return middle group of samples
            start = len(audio) // 2 - self.samples // 2
            audio = audio[start : start + self.samples]
            shifted_audio = shifted_audio[start : start + self.samples]
        else:
            # return random group of samples
            start = np.random.randint(0, len(audio) - self.samples)
            audio = audio[start : start + self.samples]
            shifted_audio = shifted_audio[start : start + self.samples]

        audio = torch.from_numpy(audio)
        audio = audio.float()
        shifted_audio = torch.from_numpy(shifted_audio)
        shifted_audio = shifted_audio.float()

        return audio, shifted_audio
    
if __name__ == "__main__":
    from pathlib import Path
    from torch.utils.data import DataLoader
    train_files = list(Path("data/val_processed").rglob("*.wav"))
    print(f"Found {len(train_files)} training files")
    train_dataset = PreShiftedAudioDataset(train_files, test=True)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True, num_workers=2)
    from tqdm import tqdm

    i = 0
    for audio, shifted_audio in tqdm(train_dataloader):
        #print(audio.shape, shifted_audio.shape)
        i += 1

    print(i)