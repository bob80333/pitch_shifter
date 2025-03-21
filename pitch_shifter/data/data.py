from torch.utils.data import Dataset
import soundfile
import numpy as np
import torch
import python_stretch as ps
from torchaudio import functional as F
import torchaudio.transforms as T

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
    def __init__(self, preprocessed_files, samples=16384, test=False):
        
        # keep only files that were shifted
        self.preprocessed_files = [str(x) for x in preprocessed_files if "baseline" not in str(x)]

        #if test:
            # only use files that were shifted up 1 octave
        self.preprocessed_files = [str(x) for x in self.preprocessed_files if "shifted_12" in str(x)]
        self.samples = samples
        self.test = test
        self.stretch = None # mono audio, 48 kHz

        #self.resampler = T.Resample(orig_freq=48_000, new_freq=16_000)

    def __len__(self):
        return len(self.preprocessed_files)

    def __getitem__(self, idx):
        shifted_file = self.preprocessed_files[idx]
        # find second last underscore to get original file
        original_file = shifted_file[:shifted_file.rindex("_", 0, shifted_file.rindex("_"))] + "_baseline.wav"
        try:
            audio, sr = soundfile.read(original_file)
            shifted_audio, sr = soundfile.read(shifted_file)
        except:
            print(f"Error reading {original_file} or {shifted_file}")
            return None, None

        audio = torch.from_numpy(audio)
        audio = audio.float()
        shifted_audio = torch.from_numpy(shifted_audio)
        shifted_audio = shifted_audio.float()

        # resample to 16 kHz
        #audio = self.resampler(audio)
        #shifted_audio = self.resampler(shifted_audio)

        if audio.shape[0] > self.samples:
            if self.test:
                # return middle group of samples
                start = audio.shape[0] // 2 - self.samples // 2
                audio = audio[start : start + self.samples]
                shifted_audio = shifted_audio[start : start + self.samples]
            else:
                # return random group of samples
                start = np.random.randint(0, audio.shape[0] - self.samples)
                audio = audio[start : start + self.samples]
                shifted_audio = shifted_audio[start : start + self.samples]
        else:
            # pad audio if necessary
            audio = torch.nn.functional.pad(audio, (0, self.samples - audio.shape[0]))
            shifted_audio = torch.nn.functional.pad(shifted_audio, (0, self.samples - shifted_audio.shape[0]))




        return audio, shifted_audio
    

class PreShiftedDownAudioDataset(Dataset):
    def __init__(self, preprocessed_files, samples=16384, test=False, only_shift_12=True):
        
        # keep only files that were shifted
        self.preprocessed_files = [str(x) for x in preprocessed_files if "baseline" not in str(x)]

        # only use files that were shifted up 1 octave
        # used for when model doesn't have input for shift amount, don't want model to learn pitch shift by artifacts, won't generalize to non artifacted audio
        if only_shift_12:
            self.preprocessed_files = [str(x) for x in self.preprocessed_files if "shifted_12" in str(x)]
        self.samples = samples
        self.test = test
        self.stretch = None # mono audio, 48 kHz

        #self.resampler = T.Resample(orig_freq=48_000, new_freq=16_000)

    def __len__(self):
        return len(self.preprocessed_files)

    def __getitem__(self, idx):
        shifted_file = self.preprocessed_files[idx]
        # find second last underscore to get original file
        original_file = shifted_file[:shifted_file.rindex("_", 0, shifted_file.rindex("_"))] + "_baseline.wav"
        try:
            audio, sr = soundfile.read(original_file)
            shifted_audio, sr = soundfile.read(shifted_file)
        except:
            print(f"Error reading {original_file} or {shifted_file}")
            return None, None

        audio = torch.from_numpy(audio)
        audio = audio.float()
        shifted_audio = torch.from_numpy(shifted_audio)
        shifted_audio = shifted_audio.float()

        # resample to 16 kHz
        #audio = self.resampler(audio)
        #shifted_audio = self.resampler(shifted_audio)

        if audio.shape[0] > self.samples:
            if self.test:
                # return middle group of samples
                start = audio.shape[0] // 2 - self.samples // 2
                audio = audio[start : start + self.samples]
                shifted_audio = shifted_audio[start : start + self.samples]
            else:
                # return random group of samples
                start = np.random.randint(0, audio.shape[0] - self.samples)
                audio = audio[start : start + self.samples]
                shifted_audio = shifted_audio[start : start + self.samples]
        else:
            # pad audio if necessary
            audio = torch.nn.functional.pad(audio, (0, self.samples - audio.shape[0]))
            shifted_audio = torch.nn.functional.pad(shifted_audio, (0, self.samples - shifted_audio.shape[0]))




        return audio, shifted_audio
    

class BetterAudioDataset(Dataset):
    def __init__(self, paths, samples=16384*3, test=False):
        self.paths = paths
        self.samples = samples
        self.test = test
        self.stretch = None # mono audio, 48 kHz

        self.resample_up = T.Resample(orig_freq=48_000, new_freq=96_000, resampling_method="sinc_interp_kaiser")
        self.resample_down = T.Resample(orig_freq=96_000, new_freq=48_000, resampling_method="sinc_interp_kaiser")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        audio, sr = soundfile.read(path)

        audio = audio.astype(np.float32) # convert to float32

        # can't pickle Stretch object, so create it on first use
        if self.stretch is None:
            self.stretch = ps.Signalsmith.Stretch()
            self.stretch.preset(1, 96_000)

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

        # resample to higher sample rate to keep high frequencies after downshift
        audio_resamp = self.resample_up(torch.from_numpy(audio)).numpy()
        # shift audio up and back down to keep same pitch but introduce pitch shifting artifacts
        #shifted_audio = audio.copy()
        self.stretch.setTransposeSemitones(shift)
        shifted_audio = self.stretch.process(audio_resamp[None, :])
        self.stretch.setTransposeSemitones(-shift)
        shifted_audio = self.stretch.process(shifted_audio)[0]

        # downsample back to 48 kHz
        shifted_audio = np.array(shifted_audio)
        shifted_audio = torch.from_numpy(shifted_audio)
        shifted_audio = self.resample_down(shifted_audio)

        audio = torch.from_numpy(audio)
        audio = audio.float()
        shifted_audio = shifted_audio.float()

        return audio, shifted_audio
    
if __name__ == "__main__":
    from pathlib import Path
    from torch.utils.data import DataLoader
    train_files = list(Path("dataset_dir/val_processed_v2").rglob("*_baseline.wav"))[:8*101] # limit # of samples to make testing faster
    print(f"Found {len(train_files)} val files")
    train_dataset = AudioDataset(train_files, test=True)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True, num_workers=0)
    from tqdm import tqdm
    from time import time

    # speed test

    i = 0
    for audio, shifted_audio in tqdm(train_dataloader):
        i += 1
        if i == 1: # let the dataloader init processes before timing
            start = time()


    print("Original Audio Dataset:")
    end = time()
    print(f"Time: {end - start:.2f}")
    print(f"Batches/second: {100 / (end - start):.2f}")
    print(f"Time: {time() - start:.2f}")

    train_dataset = BetterAudioDataset(train_files, test=True)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True, num_workers=0)

    i = 0
    for audio, shifted_audio in tqdm(train_dataloader):
        i += 1
        if i == 1: # let the dataloader init processes before timing
            start = time()


    print("Better Audio Dataset (keeps high frequencies):")
    end = time()
    print(f"Time: {end - start:.2f}")
    print(f"Batches/second: {100 / (end - start):.2f}")