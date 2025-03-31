from pathlib import Path
from tqdm import tqdm
import soundfile as sf
import numpy as np
import torch
from multiprocessing import Pool
import python_stretch as ps
from torchaudio import functional as F
import torchaudio.transforms as T
import os

def process_file_vctk(file, in_folder, out_folder):
    stretch = ps.Signalsmith.Stretch()
    stretch.preset(1, 48_000*2)

    if "val" in out_folder or "test" in out_folder:
        test = True
    else:
        test = False

    new_filename = file.replace(in_folder, out_folder)
    # make sure the directory exists
    os.makedirs(os.path.dirname(new_filename), exist_ok=True)

    audio, sr = sf.read(file)
    audio = audio.astype(np.float32) # convert to float32

    resample_up = T.Resample(sr, 48_000*2)
    resample_down = T.Resample(48_000*2, 48_000)

    # resample to 96 kHz, for the stretch algorithm
    audio_resamp = resample_up(torch.from_numpy(audio).unsqueeze(0)).squeeze().numpy()

    # do every shift from -1 octave to +1 octave
    if not test:
        shifts = np.arange(-12, 13)
    else:
        shifts = [-12, 12]
    for shift in shifts:
        stretch.setTransposeSemitones(shift)
        shifted_audio = stretch.process(audio_resamp[None, :])
        stretch.setTransposeSemitones(-shift)
        shifted_audio = stretch.process(shifted_audio)[0]

        # resample back to 48 kHz
        shifted_audio = resample_down(torch.from_numpy(shifted_audio).unsqueeze(0)).squeeze().numpy()

        # save the audio file
        sf.write(new_filename.replace(".flac", f"_shifted_{shift}.flac"), shifted_audio, sr)

    sf.write(new_filename.replace(".flac", f"_baseline.flac"), audio, sr)



def process_file_vocalset(file, in_folder, out_folder, train_prob=0.8, val_prob=0.1, test_prob=0.1):
    stretch = ps.Signalsmith.Stretch()
    stretch.preset(1, 48_000*2)

    random_number = np.random.rand()
    if random_number < train_prob:
        out_folder = out_folder.replace("FULL", "train")
        test=False
    elif random_number < train_prob + val_prob:
        out_folder = out_folder.replace("FULL", "val")
        test=False
    else:
        out_folder = out_folder.replace("FULL", "test")
        test=False
    
    new_filename = file.replace(in_folder, out_folder)
    # make sure the directory exists
    os.makedirs(os.path.dirname(new_filename), exist_ok=True)

    audio, sr = sf.read(file)
    audio = audio.astype(np.float32) # convert to float32

    resample_up = T.Resample(sr, 48_000*2)
    resample_down = T.Resample(48_000*2, 48_000)

    # resample to 96 kHz, for the stretch algorithm
    audio_resamp = resample_up(torch.from_numpy(audio).unsqueeze(0)).squeeze().numpy()

    # do every shift from -1 octave to +1 octave
    if not test:
        shifts = np.arange(-12, 13)
    else:
        shifts = [-12, 12]
    for shift in shifts:
        stretch.setTransposeSemitones(shift)
        shifted_audio = stretch.process(audio_resamp[None, :])
        stretch.setTransposeSemitones(-shift)
        shifted_audio = stretch.process(shifted_audio)[0]

        # resample back to 48 kHz
        shifted_audio = resample_down(torch.from_numpy(shifted_audio).unsqueeze(0)).squeeze().numpy()

        # save the audio file
        sf.write(new_filename.replace(".wav", f"_shifted_{shift}.flac"), shifted_audio, 48000)

    sf.write(new_filename.replace(".wav", f"_baseline.flac"), audio, 48000)

# multiprocessing

if __name__ == '__main__':
    in_folders = ["dataset_dir/vctk_dataset/train", "dataset_dir/vctk_dataset/val", "dataset_dir/vctk_dataset/test"]
    out_folders = ["dataset_dir/vctk_dataset/train_processed_v3", "dataset_dir/vctk_dataset/val_processed_v3", "dataset_dir/vctk_dataset/test_processed_v3"]
    for in_folder, out_folder in zip(in_folders, out_folders):
        files = [str(x.absolute()) for x in Path(in_folder).rglob("*.flac")]

        os.makedirs(out_folder, exist_ok=True)
        with Pool() as pool:
            results = list(tqdm(pool.imap(process_file_vctk, files, [in_folder]*len(files), [out_folder]*len(files)), total=len(files)))

    in_folders = ["dataset_dir/vocalset_dataset/FULL"]
    out_folders = ["dataset_dir/vocalset_dataset/FULL_processed_v3"]
    for in_folder, out_folder in zip(in_folders, out_folders):
        files = [str(x.absolute()) for x in Path(in_folder).rglob("*.wav")]

        os.makedirs(out_folder, exist_ok=True)
        with Pool() as pool:
            results = list(tqdm(pool.imap(process_file_vocalset, files, [in_folder]*len(files), [out_folder]*len(files)), total=len(files)))
    print("Done!")
