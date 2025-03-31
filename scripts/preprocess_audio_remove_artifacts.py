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
    print(new_filename)
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



# wrapper functions to unpack arguments for multiprocessing
def wrapper_vctk(args):
    # args will be a tuple like (file, in_folder, out_folder)
    return process_file_vctk(*args)

def wrapper_vocalset(args):
    # args will be a tuple like (file, in_folder, out_folder)
    # The default values for train_prob etc. in process_file_vocalset will be used
    return process_file_vocalset(*args)

# multiprocessing
if __name__ == '__main__':



    # --- VCTK Processing ---
    print("Processing VCTK...")
    in_folders_vctk = ["dataset_dir\\vctk_dataset\\train", "dataset_dir\\vctk_dataset\\val", "dataset_dir\\vctk_dataset\\test"]
    out_folders_vctk = ["dataset_dir\\vctk_dataset\\train_processed_v3", "dataset_dir\\vctk_dataset\\val_processed_v3", "dataset_dir\\vctk_dataset\\test_processed_v3"]

    for in_folder, out_folder in zip(in_folders_vctk, out_folders_vctk):
        print(f"  Processing folder: {in_folder}")
        # Use rglob directly inside list comprehension for potentially better memory usage if needed
        files = [str(x.absolute()) for x in Path(in_folder).rglob("*.flac")]

        if not files:
            print(f"    No *.flac files found in {in_folder}. Skipping.")
            continue
        print(f"    Found {len(files)} files.")

        os.makedirs(out_folder, exist_ok=True)

        # Create an iterable of argument tuples
        # Each element is a tuple: (file, in_folder, out_folder)
        args_iterable_vctk = zip(files, [in_folder] * len(files), [out_folder] * len(files))

        with Pool() as pool:
            # Pass the wrapper function and the iterable of argument tuples to imap
            results_vctk = list(tqdm(pool.imap(wrapper_vctk, args_iterable_vctk),
                                    total=len(files),
                                    desc=f"Processing {Path(in_folder).name}"))

    # --- VocalSet Processing ---
    print("\nProcessing VocalSet...")
    in_folders_vocalset = ["dataset_dir\\vocalset_dataset\\FULL"]
    out_folders_vocalset = ["dataset_dir\\vocalset_dataset\\FULL_processed_v3"]

    for in_folder, out_folder in zip(in_folders_vocalset, out_folders_vocalset):
        print(f"  Processing folder: {in_folder}")
        files = [str(x.absolute()) for x in Path(in_folder).rglob("*.wav")]

        if not files:
            print(f"    No *.wav files found in {in_folder}. Skipping.")
            continue
        print(f"    Found {len(files)} files.")

        os.makedirs(out_folder, exist_ok=True)

        # Create an iterable of argument tuples
        # Each element is a tuple: (file, in_folder, out_folder)
        args_iterable_vocalset = zip(files, [in_folder] * len(files), [out_folder] * len(files))

        with Pool() as pool:
            # Pass the wrapper function and the iterable of argument tuples to imap
            results_vocalset = list(tqdm(pool.imap(wrapper_vocalset, args_iterable_vocalset),
                                        total=len(files),
                                        desc=f"Processing {Path(in_folder).name}"))

    print("\nDone!")