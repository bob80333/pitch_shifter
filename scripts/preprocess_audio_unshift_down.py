from pathlib import Path
from tqdm import tqdm
import soundfile as sf
import numpy as np
from multiprocessing import Pool
import python_stretch as ps
import torch
from torchaudio.functional import resample
import os

in_folder = "dataset_dir\\train"
out_folder = "dataset_dir\\train_processed_unshift_down_2"

files = [str(x.absolute()) for x in Path(in_folder).rglob("*.flac")]



os.makedirs(out_folder, exist_ok=True)

def process_file(file, in_folder=in_folder, out_folder=out_folder):
    stretch = ps.Signalsmith.Stretch()
    stretch.preset(1, 48_000)

    new_filename = file.replace(in_folder, out_folder)
    # make sure the directory exists
    os.makedirs(os.path.dirname(new_filename), exist_ok=True)

    audio, sr = sf.read(file)
    audio = audio.astype(np.float32) # convert to float32

    # resample audio down to 24kHz and then back up to remove high frequencies
    # use kaiser and more filter width to get better quality
    audio = torch.from_numpy(audio).unsqueeze(0)
    audio = resample(audio, sr, 24_000, lowpass_filter_width=200, resampling_method="sinc_interp_kaiser")
    audio = resample(audio, 24_000, 48_000, lowpass_filter_width=200, resampling_method="sinc_interp_kaiser")
    audio = audio.squeeze().numpy()

    # only deal with large shift for now
    shifts = [12]
    for shift in shifts:
        stretch.setTransposeSemitones(shift)
        shifted_audio = stretch.process(audio[None, :])[0]

        # save the audio file
        sf.write(new_filename.replace(".flac", f"_shifted_{shift}.wav"), shifted_audio, sr)

    sf.write(new_filename.replace(".flac", f"_baseline.wav"), audio, sr)

# multiprocessing

if __name__ == '__main__':
    with Pool() as pool:
        results = list(tqdm(pool.imap(process_file, files), total=len(files)))
    print("Done!")
