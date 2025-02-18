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

in_folder = "data\\train"
out_folder = "data\\train_processed_v2"

files = [str(x.absolute()) for x in Path(in_folder).rglob("*.flac")]



os.makedirs(out_folder, exist_ok=True)

def process_file(file, in_folder=in_folder, out_folder=out_folder):
    stretch = ps.Signalsmith.Stretch()
    stretch.preset(1, 48_000*2)
    resample_up = T.Resample(48_000, 48_000*2)
    resample_down = T.Resample(48_000*2, 48_000)

    new_filename = file.replace(in_folder, out_folder)
    # make sure the directory exists
    os.makedirs(os.path.dirname(new_filename), exist_ok=True)

    audio, sr = sf.read(file)
    audio = audio.astype(np.float32) # convert to float32

    # resample to 96 kHz, for the stretch algorithm
    audio_resamp = resample_up(torch.from_numpy(audio).unsqueeze(0)).squeeze().numpy()

    # only deal with large shifts for now
    shifts = [-12, -10, 10, 12]
    for shift in shifts:
        stretch.setTransposeSemitones(shift)
        shifted_audio = stretch.process(audio_resamp[None, :])
        stretch.setTransposeSemitones(-shift)
        shifted_audio = stretch.process(shifted_audio)[0]

        # resample back to 48 kHz
        shifted_audio = resample_down(torch.from_numpy(shifted_audio).unsqueeze(0)).squeeze().numpy()

        # save the audio file
        sf.write(new_filename.replace(".flac", f"_shifted_{shift}.wav"), shifted_audio, sr)

    sf.write(new_filename.replace(".flac", f"_baseline.wav"), audio, sr)

# multiprocessing

if __name__ == '__main__':
    with Pool() as pool:
        results = list(tqdm(pool.imap(process_file, files), total=len(files)))
    print("Done!")
