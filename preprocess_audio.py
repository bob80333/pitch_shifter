from pathlib import Path
from tqdm import tqdm
import soundfile as sf
import numpy as np
from multiprocessing import Pool
import python_stretch as ps
from torchaudio import functional as F
import os

in_folder = "data\\val"
out_folder = "data\\val_processed"

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

    for shift in range(-12, 13, 1):
        stretch.setTransposeSemitones(shift)

        shifted_audio = stretch.process(audio[None, :])[0]

        # save the audio file
        sf.write(new_filename.replace(".flac", f"_shifted_{shift}.wav"), shifted_audio, sr)

# multiprocessing

if __name__ == '__main__':
    with Pool() as pool:
        results = list(tqdm(pool.imap(process_file, files), total=len(files)))
    print("Done!")
