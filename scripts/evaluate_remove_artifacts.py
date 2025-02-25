import soundfile as sf
import python_stretch as ps
import numpy as np
import torch
from torchaudio.functional import resample
from auraloss.time import SISDRLoss
si_sdr = SISDRLoss()
import torch
from pitch_shifter.model.model_1d_v2 import WavUNet

model = WavUNet().to("cuda")
model.load_state_dict(torch.load("runs/outputs/output84/model_10000.pt"))

audio, sr = sf.read("example/p250_003_mic2.flac")
#audio, sr = sf.read("ado_singing.wav")

print(audio.shape, sr)

# run the model on the audio after it has been shifted up
# and then shift back down, and see if the shifted down audio is better than baseline shifted down audio

stretcher = ps.Signalsmith.Stretch()
stretcher.preset(1, sr)

sf.write("example/evaluate/original.wav", audio, sr)

shifts = [i for i in range(1, 13)]

for shift in shifts:
    stretcher.setTransposeSemitones(shift)
    shifted_up = stretcher.process(audio[None, :])

    with torch.no_grad():
        shifted_up_no_artifact = torch.tensor(shifted_up)
        shifted_up_no_artifact = shifted_up_no_artifact.unsqueeze(0).to("cuda")

        # pad audio to multiple of 16384
        pad = 16384 - shifted_up_no_artifact.shape[2] % 16384
        shifted_up_no_artifact = torch.nn.functional.pad(shifted_up_no_artifact, (0, pad))

        shifted_up_no_artifact = model(shifted_up_no_artifact)

        # remove padding
        shifted_up_no_artifact = shifted_up_no_artifact[:, :, :-pad]
        
        shifted_up_no_artifact = shifted_up_no_artifact.squeeze().detach().cpu().numpy()
    
    # shift both back down
    stretcher.setTransposeSemitones(-shift)
    shifted_down = stretcher.process(shifted_up)
    shifted_down_no_artifact = stretcher.process(shifted_up_no_artifact[None, :])

    sf.write(f"example/evaluate/shifted_{shift}.wav", shifted_down[0], sr)
    sf.write(f"example/evaluate/shifted_no_artifact_{shift}.wav", shifted_down_no_artifact[0], sr)

    # calculate si-sdr for both (negative due to it being a loss)
    si_sdr_shifted = -si_sdr(torch.tensor(shifted_down), torch.tensor(audio))
    si_sdr_shifted_no_artifact = -si_sdr(torch.tensor(shifted_down_no_artifact), torch.tensor(audio))

    print(f"Shift: {shift}, si-sdr shifted: {si_sdr_shifted}, si-sdr shifted no artifact: {si_sdr_shifted_no_artifact}")

