import soundfile as sf
import python_stretch as ps
import numpy as np
import torch
from torchaudio.functional import resample
from auraloss.time import SISDRLoss
si_sdr = SISDRLoss()
import torch
from pitch_shifter.model.model_1d_v2 import WavUNet
audio, sr = sf.read("example/p250_003_mic2.flac")
#audio, sr = sf.read("example/ado_singing.wav")
print(audio.shape, sr)

stretcher = ps.Signalsmith.Stretch()
stretcher.preset(1, sr)


model = WavUNet().to("cuda")

model_checkpoints = range(1000, 15001, 1000)

for checkpoint in model_checkpoints:
    print("Step: ", checkpoint)
    model.load_state_dict(torch.load(f"runs/outputs/output91/model_{checkpoint}.pt"))
#model.load_state_dict(torch.load("runs/outputs/output84/model_1000.pt"))

    # run the model on the audio after it has been shifted up
    # and then shift back down, and see if the shifted down audio is better than baseline shifted down audio

    #sf.write("example/evaluate/original.wav", audio, sr)

    shifts = [i for i in range(1, 13)]

    biggest_improvement = float("-inf")
    best_shift = None
    shifted_sisdr = None
    shifted_artifact_removed = None
    shifted_subtracted = None

    shifted_sisdrs = []
    shifted_artifact_removeds = []
    shifted_subtracteds = []

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

        # dumb baseline: calculate error from shifting up and then back down, and remove it from shifted up audio before shifting down again

        difference = audio - shifted_down
        shifted_up_subtracted = shifted_up - difference
        shifted_down_subtracted = stretcher.process(shifted_up_subtracted)

        #sf.write(f"example/evaluate/shifted_{shift}.wav", shifted_down[0], sr)
        #sf.write(f"example/evaluate/shifted_no_artifact_{shift}.wav", shifted_down_no_artifact[0], sr)
        #sf.write(f"example/evaluate/shifted_subtracted_{shift}.wav", shifted_down_subtracted[0], sr)

        # calculate si-sdr for both (negative due to it being a loss)
        si_sdr_shifted = -si_sdr(torch.tensor(shifted_down), torch.tensor(audio))
        si_sdr_shifted_no_artifact = -si_sdr(torch.tensor(shifted_down_no_artifact), torch.tensor(audio))
        si_sdr_shifted_subtracted = -si_sdr(torch.tensor(shifted_down_subtracted), torch.tensor(audio))

        shifted_sisdrs.append(si_sdr_shifted.item())
        shifted_artifact_removeds.append(si_sdr_shifted_no_artifact.item())
        shifted_subtracteds.append(si_sdr_shifted_subtracted.item())

        if si_sdr_shifted_no_artifact > si_sdr_shifted:
            improvement = si_sdr_shifted_no_artifact - si_sdr_shifted
            if improvement > biggest_improvement:
                biggest_improvement = improvement
                best_shift = shift
                shifted_sisdr = si_sdr_shifted
                shifted_artifact_removed = si_sdr_shifted_no_artifact
                shifted_subtracted = si_sdr_shifted_subtracted

        #print(f"Shift: {shift}, si-sdr shifted: {si_sdr_shifted}, si-sdr shifted no artifact: {si_sdr_shifted_no_artifact}, si-sdr shifted subtracted: {si_sdr_shifted_subtracted}")

    #print(f"Best shift: {best_shift}, improvement (si-sdr): {biggest_improvement}")
    #print(f"Best shift results: si-sdr shifted: {shifted_sisdr}, si-sdr shifted no artifact: {shifted_artifact_removed}, si-sdr shifted subtracted: {shifted_subtracted}")

    #print(f"Average si-sdr shifted: {np.mean(shifted_sisdrs)}, si-sdr shifted no artifact: {np.mean(shifted_artifact_removeds)}, si-sdr shifted subtracted: {np.mean(shifted_subtracteds)}")
    #print(f"Median si-sdr shifted: {np.median(shifted_sisdrs)}, si-sdr shifted no artifact: {np.median(shifted_artifact_removeds)}, si-sdr shifted subtracted: {np.median(shifted_subtracteds)}")
    #print(f"Std si-sdr shifted: {np.std(shifted_sisdrs)}, si-sdr shifted no artifact: {np.std(shifted_artifact_removeds)}, si-sdr shifted subtracted: {np.std(shifted_subtracteds)}")

    print(f"Average improvements: Model - {- (np.mean(shifted_sisdrs) - np.mean(shifted_artifact_removeds))} Subtraction - {- (np.mean(shifted_sisdrs) - np.mean(shifted_subtracteds))}")

