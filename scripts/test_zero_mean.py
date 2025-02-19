import torch
from auraloss.time import SISDRLoss
from pathlib import Path
from data.data import PreShiftedAudioDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.model_1d_v2 import WavUNet

if __name__ == "__main__":

    val_files = list(Path("dataset_dir/val_processed").rglob("*.wav"))
    print(f"Found {len(val_files)} validation files")
    val_dataset = PreShiftedAudioDataset(val_files, test=True, samples=16384*16)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, drop_last=True, num_workers=0)

    model = WavUNet().to("cuda")
    model.load_state_dict(torch.load("outputs/output78/model_30000.pt"))

    si_sdr_zero_mean = SISDRLoss(zero_mean=True)
    si_sdr_normal_mean = SISDRLoss(zero_mean=False)

    model.eval()

    with torch.no_grad():
        total_sisdr_zero_mean = 0
        total_sisdr_normal_mean = 0

        total_shifted_si_sdr_zero_mean = 0
        total_shifted_si_sdr_normal_mean = 0
        i = 0

        for original, shifted in tqdm(val_dataloader):
            # move to gpu
            original = original.to("cuda")
            shifted = shifted.to("cuda")
            # add channel dimension
            original = original.unsqueeze(1)
            shifted = shifted.unsqueeze(1)

            # run model
            unshifted_audio = model(shifted)

            # calculate SISDR for model outputs
            # add negative since its a loss function, so it was inverted
            total_sisdr_normal_mean += -si_sdr_normal_mean(unshifted_audio, original).sum()
            total_sisdr_zero_mean += -si_sdr_zero_mean(unshifted_audio, original).sum()

            # calculate SISDR for shifted audio (model input vs original to see how much the model is fixing)
            total_shifted_si_sdr_normal_mean += -si_sdr_normal_mean(shifted, original).sum()
            total_shifted_si_sdr_zero_mean += -si_sdr_zero_mean(shifted, original).sum()

            i += 1

        print(f"Average SISDR zero mean for model output: {total_sisdr_zero_mean / i}")
        print(f"Average SISDR normal mean for model output: {total_sisdr_normal_mean / i}")

        print(f"Average SISDR zero mean for shifted audio: {total_shifted_si_sdr_zero_mean / i}")
        print(f"Average SISDR normal mean for shifted audio: {total_shifted_si_sdr_normal_mean / i}")