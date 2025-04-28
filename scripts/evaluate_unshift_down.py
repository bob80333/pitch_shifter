# done training, final evaluation using audiobox + sisdr
from audiobox_aesthetics.infer import AesPredictor
import torch
from auraloss.time import SISDRLoss
import numpy as np
import os
from pathlib import Path
from pitch_shifter.model.model_1d_dac import WavUNetDAC
from pitch_shifter.data.data import PreShiftedAudioDataset
from torch.utils.data import DataLoader

path = "runs/outputs_unshift_down_gan/output5"

AMP_ENABLE = True
AMP_DTYPE = torch.float16

# performance tweaks
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = WavUNetDAC().to(device)
model.load_state_dict(torch.load(f"{path}/model_100000.pt"))

opt_model = torch.compile(model)

sr = 48_000

val_vctk_files = list(Path("dataset_dir/vctk_dataset/val_processed_unshift_down_2").rglob("*.wav"))
val_vctk_dataset = PreShiftedAudioDataset(val_vctk_files, test=True, samples=16384 * 8)
val_vctk_dataloader = DataLoader(
    val_vctk_dataset,
    batch_size=4,
    shuffle=False,
    drop_last=True,
    num_workers=8,
)

val_dl = [val_vctk_dataloader]
val_dl_name = ["vctk"]


predictor = AesPredictor(checkpoint_pth=None, batch_size=4)

# sisdr loss no averaging
sisdr_loss = SISDRLoss(reduction="none")

shifted_si_sdrs = []
unshifted_si_sdrs = []

original_pqs = []
shifted_pqs = []
unshifted_pqs = []

with torch.no_grad():
    from tqdm import tqdm

    model.eval()
    for loader, name in zip(val_dl, val_dl_name):
        for audio, shifted_audio in tqdm(loader):
            audio, shifted_audio = audio.to(device), shifted_audio.to(device)
            # add channels
            audio = audio.unsqueeze(1)
            shifted_audio = shifted_audio.unsqueeze(1)

            with torch.autocast("cuda", enabled=AMP_ENABLE, dtype=AMP_DTYPE):
                # predict clean audio from shifted audio
                unshifted_audio = opt_model(shifted_audio)

            unshifted_audio = (
                unshifted_audio.float()
            )  # convert back to fp32 for metrics

            # negate because loss is inverted
            shifted_si_sdr = -sisdr_loss(shifted_audio, audio)
            unshifted_si_sdr = -sisdr_loss(unshifted_audio, audio)

            # pq = Production Quality
            audio_input = [{"path": a, "sample_rate": sr} for a in audio]
            shifted_audio_input = [
                {"path": a, "sample_rate": sr} for a in shifted_audio
            ]
            unshifted_audio_input = [
                {"path": a, "sample_rate": sr} for a in unshifted_audio
            ]
            original_pq = predictor.forward(audio_input)
            shifted_pq = predictor.forward(shifted_audio_input)
            unshifted_pq = predictor.forward(unshifted_audio_input)

            shifted_si_sdrs.append(shifted_si_sdr.tolist())
            unshifted_si_sdrs.append(unshifted_si_sdr.tolist())

            # pull out PQ values
            original_pqs.append([pq["PQ"] for pq in original_pq])
            shifted_pqs.append([pq["PQ"] for pq in shifted_pq])
            unshifted_pqs.append([pq["PQ"] for pq in unshifted_pq])

        # save results
        with open(os.path.join(path, f"{name}_results.txt"), "w") as f:
            f.write("Shifted PQs\n")
            f.write(str(shifted_pqs))
            f.write("\n")
            f.write("Unshifted PQs\n")
            f.write(str(unshifted_pqs))
            f.write("\n")
            f.write("Shifted SI-SDRs\n")
            f.write(str(shifted_si_sdrs))
            f.write("\n")
            f.write("Unshifted SI-SDRs\n")
            f.write(str(unshifted_si_sdrs))
            f.write("\n")

        with open(os.path.join(path, f"{name}_summary.txt"), "w") as f:
            f.write(f"What, mean, std, median, min, max\n")
            f.write(
                f"Original PQ, {np.mean(original_pqs)}, {np.std(original_pqs)}, {np.median(original_pqs)}, {np.min(original_pqs)}, {np.max(original_pqs)}\n"
            )
            f.write(
                f"Shifted PQ, {np.mean(shifted_pqs)}, {np.std(shifted_pqs)}, {np.median(shifted_pqs)}, {np.min(shifted_pqs)}, {np.max(shifted_pqs)}\n"
            )
            f.write(
                f"Unshifted PQ, {np.mean(unshifted_pqs)}, {np.std(unshifted_pqs)}, {np.median(unshifted_pqs)}, {np.min(unshifted_pqs)}, {np.max(unshifted_pqs)}\n"
            )
            f.write(
                f"Shifted SI-SDR, {np.mean(shifted_si_sdrs)}, {np.std(shifted_si_sdrs)}, {np.median(shifted_si_sdrs)}, {np.min(shifted_si_sdrs)}, {np.max(shifted_si_sdrs)}\n"
            )
            f.write(
                f"Unshifted SI-SDR, {np.mean(unshifted_si_sdrs)}, {np.std(unshifted_si_sdrs)}, {np.median(unshifted_si_sdrs)}, {np.min(unshifted_si_sdrs)}, {np.max(unshifted_si_sdrs)}\n"
            )