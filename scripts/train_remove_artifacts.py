from muon import Muon
import torch
from pitch_shifter.model.model_1d_dac import WavUNetDAC
from pitch_shifter.data.data import PreShiftedAudioDatasetV2
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
from tqdm import trange
from auraloss.time import SISDRLoss
from torch.utils.tensorboard import SummaryWriter
import os
import torchaudio.transforms as T
from audiotools.metrics.spectral import MelSpectrogramLoss
from audiotools.core.audio_signal import AudioSignal
import numpy as np
import random

sr = 48_000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hop = sr // 200

# performance tweaks
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


# set seed for reproducibility
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def inf_train_generator(train_loader):
    while True:
        for data in train_loader:
            yield data


def main(args):

    # set seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    g = torch.Generator()
    g.manual_seed(0)

    model = WavUNetDAC()
    model.to(device)

    opt_model = torch.compile(model)

    # for newer version of Muon
    # Find ≥2D parameters in the body of the network -- these will be optimized by Muon
    muon_params = [p for p in model.parameters() if p.ndim >= 2]
    # Find everything else -- these will be optimized by AdamW
    adamw_params = [p for p in model.parameters() if p.ndim < 2]
    # Create the optimizer
    optimizers = [
        Muon(muon_params, lr=args.muon_lr, momentum=0.95, weight_decay=0.01),
        torch.optim.AdamW(adamw_params, lr=args.adam_lr, betas=(0.90, 0.95), weight_decay=0.01),
    ]
    # decay lr linearly to 0 across training steps for each optimizer
    schedulers = [
        torch.optim.lr_scheduler.LambdaLR(optimizers[0], lambda step: 1 - step / args.n_steps),
        torch.optim.lr_scheduler.LambdaLR(optimizers[1], lambda step: 1 - step / args.n_steps),
    ]

    train_vctk_files = list(Path("dataset_dir/vctk_dataset/train_processed_v3").rglob("*.flac"))
    train_vctk_dataset = PreShiftedAudioDatasetV2(train_vctk_files, samples=16384 * 3)
    train_vctk_dataloader = DataLoader(
        train_vctk_dataset,
        batch_size=args.batch_size//2, # half batch size for each dataset
        shuffle=True,
        drop_last=True,
        num_workers=args.n_workers,
        persistent_workers=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    train_vocalset_files = list(Path("dataset_dir/vocalset_dataset/train_processed_v3").rglob("*.flac"))
    train_vocalset_dataset = PreShiftedAudioDatasetV2(train_vocalset_files, samples=16384 * 3)
    train_vocalset_dataloader = DataLoader(
        train_vocalset_dataset,
        batch_size=args.batch_size//2, # half batch size for each dataset
        shuffle=True,
        drop_last=True,
        num_workers=args.n_workers,
        persistent_workers=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    val_vctk_files = list(Path("dataset_dir/val_processed_v3").rglob("*.flac"))
    val_vctk_dataset = PreShiftedAudioDatasetV2(val_vctk_files, test=True, samples=16384 * 12)
    val_vctk_dataloader = DataLoader(
        val_vctk_dataset,
        batch_size=32,
        shuffle=False,
        drop_last=True,
        num_workers=args.n_workers,
        persistent_workers=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    val_vocalset_files = list(Path("dataset_dir/vocalset_dataset/val_processed_v3").rglob("*.flac"))
    val_vocalset_dataset = PreShiftedAudioDatasetV2(val_vocalset_files, test=True, samples=16384 * 12)
    val_vocalset_dataloader = DataLoader(
        val_vocalset_dataset,
        batch_size=32,
        shuffle=False,
        drop_last=True,
        num_workers=args.n_workers,
        persistent_workers=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    sisdr_loss = SISDRLoss()
    l1_loss_fn = torch.nn.L1Loss()

    melspec_loss = MelSpectrogramLoss(
        n_mels=[5, 10, 20, 40, 80, 160, 320],
        window_lengths=[32, 64, 128, 256, 512, 1024, 2048],
        mel_fmin=[0, 0, 0, 0, 0, 0, 0],
        mel_fmax=[None, None, None, None, None, None, None],
        pow=1.0,
        clamp_eps=1.0e-5,
        mag_weight=0.0,
    )

    
    to_spectrogram = T.Spectrogram(1024, 1024, 256, power=2).to(device)
    to_log = T.AmplitudeToDB().to(device)

    writer = SummaryWriter(args.save_dir)

    vctk_train_gen = inf_train_generator(train_vctk_dataloader)
    vocalset_train_gen = inf_train_generator(train_vocalset_dataloader)

    for step in trange(args.n_steps):
        model.train()
        for opt in optimizers:
            opt.zero_grad()

        audio_vctk, shifted_audio_vctk = next(vctk_train_gen)
        audio_vocalset, shifted_audio_vocalset = next(vocalset_train_gen)

        audio = torch.cat([audio_vctk, audio_vocalset], dim=0).to(device)
        shifted_audio = torch.cat([shifted_audio_vctk, shifted_audio_vocalset], dim=0).to(device)

        # add channels dimension for losses calculation
        audio = audio.unsqueeze(1)
        shifted_audio = shifted_audio.unsqueeze(1)

        # predict clean audio from shifted audio
        unshifted_audio = opt_model(shifted_audio)

        l1_loss = l1_loss_fn(unshifted_audio, audio)

        # make tensors AudioSignals for MelSpectrogramLoss (takes in tensors, so should preserve gradients)
        audio = AudioSignal(audio, sr)
        unshifted_audio = AudioSignal(unshifted_audio, sr)

        mel_loss = melspec_loss(unshifted_audio, audio)

        loss = mel_loss + 10 * l1_loss
        loss.backward()
        # log / clip grad norm
        writer.add_scalar(
            "train/grad_norm",
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1e3).item(),
            step + 1,
        )

        for opt in optimizers:
            opt.step()
        
        for scheduler in schedulers:
            scheduler.step()

        writer.add_scalar("train/loss", loss, step + 1)
        writer.add_scalar("train/mel_loss", mel_loss, step + 1)
        writer.add_scalar("train/l1_loss", l1_loss, step + 1)

        if (step + 1) % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                
                val_dl = [val_vctk_dataloader, val_vocalset_dataloader]
                val_dl_name = ["vctk", "vocalset"]
                for loader, name in zip(val_dl, val_dl_name):
                    total_val_loss = 0
                    total_val_sisdr = 0
                    total_shifted_loss = 0
                    i = 0
                    for audio, shifted_audio in loader:
                        audio = audio.to(device)
                        shifted_audio = shifted_audio.to(device)

                        # add channels dimension for losses calculation
                        audio = audio.unsqueeze(1)
                        shifted_audio = shifted_audio.unsqueeze(1)

                        # remove pitch artifacts from shifted audio
                        unshifted_audio = opt_model(shifted_audio)


                        # calculate sisdr loss
                        val_sisdr = sisdr_loss(unshifted_audio, audio)
                        # convert to AudioSignal for MelSpectrogramLoss
                        audio_sig = AudioSignal(audio, sr)
                        unshifted_audio_sig = AudioSignal(unshifted_audio, sr)
                        shifted_audio_sig = AudioSignal(shifted_audio, sr)

                        # calculate stft error for unshifted audio, should not have artifacts from shifting and should be back to original pitch
                        val_loss = melspec_loss(unshifted_audio_sig, audio_sig)
                        # calculate the error for the shifted audio, should have artifacts from shifting, the model output should have a better error than this
                        shifted_loss = melspec_loss(shifted_audio_sig, audio_sig)

                        total_val_loss += val_loss
                        total_val_sisdr += val_sisdr
                        total_shifted_loss += shifted_loss
                        i += 1

                    total_val_loss /= i
                    total_shifted_loss /= i
                    total_val_sisdr /= i

                    writer.add_scalar(f"val_{name}/mel_loss", total_val_loss, step + 1)
                    writer.add_scalar(f"val_{name}/sisdr", total_val_sisdr, step + 1)
                    # baseline, if model output is worse than this, it's not useful
                    writer.add_scalar(f"val_{name}/shifted_mel_loss", total_shifted_loss, step + 1)

                    # save an example output
                    writer.add_audio(f"val_{name}/audio", audio[0], step + 1, sample_rate=sr)
                    writer.add_audio(
                        f"val_{name}/shifted_audio", shifted_audio[0], step + 1, sample_rate=sr
                    )
                    writer.add_audio(
                        f"val_{name}/unshifted_audio", unshifted_audio[0], step + 1, sample_rate=sr
                    )

                    # spectrogram visualization

                    audio_spec = to_log(to_spectrogram(audio[0]))
                    # normalize to between 0 and 1
                    audio_spec = (audio_spec - audio_spec.min()) / (audio_spec.max() - audio_spec.min())
                    # do for the rest
                    shifted_audio_spec = to_log(to_spectrogram(shifted_audio[0]))
                    shifted_audio_spec = (shifted_audio_spec - shifted_audio_spec.min()) / (shifted_audio_spec.max() - shifted_audio_spec.min())
                    unshifted_audio_spec = to_log(to_spectrogram(unshifted_audio[0]))
                    unshifted_audio_spec = (unshifted_audio_spec - unshifted_audio_spec.min()) / (unshifted_audio_spec.max() - unshifted_audio_spec.min())


                    # save spectrograms
                    writer.add_image(f"val_{name}/audio_spec", audio_spec, step+1)
                    writer.add_image(f"val_{name}/unshifted_audio_spec", unshifted_audio_spec, step+1)
                    writer.add_image(f"val_{name}/shifted_audio_spec", shifted_audio_spec, step+1)

                print(f"Step {step+1}, {name}_val_loss: {total_val_loss}")

            torch.save(
                model.state_dict(), os.path.join(args.save_dir, f"model_{step+1}.pt")
            )

    # done training, final evaluation using audiobox + sisdr
    from audiobox_aesthetics.infer import AesPredictor
    predictor = AesPredictor(checkpoint_pth=None, batch_size=args.batch_size)

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

                unshifted_audio = opt_model(shifted_audio)

                # negate because loss is inverted
                shifted_si_sdr = -sisdr_loss(shifted_audio, audio)
                unshifted_si_sdr = -sisdr_loss(unshifted_audio, audio)

                # pq = Production Quality
                audio_input = [{"path": a, "sample_rate": sr} for a in audio]
                shifted_audio_input = [{"path": a, "sample_rate": sr} for a in shifted_audio]
                unshifted_audio_input = [{"path": a, "sample_rate": sr} for a in unshifted_audio]
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
        with open(os.path.join(args.save_dir, f"{name}_results.txt"), "w") as f:
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

        with open(os.path.join(args.save_dir, f"{name}_summary.txt"), "w") as f:
            f.write(f"What, mean, std, median, min, max\n")
            f.write(f"Original PQ, {np.mean(original_pqs)}, {np.std(original_pqs)}, {np.median(original_pqs)}, {np.min(original_pqs)}, {np.max(original_pqs)}\n")
            f.write(f"Shifted PQ, {np.mean(shifted_pqs)}, {np.std(shifted_pqs)}, {np.median(shifted_pqs)}, {np.min(shifted_pqs)}, {np.max(shifted_pqs)}\n")
            f.write(f"Unshifted PQ, {np.mean(unshifted_pqs)}, {np.std(unshifted_pqs)}, {np.median(unshifted_pqs)}, {np.min(unshifted_pqs)}, {np.max(unshifted_pqs)}\n")
            f.write(f"Shifted SI-SDR, {np.mean(shifted_si_sdrs)}, {np.std(shifted_si_sdrs)}, {np.median(shifted_si_sdrs)}, {np.min(shifted_si_sdrs)}, {np.max(shifted_si_sdrs)}\n")
            f.write(f"Unshifted SI-SDR, {np.mean(unshifted_si_sdrs)}, {np.std(unshifted_si_sdrs)}, {np.median(unshifted_si_sdrs)}, {np.min(unshifted_si_sdrs)}, {np.max(unshifted_si_sdrs)}\n")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--n_steps", type=int, default=20_000)
    argparser.add_argument("--eval_every", type=int, default=1000)
    argparser.add_argument("--batch_size", type=int, default=32)
    argparser.add_argument("--n_workers", type=int, default=4)
    argparser.add_argument("--save_dir", type=str, default="runs/outputs/output110")
    argparser.add_argument("--muon_lr", type=float, default=1e-3)
    argparser.add_argument("--adam_lr", type=float, default=1e-4)

    args = argparser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    main(args)
