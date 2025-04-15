from muon import Muon
import torch
import torch._dynamo.cache_size
from pitch_shifter.model.model_1d_dac import WavUNetDAC
from pitch_shifter.data.data import PreShiftedAudioDataset
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
from tqdm import trange
from auraloss.time import SISDRLoss
from torch.utils.tensorboard import SummaryWriter
import os
import torchaudio.transforms as T
from audiotools.core.audio_signal import AudioSignal

# from dac.model import Discriminator
from pitch_shifter.model.dac_discriminator import Discriminator
from dac.nn.loss import GANLoss, MelSpectrogramLoss
import numpy as np
import random
from torch import amp

from torch.amp.grad_scaler import GradScaler

sr = 48_000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hop = sr // 200

AMP_ENABLE = False
AMP_DTYPE = torch.bfloat16

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
    # increase torch compile cache limit
    print(
        f"Current cache size limit: {torch._dynamo.config.cache_size_limit}, upping to 64"
    )
    torch._dynamo.config.cache_size_limit = 64

    # set seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    g = torch.Generator()
    g.manual_seed(0)

    model = WavUNetDAC()
    model.to(device)

    opt_model = torch.compile(model)

    disc = Discriminator(
        sample_rate=sr,
        bands=[[0.0, 0.1], [0.1, 0.25], [0.25, 0.5], [0.5, 0.75], [0.75, 1.0]],
    )
    disc.to(device)

    # just use adamw as optimizer, due to muon being slow
    g_optimizers = [
        torch.optim.AdamW(
            model.parameters(), lr=1e-4, betas=(0.8, 0.99), weight_decay=0.01
        ),
    ]

    d_optimizers = [
        torch.optim.AdamW(
            disc.parameters(), lr=1e-4, betas=(0.8, 0.99), weight_decay=0.01
        ),
    ]

    # decay lr linearly to 0
    g_schedulers = [
        torch.optim.lr_scheduler.LambdaLR(optim, lambda step: 1 - step / args.n_steps)
        for optim in g_optimizers
    ]

    d_schedulers = [
        torch.optim.lr_scheduler.LambdaLR(optim, lambda step: 1 - step / args.n_steps)
        for optim in d_optimizers
    ]

    train_vctk_files = list(
        Path("dataset_dir/vctk_dataset/train_processed_unshift_down_2").rglob("*.wav")
    )
    train_vctk_dataset = PreShiftedAudioDataset(train_vctk_files)
    train_vctk_dataloader = DataLoader(
        train_vctk_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.n_workers,
        persistent_workers=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    val_vctk_files = list(Path("dataset_dir/vctk_dataset/val_processed_unshift_down_2").rglob("*.wav"))
    val_vctk_dataset = PreShiftedAudioDataset(val_vctk_files, test=True, samples=16384 * 8)
    val_vctk_dataloader = DataLoader(
        val_vctk_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.n_workers,
        persistent_workers=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    sisdr_loss = SISDRLoss()

    # train losses
    gan_loss = GANLoss(disc)

    # using setting from DAC base.yml:
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
    
    val_dl = [val_vctk_dataloader]
    val_dl_name = ["vctk"]
    
    grad_scaler_g = GradScaler()
    grad_scaler_d = GradScaler()

    for step in trange(args.n_steps):
        model.train()

        audio, shifted_audio = next(vctk_train_gen)
        audio = audio.to(device)
        shifted_audio = shifted_audio.to(device)

        # add channels dimension for losses calculation
        audio = audio.unsqueeze(1)
        shifted_audio = shifted_audio.unsqueeze(1)

        audio = AudioSignal(audio, sample_rate=sr)

        for d_opt in d_optimizers:
            d_opt.zero_grad()

        with amp.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=AMP_ENABLE):

            # discriminator loss
            # no grad for generator
            with torch.no_grad():
                unshifted_audio = opt_model(shifted_audio)
                unshifted_audio = AudioSignal(unshifted_audio, sample_rate=sr)

            d_loss = gan_loss.discriminator_loss(unshifted_audio, audio)

        grad_scaler_d.scale(d_loss).backward()
        
        # unscale for grad clipping
        for opt in d_optimizers:
            grad_scaler_d.unscale_(opt)

        writer.add_scalar(
            "train/disc_grad_norm",
            torch.nn.utils.clip_grad_norm_(disc.parameters(), 1e2).item(),
            step + 1,
        )

        for d_opt in d_optimizers:
            grad_scaler_d.step(opt)

        for d_sched in d_schedulers:
            d_sched.step()

        grad_scaler_d.update()

        # generator loss
        for g_opt in g_optimizers:
            g_opt.zero_grad()

        with torch.autocast("cuda", dtype=AMP_DTYPE, enabled=AMP_ENABLE):

            unshifted_audio = opt_model(shifted_audio)
            unshifted_audio = AudioSignal(unshifted_audio, sample_rate=sr)

            mel_loss = melspec_loss(unshifted_audio, audio)

            gen_loss, feat_loss = gan_loss.generator_loss(unshifted_audio, audio)
            g_loss = gen_loss + 2.0 * feat_loss + 15.0 * mel_loss
            
        grad_scaler_g.scale(g_loss).backward()
        
        for opt in g_optimizers:
            grad_scaler_g.unscale_(opt)

        writer.add_scalar(
            "train/gen_grad_norm",
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1e3).item(),
            step + 1,
        )

        for g_opt in g_optimizers:
            grad_scaler_g.step(g_opt)

        for g_sched in g_schedulers:
            g_sched.step()
            
        grad_scaler_g.update()

        writer.add_scalar("train_g/loss", g_loss, step + 1)
        writer.add_scalar("train_g/feat_loss", feat_loss, step + 1)
        writer.add_scalar("train_g/gen_loss", gen_loss, step + 1)
        writer.add_scalar("train_g/mel_loss", mel_loss, step + 1)
        writer.add_scalar("train_d/loss", d_loss, step + 1)

        if (step + 1) % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                for loader, name in zip(val_dl, val_dl_name):
                    total_val_loss = 0
                    total_val_sisdr = 0
                    total_shifted_loss = 0
                    i = 0
                    for audio, shifted_audio in val_dl:
                        audio = audio.to(device)
                        shifted_audio = shifted_audio.to(device)

                        # add channels dimension for losses calculation
                        audio = audio.unsqueeze(1)
                        shifted_audio = shifted_audio.unsqueeze(1)

                        with torch.autocast("cuda", dtype=AMP_DTYPE, enabled=AMP_ENABLE):

                            # remove pitch artifacts from shifted audio
                            unshifted_audio = opt_model(shifted_audio)

                        val_loss = melspec_loss(unshifted_audio, audio)
                        # calculate sisdr loss
                        val_sisdr = sisdr_loss(unshifted_audio, audio)
                        shifted_loss = melspec_loss(shifted_audio, audio)

                        total_val_loss += val_loss
                        total_val_sisdr += val_sisdr
                        total_shifted_loss += shifted_loss
                        i += 1

                    # val losses
                    total_val_loss /= i
                    total_val_sisdr /= i
                    # baseline, if model output is worse than this, it's not useful
                    total_shifted_loss /= i

                    writer.add_scalar(f"val_{name}/loss", total_val_loss, step + 1)
                    writer.add_scalar(f"val_{name}/sisdr", total_val_sisdr, step + 1)
                    writer.add_scalar(f"val_{name}/shifted_loss", total_shifted_loss, step + 1)

                    # save an example output
                    writer.add_audio(f"val_{name}/audio", audio[0], step + 1, sample_rate=sr)
                    writer.add_audio(
                        f"val_{name}/shifted_audio", shifted_audio[0], step + 1, sample_rate=sr
                    )
                    writer.add_audio(
                        f"val_{name}/unshifted_audio", unshifted_audio[0], step + 1, sample_rate=sr
                    )

                    audio_spec = to_log(to_spectrogram(audio[0]))
                    # normalize to between 0 and 1
                    audio_spec = (audio_spec - audio_spec.min()) / (
                        audio_spec.max() - audio_spec.min()
                    )
                    # do for the rest
                    shifted_audio_spec = to_log(to_spectrogram(shifted_audio[0]))
                    shifted_audio_spec = (shifted_audio_spec - shifted_audio_spec.min()) / (
                        shifted_audio_spec.max() - shifted_audio_spec.min()
                    )
                    unshifted_audio_spec = to_log(to_spectrogram(unshifted_audio[0]))
                    unshifted_audio_spec = (
                        unshifted_audio_spec - unshifted_audio_spec.min()
                    ) / (unshifted_audio_spec.max() - unshifted_audio_spec.min())

                    # save spectrograms
                    writer.add_image(f"val_{name}/audio_spec", audio_spec, step + 1)
                    writer.add_image(
                        f"val_{name}/unshifted_audio_spec", unshifted_audio_spec, step + 1
                    )
                    writer.add_image(f"val_{name}/shifted_audio_spec", shifted_audio_spec, step + 1)

                print(f"Step {step+1}, val_loss: {total_val_loss}")

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


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--n_steps", type=int, default=100_000)
    argparser.add_argument("--eval_every", type=int, default=1000)
    argparser.add_argument("--batch_size", type=int, default=8)
    argparser.add_argument("--n_workers", type=int, default=4)
    argparser.add_argument(
        "--save_dir", type=str, default="runs/outputs_unshift_down_gan/output3"
    )

    args = argparser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    main(args)
