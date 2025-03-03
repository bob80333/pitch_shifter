from muon import Muon
import torch
from pitch_shifter.model.model_1d_v2_diffusion import WavUNet
from pitch_shifter.data.data import PreShiftedDownAudioDataset
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
from tqdm import trange
from auraloss.freq import MultiResolutionSTFTLoss
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

# performance tweaks
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


# from k-diffusion discord channel in eleutherai

import torch.distributed as dist

def stratified_uniform(shape, grad_accum_steps=1, grad_accum_step=0, group=None, world_size=None, rank=None, dtype=None, device=None):
    """Draws stratified samples from a uniform distribution. The strata are not duplicated
    across processes or gradient accumulation steps."""
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size(group) if world_size is None else world_size
        rank = dist.get_rank(group) if rank is None else rank
    else:
        world_size = 1 if world_size is None else world_size
        rank = 0 if rank is None else rank
    world_size = world_size * grad_accum_steps
    rank = rank * grad_accum_steps + grad_accum_step
    n = shape[-1] * world_size
    start = rank * n // world_size
    end = (rank + 1) * n // world_size
    offsets = torch.linspace(0, 1, n + 1, dtype=dtype, device=device)[start:end]
    u = torch.rand(shape, dtype=dtype, device=device)
    return torch.clamp(offsets + u / n, 0, 1)

# based off of ddpm step code in lucidrains RIN repo

def gamma(t, ns=0.0002, ds=0.00025):
    return torch.cos(((t + ns) / (1 + ds)) * np.pi / 2)**2


def gamma_to_alpha_sigma(gamma, scale=1):
    return torch.sqrt(gamma) * scale, torch.sqrt(1 - gamma)


def safe_div(numer, denom, eps=1e-10):
    return numer / denom.clamp(min=eps)


def ddpm_step_lucidrains(x_t, eps_pred, t_now, t_next):
    t_now = torch.tensor(t_now, device=device)
    t_next = torch.tensor(t_next, device=device)
    gamma_now = gamma(t_now)
    gamma_next = gamma(t_next)
    alpha_now, sigma_now = gamma_to_alpha_sigma(gamma_now)
    alpha_next, sigma_next = gamma_to_alpha_sigma(gamma_next)

    # convert eps into x_0
    x_start = safe_div(x_t - sigma_now * eps_pred, alpha_now)

    # clip
    x_start.clamp_(-1.0, 1.0)

    # get predicted noise
    pred_noise = safe_div(x_t - alpha_now * x_start, sigma_now)

    # calculate next x_t
    x_next = x_start * alpha_next + pred_noise * sigma_next
    return x_next


def generate(steps, noise, model, conditioning):
    x_t = noise
    for step in trange(steps):
        # Get time for current and next states.
        t = 1 - step / steps
        timestep = torch.ones(x_t.shape[0], device=device) * t
        timestep = timestep.unsqueeze(-1)
        t_m1 = max(1 - (step + 1) / steps, 0)
        # Predict eps.
        
        model_in = torch.cat([x_t, conditioning], dim=1)
        eps_pred = model(model_in, timestep)
        # Estimate x at t_m1.
        x_t = ddpm_step_lucidrains(x_t, eps_pred, t, t_m1)
    return x_t

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

    model = WavUNet()
    model.to(device)

    opt_model = torch.compile(model)

    # for newer version of Muon
    # Find ≥2D parameters in the body of the network -- these will be optimized by Muon
    muon_params = [p for p in model.parameters() if p.ndim >= 2]
    # Find everything else -- these will be optimized by AdamW
    adamw_params = [p for p in model.parameters() if p.ndim < 2]
    # Create the optimizer
    optimizers = [
        Muon(muon_params, lr=5e-3, momentum=0.95, weight_decay=0.01),
        torch.optim.AdamW(adamw_params, lr=5e-4, betas=(0.90, 0.95), weight_decay=0.01),
    ]

    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.01)

    train_files = list(Path("dataset_dir/train_processed_unshift_down").rglob("*.wav"))
    print(f"Found {len(train_files)} training files")
    train_dataset = PreShiftedDownAudioDataset(train_files, samples=16384 * 2)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.n_workers,
        persistent_workers=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    val_files = list(Path("dataset_dir/val_processed_unshift_down").rglob("*.wav"))
    print(f"Found {len(val_files)} validation files")
    val_dataset = PreShiftedDownAudioDataset(val_files, test=True, samples=16384 * 8)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.n_workers,
        persistent_workers=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    stft_loss = MultiResolutionSTFTLoss(
        fft_sizes=[4096, 2048, 1024],
        hop_sizes=[480, 240, 120],
        win_lengths=[2400, 1200, 600],
        scale="mel",
        n_bins=128,
        sample_rate=sr,
        perceptual_weighting=True,
    )

    sisdr_loss = SISDRLoss()
    l1_loss_fn = torch.nn.L1Loss()

    # cdpam_loss = cdpam.CDPAM(dev='cuda:0')

    # resampler = T.Resample(48000, 22050).to(device)

    # dac_loss = DACFeatureMatchingLoss(device)

    # wavlm_loss = WavLMFeatureMatchingLoss(device)

    # using setting from DAC base.yml:
    # MelSpectrogramLoss.n_mels: [5, 10, 20, 40, 80, 160, 320]
    # MelSpectrogramLoss.window_lengths: [32, 64, 128, 256, 512, 1024, 2048]
    # MelSpectrogramLoss.mel_fmin: [0, 0, 0, 0, 0, 0, 0]
    # MelSpectrogramLoss.mel_fmax: [null, null, null, null, null, null, null]
    # MelSpectrogramLoss.pow: 1.0
    # MelSpectrogramLoss.clamp_eps: 1.0e-5
    # MelSpectrogramLoss.mag_weight: 0.0
    melspec_loss = MelSpectrogramLoss(
        n_mels=[5, 10, 20, 40, 80, 160, 320],
        window_lengths=[32, 64, 128, 256, 512, 1024, 2048],
        mel_fmin=[0, 0, 0, 0, 0, 0, 0],
        mel_fmax=[None, None, None, None, None, None, None],
        pow=1.0,
        clamp_eps=1.0e-5,
        mag_weight=0.0,
    )

    writer = SummaryWriter(args.save_dir)

    train_gen = inf_train_generator(train_dataloader)

    for step in trange(args.n_steps):
        model.train()
        # optimizer.zero_grad()
        for opt in optimizers:
            opt.zero_grad()

        audio, shifted_audio = next(train_gen)
        audio = audio.to(device)
        shifted_audio = shifted_audio.to(device)

        # add channels dimension for losses calculation
        audio = audio.unsqueeze(1)
        shifted_audio = shifted_audio.unsqueeze(1)

        noise = torch.randn_like(shifted_audio)

        timesteps = stratified_uniform((shifted_audio.shape[0], 1, 1), grad_accum_steps=1, grad_accum_step=0, dtype=torch.float32, device=device)

        #timesteps = torch.rand(shifted_audio.shape[0], 1, 1, device=device)

        # create input for model
        x_t = audio * gamma(timesteps).sqrt() + noise * (1 - gamma(timesteps)).sqrt()

        # combine input with shifted audio (conditioning)
        conditioned_input = torch.cat([x_t, shifted_audio], dim=1)

        # predict fixed input audio

        # predict noise (eps style)
        pred_noise = opt_model(conditioned_input, timesteps[:, :, 0])

        # calculate stft error for unshifted audio, should not have artifacts from shifting and should be back to original pitch
        # loss = stft_loss(unshifted_audio, audio)

        l1_loss = l1_loss_fn(pred_noise, noise)

        # loss = cdpam_loss.forward(resampler(audio), resampler(unshifted_audio)).mean()

        # loss = dac_loss(unshifted_audio, audio)
        # feature_loss = wavlm_loss(unshifted_audio, audio)

        # make tensors AudioSignals for MelSpectrogramLoss (takes in tensors, so should preserve gradients)
        #audio = AudioSignal(audio, sr)
        #unshifted_audio = AudioSignal(unshifted_audio, sr)

        noise = AudioSignal(noise, sr)
        pred_noise = AudioSignal(pred_noise, sr)

        mel_loss = melspec_loss(pred_noise, noise)

        loss = mel_loss + 10 * l1_loss
        # loss = l1_loss(unshifted_audio, audio)
        loss.backward()
        # log / clip grad norm
        writer.add_scalar(
            "train/grad_norm",
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1e3).item(),
            step + 1,
        )

        # optimizer.step()
        for opt in optimizers:
            opt.step()

        writer.add_scalar("train/loss", loss, step + 1)
        writer.add_scalar("train/mel_loss", mel_loss, step + 1)
        writer.add_scalar("train/l1_loss", l1_loss, step + 1)
        # writer.add_scalar("train/wavlm_loss", feature_loss, step+1)

        if (step + 1) % args.eval_every == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                total_val_loss = 0
                total_val_sisdr = 0
                total_shifted_loss = 0
                i = 0
                for audio, shifted_audio in val_dataloader:
                    audio = audio.to(device)
                    shifted_audio = shifted_audio.to(device)

                    # add channels dimension for losses calculation
                    audio = audio.unsqueeze(1)
                    shifted_audio = shifted_audio.unsqueeze(1)

                    # remove pitch artifacts from shifted audio
                    #unshifted_audio = opt_model(shifted_audio)
                    
                    noise = torch.randn_like(audio)
                    
                    unshifted_audio = generate(100, noise, opt_model, shifted_audio)

                    # calculate stft error for unshifted audio, should not have artifacts from shifting and should be back to original pitch
                    val_loss = stft_loss(unshifted_audio, audio)
                    # calculate sisdr loss
                    val_sisdr = sisdr_loss(unshifted_audio, audio)
                    # calculate the error for the shifted audio, should have artifacts from shifting, the model output should have a better error than this
                    shifted_loss = stft_loss(shifted_audio, audio)

                    total_val_loss += val_loss
                    total_val_sisdr += val_sisdr
                    total_shifted_loss += shifted_loss
                    i += 1

                    # stop after ~100 examples (3 x batch size) (for faster evaluation, since it does 100 steps of diffusion)
                    if i >= 2:
                        break

                total_val_loss /= i
                total_shifted_loss /= i
                total_val_sisdr /= i

                writer.add_scalar("val/loss", total_val_loss, step + 1)
                writer.add_scalar("val/sisdr", total_val_sisdr, step + 1)
                # baseline, if model output is worse than this, it's not useful
                writer.add_scalar("val/shifted_loss", total_shifted_loss, step + 1)

                # save an example output
                writer.add_audio("val/audio", audio[0], step + 1, sample_rate=sr)
                writer.add_audio(
                    "val/shifted_audio", shifted_audio[0], step + 1, sample_rate=sr
                )
                writer.add_audio(
                    "val/unshifted_audio", unshifted_audio[0], step + 1, sample_rate=sr
                )

            print(f"Step {step+1}, val_loss: {total_val_loss}")

            torch.save(
                model.state_dict(), os.path.join(args.save_dir, f"model_{step+1}.pt")
            )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--n_steps", type=int, default=1_000)
    argparser.add_argument("--eval_every", type=int, default=100)
    argparser.add_argument("--batch_size", type=int, default=32)
    argparser.add_argument("--n_workers", type=int, default=6)
    argparser.add_argument(
        "--save_dir", type=str, default="runs/outputs_unshift_down_diffusion/output3"
    )

    args = argparser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    main(args)
