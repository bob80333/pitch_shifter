from muon import Muon
import torch
from model.model_1d_v2 import WavUNet
from data.data import PreShiftedAudioDataset
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
from tqdm import trange
from auraloss.freq import MultiResolutionSTFTLoss
from auraloss.time import SISDRLoss
from torch.utils.tensorboard import SummaryWriter
import os
import torchaudio.transforms as T
from audiotools.core.audio_signal import AudioSignal
from dac.model import Discriminator
from dac.nn.loss import GANLoss, MelSpectrogramLoss
import numpy as np
import random
from torch import amp

sr = 48_000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hop = sr // 200
AMP_ENABLE = False

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


    model = WavUNet()
    model.to(device)

    disc = Discriminator(sample_rate=sr, bands=[[0.0, 0.1], [0.1, 0.25], [0.25, 0.5], [0.5, 0.75], [0.75, 1.0]])
    disc.to(device)

    # # Find ≥2D parameters in the body of the network -- these will be optimized by Muon
    # muon_params = [p for p in model.parameters() if p.ndim >= 2]
    # # Find everything else -- these will be optimized by AdamW
    # adamw_params = [p for p in model.parameters() if p.ndim < 2]
    # # Create the optimizer
    # g_optim = Muon(muon_params, lr=1e-3, momentum=0.95,
    #                 adamw_params=adamw_params, adamw_lr=1e-4, adamw_betas=(0.8, 0.99), adamw_wd=0.01)
    
    # # Find ≥2D parameters in the body of the network -- these will be optimized by Muon
    # muon_params = [p for p in disc.parameters() if p.ndim >= 2]
    # # Find everything else -- these will be optimized by AdamW
    # adamw_params = [p for p in disc.parameters() if p.ndim < 2]
    # # Create the optimizer
    # d_optim = Muon(muon_params, lr=1e-3, momentum=0.95,
    #                 adamw_params=adamw_params, adamw_lr=1e-4, adamw_betas=(0.8, 0.99), adamw_wd=0.01)

    g_optim = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.8, 0.99), weight_decay=0.01)
    d_optim = torch.optim.AdamW(disc.parameters(), lr=1e-4, betas=(0.8, 0.99), weight_decay=0.01)

    train_files = list(Path("dataset_dir/train_processed").rglob("*.wav"))
    print(f"Found {len(train_files)} training files")
    train_dataset = PreShiftedAudioDataset(train_files)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.n_workers, persistent_workers=True, worker_init_fn=seed_worker, generator=g)

    val_files = list(Path("dataset_dir/val_processed").rglob("*.wav"))
    print(f"Found {len(val_files)} validation files")
    val_dataset = PreShiftedAudioDataset(val_files, test=True, samples=16384*4)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=args.n_workers, persistent_workers=True, worker_init_fn=seed_worker, generator=g)

    # val losses
    stft_loss = MultiResolutionSTFTLoss(fft_sizes = [4096, 2048, 1024], hop_sizes = [480, 240, 120], win_lengths = [2400, 1200, 600], scale="mel", n_bins=128, sample_rate=sr, perceptual_weighting=True)
    
    sisdr_loss = SISDRLoss()
    # l1 loss on wav directly as part of val    
    l1_loss = torch.nn.L1Loss()

    # train losses
    gan_loss = GANLoss(disc)


    # using setting from DAC base.yml:
    # MelSpectrogramLoss.n_mels: [5, 10, 20, 40, 80, 160, 320]
    # MelSpectrogramLoss.window_lengths: [32, 64, 128, 256, 512, 1024, 2048]
    # MelSpectrogramLoss.mel_fmin: [0, 0, 0, 0, 0, 0, 0]
    # MelSpectrogramLoss.mel_fmax: [null, null, null, null, null, null, null]
    # MelSpectrogramLoss.pow: 1.0
    # MelSpectrogramLoss.clamp_eps: 1.0e-5
    # MelSpectrogramLoss.mag_weight: 0.0
    melspec_loss = MelSpectrogramLoss(n_mels=[5, 10, 20, 40, 80, 160, 320], window_lengths=[32, 64, 128, 256, 512, 1024, 2048], mel_fmin=[0, 0, 0, 0, 0, 0, 0], mel_fmax=[None, None, None, None, None, None, None], pow=1.0, clamp_eps=1.0e-5, mag_weight=0.0)

    writer = SummaryWriter(args.save_dir)

    train_gen = inf_train_generator(train_dataloader)


    for step in trange(args.n_steps):
        model.train()

        audio, shifted_audio = next(train_gen)
        audio = audio.to(device)
        shifted_audio = shifted_audio.to(device)

        # add channels dimension for losses calculation
        audio = audio.unsqueeze(1)
        shifted_audio = shifted_audio.unsqueeze(1)

        audio = AudioSignal(audio, sample_rate=sr)

        d_optim.zero_grad()

        with amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=AMP_ENABLE):

            # discriminator loss
            # no grad for generator
            with torch.no_grad():
                unshifted_audio = model(shifted_audio)
                unshifted_audio = AudioSignal(unshifted_audio, sample_rate=sr)
            

            d_loss = gan_loss.discriminator_loss(unshifted_audio, audio)

        d_loss.backward()

        writer.add_scalar("train/disc_grad_norm", torch.nn.utils.clip_grad_norm_(disc.parameters(), 1e2).item(), step+1)

        d_optim.step()

        # generator loss
        g_optim.zero_grad()

        with amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=AMP_ENABLE):

            unshifted_audio = model(shifted_audio)
            unshifted_audio = AudioSignal(unshifted_audio, sample_rate=sr)

            mel_loss = melspec_loss(unshifted_audio, audio)

            gen_loss, feat_loss  = gan_loss.generator_loss(unshifted_audio, audio)
            g_loss = gen_loss + 2.0 * feat_loss + 15.0 * mel_loss
        g_loss.backward()

        writer.add_scalar("train/gen_grad_norm", torch.nn.utils.clip_grad_norm_(model.parameters(), 1e3).item(), step+1)

        g_optim.step()

        writer.add_scalar("train_g/loss", g_loss, step+1)
        writer.add_scalar("train_g/feat_loss", feat_loss, step+1)
        writer.add_scalar("train_g/gen_loss", gen_loss, step+1)
        writer.add_scalar("train_g/mel_loss", mel_loss, step+1)
        writer.add_scalar("train_d/loss", d_loss, step+1)

        if (step + 1) % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                total_val_loss = 0
                total_val_sisdr = 0
                total_val_l1 = 0
                total_shifted_loss = 0
                i = 0
                for audio, shifted_audio in val_dataloader:
                    audio = audio.to(device)
                    shifted_audio = shifted_audio.to(device)

                    # add channels dimension for losses calculation
                    audio = audio.unsqueeze(1)
                    shifted_audio = shifted_audio.unsqueeze(1)

                    with amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=AMP_ENABLE):
                                            
                        # remove pitch artifacts from shifted audio
                        unshifted_audio = model(shifted_audio)

                    # calculate stft error for unshifted audio, should not have artifacts from shifting and should be back to original pitch
                    val_loss = stft_loss(unshifted_audio, audio)
                    # calculate sisdr loss
                    val_sisdr = sisdr_loss(unshifted_audio, audio)
                    # calculate l1 loss
                    val_l1 = l1_loss(unshifted_audio, audio)
                    # calculate the error for the shifted audio, should have artifacts from shifting, the model output should have a better error than this
                    shifted_loss = stft_loss(shifted_audio, audio)

                    total_val_loss += val_loss
                    total_val_sisdr += val_sisdr
                    total_val_l1 += val_l1
                    total_shifted_loss += shifted_loss
                    i += 1
                
                # val losses
                total_val_loss /= i
                total_val_sisdr /= i
                total_val_l1 /= i
                # baseline, if model output is worse than this, it's not useful
                total_shifted_loss /= i
                
                writer.add_scalar("val/loss", total_val_loss, step+1)
                writer.add_scalar("val/sisdr", total_val_sisdr, step+1)
                writer.add_scalar("val/l1", total_val_l1, step+1)
                writer.add_scalar("val/shifted_loss", total_shifted_loss, step+1)

                # save an example output
                writer.add_audio("val/audio", audio[0], step+1, sample_rate=sr)
                writer.add_audio("val/shifted_audio", shifted_audio[0], step+1, sample_rate=sr)
                writer.add_audio("val/unshifted_audio", unshifted_audio[0], step+1, sample_rate=sr)

            print(f"Step {step+1}, val_loss: {total_val_loss}")

            torch.save(model.state_dict(), os.path.join(args.save_dir, f"model_{step+1}.pt"))


    



if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--n_steps", type=int, default=10_000)
    argparser.add_argument("--eval_every", type=int, default=1000)
    argparser.add_argument("--batch_size", type=int, default=16)
    argparser.add_argument("--n_workers", type=int, default=3)
    argparser.add_argument("--save_dir", type=str, default="runs/outputs_gan/output10" )

    args = argparser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    main(args)