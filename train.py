from muon import Muon
import torch
import torch.optim as optim
from model import AudioUNet
from data import AudioDataset
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
from tqdm import trange
from auraloss.freq import MultiResolutionSTFTLoss
import torchaudio.functional as F
from torch.utils.tensorboard import SummaryWriter
import os

sr = 48000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def inf_train_generator(train_loader):
    while True:
        for data in train_loader:
            yield data

def main(args):
    model = AudioUNet()
    model.to(device)

    # Find ≥2D parameters in the body of the network -- these will be optimized by Muon
    muon_params = [p for p in model.parameters() if p.ndim >= 2]
    # Find everything else -- these will be optimized by AdamW
    adamw_params = [p for p in model.parameters() if p.ndim < 2]
    # Create the optimizer
    optimizer = Muon(muon_params, lr=0.02, momentum=0.95,
                    adamw_params=adamw_params, adamw_lr=3e-4, adamw_betas=(0.90, 0.95), adamw_wd=0.01)

    train_files = list(Path("data/train").rglob("*.flac"))
    print(f"Found {len(train_files)} training files")
    train_dataset = AudioDataset(train_files)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.n_workers, persistent_workers=True)

    val_files = list(Path("data/val").rglob("*.flac"))
    print(f"Found {len(val_files)} validation files")
    val_dataset = AudioDataset(val_files)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=args.n_workers, persistent_workers=True)

    pitch_loss = torch.nn.MSELoss()
    stft_loss = MultiResolutionSTFTLoss(fft_sizes = [4096, 2048, 1024], hop_sizes = [480, 240, 120], win_lengths = [2400, 1200, 600], scale="mel", n_bins=128, sample_rate=sr, perceptual_weighting=True)

    writer = SummaryWriter(args.save_dir)

    train_gen = inf_train_generator(train_dataloader)

    for step in trange(args.n_steps):
        model.train()
        optimizer.zero_grad()

        audio = next(train_gen)
        audio = audio.to(device).float()



        # pitch shift audio, multipler so that pitch is multiplied by 0.125 to 8.0 = -3 to +3 octaves at uniform random
        pitch_multiplier = torch.rand(args.batch_size, 1, device=device) * 7.875 + 0.125
        pitch_multiplier = pitch_multiplier.clamp(0.125, 8.0).float()

        # shift audio and then shift it back
        shifted_audio = model(audio, pitch_multiplier)
        unshifted_audio = model(shifted_audio, 1 / pitch_multiplier)

        # add channels dimension for losses calculation
        audio = audio.unsqueeze(1)
        shifted_audio = shifted_audio.unsqueeze(1)
        unshifted_audio = unshifted_audio.unsqueeze(1)

        # measure pitch of original / shifted / unshifted audio
        pitch = F.detect_pitch_frequency(audio, sr)
        shifted_audio_pitch = F.detect_pitch_frequency(shifted_audio, sr)
        unshifted_audio_pitch = F.detect_pitch_frequency(unshifted_audio, sr)

        print("pitch", pitch.shape)
        print("shifted", shifted_audio_pitch.shape)
        print("unshifted", unshifted_audio_pitch.shape)

        print("audio", audio.shape)
        print("shifted_audio", shifted_audio.shape)

        # calculate pitch error for shifted (should be pitch * pitch_multiplier) and unshifted (should be pitch)
        pitch_error = pitch_loss(pitch * pitch_multiplier, shifted_audio_pitch)
        pitch_error += pitch_loss(pitch, unshifted_audio_pitch)

        # calculate stft error for unshifted audio, should not have artifacts from shifting
        stft_error = stft_loss(audio, unshifted_audio)

        loss = args.pitch_loss_weight * pitch_error +  args.stft_loss_weight * stft_error

        loss.backward()
        optimizer.step()

        writer.add_scalar("train/loss", loss, step)
        writer.add_scalar("train/pitch_error", pitch_error, step)
        writer.add_scalar("train/stft_error", stft_error, step)

        if step % args.eval_every == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for audio in val_dataloader:
                    audio = audio.to(device).float()
                    shifted_audio = model(audio, pitch_multiplier)
                    unshifted_audio = model(shifted_audio, 1 / pitch_multiplier)

                    # add channels dimension for losses calculation
                    audio = audio.unsqueeze(1)
                    shifted_audio = shifted_audio.unsqueeze(1)
                    unshifted_audio = unshifted_audio.unsqueeze(1)

                    pitch = F.detect_pitch_frequency(audio, sr)
                    pitch_error = pitch_loss(pitch * pitch_multiplier, F.detect_pitch_frequency(shifted_audio, sr))
                    pitch_error += pitch_loss(pitch, F.detect_pitch_frequency(unshifted_audio, sr))

                    stft_error = stft_loss(audio, unshifted_audio)

                    val_loss += args.pitch_loss_weight * pitch_error +  args.stft_loss_weight * stft_error
                
                val_loss /= len(val_dataloader)
                pitch_error /= len(val_dataloader)
                stft_error /= len(val_dataloader)
                writer.add_scalar("val/loss", val_loss, step)
                writer.add_scalar("val/pitch_error", pitch_error, step)
                writer.add_scalar("val/stft_error", stft_error, step)

                # save an example output
                writer.add_audio("val/audio", audio[0], step, sample_rate=sr)
                writer.add_audio("val/shifted_audio", shifted_audio[0], step, sample_rate=sr)
                writer.add_audio("val/unshifted_audio", unshifted_audio[0], step, sample_rate=sr)

            print(f"Step {step}, val_loss: {val_loss}, pitch_error: {pitch_error}, stft_error: {stft_error}")


    



if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--n_steps", type=int, default=5000)
    argparser.add_argument("--eval_every", type=int, default=500)
    argparser.add_argument("--batch_size", type=int, default=32)
    argparser.add_argument("--n_workers", type=int, default=2)
    argparser.add_argument("--pitch_loss_weight", type=float, default=1.0)
    argparser.add_argument("--stft_loss_weight", type=float, default=0.0)
    argparser.add_argument("--save_dir", type=str, default="outputs/output1" )

    args = argparser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    main(args)