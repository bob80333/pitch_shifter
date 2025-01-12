from muon import Muon
import torch
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
from pesto import load_model
from torch_pitch_shift import get_fast_shifts, pitch_shift


sr = 48000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hop = sr // 200

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
    optimizer = Muon(muon_params, lr=2e-3, momentum=0.95,
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
    pesto_model = load_model("mir-1k", step_size=20., sampling_rate=sr).to(device)
    pesto_model.eval()

    stft_loss = MultiResolutionSTFTLoss(fft_sizes = [4096, 2048, 1024], hop_sizes = [480, 240, 120], win_lengths = [2400, 1200, 600], scale="mel", n_bins=128, sample_rate=sr, perceptual_weighting=True)

    writer = SummaryWriter(args.save_dir)

    print("Pitch Shift Ratios", [float(x) for x in get_fast_shifts(sr)])

    fast_shift_ratios = torch.tensor([float(x) for x in get_fast_shifts(sr)]).to(device)

    train_gen = inf_train_generator(train_dataloader)

    for step in trange(args.n_steps):
        model.train()
        optimizer.zero_grad()

        audio = next(train_gen)
        audio = audio.to(device).float()

        # pick multiplier from a set of fast shifts
        pitch_multiplier = fast_shift_ratios[torch.randint(0, len(fast_shift_ratios), (1, 1), device=device)]

        # shift audio and then shift it back
        shifted_audio = model(audio, pitch_multiplier)
        unshifted_audio = model(shifted_audio, 1 / pitch_multiplier)

        # get embeddings of shifted audio vs original audio shifted to the same pitch
        traditional_shifted_audio = pitch_shift(audio.unsqueeze(1), pitch_multiplier.item(), sr)
        audio_pitch_embeds = pesto_model(traditional_shifted_audio.squeeze(), sr, return_activations=True)[2]
        shifted_pitch_embeds = pesto_model(shifted_audio, sr, return_activations=True)[2]
        #print(audio_pitch_embeds.shape, shifted_pitch_embeds.shape)
        #break

        # calculate pitch error for shifted (should be pitch * pitch_multiplier)
        pitch_error = pitch_loss(shifted_pitch_embeds, audio_pitch_embeds)

        # add channels dimension for losses calculation
        audio = audio.unsqueeze(1)
        unshifted_audio = unshifted_audio.unsqueeze(1)

        # calculate stft error for unshifted audio, should not have artifacts from shifting and should be back to original pitch
        stft_error = stft_loss(unshifted_audio, audio)

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
                total_val_loss = 0
                total_pitch_error = 0
                total_stft_error = 0
                i = 0
                for audio in val_dataloader:
                    audio = audio.to(device).float()
                    # always shift up by 1/2 octave (1.5x pitch)
                    pitch_multiplier = torch.tensor([1.5]).unsqueeze(1).to(device)
                    
                    # shift audio and then shift it back
                    shifted_audio = model(audio, pitch_multiplier)
                    unshifted_audio = model(shifted_audio, 1 / pitch_multiplier)

                    # get embeddings of shifted audio vs original audio shifted to the same pitch
                    traditional_shifted_audio = pitch_shift(audio.unsqueeze(1), pitch_multiplier.item(), sr)
                    audio_pitch_embeds = pesto_model(traditional_shifted_audio.squeeze(), sr, return_activations=True)[2]
                    shifted_pitch_embeds = pesto_model(shifted_audio, sr, return_activations=True)[2]
                    #print(audio_pitch_embeds.shape, shifted_pitch_embeds.shape)
                    #break

                    # calculate pitch error for shifted (should be pitch * pitch_multiplier)
                    pitch_error = pitch_loss(shifted_pitch_embeds, audio_pitch_embeds)

                    # add channels dimension for losses calculation
                    audio = audio.unsqueeze(1)
                    unshifted_audio = unshifted_audio.unsqueeze(1)

                    stft_error = stft_loss(unshifted_audio, audio)

                    val_loss = args.pitch_loss_weight * pitch_error +  args.stft_loss_weight * stft_error

                    total_val_loss += val_loss
                    total_pitch_error += pitch_error
                    total_stft_error += stft_error
                    i += 1
                
                total_val_loss /= i
                total_pitch_error /= i
                total_stft_error /= i
                writer.add_scalar("val/loss", total_val_loss, step)
                writer.add_scalar("val/pitch_error", total_pitch_error, step)
                writer.add_scalar("val/stft_error", total_stft_error, step)

                # save an example output
                writer.add_audio("val/audio", audio[0], step, sample_rate=sr)
                writer.add_audio("val/shifted_audio", shifted_audio[0], step, sample_rate=sr)
                writer.add_audio("val/unshifted_audio", unshifted_audio[0], step, sample_rate=sr)

            print(f"Step {step}, val_loss: {total_val_loss}, pitch_error: {total_pitch_error}, stft_error: {total_stft_error}")


    



if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--n_steps", type=int, default=5000)
    argparser.add_argument("--eval_every", type=int, default=50)
    argparser.add_argument("--batch_size", type=int, default=16)
    argparser.add_argument("--n_workers", type=int, default=4)
    argparser.add_argument("--pitch_loss_weight", type=float, default=1.0)
    argparser.add_argument("--stft_loss_weight", type=float, default=1000.0)
    argparser.add_argument("--save_dir", type=str, default="outputs/output7" )

    args = argparser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    main(args)