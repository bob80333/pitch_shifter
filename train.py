from muon import Muon
import torch
from model_2d import AudioUNet
from model_1d import WavUNet
from data import AudioDataset, PreShiftedAudioDataset
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
from tqdm import trange
from auraloss.freq import MultiResolutionSTFTLoss
from torch.utils.tensorboard import SummaryWriter
import os
import torchaudio.transforms as T
from dac_loss import DACFeatureMatchingLoss
from audiotools.metrics.spectral import MelSpectrogramLoss
from audiotools.core.audio_signal import AudioSignal
from heavyball import ForeachPSGDKron, ForeachSOAP
import heavyball


sr = 48000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hop = sr // 200

# performance tweaks
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


def inf_train_generator(train_loader):
    while True:
        for data in train_loader:
            yield data

def main(args):
    model = WavUNet()
    model.to(device)

    # Find ≥2D parameters in the body of the network -- these will be optimized by Muon
    # muon_params = [p for p in model.parameters() if p.ndim >= 2]
    # # Find everything else -- these will be optimized by AdamW
    # adamw_params = [p for p in model.parameters() if p.ndim < 2]
    # # Create the optimizer
    # optimizer = Muon(muon_params, lr=1e-2, momentum=0.95,
    #                 adamw_params=adamw_params, adamw_lr=1e-3, adamw_betas=(0.90, 0.95), adamw_wd=0.01)

    heavyball.utils.compile_mode = None # disable triton compiling on windows

    optimizer = ForeachPSGDKron(model.parameters(), lr=5e-4, beta = 0.95, weight_decay=0.01)

    #optimizer = ForeachSOAP(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.01)

    #optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.01)

    train_files = list(Path("data/train_processed").rglob("*.wav"))
    print(f"Found {len(train_files)} training files")
    train_dataset = PreShiftedAudioDataset(train_files)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.n_workers, persistent_workers=True)

    val_files = list(Path("data/val_processed").rglob("*.wav"))
    print(f"Found {len(val_files)} validation files")
    val_dataset = PreShiftedAudioDataset(val_files, test=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=args.n_workers, persistent_workers=True)

    stft_loss = MultiResolutionSTFTLoss(fft_sizes = [4096, 2048, 1024], hop_sizes = [480, 240, 120], win_lengths = [2400, 1200, 600], scale="mel", n_bins=128, sample_rate=sr, perceptual_weighting=True)
    #l1_loss = torch.nn.L1Loss()

    #cdpam_loss = cdpam.CDPAM(dev='cuda:0')

    #resampler = T.Resample(48000, 22050).to(device)

    #dac_loss = DACFeatureMatchingLoss(device)


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
        optimizer.zero_grad()

        audio, shifted_audio = next(train_gen)
        audio = audio.to(device)
        shifted_audio = shifted_audio.to(device)

        # add channels dimension for losses calculation
        audio = audio.unsqueeze(1)
        shifted_audio = shifted_audio.unsqueeze(1)

        # predict differences to fix the input audio
        unshifted_audio = model(shifted_audio)

        # calculate stft error for unshifted audio, should not have artifacts from shifting and should be back to original pitch
        #loss = stft_loss(unshifted_audio, audio)

        # loss = l1_loss(unshifted_audio, audio)

        #loss = cdpam_loss.forward(resampler(audio), resampler(unshifted_audio)).mean()

        #loss = dac_loss(unshifted_audio, audio)

        # make tensors AudioSignals for MelSpectrogramLoss (takes in tensors, so should preserve gradients)
        audio = AudioSignal(audio, sr)
        unshifted_audio = AudioSignal(unshifted_audio, sr)

        loss = melspec_loss(unshifted_audio, audio)

        loss.backward()
        # log / clip grad norm
        writer.add_scalar("train/grad_norm", torch.nn.utils.clip_grad_norm_(model.parameters(), 1e3).item(), step+1)

        optimizer.step()

        writer.add_scalar("train/loss", loss, step+1)

        if (step + 1) % args.eval_every == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                total_val_loss = 0
                total_shifted_loss = 0
                i = 0
                for audio, shifted_audio in val_dataloader:
                    audio = audio.to(device)
                    shifted_audio = shifted_audio.to(device)

                    # add channels dimension for losses calculation
                    audio = audio.unsqueeze(1)
                    shifted_audio = shifted_audio.unsqueeze(1)
                                        
                    # remove pitch artifacts from shifted audio
                    unshifted_audio = model(shifted_audio)

                    # calculate stft error for unshifted audio, should not have artifacts from shifting and should be back to original pitch
                    val_loss = stft_loss(unshifted_audio, audio)
                    # calculate the error for the shifted audio, should have artifacts from shifting, the model output should have a better error than this
                    shifted_loss = stft_loss(shifted_audio, audio)

                    total_val_loss += val_loss
                    total_shifted_loss += shifted_loss
                    i += 1
                
                total_val_loss /= i
                total_shifted_loss /= i
                
                writer.add_scalar("val/loss", total_val_loss, step+1)
                # baseline, if model output is worse than this, it's not useful
                writer.add_scalar("val/shifted_loss", total_shifted_loss, step+1)

                # save an example output
                writer.add_audio("val/audio", audio[0], step+1, sample_rate=sr)
                writer.add_audio("val/shifted_audio", shifted_audio[0], step+1, sample_rate=sr)
                writer.add_audio("val/unshifted_audio", unshifted_audio[0], step+1, sample_rate=sr)

            print(f"Step {step+1}, val_loss: {total_val_loss}")

            torch.save(model.state_dict(), os.path.join(args.save_dir, f"model_{step+1}.pt"))


    



if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--n_steps", type=int, default=100_000)
    argparser.add_argument("--eval_every", type=int, default=5000)
    argparser.add_argument("--batch_size", type=int, default=32)
    argparser.add_argument("--n_workers", type=int, default=6)
    argparser.add_argument("--save_dir", type=str, default="outputs/output55" )

    args = argparser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    main(args)