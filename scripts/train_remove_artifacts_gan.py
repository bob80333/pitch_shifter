from muon import Muon
import torch
import torch._dynamo.cache_size
from pitch_shifter.model.model_1d_v2 import WavUNet
from pitch_shifter.data.data import PreShiftedAudioDataset
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
#from dac.model import Discriminator
from pitch_shifter.model.dac_discriminator import Discriminator
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

    # increase torch compile cache limit
    print(f"Current cache size limit: {torch._dynamo.config.cache_size_limit}, upping to 64")
    torch._dynamo.config.cache_size_limit = 64


    # set seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    g = torch.Generator()
    g.manual_seed(0)


    model = WavUNet()
    model.to(device)

    opt_model = torch.compile(model)

    disc = Discriminator(sample_rate=sr, bands=[[0.0, 0.1], [0.1, 0.25], [0.25, 0.5], [0.5, 0.75], [0.75, 1.0]])
    disc.to(device)

    # this used to work, but then i restarted my computer and now almost none of the discriminators compile
    # idk

    for i in range(len(disc.discriminators)):
        if i == 1 or i == 2 or i == 3 or i == 4:
            continue # skip the second one
        if i < 5:
            disc.discriminators[i] = torch.compile(disc.discriminators[i])
        else:
            # compile each conv separately, the whole model can't be compiled
            # doubly nested list comprehension to compile each conv in each band (originally is doubly nested modulelist)
            #disc.discriminators[i].band_convs = torch.nn.ModuleList([torch.nn.ModuleList([torch.compile(conv) for conv in band]) for band in disc.discriminators[i].band_convs])
            pass # faster to not compile these for some reason

    # # compile the convs of each discriminator for faster training
    # i = 0
    # for discriminator in disc.discriminators:
    #     i += 1
    #     if i == 2:
    #         continue # this one breaks compilation for some reason /shrug
    #         # get an error about unexpected dtype "fp32"??? idk
    #     if hasattr(discriminator, "convs"):
    #         # can't compile the whole modulelist, as each block needs to be run separately for the multiscale feature matching loss
    #         discriminator.convs = torch.nn.ModuleList([torch.compile(conv) for conv in discriminator.convs])
        
        # only compile first 7
        #if i > 6:
        #    break
    #disc_opt = torch.compile(disc)



    # for newer version of Muon
    # Find ≥2D parameters in the body of the network -- these will be optimized by Muon
    muon_params = [p for p in model.parameters() if p.ndim >= 2]
    # Find everything else -- these will be optimized by AdamW
    adamw_params = [p for p in model.parameters() if p.ndim < 2]
    # Create the optimizer
    g_optimizers = [
        Muon(muon_params, lr=1e-3, momentum=0.95, weight_decay=0.01),
        torch.optim.AdamW(adamw_params, lr=1e-4, betas=(0.8, 0.99), weight_decay=0.01),
    ]

    # for newer version of Muon
    # Find ≥2D parameters in the body of the network -- these will be optimized by Muon
    muon_params = [p for p in disc.parameters() if p.ndim >= 2]
    # Find everything else -- these will be optimized by AdamW
    adamw_params = [p for p in disc.parameters() if p.ndim < 2]
    # Create the optimizer
    d_optimizers = [
        Muon(muon_params, lr=1e-3, momentum=0.95, weight_decay=0.01),
        torch.optim.AdamW(adamw_params, lr=1e-4, betas=(0.8, 0.99), weight_decay=0.01),
    ]

    # g_optim = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.8, 0.99), weight_decay=0.01)
    # d_optim = torch.optim.AdamW(disc.parameters(), lr=1e-4, betas=(0.8, 0.99), weight_decay=0.01)

    # multiply the LR by gamma every step, as in DAC
    LR_GAMMA = 0.999996
    LR_STEP_SIZE = 1

    # g_scheduler = torch.optim.lr_scheduler.StepLR(g_optim, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)
    # d_scheduler = torch.optim.lr_scheduler.StepLR(d_optim, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

    g_schedulers = [torch.optim.lr_scheduler.StepLR(optim, step_size=LR_STEP_SIZE, gamma=LR_GAMMA) for optim in g_optimizers]
    d_schedulers = [torch.optim.lr_scheduler.StepLR(optim, step_size=LR_STEP_SIZE, gamma=LR_GAMMA) for optim in d_optimizers]

    train_files = list(Path("dataset_dir/train_processed_v2").rglob("*.wav"))
    print(f"Found {len(train_files)} training files")
    train_dataset = PreShiftedAudioDataset(train_files)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.n_workers, persistent_workers=True, worker_init_fn=seed_worker, generator=g)

    val_files = list(Path("dataset_dir/val_processed_v2").rglob("*.wav"))
    print(f"Found {len(val_files)} validation files")
    val_dataset = PreShiftedAudioDataset(val_files, test=True, samples=16384*8)
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

    to_spectrogram = T.Spectrogram(1024, 1024, 256, power=2).to(device)
    to_log = T.AmplitudeToDB().to(device)

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

        # d_optim.zero_grad()

        for d_opt in d_optimizers:
            d_opt.zero_grad()

        with amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=AMP_ENABLE):

            # discriminator loss
            # no grad for generator
            with torch.no_grad():
                unshifted_audio = opt_model(shifted_audio)
                unshifted_audio = AudioSignal(unshifted_audio, sample_rate=sr)
            

            d_loss = gan_loss.discriminator_loss(unshifted_audio, audio)

        d_loss.backward()

        writer.add_scalar("train/disc_grad_norm", torch.nn.utils.clip_grad_norm_(disc.parameters(), 1e2).item(), step+1)

        # d_optim.step()

        # d_scheduler.step()

        for d_opt in d_optimizers:
            d_opt.step()

        for d_sched in d_schedulers:
            d_sched.step()

        # generator loss
        # g_optim.zero_grad()
        for g_opt in g_optimizers:
            g_opt.zero_grad()

        with amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=AMP_ENABLE):

            unshifted_audio = opt_model(shifted_audio)
            unshifted_audio = AudioSignal(unshifted_audio, sample_rate=sr)

            mel_loss = melspec_loss(unshifted_audio, audio)

            gen_loss, feat_loss  = gan_loss.generator_loss(unshifted_audio, audio)
            g_loss = gen_loss + 2.0 * feat_loss + 15.0 * mel_loss
        g_loss.backward()

        writer.add_scalar("train/gen_grad_norm", torch.nn.utils.clip_grad_norm_(model.parameters(), 1e3).item(), step+1)

        # g_optim.step()

        for g_opt in g_optimizers:
            g_opt.step()

        # g_scheduler.step()

        for g_sched in g_schedulers:
            g_sched.step()

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
                        unshifted_audio = opt_model(shifted_audio)

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

                audio_spec = to_log(to_spectrogram(audio[0]))
                # normalize to between 0 and 1
                audio_spec = (audio_spec - audio_spec.min()) / (audio_spec.max() - audio_spec.min())
                # do for the rest
                shifted_audio_spec = to_log(to_spectrogram(shifted_audio[0]))
                shifted_audio_spec = (shifted_audio_spec - shifted_audio_spec.min()) / (shifted_audio_spec.max() - shifted_audio_spec.min())
                unshifted_audio_spec = to_log(to_spectrogram(unshifted_audio[0]))
                unshifted_audio_spec = (unshifted_audio_spec - unshifted_audio_spec.min()) / (unshifted_audio_spec.max() - unshifted_audio_spec.min())


                # save spectrograms
                writer.add_image("val/audio_spec", audio_spec, step+1)
                writer.add_image("val/unshifted_audio_spec", unshifted_audio_spec, step+1)
                writer.add_image("val/shifted_audio_spec", shifted_audio_spec, step+1)

            print(f"Step {step+1}, val_loss: {total_val_loss}")

            torch.save(model.state_dict(), os.path.join(args.save_dir, f"model_{step+1}.pt"))


    



if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--n_steps", type=int, default=100_000)
    argparser.add_argument("--eval_every", type=int, default=1000)
    argparser.add_argument("--batch_size", type=int, default=8)
    argparser.add_argument("--n_workers", type=int, default=4)
    argparser.add_argument("--save_dir", type=str, default="runs/outputs_gan/output11" )

    args = argparser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    main(args)