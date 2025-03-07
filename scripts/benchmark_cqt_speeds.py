from tqdm import trange
import torch
from time import time
# sisdr loss
from auraloss.time import SISDRLoss
si_sdr = SISDRLoss().to("cuda")

from cqt_pytorch import CQT

# test this library: https://github.com/archinetai/cqt-pytorch
def test_cqt_1(iterations = 200, batch_size = 32, channels = 1, timesteps = 16384*2, compile=True):
    transform = CQT(
        num_octaves = 8,
        num_bins_per_octave = 64,
        sample_rate = 48000,
        block_length = timesteps
    ).to("cuda")

    def test():
        x = torch.randn(batch_size, channels, timesteps).to("cuda")
        y = transform.encode(x)
        z = transform.decode(y)

    if compile:
        test = torch.compile(test)
    
    with torch.no_grad():
        # warmup
        for _ in trange(50):
            test()

        start = time()
        for _ in trange(iterations):
            test()
        end = time()

    print(f"Time taken: {end-start:.2f}s")

    # check how well it reconstructs:
    x = torch.randn(batch_size, channels, timesteps).to("cuda")
    y = transform.encode(x)
    z = transform.decode(y)
    print(f"Reconstruction error: {(x - z).abs().max()}")
    print(f"negative si-sdr: {si_sdr(z, x)}")


# test this library: https://github.com/eloimoliner/CQT_pytorch
from cqt_nsgt_pytorch import CQT_nsgt

def test_cqt_2(iterations = 200, batch_size = 32, channels = 1, timesteps = 16384*2, compile=True):
    transform = CQT_nsgt(
        numocts =8,
        binsoct = 64,
        fs = 48000,
        audio_len = timesteps,
        device="cuda"
    )

    def test():
        x = torch.randn(batch_size, channels, timesteps).to("cuda")
        y = transform.fwd(x)
        z = transform.bwd(y)

    if compile:
        test = torch.compile(test)
    
    with torch.no_grad():
        # warmup
        for _ in trange(50):
            test()

        start = time()
        for _ in trange(iterations):
            test()
        end = time()

    print(f"Time taken: {end-start:.2f}s")

    # check how well it reconstructs:
    x = torch.randn(batch_size, channels, timesteps).to("cuda")
    y = transform.fwd(x)
    z = transform.bwd(y)
    print(f"Reconstruction error: {(x - z).abs().max()}")
    print(f"negative si-sdr: {si_sdr(z, x)}")


if __name__ == "__main__":
    print("Testing cqt-pytorch")
    # measure max memory usage before and after
    max_memory_before = torch.cuda.max_memory_allocated()
    test_cqt_1(compile=False)
    max_memory_after = torch.cuda.max_memory_allocated()
    print(f"Max memory used: {max_memory_after - max_memory_before}")
    # reset max memory usage
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    print("Testing cqt-nsgt-pytorch")
    test_cqt_2(compile=False)
    max_memory_after = torch.cuda.max_memory_allocated()
    print(f"Max memory used: {max_memory_after - max_memory_before}")

    # reconstruction error when adding random noise with scale 1e-4:
    x = torch.randn(32, 1, 16384*2).to("cuda")
    z = x + torch.randn_like(x) * 1e-3
    print(f"Reconstruction error with noise: {(x - z).abs().max()}")
    print(f"negative si-sdr with noise: {si_sdr(z, x)}")