import torch
from pitch_shifter.model.model_1d_dac import WavUNetDAC

model = WavUNetDAC().to("cuda")
input = torch.rand((4, 1, 16384*4), device="cuda")
output = model(input)
output.sum().backward()

# exit cleanly if we are on a device that doesn't support torch.compile
if torch.cuda.get_device_capability() < (7, 0):
    print("Exiting because torch.compile is not supported on this device.")
    import sys
    sys.exit(0)


opt = torch.optim.AdamW(model.parameters(), lr=0.01, fused=True)
opt2 = torch.optim.AdamW(model.parameters(), lr=0.01, fused=False)


@torch.compile(fullgraph=False)
def fn():
    opt.step()


# Let's define a helpful benchmarking function:
import torch.utils.benchmark as benchmark


def benchmark_torch_function_in_microseconds(f, *args, **kwargs):
    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}
    )
    return t0.blocked_autorange().mean * 1e6


# # Warmup runs to compile the function
# for _ in range(5):
#     fn()

eager_runtime = benchmark_torch_function_in_microseconds(opt.step)
compiled_runtime = benchmark_torch_function_in_microseconds(opt2.step)

#assert eager_runtime > compiled_runtime

print(f"fused runtime: {eager_runtime}us")
print(f"non-fused runtime: {compiled_runtime}us")