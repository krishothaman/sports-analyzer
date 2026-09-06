"""Environment check: is PyTorch installed correctly and can it see the GPU?

Run this any time something seems off. It answers, in order:
  1. Which Python and which PyTorch build am I running?
  2. Was this PyTorch compiled with CUDA support at all?
  3. Can it actually find and talk to the GPU right now?

A "+cpu" build will always report CUDA unavailable no matter how good the
GPU is -- the support simply is not compiled in.
"""

import sys

import torch


def main():
    print(f"Python:        {sys.version.split()[0]}")
    print(f"PyTorch:       {torch.__version__}")

    # torch.version.cuda is the CUDA version this build was COMPILED against.
    # It is None for a CPU-only build -- that alone tells you the wrong wheel
    # is installed, before we even look for a GPU.
    print(f"Built w/ CUDA: {torch.version.cuda or 'NO -- this is a CPU-only build'}")

    available = torch.cuda.is_available()
    print(f"CUDA usable:   {available}")

    if not available:
        print("\nGPU not usable. Either the CPU-only wheel is installed,")
        print("or the NVIDIA driver is not visible to PyTorch.")
        print("See Task 1 of the Phase 0 plan for the reinstall command.")
        return 1

    # Only reachable when a GPU is genuinely available.
    props = torch.cuda.get_device_properties(0)
    print(f"GPU:           {props.name}")
    print(f"VRAM:          {props.total_memory / 1024**3:.1f} GB")

    # A real computation on the device. is_available() can be true while an
    # actual operation still fails (driver mismatch), so we prove it works
    # rather than trusting the flag.
    x = torch.randn(1000, 1000, device="cuda")
    y = x @ x
    torch.cuda.synchronize()  # GPU work is async; wait for it before claiming success
    print(f"Test matmul:   OK (result sum {y.sum().item():.1f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
