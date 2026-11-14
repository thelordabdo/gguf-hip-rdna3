"""Direct test of WanGP's GGUF linear dispatch predicates on the GPU.

Confirms the llama.cpp HIP fast path will be taken for IQ3_S (and other
qtypes) when Deepy loads the Qwen3.8-27B model — without a full model load.

Usage:
    python test_dispatch.py

Expected output:
    1. _gguf_cuda_kernels_enabled(): True
    2. _gguf_cuda_module(): llamacpp_gguf_cuda._prism
    3. linear callable: True
    4. embedding callable: True
    5. may_support_linear[IQ3_S]: True
    ...
"""
import inspect
import sys

# Add WanGP to the path (adjust to your installation)
sys.path.insert(0, r"C:\Program Files\Wan2gp")

import torch

print("torch:", torch.__version__, "| hip:", torch.version.hip)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0))
print("-" * 60)

from shared.qtypes import gguf

enabled = gguf._gguf_cuda_kernels_enabled()
print("1. _gguf_cuda_kernels_enabled():", enabled)

mod = gguf._gguf_cuda_module()
print("2. _gguf_cuda_module():", "None" if mod is None else mod.__name__)
if mod is not None:
    try:
        print("   version:", mod.__version__)
    except Exception as e:
        print("   version: <n/a>", e)
    print("3. linear callable:", callable(getattr(mod, "linear", None)))
    print("4. embedding callable:", callable(getattr(mod, "embedding", None)))
    for qt in ("IQ3_S", "IQ2_S", "Q4_K", "Q4_K_M", "Q5_K", "Q6_K", "Q8_0"):
        try:
            print(f"5. may_support_linear[{qt}]:", mod.may_support_linear_qtype_name(qt))
        except Exception as e:
            print(f"5. may_support_linear[{qt}]: <error> {e}")

# End-to-end: build a real IQ3_S GGUF weight on GPU and run linear()
print("-" * 60)
print("End-to-end linear() on GPU with IQ3_S weight:")
try:
    from shared.qtypes.gguf import GGUFLinear  # may have a different name
except Exception:
    GGUFLinear = None

# Find the linear layer class
cands = [n for n, o in vars(gguf).items()
         if inspect.isclass(o) and hasattr(o, "linear") and hasattr(o, "dequantize")]
print("linear layer classes found:", cands)
