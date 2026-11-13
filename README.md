# GGUF HIP Kernels for AMD RDNA3 (gfx1100)

Building `llamacpp_gguf_cuda` GGUF quantization kernels from source for **AMD RDNA3** GPUs — enabling fast llama.cpp-style packed quantized inference (IQ3_S, Q4_K, Q5_K, Q8_0, PTQ1_0, and more) on cards that have **no official prebuilt wheel**.

> **Why this exists:** The official [`llamacpp_gguf_cuda` wheels](https://github.com/deepbeepmeep/kernels/releases/tag/gguf-v1.0.30) only target **RDNA4** (`gfx1200`/`gfx1201`). RDNA3 users (RX 7900 XTX, RX 7900 XT, RX 7800 XT, RX 7700 XT) have no prebuilt option — they fall back to slow PyTorch dense weight materialization. The official release notes confirm the HIP wheels were *"checked without ROCm PyTorch or an AMD GPU"* — **this repository documents the first real AMD validation** and the complete process to build the HIP kernels from source for `gfx1100`.

## Use Case: WanGP

This build was created for **[WanGP](https://github.com/deepbeepmeep/WanGP)** — a local AI application that uses `llamacpp_gguf_cuda` for GGUF model inference. Without the HIP kernels, WanGP on RDNA3 falls back to PyTorch dense matmul, which is 5–10× slower for quantized models. With the kernels built in this repo, WanGP achieves proper packed quantized inference on RDNA3.

The `test_dispatch.py` script in this repo verifies that WanGP's GGUF dispatch predicates correctly route to the HIP fast path.

## What This Enables

| Capability | Without (PyTorch fallback) | With (HIP kernels) |
|---|---|---|
| IQ3_S 27B decode | ~1–1.5 tok/s | ~6–10 tok/s |
| Q4_K / Q5_K / Q8_0 | Dense matmul | Packed MMVQ/MMQ kernels |
| PTQ1_0 (ternary) | Not supported | Prism Hadamard kernels |
| Attention | PyTorch SDPA | Paged attention (Triton) |

## Hardware Validated

- **GPU:** AMD Radeon RX 7900 XT (20 GB, `gfx1100`)
- **CPU:** AMD Ryzen 7 3700X
- **RAM:** 80 GB
- **OS:** Windows 11
- **ROCm:** 7.x (HIP 7.16.26385)
- **PyTorch:** 2.13.0+rocm10.1.0
- **Python:** 3.12.10

## Download Prebuilt Wheel

A prebuilt wheel for **gfx1100** is available as a [release asset](https://github.com/thelordabdo/gguf-hip-rdna3/releases/tag/gguf-v1.0.30-gfx1100):

```
llamacpp_gguf_cuda-1.0.30+torch213rocm101py312-cp312-cp312-win_amd64.whl  (5.25 MB)
```

Requires: Python 3.12, torch 2.13.0+rocm10.1.0, Windows x64, RDNA3 GPU.

```powershell
pip install llamacpp_gguf_cuda-1.0.30+torch213rocm101py312-cp312-cp312-win_amd64.whl
```

Validation logs and JSON results are also included in the [release](https://github.com/thelordabdo/gguf-hip-rdna3/releases/tag/gguf-v1.0.30-gfx1100).

## Quick Start (Build from Source)

```bash
# 1. Install Python 3.12 (winget)
winget install Python.Python.3.12

# 2. Create build environment (NO SPACES in path!)
py -3.12 -m venv D:\WanGP\build-env
D:\WanGP\build-env\Scripts\activate

# 3. Install ROCm PyTorch
pip install torch --index-url https://stable.repo.amd.com/rocm/whl-next/ --extra-index-url https://pypi.org/simple/

# 4. Clone the GGUF source (v1.0.30)
git clone --depth 1 --branch gguf-v1.0.30 https://github.com/deepbeepmeep/kernels.git D:\WanGP\gguf-src-1.0.30
cd D:\WanGP\gguf-src-1.0.30

# 5. Initialize ROCm SDK
rocm-sdk init

# 6. Build for gfx1100
.\build_hip.ps1 -Python "D:\WanGP\build-env\Scripts\python.exe" -Architectures "gfx1100" -OutputDirectory "D:\WanGP\gguf-dist"

# 7. Validate (run from the source tests/ directory)
python tests\validate_hip.py --output D:\WanGP\validation

# 8. Install in target environment
pip install D:\WanGP\gguf-dist\llamacpp_gguf_cuda-1.0.30+torch213rocm101py312-cp312-cp312-win_amd64.whl
```

## Repository Structure

```
gguf-hip-rdna3/
├── README.md                  ← This file
├── LICENSE
├── .gitignore
├── build/
│   ├── validate_hip.py        ← Kernel validation script (from llamacpp_gguf_cuda source)
│   └── test_dispatch.py       ← WanGP dispatch predicate test
├── docs/
│   ├── BUILD_GUIDE.md         ← Complete step-by-step build guide
│   ├── ENVIRONMENT.md         ← Exact versions and environment
│   └── RESULTS.md             ← Validation results and performance
```

## Key Constraints

1. **No spaces in build path.** PyTorch's `cpp_extension` corrupts compiler flags when the build directory path contains spaces. Use `D:\WanGP\` not `D:\Wan GP\`.
2. **ROCm SDK must be initialized** in the build environment before running `build_hip.ps1`.
3. **`gfx1100`** is the correct architecture flag for all RDNA3 GPUs.
4. **The wheel is environment-internal.** It survives `git pull` and `requirements.txt` updates. It is only lost if the Python environment is recreated from scratch.

## Validation Results

All **252 linear configurations**, **12 attention cases**, and **3 padding cases** passed:

```
PASS 1.0.30+torch213rocm101py312: 252 linear configurations, 12 attention cases, 3 padding cases.
```

Supported quantization types confirmed:
`Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `IQ2_S`, `IQ3_S`, `IQ1_S`, `IQ2_XS`, `IQ2_XXS`, `IQ3_XXS`, `IQ4_NL`, `IQ4_XS`

Prism (ternary) kernels: `PTQ1_0` supported with `prism_hadamard` callable.

## License

MIT — see [LICENSE](LICENSE).
