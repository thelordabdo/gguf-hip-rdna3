# Build Guide: GGUF HIP Kernels for gfx1100 (RDNA3)

Complete step-by-step guide to building `llamacpp_gguf_cuda` GGUF quantization kernels from source for AMD RDNA3 GPUs on Windows.

## Prerequisites

| Component | Version | Notes |
|---|---|---|
| Windows | 10/11 | x64 |
| AMD GPU | RDNA3 (gfx1100) | RX 7900 XTX/XT, 7800 XT, 7700 XT |
| ROCm SDK | 7.x | Installed via `rocm-sdk` |
| Python | 3.12.x | Must be 3.12 for torch 2.13 compatibility |
| Visual Studio 2022 | Community/Pro | C++ workload + Windows SDK |
| Git | Latest | For cloning source |
| Disk space | ~15 GB | Build env + source + wheel |

## Step 1: Install Python 3.12

```powershell
winget install Python.Python.3.12
```

Verify:
```powershell
py -3.12 --version
# Expected: Python 3.12.10
```

## Step 2: Create Build Environment

> **CRITICAL:** The build path must contain **NO SPACES**. PyTorch's `cpp_extension` corrupts compiler flags (`cflags`) when the path contains spaces, causing build failures.

```powershell
# Create a no-space build directory
mkdir D:\WanGP

# Create virtual environment
py -3.12 -m venv D:\WanGP\build-env

# Activate
D:\WanGP\build-env\Scripts\activate
```

## Step 3: Install PyTorch with ROCm

```powershell
pip install torch --index-url https://stable.repo.amd.com/rocm/whl-next/ --extra-index-url https://pypi.org/simple/
```

Verify:
```python
import torch
print(torch.__version__)       # 2.13.0+rocm10.1.0
print(torch.version.hip)       # 7.16.26385
print(torch.cuda.is_available())  # True
print(torch.cuda.get_device_name(0))  # AMD Radeon RX 7900 XT
```

## Step 4: Clone GGUF Source

```powershell
git clone --depth 1 https://github.com/.../llamacpp_gguf_cuda.git D:\WanGP\gguf-src-1.0.30
cd D:\WanGP\gguf-src-1.0.30
```

## Step 5: Initialize ROCm SDK

```powershell
rocm-sdk init
```

This sets up the HIP compiler toolchain (`clang-cl`) and SDK paths required by `build_hip.ps1`.

## Step 6: Build for gfx1100

```powershell
.\build_hip.ps1 `
    -Python "D:\WanGP\build-env\Scripts\python.exe" `
    -Architectures "gfx1100" `
    -OutputDirectory "D:\WanGP\gguf-dist"
```

The build:
1. Detects the ROCm SDK root from the Python environment
2. Sets up `clang-cl` compiler environment (CC/CXX)
3. Sets `PYTORCH_ROCM_ARCH=gfx1100`
4. Runs `setup.py bdist_wheel`
5. Produces: `D:\WanGP\gguf-dist\llamacpp_gguf_cuda-1.0.30+torch213rocm101py312-cp312-cp312-win_amd64.whl`

Build time: ~10–20 minutes depending on CPU.

## Step 7: Validate

```powershell
python validate_hip.py
```

Expected output:
```
{
  "passed": true,
  "torch": "2.13.0+rocm10.1.0",
  "hip": "7.16.26385",
  "gpu": "AMD Radeon RX 7900 XT",
  "kernels": "1.0.30+torch213rocm101py312",
  "scope": "Native kernel numerical and graph tests; full Deepy checkpoints must also be tested."
}
```

## Step 8: Install in Target Environment

```powershell
# In your target environment (e.g., WanGP)
pip install D:\WanGP\gguf-dist\llamacpp_gguf_cuda-1.0.30+torch213rocm101py312-cp312-cp312-win_amd64.whl
```

Verify dispatch:
```python
import sys
sys.path.insert(0, r"C:\Program Files\Wan2gp")
from shared.qtypes import gguf

print(gguf._gguf_cuda_kernels_enabled())  # True
mod = gguf._gguf_cuda_module()
print(mod.__version__)                     # 1.0.30+torch213rocm101py312
print(mod.may_support_linear_qtype_name("IQ3_S"))  # True
```

## Troubleshooting

### Build fails with compiler errors
- Ensure the build path has **no spaces**
- Ensure `rocm-sdk init` was run in the build environment
- Ensure Visual Studio 2022 C++ workload is installed

### `torch.cuda.is_available()` returns False
- Ensure ROCm SDK is properly installed
- Check `rocm-smi` shows your GPU
- Reinstall torch with the ROCm index URL

### Wheel not found after build
- Check `D:\WanGP\gguf-dist\` for the `.whl` file
- The wheel name includes torch version + ROCm version + Python version

### Slow inference after install
- Ensure the wheel is installed in the **correct** environment
- Check `gguf._gguf_cuda_kernels_enabled()` returns `True`
- First run includes Triton JIT compilation (one-time, ~5 min)
