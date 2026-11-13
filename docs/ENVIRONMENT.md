# Environment Details

Exact versions and configurations used for the validated build.

## Hardware

| Component | Specification |
|---|---|
| GPU | AMD Radeon RX 7900 XT (20 GB GDDR6, `gfx1100`) |
| CPU | AMD Ryzen 7 3700X (8 cores / 16 threads) |
| RAM | 80 GB DDR4 |
| OS | Windows 11 Pro x64 |

## Software Stack

| Package | Version | Source |
|---|---|---|
| Python | 3.12.10 | `winget install Python.Python.3.12` |
| PyTorch | 2.13.0+rocm10.1.0 | `https://stable.repo.amd.com/rocm/whl-next/` |
| Triton | 3.7.1 | Bundled with torch |
| ROCm / HIP | 7.16.26385 | `rocm-sdk` |
| llamacpp_gguf_cuda | 1.0.30+torch213rocm101py312 | Built from source |
| flash_attn | 2.8.4 | Pure Python + Triton (no pyd) |
| amd-aiter | 0.1.14rc1.dev177 | Triton ops only |
| sageattention | 1.0.6 | — |

## Build Environment

```
Path:        D:\WanGP\build-env
Python:      3.12.10
Base:        C:\Users\Abdo\AppData\Local\Programs\Python\Python312\python.exe
torch:       2.13.0+rocm10.1.0
```

## Target Environment

```
Path:        C:\Program Files\Wan2gp\wan2gp-env
Python:      3.12.10
torch:       2.13.0+rocm10.1.0
llamacpp_gguf_cuda: 1.0.30+torch213rocm101py312
```

## Key Environment Variables (at runtime)

```
FLASH_ATTENTION_TRITON_AMD_ENABLE=True
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
MIOPEN_FIND_MODE=FAST
MIOPEN_LOG_LEVEL=4
CC=clang-cl
CXX=clang-cl
```

## Build Path Constraint

> **The build path must not contain spaces.**
>
> PyTorch's `cpp_extension.load()` constructs compiler command lines by
> joining flags with spaces. When the build directory path itself contains
> spaces (e.g., `D:\Programs\Ai Programs\WanGP\`), the resulting `cflags`
> string is mis-parsed, causing silent build corruption.
>
> **Solution:** Use a no-space path like `D:\WanGP\` for all C++/HIP builds.

## Wheel Compatibility

The built wheel `llamacpp_gguf_cuda-1.0.30+torch213rocm101py312-cp312-cp312-win_amd64.whl` is compatible with:

- Python 3.12 (cp312)
- torch 2.13.0+rocm10.1.0
- Windows x64
- gfx1100 (RDNA3)

It is **not** compatible with:
- Python 3.13 (cp313)
- Different torch/ROCm versions
- RDNA4 (gfx1200/gfx1201) — use official wheels for those
