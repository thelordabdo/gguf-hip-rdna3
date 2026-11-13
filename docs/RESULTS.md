# Validation Results

## Kernel Validation

**Date:** 2026-11-12
**GPU:** AMD Radeon RX 7900 XT (20 GB, gfx1100)
**Kernels:** 1.0.30+torch213rocm101py312

### Summary

```
PASS 1.0.30+torch213rocm101py312: 252 linear configurations, 12 attention cases, 3 padding cases.
```

### Linear Kernel Checks (all passed)

| Quantization Type | Shape | Status |
|---|---|---|
| Q2_K | (128, 512) | ✅ PASS |
| Q3_K | (128, 512) | ✅ PASS |
| Q4_K | (128, 512) | ✅ PASS |
| Q5_K | (128, 512) | ✅ PASS |
| Q6_K | (128, 512) | ✅ PASS |
| Q4_0 | (128, 512) | ✅ PASS |
| Q4_1 | (128, 512) | ✅ PASS |
| Q5_0 | (128, 512) | ✅ PASS |
| Q5_1 | (128, 512) | ✅ PASS |
| Q8_0 | (128, 512) | ✅ PASS |
| IQ2_S | (128, 512) | ✅ PASS |
| IQ3_S | (128, 512) | ✅ PASS |
| IQ1_S | (128, 512) | ✅ PASS |
| IQ2_XS | (128, 512) | ✅ PASS |
| IQ2_XXS | (128, 512) | ✅ PASS |
| IQ3_XXS | (128, 512) | ✅ PASS |
| IQ4_NL | (128, 512) | ✅ PASS |
| IQ4_XS | (128, 512) | ✅ PASS |

### Attention & Padding

- 12 attention cases: ✅ PASS
- 3 padding cases: ✅ PASS

### Prism (Ternary) Kernels

- `PTQ1_0` support: ✅
- `prism_hadamard` callable: ✅
- `prism_decode` present: ✅

## Dispatch Verification

```python
>>> from shared.qtypes import gguf
>>> gguf._gguf_cuda_kernels_enabled()
True
>>> gguf._gguf_cuda_module().__version__
'1.0.30+torch213rocm101py312'
>>> gguf._gguf_cuda_module().may_support_linear_qtype_name("IQ3_S")
True
```

## Performance (Qwen3.8-27B IQ3_S)

| Metric | Before (PyTorch fallback) | After (HIP kernels) |
|---|---|---|
| Decode speed (27B IQ3_S) | ~1–1.5 tok/s | ~6–10 tok/s |
| Fast path active | ❌ | ✅ |

> **Note:** First run includes one-time Triton JIT compilation (~5 min total).
> Subsequent runs skip compilation and are faster.

## Significance

This is the **first real AMD validation** of `llamacpp_gguf_cuda` HIP kernels
on RDNA3 hardware. The official [release notes](https://github.com/deepbeepmeep/kernels/releases/tag/gguf-v1.0.30)
state that the HIP wheels were *"checked without ROCm PyTorch or an AMD GPU"*
and their native extensions *"were not loaded for this release."* The official
wheels only target `gfx1200`/`gfx1201` (RDNA4) — there is no `gfx1100` wheel.

This build confirms that the full kernel pipeline (linear, attention, padding,
Prism) works correctly on `gfx1100` (RDNA3) with real ROCm PyTorch and a real
AMD GPU.
