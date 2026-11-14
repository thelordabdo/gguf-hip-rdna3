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

Each of the 18 quantization types below was tested across **2 dtypes** (`torch.float16`, `torch.bfloat16`) and **7 batch sizes** (rows = 1, 2, 4, 8, 17, 64, 129): 18 × 2 × 7 = **252 linear configurations**, all within tolerance (`relative_rms` < 0.025):

| Quantization Type | Status |
|---|---|
| Q2_K, Q3_K, Q4_K, Q5_K, Q6_K | ✅ PASS |
| Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 | ✅ PASS |
| IQ1_S, IQ2_S, IQ2_XS, IQ2_XXS | ✅ PASS |
| IQ3_S, IQ3_XXS, IQ4_NL, IQ4_XS | ✅ PASS |

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

> **Note:** The first run pays a large one-time Triton JIT cost before decoding
> starts. Measured on this machine (RX 7900 XT, Ryzen 7 3700X):
> `l2norm` 124 s, `chunk_gated_delta_rule` 94 s, `q8_grouped_partials_kernel`
> 167 s, `q8_paged_prefill_kernel` 133 s — **~8.6 minutes for those four kernels
> alone**, plus further autotuning. Compilation results are cached, so subsequent
> runs start fast and decode at the steady-state rate above.

## Significance

This is the **first real AMD validation** of `llamacpp_gguf_cuda` HIP kernels
on RDNA3 hardware. The official [release notes](https://github.com/deepbeepmeep/kernels/releases/tag/gguf-v1.0.30)
state that the HIP wheels were *"checked without ROCm PyTorch or an AMD GPU"*
and their native extensions *"were not loaded for this release."* The official
wheels only target `gfx1200`/`gfx1201` (RDNA4) — there is no `gfx1100` wheel.

This build confirms that the full kernel pipeline (linear, attention, padding,
Prism) works correctly on `gfx1100` (RDNA3) with real ROCm PyTorch and a real
AMD GPU.
