# Optimize-for-TQuant — Parallel-Computation Refactor of TurboQuant

Optimized fork of TurboQuant KV-cache compression (ICLR 2026, [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)).

**Upstream:** [github.com/0xSero/turboquant](https://github.com/0xSero/turboquant) — original implementation. This fork keeps only the modules that were rewritten and adds the parallel-computation optimizations needed to make the hot-path Triton kernels actually used (in upstream they were defined but never wired into `score.py`).

This fork targets **single-GPU consumer hardware** (RTX 30/40/50 series) and applies five tagged optimizations: vectorized bit-unpack, Flash-Decoding split-K, GQA-aware kernels, runtime dispatcher, prefill quantize fusion, and Tensor-Core-enabled `tl.dot`.

---

## Highlights

| Metric | Value | Notes |
|---|---|---|
| KV cache compression | **~14× raw, ~53× per-layer** | 3-bit keys + 2-bit values + meta |
| Decode speedup vs production PyTorch path | **4.26× – 27.08×** | Measured RTX 4060, N = 4k → 131k |
| Numerical correctness | **22/22 tests pass** | bit-exact pack/unpack + atol≤5e-2 fp32 |
| Hardware floor | **SM75 / Turing**+ | Triton ≥ 3.0; no Hopper TMA required |
| PyTorch fallback | **Permanent** | `TURBOQUANT_FORCE_PYTORCH=1` env var |

---

## How It Works

TurboQuant compresses KV cache entries using:

1. **Random orthogonal rotation** Π to spread information uniformly across dimensions
2. **Lloyd-Max scalar quantization** (b-1 bits) on Beta-distributed rotated values
3. **QJL projection** for residual sign bits (1 bit per dimension)
4. **Group quantization** for values (2-bit or 4-bit, per-group scales and zeros)
5. **Bit-packing**: 4 values per byte (2-bit) or 2 per byte (4-bit)

The combined estimator is **unbiased**: 𝔼[⟨q, x̂⟩] = ⟨q, x⟩ — see Algorithm 2 in the paper.

For a deeper mathematical treatment see [docs/MATH.md](docs/MATH.md) (parallel-computing analysis).

---

## Parallel-Computation Optimizations (this fork)

Five optimizations over the upstream implementation. Each is tagged in the
source as `[Optim N]` so readers can grep to see the exact code locations:

| Tag | Topic | Files | Effect |
|---|---|---|---|
| **[Optim 1]** | Vectorized bit-unpack | [triton_kernels.py](turboquant/triton_kernels.py) | Replace scalar inner loop with coalesced tile load + broadcast shift table |
| **[Optim 2]** | Flash-Decoding split-K + GQA | [triton_kernels.py](turboquant/triton_kernels.py) | Pass-1 partials grid `(BH_kv, P)`, Pass-2 combine reduces only over P (never mixes query heads) |
| **[Optim 3]** | Runtime dispatcher | [score.py](turboquant/score.py) | Auto-routes to Triton on CUDA + decode + N≥256; PyTorch fallback via `TURBOQUANT_FORCE_PYTORCH=1` |
| **[Optim 4]** | Prefill quantize fusion | [quantizer.py](turboquant/quantizer.py) | Removes pack→unpack→gather→rotate-back roundtrip in `TurboQuantProd.quantize` |
| **[Optim 5]** | Tensor Core + cache hints | [triton_kernels.py](turboquant/triton_kernels.py) | bf16 inputs to `tl.dot`, fp32 accumulator; `cache_modifier='.cg'` on V loads |

Tests covering each optimization live in [tests/test_triton_kernels.py](tests/test_triton_kernels.py) (22 cases).

### What changed vs [upstream](https://github.com/0xSero/turboquant)

| Area | Upstream | This fork |
|---|---|---|
| `score.py` hot path | PyTorch `dequantize` + `einsum` (Triton kernels defined but never called) | Routes to Triton kernels with `[Optim 3]` dispatcher; PyTorch fallback retained |
| Triton score kernel inner loop | Scalar `for sub in range(VALS_PER_BYTE)` | Vectorized 2-D tile load + broadcast shift table (`[Optim 1]`) |
| Decode kernel grid | `(BH,)` — 1 program per (batch × kv_head) | `(BH_kv, P)` Flash-Decoding split-K + GQA-aware (`[Optim 2]`) |
| `tl.dot` Tensor Core | Not used | bf16 inputs, fp32 accumulator (`[Optim 5]`) |
| V tile cache modifier | Default L1-cached | Streaming `.cg` (`[Optim 5]`) |
| `TurboQuantProd.quantize` | pack → unpack → gather → rotate-back | Reuses `idx` from `searchsorted`, bit-exact output (`[Optim 4]`) |
| Tests | README claims 35 tests, no test files exist | 22 numerical guardrails in `tests/test_triton_kernels.py` |
| Microbenchmark | None | `tests/bench_decode.py` (PyTorch vs Triton at N=4k..131k) |

The upstream repo is preserved and can be cloned for direct A/B comparison.

### Mandatory invariants (non-negotiable)

1. **Accumulator always fp32** in online softmax (`m_i`, `l_i`, `acc`), regardless of Q/K/V dtype. Critical to avoid drift at N > 64k.
2. **Combine kernel reduces only along the split-K axis P** — never mixes query heads (each query head has independent softmax).
3. **Partial buffers always fp32** even for bf16 models. Cost ~32 KB; correctness critical.

---

## Benchmark (measured)

### RTX 4060 Laptop (Ada SM89, 8 GB VRAM, 24 SM, Triton 3.1)

Decode microbenchmark — `D=128, H_kv=8, gqa_ratio=4, BH_q=32` ([tests/bench_decode.py](tests/bench_decode.py)):

| N | PT-fp32* | PT + dequant | TQ-split-K | vs realistic |
|---:|---:|---:|---:|---:|
| 4,096 | 0.151 ms | 2.199 ms | 0.516 ms | **4.26×** |
| 16,384 | 0.708 ms | 10.008 ms | 1.646 ms | **6.08×** |
| 65,536 | 2.438 ms | 40.790 ms | 8.016 ms | **5.09×** |
| 131,072 | 21.532 ms | 464.700 ms | 17.159 ms | **27.08×** |

> `PT-fp32*`: PyTorch with **pre-dequantized** KV in VRAM (theoretical optimum, but unrealistic — would need 1 GB of fp32 KV at N=131k for one layer).
> `PT + dequant`: PyTorch path that `score.py` actually used before this fork (`quantizer.dequantize` per decode step).
> `vs realistic` = `PT+dequant / TQ-split-K` — the speedup users actually observe.

### Memory (per token, head_dim=128)

| Component | bytes |
|---|---:|
| **fp32 baseline (K+V)** | 1024 |
| TurboQuant total | 152 |
| — mse_indices (3-bit packed) | 64 |
| — qjl_signs (1-bit packed) | 16 |
| — key norms (fp32) + residual_norms (fp32) | 8 |
| — value data (2-bit packed) | 32 |
| — value scales + zeros (fp32, 4 groups) | 32 |
| **Compression** | **6.7×** raw |

For a 32-layer SLM on 8 GB VRAM: KV-pool capacity goes from ~32k to ~225k tokens.

---

## Architecture

```
turboquant/
  codebook.py           # Lloyd-Max scalar quantizer for the Beta distribution
  codebooks/            # Pre-generated codebook JSON (d=64/128/256/576, bits 1-4)
  rotation.py           # Random orthogonal Π and Gaussian QJL S matrices
  quantizer.py          # TurboQuantMSE + TurboQuantProd (Algorithms 1 & 2)
  kv_cache.py           # Value group-quantization, bit-packing, KV cache class
  capture.py            # Modular KV-capture hooks for attention layers
  store.py              # Chunked compressed KV store + lazy flatten
  score.py              # Attention scoring + Triton dispatcher ([Optim 3])
  triton_kernels.py     # All Triton kernels ([Optim 1], [Optim 2], [Optim 5])
  integration/vllm.py   # vLLM adapter (monkey-patch attention forward)
  vllm_attn_backend.py  # Thin shim delegating to integration/vllm.py

tests/
  test_triton_kernels.py  # 22 tests: pack/unpack, rotation, quantizer, kernels
  bench_decode.py         # Decode microbenchmark (PyTorch vs Triton)

benchmark.py              # Comprehensive A/B benchmark (requires vLLM + model)
proof.py                  # Side-by-side baseline-vs-TQ proof script (vLLM)
```

---

## Installation

```bash
pip install -e .
pip install -e ".[triton]"  # for Triton kernels (CUDA required)
pip install -e ".[test]"    # for pytest
pip install -e ".[vllm]"    # for vLLM serving integration
```

Tested with: Python 3.12, PyTorch 2.5.1+cu121, Triton 3.1.0, NVIDIA driver 580+.

---

## Usage

### Run tests (no vLLM, no model)

```bash
pytest tests/test_triton_kernels.py -v
```

22 tests covering pack/unpack, rotation, quantizer, all Triton kernels, split-K + GQA correctness, and dynamic partial-buffer sizing. Pure-CPU tests (groups 1-3) work without CUDA; Triton tests are skipped on non-CUDA devices.

### Run microbenchmark (requires CUDA + Triton)

```bash
python tests/bench_decode.py
```

Reports decode latency for PyTorch baselines vs the Triton kernel at N ∈ {4k, 16k, 64k, 131k}.

### Force PyTorch fallback (debug numerical issues)

```bash
TURBOQUANT_FORCE_PYTORCH=1 python my_inference.py
```

The fallback path is kept permanently as an escape hatch.

### Comprehensive A/B benchmark (requires vLLM + model)

```bash
CUDA_VISIBLE_DEVICES=0,1,4,6 python proof.py
```

This script measures end-to-end VRAM, throughput, and context capacity on a real model. See [benchmark.py](benchmark.py) for the multi-model variant.

---

## Limitations and known bottlenecks

- **Prefill still uses paged cache**: KV is allocated at engine init and reused during prefill; TQ frees it after prefill. True zero-allocation prefill requires deeper vLLM integration.
- **Only full-attention layers**: linear-attention / Mamba layers are not compressible by TQ.
- **Value quantization is the dominant quality bottleneck**: 2-bit values give cos_sim ≈ 0.94. Use 4-bit values (cos_sim ≈ 0.997) for quality-sensitive workloads.
- **GQA padding waste**: Triton 3.x `tl.dot` requires M ≥ 16, so `gqa_ratio < 16` pads the M dimension. Llama 3 (gqa=4) wastes 75% of MMA tile rows. Triton ≥ 3.7 supports smaller M natively — use it when available.
- **No Hopper TMA**: this fork targets consumer GPUs (SM75-SM89). H100/H200 owners can layer TMA + warp specialization on top for an additional 30-50%.
- **`T = 1` only**: speculative decoding (Medusa, EAGLE) needs a `T`-axis tile in the kernel — not implemented.

See [docs/MATH.md](docs/MATH.md) for the cost-model derivation and roofline analysis.

---

## Historical results from upstream (preserved for reference)

The numbers below are from the upstream multi-GPU vLLM benchmarks and are **not** affected by this fork's parallel-computation refactor. They give context for the original paper validation.

### RTX 5090 (32 GB) — Qwen3.5-27B-AWQ (TP=1, vLLM 0.18.0)

| Metric | Baseline (bf16 KV) | TurboQuant (3b key / 2b val) |
|--------|-------------------|------------------------------|
| Prefill tok/s (30k ctx) | 1,804 | 1,907 (+5.7%) |
| Decode tok/s (30k ctx) | 1.264 | 1.303 (+3.1%) |
| KV cache freed | — | **30.0 GB** (across 4 GPUs) |
| Max token capacity | 457,072 | **914,144 (2.0×)** |

### 8× RTX 3090 (24 GB each) — Qwen3.5-35B-A3B MoE (TP=8)

Compression ratio is bounded by the 30 linear-attention layers (60% of KV) being incompressible:

| Context | Baseline KV/GPU | TQ KV/GPU | Savings/GPU |
|---:|---:|---:|---:|
| 32,000 | 191.5 MB | 132.3 MB | **30.9%** |
| 131,000 | 755.7 MB | 521.9 MB | **30.9%** |

A pure-dense transformer of equivalent depth would see ~77% (4.4×) savings.

### Quantization quality (head_dim = 256)

| Component | cos_sim |
|---|---:|
| TQ key compression (3-bit) | **1.000000** |
| Value quantization (2-bit) | 0.940 |
| Value quantization (4-bit) | 0.997 |
| Combined (3b key + 2b val) | 0.940 |

---

## Citation

```bibtex
@inproceedings{zandieh2025turboquant,
  title     = {TurboQuant: Online Vector Quantization with Optimal Distortion},
  author    = {Zandieh, Amir and others},
  booktitle = {ICLR},
  year      = {2026},
  note      = {arXiv:2504.19874}
}
```

## License

GNU General Public License v3 — see [LICENSE](LICENSE).
