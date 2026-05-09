"""
Microbenchmark for TurboQuant decode attention.

Compares three paths to isolate the effect of each optimization:

  1. PyTorch reference (dequantize + matmul + softmax + matmul)
     Matches the upstream 0xSero/turboquant decode path.

  2. Triton turboquant_fused_decode (single program / BH)
     Applies [Optim 1] vectorized unpack only — no split-K.

  3. Triton turboquant_fused_decode_split_k
     Applies [Optim 1] + [Optim 2] (split-K + GQA) + [Optim 5] (Tensor Core
     via tl.dot + '.cg' cache modifier). Production hot path.

Run on a CUDA box with Triton installed:
    python tests/bench_decode.py

Reports latency and speedup vs the PyTorch baselines across N = 4k..131k.
"""

from __future__ import annotations

import math
import sys
import time
import torch

try:
    import triton  # noqa: F401
except ImportError:
    print("Triton not installed — bench requires GPU + triton")
    sys.exit(1)

if not torch.cuda.is_available():
    print("CUDA not available — bench requires GPU")
    sys.exit(1)

from turboquant.quantizer import TurboQuantProd
from turboquant.kv_cache import quantize_values, dequantize_values
from turboquant.triton_kernels import (
    turboquant_fused_decode,
    turboquant_fused_decode_split_k,
)


def bench_pytorch(query, keys_dq, values_dq, gqa_ratio, sm_scale, iters=20):
    """Reference: full precision attention with already-dequantized KV.
    Note: this benchmark gives PyTorch the unfair advantage of NOT counting
    the dequantize cost (in real TQ flow, dequant is done in the kernel).
    """
    H_kv, N, D = keys_dq.shape
    BH_q = H_kv * gqa_ratio
    q = query.float().view(H_kv, gqa_ratio, D)

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        scores = torch.einsum("hgd,hnd->hgn", q, keys_dq.float()) * sm_scale
        weights = torch.softmax(scores, dim=-1)
        out = torch.einsum("hgn,hnd->hgd", weights, values_dq.float())
        out = out.reshape(BH_q, D)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000, out  # ms, output


def bench_pytorch_with_dequant(query, prod_q, val_q, quantizer, gqa_ratio, sm_scale,
                                iters=10):
    """Realistic baseline: dequantize in the loop (matches what score.py PyTorch path does)."""
    from turboquant.kv_cache import dequantize_values
    H_kv = prod_q.norms.shape[0]
    D = query.shape[-1]
    BH_q = H_kv * gqa_ratio

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        keys_dq = quantizer.dequantize(prod_q)
        values_dq = dequantize_values(val_q, group_size=32)
        q = query.float().view(H_kv, gqa_ratio, D)
        scores = torch.einsum("hgd,hnd->hgn", q, keys_dq.float()) * sm_scale
        weights = torch.softmax(scores, dim=-1)
        out = torch.einsum("hgn,hnd->hgd", weights, values_dq.float())
        out = out.reshape(BH_q, D)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000, out


def bench_triton_v1(query, prod_q, val_q, quantizer, sm_scale, H_kv, iters=20):
    """Single-program-per-BH kernel — [Optim 1] vector unpack only."""
    q_per_kv = query[:H_kv].float()  # (H_kv, D), drop GQA — bench non-GQA path
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        out = turboquant_fused_decode(
            q_per_kv, prod_q, val_q,
            quantizer.mse_quantizer.Pi, quantizer.S,
            quantizer.mse_quantizer.centroids,
            prod_q.mse_bits, quantizer.qjl_scale, sm_scale,
            group_size=32,
        )
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000, out


def bench_triton_split_k(query, prod_q, val_q, quantizer, sm_scale, H_kv, gqa_ratio, iters=20):
    """Split-K + GQA fused kernel — [Optim 1] + [Optim 2] + [Optim 5]."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        out = turboquant_fused_decode_split_k(
            query, prod_q, val_q,
            quantizer.mse_quantizer.Pi, quantizer.S,
            quantizer.mse_quantizer.centroids,
            prod_q.mse_bits, quantizer.qjl_scale, sm_scale,
            num_kv_heads=H_kv, gqa_ratio=gqa_ratio,
            group_size=32,
        )
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000, out


def main():
    torch.manual_seed(0)
    device = torch.device("cuda")
    D = 128
    H_kv = 8
    gqa_ratio = 4
    BH_q = H_kv * gqa_ratio
    sm_scale = 1.0 / math.sqrt(D)

    quantizer = TurboQuantProd(dim=D, bits=3, device=device, dtype=torch.float32)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"D={D}, H_kv={H_kv}, gqa_ratio={gqa_ratio}, BH_q={BH_q}")
    print()
    print(f"{'N':>8} | {'PT-fp32*':>10} | {'PT+dequant':>11} | {'TQ-split-K':>11} | "
          f"{'sp vs PT*':>10} | {'sp vs realistic':>16}")
    print("-" * 90)
    print("(* = pre-dequantized KV; not realistic — does not count dequant cost)")
    print()

    for N in [4096, 16384, 65536, 131072]:
        keys = torch.randn(H_kv, N, D, device=device)
        values = torch.randn(H_kv, N, D, device=device)
        query = torch.randn(BH_q, D, device=device)

        prod_q = quantizer.quantize(keys)
        val_q = quantize_values(values, bits=2, group_size=32)
        keys_dq = quantizer.dequantize(prod_q)
        values_dq = dequantize_values(val_q, group_size=32)

        # Warm up
        for _ in range(3):
            _ = bench_pytorch(query, keys_dq, values_dq, gqa_ratio, sm_scale, iters=1)
            _ = bench_pytorch_with_dequant(query, prod_q, val_q, quantizer,
                                            gqa_ratio, sm_scale, iters=1)
            _ = bench_triton_split_k(query, prod_q, val_q, quantizer, sm_scale,
                                     H_kv, gqa_ratio, iters=1)

        ms_pt, _ = bench_pytorch(query, keys_dq, values_dq, gqa_ratio, sm_scale)
        ms_pt_dq, _ = bench_pytorch_with_dequant(query, prod_q, val_q, quantizer,
                                                  gqa_ratio, sm_scale)
        ms_sk, _ = bench_triton_split_k(query, prod_q, val_q, quantizer, sm_scale,
                                        H_kv, gqa_ratio)

        sp_pt = ms_pt / ms_sk
        sp_realistic = ms_pt_dq / ms_sk
        print(f"{N:>8} | {ms_pt:>9.3f}ms | {ms_pt_dq:>10.3f}ms | {ms_sk:>10.3f}ms | "
              f"{sp_pt:>9.2f}x | {sp_realistic:>15.2f}x")


if __name__ == "__main__":
    main()
