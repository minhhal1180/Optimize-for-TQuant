"""
TurboQuant score module — attention computation over compressed + exact segments.

Handles the read path:
  - Attention over compressed historical KV (Triton fast path or PyTorch fallback)
  - Attention over exact recent buffer (standard matmul / SDPA)
  - Hybrid merge of both segments via online-softmax LSE

[Optim 3] Runtime dispatcher
  Routes to Triton when: CUDA tensor + decode (T=1) + N >= MIN_TRITON_HISTORY.
  Env var TURBOQUANT_FORCE_PYTORCH=1 forces the PyTorch fallback (kept
  permanently as an escape hatch for numerical debugging or no-Triton envs).

Hybrid (compressed + recent):
  - TQ kernel called with return_aux=True → (acc_tq, m_tq, l_tq), un-normalized.
  - Recent computed in PyTorch → (acc_r, m_r, l_r), un-normalized.
  - Final output via online-softmax LSE merge (see _attend_hybrid_triton).
"""

from __future__ import annotations

import math
import os
import logging
import torch
import torch.nn.functional as F

from turboquant.store import FlatCache, CompressedKVStore
from turboquant.kv_cache import dequantize_values
from turboquant.quantizer import TurboQuantProd

logger = logging.getLogger("turboquant.score")

MIN_HISTORY_FOR_TQ = 16
MIN_TRITON_HISTORY = 256  # below this, kernel launch overhead dominates

# Try to import Triton kernels — module-level so the check is cheap per call.
try:
    from turboquant.triton_kernels import turboquant_fused_decode_split_k
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False
    turboquant_fused_decode_split_k = None  # type: ignore


def _force_pytorch() -> bool:
    """Read env var fresh each call so tests can flip it via monkeypatch."""
    return os.environ.get("TURBOQUANT_FORCE_PYTORCH", "0") == "1"


def _can_use_triton(query: torch.Tensor, n_history: int) -> bool:
    return (
        _TRITON_AVAILABLE
        and not _force_pytorch()
        and query.is_cuda
        and query.shape[0] == 1
        and n_history >= MIN_TRITON_HISTORY
    )


def compute_hybrid_attention(
    query: torch.Tensor,
    store: CompressedKVStore,
    recent_k: "torch.Tensor | None",
    recent_v: "torch.Tensor | None",
    num_query_heads: int,
    scale: "float | None" = None,
) -> torch.Tensor:
    """Compute attention output combining compressed history and exact recent buffer.

    Args:
        query: (num_tokens, num_query_heads, head_dim) — typically num_tokens=1 for decode
        store: compressed KV store with historical tokens
        recent_k: (recent_len, num_kv_heads, head_dim) or None
        recent_v: (recent_len, num_kv_heads, head_dim) or None
        num_query_heads: total query heads (for GQA expansion)
        scale: attention scale factor (default: 1/sqrt(head_dim))

    Returns:
        output: (num_tokens, num_query_heads, head_dim)
    """
    head_dim = store.head_dim
    num_kv_heads = store.num_kv_heads
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    flat = store.get_flat_cache()
    has_history = flat is not None and flat.num_tokens >= MIN_HISTORY_FOR_TQ
    has_recent = recent_k is not None and recent_k.shape[0] > 0

    if not has_history and not has_recent:
        return torch.zeros(
            query.shape[0], num_query_heads, head_dim,
            device=query.device, dtype=query.dtype,
        )

    gqa_ratio = num_query_heads // num_kv_heads

    if has_history and not has_recent:
        return _attend_compressed_only(
            query, flat, store.quantizer, gqa_ratio, num_kv_heads, scale,
        )

    if not has_history and has_recent:
        return _attend_exact_only(
            query, recent_k, recent_v, gqa_ratio, num_kv_heads, scale,
        )

    # Both segments present — merge via log-sum-exp trick
    return _attend_hybrid(
        query, flat, store.quantizer, recent_k, recent_v,
        gqa_ratio, num_kv_heads, head_dim, scale,
    )


# ─── Compressed-only attention ───────────────────────────────────────────


def _attend_compressed_only(
    query: torch.Tensor,
    flat: FlatCache,
    quantizer: TurboQuantProd,
    gqa_ratio: int,
    num_kv_heads: int,
    scale: float,
) -> torch.Tensor:
    """Attention over compressed history only.

    Triton fast path takes (T=1, CUDA, N>=MIN_TRITON_HISTORY); else PyTorch.
    """
    if _can_use_triton(query, flat.num_tokens):
        return _attend_compressed_only_triton(
            query, flat, quantizer, gqa_ratio, num_kv_heads, scale,
        )
    return _attend_compressed_only_pytorch(
        query, flat, quantizer, gqa_ratio, num_kv_heads, scale,
    )


def _attend_compressed_only_pytorch(
    query: torch.Tensor,
    flat: FlatCache,
    quantizer: TurboQuantProd,
    gqa_ratio: int,
    num_kv_heads: int,
    scale: float,
) -> torch.Tensor:
    k_dequant = quantizer.dequantize(flat.prod_q)              # (H_kv, N, D)
    v_dequant = dequantize_values(flat.value_q, 32)
    return _matmul_attend(query, k_dequant, v_dequant, gqa_ratio, num_kv_heads, scale)


def _attend_compressed_only_triton(
    query: torch.Tensor,
    flat: FlatCache,
    quantizer: TurboQuantProd,
    gqa_ratio: int,
    num_kv_heads: int,
    scale: float,
) -> torch.Tensor:
    # query: (1, num_q_heads, D) → (num_q_heads, D) for kernel
    T, Q_heads, D = query.shape
    q_2d = query.squeeze(0)
    out = turboquant_fused_decode_split_k(
        q_2d,
        flat.prod_q,
        flat.value_q,
        quantizer.mse_quantizer.Pi,
        quantizer.S,
        quantizer.mse_quantizer.centroids,
        flat.prod_q.mse_bits,
        quantizer.qjl_scale,
        scale,
        num_kv_heads=num_kv_heads,
        gqa_ratio=gqa_ratio,
        group_size=32,
    )
    return out.view(T, Q_heads, D).to(query.dtype)


# ─── Exact-recent-only attention ─────────────────────────────────────────


def _attend_exact_only(
    query: torch.Tensor,
    recent_k: torch.Tensor,
    recent_v: torch.Tensor,
    gqa_ratio: int,
    num_kv_heads: int,
    scale: float,
) -> torch.Tensor:
    """Attention over exact recent buffer only."""
    return _matmul_attend(
        query, recent_k.transpose(0, 1), recent_v.transpose(0, 1),
        gqa_ratio, num_kv_heads, scale,
    )


# ─── Hybrid (compressed + recent) attention ──────────────────────────────


def _attend_hybrid(
    query: torch.Tensor,
    flat: FlatCache,
    quantizer: TurboQuantProd,
    recent_k: torch.Tensor,
    recent_v: torch.Tensor,
    gqa_ratio: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
) -> torch.Tensor:
    """Merge compressed history + exact recent.

    Triton fast path: TQ kernel returns (acc_tq, m_tq, l_tq) un-normalized;
    we compute recent attention in PyTorch and merge via online-softmax.
    Else: PyTorch concat dequant + matmul (existing path).
    """
    if _can_use_triton(query, flat.num_tokens):
        return _attend_hybrid_triton(
            query, flat, quantizer, recent_k, recent_v,
            gqa_ratio, num_kv_heads, head_dim, scale,
        )
    return _attend_hybrid_pytorch(
        query, flat, quantizer, recent_k, recent_v,
        gqa_ratio, num_kv_heads, head_dim, scale,
    )


def _attend_hybrid_pytorch(
    query: torch.Tensor,
    flat: FlatCache,
    quantizer: TurboQuantProd,
    recent_k: torch.Tensor,
    recent_v: torch.Tensor,
    gqa_ratio: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
) -> torch.Tensor:
    k_hist = quantizer.dequantize(flat.prod_q)  # (H_kv, N_hist, D)
    v_hist = dequantize_values(flat.value_q, 32)

    k_recent = recent_k.transpose(0, 1)   # (H_kv, N_recent, D)
    v_recent = recent_v.transpose(0, 1)

    k_all = torch.cat([k_hist.float(), k_recent.float()], dim=1)
    v_all = torch.cat([v_hist.float(), v_recent.float()], dim=1)

    return _matmul_attend(query, k_all, v_all, gqa_ratio, num_kv_heads, scale)


def _attend_hybrid_triton(
    query: torch.Tensor,
    flat: FlatCache,
    quantizer: TurboQuantProd,
    recent_k: torch.Tensor,
    recent_v: torch.Tensor,
    gqa_ratio: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
) -> torch.Tensor:
    T, Q_heads, D = query.shape
    q_2d = query.squeeze(0)  # (Q_heads, D)

    # ── Compressed segment via TQ (un-normalized) ──
    acc_tq, m_tq, l_tq = turboquant_fused_decode_split_k(
        q_2d,
        flat.prod_q,
        flat.value_q,
        quantizer.mse_quantizer.Pi,
        quantizer.S,
        quantizer.mse_quantizer.centroids,
        flat.prod_q.mse_bits,
        quantizer.qjl_scale,
        scale,
        num_kv_heads=num_kv_heads,
        gqa_ratio=gqa_ratio,
        group_size=32,
        return_aux=True,
    )
    # acc_tq: (Q_heads, D); m_tq, l_tq: (Q_heads,)

    # ── Recent segment via PyTorch (also un-normalized) ──
    # recent_k: (N_r, H_kv, D) → (H_kv, N_r, D)
    k_r = recent_k.transpose(0, 1).float()
    v_r = recent_v.transpose(0, 1).float()
    H_kv = num_kv_heads
    N_r = k_r.shape[1]

    # Reshape query to (H_kv, gqa_ratio, D)
    q_grp = q_2d.float().view(H_kv, gqa_ratio, D)
    # scores: (H_kv, gqa_ratio, N_r)
    scores_r = torch.einsum("hgd,hnd->hgn", q_grp, k_r) * scale
    m_r = scores_r.amax(dim=-1)                                    # (H_kv, gqa_ratio)
    exp_r = torch.exp(scores_r - m_r.unsqueeze(-1))
    l_r = exp_r.sum(dim=-1)                                        # (H_kv, gqa_ratio)
    # acc_r (un-normalized): (H_kv, gqa_ratio, D)
    acc_r = torch.einsum("hgn,hnd->hgd", exp_r, v_r)

    # Flatten to (Q_heads,) like TQ outputs
    m_r_flat = m_r.reshape(Q_heads)
    l_r_flat = l_r.reshape(Q_heads)
    acc_r_flat = acc_r.reshape(Q_heads, D)

    # ── Online-softmax merge ──
    m_global = torch.maximum(m_tq, m_r_flat)
    alpha_tq = torch.exp(m_tq - m_global)               # (Q_heads,)
    alpha_r = torch.exp(m_r_flat - m_global)
    l_global = alpha_tq * l_tq + alpha_r * l_r_flat
    acc_global = alpha_tq.unsqueeze(-1) * acc_tq + alpha_r.unsqueeze(-1) * acc_r_flat
    out = acc_global / l_global.unsqueeze(-1)                       # (Q_heads, D)

    return out.view(T, Q_heads, D).to(query.dtype)


# ─── Standard PyTorch attention helper (used as fallback) ────────────────


def _matmul_attend(
    query: torch.Tensor,
    kv_keys: torch.Tensor,
    kv_values: torch.Tensor,
    gqa_ratio: int,
    num_kv_heads: int,
    scale: float,
) -> torch.Tensor:
    """Standard matmul attention with GQA support.

    query: (T, Q_heads, D)
    kv_keys: (H_kv, N, D)
    kv_values: (H_kv, N, D)

    Returns: (T, Q_heads, D)
    """
    T, Q, D = query.shape
    H_kv = num_kv_heads
    if Q != H_kv * gqa_ratio:
        raise ValueError(
            f"Incompatible GQA shapes: Q={Q}, H_kv={H_kv}, gqa_ratio={gqa_ratio}"
        )

    # Avoid repeat_interleave(Q/H) on KV tensors to keep memory bounded at long context.
    # q: (T, Q, D) -> (H_kv, G, T, D)
    q = query.float().view(T, H_kv, gqa_ratio, D).permute(1, 2, 0, 3)
    k = kv_keys.float().unsqueeze(1)   # (H_kv, 1, N, D) broadcast over G
    v = kv_values.float().unsqueeze(1)  # (H_kv, 1, N, D) broadcast over G

    # scores: (H_kv, G, T, N)
    scores = torch.einsum("hgtd,hgnd->hgtn", q, k) * scale
    weights = F.softmax(scores, dim=-1)
    out = torch.einsum("hgtn,hgnd->hgtd", weights, v)

    # Back to (T, Q, D)
    return out.permute(2, 0, 1, 3).reshape(T, Q, D).to(query.dtype)
