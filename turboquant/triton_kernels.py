"""
TurboQuant fused Triton kernels for decode attention.

Optimizations applied (see docs/MATH.md for the math):
  [Optim 1] Vectorized bit-unpack          → kernels A, B, C, D
  [Optim 2] Split-K Flash-Decoding + GQA   → kernels D + E
  [Optim 5] Tensor Core via tl.dot (bf16)  → kernel D
  [Optim 5] cache_modifier '.cg' for V      → kernels C, D

Invariants (mandatory):
  - m_i / l_i / acc are ALWAYS fp32 in online softmax (no drift at large N).
  - Combine kernel reduces only over P axis — never mixes query heads.

Production hot path: turboquant_fused_decode_split_k (kernels D + E).
"""

import math
import torch
import triton
import triton.language as tl


# ─── Internal helpers ────────────────────────────────────────────────────


def _get_packing_params(bits: int):
    """Match the bit-packing scheme of `quantizer._pack_indices`.

    Returns (effective_bits, vals_per_byte). 3-bit indices are stored as
    4-bit (2 per byte) for simpler unpacking.
    """
    if bits == 1:
        return 1, 8
    elif bits == 2:
        return 2, 4
    elif bits <= 4:
        return 4, 2  # 3-bit rounds up to 4-bit packing
    else:
        return 8, 1


# ─── Kernel A: MSE score ─────────────  [Optim 1]  ──────────────────────
# scores[bh, n] = norms[bh, n] * <q_rot[bh], centroids[mse_idx[bh, n, :]]>
# Rotate query once (q_rot = q @ Π^T) instead of dequantizing each key — Π
# is orthogonal so the inner products are identical, no D×D matmul needed.

@triton.jit
def _turboquant_mse_score_kernel(
    Q_ptr,          # (BH, D)  q @ Pi^T (rotated query)
    MSE_ptr,        # (BH, N, PACKED_D) bit-packed indices (uint8)
    NORMS_ptr,      # (BH, N)  original key vector L2 norms
    CENTROIDS_ptr,  # (n_clusters,)  Lloyd-Max codebook
    OUT_ptr,        # (BH, N)  output scores
    # Strides
    stride_q_bh, stride_q_d,
    stride_m_bh, stride_m_n, stride_m_d,
    stride_n_bh, stride_n_n,
    stride_o_bh, stride_o_n,
    # Dimensions
    N,
    D: tl.constexpr,
    PACKED_D: tl.constexpr,
    PADDED_D: tl.constexpr,            # PACKED_D * VALS_PER_BYTE (>= D)
    BITS: tl.constexpr,
    VALS_PER_BYTE: tl.constexpr,
    BIT_MASK: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_n = tl.program_id(1)

    n_start = pid_n * BLOCK_N
    n_offs = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N

    pd_offs = tl.arange(0, PACKED_D)                           # (PACKED_D,)
    d_offs = tl.arange(0, PADDED_D)                             # (PADDED_D,)
    d_mask = d_offs < D

    # ── Load packed tile: (BLOCK_N, PACKED_D) uint8 ──
    packed_tile = tl.load(
        MSE_ptr
        + pid_bh * stride_m_bh
        + n_offs[:, None] * stride_m_n
        + pd_offs[None, :] * stride_m_d,
        mask=n_mask[:, None],
        other=0,
    ).to(tl.int32)

    # ── Vectorized unpack: (BLOCK_N, PACKED_D, VALS_PER_BYTE) ──
    shifts = (tl.arange(0, VALS_PER_BYTE) * BITS).to(tl.int32)  # (VPB,)
    idx_3d = (packed_tile[:, :, None] >> shifts[None, None, :]) & BIT_MASK
    # Reshape to (BLOCK_N, PADDED_D)
    idx_2d = tl.reshape(idx_3d, (BLOCK_N, PADDED_D))

    # ── Gather centroids: (BLOCK_N, PADDED_D) fp32 ──
    # Codebook is tiny (≤ 16 floats) — sits in L1/constant cache.
    centroid_tile = tl.load(CENTROIDS_ptr + idx_2d).to(tl.float32)
    centroid_tile = tl.where(d_mask[None, :], centroid_tile, 0.0)

    # ── Load rotated query for this BH: (PADDED_D,) ──
    q_tile = tl.load(
        Q_ptr + pid_bh * stride_q_bh + d_offs * stride_q_d,
        mask=d_mask, other=0.0,
    ).to(tl.float32)

    # ── Score reduction: sum_d centroid_tile[n, d] * q_tile[d] ──
    scores = tl.sum(centroid_tile * q_tile[None, :], axis=1)  # (BLOCK_N,) fp32

    # Multiply by per-key norms
    norms = tl.load(
        NORMS_ptr + pid_bh * stride_n_bh + n_offs * stride_n_n,
        mask=n_mask, other=0.0,
    ).to(tl.float32)
    scores = scores * norms

    tl.store(
        OUT_ptr + pid_bh * stride_o_bh + n_offs * stride_o_n,
        scores, mask=n_mask,
    )


# ─── Kernel B: QJL score ─────────────  [Optim 1]  ──────────────────────
# scores[bh, n] = qjl_scale * res_norms[n] * <q_sketch[bh], signs[bh, n, :]>

@triton.jit
def _turboquant_qjl_score_kernel(
    Q_SKETCH_ptr,    # (BH, D)  q @ S^T
    SIGNS_ptr,       # (BH, N, PACKED_D_SIGNS=ceil(D/8)) packed sign bits
    RES_NORMS_ptr,   # (BH, N) residual L2 norms
    OUT_ptr,         # (BH, N) output (added to existing)
    stride_qs_bh, stride_qs_d,
    stride_s_bh, stride_s_n, stride_s_d,
    stride_rn_bh, stride_rn_n,
    stride_o_bh, stride_o_n,
    N,
    D: tl.constexpr,
    PACKED_D_SIGNS: tl.constexpr,    # = ceil(D / 8)
    PADDED_D: tl.constexpr,           # = PACKED_D_SIGNS * 8 (>= D)
    QJL_SCALE,
    BLOCK_N: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_n = tl.program_id(1)

    n_start = pid_n * BLOCK_N
    n_offs = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N

    pd_offs = tl.arange(0, PACKED_D_SIGNS)
    d_offs = tl.arange(0, PADDED_D)
    d_mask = d_offs < D

    # ── Load sign bytes: (BLOCK_N, PACKED_D_SIGNS) ──
    signs_packed = tl.load(
        SIGNS_ptr
        + pid_bh * stride_s_bh
        + n_offs[:, None] * stride_s_n
        + pd_offs[None, :] * stride_s_d,
        mask=n_mask[:, None], other=0,
    ).to(tl.int32)

    # ── Vectorized 8-bit unpack to (BLOCK_N, PADDED_D) of {0, 1} ──
    bit_shifts = tl.arange(0, 8).to(tl.int32)                              # (8,)
    bits_3d = (signs_packed[:, :, None] >> bit_shifts[None, None, :]) & 1  # (BLOCK_N, PACKED_D_SIGNS, 8)
    bits_2d = tl.reshape(bits_3d, (BLOCK_N, PADDED_D))

    # Convert {0, 1} → {-1, +1}
    sign_tile = (2.0 * bits_2d.to(tl.float32) - 1.0)
    sign_tile = tl.where(d_mask[None, :], sign_tile, 0.0)

    # ── Load sketched query (PADDED_D,) ──
    q_tile = tl.load(
        Q_SKETCH_ptr + pid_bh * stride_qs_bh + d_offs * stride_qs_d,
        mask=d_mask, other=0.0,
    ).to(tl.float32)

    # ── Dot reduction ──
    dot = tl.sum(sign_tile * q_tile[None, :], axis=1)  # (BLOCK_N,) fp32

    # Scale by residual norm and QJL constant
    res_norms = tl.load(
        RES_NORMS_ptr + pid_bh * stride_rn_bh + n_offs * stride_rn_n,
        mask=n_mask, other=0.0,
    ).to(tl.float32)
    qjl_scores = dot * res_norms * QJL_SCALE

    # Add to existing buffer (caller may have written MSE scores there)
    existing = tl.load(
        OUT_ptr + pid_bh * stride_o_bh + n_offs * stride_o_n,
        mask=n_mask, other=0.0,
    ).to(tl.float32)
    tl.store(
        OUT_ptr + pid_bh * stride_o_bh + n_offs * stride_o_n,
        existing + qjl_scores, mask=n_mask,
    )


# ─── Kernel C: Fused decode  ───  [Optim 1 + 5]  ────────────────────────
# Single-program-per-BH path: scores + online softmax + V aggregation in
# one pass. Uses tl.sum (no batched M dim → no Tensor Core, see kernel D).

@triton.jit
def _turboquant_fused_decode_kernel(
    # Pre-rotated / pre-sketched query
    Q_ROT_ptr,       # (BH, D)  q @ Pi^T
    Q_SKETCH_ptr,    # (BH, D)  q @ S^T
    # Quantized keys
    MSE_ptr,         # (BH, N, PACKED_D_MSE)
    SIGNS_ptr,       # (BH, N, PACKED_D_SIGNS)
    NORMS_ptr,       # (BH, N)
    RES_NORMS_ptr,   # (BH, N)
    CENTROIDS_ptr,   # (n_clusters,)
    # Group-quantized values (already unpacked to per-element uint8)
    V_DATA_ptr,      # (BH, N, D) uint8 in [0, 2^v_bits-1]
    V_SCALES_ptr,    # (BH, N, N_GROUPS) fp32
    V_ZEROS_ptr,     # (BH, N, N_GROUPS) fp32
    # Output
    OUT_ptr,         # (BH, D)
    # Strides
    stride_q_bh, stride_q_d,
    stride_qs_bh, stride_qs_d,
    stride_m_bh, stride_m_n, stride_m_d,
    stride_s_bh, stride_s_n, stride_s_d,
    stride_n_bh, stride_n_n,
    stride_rn_bh, stride_rn_n,
    stride_v_bh, stride_v_n, stride_v_d,
    stride_vs_bh, stride_vs_n, stride_vs_g,
    stride_vz_bh, stride_vz_n, stride_vz_g,
    stride_o_bh, stride_o_d,
    # Dims
    N,
    D: tl.constexpr,
    PACKED_D_MSE: tl.constexpr,
    PADDED_D_MSE: tl.constexpr,
    PACKED_D_SIGNS: tl.constexpr,
    PADDED_D_SIGNS: tl.constexpr,
    N_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BITS: tl.constexpr,
    VALS_PER_BYTE: tl.constexpr,
    BIT_MASK: tl.constexpr,
    QJL_SCALE,
    SM_SCALE,
    BLOCK_N: tl.constexpr,
):
    pid_bh = tl.program_id(0)

    pd_mse_offs = tl.arange(0, PACKED_D_MSE)
    d_mse_offs = tl.arange(0, PADDED_D_MSE)
    d_mse_mask = d_mse_offs < D

    pd_signs_offs = tl.arange(0, PACKED_D_SIGNS)
    d_signs_offs = tl.arange(0, PADDED_D_SIGNS)
    d_signs_mask = d_signs_offs < D

    d_offs = tl.arange(0, D)

    # Load rotated/sketched query — shared across all KV blocks
    q_rot = tl.load(
        Q_ROT_ptr + pid_bh * stride_q_bh + d_mse_offs * stride_q_d,
        mask=d_mse_mask, other=0.0,
    ).to(tl.float32)
    q_sketch = tl.load(
        Q_SKETCH_ptr + pid_bh * stride_qs_bh + d_signs_offs * stride_qs_d,
        mask=d_signs_mask, other=0.0,
    ).to(tl.float32)

    # ── Online softmax state — INVARIANT: fp32 ──
    m_i = tl.full([1], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([1], dtype=tl.float32)
    acc = tl.zeros([D], dtype=tl.float32)

    bit_shifts_signs = tl.arange(0, 8).to(tl.int32)
    shifts_mse = (tl.arange(0, VALS_PER_BYTE) * BITS).to(tl.int32)

    num_blocks = tl.cdiv(N, BLOCK_N)

    for block_idx in range(num_blocks):
        n_start = block_idx * BLOCK_N
        n_offs = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_offs < N

        # ── Part 1: MSE score (vectorized unpack) ──
        packed_mse = tl.load(
            MSE_ptr
            + pid_bh * stride_m_bh
            + n_offs[:, None] * stride_m_n
            + pd_mse_offs[None, :] * stride_m_d,
            mask=n_mask[:, None], other=0,
        ).to(tl.int32)

        idx_3d = (packed_mse[:, :, None] >> shifts_mse[None, None, :]) & BIT_MASK
        idx_2d = tl.reshape(idx_3d, (BLOCK_N, PADDED_D_MSE))
        centroid_tile = tl.load(CENTROIDS_ptr + idx_2d).to(tl.float32)
        centroid_tile = tl.where(d_mse_mask[None, :], centroid_tile, 0.0)

        mse_scores = tl.sum(centroid_tile * q_rot[None, :], axis=1)  # (BLOCK_N,)

        key_norms = tl.load(
            NORMS_ptr + pid_bh * stride_n_bh + n_offs * stride_n_n,
            mask=n_mask, other=0.0,
        ).to(tl.float32)
        mse_scores = mse_scores * key_norms

        # ── Part 2: QJL score (vectorized sign unpack) ──
        signs_packed = tl.load(
            SIGNS_ptr
            + pid_bh * stride_s_bh
            + n_offs[:, None] * stride_s_n
            + pd_signs_offs[None, :] * stride_s_d,
            mask=n_mask[:, None], other=0,
        ).to(tl.int32)
        bits_3d = (signs_packed[:, :, None] >> bit_shifts_signs[None, None, :]) & 1
        bits_2d = tl.reshape(bits_3d, (BLOCK_N, PADDED_D_SIGNS))
        sign_tile = (2.0 * bits_2d.to(tl.float32) - 1.0)
        sign_tile = tl.where(d_signs_mask[None, :], sign_tile, 0.0)

        qjl_dot = tl.sum(sign_tile * q_sketch[None, :], axis=1)  # (BLOCK_N,)

        res_norms = tl.load(
            RES_NORMS_ptr + pid_bh * stride_rn_bh + n_offs * stride_rn_n,
            mask=n_mask, other=0.0,
        ).to(tl.float32)
        qjl_scores = qjl_dot * res_norms * QJL_SCALE

        # ── Combined logits, masked, scaled ──
        scores = (mse_scores + qjl_scores) * SM_SCALE
        scores = tl.where(n_mask, scores, -float("inf"))

        # ── Online softmax update (fp32 throughout) ──
        block_max = tl.max(scores, 0)
        m_new = tl.maximum(m_i, block_max)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)                # (BLOCK_N,)

        l_i = l_i * alpha + tl.sum(p, 0)
        acc = acc * alpha

        # ── Dequantize values for this block ──
        # Use streaming cache modifier: V tile is read once per decode step,
        # caching in L1 evicts useful K data. SM75+ supports `.cg`.
        v_quant = tl.load(
            V_DATA_ptr
            + pid_bh * stride_v_bh
            + n_offs[:, None] * stride_v_n
            + d_offs[None, :] * stride_v_d,
            mask=n_mask[:, None], other=0,
            cache_modifier=".cg",
        ).to(tl.float32)

        g_offs = d_offs // GROUP_SIZE
        v_scale = tl.load(
            V_SCALES_ptr
            + pid_bh * stride_vs_bh
            + n_offs[:, None] * stride_vs_n
            + g_offs[None, :] * stride_vs_g,
            mask=n_mask[:, None], other=1.0,
        ).to(tl.float32)
        v_zero = tl.load(
            V_ZEROS_ptr
            + pid_bh * stride_vz_bh
            + n_offs[:, None] * stride_vz_n
            + g_offs[None, :] * stride_vz_g,
            mask=n_mask[:, None], other=0.0,
        ).to(tl.float32)
        v_dequant = v_quant * v_scale + v_zero  # (BLOCK_N, D)
        v_dequant = tl.where(n_mask[:, None], v_dequant, 0.0)

        # ── Weighted accumulation ──
        acc += tl.sum(p[:, None] * v_dequant, 0)

        m_i = m_new

    # Final normalization
    out = acc / l_i

    tl.store(
        OUT_ptr + pid_bh * stride_o_bh + d_offs * stride_o_d,
        out,
    )


# ─── Python wrappers ─────────────────────────────────────────────────────


def turboquant_mse_score(
    query_rot: torch.Tensor,
    mse_packed: torch.Tensor,
    norms: torch.Tensor,
    centroids: torch.Tensor,
    mse_bits: int,
) -> torch.Tensor:
    """Compute MSE attention scores: <q_rot, dequant_mse(k)> * norm.

    Args:
        query_rot: (BH, D) — query already rotated by Pi^T.
        mse_packed: (BH, N, packed_d) uint8 bit-packed indices.
        norms: (BH, N) original key norms.
        centroids: (n_clusters,) Lloyd-Max codebook (fp32).
        mse_bits: original bit width before packing (3 → packed as 4-bit).

    Returns: (BH, N) fp32 scores.
    """
    if query_rot.dim() == 3:
        query_rot = query_rot.squeeze(1)

    BH, D = query_rot.shape
    N = mse_packed.shape[1]
    packed_d = mse_packed.shape[2]
    eff_bits, vals_per_byte = _get_packing_params(mse_bits)
    bit_mask = (1 << eff_bits) - 1
    padded_d = packed_d * vals_per_byte

    out = torch.zeros(BH, N, device=query_rot.device, dtype=torch.float32)

    BLOCK_N = min(64, max(16, triton.next_power_of_2(min(N, 64))))
    grid = (BH, triton.cdiv(N, BLOCK_N))

    _turboquant_mse_score_kernel[grid](
        query_rot, mse_packed, norms, centroids, out,
        query_rot.stride(0), query_rot.stride(1),
        mse_packed.stride(0), mse_packed.stride(1), mse_packed.stride(2),
        norms.stride(0), norms.stride(1),
        out.stride(0), out.stride(1),
        N=N, D=D,
        PACKED_D=packed_d, PADDED_D=padded_d,
        BITS=eff_bits, VALS_PER_BYTE=vals_per_byte, BIT_MASK=bit_mask,
        BLOCK_N=BLOCK_N,
        num_warps=4,
    )
    return out


def turboquant_qjl_score(
    q_sketched: torch.Tensor,
    qjl_signs: torch.Tensor,
    residual_norms: torch.Tensor,
    qjl_scale: float,
    out: torch.Tensor = None,
) -> torch.Tensor:
    """Compute QJL residual contribution to attention scores.

    If `out` is provided, ADD QJL scores to it in-place. Otherwise allocate.
    Returns: (BH, N) fp32 scores.
    """
    if q_sketched.dim() == 3:
        q_sketched = q_sketched.squeeze(1)

    BH, D = q_sketched.shape
    N = qjl_signs.shape[1]
    packed_d_signs = qjl_signs.shape[2]
    padded_d = packed_d_signs * 8

    if out is None:
        out = torch.zeros(BH, N, device=q_sketched.device, dtype=torch.float32)

    BLOCK_N = min(64, max(16, triton.next_power_of_2(min(N, 64))))
    grid = (BH, triton.cdiv(N, BLOCK_N))

    _turboquant_qjl_score_kernel[grid](
        q_sketched, qjl_signs, residual_norms, out,
        q_sketched.stride(0), q_sketched.stride(1),
        qjl_signs.stride(0), qjl_signs.stride(1), qjl_signs.stride(2),
        residual_norms.stride(0), residual_norms.stride(1),
        out.stride(0), out.stride(1),
        N=N, D=D,
        PACKED_D_SIGNS=packed_d_signs, PADDED_D=padded_d,
        QJL_SCALE=qjl_scale,
        BLOCK_N=BLOCK_N,
        num_warps=4,
    )
    return out


def turboquant_attention_score(
    query: torch.Tensor,
    quantized_key,
    Pi: torch.Tensor,
    S: torch.Tensor,
    centroids: torch.Tensor,
    mse_bits: int,
    qjl_scale: float,
) -> torch.Tensor:
    """High-level: compute combined MSE + QJL attention scores.

    Args:
        query: (B, H, 1, D) or (BH, 1, D) — un-rotated query.
        quantized_key: ProdQuantized.
        Pi, S: rotation matrices.
        centroids: codebook.

    Returns: (BH, N) raw logits.
    """
    if query.dim() == 4:
        B, H, Q, D = query.shape
        query_flat = query.reshape(B * H, Q, D)
    else:
        query_flat = query
        D = query.shape[-1]

    q_squeezed = query_flat.squeeze(1).float()
    q_rot = torch.matmul(q_squeezed, Pi.T)
    q_sketch = torch.matmul(q_squeezed, S.T)

    mse_packed = quantized_key.mse_indices
    qjl_signs = quantized_key.qjl_signs
    norms = quantized_key.norms
    res_norms = quantized_key.residual_norms

    if mse_packed.dim() == 4:
        BH_shape = mse_packed.shape[:2]
        BH = BH_shape[0] * BH_shape[1]
        mse_packed = mse_packed.reshape(BH, *mse_packed.shape[2:])
        qjl_signs = qjl_signs.reshape(BH, *qjl_signs.shape[2:])
        norms = norms.reshape(BH, -1)
        res_norms = res_norms.reshape(BH, -1)

    scores = turboquant_mse_score(q_rot, mse_packed, norms, centroids, mse_bits)
    scores = turboquant_qjl_score(q_sketch, qjl_signs, res_norms, qjl_scale, out=scores)
    return scores


def turboquant_fused_decode(
    query: torch.Tensor,
    quantized_key,
    value_quantized,
    Pi: torch.Tensor,
    S: torch.Tensor,
    centroids: torch.Tensor,
    mse_bits: int,
    qjl_scale: float,
    sm_scale: float,
    group_size: int = 32,
) -> torch.Tensor:
    """Fused decode: scores + softmax + V aggregation in one kernel.

    Args:
        query: (BH, D) or (BH, 1, D) un-rotated.
        quantized_key: ProdQuantized.
        value_quantized: ValueQuantized.
        Pi, S: rotation matrices.
        centroids: codebook.
        sm_scale: softmax scale (typically 1/sqrt(D)).

    Returns: (BH, D) attention output in query.dtype.
    """
    if query.dim() == 3:
        query = query.squeeze(1)
    BH, D = query.shape

    q_rot = torch.matmul(query.float(), Pi.T)
    q_sketch = torch.matmul(query.float(), S.T)

    mse_packed = quantized_key.mse_indices
    qjl_signs = quantized_key.qjl_signs
    norms = quantized_key.norms
    res_norms = quantized_key.residual_norms

    if mse_packed.dim() > 3:
        BH_actual = mse_packed.shape[0] * mse_packed.shape[1]
        mse_packed = mse_packed.reshape(BH_actual, *mse_packed.shape[2:])
        qjl_signs = qjl_signs.reshape(BH_actual, *qjl_signs.shape[2:])
        norms = norms.reshape(BH_actual, -1)
        res_norms = res_norms.reshape(BH_actual, -1)

    N = mse_packed.shape[1]
    packed_d_mse = mse_packed.shape[2]
    packed_d_signs = qjl_signs.shape[2]

    eff_bits, vals_per_byte = _get_packing_params(mse_bits)
    bit_mask = (1 << eff_bits) - 1
    padded_d_mse = packed_d_mse * vals_per_byte
    padded_d_signs = packed_d_signs * 8

    # Value data: unpack to (BH, N, D) uint8 if currently bit-packed.
    v_data = value_quantized.data
    v_bits = value_quantized.bits if len(value_quantized) > 3 else 2
    if v_data.shape[-1] != D:
        from turboquant.kv_cache import unpack_values
        v_data = unpack_values(value_quantized)

    v_scales = value_quantized.scales
    v_zeros = value_quantized.zeros

    if v_data.dim() > 3:
        v_data = v_data.reshape(BH, N, -1)
        v_scales = v_scales.reshape(BH, N, -1)
        v_zeros = v_zeros.reshape(BH, N, -1)

    # Ensure dtype/contiguous
    v_data = v_data.contiguous()
    v_scales = v_scales.float().contiguous()
    v_zeros = v_zeros.float().contiguous()

    n_groups = D // group_size

    out = torch.zeros(BH, D, device=query.device, dtype=torch.float32)
    BLOCK_N = min(64, max(16, triton.next_power_of_2(min(N, 64))))
    grid = (BH,)

    _turboquant_fused_decode_kernel[grid](
        q_rot, q_sketch,
        mse_packed, qjl_signs, norms, res_norms, centroids,
        v_data, v_scales, v_zeros,
        out,
        # Q strides
        q_rot.stride(0), q_rot.stride(1),
        q_sketch.stride(0), q_sketch.stride(1),
        # MSE strides
        mse_packed.stride(0), mse_packed.stride(1), mse_packed.stride(2),
        # Signs strides
        qjl_signs.stride(0), qjl_signs.stride(1), qjl_signs.stride(2),
        # Norms strides
        norms.stride(0), norms.stride(1),
        res_norms.stride(0), res_norms.stride(1),
        # Value strides
        v_data.stride(0), v_data.stride(1), v_data.stride(2),
        v_scales.stride(0), v_scales.stride(1), v_scales.stride(2),
        v_zeros.stride(0), v_zeros.stride(1), v_zeros.stride(2),
        # Out strides
        out.stride(0), out.stride(1),
        # Dims
        N=N, D=D,
        PACKED_D_MSE=packed_d_mse, PADDED_D_MSE=padded_d_mse,
        PACKED_D_SIGNS=packed_d_signs, PADDED_D_SIGNS=padded_d_signs,
        N_GROUPS=n_groups, GROUP_SIZE=group_size,
        BITS=eff_bits, VALS_PER_BYTE=vals_per_byte, BIT_MASK=bit_mask,
        QJL_SCALE=qjl_scale, SM_SCALE=sm_scale,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=3,
    )

    return out.to(query.dtype)


# ═════════════ [Optim 2] Split-K Flash-Decoding + GQA  ═══════════════════
# Pass 1 (kernel D): grid (BH_kv, P) — one program per (kv_head, split),
#                    processes GQA_G query heads at once → output partials.
# Pass 2 (kernel E): grid (BH_kv, GQA_G) — reduces only over P axis
#                    (never mixes query heads; each head has its own softmax).
# Partial buffers (fp32): m_p, l_p (BH_kv, P, GQA_G); acc_p (BH_kv, P, GQA_G, D).
# LSE associativity proof: docs/MATH.md §F.


@triton.jit
def _fused_decode_partial_kernel(
    Q_ROT_ptr,       # (BH_kv, GQA_G, D)
    Q_SKETCH_ptr,    # (BH_kv, GQA_G, D)
    MSE_ptr,         # (BH_kv, N, PACKED_D_MSE)
    SIGNS_ptr,       # (BH_kv, N, PACKED_D_SIGNS)
    NORMS_ptr,       # (BH_kv, N)
    RES_NORMS_ptr,   # (BH_kv, N)
    CENTROIDS_ptr,
    V_DATA_ptr,      # (BH_kv, N, D)
    V_SCALES_ptr,    # (BH_kv, N, N_GROUPS)
    V_ZEROS_ptr,     # (BH_kv, N, N_GROUPS)
    Mp_ptr,          # (BH_kv, P, GQA_G)
    Lp_ptr,          # (BH_kv, P, GQA_G)
    ACCp_ptr,        # (BH_kv, P, GQA_G, D)
    # Strides — query
    s_qr_bh, s_qr_g, s_qr_d,
    s_qs_bh, s_qs_g, s_qs_d,
    # Strides — keys
    s_m_bh, s_m_n, s_m_d,
    s_s_bh, s_s_n, s_s_d,
    s_n_bh, s_n_n,
    s_rn_bh, s_rn_n,
    # Strides — values
    s_v_bh, s_v_n, s_v_d,
    s_vs_bh, s_vs_n, s_vs_g,
    s_vz_bh, s_vz_n, s_vz_g,
    # Strides — partial buffers
    s_mp_bh, s_mp_p, s_mp_g,
    s_lp_bh, s_lp_p, s_lp_g,
    s_ap_bh, s_ap_p, s_ap_g, s_ap_d,
    # Dims
    N,
    SPLIT_SIZE: tl.constexpr,           # tokens per Split (must be multiple of BLOCK_N)
    D: tl.constexpr,
    PACKED_D_MSE: tl.constexpr,
    PADDED_D_MSE: tl.constexpr,
    PACKED_D_SIGNS: tl.constexpr,
    PADDED_D_SIGNS: tl.constexpr,
    N_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    GQA_G: tl.constexpr,
    GQA_G_PAD: tl.constexpr,        # padded to >= 16 for tl.dot
    BITS: tl.constexpr,
    VALS_PER_BYTE: tl.constexpr,
    BIT_MASK: tl.constexpr,
    QJL_SCALE,
    SM_SCALE,
    BLOCK_N: tl.constexpr,
):
    pid_bh_kv = tl.program_id(0)
    pid_p = tl.program_id(1)

    # Range of N tokens this split handles
    n_split_start = pid_p * SPLIT_SIZE
    n_split_end_capped = tl.minimum(n_split_start + SPLIT_SIZE, N)

    # Triton 3.x tl.dot requires M, K, N >= 16. We pad GQA_G up to GQA_G_PAD
    # along the M axis: rows [0, GQA_G) are valid; rows [GQA_G, GQA_G_PAD) are
    # zero-padded and discarded at write time.
    g_offs = tl.arange(0, GQA_G_PAD)
    g_mask = g_offs < GQA_G
    d_offs = tl.arange(0, D)

    pd_mse_offs = tl.arange(0, PACKED_D_MSE)
    d_mse_offs = tl.arange(0, PADDED_D_MSE)
    d_mse_mask = d_mse_offs < D

    pd_signs_offs = tl.arange(0, PACKED_D_SIGNS)
    d_signs_offs = tl.arange(0, PADDED_D_SIGNS)
    d_signs_mask = d_signs_offs < D

    # ── Load Q tile for this kv_head: (GQA_G_PAD, D), zero-pad rows g >= GQA_G ──
    load_mask_q_mse = g_mask[:, None] & d_mse_mask[None, :]
    q_rot = tl.load(
        Q_ROT_ptr
        + pid_bh_kv * s_qr_bh
        + g_offs[:, None] * s_qr_g
        + d_mse_offs[None, :] * s_qr_d,
        mask=load_mask_q_mse, other=0.0,
    ).to(tl.float32)
    load_mask_q_signs = g_mask[:, None] & d_signs_mask[None, :]
    q_sketch = tl.load(
        Q_SKETCH_ptr
        + pid_bh_kv * s_qs_bh
        + g_offs[:, None] * s_qs_g
        + d_signs_offs[None, :] * s_qs_d,
        mask=load_mask_q_signs, other=0.0,
    ).to(tl.float32)

    # ── Online softmax state (fp32, per query head g, padded to GQA_G_PAD) ──
    m_i = tl.full([GQA_G_PAD], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([GQA_G_PAD], dtype=tl.float32)
    acc = tl.zeros([GQA_G_PAD, D], dtype=tl.float32)

    bit_shifts_signs = tl.arange(0, 8).to(tl.int32)
    shifts_mse = (tl.arange(0, VALS_PER_BYTE) * BITS).to(tl.int32)

    num_blocks_in_split: tl.constexpr = SPLIT_SIZE // BLOCK_N

    for block_idx in range(num_blocks_in_split):
        n_offs = n_split_start + block_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = n_offs < n_split_end_capped

        # ── MSE score: vectorized unpack → (BLOCK_N, D) → (GQA_G, BLOCK_N) ──
        packed_mse = tl.load(
            MSE_ptr
            + pid_bh_kv * s_m_bh
            + n_offs[:, None] * s_m_n
            + pd_mse_offs[None, :] * s_m_d,
            mask=n_mask[:, None], other=0,
        ).to(tl.int32)

        idx_3d = (packed_mse[:, :, None] >> shifts_mse[None, None, :]) & BIT_MASK
        idx_2d = tl.reshape(idx_3d, (BLOCK_N, PADDED_D_MSE))
        centroid_tile = tl.load(CENTROIDS_ptr + idx_2d).to(tl.float32)
        centroid_tile = tl.where(d_mse_mask[None, :], centroid_tile, 0.0)

        key_norms = tl.load(
            NORMS_ptr + pid_bh_kv * s_n_bh + n_offs * s_n_n,
            mask=n_mask, other=0.0,
        ).to(tl.float32)

        # MSE scores: (GQA_G, BLOCK_N) = q_rot @ centroid_tile.T
        # tl.dot hits Tensor Cores even when GQA_G < 16 (Triton pads M to 16).
        # Cast operands to bf16 for TC; accumulator stays fp32.
        mse_scores = tl.dot(
            q_rot.to(tl.bfloat16),
            tl.trans(centroid_tile).to(tl.bfloat16),
            out_dtype=tl.float32,
        )                                                              # (GQA_G, BLOCK_N) fp32
        mse_scores = mse_scores * key_norms[None, :]

        # ── QJL score ──
        signs_packed = tl.load(
            SIGNS_ptr
            + pid_bh_kv * s_s_bh
            + n_offs[:, None] * s_s_n
            + pd_signs_offs[None, :] * s_s_d,
            mask=n_mask[:, None], other=0,
        ).to(tl.int32)
        bits_3d = (signs_packed[:, :, None] >> bit_shifts_signs[None, None, :]) & 1
        bits_2d = tl.reshape(bits_3d, (BLOCK_N, PADDED_D_SIGNS))
        sign_tile = (2.0 * bits_2d.to(tl.float32) - 1.0)
        sign_tile = tl.where(d_signs_mask[None, :], sign_tile, 0.0)

        res_norms = tl.load(
            RES_NORMS_ptr + pid_bh_kv * s_rn_bh + n_offs * s_rn_n,
            mask=n_mask, other=0.0,
        ).to(tl.float32)

        qjl_dot = tl.dot(
            q_sketch.to(tl.bfloat16),
            tl.trans(sign_tile).to(tl.bfloat16),
            out_dtype=tl.float32,
        )                                                              # (GQA_G, BLOCK_N) fp32
        qjl_scores = qjl_dot * res_norms[None, :] * QJL_SCALE

        # ── Combined logits ──
        scores = (mse_scores + qjl_scores) * SM_SCALE                      # (GQA_G, BLOCK_N)
        scores = tl.where(n_mask[None, :], scores, -float("inf"))

        # ── Online softmax per g (INVARIANT fp32) ──
        block_max = tl.max(scores, axis=1)              # (GQA_G,)
        m_new = tl.maximum(m_i, block_max)
        alpha = tl.exp(m_i - m_new)                     # (GQA_G,)
        p = tl.exp(scores - m_new[:, None])              # (GQA_G, BLOCK_N)

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        # ── Dequantize V (BLOCK_N, D), streaming cache (.cg, SM75+) ──
        v_quant = tl.load(
            V_DATA_ptr
            + pid_bh_kv * s_v_bh
            + n_offs[:, None] * s_v_n
            + d_offs[None, :] * s_v_d,
            mask=n_mask[:, None], other=0,
            cache_modifier=".cg",
        ).to(tl.float32)
        g_grp = d_offs // GROUP_SIZE
        v_scale = tl.load(
            V_SCALES_ptr
            + pid_bh_kv * s_vs_bh
            + n_offs[:, None] * s_vs_n
            + g_grp[None, :] * s_vs_g,
            mask=n_mask[:, None], other=1.0,
        ).to(tl.float32)
        v_zero = tl.load(
            V_ZEROS_ptr
            + pid_bh_kv * s_vz_bh
            + n_offs[:, None] * s_vz_n
            + g_grp[None, :] * s_vz_g,
            mask=n_mask[:, None], other=0.0,
        ).to(tl.float32)
        v_dequant = v_quant * v_scale + v_zero
        v_dequant = tl.where(n_mask[:, None], v_dequant, 0.0)

        # Weighted accumulation: (GQA_G, BLOCK_N) × (BLOCK_N, D) → (GQA_G, D)
        # tl.dot for tensor-core path; accumulator fp32.
        acc += tl.dot(
            p.to(tl.bfloat16),
            v_dequant.to(tl.bfloat16),
            out_dtype=tl.float32,
        )

        m_i = m_new

    # ── Write partials (NO normalization yet — combine kernel does it) ──
    # Mask write to only valid rows g < GQA_G (skip padded rows)
    tl.store(
        Mp_ptr + pid_bh_kv * s_mp_bh + pid_p * s_mp_p + g_offs * s_mp_g,
        m_i, mask=g_mask,
    )
    tl.store(
        Lp_ptr + pid_bh_kv * s_lp_bh + pid_p * s_lp_p + g_offs * s_lp_g,
        l_i, mask=g_mask,
    )
    tl.store(
        ACCp_ptr
        + pid_bh_kv * s_ap_bh
        + pid_p * s_ap_p
        + g_offs[:, None] * s_ap_g
        + d_offs[None, :] * s_ap_d,
        acc,
        mask=g_mask[:, None],
    )


@triton.jit
def _combine_partials_kernel(
    Mp_ptr,          # (BH_kv, P, GQA_G)
    Lp_ptr,          # (BH_kv, P, GQA_G)
    ACCp_ptr,        # (BH_kv, P, GQA_G, D)
    OUT_ptr,         # (BH_kv * GQA_G, D) — final output, bh_q = bh_kv*GQA_G + g
    M_GLOBAL_ptr,    # (BH_kv * GQA_G,) optional — per-row softmax max
    L_GLOBAL_ptr,    # (BH_kv * GQA_G,) optional — per-row softmax denom
    s_mp_bh, s_mp_p, s_mp_g,
    s_lp_bh, s_lp_p, s_lp_g,
    s_ap_bh, s_ap_p, s_ap_g, s_ap_d,
    s_o_bh, s_o_d,
    s_mg, s_lg,
    P: tl.constexpr,
    D: tl.constexpr,
    GQA_G: tl.constexpr,
    NORMALIZE: tl.constexpr,         # if False, write `acc` (un-normalized) and aux
    WRITE_AUX: tl.constexpr,
):
    """Reduce P partials → one final (D,) per (bh_kv, g).

    INVARIANT: reduces along axis P ONLY. GQA_G axis is grid-parallel —
    different query heads have independent softmax distributions and must
    not be merged.

    If NORMALIZE: write out = acc / l_global (standard final attention output).
    Else: write out = acc (un-normalized weighted sum). Set WRITE_AUX=True to
    also record (m_global, l_global) for later merge with another segment.
    """
    pid_bh_kv = tl.program_id(0)
    pid_g = tl.program_id(1)

    p_offs = tl.arange(0, P)
    d_offs = tl.arange(0, D)

    # Load partials for THIS specific (bh_kv, g) — vector over P
    m_p = tl.load(Mp_ptr + pid_bh_kv * s_mp_bh + p_offs * s_mp_p + pid_g * s_mp_g)
    l_p = tl.load(Lp_ptr + pid_bh_kv * s_lp_bh + p_offs * s_lp_p + pid_g * s_lp_g)

    m_global = tl.max(m_p, 0)
    alpha = tl.exp(m_p - m_global)               # (P,)
    l_global = tl.sum(l_p * alpha, 0)

    acc = tl.load(
        ACCp_ptr
        + pid_bh_kv * s_ap_bh
        + p_offs[:, None] * s_ap_p
        + pid_g * s_ap_g
        + d_offs[None, :] * s_ap_d,
    )                                             # (P, D)

    summed_acc = tl.sum(acc * alpha[:, None], axis=0)        # (D,)

    bh_q = pid_bh_kv * GQA_G + pid_g

    if NORMALIZE:
        out = summed_acc / l_global
    else:
        out = summed_acc

    tl.store(OUT_ptr + bh_q * s_o_bh + d_offs * s_o_d, out)

    if WRITE_AUX:
        tl.store(M_GLOBAL_ptr + bh_q * s_mg, m_global)
        tl.store(L_GLOBAL_ptr + bh_q * s_lg, l_global)


def _choose_split_k(BH_kv: int, N: int, device) -> int:
    """Pick number of splits P to balance SM occupancy without going too small.

    Heuristic: target ~2 programs per SM. Each split must hold enough work
    to amortize launch — minimum 256 tokens per split.
    """
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    target_programs = sm_count * 2
    P = max(1, target_programs // max(1, BH_kv))
    P = min(P, max(1, N // 256))
    # Round to a power of 2 for cleaner SPLIT_SIZE
    P = max(1, 1 << (P.bit_length() - 1))
    return P


def _choose_block_n(N: int) -> int:
    """Pick BLOCK_N adapted to consumer GPU constraints.

    Smaller BLOCK_N → less shared mem (RTX 4060 = 100KB / SM, RTX 3060 = 48KB).
    With GQA_G_PAD=16 + D=128 + fp32 acc + num_stages=2, BLOCK_N=32 stays
    safely under the limit on all SM75-SM89 GPUs.
    """
    if N >= 4096:
        return 32
    if N >= 1024:
        return 32
    return max(16, triton.next_power_of_2(min(N, 32)))


def turboquant_fused_decode_split_k(
    query: torch.Tensor,                # (BH_q, D)  un-rotated, BH_q = B*H_q
    quantized_key,                       # ProdQuantized — leading dim BH_kv
    value_quantized,                     # ValueQuantized — leading dim BH_kv
    Pi: torch.Tensor,
    S: torch.Tensor,
    centroids: torch.Tensor,
    mse_bits: int,
    qjl_scale: float,
    sm_scale: float,
    num_kv_heads: int,
    gqa_ratio: int,
    group_size: int = 32,
    split_k: int = None,
    block_n: int = None,
    partial_buffers: tuple = None,
    return_aux: bool = False,
):
    """Fused decode with Flash-Decoding split-K and GQA-aware q-batching.

    Args:
        query: (B*H_q, D). H_q = num_kv_heads * gqa_ratio.
        quantized_key: ProdQuantized whose leading dim is B*H_kv.
        value_quantized: ValueQuantized matching shape.
        num_kv_heads, gqa_ratio: must satisfy BH_q = BH_kv * gqa_ratio.
        split_k: P (number of splits). Default: chosen by `_choose_split_k`.
        partial_buffers: optional (Mp, Lp, ACCp) to reuse across decode steps.
        return_aux: when True, return (acc_unnormalized, m_global, l_global)
            instead of the normalized output. Use this when merging with
            another softmax segment (e.g., the recent buffer in hybrid
            decode). The caller must perform the final divide.

    Returns:
        - normalized: (B*H_q, D) attention output in query.dtype.
        - if return_aux=True: (acc, m_global, l_global) with shapes
          (B*H_q, D), (B*H_q,), (B*H_q,) all fp32.
    """
    if query.dim() == 3:
        query = query.squeeze(1)
    BH_q, D = query.shape
    if BH_q % gqa_ratio != 0:
        raise ValueError(f"BH_q={BH_q} not divisible by gqa_ratio={gqa_ratio}")
    BH_kv = BH_q // gqa_ratio

    mse_packed = quantized_key.mse_indices
    qjl_signs = quantized_key.qjl_signs
    norms = quantized_key.norms
    res_norms = quantized_key.residual_norms

    if mse_packed.dim() > 3:
        mse_packed = mse_packed.reshape(BH_kv, *mse_packed.shape[-2:])
        qjl_signs = qjl_signs.reshape(BH_kv, *qjl_signs.shape[-2:])
        norms = norms.reshape(BH_kv, -1)
        res_norms = res_norms.reshape(BH_kv, -1)

    if mse_packed.shape[0] != BH_kv:
        raise ValueError(
            f"BH_kv mismatch: query gives {BH_kv}, key has {mse_packed.shape[0]}"
        )

    N = mse_packed.shape[1]
    packed_d_mse = mse_packed.shape[2]
    packed_d_signs = qjl_signs.shape[2]

    # Pre-rotate / pre-sketch query (one-time)
    q_rot = torch.matmul(query.float(), Pi.T).reshape(BH_kv, gqa_ratio, D).contiguous()
    q_sketch = torch.matmul(query.float(), S.T).reshape(BH_kv, gqa_ratio, D).contiguous()

    # Choose split-K and BLOCK_N ([Optim 5] — tuned for consumer GPUs)
    if split_k is None:
        split_k = _choose_split_k(BH_kv, N, query.device)
    if block_n is None:
        block_n = _choose_block_n(N)
    # SPLIT_SIZE must be a multiple of BLOCK_N. Pad N up to split_k * SPLIT_SIZE
    # via the existing n_mask mechanism (loads beyond N are masked to 0/-inf).
    split_size = ((N + split_k - 1) // split_k + block_n - 1) // block_n * block_n
    # Re-derive split_k from the rounded split_size
    split_k = (N + split_size - 1) // split_size

    # Value layout
    eff_bits, vals_per_byte = _get_packing_params(mse_bits)
    bit_mask = (1 << eff_bits) - 1
    padded_d_mse = packed_d_mse * vals_per_byte
    padded_d_signs = packed_d_signs * 8

    v_data = value_quantized.data
    if v_data.shape[-1] != D:
        from turboquant.kv_cache import unpack_values
        v_data = unpack_values(value_quantized)
    v_scales = value_quantized.scales
    v_zeros = value_quantized.zeros
    if v_data.dim() > 3:
        v_data = v_data.reshape(BH_kv, N, -1)
        v_scales = v_scales.reshape(BH_kv, N, -1)
        v_zeros = v_zeros.reshape(BH_kv, N, -1)
    v_data = v_data.contiguous()
    v_scales = v_scales.float().contiguous()
    v_zeros = v_zeros.float().contiguous()
    n_groups = D // group_size

    # Allocate partial buffers (or reuse if caller provided).
    if partial_buffers is None:
        Mp = torch.empty(BH_kv, split_k, gqa_ratio, device=query.device, dtype=torch.float32)
        Lp = torch.empty(BH_kv, split_k, gqa_ratio, device=query.device, dtype=torch.float32)
        ACCp = torch.empty(BH_kv, split_k, gqa_ratio, D, device=query.device, dtype=torch.float32)
    else:
        Mp, Lp, ACCp = partial_buffers

    out = torch.empty(BH_q, D, device=query.device, dtype=torch.float32)

    # ── Pass 1: partial computation, grid (BH_kv, P) ──
    grid_p1 = (BH_kv, split_k)
    _fused_decode_partial_kernel[grid_p1](
        q_rot, q_sketch,
        mse_packed, qjl_signs, norms, res_norms, centroids,
        v_data, v_scales, v_zeros,
        Mp, Lp, ACCp,
        # Q strides
        q_rot.stride(0), q_rot.stride(1), q_rot.stride(2),
        q_sketch.stride(0), q_sketch.stride(1), q_sketch.stride(2),
        # Key strides
        mse_packed.stride(0), mse_packed.stride(1), mse_packed.stride(2),
        qjl_signs.stride(0), qjl_signs.stride(1), qjl_signs.stride(2),
        norms.stride(0), norms.stride(1),
        res_norms.stride(0), res_norms.stride(1),
        # Value strides
        v_data.stride(0), v_data.stride(1), v_data.stride(2),
        v_scales.stride(0), v_scales.stride(1), v_scales.stride(2),
        v_zeros.stride(0), v_zeros.stride(1), v_zeros.stride(2),
        # Partial strides
        Mp.stride(0), Mp.stride(1), Mp.stride(2),
        Lp.stride(0), Lp.stride(1), Lp.stride(2),
        ACCp.stride(0), ACCp.stride(1), ACCp.stride(2), ACCp.stride(3),
        N=N,
        SPLIT_SIZE=split_size,
        D=D,
        PACKED_D_MSE=packed_d_mse, PADDED_D_MSE=padded_d_mse,
        PACKED_D_SIGNS=packed_d_signs, PADDED_D_SIGNS=padded_d_signs,
        N_GROUPS=n_groups, GROUP_SIZE=group_size,
        GQA_G=gqa_ratio,
        GQA_G_PAD=max(16, triton.next_power_of_2(gqa_ratio)),
        BITS=eff_bits, VALS_PER_BYTE=vals_per_byte, BIT_MASK=bit_mask,
        QJL_SCALE=qjl_scale, SM_SCALE=sm_scale,
        BLOCK_N=block_n,
        num_warps=8,
        num_stages=2,
    )

    # ── Pass 2: combine partials, grid (BH_kv, GQA_G) ──
    if return_aux:
        m_global = torch.empty(BH_q, device=query.device, dtype=torch.float32)
        l_global = torch.empty(BH_q, device=query.device, dtype=torch.float32)
        s_mg = m_global.stride(0)
        s_lg = l_global.stride(0)
    else:
        # Pass dummy tensors with valid strides — kernel won't write to them.
        m_global = torch.empty(1, device=query.device, dtype=torch.float32)
        l_global = torch.empty(1, device=query.device, dtype=torch.float32)
        s_mg = 1
        s_lg = 1

    grid_p2 = (BH_kv, gqa_ratio)
    _combine_partials_kernel[grid_p2](
        Mp, Lp, ACCp, out, m_global, l_global,
        Mp.stride(0), Mp.stride(1), Mp.stride(2),
        Lp.stride(0), Lp.stride(1), Lp.stride(2),
        ACCp.stride(0), ACCp.stride(1), ACCp.stride(2), ACCp.stride(3),
        out.stride(0), out.stride(1),
        s_mg, s_lg,
        P=split_k, D=D, GQA_G=gqa_ratio,
        NORMALIZE=(not return_aux),
        WRITE_AUX=return_aux,
        num_warps=4,
    )

    if return_aux:
        return out, m_global, l_global   # acc (un-normalized), m_global, l_global
    return out.to(query.dtype)


def get_or_resize_partial_buffers(
    state,
    B: int,
    num_kv_heads: int,
    P: int,
    gqa_ratio: int,
    D: int,
    device,
    dtype=torch.float32,
):
    """Lazily allocate / grow partial buffers for split-K decode.

    Buffers grow with 1.5× headroom to avoid thrashing under continuous
    batching (vLLM's batch size oscillates per decode step).

    `state` is any object with mutable attributes:
        _partial_buf_m, _partial_buf_l, _partial_buf_acc,
        _partial_buf_capacity (tuple (BH_kv, P)).
    Initialized to None / (0, 0) on first call.
    """
    bh_kv_needed = B * num_kv_heads
    cap = getattr(state, "_partial_buf_capacity", (0, 0))
    bh_kv_alloc, p_alloc = cap

    if (bh_kv_needed > bh_kv_alloc) or (P > p_alloc) or getattr(state, "_partial_buf_m", None) is None:
        new_bh = max(bh_kv_needed, int(bh_kv_alloc * 1.5))
        new_p = max(P, p_alloc, 4)
        state._partial_buf_m = torch.empty(new_bh, new_p, gqa_ratio, device=device, dtype=dtype)
        state._partial_buf_l = torch.empty(new_bh, new_p, gqa_ratio, device=device, dtype=dtype)
        state._partial_buf_acc = torch.empty(new_bh, new_p, gqa_ratio, D, device=device, dtype=dtype)
        state._partial_buf_capacity = (new_bh, new_p)

    return (
        state._partial_buf_m[:bh_kv_needed, :P],
        state._partial_buf_l[:bh_kv_needed, :P],
        state._partial_buf_acc[:bh_kv_needed, :P],
    )
