"""
Numerical guardrails for TurboQuant Triton kernels.

Reference oracles use the existing PyTorch implementations:
  - TurboQuantProd.quantize/dequantize  (turboquant/quantizer.py)
  - TurboQuantProd.attention_score      (turboquant/quantizer.py:274)
  - quantize_values/dequantize_values   (turboquant/kv_cache.py)

These tests gate every kernel refactor in the optimization plan.

Run:
    pytest tests/test_triton_kernels.py -v

No vLLM, no model — just mock data on whatever CUDA device is available.
A few tests fall back to CPU if CUDA is unavailable (pack/unpack determinism).
"""
from __future__ import annotations

import math
import pytest
import torch

from turboquant.quantizer import (
    TurboQuantMSE,
    TurboQuantProd,
    MSEQuantized,
    ProdQuantized,
    _pack_indices,
    _unpack_indices,
)
from turboquant.kv_cache import (
    quantize_values,
    dequantize_values,
    unpack_values,
    ValueQuantized,
)
from turboquant.rotation import (
    generate_rotation_matrix,
    generate_qjl_matrix,
    rotate_forward,
    rotate_backward,
)


# ─── Fixtures ──────────────────────────────────────────────────────────────

CUDA_AVAILABLE = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")

try:
    import triton  # noqa: F401
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

requires_triton = pytest.mark.skipif(
    not (CUDA_AVAILABLE and TRITON_AVAILABLE),
    reason="Triton+CUDA required",
)


def _device():
    return torch.device("cuda" if CUDA_AVAILABLE else "cpu")


@pytest.fixture
def small_setup():
    """Small mock setup for fast tests."""
    torch.manual_seed(0)
    return {
        "D": 128,
        "H_kv": 4,
        "gqa_ratio": 4,
        "N": 1024,
        "device": _device(),
        "dtype": torch.float32,
    }


@pytest.fixture
def medium_setup():
    """Medium setup matching realistic SLM decode."""
    torch.manual_seed(0)
    return {
        "D": 128,
        "H_kv": 8,
        "gqa_ratio": 4,
        "N": 4096,
        "device": _device(),
        "dtype": torch.float32,
    }


def _make_keys_values(setup):
    H_kv, N, D = setup["H_kv"], setup["N"], setup["D"]
    device, dtype = setup["device"], setup["dtype"]
    keys = torch.randn(H_kv, N, D, device=device, dtype=dtype)
    values = torch.randn(H_kv, N, D, device=device, dtype=dtype)
    return keys, values


def _make_query(setup):
    H_kv, gqa, D = setup["H_kv"], setup["gqa_ratio"], setup["D"]
    device, dtype = setup["device"], setup["dtype"]
    Q_heads = H_kv * gqa
    return torch.randn(1, Q_heads, D, device=device, dtype=dtype)


def _make_quantizer(setup, bits=3):
    return TurboQuantProd(
        dim=setup["D"],
        bits=bits,
        device=setup["device"],
        dtype=torch.float32,
        seed=42,
    )


# ─── Group 1: Bit pack/unpack determinism (pure CPU OK) ────────────────────

class TestPackUnpack:
    """Roundtrip tests for bit-packing primitives. Run on CPU if no CUDA."""

    @pytest.mark.parametrize("bits", [1, 2, 3, 4])
    def test_pack_unpack_indices_roundtrip(self, bits):
        torch.manual_seed(bits)
        d = 128
        n = 64
        max_val = (1 << (4 if bits == 3 else bits)) - 1
        idx = torch.randint(0, max_val + 1, (n, d), dtype=torch.int64)
        packed = _pack_indices(idx, bits)
        unpacked = _unpack_indices(packed, bits, d)
        assert torch.equal(idx, unpacked), \
            f"bits={bits} roundtrip failed: max diff={(idx - unpacked).abs().max()}"

    def test_qjl_signs_pack_unpack_roundtrip(self):
        """Sign bits packed in TurboQuantProd._pack_qjl_signs roundtrip."""
        torch.manual_seed(0)
        D = 128
        n = 64
        device = _device()
        prod = TurboQuantProd(dim=D, bits=3, device=device, dtype=torch.float32)
        # Random projected values; only sign matters
        projected = torch.randn(n, D, device=device)
        packed = prod._pack_qjl_signs(projected)
        signs = prod._unpack_qjl_signs(packed)
        expected = torch.where(projected > 0,
                               torch.tensor(1.0, device=device),
                               torch.tensor(-1.0, device=device))
        assert torch.equal(signs, expected.to(signs.dtype))

    def test_value_pack_unpack_roundtrip_2bit(self):
        """quantize_values then unpack_values then dequantize roundtrip."""
        torch.manual_seed(0)
        device = _device()
        v = torch.randn(4, 256, 128, device=device)  # (H_kv, N, D)
        vq = quantize_values(v, bits=2, group_size=32)
        unpacked = unpack_values(vq)
        assert unpacked.shape == v.shape
        # Reconstruction error reasonable
        recon = dequantize_values(vq, group_size=32)
        # 2-bit values: relative MSE per group bounded
        rel_err = ((recon - v) ** 2).mean() / (v ** 2).mean()
        assert rel_err < 0.5, f"2-bit value rel MSE too high: {rel_err}"

    def test_value_pack_unpack_roundtrip_4bit(self):
        torch.manual_seed(0)
        device = _device()
        v = torch.randn(4, 256, 128, device=device)
        vq = quantize_values(v, bits=4, group_size=32)
        recon = dequantize_values(vq, group_size=32)
        rel_err = ((recon - v) ** 2).mean() / (v ** 2).mean()
        assert rel_err < 0.05, f"4-bit value rel MSE too high: {rel_err}"


# ─── Group 2: Rotation invariants ─────────────────────────────────────────

class TestRotation:
    def test_pi_orthogonality(self):
        D = 128
        Pi = generate_rotation_matrix(D, _device(), torch.float32, seed=42)
        I = torch.eye(D, device=Pi.device)
        torch.testing.assert_close(Pi @ Pi.T, I, atol=1e-5, rtol=1e-5)

    def test_rotate_preserves_norm(self):
        torch.manual_seed(0)
        device = _device()
        D = 128
        Pi = generate_rotation_matrix(D, device, torch.float32, seed=42)
        x = torch.randn(32, D, device=device)
        y = rotate_forward(x, Pi)
        torch.testing.assert_close(x.norm(dim=-1), y.norm(dim=-1), atol=1e-4, rtol=1e-4)

    def test_rotate_backward_inverse(self):
        torch.manual_seed(0)
        device = _device()
        D = 128
        Pi = generate_rotation_matrix(D, device, torch.float32, seed=42)
        x = torch.randn(32, D, device=device)
        x_back = rotate_backward(rotate_forward(x, Pi), Pi)
        torch.testing.assert_close(x, x_back, atol=1e-4, rtol=1e-4)


# ─── Group 3: Quantizer roundtrip / unbiasedness ─────────────────────────

class TestQuantizer:
    def test_mse_quantize_dequantize_cos_sim(self, medium_setup):
        torch.manual_seed(0)
        D = medium_setup["D"]
        device = medium_setup["device"]
        mse_q = TurboQuantMSE(dim=D, bits=3, device=device, dtype=torch.float32)
        x = torch.randn(64, D, device=device)
        x_hat = mse_q(x)
        cos = torch.nn.functional.cosine_similarity(x, x_hat, dim=-1).mean()
        # 3-bit MSE quantization on Beta distribution: paper Table 1 says cos≈0.99+
        assert cos > 0.95, f"cos_sim={cos} too low for 3-bit MSE"

    def test_prod_unbiasedness(self, medium_setup):
        """Algorithm 2 estimator should be unbiased: E[<q, x_hat>] ≈ <q, x>."""
        torch.manual_seed(0)
        D = medium_setup["D"]
        device = medium_setup["device"]
        prod = TurboQuantProd(dim=D, bits=3, device=device, dtype=torch.float32)
        # Average over many random rotations would be ideal, but Π is fixed by seed.
        # Instead average over many vectors — mean error should be small.
        n = 256
        x = torch.randn(n, D, device=device)
        q = torch.randn(n, D, device=device)
        # Reference inner product
        true_ip = (q * x).sum(-1)
        # Estimator
        x_q = prod.quantize(x)
        x_hat = prod.dequantize(x_q)
        est_ip = (q * x_hat).sum(-1)
        # Bias should be near 0 relative to the true score scale.
        true_scale = true_ip.abs().mean().clamp_min(1e-6)
        rel_bias = (est_ip - true_ip).mean().abs() / true_scale
        # Variance of the noise vs variance of the signal — should be bounded.
        rel_std = (est_ip - true_ip).std() / true_ip.std().clamp_min(1e-6)
        assert rel_bias < 0.2, f"relative bias = {rel_bias}"
        assert rel_std < 1.0, f"relative noise std = {rel_std} (3-bit estimator)"

    def test_prod_attention_score_matches_dequant_matmul(self, medium_setup):
        """attention_score() must match dequantize() + manual matmul."""
        torch.manual_seed(0)
        D = medium_setup["D"]
        H_kv = medium_setup["H_kv"]
        N = medium_setup["N"]
        device = medium_setup["device"]
        prod = TurboQuantProd(dim=D, bits=3, device=device, dtype=torch.float32)

        keys = torch.randn(H_kv, N, D, device=device)
        query = torch.randn(H_kv, 1, D, device=device)
        key_q = prod.quantize(keys)

        # Path A: attention_score (asymmetric Q-not-quantized estimator)
        scores_a = prod.attention_score(query, key_q)  # (H_kv, 1, N)

        # Path B: dequantize then matmul (this is what score.py uses)
        k_hat = prod.dequantize(key_q)  # (H_kv, N, D)
        scores_b = torch.matmul(query, k_hat.transpose(-2, -1))  # (H_kv, 1, N)

        # These should match closely — same estimator
        torch.testing.assert_close(scores_a, scores_b, atol=1e-3, rtol=1e-3)


# ─── Group 4: Triton kernel correctness (gates Step A) ───────────────────

@requires_triton
class TestTritonKernels:
    """Compare current Triton kernels against PyTorch oracle.

    These tests are the gate for refactor in Step A. They MUST pass before
    any kernel changes. After refactor, they MUST still pass at same tolerance.
    """

    def _setup(self, setup, bits=3):
        keys, values = _make_keys_values(setup)
        query = _make_query(setup)
        quantizer = _make_quantizer(setup, bits=bits)
        return query, keys, values, quantizer

    def test_mse_score_kernel_matches_oracle(self, medium_setup):
        """turboquant_mse_score vs reference: <q, dequant_mse_only(k)>."""
        from turboquant.triton_kernels import turboquant_mse_score
        query, keys, _, quantizer = self._setup(medium_setup)
        H_kv, N, D = keys.shape

        prod_q = quantizer.quantize(keys)

        # Reference: dequantize MSE-only (drop QJL residual contribution)
        mse_only = MSEQuantized(
            indices=prod_q.mse_indices,
            norms=prod_q.norms,
            bits=prod_q.mse_bits,
        )
        k_mse = quantizer.mse_quantizer.dequantize(mse_only)  # (H_kv, N, D)
        # query: (1, Q_heads, D). Use just first H_kv heads to match key shape.
        q_per_kv = query[0, :H_kv, :]  # (H_kv, D)
        ref_scores = torch.einsum("hd,hnd->hn", q_per_kv.float(), k_mse.float())

        # Triton kernel: needs rotated query
        q_rot = torch.matmul(q_per_kv.float(), quantizer.mse_quantizer.Pi.T)
        kernel_scores = turboquant_mse_score(
            q_rot, prod_q.mse_indices, prod_q.norms,
            quantizer.mse_quantizer.centroids, prod_q.mse_bits,
        )

        torch.testing.assert_close(
            kernel_scores.float(), ref_scores.float(),
            atol=1e-2, rtol=1e-2,
            msg="MSE score kernel diverges from PyTorch dequant+matmul",
        )

    def test_qjl_score_kernel_matches_oracle(self, medium_setup):
        """turboquant_qjl_score vs reference: residual contribution to <q, k>."""
        from turboquant.triton_kernels import turboquant_qjl_score
        query, keys, _, quantizer = self._setup(medium_setup)
        H_kv, N, D = keys.shape

        prod_q = quantizer.quantize(keys)

        # Reference: full dequantize - mse_only_dequantize = qjl contribution
        mse_only = MSEQuantized(
            indices=prod_q.mse_indices, norms=prod_q.norms, bits=prod_q.mse_bits,
        )
        k_full = quantizer.dequantize(prod_q)
        k_mse = quantizer.mse_quantizer.dequantize(mse_only)
        k_qjl = k_full - k_mse

        q_per_kv = query[0, :H_kv, :].float()
        ref_scores = torch.einsum("hd,hnd->hn", q_per_kv, k_qjl.float())

        # Triton kernel: needs sketched query (q @ S^T)
        q_sketch = torch.matmul(q_per_kv, quantizer.S.float().T)
        kernel_scores = turboquant_qjl_score(
            q_sketch, prod_q.qjl_signs, prod_q.residual_norms,
            quantizer.qjl_scale,
        )

        torch.testing.assert_close(
            kernel_scores.float(), ref_scores.float(),
            atol=1e-2, rtol=1e-2,
            msg="QJL score kernel diverges from PyTorch reference",
        )

    def test_full_attention_score_kernel_matches_oracle(self, medium_setup):
        """Combined MSE+QJL Triton path matches quantizer.attention_score."""
        from turboquant.triton_kernels import turboquant_attention_score
        query, keys, _, quantizer = self._setup(medium_setup)
        H_kv, N, D = keys.shape

        prod_q = quantizer.quantize(keys)
        q_per_kv = query[0, :H_kv, :].unsqueeze(1)  # (H_kv, 1, D)

        ref_scores = quantizer.attention_score(q_per_kv, prod_q)  # (H_kv, 1, N)

        kernel_scores = turboquant_attention_score(
            q_per_kv,  # (H_kv, 1, D)
            prod_q,
            quantizer.mse_quantizer.Pi,
            quantizer.S,
            quantizer.mse_quantizer.centroids,
            prod_q.mse_bits,
            quantizer.qjl_scale,
        )

        # kernel_scores is (BH, N) — reshape to match
        kernel_scores = kernel_scores.view(H_kv, 1, N)
        torch.testing.assert_close(
            kernel_scores.float(), ref_scores.float(),
            atol=1e-2, rtol=1e-2,
        )

    def test_fused_decode_matches_pytorch_pipeline(self, medium_setup):
        """End-to-end fused decode vs PyTorch dequant+matmul+softmax+matmul."""
        from turboquant.triton_kernels import turboquant_fused_decode
        query, keys, values, quantizer = self._setup(medium_setup)
        H_kv, N, D = keys.shape

        prod_q = quantizer.quantize(keys)
        val_q = quantize_values(values, bits=2, group_size=32)

        # Reference path: dequant + softmax + matmul (NO GQA — query per H_kv head)
        k_dequant = quantizer.dequantize(prod_q)              # (H_kv, N, D)
        v_dequant = dequantize_values(val_q, group_size=32)   # (H_kv, N, D)
        q_per_kv = query[0, :H_kv, :].float()                  # (H_kv, D)
        sm_scale = 1.0 / math.sqrt(D)
        scores = torch.einsum("hd,hnd->hn", q_per_kv, k_dequant.float()) * sm_scale
        weights = torch.softmax(scores, dim=-1)                 # (H_kv, N)
        ref_out = torch.einsum("hn,hnd->hd", weights, v_dequant.float())

        kernel_out = turboquant_fused_decode(
            q_per_kv,      # (BH=H_kv, D)
            prod_q,
            val_q,
            quantizer.mse_quantizer.Pi,
            quantizer.S,
            quantizer.mse_quantizer.centroids,
            prod_q.mse_bits,
            quantizer.qjl_scale,
            sm_scale,
            group_size=32,
        )

        # End-to-end tolerance is looser because of online softmax + 2-bit value quant
        torch.testing.assert_close(
            kernel_out.float(), ref_out.float(),
            atol=5e-2, rtol=5e-2,
            msg="Fused decode kernel diverges from PyTorch oracle (E2E)",
        )


# ─── Group 5: Architectural variants ([Optim 2]) ─────────────────────────


@requires_triton
class TestSplitKGQA:
    """Tests for [Optim 2] — Flash-Decoding split-K + GQA-aware kernel.

    Reference oracle = PyTorch dequantize + matmul + softmax + matmul,
    with explicit GQA expansion via einsum (matches `score._matmul_attend`).
    """

    def _setup(self, medium_setup):
        return TestTritonKernels()._setup(medium_setup)

    def _pytorch_oracle(self, query_per_q, keys, values, quantizer, gqa_ratio, sm_scale):
        """PyTorch reference: full attention with GQA broadcast, returning (BH_q, D)."""
        prod_q = quantizer.quantize(keys)
        val_q = quantize_values(values, bits=2, group_size=32)
        k_dequant = quantizer.dequantize(prod_q)              # (H_kv, N, D)
        v_dequant = dequantize_values(val_q, group_size=32)   # (H_kv, N, D)
        H_kv, N, D = keys.shape
        # query_per_q: (H_kv * gqa_ratio, D)
        q_grouped = query_per_q.float().view(H_kv, gqa_ratio, D)
        # scores: (H_kv, gqa_ratio, N)
        scores = torch.einsum("hgd,hnd->hgn", q_grouped, k_dequant.float()) * sm_scale
        weights = torch.softmax(scores, dim=-1)
        out = torch.einsum("hgn,hnd->hgd", weights, v_dequant.float())
        return out.reshape(H_kv * gqa_ratio, D)

    def test_split_k_gqa_matches_oracle(self, medium_setup):
        """Combined split-K + GQA path matches PyTorch reference."""
        from turboquant.triton_kernels import turboquant_fused_decode_split_k

        query, keys, values, quantizer = self._setup(medium_setup)
        H_kv, N, D = keys.shape
        gqa_ratio = medium_setup["gqa_ratio"]
        BH_q = H_kv * gqa_ratio
        sm_scale = 1.0 / math.sqrt(D)

        prod_q = quantizer.quantize(keys)
        val_q = quantize_values(values, bits=2, group_size=32)
        query_per_q = query.squeeze(0)  # (BH_q, D)

        ref_out = self._pytorch_oracle(query_per_q, keys, values, quantizer,
                                       gqa_ratio, sm_scale)

        kernel_out = turboquant_fused_decode_split_k(
            query_per_q, prod_q, val_q,
            quantizer.mse_quantizer.Pi, quantizer.S,
            quantizer.mse_quantizer.centroids,
            prod_q.mse_bits, quantizer.qjl_scale, sm_scale,
            num_kv_heads=H_kv, gqa_ratio=gqa_ratio,
            group_size=32,
        )

        torch.testing.assert_close(
            kernel_out.float(), ref_out.float(),
            atol=5e-2, rtol=5e-2,
            msg="Split-K + GQA kernel diverges from PyTorch oracle",
        )

    def test_split_k_p1_matches_p4(self, medium_setup):
        """P=1 and P=4 must produce nearly identical output (correctness gate)."""
        from turboquant.triton_kernels import turboquant_fused_decode_split_k

        query, keys, values, quantizer = self._setup(medium_setup)
        H_kv, N, D = keys.shape
        gqa_ratio = medium_setup["gqa_ratio"]
        sm_scale = 1.0 / math.sqrt(D)
        prod_q = quantizer.quantize(keys)
        val_q = quantize_values(values, bits=2, group_size=32)
        query_per_q = query.squeeze(0)

        kwargs = dict(
            quantized_key=prod_q, value_quantized=val_q,
            Pi=quantizer.mse_quantizer.Pi, S=quantizer.S,
            centroids=quantizer.mse_quantizer.centroids,
            mse_bits=prod_q.mse_bits, qjl_scale=quantizer.qjl_scale,
            sm_scale=sm_scale, num_kv_heads=H_kv, gqa_ratio=gqa_ratio,
            group_size=32,
        )
        out_p1 = turboquant_fused_decode_split_k(query_per_q, split_k=1, **kwargs)
        out_p4 = turboquant_fused_decode_split_k(query_per_q, split_k=4, **kwargs)

        # Split-K adds one extra LSE merge stage — tolerance ~5e-4 in fp32.
        torch.testing.assert_close(
            out_p1.float(), out_p4.float(),
            atol=1e-3, rtol=1e-3,
            msg="split_k=4 diverges from split_k=1 — Online softmax merge buggy",
        )

    def test_gqa_independent_softmax_per_query_head(self, medium_setup):
        """Two query heads sharing one KV head must NOT have merged softmax.

        Construct queries where head 0 and head 1 attend strongly to different
        token positions. If the kernel incorrectly reduces over GQA_G, the
        outputs will be averaged.
        """
        from turboquant.triton_kernels import turboquant_fused_decode_split_k

        torch.manual_seed(0)
        device = medium_setup["device"]
        D = medium_setup["D"]
        H_kv = 2
        gqa_ratio = 2
        N = 256
        sm_scale = 1.0 / math.sqrt(D)

        keys = torch.randn(H_kv, N, D, device=device)
        values = torch.randn(H_kv, N, D, device=device)
        quantizer = TurboQuantProd(dim=D, bits=3, device=device, dtype=torch.float32)

        # Build queries that strongly correlate with two different keys per kv_head.
        # Head 0 queries match key at position 10; head 1 queries match key at 200.
        q0_target_a = keys[0, 10, :].clone()                    # kv_head=0, q_local=0
        q0_target_b = keys[0, 200, :].clone()                   # kv_head=0, q_local=1
        q1_target_a = keys[1, 50, :].clone()                    # kv_head=1, q_local=0
        q1_target_b = keys[1, 150, :].clone()                   # kv_head=1, q_local=1

        # Layout: bh_q = bh_kv * gqa_ratio + g
        query = torch.stack([
            q0_target_a * 5,
            q0_target_b * 5,
            q1_target_a * 5,
            q1_target_b * 5,
        ], dim=0)  # (4, D)

        prod_q = quantizer.quantize(keys)
        val_q = quantize_values(values, bits=2, group_size=32)

        out = turboquant_fused_decode_split_k(
            query, prod_q, val_q,
            quantizer.mse_quantizer.Pi, quantizer.S,
            quantizer.mse_quantizer.centroids,
            prod_q.mse_bits, quantizer.qjl_scale, sm_scale,
            num_kv_heads=H_kv, gqa_ratio=gqa_ratio,
            group_size=32,
        )  # (4, D)

        # Output for head 0 (q targeting position 10) should be close to values[0, 10]
        # Output for head 1 (q targeting position 200) should be close to values[0, 200]
        # If GQA softmax was merged, both outputs would be similar (averaged).
        diff_within_kv0 = (out[0] - out[1]).abs().mean()
        diff_within_kv1 = (out[2] - out[3]).abs().mean()
        # Non-trivial difference expected
        assert diff_within_kv0 > 0.05, (
            f"head 0 and 1 (same kv_head) too similar: {diff_within_kv0} — "
            "softmax may be incorrectly merged across GQA_G"
        )
        assert diff_within_kv1 > 0.05, (
            f"head 2 and 3 (same kv_head) too similar: {diff_within_kv1}"
        )

    def test_partial_buffer_resize(self):
        """Dynamic buffer alloc grows correctly under varying batch."""
        from turboquant.triton_kernels import get_or_resize_partial_buffers

        class State:
            pass

        s = State()
        device = _device()
        # First call B=1
        m1, l1, a1 = get_or_resize_partial_buffers(s, B=1, num_kv_heads=4,
                                                    P=4, gqa_ratio=4, D=128,
                                                    device=device)
        cap1 = s._partial_buf_capacity
        assert m1.shape == (4, 4, 4)
        assert a1.shape == (4, 4, 4, 128)

        # Second call B=8 — must grow
        m2, l2, a2 = get_or_resize_partial_buffers(s, B=8, num_kv_heads=4,
                                                    P=4, gqa_ratio=4, D=128,
                                                    device=device)
        cap2 = s._partial_buf_capacity
        assert cap2[0] >= cap1[0], "buffer should grow"
        assert m2.shape == (32, 4, 4)
        assert a2.shape == (32, 4, 4, 128)

        # Third call B=2 — must reuse, not shrink
        m3, l3, a3 = get_or_resize_partial_buffers(s, B=2, num_kv_heads=4,
                                                    P=4, gqa_ratio=4, D=128,
                                                    device=device)
        cap3 = s._partial_buf_capacity
        assert cap3 == cap2, "buffer should not shrink"
        assert m3.shape == (8, 4, 4)


@requires_triton
class TestArchitecturalVariants:
    """Placeholder for [Optim 4] — prefill batch quantize."""

    @pytest.mark.xfail(reason="Triton prefill batch-quantize kernel ([Optim 4])")
    def test_prefill_batch_quantize_bit_exact(self, medium_setup):
        pass
