import pytest 
import torch
import ttnn
from loguru import logger
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_pcc,
)
import os

def fa_rand(*shape):
    normal_1 = torch.randn(shape)
    normal_2 = torch.randn(shape) * 10
    bernoulli = torch.bernoulli(torch.full(shape, 0.001))
    return normal_1 + normal_2 * bernoulli


def is_watcher_enabled():
    return os.environ.get("TT_METAL_WATCHER") is not None



def create_sliding_window_mask_prefill(b, nh, seq_len, sliding_window=None, is_causal=True):
    """
    Create attention mask for sliding window attention in prefill mode.

    Args:
        b: batch size
        nh: number of heads
        seq_len: sequence length
        sliding_window: sliding window size
        is_causal: whether to apply causal constraint

    Returns:
        attn_mask: [b, nh, seq_len, seq_len] mask with -inf for positions outside window
    """
    attn_mask = torch.zeros((b, nh, seq_len, seq_len))

    for i in range(b):
        for q_pos in range(seq_len):
            if is_causal:
                # Causal sliding window: spans from (q_pos - sliding_window + 1) to q_pos (inclusive)
                window_end = q_pos + 1  # exclusive (causal constraint)
                window_start = max(0, window_end - sliding_window if sliding_window is not None else 0)

                # Mask positions before sliding window start
                if window_start > 0:
                    attn_mask[i, :, q_pos, :window_start] = torch.finfo(torch.float32).min

                # Mask positions after current position (causal constraint)
                if q_pos + 1 < seq_len:
                    attn_mask[i, :, q_pos, q_pos + 1 :] = torch.finfo(torch.float32).min
            elif sliding_window is not None:
                # Non-causal sliding window: centered on diagonal with half before and half after
                half_window = sliding_window // 2
                window_start = max(0, q_pos - half_window)
                window_end = min(seq_len, q_pos + half_window + 1)  # exclusive

                # Mask positions outside the sliding window
                if window_start > 0:
                    attn_mask[i, :, q_pos, :window_start] = torch.finfo(torch.float32).min
                if window_end < seq_len:
                    attn_mask[i, :, q_pos, window_end:] = torch.finfo(torch.float32).min

    return attn_mask


def run_test_sdpa_sliding_window(
    device, b, nh, nkv, s, d, q_chunk_size, k_chunk_size, dtype, sliding_window, is_causal=True, rmse_threshold=None
):
    """Test sliding window attention in prefill mode."""
    torch.manual_seed(1234)

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        # compute_with_storage_grid_size=(1, 1),
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=True,
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    Q = fa_rand(b, nh, s, d)
    K = fa_rand(b, nkv, s, d)
    V = fa_rand(b, nkv, s, d)

    tt_Q = ttnn.from_torch(Q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)
    tt_K = ttnn.from_torch(K, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)
    tt_V = ttnn.from_torch(V, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)

    tt_back = ttnn.transformer.scaled_dot_product_attention(
        tt_Q,
        tt_K,
        tt_V,
        is_causal=is_causal,
        sliding_window=sliding_window,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
    )
    tt_back = ttnn.to_torch(tt_back)
    # Slice out any tile-padding
    tt_back = tt_back[:, :, :s, :]

    # Create reference with sliding window mask
    K_repeated = torch.cat([K[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1)  # b, nh, s, s
    V_repeated = torch.cat([V[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1)  # b, nh, s, s

    # Create sliding window mask
    sliding_window_mask = create_sliding_window_mask_prefill(b, nh, s, sliding_window, is_causal)
    gt = torch.nn.functional.scaled_dot_product_attention(
        Q, K_repeated, V_repeated, attn_mask=sliding_window_mask, is_causal=False
    )

    out_pass, out_pcc = comp_pcc(gt, tt_back, 0.994)
    logger.debug(f"python vs pytorch: {out_pcc}")
    rmse = torch.sqrt(((gt - tt_back) ** 2).mean()).item()
    logger.debug(f"rmse: {rmse}")
    breakpoint()
    if rmse_threshold is not None:
        assert rmse < rmse_threshold
    else:
        assert out_pass


@pytest.mark.skipif(is_watcher_enabled(), reason="Kernel OOM with watcher enabled")
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16], ids=["bfp8", "bf16"])
@pytest.mark.parametrize("q_chunk_size", [32, 128, 256], ids=["q32", "q128", "q256"])
@pytest.mark.parametrize("k_chunk_size", [32, 128, 256], ids=["k32", "k128", "k256"])
@pytest.mark.parametrize(
    "b, nh, nkv, s, d, sliding_window",
    [
        # Test different sliding window sizes
        # [1, 8, 1, 1024, 128, 64],  # Small window
        # [1, 8, 1, 1024, 128, 128],  # Medium window
        # [1, 8, 1, 1024, 128, 256],  # Large window
        # [1, 8, 1, 2048, 128, 128],  # Longer sequence
        # [2, 8, 1, 512, 128, 64],  # Batch size > 1
        # [1, 16, 2, 1024, 128, 128],  # GQA with sliding window
        [1, 4, 2, 32*1024, 128, 1024], # gemma
    ],
)
def test_sdpa_sliding_window(device, b, nh, nkv, s, d, dtype, q_chunk_size, k_chunk_size, sliding_window):
    """Test sliding window attention functionality in SDPA prefill."""
    if (s % q_chunk_size != 0) or (s % k_chunk_size != 0):
        pytest.skip("s must be divisible by q_chunk_size and k_chunk_size")
    # if sliding_window >= s:
    #     pytest.skip("sliding_window must be smaller than sequence length")

    ttnn.device.DisablePersistentKernelCache()
    # rmse_threshold = 0.01
    rmse_threshold = None
    run_test_sdpa_sliding_window(
        device, b, nh, nkv, s, d, q_chunk_size, k_chunk_size, dtype, sliding_window, rmse_threshold=rmse_threshold
    )



def reference_sdpa_with_attention_sinks(Q, K, V, S, is_causal=True):
    """
    Reference implementation of scaled dot product attention with attention sinks.
    
    Args:
        Q: Query tensor [b, nh, s, d]
        K: Key tensor [b, nh, s, d]
        V: Value tensor [b, nh, s, d]
        S: Attention sink tensor [b, nh, s, 1] - one sink value per query position
        is_causal: Whether to apply causal masking
    
    Returns:
        Output tensor [b, nh, s, d]
    """
    b, nh, s, d = Q.shape
    assert K.shape == (b, nh, s, d)
    assert V.shape == (b, nh, s, d)
    assert S.shape == (b, nh, s, 1), f"Expected S shape {(b, nh, s, 1)}, got {S.shape}"
    
    # Compute attention scores: QK = Q @ K^T
    # Q: [b, nh, s, d], K: [b, nh, s, d] -> QK: [b, nh, s, s]
    QK = torch.matmul(Q, K.transpose(-2, -1))
    
    # Scale
    sm_scale = 1.0 / math.sqrt(d)
    QK = QK * sm_scale
    
    # Apply causal mask if needed
    if is_causal:
        causal_mask = torch.triu(torch.full((s, s), float("-inf"), device=Q.device, dtype=Q.dtype), diagonal=1)
        QK = QK + causal_mask[None, None, :, :]
    
    # Concatenate attention sink scores
    # QK: [b, nh, s, s], S: [b, nh, s, 1] -> QK_with_sink: [b, nh, s, s+1]
    QK_with_sink = torch.cat([QK, S], dim=-1)
    
    # Apply softmax over extended dimension (including sink)
    W = torch.softmax(QK_with_sink, dim=-1)
    
    # Slice off attention sink weights (they don't contribute to output)
    W = W[..., :-1]  # [b, nh, s, s]
    
    # Compute final output
    output = torch.matmul(W, V)  # [b, nh, s, d]
    
    return output


def run_test_sdpa_with_attention_sink(
    device, b, nh, nkv, s, d, q_chunk_size, k_chunk_size, dtype, sink_scale=1.0, rmse_threshold=None
):
    """Test SDPA with attention sinks."""
    torch.manual_seed(1234)

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=True,
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    Q = fa_rand(b, nh, s, d)
    K = fa_rand(b, nkv, s, d)
    V = fa_rand(b, nkv, s, d)
    
    # Create attention sink tensor
    # Shape: [b, nh, q_chunk_size, 1] for each Q chunk
    # We need to create a sink value for each query position
    num_q_chunks = (s + q_chunk_size - 1) // q_chunk_size
    
    # Create sink values for all positions (we'll reshape/pad as needed)
    # For simplicity, create [b, nh, s, 1] and we'll chunk it appropriately
    S_full = torch.randn(b, nh, s, 1) * sink_scale

    tt_Q = ttnn.from_torch(Q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)
    tt_K = ttnn.from_torch(K, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)
    tt_V = ttnn.from_torch(V, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)
    
    # For the TT implementation, we only pass the sink for the first Q chunk
    # Since our kernel processes Q in chunks, it reads the appropriate sink values per chunk
    # The sink tensor on device should have shape [b, nh, q_chunk_size, 1]
    S_chunk = S_full[:, :, :q_chunk_size, :]
    tt_S = ttnn.from_torch(S_chunk, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)

    tt_back = ttnn.transformer.scaled_dot_product_attention(
        tt_Q,
        tt_K,
        tt_V,
        is_causal=True,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
        attention_sink=tt_S,
    )
    tt_back = ttnn.to_torch(tt_back)
    # Slice out any tile-padding
    tt_back = tt_back[:, :, :s, :]

    # Compute reference with GQA expansion
    K_repeated = torch.cat([K[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1)
    V_repeated = torch.cat([V[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1)
    
    # For reference, we need to handle chunked processing
    # Since TT processes Q in chunks and applies sink per chunk, we need to match that behavior
    # For now, we'll test with the first Q chunk only by comparing just that portion
    gt_chunk = reference_sdpa_with_attention_sinks(
        Q[:, :, :q_chunk_size, :],
        K_repeated[:, :, :q_chunk_size, :],
        V_repeated[:, :, :q_chunk_size, :],
        S_chunk,
        is_causal=True
    )
    
    # Compare only the first Q chunk
    tt_back_chunk = tt_back[:, :, :q_chunk_size, :]

    out_pass, out_pcc = comp_pcc(gt_chunk, tt_back_chunk, 0.99)
    logger.debug(f"python vs pytorch: {out_pcc}")
    rmse = torch.sqrt(((gt_chunk - tt_back_chunk) ** 2).mean()).item()
    logger.debug(f"rmse: {rmse}")
    
    if rmse_threshold is not None:
        assert rmse < rmse_threshold, f"RMSE {rmse} exceeds threshold {rmse_threshold}"
    else:
        assert out_pass, f"PCC check failed: {out_pcc}"


@pytest.mark.skipif(is_watcher_enabled(), reason="Kernel OOM with watcher enabled")
@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("q_chunk_size", [32, 128], ids=["q32", "q128"])
@pytest.mark.parametrize("k_chunk_size", [128], ids=["k128"])
@pytest.mark.parametrize(
    "b, nh, nkv, s, d",
    [
        [1, 8, 1, 128, 128],  # Basic test
        # [1, 16, 1, 256, 64],  # Different head/dim config
        # [2, 8, 1, 128, 128],  # Batch size > 1
        # [1, 8, 2, 128, 128],  # GQA
    ],
)
@pytest.mark.parametrize("sink_scale", [0.5, 2.0], ids=["weak_sink", "strong_sink"])
def test_sdpa_with_attention_sink(device, b, nh, nkv, s, d, dtype, q_chunk_size, k_chunk_size, sink_scale):
    """Test SDPA with attention sinks on device."""
    if (s % q_chunk_size != 0) or (s % k_chunk_size != 0):
        pytest.skip("s must be divisible by q_chunk_size and k_chunk_size")
    if nh % nkv != 0:
        pytest.skip("nkv must divide nh")

    ttnn.device.DisablePersistentKernelCache()
    rmse_threshold = 0.015  # Slightly higher threshold due to sink approximation
    run_test_sdpa_with_attention_sink(
        device, b, nh, nkv, s, d, q_chunk_size, k_chunk_size, dtype, sink_scale, rmse_threshold=rmse_threshold
    )


def test_attention_sink_effect(device):
    """Test that attention sinks actually reduce attention on real tokens."""
    torch.manual_seed(1234)
    
    b, nh, s, d = 1, 8, 128, 128
    q_chunk_size, k_chunk_size = 128, 128
    dtype = ttnn.bfloat16

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=True,
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    Q = fa_rand(b, nh, s, d)
    K = fa_rand(b, nh, s, d)
    V = fa_rand(b, nh, s, d)

    tt_Q = ttnn.from_torch(Q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)
    tt_K = ttnn.from_torch(K, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)
    tt_V = ttnn.from_torch(V, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)

    # Run without attention sink
    tt_out_no_sink = ttnn.transformer.scaled_dot_product_attention(
        tt_Q, tt_K, tt_V, is_causal=True, program_config=program_config, compute_kernel_config=compute_kernel_config
    )
    tt_out_no_sink = ttnn.to_torch(tt_out_no_sink)[:, :, :s, :]

    # Run with strong positive attention sink (should absorb significant attention)
    S = torch.full((b, nh, q_chunk_size, 1), 5.0)  # Strong positive sink
    tt_S = ttnn.from_torch(S, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)
    
    tt_out_with_sink = ttnn.transformer.scaled_dot_product_attention(
        tt_Q,
        tt_K,
        tt_V,
        is_causal=True,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
        attention_sink=tt_S,
    )
    tt_out_with_sink = ttnn.to_torch(tt_out_with_sink)[:, :, :s, :]

    # Outputs should be different
    assert not torch.allclose(tt_out_no_sink, tt_out_with_sink, rtol=1e-2), \
        "Attention sink should change the output"
    
    # With a strong positive sink, output magnitude should generally be reduced
    # (more attention probability goes to the sink)
    norm_no_sink = torch.norm(tt_out_no_sink).item()
    norm_with_sink = torch.norm(tt_out_with_sink).item()
    
    logger.debug(f"Output norm without sink: {norm_no_sink}")
    logger.debug(f"Output norm with sink: {norm_with_sink}")
    logger.debug(f"Ratio: {norm_with_sink / norm_no_sink}")
    
    # The sink should reduce the output magnitude
    assert norm_with_sink < norm_no_sink, \
        f"Strong attention sink should reduce output magnitude. Got {norm_with_sink} >= {norm_no_sink}"
