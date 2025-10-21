# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import math


def reference_sdpa_with_sinks(Q, K, V, S, sm_scale, sliding_window=0):
    """
    Reference implementation of scaled dot product attention with attention sinks.

    Args:
        Q: Query tensor [n_tokens, n_heads, q_mult, d_head]
        K: Key tensor [n_tokens, n_heads, d_head]
        V: Value tensor [n_tokens, n_heads, d_head]
        S: Attention sink tensor [n_heads, q_mult, n_tokens, 1] - one sink value per query position
        sm_scale: Softmax scale factor
        sliding_window: Sliding window size (0 = no sliding window)

    Returns:
        Output tensor [n_tokens, n_heads * q_mult]
    """
    n_tokens, n_heads, q_mult, d_head = Q.shape
    assert K.shape == (n_tokens, n_heads, d_head)
    assert V.shape == (n_tokens, n_heads, d_head)
    assert S.shape == (
        n_heads,
        q_mult,
        n_tokens,
        1,
    ), f"Expected S shape {(n_heads, q_mult, n_tokens, 1)}, got {S.shape}"

    # Expand K and V to match Q's q_mult dimension
    K = K[:, :, None, :].expand(-1, -1, q_mult, -1)
    V = V[:, :, None, :].expand(-1, -1, q_mult, -1)

    # S is already in the right shape: [n_heads, q_mult, n_tokens, 1]
    # Each query position has its own sink value

    # Create causal mask
    mask = torch.triu(Q.new_full((n_tokens, n_tokens), float("-inf")), diagonal=1)

    # Apply sliding window if specified
    if sliding_window > 0:
        mask += torch.tril(mask.new_full((n_tokens, n_tokens), float("-inf")), diagonal=-sliding_window)

    # Compute attention scores: QK = Q @ K^T
    QK = torch.einsum("qhmd,khmd->hmqk", Q, K)
    QK *= sm_scale
    QK += mask[None, None, :, :]

    # Concatenate attention sink scores
    QK = torch.cat([QK, S], dim=-1)  # [heads, q_mult, n_tokens, n_tokens+1]

    # Apply softmax over extended dimension (including sink)
    W = torch.softmax(QK, dim=-1)

    # Slice off attention sink weights (they don't contribute to output)
    W = W[..., :-1]  # [heads, q_mult, n_tokens, n_tokens]

    # Compute final output
    attn = torch.einsum("hmqk,khmd->qhmd", W, V)
    return attn.reshape(n_tokens, -1)


def flash_attention_with_sinks(Q, K, V, S, sm_scale, q_chunk_size=32, k_chunk_size=32):
    """
    Flash Attention implementation with attention sinks using chunked processing.

    Args:
        Q: Query tensor [n_tokens, n_heads, q_mult, d_head]
        K: Key tensor [n_tokens, n_heads, d_head]
        V: Value tensor [n_tokens, n_heads, d_head]
        S: Attention sink tensor [n_heads, q_mult, n_tokens, 1]
        sm_scale: Softmax scale factor
        q_chunk_size: Size of Q chunks for tiling
        k_chunk_size: Size of K chunks for tiling

    Returns:
        Output tensor [n_tokens, n_heads * q_mult]
    """
    n_tokens, n_heads, q_mult, d_head = Q.shape
    assert K.shape == (n_tokens, n_heads, d_head)
    assert V.shape == (n_tokens, n_heads, d_head)

    # Expand K and V to match Q's q_mult dimension
    K = K[:, :, None, :].expand(-1, -1, q_mult, -1)
    V = V[:, :, None, :].expand(-1, -1, q_mult, -1)

    # Reshape tensors for easier processing: [n_heads, q_mult, n_tokens, d_head]
    Q = Q.permute(1, 2, 0, 3)
    K = K.permute(1, 2, 0, 3)
    V = V.permute(1, 2, 0, 3)

    # Initialize output accumulator
    output = torch.zeros_like(Q)

    # Process in Q chunks
    num_q_chunks = (n_tokens + q_chunk_size - 1) // q_chunk_size

    for q_chunk_idx in range(num_q_chunks):
        q_start = q_chunk_idx * q_chunk_size
        q_end = min(q_start + q_chunk_size, n_tokens)
        q_chunk_len = q_end - q_start

        Q_chunk = Q[:, :, q_start:q_end, :]  # [n_heads, q_mult, q_chunk_len, d_head]
        S_chunk = S[:, :, q_start:q_end, :]  # [n_heads, q_mult, q_chunk_len, 1]

        # Initialize running statistics for this Q chunk
        running_max = torch.full((n_heads, q_mult, q_chunk_len, 1), float("-inf"), device=Q.device, dtype=Q.dtype)
        running_sum = torch.zeros((n_heads, q_mult, q_chunk_len, 1), device=Q.device, dtype=Q.dtype)
        running_output = torch.zeros((n_heads, q_mult, q_chunk_len, d_head), device=Q.device, dtype=Q.dtype)

        # Process K chunks (all chunks up to and including current Q chunk for causal)
        # For causal attention, we attend to all K positions up to each Q position
        for k_chunk_idx in range((q_end + k_chunk_size - 1) // k_chunk_size):
            k_start = k_chunk_idx * k_chunk_size
            k_end = min(k_start + k_chunk_size, n_tokens)
            k_chunk_len = k_end - k_start

            # Only process if this K chunk contains tokens that should be attended to
            if k_start >= q_end:
                break

            K_chunk = K[:, :, k_start:k_end, :]  # [n_heads, q_mult, k_chunk_len, d_head]
            V_chunk = V[:, :, k_start:k_end, :]  # [n_heads, q_mult, k_chunk_len, d_head]

            # Compute attention scores for this QK chunk pair
            QK = (
                torch.matmul(Q_chunk, K_chunk.transpose(-2, -1)) * sm_scale
            )  # [n_heads, q_mult, q_chunk_len, k_chunk_len]

            # Apply causal mask
            q_indices = torch.arange(q_start, q_end, device=Q.device)[:, None]
            k_indices = torch.arange(k_start, k_end, device=Q.device)[None, :]
            causal_mask = q_indices < k_indices
            QK = QK.masked_fill(causal_mask[None, None, :, :], float("-inf"))

            # Compute max for this chunk (handling -inf properly)
            chunk_max = QK.max(dim=-1, keepdim=True).values  # [n_heads, q_mult, q_chunk_len, 1]

            # Update running max
            new_max = torch.maximum(running_max, chunk_max)

            # Rescale previous statistics if we're not on the first K chunk
            if k_chunk_idx > 0 or torch.any(running_max > float("-inf")):
                exp_diff_prev = torch.exp(running_max - new_max)
                # Handle -inf cases
                exp_diff_prev = torch.nan_to_num(exp_diff_prev, nan=0.0, posinf=0.0, neginf=0.0)
                running_sum = running_sum * exp_diff_prev
                running_output = running_output * exp_diff_prev

            # Compute softmax for current chunk
            QK_exp = torch.exp(QK - new_max)
            # Handle -inf cases (masked positions)
            QK_exp = torch.nan_to_num(QK_exp, nan=0.0, posinf=0.0, neginf=0.0)

            chunk_sum = QK_exp.sum(dim=-1, keepdim=True)

            # Update running sum
            running_sum = running_sum + chunk_sum

            # Accumulate weighted values
            running_output = running_output + torch.matmul(QK_exp, V_chunk)

            # Update running max
            running_max = new_max

        # Process attention sink as a virtual K chunk
        # S_chunk shape: [n_heads, q_mult, q_chunk_len, 1]
        # Update running max with sink values
        new_max = torch.maximum(running_max, S_chunk)

        # Rescale previous statistics
        exp_diff_prev = torch.exp(running_max - new_max)
        exp_diff_prev = torch.nan_to_num(exp_diff_prev, nan=0.0, posinf=0.0, neginf=0.0)
        running_sum = running_sum * exp_diff_prev
        running_output = running_output * exp_diff_prev

        # Add sink contribution to sum (sink doesn't contribute to output)
        sink_exp = torch.exp(S_chunk - new_max)
        running_sum = running_sum + sink_exp

        # Final normalization
        output[:, :, q_start:q_end, :] = running_output / running_sum

    # Reshape output back to original format
    output = output.permute(2, 0, 1, 3)  # [n_tokens, n_heads, q_mult, d_head]
    return output.reshape(n_tokens, -1)


@pytest.mark.parametrize("n_tokens", [32, 64, 128])
@pytest.mark.parametrize("n_heads", [1, 4, 8])
@pytest.mark.parametrize("q_mult", [32, 64, 128])
@pytest.mark.parametrize("d_head", [32, 64, 128])
@pytest.mark.parametrize("q_chunk_size", [16, 32])
@pytest.mark.parametrize("k_chunk_size", [16, 32])
def test_flash_attention_with_sinks(n_tokens, n_heads, q_mult, d_head, q_chunk_size, k_chunk_size):
    """Test that flash attention with sinks matches reference implementation."""

    # Skip very large tests to keep runtime reasonable
    if n_tokens * d_head > 8192:
        pytest.skip("Skipping large test configuration")

    torch.manual_seed(42)
    device = torch.device("cpu")

    # Create input tensors
    Q = torch.randn(n_tokens, n_heads, q_mult, d_head, device=device)
    K = torch.randn(n_tokens, n_heads, d_head, device=device)
    V = torch.randn(n_tokens, n_heads, d_head, device=device)

    # Create attention sink tensor
    # Each query position has one sink value
    S = torch.randn(n_heads, q_mult, n_tokens, 1, device=device)

    # Scale factor
    sm_scale = 1.0 / math.sqrt(d_head)

    # Compute reference output
    ref_output = reference_sdpa_with_sinks(Q, K, V, S, sm_scale)

    # Compute flash attention output
    flash_output = flash_attention_with_sinks(Q, K, V, S, sm_scale, q_chunk_size, k_chunk_size)

    # Compare outputs
    torch.testing.assert_close(flash_output, ref_output, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("n_tokens", [64, 128])
@pytest.mark.parametrize("n_heads", [4, 8])
def test_attention_sink_reduces_token_attention(n_tokens, n_heads):
    """Test that attention sinks reduce attention weights on actual tokens."""

    torch.manual_seed(42)
    device = torch.device("cpu")
    q_mult = 1
    d_head = 64

    # Create input tensors
    Q = torch.randn(n_tokens, n_heads, q_mult, d_head, device=device)
    K = torch.randn(n_tokens, n_heads, d_head, device=device)
    V = torch.randn(n_tokens, n_heads, d_head, device=device)

    sm_scale = 1.0 / math.sqrt(d_head)

    # Test with zero sink (should be equivalent to no sink)
    S_zero = torch.full((n_heads, q_mult, n_tokens, 1), float("-inf"), device=device)
    output_no_sink = reference_sdpa_with_sinks(Q, K, V, S_zero, sm_scale)

    # Test with positive sink values
    S_positive = torch.full((n_heads, q_mult, n_tokens, 1), 5.0, device=device)
    output_with_sink = reference_sdpa_with_sinks(Q, K, V, S_positive, sm_scale)

    # Outputs should be different (sink absorbs attention)
    assert not torch.allclose(output_no_sink, output_with_sink)

    # With a strong positive sink, output magnitude should be reduced
    # (more attention probability goes to the sink)
    output_no_sink_norm = torch.norm(output_no_sink)
    output_with_sink_norm = torch.norm(output_with_sink)
    assert output_with_sink_norm < output_no_sink_norm


@pytest.mark.parametrize(
    "n_tokens,q_chunk_size,k_chunk_size",
    [
        (64, 16, 16),
        (128, 32, 32),
        (96, 32, 24),  # Test non-perfect chunk divisions
    ],
)
def test_flash_attention_chunk_sizes(n_tokens, q_chunk_size, k_chunk_size):
    """Test flash attention with various chunk size configurations."""

    torch.manual_seed(42)
    device = torch.device("cpu")
    n_heads = 4
    q_mult = 1
    d_head = 64

    # Create input tensors
    Q = torch.randn(n_tokens, n_heads, q_mult, d_head, device=device)
    K = torch.randn(n_tokens, n_heads, d_head, device=device)
    V = torch.randn(n_tokens, n_heads, d_head, device=device)
    S = torch.randn(n_heads, q_mult, n_tokens, 1, device=device)

    sm_scale = 1.0 / math.sqrt(d_head)

    # Reference output
    ref_output = reference_sdpa_with_sinks(Q, K, V, S, sm_scale)

    # Flash attention with specified chunk sizes
    flash_output = flash_attention_with_sinks(Q, K, V, S, sm_scale, q_chunk_size, k_chunk_size)

    # Should match regardless of chunk sizes
    torch.testing.assert_close(flash_output, ref_output, rtol=1e-4, atol=1e-4)


def test_attention_sink_shapes():
    """Test that attention sink tensor must have correct shape."""

    torch.manual_seed(42)
    device = torch.device("cpu")
    n_tokens = 64
    n_heads = 4
    q_mult = 2
    d_head = 64

    Q = torch.randn(n_tokens, n_heads, q_mult, d_head, device=device)
    K = torch.randn(n_tokens, n_heads, d_head, device=device)
    V = torch.randn(n_tokens, n_heads, d_head, device=device)

    # Correct shape
    S_correct = torch.randn(n_heads, q_mult, n_tokens, 1, device=device)
    sm_scale = 1.0 / math.sqrt(d_head)

    # Should not raise
    output = reference_sdpa_with_sinks(Q, K, V, S_correct, sm_scale)
    import pdb

    pdb.set_trace()
    assert output.shape == (n_tokens, n_heads * q_mult)


if __name__ == "__main__":
    # Run a quick sanity check
    print("Running sanity check...")
    test_flash_attention_with_sinks(n_tokens=64, n_heads=4, q_mult=1, d_head=64, q_chunk_size=32, k_chunk_size=32)
    print("✓ Basic test passed")

    test_attention_sink_reduces_token_attention(n_tokens=64, n_heads=4)
    print("✓ Attention sink effect verified")

    test_flash_attention_chunk_sizes(n_tokens=96, q_chunk_size=32, k_chunk_size=24)
    print("✓ Chunk size handling verified")

    print("\nAll sanity checks passed! Run with pytest for full test suite.")
