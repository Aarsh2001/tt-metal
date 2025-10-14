# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from tests.ttnn.utils_for_testing import (
    check_with_pcc,
    start_measuring_time,
    stop_measuring_time,
)
from models.common.utility_functions import torch_random


# Override default timeout (in seconds) for hang detection
TIMEOUT = 30


# Parameter suite
parameters = {
    "nightly": {
        "batch_sizes": [(1,), (4,)],
        "height": [384, 1024],
        "width": [1024, 4096],
        "input_dtype": [ttnn.bfloat16],
        "layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
        "input_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "op_name": ["reciprocal", "log", "exp", "gelu", "tanh"],
    },
}


# Main run function
def run(
    batch_sizes,
    height,
    width,
    input_dtype,
    layout,
    input_memory_config,
    output_memory_config,
    op_name,
    *,
    device,
) -> list:
    input_shape = (*batch_sizes, height, width)

    # Define value ranges depending on op (for numerical stability)
    if op_name in ["reciprocal", "log"]:
        low, high = 0.1, 10.0  # avoid zeros/negatives
    else:
        low, high = -3.0, 3.0

    # Generate random input tensor
    torch_input_tensor = torch_random(input_shape, low, high, dtype=torch.float32)

    # Resolve TTNN and PyTorch ops dynamically
    ttnn_op = getattr(ttnn, op_name)
    torch_op = ttnn.get_golden_function(ttnn_op)

    # Compute golden reference
    torch_output_tensor = torch_op(torch_input_tensor)

    # Convert input to TTNN tensor
    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=input_dtype,
        layout=layout,
        device=device,
        memory_config=input_memory_config,
    )

    # Measure execution time
    start_time = start_measuring_time()
    output_tensor = ttnn_op(input_tensor, memory_config=output_memory_config)
    output_tensor = ttnn.to_torch(output_tensor)
    e2e_perf = stop_measuring_time(start_time)

    # Return correctness + perf
    return [check_with_pcc(torch_output_tensor, output_tensor, 0.999), e2e_perf]
