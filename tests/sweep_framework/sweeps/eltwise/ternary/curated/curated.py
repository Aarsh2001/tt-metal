# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
from functools import partial
import torch
import random
import ttnn

from tests.sweep_framework.sweep_utils.sharding_utils import (
    gen_sharded_spec_unary,
    parse_sharding_spec,
    invalidate_vector_sharding,
)
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt, gen_bin
from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.common.utility_functions import torch_random

# Override the default timeout (seconds) for hang detection.
TIMEOUT = 30

random.seed(0)


sharded_specs = gen_sharded_spec_unary(2, max_tensor_size_per_core=20 * 1024, layouts=["TILE_LAYOUT"])
parameters = {
    "nightly": {
        "input_spec": [None, *random.sample(sharded_specs, 2)],
        "input_a_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_b_dtype": [ttnn.bfloat16],
        "input_c_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_b_layout": [ttnn.TILE_LAYOUT],
        "input_c_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG],
        "input_b_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "input_c_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
    },
}


def invalidate_vector(test_vector) -> Tuple[bool, Optional[str]]:
    input_spec = test_vector["input_spec"]

    if input_spec is None:
        return False, None

    memory_config = test_vector["input_a_memory_config"]

    if memory_config != ttnn.L1_MEMORY_CONFIG:
        return True, "Only test sharding in L1"

    input_layout = test_vector["input_spec"]["input_layout"]
    sharding_invalidated, output_str = invalidate_vector_sharding(test_vector["input_spec"])

    if input_layout == "ROW_MAJOR_LAYOUT":
        return True, "Inputs to eltwise binary must be tilized"
    if sharding_invalidated:
        return sharding_invalidated, output_str
    return False, None


def run(
    input_spec,
    input_a_dtype,
    input_b_dtype,
    input_c_dtype,
    input_a_layout,
    input_b_layout,
    input_c_layout,
    input_a_memory_config,
    input_b_memory_config,
    input_c_memory_config,
    output_memory_config,
    *,
    device,
) -> list:
    data_seed = random.randint(0, 20000000)
    torch.manual_seed(data_seed)

    (
        input_shape,
        core_grid,
        sharding_strategy,
        shard_orientation,
        tensor_hw_as_shard_shape,
        _,
        shard_height_mul_of_32,
    ) = parse_sharding_spec(sharded_specs[0] if input_spec is None else input_spec)

    # Generate test tensors
    torch_input_tensor_a = gen_func_with_cast_tt(gen_bin, input_a_dtype)(input_shape)
    torch_input_tensor_b = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_b_dtype
    )(input_shape)
    torch_input_tensor_c = gen_func_with_cast_tt(
        partial(torch_random, low=-100, high=100, dtype=torch.float32), input_c_dtype
    )(input_shape)

    # Golden reference
    torch_output_tensor = torch.where(torch_input_tensor_a > 0, torch_input_tensor_b, torch_input_tensor_c)

    config = (
        input_a_memory_config
        if input_spec is None
        else ttnn.create_sharded_memory_config_(
            shape=input_shape,
            core_grid=core_grid,
            strategy=sharding_strategy,
            orientation=shard_orientation,
            use_height_and_width_as_shard_shape=tensor_hw_as_shard_shape,
            tile_layout=shard_height_mul_of_32,
        )
    )

    # Convert to ttnn tensors
    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=input_a_dtype,
        layout=input_a_layout,
        device=device,
        memory_config=config,
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=input_b_dtype,
        layout=input_b_layout,
        device=device,
        memory_config=input_b_memory_config,
    )
    input_tensor_c = ttnn.from_torch(
        torch_input_tensor_c,
        dtype=input_c_dtype,
        layout=input_c_layout,
        device=device,
        memory_config=input_c_memory_config,
    )

    # Run op
    start_time = start_measuring_time()
    output_tensor = ttnn.where(input_tensor_a, input_tensor_b, input_tensor_c, memory_config=output_memory_config)
    output_tensor = ttnn.to_torch(output_tensor)
    e2e_perf = stop_measuring_time(start_time)

    # Compare results
    return [check_with_pcc(torch_output_tensor, output_tensor, 0.999), e2e_perf]
