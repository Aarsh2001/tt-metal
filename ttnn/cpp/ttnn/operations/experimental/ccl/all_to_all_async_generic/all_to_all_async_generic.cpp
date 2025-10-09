// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_to_all_async_generic.hpp"
#include <utility>
#include "ttnn/operations/experimental/ccl/all_to_all_async_generic/device/all_to_all_async_generic_op.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn::operations::experimental::ccl {

ttnn::Tensor ExecuteAllToAllAsyncGeneric::invoke(
    const ttnn::Tensor& input_tensor,
    const std::optional<Tensor>& persistent_output_buffer,
    const int32_t in_dim,
    const int32_t out_dim,
    const uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id) {
    return ttnn::operations::experimental::ccl::all_to_all_async_generic(
        input_tensor, persistent_output_buffer, in_dim, out_dim, num_links, memory_config, topology, subdevice_id);
}

}  // namespace ttnn::operations::experimental::ccl
