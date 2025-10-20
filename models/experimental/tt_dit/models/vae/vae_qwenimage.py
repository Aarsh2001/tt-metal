# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import ttnn

from ...layers.conv2d import Conv2d
from ...layers.linear import Linear
from ...layers.module import Module, ModuleList
from ...layers.normalization import RMSNorm
from ...parallel.config import VAEParallelConfig
from ...parallel.manager import CCLManager
from ...utils.substate import pop_substate, rename_substate

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch


class QwenImageConv(Module):
    """Qwen-Image causal convolution without temporal dimension.

    The original QwenImage VAE supports video so the convolution is three-dimensional. Since this is
    not needed for image generation, the temporal dimension is removed here.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: tuple[int, int, int],
        stride: tuple[int, int, int] = (1, 1, 1),
        padding: tuple[int, int, int] = (0, 0, 0),
        tp_axis: int | None = None,
        device: ttnn.MeshDevice,
        ccl_manager: CCLManager | None = None,
    ) -> None:
        super().__init__()

        self.inner = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size[1:],
            stride=stride[1:],
            padding=padding[1:],
            mesh_device=device,
            ccl_manager=ccl_manager,
            mesh_axis=tp_axis,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        # remove temporal dimension and rename
        if "weight" in state:
            state["inner.weight"] = state.pop("weight")[:, :, -1, :, :]
        if "bias" in state:
            state["inner.bias"] = state.pop("bias")

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return self.inner.forward(x)


class QwenImageRmsNorm(Module):
    def __init__(self, dim: int, *, device: ttnn.MeshDevice) -> None:
        super().__init__()

        self.norm = RMSNorm(dim, norm_eps=1e-12, bias=False, mesh_device=device)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "gamma" in state:
            state["norm.weight"] = state.pop("gamma").reshape([-1])

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return self.norm.forward(x)


class QwenImageResample(Module):
    def __init__(self, *, dim: int, mode: str, device: ttnn.MeshDevice) -> None:
        super().__init__()

        if mode in {"upsample2d", "upsample3d"}:
            self.conv = Conv2d(dim, dim // 2, kernel_size=(3, 3), padding=(1, 1), mesh_device=device)
        else:
            msg = f"unsupported resample mode '{mode}'"
            raise ValueError(msg)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        pop_substate(state, "time_conv")  # only needed for temporal upsampling
        rename_substate(state, "resample.1", "conv")

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = ttnn.upsample(x, scale_factor=2)
        return self.conv.forward(x)


class QwenImageResidualBlock(Module):
    def __init__(
        self,
        *,
        in_dim: int,
        out_dim: int,
        non_linearity: str,
        device: ttnn.MeshDevice,
    ) -> None:
        super().__init__()

        assert non_linearity == "silu"
        self._nonlinearity = ttnn.silu

        self.norm1 = QwenImageRmsNorm(in_dim, device=device)
        self.conv1 = QwenImageConv(in_dim, out_dim, kernel_size=(3, 3, 3), padding=(1, 1, 1), device=device)
        self.norm2 = QwenImageRmsNorm(out_dim, device=device)
        self.conv2 = QwenImageConv(out_dim, out_dim, kernel_size=(3, 3, 3), padding=(1, 1, 1), device=device)
        self.conv_shortcut = Linear(in_dim, out_dim, mesh_device=device) if in_dim != out_dim else None

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "conv_shortcut.weight" in state:
            state["conv_shortcut.weight"] = state["conv_shortcut.weight"].flatten(1)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        h = self.conv_shortcut(x) if self.conv_shortcut is not None else x

        x = self.norm1.forward(x)
        x = self._nonlinearity(x)
        x = self.conv1.forward(x)

        x = self.norm2.forward(x)
        x = self._nonlinearity(x)
        x = self.conv2.forward(x)

        return x + h


class QwenImageAttentionBlock(Module):
    def __init__(self, *, dim: int, device: ttnn.MeshDevice) -> None:
        super().__init__()

        self.norm = QwenImageRmsNorm(dim, device=device)
        self.to_qkv = Linear(dim, dim * 3, mesh_device=device)
        self.proj = Linear(dim, dim, mesh_device=device)

        grid_size = device.compute_with_storage_grid_size()

        self._sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid_size,
            q_chunk_size=128,
            k_chunk_size=128,
            exp_approx_mode=False,
        )
        self._sdpa_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "to_qkv.weight" in state:
            state["to_qkv.weight"] = state["to_qkv.weight"].flatten(1)
        if "proj.weight" in state:
            state["proj.weight"] = state["proj.weight"].flatten(1)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        identity = x
        batch_size, height, width, channels = x.shape

        x = self.norm.forward(x)

        # convert to 1d sequence
        x = x.reshape([batch_size, height * width, channels])

        qkv = self.to_qkv.forward(x)
        qkv = ttnn.unsqueeze(qkv, 1)  # add head dimension

        # equivalent to split_query_key_value_and_split_heads with a single head
        q, k, v = ttnn.chunk(qkv, 3, dim=-1)  # batch_size, 1, height * width, head_size

        import math

        scale_factor = 1 / math.sqrt(q.shape[-1])
        w = q @ ttnn.transpose(k, -2, -1) * scale_factor
        # w = ttnn.softmax(w, dim=-1, numeric_stable=False)
        w = ttnn.exp(w)
        w = ttnn.div(w, ttnn.sum(w, dim=-1, keepdim=True))
        x = w @ v

        # TODO: this does not work:
        # x = ttnn.transformer.scaled_dot_product_attention(
        #     q,
        #     k,
        #     v,
        #     is_causal=False,
        #     program_config=self._sdpa_program_config,
        #     compute_kernel_config=self._sdpa_compute_kernel_config,
        # )

        # remove head dimension
        x = ttnn.squeeze(x, 1)

        x = self.proj.forward(x)

        # convert back to 2d
        x = x.reshape([batch_size, height, width, channels])

        return x + identity


class QwenImageMidBlock(Module):
    def __init__(
        self,
        *,
        dim: int,
        non_linearity: str,
        num_layers: int = 1,
        device: ttnn.MeshDevice,
    ) -> None:
        super().__init__()

        self.resnets = ModuleList(
            QwenImageResidualBlock(in_dim=dim, out_dim=dim, non_linearity=non_linearity, device=device)
            for _ in range(num_layers + 1)
        )
        self.attentions = ModuleList(QwenImageAttentionBlock(dim=dim, device=device) for _ in range(num_layers))

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        first_resnet, *other_resnets = self.resnets

        x = first_resnet.forward(x)

        for attn, resnet in zip(self.attentions, other_resnets, strict=True):
            x = attn.forward(x)
            x = resnet.forward(x)

        return x


class QwenImageUpBlock(Module):
    def __init__(
        self,
        *,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        upsample_mode: str | None = None,
        non_linearity: str,
        device: ttnn.MeshDevice,
    ) -> None:
        super().__init__()

        self.resnets = ModuleList([])
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            self.resnets.append(
                QwenImageResidualBlock(in_dim=current_dim, out_dim=out_dim, non_linearity=non_linearity, device=device)
            )
            current_dim = out_dim

        self.upsampler = (
            QwenImageResample(dim=out_dim, mode=upsample_mode, device=device) if upsample_mode is not None else None
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        rename_substate(state, "upsamplers.0", "upsampler")

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        for resnet in self.resnets:
            x = resnet.forward(x)

        if self.upsampler is not None:
            x = self.upsampler.forward(x)

        return x


class QwenImageVaeDecoder(Module):
    """Qwen-Image VAE decoder without support for temporal dimension."""

    def __init__(
        self,
        *,
        base_dim: int = 96,
        z_dim: int = 16,
        dim_mult: Sequence[int] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        temperal_downsample: Sequence[bool] = (False, True, True),
        non_linearity: str = "silu",
        parallel_config: VAEParallelConfig | None = None,
        device: ttnn.MeshDevice,
        ccl_manager: CCLManager | None = None,
    ) -> None:
        super().__init__()

        tp_axis = parallel_config.tensor_parallel.mesh_axis if parallel_config is not None else None

        assert non_linearity == "silu"
        self._nonlinearity = ttnn.silu

        dims = [base_dim * u for u in [dim_mult[-1], *dim_mult[::-1]]]

        self.post_quant_conv = Linear(z_dim, z_dim, mesh_device=device)
        self.conv_in = QwenImageConv(z_dim, dims[0], kernel_size=(3, 3, 3), padding=(1, 1, 1), device=device)
        self.mid_block = QwenImageMidBlock(dim=dims[0], non_linearity=non_linearity, num_layers=1, device=device)

        self.up_blocks = ModuleList([])
        for i, (in_dim, out_dim) in enumerate(itertools.pairwise(dims)):
            if i == len(dim_mult) - 1:
                upsample_mode = None
            else:
                upsample_mode = "upsample3d" if temperal_downsample[-i - 1] else "upsample2d"

            up_block = QwenImageUpBlock(
                in_dim=in_dim // 2 if i > 0 else in_dim,
                out_dim=out_dim,
                num_res_blocks=num_res_blocks,
                upsample_mode=upsample_mode,
                non_linearity=non_linearity,
                device=device,
            )
            self.up_blocks.append(up_block)

        self.norm_out = QwenImageRmsNorm(out_dim, device=device)
        self.conv_out = QwenImageConv(out_dim, 3, kernel_size=(3, 3, 3), padding=(1, 1, 1), device=device)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "post_quant_conv.weight" in state:
            state["post_quant_conv.weight"] = state["post_quant_conv.weight"].flatten(1)

        rename_substate(state, "decoder", "")

        # remove encoder state
        pop_substate(state, "quant_conv")
        pop_substate(state, "encoder")

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = self.post_quant_conv.forward(x)
        x = self.conv_in.forward(x)
        x = self.mid_block.forward(x)

        for block in self.up_blocks:
            x = block.forward(x)

        x = self.norm_out.forward(x)
        x = self._nonlinearity(x)
        x = self.conv_out.forward(x)

        return ttnn.clamp(x, min=-1.0, max=1.0)
