# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Kimi-K3 AMD router projection dispatch."""

import torch
from torch.nn.parameter import Parameter

from vllm._aiter_ops import rocm_aiter_ops
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op


def _kimi_k3_aiter_gate_projection_impl(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
) -> torch.Tensor:
    """Use the narrow AITER kernel when available, preserving the fallback."""

    try:
        from aiter.ops.flydsl.kimi_k3_gate import (
            kimi_k3_b1_gate_projection,
            supports_kimi_k3_b1_gate_projection,
        )
    except (ImportError, ModuleNotFoundError):
        pass
    else:
        if supports_kimi_k3_b1_gate_projection(hidden_states, router_weight):
            return kimi_k3_b1_gate_projection(hidden_states, router_weight)

    projected = torch.nn.functional.linear(
        hidden_states.to(router_weight.dtype),
        router_weight,
    )
    return projected.to(torch.float32)


def _kimi_k3_aiter_gate_projection_fake(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
) -> torch.Tensor:
    return hidden_states.new_empty(
        (hidden_states.shape[0], router_weight.shape[0]),
        dtype=torch.float32,
    )


direct_register_custom_op(
    op_name="kimi_k3_aiter_gate_projection",
    op_func=_kimi_k3_aiter_gate_projection_impl,
    mutates_args=[],
    fake_impl=_kimi_k3_aiter_gate_projection_fake,
    dispatch_key=current_platform.dispatch_key,
)


def kimi_k3_aiter_gate_projection(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
) -> torch.Tensor:
    """Run the compile-safe Kimi-K3 gate projection custom op."""

    return torch.ops.vllm.kimi_k3_aiter_gate_projection(
        hidden_states,
        router_weight,
    )


class KimiK3AiterGateLinear(GateLinear):
    """GateLinear with an optional AITER Kimi-K3 projection backend."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        out_dtype: torch.dtype | None = None,
        params_dtype: torch.dtype | None = None,
        force_fp32_compute: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            bias=bias,
            out_dtype=out_dtype,
            params_dtype=params_dtype,
            force_fp32_compute=force_fp32_compute,
            prefix=prefix,
        )
        self._use_aiter_projection = rocm_aiter_ops.is_enabled()

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        if not self._use_aiter_projection:
            return super().forward(hidden_states)

        router_logits = kimi_k3_aiter_gate_projection(
            hidden_states,
            self.weight,
        )
        return router_logits, None
