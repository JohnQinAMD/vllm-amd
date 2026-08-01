# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compile-safe Kimi-K3 AMD pre-route projection dispatch."""

import torch

from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

_FP8_MAX = 448.0


def supports_kimi_k3_preroute_bf16(
    hidden_states: torch.Tensor,
    routed_weight: torch.Tensor,
    shared_gate_up_weight: torch.Tensor,
    shared_down_weight: torch.Tensor,
) -> bool:
    """Return whether AITER supports the exact-BF16 B1 projection cluster."""

    try:
        from aiter.ops.flydsl.kimi_k3_moe_preroute_projection import (
            supports_kimi_k3_moe_preroute_projection,
        )
    except (ImportError, ModuleNotFoundError):
        return False
    return supports_kimi_k3_moe_preroute_projection(
        hidden_states,
        routed_weight,
        shared_gate_up_weight,
        shared_down_weight,
    )


def _kimi_k3_preroute_bf16_impl(
    hidden_states: torch.Tensor,
    routed_weight: torch.Tensor,
    shared_gate_up_weight: torch.Tensor,
    shared_down_weight: torch.Tensor,
    situ_beta: float,
    situ_linear_beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    from aiter.ops.flydsl.kimi_k3_moe_preroute_projection import (
        kimi_k3_moe_preroute_projection,
    )

    return kimi_k3_moe_preroute_projection(
        hidden_states,
        routed_weight,
        shared_gate_up_weight,
        shared_down_weight,
        situ_beta,
        situ_linear_beta,
    )


def _kimi_k3_preroute_bf16_fake(
    hidden_states: torch.Tensor,
    routed_weight: torch.Tensor,
    shared_gate_up_weight: torch.Tensor,
    shared_down_weight: torch.Tensor,
    situ_beta: float,
    situ_linear_beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    del shared_gate_up_weight, shared_down_weight, situ_beta, situ_linear_beta
    return (
        hidden_states.new_empty((hidden_states.shape[0], routed_weight.shape[0])),
        hidden_states.new_empty((hidden_states.shape[0], hidden_states.shape[1])),
    )


direct_register_custom_op(
    op_name="kimi_k3_preroute_bf16",
    op_func=_kimi_k3_preroute_bf16_impl,
    mutates_args=[],
    fake_impl=_kimi_k3_preroute_bf16_fake,
    dispatch_key=current_platform.dispatch_key,
)


def kimi_k3_preroute_bf16(
    hidden_states: torch.Tensor,
    routed_weight: torch.Tensor,
    shared_gate_up_weight: torch.Tensor,
    shared_down_weight: torch.Tensor,
    situ_beta: float,
    situ_linear_beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the fixed gfx950 B1 exact-BF16 pre-route projection custom op."""

    return torch.ops.vllm.kimi_k3_preroute_bf16(
        hidden_states,
        routed_weight,
        shared_gate_up_weight,
        shared_down_weight,
        situ_beta,
        situ_linear_beta,
    )


def quantize_kimi_k3_preroute_weight(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a BF16 projection weight per output row to OCP FP8 E4M3."""

    if not weight.is_cuda or weight.dtype != torch.bfloat16 or weight.dim() != 2:
        raise ValueError("pre-route source weight must be a CUDA BF16 matrix")
    weight_f32 = weight.float()
    amax = weight_f32.abs().amax(dim=1)
    scale = torch.where(
        amax > 0,
        amax / _FP8_MAX,
        torch.ones_like(amax),
    )
    quantized = (
        (weight_f32 / scale[:, None])
        .clamp(min=-_FP8_MAX, max=_FP8_MAX)
        .to(torch.float8_e4m3fn)
        .contiguous()
    )
    return quantized, scale.contiguous()


def supports_kimi_k3_preroute_fp8(
    hidden_states: torch.Tensor,
    routed_weight: torch.Tensor,
    routed_scale: torch.Tensor,
    shared_gate_up_weight: torch.Tensor,
    shared_gate_up_scale: torch.Tensor,
    shared_down_weight: torch.Tensor,
    shared_down_scale: torch.Tensor,
) -> bool:
    """Fail closed unless both fixed-shape AITER primitives are available."""

    try:
        from aiter.ops.flydsl.kimi_k3_moe_dual_projection import (
            supports_kimi_k3_moe_dual_projection_fp8_weight,
        )
    except (ImportError, ModuleNotFoundError):
        return False

    tensors = (shared_down_weight, shared_down_scale)
    return supports_kimi_k3_moe_dual_projection_fp8_weight(
        hidden_states,
        routed_weight,
        routed_scale,
        shared_gate_up_weight,
        shared_gate_up_scale,
    ) and (
        shared_down_weight.is_cuda
        and shared_down_scale.is_cuda
        and shared_down_weight.dtype == torch.float8_e4m3fn
        and shared_down_scale.dtype == torch.float32
        and tuple(shared_down_weight.shape) == (7168, 768)
        and tuple(shared_down_scale.shape) == (7168,)
        and shared_down_weight.is_contiguous()
        and shared_down_scale.is_contiguous()
        and len({tensor.device for tensor in tensors} | {hidden_states.device}) == 1
    )


def _kimi_k3_preroute_fp8_impl(
    hidden_states: torch.Tensor,
    routed_weight: torch.Tensor,
    routed_scale: torch.Tensor,
    shared_gate_up_weight: torch.Tensor,
    shared_gate_up_scale: torch.Tensor,
    shared_down_weight: torch.Tensor,
    shared_down_scale: torch.Tensor,
    situ_beta: float,
    situ_linear_beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    from aiter.ops.flydsl.kimi_k3_moe_dual_projection import (
        kimi_k3_moe_dual_projection_fp8_weight,
        kimi_k3_shared_down_fp8_weight,
    )

    routed, shared_gate_up = kimi_k3_moe_dual_projection_fp8_weight(
        hidden_states,
        routed_weight,
        routed_scale,
        shared_gate_up_weight,
        shared_gate_up_scale,
    )
    shared_output = kimi_k3_shared_down_fp8_weight(
        shared_gate_up,
        shared_down_weight,
        shared_down_scale,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
    )
    return routed, shared_output


def _kimi_k3_preroute_fp8_fake(
    hidden_states: torch.Tensor,
    routed_weight: torch.Tensor,
    routed_scale: torch.Tensor,
    shared_gate_up_weight: torch.Tensor,
    shared_gate_up_scale: torch.Tensor,
    shared_down_weight: torch.Tensor,
    shared_down_scale: torch.Tensor,
    situ_beta: float,
    situ_linear_beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    del (
        routed_scale,
        shared_gate_up_weight,
        shared_gate_up_scale,
        shared_down_weight,
        shared_down_scale,
        situ_beta,
        situ_linear_beta,
    )
    return (
        hidden_states.new_empty((hidden_states.shape[0], routed_weight.shape[0])),
        hidden_states.new_empty((hidden_states.shape[0], hidden_states.shape[1])),
    )


direct_register_custom_op(
    op_name="kimi_k3_preroute_fp8",
    op_func=_kimi_k3_preroute_fp8_impl,
    mutates_args=[],
    fake_impl=_kimi_k3_preroute_fp8_fake,
    dispatch_key=current_platform.dispatch_key,
)


def kimi_k3_preroute_fp8(
    hidden_states: torch.Tensor,
    routed_weight: torch.Tensor,
    routed_scale: torch.Tensor,
    shared_gate_up_weight: torch.Tensor,
    shared_gate_up_scale: torch.Tensor,
    shared_down_weight: torch.Tensor,
    shared_down_scale: torch.Tensor,
    situ_beta: float,
    situ_linear_beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the fixed gfx950 B1 pre-route projection custom op."""

    return torch.ops.vllm.kimi_k3_preroute_fp8(
        hidden_states,
        routed_weight,
        routed_scale,
        shared_gate_up_weight,
        shared_gate_up_scale,
        shared_down_weight,
        shared_down_scale,
        situ_beta,
        situ_linear_beta,
    )


def supports_kimi_k3_preroute_fp8_tri(
    hidden_states: torch.Tensor,
    routed_weight: torch.Tensor,
    routed_scale: torch.Tensor,
    shared_gate_up_weight: torch.Tensor,
    shared_gate_up_scale: torch.Tensor,
    shared_down_weight: torch.Tensor,
    shared_down_scale: torch.Tensor,
    router_weight: torch.Tensor,
) -> bool:
    """Fail closed unless the router GEMM can also be folded into the grid."""

    try:
        from aiter.ops.flydsl.kimi_k3_moe_preroute_fp8 import (
            supports_kimi_k3_moe_tri_projection_fp8,
        )
    except (ImportError, ModuleNotFoundError):
        return False

    return supports_kimi_k3_preroute_fp8(
        hidden_states,
        routed_weight,
        routed_scale,
        shared_gate_up_weight,
        shared_gate_up_scale,
        shared_down_weight,
        shared_down_scale,
    ) and supports_kimi_k3_moe_tri_projection_fp8(
        hidden_states,
        routed_weight,
        routed_scale,
        shared_gate_up_weight,
        shared_gate_up_scale,
        router_weight,
    )


def _kimi_k3_preroute_fp8_tri_impl(
    hidden_states: torch.Tensor,
    routed_weight: torch.Tensor,
    routed_scale: torch.Tensor,
    shared_gate_up_weight: torch.Tensor,
    shared_gate_up_scale: torch.Tensor,
    shared_down_weight: torch.Tensor,
    shared_down_scale: torch.Tensor,
    router_weight: torch.Tensor,
    situ_beta: float,
    situ_linear_beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from aiter.ops.flydsl.kimi_k3_moe_dual_projection import (
        kimi_k3_shared_down_fp8_weight,
    )
    from aiter.ops.flydsl.kimi_k3_moe_preroute_fp8 import (
        kimi_k3_moe_tri_projection_fp8,
    )

    routed, shared_gate_up, router_logits = kimi_k3_moe_tri_projection_fp8(
        hidden_states,
        routed_weight,
        routed_scale,
        shared_gate_up_weight,
        shared_gate_up_scale,
        router_weight,
    )
    shared_output = kimi_k3_shared_down_fp8_weight(
        shared_gate_up,
        shared_down_weight,
        shared_down_scale,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
    )
    return routed, shared_output, router_logits


def _kimi_k3_preroute_fp8_tri_fake(
    hidden_states: torch.Tensor,
    routed_weight: torch.Tensor,
    routed_scale: torch.Tensor,
    shared_gate_up_weight: torch.Tensor,
    shared_gate_up_scale: torch.Tensor,
    shared_down_weight: torch.Tensor,
    shared_down_scale: torch.Tensor,
    router_weight: torch.Tensor,
    situ_beta: float,
    situ_linear_beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del (
        routed_scale,
        shared_gate_up_weight,
        shared_gate_up_scale,
        shared_down_weight,
        shared_down_scale,
        situ_beta,
        situ_linear_beta,
    )
    return (
        hidden_states.new_empty((hidden_states.shape[0], routed_weight.shape[0])),
        hidden_states.new_empty((hidden_states.shape[0], hidden_states.shape[1])),
        hidden_states.new_empty(
            (hidden_states.shape[0], router_weight.shape[0]),
            dtype=torch.float32,
        ),
    )


direct_register_custom_op(
    op_name="kimi_k3_preroute_fp8_tri",
    op_func=_kimi_k3_preroute_fp8_tri_impl,
    mutates_args=[],
    fake_impl=_kimi_k3_preroute_fp8_tri_fake,
    dispatch_key=current_platform.dispatch_key,
)


def kimi_k3_preroute_fp8_tri(
    hidden_states: torch.Tensor,
    routed_weight: torch.Tensor,
    routed_scale: torch.Tensor,
    shared_gate_up_weight: torch.Tensor,
    shared_gate_up_scale: torch.Tensor,
    shared_down_weight: torch.Tensor,
    shared_down_scale: torch.Tensor,
    router_weight: torch.Tensor,
    situ_beta: float,
    situ_linear_beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the B1 pre-route cluster with the router GEMM folded in."""

    return torch.ops.vllm.kimi_k3_preroute_fp8_tri(
        hidden_states,
        routed_weight,
        routed_scale,
        shared_gate_up_weight,
        shared_gate_up_scale,
        shared_down_weight,
        shared_down_scale,
        router_weight,
        situ_beta,
        situ_linear_beta,
    )
