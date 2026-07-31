# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import weakref
from dataclasses import dataclass
from typing import ClassVar

import torch

import vllm.envs as envs
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce_dual,
)
from vllm.platforms.rocm import on_gfx950

_HIDDEN_SIZE = 7168
_LATENT_SIZE = 3584
_MAX_NUM_TOKENS = 1
_SUPPORTED_TP_SIZE = 8


@dataclass(frozen=True)
class KimiK3LatentMoETailContract:
    tp_size: int
    device: torch.device
    dtype: torch.dtype
    hidden_size: int
    latent_size: int
    max_num_tokens: int
    rms_eps: float


class KimiK3LatentMoETailOp:
    """gfx950 Kimi-K3 latent-MoE tail with one combined TP reduction."""

    _instances: ClassVar[
        dict[KimiK3LatentMoETailContract, "KimiK3LatentMoETailOp"]
    ] = {}
    _packed_up_weights: ClassVar[
        dict[
            tuple[torch.device, int],
            tuple[
                weakref.ReferenceType[torch.Tensor],
                torch.Tensor,
                torch.Tensor,
            ],
        ]
    ] = {}

    @classmethod
    @torch.no_grad()
    def prepack_up_weight(cls, up_weight: torch.Tensor) -> None:
        """Build the FP8 representation after loading, before graph capture."""

        if not envs.VLLM_ROCM_USE_KIMI_K3_LATENT_TAIL_FP8:
            return
        if (
            not up_weight.is_cuda
            or up_weight.dtype != torch.bfloat16
            or tuple(up_weight.shape) != (_HIDDEN_SIZE, _LATENT_SIZE)
            or not up_weight.is_contiguous()
        ):
            raise RuntimeError(
                "Kimi-K3 FP8 latent tail requires a contiguous CUDA BF16 "
                "up-projection with shape [7168,3584]"
            )
        key = (up_weight.device, up_weight.data_ptr())
        existing = cls._packed_up_weights.get(key)
        if existing is not None and existing[0]() is up_weight:
            return

        from aiter.ops.flydsl.latent_moe_tail_fp8 import (
            quantize_latent_moe_tail_weight,
        )

        packed, scale = quantize_latent_moe_tail_weight(up_weight)

        def remove_stale(
            reference: weakref.ReferenceType[torch.Tensor],
        ) -> None:
            current = cls._packed_up_weights.get(key)
            if current is not None and current[0] is reference:
                cls._packed_up_weights.pop(key, None)

        cls._packed_up_weights[key] = (
            weakref.ref(up_weight, remove_stale),
            packed,
            scale,
        )

    @classmethod
    def initialize(
        cls,
        *,
        hidden_size: int,
        latent_size: int,
        dtype: torch.dtype,
        device: torch.device,
        rms_eps: float,
    ) -> "KimiK3LatentMoETailOp":
        contract = KimiK3LatentMoETailContract(
            tp_size=get_tensor_model_parallel_world_size(),
            device=torch.device(device),
            dtype=dtype,
            hidden_size=hidden_size,
            latent_size=latent_size,
            max_num_tokens=_MAX_NUM_TOKENS,
            rms_eps=float(rms_eps),
        )
        op = cls._instances.get(contract)
        if op is None:
            op = cls(contract)
            cls._instances[contract] = op
        return op

    def __init__(self, contract: KimiK3LatentMoETailContract) -> None:
        if contract.tp_size != _SUPPORTED_TP_SIZE:
            raise ValueError(
                "Kimi-K3 ROCm latent-MoE tail fusion requires "
                f"TP={_SUPPORTED_TP_SIZE}, got TP={contract.tp_size}."
            )
        if contract.device.type != "cuda":
            raise ValueError(
                "Kimi-K3 ROCm latent-MoE tail fusion requires a GPU device."
            )
        if not on_gfx950():
            raise ValueError("Kimi-K3 ROCm latent-MoE tail fusion requires gfx950.")
        if contract.dtype != torch.bfloat16:
            raise ValueError("Kimi-K3 ROCm latent-MoE tail fusion requires bfloat16.")
        if (contract.hidden_size, contract.latent_size) != (
            _HIDDEN_SIZE,
            _LATENT_SIZE,
        ):
            raise ValueError(
                "Kimi-K3 ROCm latent-MoE tail fusion requires "
                f"hidden_size={_HIDDEN_SIZE} and latent_size={_LATENT_SIZE}."
            )
        self.contract = contract

    def __call__(
        self,
        routed_output: torch.Tensor,
        shared_output: torch.Tensor,
        rms_weight: torch.Tensor,
        up_weight: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_inputs(
            routed_output,
            shared_output,
            rms_weight,
            up_weight,
        )
        routed_reduced, shared_reduced = tensor_model_parallel_all_reduce_dual(
            routed_output,
            shared_output,
        )

        if envs.VLLM_ROCM_USE_KIMI_K3_LATENT_TAIL_FP8:
            key = (up_weight.device, up_weight.data_ptr())
            packed = self._packed_up_weights.get(key)
            if packed is None or packed[0]() is not up_weight:
                raise RuntimeError(
                    "Kimi-K3 FP8 latent-tail weight was not prepacked before "
                    "decode; refusing graph-capture-time allocation"
                )
            from aiter.ops.flydsl.latent_moe_tail_fp8 import (
                latent_moe_tail_fp8,
            )

            return latent_moe_tail_fp8(
                routed_reduced,
                shared_reduced,
                rms_weight,
                packed[1],
                packed[2],
                self.contract.rms_eps,
            )

        from aiter.ops.flydsl.latent_moe_tail import latent_moe_tail

        return latent_moe_tail(
            routed_reduced,
            shared_reduced,
            rms_weight,
            up_weight,
            self.contract.rms_eps,
        )

    def _validate_inputs(
        self,
        routed_output: torch.Tensor,
        shared_output: torch.Tensor,
        rms_weight: torch.Tensor,
        up_weight: torch.Tensor,
    ) -> None:
        contract = self.contract
        num_tokens = routed_output.shape[0]
        expected_shapes = (
            (num_tokens, contract.latent_size),
            (num_tokens, contract.hidden_size),
            (contract.latent_size,),
            (contract.hidden_size, contract.latent_size),
        )
        tensors = (routed_output, shared_output, rms_weight, up_weight)
        if tuple(tensor.shape for tensor in tensors) != expected_shapes:
            raise ValueError("Unexpected Kimi-K3 ROCm latent-MoE tail tensor shapes.")
        if not 1 <= num_tokens <= contract.max_num_tokens:
            raise ValueError(
                "Kimi-K3 ROCm latent-MoE tail fusion currently supports B1."
            )
        if any(tensor.device != contract.device for tensor in tensors):
            raise ValueError("All latent-MoE tail inputs must use the same device.")
        if any(tensor.dtype != contract.dtype for tensor in tensors):
            raise ValueError("All latent-MoE tail inputs must use bfloat16.")
        if any(not tensor.is_contiguous() for tensor in tensors):
            raise ValueError("All latent-MoE tail inputs must be contiguous.")
