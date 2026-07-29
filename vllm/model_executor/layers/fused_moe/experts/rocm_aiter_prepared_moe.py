# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""AITER adapter for a synchronously prepared MoE boundary."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.fused_moe.router.fused_moe_router import (
    FusedMoERouter,
)
from vllm.model_executor.layers.fused_moe.runner.prepared_moe import (
    PreparedMoEMetadata,
)

if TYPE_CHECKING:
    from aiter.ops.flydsl.kimi_k3_persistent_moe import (
        KimiK3PersistentMoEMetadata,
        KimiK3PersistentMoERequest,
    )


@dataclass(frozen=True)
class _AiterPreparedMoEMetadata:
    request: "KimiK3PersistentMoERequest"
    backend_metadata: "KimiK3PersistentMoEMetadata"
    routing_weights: torch.Tensor
    expert_ids: torch.Tensor


class AiterPreparedMoEBackend:
    """Translate generic layer ownership into AITER's typed contract."""

    @staticmethod
    def _make_request(
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None,
        router: FusedMoERouter,
        routed_experts: RoutedExperts,
    ):
        from aiter.ops.flydsl.kimi_k3_persistent_moe import (
            KimiK3PersistentMoERequest,
        )

        quant_method = routed_experts.quant_method
        config = routed_experts.moe_config
        parallel = config.moe_parallel_config
        return KimiK3PersistentMoERequest(
            hidden_states=hidden_states,
            router_logits=router_logits,
            correction_bias=getattr(
                routed_experts,
                "e_score_correction_bias",
                None,
            ),
            w1=getattr(routed_experts, "w13_weight", None),
            w2=getattr(routed_experts, "w2_weight", None),
            w1_scale=getattr(routed_experts, "w13_weight_scale", None),
            w2_scale=getattr(routed_experts, "w2_weight_scale", None),
            situ_beta=float(config.activation_situ_beta or 0.0),
            situ_linear_beta=float(config.activation_situ_linear_beta or 0.0),
            w13_layout=getattr(quant_method, "kimi_k3_w13_layout", None),
            weights_shuffled=bool(
                getattr(quant_method, "kimi_k3_weights_shuffled", False)
            ),
            quantization_supported=bool(
                getattr(quant_method, "kimi_k3_persistent_moe_supported", False)
                or getattr(quant_method, "is_k3_situ_aiter", False)
            ),
            activation=config.activation.value,
            num_experts=config.num_experts,
            topk=config.experts_per_token,
            num_expert_group=routed_experts.num_expert_group or 0,
            topk_group=routed_experts.topk_group or 0,
            renormalize=routed_experts.renormalize,
            scoring_func=routed_experts.scoring_func,
            routed_scaling_factor=routed_experts.routed_scaling_factor,
            expert_parallel=parallel.use_ep,
            eplb_enabled=parallel.enable_eplb,
            lora_enabled=config.is_lora_enabled,
            has_expert_bias=config.has_bias,
            apply_router_weight_on_input=routed_experts.apply_router_weight_on_input,
            expert_map_active=routed_experts.expert_map is not None,
            routing_capture_enabled=(
                getattr(router, "capture_fn", None) is not None
                or getattr(router, "_routing_replay_out", None) is not None
            ),
            custom_routing_active=(routed_experts.custom_routing_function is not None),
            input_ids_active=input_ids is not None,
            routing_method=config.routing_method.name,
        )

    def prepare(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None,
        router: FusedMoERouter,
        routed_experts: RoutedExperts,
    ) -> PreparedMoEMetadata | None:
        from aiter.ops.flydsl.kimi_k3_persistent_moe import (
            prepare_kimi_k3_b1_persistent_moe,
        )

        request = self._make_request(
            hidden_states,
            router_logits,
            input_ids,
            router,
            routed_experts,
        )
        metadata = prepare_kimi_k3_b1_persistent_moe(request)
        if metadata is None:
            return None
        return _AiterPreparedMoEMetadata(
            request=request,
            backend_metadata=metadata,
            routing_weights=metadata.routing_weights,
            expert_ids=metadata.expert_ids,
        )

    def consume(
        self,
        hidden_states: torch.Tensor,
        routed_experts: RoutedExperts,
        metadata: PreparedMoEMetadata,
    ) -> torch.Tensor:
        del hidden_states, routed_experts
        if not isinstance(metadata, _AiterPreparedMoEMetadata):
            raise TypeError("AITER received metadata owned by another backend")

        from aiter.ops.flydsl.kimi_k3_persistent_moe import (
            consume_kimi_k3_b1_persistent_moe,
        )

        return consume_kimi_k3_b1_persistent_moe(
            metadata.request,
            metadata.backend_metadata,
        )


def make_aiter_prepared_moe_backend() -> AiterPreparedMoEBackend:
    """Create an independent per-layer prepared-routing owner."""

    return AiterPreparedMoEBackend()


__all__ = ["AiterPreparedMoEBackend", "make_aiter_prepared_moe_backend"]
