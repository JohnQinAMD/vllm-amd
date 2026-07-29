# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Backend-neutral contracts for synchronously prepared MoE routing."""

from typing import TYPE_CHECKING, Protocol

import torch

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.model_executor.layers.fused_moe.router.fused_moe_router import (
        FusedMoERouter,
    )


class PreparedMoEMetadata(Protocol):
    """Opaque metadata exposing only its logical routing decision."""

    routing_weights: torch.Tensor
    expert_ids: torch.Tensor


class PreparedMoEBackend(Protocol):
    """Own backend-specific route preparation and synchronous consumption."""

    def prepare(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None,
        router: "FusedMoERouter",
        routed_experts: "RoutedExperts",
    ) -> PreparedMoEMetadata | None:
        """Return prepared metadata, or ``None`` for the generic path."""

    def consume(
        self,
        hidden_states: torch.Tensor,
        routed_experts: "RoutedExperts",
        metadata: PreparedMoEMetadata,
    ) -> torch.Tensor:
        """Consume metadata exactly once and return routed-expert output."""
