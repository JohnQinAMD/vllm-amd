# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch import nn

import vllm.models.kimi_k3.amd.kda as amd_kda
import vllm.models.kimi_k3.amd.linear as amd_linear
from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import (
    KimiGatedDeltaNetAttention as CommonKimiGatedDeltaNetAttention,
)
from vllm.models.kimi_k3.amd.kda import KimiGatedDeltaNetAttention
from vllm.models.kimi_k3.amd.ops.kda_input_projection import (
    kda_input_projection,
)


class _Projection(nn.Module):
    def forward(self, hidden_states: torch.Tensor):
        return hidden_states + 1, None


def test_default_projection_seam_preserves_exact_result() -> None:
    layer = object.__new__(CommonKimiGatedDeltaNetAttention)
    nn.Module.__init__(layer)
    layer.in_proj_qkvgfab = _Projection()
    hidden_states = torch.arange(8, dtype=torch.float32).reshape(1, 8)

    actual = CommonKimiGatedDeltaNetAttention._project_qkvgfab(layer, hidden_states)

    assert torch.equal(actual, layer.in_proj_qkvgfab(hidden_states)[0])


def test_group64_dispatch_falls_back_without_prepacked_weight() -> None:
    hidden_states = torch.arange(8, dtype=torch.float32).reshape(1, 8)
    fallback_calls = 0

    def fallback(value: torch.Tensor) -> torch.Tensor:
        nonlocal fallback_calls
        fallback_calls += 1
        return value + 1

    actual = kda_input_projection(
        hidden_states,
        fallback,
        packed_weight=None,
        packed_scale=None,
    )

    assert fallback_calls == 1
    assert torch.equal(actual, hidden_states + 1)


def test_amd_projection_seam_preserves_fallback_result() -> None:
    layer = object.__new__(KimiGatedDeltaNetAttention)
    nn.Module.__init__(layer)
    layer.in_proj_qkvgfab = _Projection()
    layer.register_buffer("_kda_group64_weight", None, persistent=False)
    layer.register_buffer("_kda_group64_scale", None, persistent=False)
    hidden_states = torch.arange(8, dtype=torch.float32).reshape(1, 8)

    actual = KimiGatedDeltaNetAttention._project_qkvgfab(layer, hidden_states)

    assert torch.equal(actual, hidden_states + 1)


def test_finalize_installs_prepacked_buffers(monkeypatch) -> None:
    layer = object.__new__(KimiGatedDeltaNetAttention)
    nn.Module.__init__(layer)
    layer.in_proj_qkvgfab = nn.Linear(8, 8, bias=False)
    layer.register_buffer("_kda_group64_weight", None, persistent=False)
    layer.register_buffer("_kda_group64_scale", None, persistent=False)
    packed_weight = torch.empty(2, 2)
    packed_scale = torch.empty(2)
    monkeypatch.setattr(
        amd_kda,
        "prepack_kda_input_group64",
        lambda weight: (packed_weight, packed_scale),
    )

    layer.finalize_kda_group64_weight()

    assert layer._kda_group64_weight is packed_weight
    assert layer._kda_group64_scale is packed_scale


def test_model_finalizer_visits_only_amd_kda_layers(monkeypatch) -> None:
    class _FakeKDA(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def finalize_kda_group64_weight(self) -> None:
            self.calls += 1

    model = object.__new__(amd_linear.KimiLinearModel)
    nn.Module.__init__(model)
    kda_a = _FakeKDA()
    kda_b = _FakeKDA()
    model.add_module("kda_a", kda_a)
    model.add_module("other", nn.Linear(1, 1))
    model.add_module("kda_b", kda_b)
    monkeypatch.setattr(amd_linear, "KimiGatedDeltaNetAttention", _FakeKDA)

    model.finalize_kda_group64_weights()

    assert kda_a.calls == 1
    assert kda_b.calls == 1
