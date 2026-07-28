# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm._aiter_ops import rocm_aiter_ops
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExpertsOrder,
)
from vllm.model_executor.layers.quantization import mxfp4 as mxfp4_module
from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod
from vllm.models.kimi_k3.amd.linear import (
    KimiAMDLatentMoERunner,
    KimiRoutedOutputTransform,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="Kimi-K3 AITER latent-MoE tail requires ROCm",
)


def _transform() -> KimiRoutedOutputTransform:
    transform = object.__new__(KimiRoutedOutputTransform)
    nn.Module.__init__(transform)
    transform.norm = SimpleNamespace(
        weight=torch.empty(3584, device="meta"),
        variance_epsilon=1.0e-6,
    )
    transform.up_proj = SimpleNamespace(weight=torch.empty(7168, 3584, device="meta"))
    return transform


def test_forward_with_shared_delegates_to_supported_aiter_kernel(monkeypatch):
    latent_moe_tail_module = importlib.import_module("aiter.ops.flydsl.latent_moe_tail")

    transform = _transform()
    routed = torch.empty(1, 3584)
    shared = torch.empty(1, 7168)
    expected = torch.empty_like(shared)
    calls = []

    monkeypatch.setattr(rocm_aiter_ops, "is_enabled", lambda: True)
    monkeypatch.setattr(
        latent_moe_tail_module,
        "supports_latent_moe_tail",
        lambda *args: True,
    )

    def fused_tail(*args):
        calls.append(args)
        return expected

    monkeypatch.setattr(latent_moe_tail_module, "latent_moe_tail", fused_tail)

    assert transform.forward_with_shared(routed, shared) is expected
    assert len(calls) == 1
    assert calls[0][0] is routed
    assert calls[0][1] is shared
    assert calls[0][2] is transform.norm.weight
    assert calls[0][3] is transform.up_proj.weight
    assert calls[0][4] == transform.norm.variance_epsilon


def test_forward_with_shared_preserves_fallbacks(monkeypatch):
    transform = _transform()
    routed = torch.empty(8, 3584)
    shared = torch.empty(8, 7168)

    monkeypatch.setattr(rocm_aiter_ops, "is_enabled", lambda: False)
    assert transform.forward_with_shared(routed, shared) is None

    monkeypatch.setattr(rocm_aiter_ops, "is_enabled", lambda: True)
    latent_moe_tail_module = importlib.import_module("aiter.ops.flydsl.latent_moe_tail")

    monkeypatch.setattr(
        latent_moe_tail_module,
        "supports_latent_moe_tail",
        lambda *args: False,
    )
    monkeypatch.setattr(
        latent_moe_tail_module,
        "latent_moe_tail",
        lambda *args: pytest.fail("unsupported inputs must use the fallback"),
    )
    assert transform.forward_with_shared(routed, shared) is None


def test_runner_consumes_fused_tail_once_without_leaking_state(monkeypatch):
    runner = object.__new__(KimiAMDLatentMoERunner)
    nn.Module.__init__(runner)
    runner.routed_scaling_factor = 1.0
    transform = _transform()
    runner.routed_output_transform = transform

    routed = torch.tensor([[1.0]])
    shared = torch.tensor([[2.0]])
    fused_result = torch.tensor([[3.0]])

    monkeypatch.setattr(
        transform,
        "forward_with_shared",
        lambda routed, shared: fused_result,
    )
    remaining_shared, result = runner._maybe_apply_routed_scale_to_output(
        shared, routed
    )
    assert remaining_shared is None
    assert result is fused_result
    assert runner.apply_routed_output_transform(result) is fused_result

    fallback_result = torch.tensor([[4.0]])
    monkeypatch.setattr(
        transform,
        "forward_with_shared",
        lambda routed, shared: None,
    )
    monkeypatch.setattr(transform, "forward", lambda routed: fallback_result)
    remaining_shared, result = runner._maybe_apply_routed_scale_to_output(
        shared, routed
    )
    assert remaining_shared is shared
    assert result is routed
    assert runner.apply_routed_output_transform(result) is fallback_result


def test_runner_uses_typed_prepared_route_handoff_once(monkeypatch):
    handoff_module = importlib.import_module("aiter.ops.flydsl.kimi_k3_moe_handoff")
    runner = object.__new__(KimiAMDLatentMoERunner)
    nn.Module.__init__(runner)

    quant_method = object.__new__(Mxfp4MoEMethod)
    quant_method.is_k3_situ_aiter = True
    correction_bias = torch.empty(896)
    w13_weight = torch.empty(1)
    w13_weight.kimi_k3_w13_layout = "gate_up_interleaved_preshuffled"
    runner.routed_experts = SimpleNamespace(
        quant_method=quant_method,
        e_score_correction_bias=correction_bias,
        w13_weight=w13_weight,
        w2_weight=torch.empty(1),
        w13_weight_scale=torch.empty(1),
        w2_weight_scale=torch.empty(1),
    )
    runner.moe_config = SimpleNamespace(
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
    )
    shared_output = torch.tensor([[7.0]])
    shared_orders = []

    class FakeSharedExperts:
        output = shared_output

        def __call__(self, _inputs, order):
            shared_orders.append(order)

    runner._shared_experts = FakeSharedExperts()
    hidden_states = torch.empty(1, 3584)
    router_logits = torch.empty(1, 896)
    expert_output = torch.tensor([[11.0]])
    calls = []

    monkeypatch.setattr(rocm_aiter_ops, "is_enabled", lambda: True)
    monkeypatch.setattr(
        handoff_module,
        "supports_kimi_k3_mxfp4_expert_handoff",
        lambda request: True,
    )

    def handoff(request):
        calls.append(request)
        return SimpleNamespace(expert_output=expert_output)

    monkeypatch.setattr(handoff_module, "kimi_k3_mxfp4_expert_handoff", handoff)
    actual_shared, actual_expert = runner._apply_quant_method(
        hidden_states, router_logits, torch.empty_like(hidden_states)
    )

    assert actual_shared is shared_output
    assert actual_expert is expert_output
    assert len(calls) == 1
    assert calls[0].hidden_states is hidden_states
    assert calls[0].router_logits is router_logits
    assert calls[0].correction_bias is correction_bias
    assert calls[0].w1.kimi_k3_w13_layout == "gate_up_interleaved_preshuffled"
    assert shared_orders == [
        SharedExpertsOrder.NO_OVERLAP,
        SharedExpertsOrder.MULTI_STREAM_OVERLAPPED,
    ]


def test_runner_prepared_route_handoff_preserves_fallback(monkeypatch):
    runner = object.__new__(KimiAMDLatentMoERunner)
    nn.Module.__init__(runner)
    runner._make_prepared_route_request = lambda *_args: None
    expected = (torch.tensor([[1.0]]), torch.tensor([[2.0]]))
    calls = []

    def fallback(self, *args, **kwargs):
        calls.append((self, args, kwargs))
        return expected

    monkeypatch.setattr(MoERunner, "_apply_quant_method", fallback)
    actual = runner._apply_quant_method(
        torch.empty(8, 3584),
        torch.empty(8, 896),
        torch.empty(8, 3584),
    )
    assert actual is expected
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("a8w4", "expected_layout"),
    [
        ("0", "gate_up_separated_preshuffled"),
        ("1", "gate_up_interleaved_preshuffled"),
    ],
)
def test_k3_weight_loading_records_stable_w13_layout(
    monkeypatch, a8w4, expected_layout
):
    monkeypatch.setenv("AITER_SITUV2_A8W4", a8w4)
    calls = []

    def shuffle_weight(tensor, _nlane, gate_up):
        calls.append(("weight", gate_up))
        return tensor.clone()

    def shuffle_scale(tensor, _experts, gate_up):
        calls.append(("scale", gate_up))
        return tensor.clone()

    monkeypatch.setattr(rocm_aiter_ops, "shuffle_weight_a16w4", shuffle_weight)
    monkeypatch.setattr(rocm_aiter_ops, "shuffle_scale_a16w4", shuffle_scale)
    fp4_utils = importlib.import_module("aiter.utility.fp4_utils")
    monkeypatch.setattr(fp4_utils, "e8m0_shuffle", lambda tensor: tensor.clone())
    monkeypatch.setattr(
        mxfp4_module,
        "replace_parameter",
        lambda layer, name, tensor: setattr(layer, name, tensor),
    )

    layer = SimpleNamespace(
        w13_weight=torch.zeros((2, 4, 4), dtype=torch.uint8),
        w2_weight=torch.zeros((2, 4, 4), dtype=torch.uint8),
        w13_weight_scale=torch.zeros((2, 4, 1), dtype=torch.uint8),
        w2_weight_scale=torch.zeros((2, 4, 1), dtype=torch.uint8),
    )
    method = object.__new__(Mxfp4MoEMethod)
    method.experts_cls = None
    method.get_fused_moe_quant_config = lambda _layer: None
    method._setup_kernel_k3_situ(layer)

    expected_interleave = a8w4 == "1"
    assert calls == [
        ("weight", expected_interleave),
        ("weight", False),
        ("scale", expected_interleave),
    ]
    assert layer.w13_weight.kimi_k3_w13_layout == expected_layout
    assert layer.w13_weight.is_shuffled
    assert layer.w2_weight.is_shuffled
