# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_prepared_moe import (
    AiterPreparedMoEBackend,
)
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner


def _runner(backend, router):
    runner = object.__new__(MoERunner)
    nn.Module.__init__(runner)
    runner.prepared_moe_backend = backend
    runner.router = router
    runner._shared_experts = None
    runner.routed_experts = SimpleNamespace(
        quant_method=SimpleNamespace(
            is_monolithic=False,
            topk_indices_dtype=torch.int32,
        )
    )
    return runner


def test_prepared_metadata_is_consumed_without_rerouting():
    metadata = SimpleNamespace(
        routing_weights=torch.empty(1, 16),
        expert_ids=torch.empty(1, 16, dtype=torch.int32),
    )
    expected = torch.empty(1, 3584)
    calls = []

    class Backend:
        def prepare(self, hidden, logits, input_ids, router, experts):
            calls.append(("prepare", hidden, logits, input_ids, router, experts))
            return metadata

        def consume(self, hidden, experts, prepared):
            calls.append(("consume", hidden, experts, prepared))
            return expected

    router = SimpleNamespace(
        eplb_state=None,
        select_experts=lambda **_kwargs: pytest.fail(
            "prepared routing must not run grouped top-k"
        ),
    )
    runner = _runner(Backend(), router)
    runner.routed_experts.forward_modular = lambda **_kwargs: pytest.fail(
        "prepared routing must not enter the generic expert path"
    )
    hidden = torch.empty(1, 3584)
    logits = torch.empty(1, 896)

    shared, actual = runner._apply_quant_method(hidden, logits, None)

    assert shared is None
    assert actual is expected
    assert [call[0] for call in calls] == ["prepare", "consume"]
    assert calls[0][1] is hidden
    assert calls[0][2] is logits
    assert calls[1][3] is metadata


def test_backend_decline_preserves_generic_path():
    topk_weights = torch.empty(2, 16)
    topk_ids = torch.empty(2, 16, dtype=torch.int32)
    expected = torch.empty(2, 3584)
    calls = []

    class Backend:
        def prepare(self, *_args):
            calls.append("prepare")
            return None

        def consume(self, *_args):
            pytest.fail("declined inputs must not consume prepared metadata")

    def select_experts(**_kwargs):
        calls.append("route")
        return topk_weights, topk_ids

    runner = _runner(
        Backend(),
        SimpleNamespace(eplb_state=None, select_experts=select_experts),
    )

    def forward_modular(**kwargs):
        calls.append("experts")
        assert kwargs["topk_weights"] is topk_weights
        assert kwargs["topk_ids"] is topk_ids
        return expected

    runner.routed_experts.forward_modular = forward_modular
    _, actual = runner._apply_quant_method(
        torch.empty(2, 3584),
        torch.empty(2, 896),
        None,
    )

    assert actual is expected
    assert calls == ["prepare", "route", "experts"]


def test_eplb_skips_prepared_backend():
    topk_weights = torch.empty(1, 16)
    topk_ids = torch.empty(1, 16, dtype=torch.int32)
    expected = torch.empty(1, 3584)
    backend = SimpleNamespace(
        prepare=lambda *_args: pytest.fail(
            "EPLB must skip prepared routing before backend work"
        ),
        consume=lambda *_args: pytest.fail("EPLB metadata must not be consumed"),
    )

    def select_experts(**_kwargs):
        return topk_weights, topk_ids

    runner = _runner(
        backend,
        SimpleNamespace(eplb_state=object(), select_experts=select_experts),
    )
    runner.routed_experts.forward_modular = lambda **_kwargs: expected

    _, actual = runner._apply_quant_method(
        torch.empty(1, 3584),
        torch.empty(1, 896),
        None,
    )

    assert actual is expected


def test_aiter_request_preserves_kimi_k3_native_weight_layout(monkeypatch):
    class Request:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "aiter.ops.flydsl.kimi_k3_persistent_moe",
        SimpleNamespace(KimiK3PersistentMoERequest=Request),
    )
    hidden = torch.empty(1, 3584)
    logits = torch.empty(1, 896)
    correction_bias = torch.empty(896)
    w1 = torch.empty(896, 768, 1792, dtype=torch.uint8, device="meta")
    w2 = torch.empty(896, 3584, 192, dtype=torch.uint8, device="meta")
    w1_scale = torch.empty(688128, 112, dtype=torch.uint8, device="meta")
    w2_scale = torch.empty(3211264, 16, dtype=torch.uint8, device="meta")
    quant_method = SimpleNamespace(
        kimi_k3_w13_layout="gate_up_interleaved_preshuffled",
        kimi_k3_weights_shuffled=True,
        kimi_k3_persistent_moe_supported=True,
    )
    config = SimpleNamespace(
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
        activation=SimpleNamespace(value="situ"),
        num_experts=896,
        experts_per_token=16,
        moe_parallel_config=SimpleNamespace(use_ep=False, enable_eplb=False),
        is_lora_enabled=False,
        has_bias=False,
        routing_method=SimpleNamespace(name="DeepSeekV3"),
    )
    experts = SimpleNamespace(
        quant_method=quant_method,
        moe_config=config,
        e_score_correction_bias=correction_bias,
        w13_weight=w1,
        w2_weight=w2,
        w13_weight_scale=w1_scale,
        w2_weight_scale=w2_scale,
        num_expert_group=1,
        topk_group=1,
        renormalize=True,
        scoring_func="sigmoid",
        routed_scaling_factor=1.0,
        apply_router_weight_on_input=False,
        expert_map=None,
        custom_routing_function=None,
    )
    router = SimpleNamespace(capture_fn=None, _routing_replay_out=None)

    request = AiterPreparedMoEBackend._make_request(
        hidden,
        logits,
        None,
        router,
        experts,
    )

    assert request.hidden_states is hidden
    assert request.router_logits is logits
    assert request.correction_bias is correction_bias
    assert request.w1 is w1
    assert request.w2 is w2
    assert request.w1_scale is w1_scale
    assert request.w2_scale is w2_scale
    assert request.quantization_supported
    assert request.w13_layout == "gate_up_interleaved_preshuffled"
    assert request.weights_shuffled
