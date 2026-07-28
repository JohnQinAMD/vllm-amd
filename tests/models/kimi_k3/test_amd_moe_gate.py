# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from vllm._aiter_ops import rocm_aiter_ops
from vllm.models.kimi_k3.amd.ops import moe_gate
from vllm.models.kimi_k3.amd.ops.moe_gate import (
    KimiK3AiterGateLinear,
    _kimi_k3_aiter_gate_projection_fake,
    kimi_k3_aiter_gate_projection,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="Kimi-K3 AITER gate projection requires ROCm",
)


def _patch_single_rank_tensor_parallel(monkeypatch):
    for module in (
        "vllm.model_executor.layers.linear",
        "vllm.model_executor.parameter",
    ):
        monkeypatch.setattr(
            f"{module}.get_tensor_model_parallel_rank",
            lambda: 0,
        )
        monkeypatch.setattr(
            f"{module}.get_tensor_model_parallel_world_size",
            lambda: 1,
        )


@pytest.mark.parametrize("num_tokens", [1, 8, 16])
def test_kimi_k3_aiter_gate_projection_matches_gate_linear(num_tokens):
    generator = torch.Generator(device="cpu").manual_seed(20260728 + num_tokens)
    hidden_states = torch.randn(
        (num_tokens, 7168),
        generator=generator,
        dtype=torch.bfloat16,
    ).cuda()
    router_weight = (
        torch.randn(
            (896, 7168),
            generator=generator,
            dtype=torch.bfloat16,
        )
        .mul_(7168**-0.5)
        .cuda()
    )

    expected = F.linear(hidden_states, router_weight).float()
    actual = kimi_k3_aiter_gate_projection(
        hidden_states,
        router_weight,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_kimi_k3_aiter_gate_projection_fake_contract():
    hidden_states = torch.empty(
        (8, 7168),
        dtype=torch.bfloat16,
        device="meta",
    )
    router_weight = torch.empty(
        (896, 7168),
        dtype=torch.bfloat16,
        device="meta",
    )

    output = _kimi_k3_aiter_gate_projection_fake(
        hidden_states,
        router_weight,
    )

    assert output.shape == (8, 896)
    assert output.dtype == torch.float32
    assert output.device.type == "meta"


def test_kimi_k3_aiter_gate_projection_custom_op_contract():
    hidden_states = torch.randn((2, 16), dtype=torch.bfloat16, device="cuda")
    router_weight = torch.randn((8, 16), dtype=torch.bfloat16, device="cuda")

    torch.library.opcheck(
        torch.ops.vllm.kimi_k3_aiter_gate_projection,
        (hidden_states, router_weight),
        test_utils=("test_schema", "test_faketensor"),
    )


def test_kimi_k3_gate_linear_delegates_to_aiter(monkeypatch):
    _patch_single_rank_tensor_parallel(monkeypatch)
    monkeypatch.setattr(rocm_aiter_ops, "is_enabled", lambda: True)

    gate = KimiK3AiterGateLinear(
        input_size=16,
        output_size=8,
        bias=False,
        out_dtype=torch.float32,
        params_dtype=torch.bfloat16,
    ).cuda()
    hidden_states = torch.randn((1, 16), dtype=torch.bfloat16, device="cuda")
    calls = []

    def projection(hidden_states, router_weight):
        calls.append((hidden_states, router_weight))
        return torch.zeros((1, 8), dtype=torch.float32, device="cuda")

    monkeypatch.setattr(moe_gate, "kimi_k3_aiter_gate_projection", projection)

    output, bias = gate(hidden_states)

    assert bias is None
    assert output.shape == (1, 8)
    assert len(calls) == 1
    assert calls[0][0] is hidden_states
    assert calls[0][1] is gate.weight


def test_kimi_k3_gate_linear_respects_aiter_opt_out(monkeypatch):
    _patch_single_rank_tensor_parallel(monkeypatch)
    monkeypatch.setattr(rocm_aiter_ops, "is_enabled", lambda: False)

    gate = KimiK3AiterGateLinear(
        input_size=16,
        output_size=8,
        bias=False,
        out_dtype=torch.float32,
        params_dtype=torch.bfloat16,
    ).cuda()
    hidden_states = torch.randn((2, 16), dtype=torch.bfloat16, device="cuda")
    with torch.no_grad():
        gate.weight.normal_()
    monkeypatch.setattr(
        moe_gate,
        "kimi_k3_aiter_gate_projection",
        lambda *_: pytest.fail("AITER projection must honor the opt-out"),
    )

    output, bias = gate(hidden_states)

    assert bias is None
    torch.testing.assert_close(
        output,
        F.linear(hidden_states, gate.weight).float(),
        rtol=0,
        atol=0,
    )
