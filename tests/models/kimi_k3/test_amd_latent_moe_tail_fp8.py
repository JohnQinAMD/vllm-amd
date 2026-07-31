# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
import weakref
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

import vllm.envs as envs
from vllm.models.kimi_k3.amd import linear
from vllm.models.kimi_k3.amd.ops import latent_moe_tail
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="Kimi-K3 FP8 latent-MoE tail requires ROCm",
)


def test_fp8_dispatch_requires_prepacked_weight(monkeypatch):
    op = object.__new__(latent_moe_tail.KimiK3LatentMoETailOp)
    op.contract = SimpleNamespace(rms_eps=1.0e-6)
    op._packed_up_weights.clear()
    monkeypatch.setattr(
        envs,
        "VLLM_ROCM_USE_KIMI_K3_LATENT_TAIL_FP8",
        True,
    )
    monkeypatch.setattr(op, "_validate_inputs", lambda *args: None)
    monkeypatch.setattr(
        latent_moe_tail,
        "tensor_model_parallel_all_reduce_dual",
        lambda routed, shared: (routed, shared),
    )
    tensors = [torch.empty(1, dtype=torch.bfloat16) for _ in range(4)]

    with pytest.raises(RuntimeError, match="was not prepacked"):
        op(*tensors)


def test_fp8_dispatch_uses_identity_checked_packed_weight(monkeypatch):
    op = object.__new__(latent_moe_tail.KimiK3LatentMoETailOp)
    op.contract = SimpleNamespace(rms_eps=1.0e-6)
    op._packed_up_weights.clear()
    monkeypatch.setattr(
        envs,
        "VLLM_ROCM_USE_KIMI_K3_LATENT_TAIL_FP8",
        True,
    )
    monkeypatch.setattr(op, "_validate_inputs", lambda *args: None)
    monkeypatch.setattr(
        latent_moe_tail,
        "tensor_model_parallel_all_reduce_dual",
        lambda routed, shared: (routed, shared),
    )

    routed, shared, rms_weight, up_weight = [
        torch.empty(1, dtype=torch.bfloat16) for _ in range(4)
    ]
    packed = torch.empty(1, dtype=torch.float8_e4m3fn)
    scale = torch.empty(1, dtype=torch.float32)
    key = (up_weight.device, up_weight.data_ptr())
    op._packed_up_weights[key] = (weakref.ref(up_weight), packed, scale)

    expected = torch.empty(1)
    calls = []
    fake_module = ModuleType("aiter.ops.flydsl.latent_moe_tail_fp8")

    def fake_tail(*args):
        calls.append(args)
        return expected

    fake_module.latent_moe_tail_fp8 = fake_tail
    monkeypatch.setitem(
        sys.modules,
        "aiter.ops.flydsl.latent_moe_tail_fp8",
        fake_module,
    )

    assert op(routed, shared, rms_weight, up_weight) is expected
    assert calls == [
        (
            routed,
            shared,
            rms_weight,
            packed,
            scale,
            op.contract.rms_eps,
        )
    ]


def test_model_finalizer_requires_exact_latent_layer_count(monkeypatch):
    model = object.__new__(linear.KimiLinearModel)
    nn.Module.__init__(model)
    model.layers = nn.ModuleList()
    for _ in range(91):
        layer = object.__new__(linear.KimiMoE)
        nn.Module.__init__(layer)
        layer.use_latent_moe = True
        model.layers.append(layer)

    monkeypatch.setattr(
        envs,
        "VLLM_ROCM_USE_KIMI_K3_LATENT_TAIL_FP8",
        True,
    )
    monkeypatch.setattr(linear, "get_tensor_model_parallel_world_size", lambda: 8)

    with pytest.raises(RuntimeError, match="weight count drift: 91 != 92"):
        model.finalize_latent_tail_fp8_weights()


def test_model_finalizer_packs_all_layers_before_capture(monkeypatch):
    model = object.__new__(linear.KimiLinearModel)
    nn.Module.__init__(model)
    model.layers = nn.ModuleList()
    for _ in range(92):
        layer = object.__new__(linear.KimiMoE)
        nn.Module.__init__(layer)
        layer.use_latent_moe = True
        model.layers.append(layer)

    packed = []
    monkeypatch.setattr(
        envs,
        "VLLM_ROCM_USE_KIMI_K3_LATENT_TAIL_FP8",
        True,
    )
    monkeypatch.setattr(linear, "get_tensor_model_parallel_world_size", lambda: 8)
    monkeypatch.setattr(
        linear.KimiMoE,
        "finalize_latent_tail_fp8_weight",
        lambda self: packed.append(self),
    )
    synchronized = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: synchronized.append(True))

    model.finalize_latent_tail_fp8_weights()

    assert len(packed) == 92
    assert synchronized == [True]
