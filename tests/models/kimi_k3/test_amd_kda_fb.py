# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

import vllm.models.kimi_k3.amd.kda as amd_kda
from vllm.models.kimi_k3.amd.kda import (
    KimiGatedDeltaNetAttention,
    _has_fused_decode_layout,
    _pure_decode_request,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

_BATCH = 2
_HEADS = 12
_DIM = 128
_CONV_WIDTH = 4
_CHANNELS = 3 * _HEADS * _DIM


def _layout_inputs() -> dict[str, torch.Tensor]:
    return {
        "f_a": torch.empty(_BATCH, _DIM, dtype=torch.bfloat16),
        "f_b_weight": torch.empty(_HEADS * _DIM, _DIM, dtype=torch.bfloat16),
        "mixed_qkv": torch.empty(_BATCH, _CHANNELS, dtype=torch.bfloat16),
        "conv_weight": torch.empty(
            _CHANNELS,
            _CONV_WIDTH,
            dtype=torch.float32,
        ),
        "conv_state": torch.empty(
            4,
            _CHANNELS,
            _CONV_WIDTH - 1,
            dtype=torch.bfloat16,
        ),
        "raw_beta": torch.empty(1, _BATCH, _HEADS, dtype=torch.bfloat16),
        "A_log": torch.empty(_HEADS, dtype=torch.float32),
        "dt_bias": torch.empty(_HEADS * _DIM, dtype=torch.float32),
        "recurrent_state": torch.empty(
            4,
            _HEADS,
            _DIM,
            _DIM,
            dtype=torch.float32,
        ),
        "state_indices": torch.empty(_BATCH, dtype=torch.int32),
        "output_gate": torch.empty(_BATCH, _HEADS, _DIM, dtype=torch.bfloat16),
        "norm_weight": torch.empty(_DIM, dtype=torch.bfloat16),
        "out": torch.empty(1, _BATCH, _HEADS, _DIM, dtype=torch.bfloat16),
    }


def test_aiter_kda_fb_layout_accepts_measured_contract() -> None:
    assert _has_fused_decode_layout(**_layout_inputs())


def test_aiter_kda_fb_layout_accepts_default_sd_cache_view() -> None:
    inputs = _layout_inputs()
    sd_cache = torch.empty(
        4,
        _CONV_WIDTH - 1,
        _CHANNELS,
        dtype=torch.bfloat16,
    )
    inputs["conv_state"] = sd_cache.transpose(1, 2)
    assert inputs["conv_state"].stride()[1:] == (1, _CHANNELS)
    assert _has_fused_decode_layout(**inputs)


def test_aiter_kda_fb_layout_selects_fallback_for_unmeasured_input() -> None:
    inputs = _layout_inputs()
    inputs["f_b_weight"] = inputs["f_b_weight"].transpose(0, 1)
    assert not _has_fused_decode_layout(**inputs)

    inputs = _layout_inputs()
    inputs["output_gate"] = inputs["output_gate"].float()
    assert not _has_fused_decode_layout(**inputs)

    inputs = _layout_inputs()
    inputs["state_indices"] = torch.empty((), dtype=torch.int32)
    assert not _has_fused_decode_layout(**inputs)

    inputs = _layout_inputs()
    inputs["conv_state"] = torch.empty(
        4,
        _CHANNELS,
        2 * (_CONV_WIDTH - 1),
        dtype=torch.bfloat16,
    )[:, :, ::2]
    assert not _has_fused_decode_layout(**inputs)


@pytest.mark.parametrize(
    ("change", "value"),
    [
        ("num_prefills", 1),
        ("num_actual_tokens", 17),
        ("spec_sequence_masks", torch.ones(2, dtype=torch.bool)),
        ("non_spec_state_indices_tensor", None),
    ],
)
def test_aiter_kda_fb_request_selects_only_measured_decode(
    change: str,
    value: object,
) -> None:
    state_indices = torch.arange(_BATCH, dtype=torch.int32)
    metadata = GDNAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=_BATCH,
        num_decode_tokens=_BATCH,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=_BATCH,
        non_spec_state_indices_tensor=state_indices,
    )
    request = _pure_decode_request({"layer": metadata}, "layer")
    assert request is not None
    batch, actual_state_indices = request
    assert batch == _BATCH
    assert torch.equal(actual_state_indices, state_indices)

    setattr(metadata, change, value)
    assert _pure_decode_request({"layer": metadata}, "layer") is None


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm-only AITER kernel")
@pytest.mark.parametrize("conv_layout", ["DS", "SD"])
@torch.inference_mode()
def test_aiter_kda_fb_adapter_matches_unfused_api(
    monkeypatch: pytest.MonkeyPatch,
    conv_layout: str,
) -> None:
    ops = amd_kda._load_aiter_kda_fb()
    if ops is None:
        pytest.skip("stacked AITER KDA f_b API is unavailable")

    from aiter.ops.flydsl import flydsl_kimi_k3_kda_decode

    batch = 8
    slots = batch + 2
    torch.manual_seed(20260728)
    f_a = torch.randn(batch, _DIM, dtype=torch.bfloat16, device="cuda")
    f_b_weight = (
        0.05
        * torch.randn(
            _HEADS * _DIM,
            _DIM,
            dtype=torch.bfloat16,
            device="cuda",
        )
    ).to(torch.bfloat16)
    mixed_qkv = torch.randn(
        batch,
        _CHANNELS,
        dtype=torch.bfloat16,
        device="cuda",
    )
    conv_weight = 0.1 * torch.randn(
        _CHANNELS,
        _CONV_WIDTH,
        dtype=torch.float32,
        device="cuda",
    )
    conv_seed_ds = torch.randn(
        slots,
        _CHANNELS,
        _CONV_WIDTH - 1,
        dtype=torch.bfloat16,
        device="cuda",
    )
    conv_seed = (
        conv_seed_ds
        if conv_layout == "DS"
        else conv_seed_ds.transpose(1, 2).contiguous()
    )
    beta = torch.randn(1, batch, _HEADS, dtype=torch.bfloat16, device="cuda")
    A_log = torch.randn(_HEADS, dtype=torch.float32, device="cuda")
    dt_bias = torch.randn(_HEADS * _DIM, dtype=torch.float32, device="cuda")
    recurrent_seed = 0.01 * torch.randn(
        slots,
        _HEADS,
        _DIM,
        _DIM,
        dtype=torch.float32,
        device="cuda",
    )
    state_indices = torch.arange(1, batch + 1, dtype=torch.int32, device="cuda")
    output_gate = torch.randn(
        batch,
        _HEADS,
        _DIM,
        dtype=torch.bfloat16,
        device="cuda",
    )
    norm_weight = torch.randn(_DIM, dtype=torch.bfloat16, device="cuda")

    metadata = GDNAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=batch,
        num_decode_tokens=batch,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=batch,
        non_spec_state_indices_tensor=state_indices,
    )
    monkeypatch.setattr(
        amd_kda,
        "get_forward_context",
        lambda: type(
            "ForwardContext",
            (),
            {"attn_metadata": {"layer": metadata}},
        )(),
    )
    monkeypatch.setattr(
        amd_kda,
        "is_conv_state_dim_first",
        lambda: conv_layout == "DS",
    )

    layer = object.__new__(KimiGatedDeltaNetAttention)
    torch.nn.Module.__init__(layer)
    layer.prefix = "layer"
    layer.gate_lower_bound = -5.0
    layer.f_b_proj = torch.nn.Linear(_DIM, _HEADS * _DIM, bias=False).to(
        device="cuda",
        dtype=torch.bfloat16,
    )
    layer.f_b_proj.weight.copy_(f_b_weight)
    layer.conv1d = torch.nn.Conv1d(
        _CHANNELS,
        _CHANNELS,
        kernel_size=_CONV_WIDTH,
        groups=_CHANNELS,
        bias=False,
        device="cuda",
        dtype=torch.float32,
    )
    layer.conv1d.weight.copy_(conv_weight.unsqueeze(1))
    layer.A_log = torch.nn.Parameter(A_log)
    layer.dt_bias = torch.nn.Parameter(dt_bias)
    layer.o_norm = torch.nn.RMSNorm(
        _DIM,
        eps=1e-5,
        device="cuda",
        dtype=torch.bfloat16,
    )
    layer.o_norm.weight.copy_(norm_weight)

    conv_actual = conv_seed.clone()
    recurrent_actual = recurrent_seed.clone()
    layer.kv_cache = (conv_actual, recurrent_actual)
    actual = torch.empty(
        1,
        batch,
        _HEADS,
        _DIM,
        dtype=torch.bfloat16,
        device="cuda",
    )
    assert layer._try_aiter_kda_fb_decode(
        f_a=f_a,
        mixed_qkv=mixed_qkv,
        beta=beta,
        output_gate=output_gate,
        out=actual,
    )

    conv_expected_cache = conv_seed.clone()
    conv_expected = (
        conv_expected_cache
        if conv_layout == "DS"
        else conv_expected_cache.transpose(1, 2)
    )
    recurrent_expected = recurrent_seed.clone()
    raw_g = F.linear(f_a, f_b_weight).view(1, batch, _HEADS, _DIM)
    expected = flydsl_kimi_k3_kda_decode(
        x=mixed_qkv,
        conv_weight=conv_weight,
        conv_bias=None,
        conv_state=conv_expected,
        raw_g=raw_g,
        raw_beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=-5.0,
        state=recurrent_expected,
        state_indices=state_indices,
        output_gate=output_gate,
        norm_weight=norm_weight,
        norm_eps=1e-5,
    )
    torch.accelerator.synchronize()

    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(
        recurrent_actual,
        recurrent_expected,
        atol=1e-6,
        rtol=1e-3,
    )
    assert torch.equal(conv_actual, conv_expected_cache)
