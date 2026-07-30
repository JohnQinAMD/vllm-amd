# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Low-concurrency Kimi-K3 decode fusion for AMD CDNA4.

The production fallback launches packed causal convolution, recurrent KDA,
an output copy, and RMSNorm with a sigmoid gate separately.  At small batches
these launches dominate device time.  This kernel assigns one complete head
to each program and fuses the chain without materializing packed QKV or the
pre-norm recurrent output.

One complete head per program is a correctness requirement, not just a tile
choice.  Independently launched V tiles consume the same pre-update Q/K
convolution state; allowing them to shift that state would introduce an
inter-program race.
"""

from __future__ import annotations

import torch

from vllm.platforms.rocm import on_gfx950
from vllm.third_party.flash_linear_attention.ops.op import exp, log
from vllm.triton_utils import tl, triton

_HEAD_DIM = 128
_MAX_FUSED_BATCH = 16
_SOFTPLUS_THRESHOLD = 20.0


def is_fused_kda_decode_supported(
    *,
    num_prefills: int,
    has_spec_decode: bool,
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor | None,
    conv_state: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
    state_indices: torch.Tensor | None,
    output_gate: torch.Tensor,
    norm_weight: torch.Tensor,
) -> bool:
    """Return whether the measured gfx950 specialization owns this input.

    Keep the complete tensor contract here so the model call site only needs
    one named predicate.  Larger batches retain the existing multi-program
    recurrent kernel: the full-head fusion crosses over between B16 and B24.
    """
    if not on_gfx950() or num_prefills != 0 or has_spec_decode:
        return False
    if state.ndim != 4:
        return False

    batch = x.shape[0] if x.ndim == 2 else 0
    _, heads, value_dim, key_dim = state.shape
    channels = 2 * heads * key_dim + heads * value_dim
    bias_supported = conv_bias is None or (
        conv_bias.shape == (channels,) and conv_bias.dtype == torch.float32
    )

    return all(
        (
            0 < batch <= _MAX_FUSED_BATCH,
            heads == 12,
            key_dim == _HEAD_DIM,
            value_dim == _HEAD_DIM,
            x.shape == (batch, channels),
            x.dtype == torch.bfloat16,
            x.stride(-1) == 1,
            conv_weight.shape == (channels, 4),
            conv_weight.dtype == torch.float32,
            conv_weight.stride(-1) == 1,
            bias_supported,
            conv_state.ndim == 3,
            conv_state.shape[1:] == (channels, 3),
            conv_state.dtype == torch.bfloat16,
            raw_g.shape == (1, batch, heads, key_dim),
            raw_g.dtype == torch.bfloat16,
            raw_g.stride(-1) == 1,
            raw_beta.shape == (1, batch, heads),
            raw_beta.dtype == torch.bfloat16,
            raw_beta.stride(-1) == 1,
            A_log.shape == (heads,),
            A_log.dtype == torch.float32,
            A_log.is_contiguous(),
            dt_bias.shape == (heads * key_dim,),
            dt_bias.dtype == torch.float32,
            dt_bias.is_contiguous(),
            state.dtype == torch.float32,
            state.stride()[1:] == (value_dim * key_dim, key_dim, 1),
            state_indices is not None,
            state_indices is not None and state_indices.shape == (batch,),
            state_indices is not None and state_indices.dtype == torch.int32,
            state_indices is not None and state_indices.stride(0) == 1,
            output_gate.shape == (batch, heads, value_dim),
            output_gate.dtype == torch.bfloat16,
            output_gate.stride(-1) == 1,
            norm_weight.shape == (value_dim,),
            norm_weight.dtype in (torch.bfloat16, torch.float32),
            norm_weight.is_contiguous(),
        )
    )


@triton.jit
def _fused_kda_decode_kernel(
    x,
    conv_weight,
    conv_bias,
    conv_state,
    raw_g,
    raw_beta,
    A_log,
    dt_bias,
    state,
    state_indices,
    output_gate,
    norm_weight,
    out,
    lower_bound,
    norm_eps,
    scale: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_weight_channel: tl.constexpr,
    stride_weight_width: tl.constexpr,
    stride_conv_state_slot,
    stride_conv_state_channel: tl.constexpr,
    stride_conv_state_width: tl.constexpr,
    stride_g_token: tl.constexpr,
    stride_beta_token: tl.constexpr,
    stride_state_slot,
    stride_output_gate_token: tl.constexpr,
    stride_output_gate_head: tl.constexpr,
    stride_out_token: tl.constexpr,
    stride_out_head: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    HAS_CONV_BIAS: tl.constexpr,
):
    token_head = tl.program_id(0)
    token, head = token_head // H, token_head % H

    offset_k = tl.arange(0, K)
    offset_v = tl.arange(0, V)
    mask_k = offset_k < K
    mask_v = offset_v < V
    mask_state = mask_v[:, None] & mask_k[None, :]

    state_idx = tl.load(state_indices + token).to(tl.int64)
    p_out = out + token * stride_out_token + head * stride_out_head + offset_v
    if state_idx <= 0:
        tl.store(p_out, tl.zeros([V], dtype=tl.float32), mask=mask_v)
        return

    q_channel = head * K + offset_k
    k_channel = H * K + head * K + offset_k
    v_channel = 2 * H * K + head * V + offset_v
    q0 = tl.load(
        conv_state
        + state_idx * stride_conv_state_slot
        + q_channel * stride_conv_state_channel,
        mask=mask_k,
        other=0,
    ).to(tl.float32)
    q1 = tl.load(
        conv_state
        + state_idx * stride_conv_state_slot
        + q_channel * stride_conv_state_channel
        + stride_conv_state_width,
        mask=mask_k,
        other=0,
    ).to(tl.float32)
    q2 = tl.load(
        conv_state
        + state_idx * stride_conv_state_slot
        + q_channel * stride_conv_state_channel
        + 2 * stride_conv_state_width,
        mask=mask_k,
        other=0,
    ).to(tl.float32)
    k0 = tl.load(
        conv_state
        + state_idx * stride_conv_state_slot
        + k_channel * stride_conv_state_channel,
        mask=mask_k,
        other=0,
    ).to(tl.float32)
    k1 = tl.load(
        conv_state
        + state_idx * stride_conv_state_slot
        + k_channel * stride_conv_state_channel
        + stride_conv_state_width,
        mask=mask_k,
        other=0,
    ).to(tl.float32)
    k2 = tl.load(
        conv_state
        + state_idx * stride_conv_state_slot
        + k_channel * stride_conv_state_channel
        + 2 * stride_conv_state_width,
        mask=mask_k,
        other=0,
    ).to(tl.float32)
    v0 = tl.load(
        conv_state
        + state_idx * stride_conv_state_slot
        + v_channel * stride_conv_state_channel,
        mask=mask_v,
        other=0,
    ).to(tl.float32)
    v1 = tl.load(
        conv_state
        + state_idx * stride_conv_state_slot
        + v_channel * stride_conv_state_channel
        + stride_conv_state_width,
        mask=mask_v,
        other=0,
    ).to(tl.float32)
    v2 = tl.load(
        conv_state
        + state_idx * stride_conv_state_slot
        + v_channel * stride_conv_state_channel
        + 2 * stride_conv_state_width,
        mask=mask_v,
        other=0,
    ).to(tl.float32)

    q3 = tl.load(x + token * stride_x_token + q_channel, mask=mask_k, other=0).to(
        tl.float32
    )
    k3 = tl.load(x + token * stride_x_token + k_channel, mask=mask_k, other=0).to(
        tl.float32
    )
    v3 = tl.load(x + token * stride_x_token + v_channel, mask=mask_v, other=0).to(
        tl.float32
    )

    qw0 = tl.load(
        conv_weight + q_channel * stride_weight_channel,
        mask=mask_k,
        other=0,
    ).to(tl.float32)
    qw1 = tl.load(
        conv_weight + q_channel * stride_weight_channel + stride_weight_width,
        mask=mask_k,
        other=0,
    ).to(tl.float32)
    qw2 = tl.load(
        conv_weight + q_channel * stride_weight_channel + 2 * stride_weight_width,
        mask=mask_k,
        other=0,
    ).to(tl.float32)
    qw3 = tl.load(
        conv_weight + q_channel * stride_weight_channel + 3 * stride_weight_width,
        mask=mask_k,
        other=0,
    ).to(tl.float32)
    kw0 = tl.load(
        conv_weight + k_channel * stride_weight_channel,
        mask=mask_k,
        other=0,
    ).to(tl.float32)
    kw1 = tl.load(
        conv_weight + k_channel * stride_weight_channel + stride_weight_width,
        mask=mask_k,
        other=0,
    ).to(tl.float32)
    kw2 = tl.load(
        conv_weight + k_channel * stride_weight_channel + 2 * stride_weight_width,
        mask=mask_k,
        other=0,
    ).to(tl.float32)
    kw3 = tl.load(
        conv_weight + k_channel * stride_weight_channel + 3 * stride_weight_width,
        mask=mask_k,
        other=0,
    ).to(tl.float32)
    vw0 = tl.load(
        conv_weight + v_channel * stride_weight_channel,
        mask=mask_v,
        other=0,
    ).to(tl.float32)
    vw1 = tl.load(
        conv_weight + v_channel * stride_weight_channel + stride_weight_width,
        mask=mask_v,
        other=0,
    ).to(tl.float32)
    vw2 = tl.load(
        conv_weight + v_channel * stride_weight_channel + 2 * stride_weight_width,
        mask=mask_v,
        other=0,
    ).to(tl.float32)
    vw3 = tl.load(
        conv_weight + v_channel * stride_weight_channel + 3 * stride_weight_width,
        mask=mask_v,
        other=0,
    ).to(tl.float32)

    q = q0 * qw0 + q1 * qw1 + q2 * qw2 + q3 * qw3
    k = k0 * kw0 + k1 * kw1 + k2 * kw2 + k3 * kw3
    v = v0 * vw0 + v1 * vw1 + v2 * vw2 + v3 * vw3
    if HAS_CONV_BIAS:
        q += tl.load(conv_bias + q_channel, mask=mask_k, other=0).to(tl.float32)
        k += tl.load(conv_bias + k_channel, mask=mask_k, other=0).to(tl.float32)
        v += tl.load(conv_bias + v_channel, mask=mask_v, other=0).to(tl.float32)
    q *= tl.sigmoid(q)
    k *= tl.sigmoid(k)
    v *= tl.sigmoid(v)
    q = q.to(x.dtype.element_ty).to(tl.float32)
    k = k.to(x.dtype.element_ty).to(tl.float32)
    v = v.to(x.dtype.element_ty).to(tl.float32)

    q_state = (
        conv_state
        + state_idx * stride_conv_state_slot
        + q_channel * stride_conv_state_channel
    )
    k_state = (
        conv_state
        + state_idx * stride_conv_state_slot
        + k_channel * stride_conv_state_channel
    )
    v_state = (
        conv_state
        + state_idx * stride_conv_state_slot
        + v_channel * stride_conv_state_channel
    )
    tl.store(q_state, q1, mask=mask_k)
    tl.store(q_state + stride_conv_state_width, q2, mask=mask_k)
    tl.store(q_state + 2 * stride_conv_state_width, q3, mask=mask_k)
    tl.store(k_state, k1, mask=mask_k)
    tl.store(k_state + stride_conv_state_width, k2, mask=mask_k)
    tl.store(k_state + 2 * stride_conv_state_width, k3, mask=mask_k)
    tl.store(v_state, v1, mask=mask_v)
    tl.store(v_state + stride_conv_state_width, v2, mask=mask_v)
    tl.store(v_state + 2 * stride_conv_state_width, v3, mask=mask_v)

    q /= tl.sqrt(tl.sum(q * q) + 1e-6)
    k /= tl.sqrt(tl.sum(k * k) + 1e-6)
    q *= scale

    p_state = state + state_idx * stride_state_slot
    p_state += head * V * K + offset_v[:, None] * K + offset_k[None, :]
    recurrent_state = tl.load(p_state, mask=mask_state, other=0).to(tl.float32)

    p_g = raw_g + token * stride_g_token + head * K + offset_k
    gate_input = tl.load(p_g, mask=mask_k, other=0).to(tl.float32)
    gate_input += tl.load(dt_bias + head * K + offset_k, mask=mask_k, other=0).to(
        tl.float32
    )
    a = exp(tl.load(A_log + head).to(tl.float32))
    if USE_LOWER_BOUND:
        gate = lower_bound * tl.sigmoid(a * gate_input)
    else:
        softplus = tl.where(
            gate_input > 20.0,
            gate_input,
            log(1.0 + tl.exp(gate_input)),
        )
        gate = -a * softplus

    recurrent_state *= exp(gate[None, :])
    v -= tl.sum(recurrent_state * k[None, :], axis=1)
    beta = tl.sigmoid(
        tl.load(raw_beta + token * stride_beta_token + head).to(tl.float32)
    )
    v *= beta
    recurrent_state += v[:, None] * k[None, :]
    result = tl.sum(recurrent_state * q[None, :], axis=1)

    # Preserve the BF16 boundary before the incumbent output norm.
    result = result.to(out.dtype.element_ty).to(tl.float32)
    variance = tl.sum(result * result, axis=0) / V
    result *= 1.0 / tl.sqrt(variance + norm_eps)
    result *= tl.load(norm_weight + offset_v, mask=mask_v, other=0).to(tl.float32)
    gate_out = tl.load(
        output_gate
        + token * stride_output_gate_token
        + head * stride_output_gate_head
        + offset_v,
        mask=mask_v,
        other=0,
    ).to(tl.float32)
    result *= tl.sigmoid(gate_out)

    tl.store(p_state, recurrent_state, mask=mask_state)
    tl.store(p_out, result, mask=mask_v)


def fused_kda_decode(
    *,
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor | None,
    conv_state: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float | None,
    state: torch.Tensor,
    state_indices: torch.Tensor,
    output_gate: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
    out: torch.Tensor,
) -> None:
    """Execute the validated gfx950 low-concurrency decode specialization."""
    _, heads, value_dim, key_dim = state.shape
    batch = x.shape[0]
    if out.shape != (1, batch, heads, value_dim):
        raise ValueError("`out` has an incompatible shape.")

    _fused_kda_decode_kernel[(batch * heads,)](
        x=x,
        conv_weight=conv_weight,
        conv_bias=conv_bias,
        conv_state=conv_state,
        raw_g=raw_g,
        raw_beta=raw_beta,
        A_log=A_log,
        dt_bias=dt_bias,
        state=state,
        state_indices=state_indices,
        output_gate=output_gate,
        norm_weight=norm_weight,
        out=out,
        lower_bound=lower_bound or 0.0,
        norm_eps=norm_eps,
        scale=key_dim**-0.5,
        stride_x_token=x.stride(0),
        stride_weight_channel=conv_weight.stride(0),
        stride_weight_width=conv_weight.stride(1),
        stride_conv_state_slot=conv_state.stride(0),
        stride_conv_state_channel=conv_state.stride(1),
        stride_conv_state_width=conv_state.stride(2),
        stride_g_token=raw_g.stride(1),
        stride_beta_token=raw_beta.stride(1),
        stride_state_slot=state.stride(0),
        stride_output_gate_token=output_gate.stride(0),
        stride_output_gate_head=output_gate.stride(1),
        stride_out_token=out.stride(1),
        stride_out_head=out.stride(2),
        H=heads,
        K=key_dim,
        V=value_dim,
        USE_LOWER_BOUND=lower_bound is not None,
        HAS_CONV_BIAS=conv_bias is not None,
        num_warps=8,
        num_stages=2,
    )
