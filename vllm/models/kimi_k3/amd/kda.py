# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""AMD-specific Kimi-K3 KDA integration."""

from __future__ import annotations

import functools
from collections.abc import Callable

import torch
from einops import rearrange

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import (
    KimiGatedDeltaNetAttention as _KimiGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    is_conv_state_dim_first,
)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

from .ops.kda_input_projection import (
    kda_input_projection,
    prepack_kda_input_group64,
)

_HEADS = 12
_DIM = 128
_CONV_WIDTH = 4
_CHANNELS = 3 * _HEADS * _DIM
_MAX_FUSED_BATCH = 16

_FusedDecode = Callable[..., torch.Tensor]
_SupportPredicate = Callable[[torch.device | str | int | None], bool]


@functools.cache
def _load_aiter_kda_fb() -> tuple[_FusedDecode, _SupportPredicate] | None:
    """Load the stacked AITER API without making it a vLLM import dependency."""
    try:
        from aiter.ops.flydsl import (
            flydsl_kimi_k3_kda_decode_with_f_b,
            is_flydsl_kimi_k3_kda_decode_supported,
        )
    except ImportError:
        return None
    return (
        flydsl_kimi_k3_kda_decode_with_f_b,
        is_flydsl_kimi_k3_kda_decode_supported,
    )


def _matches_tensor(
    tensor: torch.Tensor,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    inner_strides: tuple[int, ...] = (),
) -> bool:
    """Match a fixed tensor contract without indexing malformed inputs."""
    return (
        tensor.shape == shape
        and tensor.dtype == dtype
        and (
            not inner_strides or tensor.stride()[-len(inner_strides) :] == inner_strides
        )
    )


def _matches_cache(
    tensor: torch.Tensor,
    *,
    trailing_shape: tuple[int, ...],
    dtype: torch.dtype,
    inner_strides: tuple[int, ...],
) -> bool:
    """Match a cache with an arbitrary slot count and fixed slot layout."""
    return (
        tensor.ndim == len(trailing_shape) + 1
        and tensor.shape[1:] == trailing_shape
        and tensor.dtype == dtype
        and tensor.stride()[-len(inner_strides) :] == inner_strides
    )


def _matches_conv_state(tensor: torch.Tensor) -> bool:
    """Match either canonical vLLM convolution-cache layout.

    The AITER kernel consumes explicit strides. A ``DS`` cache is already
    contiguous as ``[slot, channel, width]``; vLLM's default ``SD`` cache
    becomes a strided view with channel contiguous after the required
    transpose.
    """
    return (
        tensor.ndim == 3
        and tensor.shape[1:] == (_CHANNELS, _CONV_WIDTH - 1)
        and tensor.dtype == torch.bfloat16
        and tensor.stride()[1:]
        in (
            (_CONV_WIDTH - 1, 1),
            (1, _CHANNELS),
        )
    )


def _has_fused_decode_layout(
    *,
    f_a: torch.Tensor,
    f_b_weight: torch.Tensor,
    mixed_qkv: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_state: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    recurrent_state: torch.Tensor,
    state_indices: torch.Tensor,
    output_gate: torch.Tensor,
    norm_weight: torch.Tensor,
    out: torch.Tensor,
) -> bool:
    """Check the measured Kimi-K3 TP8 BF16 tensor contract."""
    batch = f_a.shape[0] if f_a.ndim == 2 else 0
    channels = _CHANNELS
    device = f_a.device
    tensors = (
        f_b_weight,
        mixed_qkv,
        conv_weight,
        conv_state,
        raw_beta,
        A_log,
        dt_bias,
        recurrent_state,
        state_indices,
        output_gate,
        norm_weight,
        out,
    )
    return all(
        (
            0 < batch <= _MAX_FUSED_BATCH,
            all(tensor.device == device for tensor in tensors),
            _matches_tensor(
                f_a,
                shape=(batch, _DIM),
                dtype=torch.bfloat16,
                inner_strides=(1,),
            ),
            _matches_tensor(
                f_b_weight,
                shape=(_HEADS * _DIM, _DIM),
                dtype=torch.bfloat16,
                inner_strides=(_DIM, 1),
            ),
            _matches_tensor(
                mixed_qkv,
                shape=(batch, channels),
                dtype=torch.bfloat16,
                inner_strides=(1,),
            ),
            _matches_tensor(
                conv_weight,
                shape=(channels, _CONV_WIDTH),
                dtype=torch.float32,
                inner_strides=(1,),
            ),
            _matches_conv_state(conv_state),
            _matches_tensor(
                raw_beta,
                shape=(1, batch, _HEADS),
                dtype=torch.bfloat16,
                inner_strides=(1,),
            ),
            _matches_tensor(
                A_log,
                shape=(_HEADS,),
                dtype=torch.float32,
                inner_strides=(1,),
            ),
            _matches_tensor(
                dt_bias,
                shape=(_HEADS * _DIM,),
                dtype=torch.float32,
                inner_strides=(1,),
            ),
            _matches_cache(
                recurrent_state,
                trailing_shape=(_HEADS, _DIM, _DIM),
                dtype=torch.float32,
                inner_strides=(_DIM * _DIM, _DIM, 1),
            ),
            _matches_tensor(
                state_indices,
                shape=(batch,),
                dtype=torch.int32,
                inner_strides=(1,),
            ),
            _matches_tensor(
                output_gate,
                shape=(batch, _HEADS, _DIM),
                dtype=torch.bfloat16,
                inner_strides=(1,),
            ),
            _matches_tensor(
                norm_weight,
                shape=(_DIM,),
                dtype=torch.bfloat16,
                inner_strides=(1,),
            ),
            _matches_tensor(
                out,
                shape=(1, batch, _HEADS, _DIM),
                dtype=torch.bfloat16,
                inner_strides=(1,),
            ),
        )
    )


def _pure_decode_request(
    metadata_by_layer: object,
    prefix: str,
) -> tuple[int, torch.Tensor] | None:
    """Return the measured pure-decode batch, or select the fallback."""
    if not isinstance(metadata_by_layer, dict):
        return None
    metadata = metadata_by_layer.get(prefix)
    if not isinstance(metadata, GDNAttentionMetadata):
        return None

    batch = metadata.num_actual_tokens
    state_indices = metadata.non_spec_state_indices_tensor
    supported = all(
        (
            metadata.num_prefills == 0,
            metadata.spec_sequence_masks is None,
            state_indices is not None,
            0 < batch <= _MAX_FUSED_BATCH,
        )
    )
    return (batch, state_indices) if supported and state_indices is not None else None


class KimiGatedDeltaNetAttention(_KimiGatedDeltaNetAttention):
    """Kimi GDN layer with an AMD-only AITER pure-decode fast path."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.register_buffer("_kda_group64_weight", None, persistent=False)
        self.register_buffer("_kda_group64_scale", None, persistent=False)

    def finalize_kda_group64_weight(self) -> None:
        """Prepack the projection only after all checkpoint shards load."""

        weight = getattr(self.in_proj_qkvgfab, "weight", None)
        if not isinstance(weight, torch.Tensor):
            return
        packed = prepack_kda_input_group64(weight)
        if packed is not None:
            self._kda_group64_weight, self._kda_group64_scale = packed

    def _project_qkvgfab(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return kda_input_projection(
            hidden_states,
            super()._project_qkvgfab,
            self._kda_group64_weight,
            self._kda_group64_scale,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        num_tokens = hidden_states.size(0)
        projected_qkvgfab = self._project_qkvgfab(hidden_states)
        if self.use_full_rank_gate:
            split_sizes = [
                3 * self.local_projection_size,
                self.local_projection_size,
                self.head_dim,
                self.local_num_heads,
            ]
            if self.in_proj_padding:
                split_sizes.append(self.in_proj_padding)
            projected = projected_qkvgfab.split(split_sizes, dim=-1)
            mixed_qkv, g_proj_states, f_a, beta = projected[:4]
        else:
            mixed_qkv, beta, f_a = projected_qkvgfab.split(
                [
                    3 * self.local_projection_size,
                    self.local_num_heads,
                    self.head_dim,
                ],
                dim=-1,
            )
            g_proj_states = self.g_b_proj(self.g_a_proj(hidden_states)[0])[0]

        beta = beta.unsqueeze(0)
        g2 = rearrange(g_proj_states, "... (h d) -> ... h d", d=self.head_dim)
        core_attn_out = torch.empty(
            (1, num_tokens, self.local_num_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        if not self._try_aiter_kda_fb_decode(
            f_a=f_a,
            mixed_qkv=mixed_qkv,
            beta=beta,
            output_gate=g2,
            out=core_attn_out,
        ):
            g1 = self.f_b_proj(f_a)[0]
            g1 = rearrange(g1, "n (h d) -> 1 n h d", d=self.head_dim)
            self._forward(
                mixed_qkv=mixed_qkv,
                g1=g1,
                g2=g2,
                beta=beta,
                core_attn_out=core_attn_out,
            )

        core_attn_out = rearrange(core_attn_out, "1 n h d -> n (h d)")
        self.o_proj(core_attn_out, output=output)

    @eager_break_during_capture
    def _try_aiter_kda_fb_decode(
        self,
        *,
        f_a: torch.Tensor,
        mixed_qkv: torch.Tensor,
        beta: torch.Tensor,
        output_gate: torch.Tensor,
        out: torch.Tensor,
    ) -> bool:
        """Run the measured AITER specialization, or leave fallback untouched."""
        ops = _load_aiter_kda_fb()
        forward_context = get_forward_context()
        request = _pure_decode_request(forward_context.attn_metadata, self.prefix)
        f_b_weight = getattr(self.f_b_proj, "weight", None)
        if (
            ops is None
            or request is None
            or self.gate_lower_bound is None
            or not isinstance(f_b_weight, torch.Tensor)
        ):
            return False

        batch, state_indices = request
        fused_decode, is_supported = ops
        f_a = f_a[:batch]
        mixed_qkv = mixed_qkv[:batch]
        beta = beta[:, :batch]
        output_gate = output_gate[:batch]
        state_indices = state_indices[:batch]
        out = out[:, :batch]

        conv_state, recurrent_state = self.kv_cache
        if not is_conv_state_dim_first():
            conv_state = conv_state.transpose(-1, -2)
        conv_weight = self.conv1d.weight.view(
            self.conv1d.weight.size(0),
            self.conv1d.weight.size(2),
        )

        if not is_supported(f_a.device) or not _has_fused_decode_layout(
            f_a=f_a,
            f_b_weight=f_b_weight,
            mixed_qkv=mixed_qkv,
            conv_weight=conv_weight,
            conv_state=conv_state,
            raw_beta=beta,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            recurrent_state=recurrent_state,
            state_indices=state_indices,
            output_gate=output_gate,
            norm_weight=self.o_norm.weight,
            out=out,
        ):
            return False

        fused_decode(
            f_a=f_a,
            f_b_weight=f_b_weight.view(_HEADS, _DIM, _DIM),
            x=mixed_qkv,
            conv_weight=conv_weight,
            conv_bias=self.conv1d.bias,
            conv_state=conv_state,
            raw_beta=beta,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            lower_bound=self.gate_lower_bound,
            state=recurrent_state,
            state_indices=state_indices,
            output_gate=output_gate,
            norm_weight=self.o_norm.weight,
            norm_eps=self.o_norm.eps,
            out=out,
        )
        return True


__all__ = ["KimiGatedDeltaNetAttention"]
