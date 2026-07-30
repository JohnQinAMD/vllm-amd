# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This file contains code adapted from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li

from __future__ import annotations

import torch

from vllm.platforms.rocm import on_gfx950
from vllm.triton_utils import tl, triton

_KIMI_K3_MAX_ATTN_RES_BLOCKS = 8


@triton.jit
def _attn_res_kernel(
    prefix_ptr,
    delta_ptr,
    blocks_ptr,
    norm_weight_ptr,
    qk_weight_ptr,
    output_norm_weight_ptr,
    output_ptr,
    stride_prefix_m: tl.constexpr,
    stride_delta_m: tl.constexpr,
    stride_block_m: tl.constexpr,
    stride_block_r: tl.constexpr,
    stride_output_m: tl.constexpr,
    num_blocks: tl.constexpr,
    hidden_size: tl.constexpr,
    block_write_idx: tl.constexpr,
    eps: tl.constexpr,
    output_norm_eps: tl.constexpr,
    HAS_DELTA: tl.constexpr,
    WRITE_BLOCK: tl.constexpr,
    APPLY_OUTPUT_NORM: tl.constexpr,
    SPLIT_PREFIX: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    d_offsets = tl.max_contiguous(tl.arange(0, BLOCK_D), BLOCK_D)
    d_mask = d_offsets < hidden_size

    updated_prefix = tl.load(
        prefix_ptr + row_idx * stride_prefix_m + d_offsets,
        mask=d_mask,
        other=0.0,
    ).to(tl.float32)
    if HAS_DELTA:
        delta = tl.load(
            delta_ptr + row_idx * stride_delta_m + d_offsets,
            mask=d_mask,
            other=0.0,
        ).to(tl.float32)
        updated_prefix += delta
        updated_prefix = updated_prefix.to(prefix_ptr.dtype.element_ty).to(tl.float32)
        tl.store(
            prefix_ptr + row_idx * stride_prefix_m + d_offsets,
            updated_prefix,
            mask=d_mask,
        )
    if WRITE_BLOCK:
        tl.store(
            blocks_ptr
            + row_idx * stride_block_m
            + block_write_idx * stride_block_r
            + d_offsets,
            updated_prefix,
            mask=d_mask,
        )

    if num_blocks == 0:
        mixed = updated_prefix
    else:
        input_qk_weight = tl.load(
            norm_weight_ptr + d_offsets,
            mask=d_mask,
            other=0.0,
        ).to(tl.float32) * tl.load(
            qk_weight_ptr + d_offsets,
            mask=d_mask,
            other=0.0,
        ).to(tl.float32)
        source_offsets = tl.arange(0, BLOCK_L)
        block_ptrs = (
            blocks_ptr
            + row_idx * stride_block_m
            + source_offsets[:, None] * stride_block_r
            + d_offsets[None, :]
        )
        if SPLIT_PREFIX:
            source_mask = source_offsets < num_blocks
            block_values = tl.load(
                block_ptrs,
                mask=source_mask[:, None] & d_mask[None, :],
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            block_reciprocal_std = tl.rsqrt(
                tl.sum(block_values * block_values, axis=1) * (1.0 / hidden_size) + eps
            )
            block_logits = (
                tl.sum(block_values * input_qk_weight[None, :], axis=1)
                * block_reciprocal_std
            )
            block_scores = tl.where(
                source_mask,
                block_logits,
                -float("inf"),
            )
            prefix_reciprocal_std = tl.rsqrt(
                tl.sum(updated_prefix * updated_prefix, axis=0) * (1.0 / hidden_size)
                + eps
            )
            prefix_score = (
                tl.sum(updated_prefix * input_qk_weight, axis=0) * prefix_reciprocal_std
            )
            max_score = tl.maximum(
                tl.max(block_scores, axis=0),
                prefix_score,
            )
            block_scales = tl.exp(block_scores - max_score)
            prefix_scale = tl.exp(prefix_score - max_score)
            denominator = tl.sum(block_scales, axis=0) + prefix_scale
            mixed = (
                tl.sum(block_scales[:, None] * block_values, axis=0)
                + prefix_scale * updated_prefix
            ) / denominator
        else:
            num_sources = num_blocks + 1
            source_mask = source_offsets < num_sources
            is_prefix = source_offsets == num_blocks
            block_values = tl.load(
                block_ptrs,
                mask=(source_mask[:, None] & ~is_prefix[:, None] & d_mask[None, :]),
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            values = tl.where(
                is_prefix[:, None],
                updated_prefix[None, :],
                block_values,
            )
            reciprocal_std = tl.rsqrt(
                tl.sum(values * values, axis=1) * (1.0 / hidden_size) + eps
            )
            logits = tl.sum(values * input_qk_weight[None, :], axis=1) * reciprocal_std
            scores = tl.where(source_mask, logits, -float("inf"))
            scores -= tl.max(scores, axis=0)
            source_scales = tl.exp(scores)
            mixed = tl.sum(
                source_scales[:, None] * values,
                axis=0,
            ) / tl.sum(source_scales, axis=0)

    # Preserve the unfused AMD path's BF16 materialization before RMSNorm.
    mixed = mixed.to(output_ptr.dtype.element_ty).to(tl.float32)
    output = mixed
    if APPLY_OUTPUT_NORM:
        output_reciprocal_std = tl.rsqrt(
            tl.sum(tl.where(d_mask, mixed * mixed, 0.0), axis=0) * (1.0 / hidden_size)
            + output_norm_eps
        )
        output_norm_weight = tl.load(
            output_norm_weight_ptr + d_offsets,
            mask=d_mask,
            other=0.0,
        ).to(tl.float32)
        output = mixed * output_reciprocal_std * output_norm_weight
    tl.store(
        output_ptr + row_idx * stride_output_m + d_offsets,
        output,
        mask=d_mask,
    )


def _triton_attn_res(
    prefix: torch.Tensor,
    delta: torch.Tensor | None,
    blocks: torch.Tensor,
    norm_weight: torch.Tensor,
    qk_weight: torch.Tensor,
    output_norm_weight: torch.Tensor | None,
    num_blocks: int,
    block_write_idx: int,
    eps: float,
    output_norm_eps: float,
) -> torch.Tensor:
    num_tokens, hidden_size = prefix.shape
    assert 0 <= num_blocks <= _KIMI_K3_MAX_ATTN_RES_BLOCKS
    assert num_blocks <= blocks.shape[1]
    assert blocks.shape[0] == num_tokens
    assert delta is None or delta.shape == prefix.shape
    assert norm_weight.numel() == hidden_size
    assert qk_weight.numel() == hidden_size
    assert output_norm_weight is None or output_norm_weight.numel() == hidden_size
    assert -1 <= block_write_idx < blocks.shape[1]
    assert prefix.stride(-1) == 1
    assert delta is None or delta.stride(-1) == 1
    assert blocks.stride(-1) == 1
    assert norm_weight.stride(-1) == 1
    assert qk_weight.stride(-1) == 1
    assert output_norm_weight is None or output_norm_weight.stride(-1) == 1
    output = prefix.new_empty(prefix.shape)
    if num_tokens == 0:
        return output
    if not on_gfx950():
        # Keep every source in the single tile outside the architecture used
        # for this tuning campaign. The split-prefix profiles below have only
        # been measured on gfx950.
        split_prefix = False
        block_l = triton.next_power_of_2(num_blocks + 1)
        num_warps = 4 if num_tokens >= 256 or num_blocks <= 1 else 8
        num_stages = 2
        launch_options = {}
    elif num_blocks == 0:
        split_prefix = False
        block_l, num_warps = 1, 8
        num_stages = 1
        launch_options = {"waves_per_eu": 1}
    else:
        # A separate prefix avoids doubling the source tile when the active
        # block count is already a power of two.
        split_prefix = num_blocks > 1 and (num_blocks & (num_blocks - 1) == 0)
        tile_sources = num_blocks if split_prefix else num_blocks + 1
        block_l = triton.next_power_of_2(tile_sources)
        num_warps = 8 if num_blocks >= 6 else 4
        num_stages = 1
        launch_options = {"waves_per_eu": 1}
    _attn_res_kernel[(num_tokens,)](
        prefix,
        delta,
        blocks,
        norm_weight,
        qk_weight,
        output_norm_weight,
        output,
        prefix.stride(0),
        0 if delta is None else delta.stride(0),
        blocks.stride(0),
        blocks.stride(1),
        output.stride(0),
        num_blocks,
        hidden_size,
        block_write_idx,
        eps,
        output_norm_eps,
        HAS_DELTA=delta is not None,
        WRITE_BLOCK=block_write_idx >= 0,
        APPLY_OUTPUT_NORM=output_norm_weight is not None,
        SPLIT_PREFIX=split_prefix,
        BLOCK_L=block_l,
        BLOCK_D=triton.next_power_of_2(hidden_size),
        num_warps=num_warps,
        num_stages=num_stages,
        **launch_options,
    )
    return output


def attn_res(
    prefix: torch.Tensor,
    delta: torch.Tensor | None,
    blocks: torch.Tensor,
    norm_weight: torch.Tensor,
    qk_weight: torch.Tensor,
    output_norm_weight: torch.Tensor | None,
    num_blocks: int,
    block_write_idx: int,
    eps: float,
    output_norm_eps: float,
) -> torch.Tensor:
    """Apply the Kimi-K3 AttnRes transition and optional following RMSNorm."""
    return _triton_attn_res(
        prefix,
        delta,
        blocks,
        norm_weight,
        qk_weight,
        output_norm_weight,
        num_blocks,
        block_write_idx,
        eps,
        output_norm_eps,
    )
