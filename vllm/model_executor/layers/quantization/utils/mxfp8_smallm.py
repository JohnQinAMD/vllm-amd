# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode small-M MXFP8 HIP GEMM dispatch (MiniMax-M3, AMD gfx950).

Thin wrappers over the ``_rocm_C`` custom ops (``csrc/rocm/mxfp8/``): a dense
GEMV (with an MFMA crossover for M in {8,16,32,64}) and a MoE grouped GEMM. Each
carries the measured shape envelope (allowlist) and kernel-API mechanics
(fp8->uint8 view, M-tile padding) and returns ``None`` outside the supported
set so the caller falls back to the Triton ``dot_scaled`` path. gfx950-gated by
the caller; bf16 out only.
"""

import torch

# (K, N) -> set of M for which the dense GEMV beats Triton dot_scaled.
_DENSE_ALLOWLIST = {
    (6144, 2304): set(range(1, 9)),  # qkv_proj:    M in {1..8}
    (6144, 1536): set(range(1, 9)),  # mlp_gate_up: M in {1..8}
    (2048, 6144): set(range(1, 5)),  # o_proj:      M in {1..4}
    (1536, 6144): set(range(1, 5)),  # mlp_down:    M in {1..4}
}
_SUPPORTED_M_TILES = (1, 2, 4, 8, 16)  # kernel template instantiations

# (K, N) -> {M: (k_splits, n_sub)} where the MFMA crossover wins (M in {8,16,32,64}).
_MFMA_CFG = {
    (6144, 2304): {8: (4, 1), 16: (8, 1), 32: (4, 1), 64: (2, 1)},  # qkv
    (2048, 6144): {8: (1, 1), 16: (1, 1), 32: (1, 1), 64: (1, 1)},  # o_proj
    (6144, 1536): {8: (4, 1), 16: (8, 1), 32: (4, 1), 64: (2, 1)},  # mlp_gate_up
    (1536, 6144): {8: (1, 1), 16: (1, 1), 32: (1, 1), 64: (1, 1)},  # mlp_down
}
_MFMA_M_SET = frozenset({8, 16, 32, 64})

# (E, N, K, a_div, has_weight) -> [(M_routed_lo, M_routed_hi)] buckets the MoE
# kernel wins. Measured on MI355X / gfx950, E=128 / top_k=4, caller block_m=64.
_MOE_ALLOWLIST = {
    (128, 1536, 6144, 4, False): [(4, 16)],  # gemm1 (gate_up)
    (128, 6144, 768, 1, True): [(4, 8)],  # gemm2 weighted (K=768)
    (128, 6144, 768, 1, False): [(4, 8)],  # gemm2 no-combine (K=768)
}


def _as_u8(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.uint8) if t.dtype == torch.float8_e4m3fn else t


def _next_supported_m(m: int) -> int:
    for t in _SUPPORTED_M_TILES:
        if m <= t:
            return t
    return _SUPPORTED_M_TILES[-1]


def mxfp8_gemv(
    xq: torch.Tensor,  # [M, K] fp8 e4m3fn (or uint8)
    x_scale: torch.Tensor,  # [M, K//32] uint8 (E8M0)
    wq: torch.Tensor,  # [N, K] fp8 e4m3fn
    w_scale: torch.Tensor,  # [N, K//32] uint8 (E8M0)
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor | None:
    """Decode dense MXFP8 linear (X @ W.T) -> [M, N], or None if unsupported."""
    if out_dtype != torch.bfloat16:
        return None  # kernels only emit bf16
    M, K = xq.shape
    N = wq.shape[0]
    ops = torch.ops._rocm_C

    if M in _MFMA_M_SET:
        cfg = _MFMA_CFG.get((K, N))
        if cfg is not None and M in cfg:
            k_splits, n_sub = cfg[M]
            try:
                return ops.smallm_mxfp8_mfma(
                    _as_u8(xq), x_scale, _as_u8(wq), w_scale,
                    out_dtype, n_sub, k_splits,
                )
            except Exception:
                pass

    allowed = _DENSE_ALLOWLIST.get((K, N))
    if allowed is None or M not in allowed:
        return None

    block_n = 8  # locked: BLOCK_N=8 is the measured winner for M<=8
    xq_u, wq_u = _as_u8(xq), _as_u8(wq)
    m_tile = _next_supported_m(M)
    try:
        if m_tile == M:
            return ops.smallm_mxfp8_gemv(
                xq_u, x_scale, wq_u, w_scale, out_dtype, block_n
            )
        # Pad to a supported M_TILE (zero rows dropped by [:M]); deterministic
        # shape keeps cuda-graph capture happy.
        xq_pad = torch.zeros((m_tile, K), dtype=xq_u.dtype, device=xq_u.device)
        xq_pad[:M].copy_(xq_u)
        xs_pad = torch.zeros(
            (m_tile, x_scale.shape[1]), dtype=x_scale.dtype, device=x_scale.device
        )
        xs_pad[:M].copy_(x_scale)
        out_pad = ops.smallm_mxfp8_gemv(
            xq_pad, xs_pad, wq_u, w_scale, out_dtype, block_n
        )
        return out_pad[:M].contiguous()
    except Exception:
        return None


def grouped_gemm_mxfp8(
    a_q: torch.Tensor,
    a_scale: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    num_valid_tokens: int,
    top_k: int,
    block_m: int,
    out_dtype: torch.dtype,
    a_div: int,
    mul_weight_by: torch.Tensor | None = None,
    topk_ids: torch.Tensor | None = None,  # accepted for caller-signature parity
) -> torch.Tensor | None:
    """Decode MoE MXFP8 grouped GEMM (sorted_token_ids layout) -> [M_routed, N],
    or None outside the measured-win envelope."""
    M_act, K = a_q.shape
    E, N, K2 = w.shape
    M_routed = num_valid_tokens

    if K != K2 or out_dtype != torch.bfloat16:
        return None
    if K % 1024 != 0 and K != 768:  # multiple of K_PER_WARP_STEP=1024 or short-K
        return None
    if block_m % 4 != 0 or a_div not in (1, 4, 8):
        return None

    buckets = _MOE_ALLOWLIST.get((E, N, K, a_div, mul_weight_by is not None))
    if buckets is None or not any(lo <= M_routed <= hi for lo, hi in buckets):
        return None

    try:
        aq_u = _as_u8(a_q).contiguous()
        w_u = _as_u8(w)
        a_scale_c = a_scale.contiguous() if a_scale.stride(-1) != 1 else a_scale
        # zeros so 0-token tiles stay zero on the output side (matches Triton).
        out = torch.zeros((M_routed, N), dtype=out_dtype, device=a_q.device)
        wt = None
        if mul_weight_by is not None:
            wt = (
                mul_weight_by.to(torch.float32)
                if mul_weight_by.dtype != torch.float32
                else mul_weight_by
            )
            wt = wt.contiguous()
        sti = (
            sorted_token_ids
            if sorted_token_ids.dtype == torch.int32
            else sorted_token_ids.to(torch.int32)
        )
        ei = (
            expert_ids
            if expert_ids.dtype == torch.int32
            else expert_ids.to(torch.int32)
        )
        ntp = (
            num_tokens_post_padded
            if num_tokens_post_padded.dtype == torch.int32
            else num_tokens_post_padded.to(torch.int32)
        )
        torch.ops._rocm_C.smallm_mxfp8_moe_grouped_gemm(
            aq_u, a_scale_c, w_u, w_scale, sti, ei, ntp, out,
            E, N, K, int(M_routed), int(M_act), int(a_div), int(block_m), wt,
        )
        return out
    except Exception:
        return None
