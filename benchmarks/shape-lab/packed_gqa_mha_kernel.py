"""One fixed packed-GQA BF16 continuation-attention kernel for gfx1151."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _packed_gqa_mha_fwd(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    stride_qs,
    stride_qh,
    stride_qd,
    stride_ks,
    stride_kh,
    stride_kd,
    stride_vs,
    stride_vh,
    stride_vd,
    stride_os,
    stride_oh,
    stride_od,
    SEQLEN_Q: tl.constexpr,
    SEQLEN_K: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    QUERY_HEADS_PER_PROGRAM: tl.constexpr,
    QUERY_ROWS_PER_HEAD: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SM_SCALE: tl.constexpr,
):
    query_block = tl.program_id(0)
    query_head_group = tl.program_id(1)
    gqa_ratio: tl.constexpr = NUM_Q_HEADS // NUM_KV_HEADS
    groups_per_kv_head: tl.constexpr = gqa_ratio // QUERY_HEADS_PER_PROGRAM
    rows_per_program: tl.constexpr = QUERY_HEADS_PER_PROGRAM * QUERY_ROWS_PER_HEAD

    flat_row = tl.arange(0, rows_per_program)
    query_head = (
        query_head_group * QUERY_HEADS_PER_PROGRAM
        + flat_row // QUERY_ROWS_PER_HEAD
    )
    query_row = (
        query_block * QUERY_ROWS_PER_HEAD
        + flat_row % QUERY_ROWS_PER_HEAD
    )
    kv_head = query_head_group // groups_per_kv_head
    offs_d = tl.arange(0, HEAD_DIM)
    offs_n = tl.arange(0, BLOCK_N)

    q_offsets = (
        query_row[:, None] * stride_qs
        + query_head[:, None] * stride_qh
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(q_ptr + q_offsets)
    m_i = tl.full([rows_per_program], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([rows_per_program], dtype=tl.float32)
    acc = tl.zeros([rows_per_program, HEAD_DIM], dtype=tl.float32)
    log2e: tl.constexpr = 1.4426950408889634

    for start_n in tl.range(0, SEQLEN_K, BLOCK_N, num_stages=1):
        key_row = start_n + offs_n
        k_offsets = (
            kv_head * stride_kh
            + offs_d[:, None] * stride_kd
            + key_row[None, :] * stride_ks
        )
        k = tl.load(k_ptr + k_offsets)
        qk = tl.dot(q, k) * (SM_SCALE * log2e)
        causal_limit = query_row + (SEQLEN_K - SEQLEN_Q)
        qk = tl.where(
            key_row[None, :] <= causal_limit[:, None], qk, float("-inf")
        )

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.math.exp2(qk - m_ij[:, None])
        alpha = tl.math.exp2(m_i - m_ij)
        acc *= alpha[:, None]
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_ij

        v_offsets = (
            kv_head * stride_vh
            + key_row[:, None] * stride_vs
            + offs_d[None, :] * stride_vd
        )
        v = tl.load(v_ptr + v_offsets)
        acc += tl.dot(p.to(v.type.element_ty), v)

    acc /= l_i[:, None]
    out_offsets = (
        query_row[:, None] * stride_os
        + query_head[:, None] * stride_oh
        + offs_d[None, :] * stride_od
    )
    tl.store(out_ptr + out_offsets, acc)


def packed_gqa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    *,
    softmax_scale: float,
) -> torch.Tensor:
    """Launch the sole fixed BSHD Hq16/Hkv2/Sq8192/Sk16384/D256 arm."""
    expected_q = (1, 8192, 16, 256)
    expected_kv = (1, 16384, 2, 256)
    if tuple(q.shape) != expected_q or tuple(out.shape) != expected_q:
        raise ValueError(f"q/out shape mismatch: {tuple(q.shape)} / {tuple(out.shape)}")
    if tuple(k.shape) != expected_kv or tuple(v.shape) != expected_kv:
        raise ValueError(f"k/v shape mismatch: {tuple(k.shape)} / {tuple(v.shape)}")
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError(f"BF16 inputs required: {q.dtype}, {k.dtype}, {v.dtype}")
    if out.dtype != torch.bfloat16:
        raise ValueError(f"BF16 output required: {out.dtype}")
    if not all(t.is_contiguous() for t in (q, k, v, out)):
        raise ValueError("all tensors must be contiguous BSHD")

    query_heads_per_program = 8
    query_rows_per_head = 16
    block_n = 32
    grid = (triton.cdiv(q.shape[1], query_rows_per_head), q.shape[2] // query_heads_per_program)
    _packed_gqa_mha_fwd[grid](
        q,
        k,
        v,
        out,
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        SEQLEN_Q=8192,
        SEQLEN_K=16384,
        NUM_Q_HEADS=16,
        NUM_KV_HEADS=2,
        HEAD_DIM=256,
        QUERY_HEADS_PER_PROGRAM=query_heads_per_program,
        QUERY_ROWS_PER_HEAD=query_rows_per_head,
        BLOCK_N=block_n,
        SM_SCALE=softmax_scale,
        num_warps=8,
        num_stages=1,
        waves_per_eu=1,
    )
    return out
