"""Persistent D256 online attention for the two valid-q256 final partial chunks."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


QUERY_HEADS = 16
KV_HEADS = 2
HEAD_DIM = 256
QUERY_ROWS_PER_TILE = 64
BLOCK_N = 32
PERSISTENT_PROGRAMS = 320
SUPPORTED_SHAPES = {(7168, 261120), (7680, 261632)}


@triton.jit
def _partial_persistent_tilequeue_d256_fwd(
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
    QUERY_ROWS_PER_TILE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TOTAL_QUERY_TILES: tl.constexpr,
    PERSISTENT_PROGRAMS: tl.constexpr,
    SM_SCALE: tl.constexpr,
):
    persistent_id = tl.program_id(0)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_n = tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, QUERY_ROWS_PER_TILE)
    gqa_ratio: tl.constexpr = NUM_Q_HEADS // NUM_KV_HEADS
    log2e: tl.constexpr = 1.4426950408889634

    for tile_id in tl.range(
        persistent_id,
        TOTAL_QUERY_TILES,
        PERSISTENT_PROGRAMS,
        num_stages=1,
    ):
        query_head = tile_id % NUM_Q_HEADS
        query_block = tile_id // NUM_Q_HEADS
        kv_head = query_head // gqa_ratio
        query_row = query_block * QUERY_ROWS_PER_TILE + offs_m
        q_offsets = (
            query_row[:, None] * stride_qs
            + query_head * stride_qh
            + offs_d[None, :] * stride_qd
        )
        q = tl.load(q_ptr + q_offsets)
        m_i = tl.full([QUERY_ROWS_PER_TILE], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([QUERY_ROWS_PER_TILE], dtype=tl.float32)
        acc = tl.zeros([QUERY_ROWS_PER_TILE, HEAD_DIM], dtype=tl.float32)

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
                key_row[None, :] <= causal_limit[:, None],
                qk,
                float("-inf"),
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
            value = tl.load(v_ptr + v_offsets)
            acc += tl.dot(p.to(value.type.element_ty), value)

        output = acc / l_i[:, None]
        output_offsets = (
            query_row[:, None] * stride_os
            + query_head * stride_oh
            + offs_d[None, :] * stride_od
        )
        tl.store(out_ptr + output_offsets, output)


def partial_persistent_tilequeue_d256_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Launch only for q256 final-partial lower-right causal attention."""
    if k.ndim != 4 or v.ndim != 4 or tuple(k.shape) != tuple(v.shape):
        raise ValueError(f"matching rank-4 K/V required: {tuple(k.shape)} / {tuple(v.shape)}")
    query_tokens = int(q.shape[1]) if q.ndim == 4 else -1
    kv_tokens = int(k.shape[1])
    if (query_tokens, kv_tokens) not in SUPPORTED_SHAPES:
        raise ValueError(f"unsupported partial shape: q{query_tokens}/kv{kv_tokens}")
    expected_q = (1, query_tokens, QUERY_HEADS, HEAD_DIM)
    expected_kv = (1, kv_tokens, KV_HEADS, HEAD_DIM)
    if tuple(q.shape) != expected_q or tuple(out.shape) != expected_q:
        raise ValueError(f"q/out shape mismatch: {tuple(q.shape)} / {tuple(out.shape)}")
    if tuple(k.shape) != expected_kv:
        raise ValueError(f"K/V shape mismatch: {tuple(k.shape)}")
    if any(t.dtype != torch.bfloat16 for t in (q, k, v, out)):
        raise ValueError("BF16 q/k/v/out required")
    if not all(t.is_contiguous() for t in (q, k, v, out)):
        raise ValueError("contiguous direct-seq BSHD tensors required")

    total_query_tiles = QUERY_HEADS * (query_tokens // QUERY_ROWS_PER_TILE)
    _partial_persistent_tilequeue_d256_fwd[(PERSISTENT_PROGRAMS,)](
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
        SEQLEN_Q=query_tokens,
        SEQLEN_K=kv_tokens,
        NUM_Q_HEADS=QUERY_HEADS,
        NUM_KV_HEADS=KV_HEADS,
        HEAD_DIM=HEAD_DIM,
        QUERY_ROWS_PER_TILE=QUERY_ROWS_PER_TILE,
        BLOCK_N=BLOCK_N,
        TOTAL_QUERY_TILES=total_query_tiles,
        PERSISTENT_PROGRAMS=PERSISTENT_PROGRAMS,
        SM_SCALE=1.0 / math.sqrt(HEAD_DIM),
        num_warps=8,
        num_stages=1,
        waves_per_eu=1,
    )
    return out
