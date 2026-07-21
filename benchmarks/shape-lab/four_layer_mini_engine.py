#!/usr/bin/env python3
"""Four-layer mini-engine skeleton for Phase 3 shape-lab work."""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from q8192_compound_provider import (
    q8192_compound_provider,
    q8192_compound_provider_release,
    q8192_compound_provider_stats,
)

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

if triton is not None and tl is not None:

    @triton.jit
    def triton_fla_fused_chunk_h_o_kernel(
        q,
        k,
        w,
        u,
        g,
        h0,
        out,
        ht,
        scale,
        T: tl.constexpr,
        H: tl.constexpr,
        Hg: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BT: tl.constexpr,
        NT: tl.constexpr,
        BV: tl.constexpr,
    ) -> None:
        i_v = tl.program_id(0)
        i_h = tl.program_id(1)
        i_qh = i_h // (H // Hg)

        offs_t = tl.arange(0, BT)
        offs_v = i_v * BV + tl.arange(0, BV)
        offs_k = tl.arange(0, 64)
        mask_v = offs_v < V

        h0_1 = h0 + (i_h * V + offs_v[:, None]) * K + offs_k[None, :]
        h0_2 = h0 + (i_h * V + offs_v[:, None]) * K + (64 + offs_k[None, :])
        mask_h1 = mask_v[:, None] & (offs_k[None, :] < K)
        mask_h2 = mask_v[:, None] & ((64 + offs_k[None, :]) < K)
        b_h1 = tl.load(h0_1, mask=mask_h1, other=0.0).to(tl.float32)
        b_h2 = tl.load(h0_2, mask=mask_h2, other=0.0).to(tl.float32)

        for i_t in range(NT):
            t = i_t * BT + offs_t
            mask_t = t < T

            p_w1 = w + (t[:, None] * H + i_h) * K + offs_k[None, :]
            p_w2 = w + (t[:, None] * H + i_h) * K + (64 + offs_k[None, :])
            b_w1 = tl.load(p_w1, mask=mask_t[:, None] & (offs_k[None, :] < K), other=0.0)
            b_w2 = tl.load(
                p_w2,
                mask=mask_t[:, None] & ((64 + offs_k[None, :]) < K),
                other=0.0,
            )
            p_u = u + (t[:, None] * H + i_h) * V + offs_v[None, :]
            b_u = tl.load(p_u, mask=mask_t[:, None] & mask_v[None, :], other=0.0)

            b_v_delta = tl.dot(b_w1, tl.trans(b_h1).to(tl.bfloat16))
            b_v_delta += tl.dot(b_w2, tl.trans(b_h2).to(tl.bfloat16))
            b_v_new = b_u - b_v_delta

            p_q1 = q + (t[:, None] * Hg + i_qh) * K + offs_k[None, :]
            p_q2 = q + (t[:, None] * Hg + i_qh) * K + (64 + offs_k[None, :])
            b_q1 = tl.load(p_q1, mask=mask_t[:, None] & (offs_k[None, :] < K), other=0.0)
            b_q2 = tl.load(
                p_q2,
                mask=mask_t[:, None] & ((64 + offs_k[None, :]) < K),
                other=0.0,
            )
            b_o = tl.dot(b_q1, tl.trans(b_h1).to(tl.bfloat16))
            b_o += tl.dot(b_q2, tl.trans(b_h2).to(tl.bfloat16))

            p_k1_t = k + (t[None, :] * Hg + i_qh) * K + offs_k[:, None]
            p_k2_t = k + (t[None, :] * Hg + i_qh) * K + (64 + offs_k[:, None])
            b_k1_t = tl.load(p_k1_t, mask=(offs_k[:, None] < K) & mask_t[None, :], other=0.0)
            b_k2_t = tl.load(
                p_k2_t,
                mask=((64 + offs_k[:, None]) < K) & mask_t[None, :],
                other=0.0,
            )
            b_a = tl.dot(b_q1, b_k1_t)
            b_a += tl.dot(b_q2, b_k2_t)

            p_g = g + t * H + i_h
            b_g = tl.load(p_g, mask=mask_t, other=0.0)
            b_o *= tl.exp(b_g)[:, None]
            b_a *= tl.exp(b_g[:, None] - b_g[None, :])
            mask_a = (t[:, None] >= t[None, :]) & mask_t[:, None] & mask_t[None, :]
            b_a = tl.where(mask_a, b_a, 0.0)

            b_v_for_o = b_v_new.to(tl.bfloat16)
            b_o = b_o * scale + tl.dot(b_a.to(tl.bfloat16), b_v_for_o) * scale
            p_o = out + (t[:, None] * H + i_h) * V + offs_v[None, :]
            tl.store(p_o, b_o.to(tl.bfloat16), mask=mask_t[:, None] & mask_v[None, :])

            last_idx = min((i_t + 1) * BT, T) - 1
            b_g_last = tl.load(g + last_idx * H + i_h)
            b_v_update = b_v_new * tl.where(mask_t, tl.exp(b_g_last - b_g), 0.0)[:, None]
            b_g_last_exp = tl.exp(b_g_last)
            b_h1 *= b_g_last_exp
            b_h2 *= b_g_last_exp
            b_v_update = b_v_update.to(tl.bfloat16)
            b_h1 += tl.trans(tl.dot(b_k1_t, b_v_update))
            b_h2 += tl.trans(tl.dot(b_k2_t, b_v_update))

        p_ht1 = ht + (i_h * V + offs_v[:, None]) * K + offs_k[None, :]
        p_ht2 = ht + (i_h * V + offs_v[:, None]) * K + (64 + offs_k[None, :])
        tl.store(p_ht1, b_h1, mask=mask_h1)
        tl.store(p_ht2, b_h2, mask=mask_h2)

    @triton.jit
    def triton_fla_recompute_w_only_kernel(
        k,
        beta,
        w,
        A,
        g,
        T: tl.constexpr,
        H: tl.constexpr,
        Hg: tl.constexpr,
        K: tl.constexpr,
        BT: tl.constexpr,
        BK: tl.constexpr,
    ) -> None:
        i_t = tl.program_id(0)
        i_h = tl.program_id(1)
        i_qh = i_h // (H // Hg)

        offs_t = i_t * BT + tl.arange(0, BT)
        offs_bt = tl.arange(0, BT)
        mask_t = offs_t < T

        b_beta = tl.load(beta + offs_t * H + i_h, mask=mask_t, other=0.0)
        b_g = tl.exp(tl.load(g + offs_t * H + i_h, mask=mask_t, other=0.0))
        b_A = tl.load(
            A + ((offs_t[:, None] * H + i_h) * BT + offs_bt[None, :]),
            mask=mask_t[:, None],
            other=0.0,
        )

        for i_k in range(tl.cdiv(K, BK)):
            offs_k = i_k * BK + tl.arange(0, BK)
            mask_k = offs_k < K
            b_k = tl.load(
                k + (offs_t[:, None] * Hg + i_qh) * K + offs_k[None, :],
                mask=mask_t[:, None] & mask_k[None, :],
                other=0.0,
            )
            b_kb = (b_k * b_beta[:, None] * b_g[:, None]).to(b_k.dtype)
            b_w = tl.dot(b_A, b_kb)
            tl.store(
                w + (offs_t[:, None] * H + i_h) * K + offs_k[None, :],
                b_w.to(tl.bfloat16),
                mask=mask_t[:, None] & mask_k[None, :],
            )

    @triton.jit
    def triton_fla_fused_u_h_o_kernel(
        q,
        k,
        v,
        beta,
        w,
        A,
        g,
        h0,
        out,
        ht,
        scale,
        T: tl.constexpr,
        H: tl.constexpr,
        Hg: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BT: tl.constexpr,
        NT: tl.constexpr,
        BV: tl.constexpr,
    ) -> None:
        i_v = tl.program_id(0)
        i_h = tl.program_id(1)
        i_qh = i_h // (H // Hg)

        offs_t = tl.arange(0, BT)
        offs_bt = tl.arange(0, BT)
        offs_v = i_v * BV + tl.arange(0, BV)
        offs_k = tl.arange(0, 64)
        mask_v = offs_v < V

        h0_1 = h0 + (i_h * V + offs_v[:, None]) * K + offs_k[None, :]
        h0_2 = h0 + (i_h * V + offs_v[:, None]) * K + (64 + offs_k[None, :])
        mask_h1 = mask_v[:, None] & (offs_k[None, :] < K)
        mask_h2 = mask_v[:, None] & ((64 + offs_k[None, :]) < K)
        b_h1 = tl.load(h0_1, mask=mask_h1, other=0.0).to(tl.float32)
        b_h2 = tl.load(h0_2, mask=mask_h2, other=0.0).to(tl.float32)

        for i_t in range(NT):
            t = i_t * BT + offs_t
            mask_t = t < T

            b_beta = tl.load(beta + t * H + i_h, mask=mask_t, other=0.0)
            b_A = tl.load(
                A + ((t[:, None] * H + i_h) * BT + offs_bt[None, :]),
                mask=mask_t[:, None],
                other=0.0,
            )
            b_v = tl.load(
                v + (t[:, None] * H + i_h) * V + offs_v[None, :],
                mask=mask_t[:, None] & mask_v[None, :],
                other=0.0,
            )
            b_vb = (b_v * b_beta[:, None]).to(b_v.dtype)
            b_u = tl.dot(b_A, b_vb, allow_tf32=False).to(tl.bfloat16)

            p_w1 = w + (t[:, None] * H + i_h) * K + offs_k[None, :]
            p_w2 = w + (t[:, None] * H + i_h) * K + (64 + offs_k[None, :])
            b_w1 = tl.load(p_w1, mask=mask_t[:, None] & (offs_k[None, :] < K), other=0.0)
            b_w2 = tl.load(
                p_w2,
                mask=mask_t[:, None] & ((64 + offs_k[None, :]) < K),
                other=0.0,
            )
            b_v_delta = tl.dot(b_w1, tl.trans(b_h1).to(b_w1.dtype))
            b_v_delta += tl.dot(b_w2, tl.trans(b_h2).to(b_w2.dtype))
            b_v_new = b_u - b_v_delta

            p_q1 = q + (t[:, None] * Hg + i_qh) * K + offs_k[None, :]
            p_q2 = q + (t[:, None] * Hg + i_qh) * K + (64 + offs_k[None, :])
            b_q1 = tl.load(p_q1, mask=mask_t[:, None] & (offs_k[None, :] < K), other=0.0)
            b_q2 = tl.load(
                p_q2,
                mask=mask_t[:, None] & ((64 + offs_k[None, :]) < K),
                other=0.0,
            )
            b_o = tl.dot(b_q1, tl.trans(b_h1).to(tl.bfloat16))
            b_o += tl.dot(b_q2, tl.trans(b_h2).to(tl.bfloat16))

            p_k1_t = k + (t[None, :] * Hg + i_qh) * K + offs_k[:, None]
            p_k2_t = k + (t[None, :] * Hg + i_qh) * K + (64 + offs_k[:, None])
            b_k1_t = tl.load(p_k1_t, mask=(offs_k[:, None] < K) & mask_t[None, :], other=0.0)
            b_k2_t = tl.load(
                p_k2_t,
                mask=((64 + offs_k[:, None]) < K) & mask_t[None, :],
                other=0.0,
            )
            b_a = tl.dot(b_q1, b_k1_t)
            b_a += tl.dot(b_q2, b_k2_t)

            p_g = g + t * H + i_h
            b_g = tl.load(p_g, mask=mask_t, other=0.0)
            b_o *= tl.exp(b_g)[:, None]
            b_a *= tl.exp(b_g[:, None] - b_g[None, :])
            mask_a = (t[:, None] >= t[None, :]) & mask_t[:, None] & mask_t[None, :]
            b_a = tl.where(mask_a, b_a, 0.0)

            b_v_for_o = b_v_new.to(tl.bfloat16)
            b_o = b_o * scale + tl.dot(b_a.to(tl.bfloat16), b_v_for_o) * scale
            p_o = out + (t[:, None] * H + i_h) * V + offs_v[None, :]
            tl.store(p_o, b_o.to(tl.bfloat16), mask=mask_t[:, None] & mask_v[None, :])

            last_idx = min((i_t + 1) * BT, T) - 1
            b_g_last = tl.load(g + last_idx * H + i_h)
            b_v_update = b_v_new * tl.where(mask_t, tl.exp(b_g_last - b_g), 0.0)[:, None]
            b_g_last_exp = tl.exp(b_g_last)
            b_h1 *= b_g_last_exp
            b_h2 *= b_g_last_exp
            b_v_update = b_v_update.to(tl.bfloat16)
            b_h1 += tl.trans(tl.dot(b_k1_t, b_v_update))
            b_h2 += tl.trans(tl.dot(b_k2_t, b_v_update))

        p_ht1 = ht + (i_h * V + offs_v[:, None]) * K + offs_k[None, :]
        p_ht2 = ht + (i_h * V + offs_v[:, None]) * K + (64 + offs_k[None, :])
        tl.store(p_ht1, b_h1, mask=mask_h1)
        tl.store(p_ht2, b_h2, mask=mask_h2)

    @triton.jit
    def triton_matvec_kernel(
        x,
        weight_t,
        out,
        k_size: tl.constexpr,
        n_size: tl.constexpr,
        block_n: tl.constexpr,
        block_k: tl.constexpr,
    ) -> None:
        pid = tl.program_id(0)
        offs_n = pid * block_n + tl.arange(0, block_n)
        offs_k = tl.arange(0, block_k)
        acc = tl.zeros((block_n,), tl.float32)
        for start in range(0, k_size, block_k):
            k = start + offs_k
            x_values = tl.load(x + k, mask=k < k_size, other=0.0).to(tl.float32)
            weight_values = tl.load(
                weight_t + k[:, None] * n_size + offs_n[None, :],
                mask=(k[:, None] < k_size) & (offs_n[None, :] < n_size),
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(weight_values * x_values[:, None], axis=0)
        tl.store(out + offs_n, acc, mask=offs_n < n_size)

    @triton.jit
    def triton_full_attention_gated_o_proj_kernel(
        attn_out,
        gate,
        weight_t,
        out,
        q_dim: tl.constexpr,
        hidden_size: tl.constexpr,
        block_n: tl.constexpr,
        block_k: tl.constexpr,
    ) -> None:
        pid = tl.program_id(0)
        offs_n = pid * block_n + tl.arange(0, block_n)
        offs_k = tl.arange(0, block_k)
        valid_n = offs_n < hidden_size
        acc = tl.zeros((block_n,), tl.float32)
        for start in range(0, q_dim, block_k):
            k = start + offs_k
            valid_k = k < q_dim
            attn_values = tl.load(attn_out + k, mask=valid_k, other=0.0).to(tl.float32)
            gate_values = tl.load(gate + k, mask=valid_k, other=0.0).to(tl.float32)
            gate_scale = (1.0 / (1.0 + tl.exp(-gate_values))).to(tl.bfloat16)
            gated = (attn_values * gate_scale.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
            weight_values = tl.load(
                weight_t + k[:, None] * hidden_size + offs_n[None, :],
                mask=valid_k[:, None] & valid_n[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(gated[:, None] * weight_values, axis=0)
        tl.store(out + offs_n, acc, mask=valid_n)

    @triton.jit
    def triton_fused_shared_down_kernel(
        shared_input,
        down_weight_t,
        out,
        intermediate: tl.constexpr,
        hidden_size: tl.constexpr,
        block_n: tl.constexpr,
        block_k: tl.constexpr,
    ) -> None:
        pid = tl.program_id(0)
        offs_n = pid * block_n + tl.arange(0, block_n)
        offs_k = tl.arange(0, block_k)
        shared_gate = tl.load(shared_input + 0).to(tl.float32)
        gate_scale = tl.sigmoid(shared_gate).to(tl.bfloat16).to(tl.float32)
        acc = tl.zeros((block_n,), tl.float32)
        for start in range(0, intermediate, block_k):
            k = start + offs_k
            valid_k = k < intermediate
            gate = tl.load(shared_input + 1 + k, mask=valid_k, other=0.0).to(tl.float32)
            up = tl.load(shared_input + 1 + intermediate + k, mask=valid_k, other=0.0).to(tl.float32)
            activated = ((gate / (1.0 + tl.exp(-gate))) * up).to(tl.bfloat16).to(tl.float32)
            weights = tl.load(
                down_weight_t + k[:, None] * hidden_size + offs_n[None, :],
                mask=valid_k[:, None] & (offs_n[None, :] < hidden_size),
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(weights * activated[:, None], axis=0)
        acc = acc.to(tl.bfloat16).to(tl.float32)
        tl.store(out + offs_n, acc * gate_scale, mask=offs_n < hidden_size)

    @triton.jit
    def triton_router_topk_stage1_kernel(
        x,
        weight_t,
        partial_values,
        partial_ids,
        hidden_size: tl.constexpr,
        num_experts: tl.constexpr,
        block_e: tl.constexpr,
        block_k: tl.constexpr,
    ) -> None:
        pid = tl.program_id(0)
        offs_e = pid * block_e + tl.arange(0, block_e)
        offs_k = tl.arange(0, block_k)
        valid_e = offs_e < num_experts
        acc = tl.zeros((block_e,), tl.float32)
        for start in range(0, hidden_size, block_k):
            k = start + offs_k
            x_values = tl.load(x + k, mask=k < hidden_size, other=0.0).to(tl.float32)
            weight_values = tl.load(
                weight_t + k[:, None] * num_experts + offs_e[None, :],
                mask=(k[:, None] < hidden_size) & valid_e[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(weight_values * x_values[:, None], axis=0)

        values = tl.where(valid_e, acc, -float("inf"))
        for rank in range(0, 8):
            max_value = tl.max(values, axis=0)
            tied_ids = tl.where(values == max_value, offs_e, num_experts)
            max_id = tl.min(tied_ids, axis=0)
            tl.store(partial_values + pid * 8 + rank, max_value)
            tl.store(partial_ids + pid * 8 + rank, max_id)
            values = tl.where(offs_e == max_id, -float("inf"), values)

    @triton.jit
    def triton_router_topk_stage2_kernel(
        partial_values,
        partial_ids,
        out_values,
        out_ids,
        num_candidates: tl.constexpr,
        block_c: tl.constexpr,
    ) -> None:
        offs = tl.arange(0, block_c)
        mask = offs < num_candidates
        values = tl.load(partial_values + offs, mask=mask, other=-float("inf")).to(tl.float32)
        ids = tl.load(partial_ids + offs, mask=mask, other=2147483647)
        for rank in range(0, 8):
            max_value = tl.max(values, axis=0)
            tied_ids = tl.where(values == max_value, ids, 2147483647)
            max_id = tl.min(tied_ids, axis=0)
            tl.store(out_values + rank, max_value)
            tl.store(out_ids + rank, max_id)
            values = tl.where(ids == max_id, -float("inf"), values)

    @triton.jit
    def triton_router_topk_stage2_softmax_kernel(
        partial_values,
        partial_ids,
        out_values,
        out_ids,
        num_candidates: tl.constexpr,
        block_c: tl.constexpr,
    ) -> None:
        offs = tl.arange(0, block_c)
        mask = offs < num_candidates
        values = tl.load(partial_values + offs, mask=mask, other=-float("inf")).to(tl.float32)
        ids = tl.load(partial_ids + offs, mask=mask, other=2147483647)
        rank_offsets = tl.arange(0, 8)
        selected_values = tl.full((8,), -float("inf"), tl.float32)
        selected_ids = tl.full((8,), 2147483647, tl.int32)
        for rank in range(0, 8):
            max_value = tl.max(values, axis=0)
            tied_ids = tl.where(values == max_value, ids, 2147483647)
            max_id = tl.min(tied_ids, axis=0)
            selected_values = tl.where(rank_offsets == rank, max_value, selected_values)
            selected_ids = tl.where(rank_offsets == rank, max_id, selected_ids)
            values = tl.where(ids == max_id, -float("inf"), values)
        max_selected = tl.max(selected_values, axis=0)
        exp_values = tl.exp(selected_values - max_selected)
        weights = exp_values / tl.sum(exp_values, axis=0)
        tl.store(out_values + rank_offsets, weights)
        tl.store(out_ids + rank_offsets, selected_ids)

    @triton.jit
    def triton_decode_direct_conv_kernel(
        raw,
        state_in,
        weight,
        out,
        state_out,
        channels: tl.constexpr,
        state_len: tl.constexpr,
        kernel_dim: tl.constexpr,
        block_c: tl.constexpr,
    ) -> None:
        pid = tl.program_id(0)
        offs = pid * block_c + tl.arange(0, block_c)
        mask = offs < channels
        raw_store_values = tl.load(raw + offs, mask=mask, other=0.0)
        raw_values = raw_store_values.to(tl.float32)
        acc = tl.zeros((block_c,), tl.float32)
        for idx in range(0, state_len):
            state_values = tl.load(state_in + offs * state_len + idx, mask=mask, other=0.0).to(tl.float32)
            weight_values = tl.load(weight + offs * kernel_dim + idx, mask=mask, other=0.0).to(tl.float32)
            acc += state_values * weight_values
            if idx + 1 < state_len:
                next_state = tl.load(state_in + offs * state_len + idx + 1, mask=mask, other=0.0)
            else:
                next_state = raw_store_values
            tl.store(state_out + offs * state_len + idx, next_state, mask=mask)
        last_weight = tl.load(weight + offs * kernel_dim + state_len, mask=mask, other=0.0).to(tl.float32)
        acc += raw_values * last_weight
        activated = acc / (1.0 + tl.exp(-acc))
        tl.store(out + offs, activated, mask=mask)

    @triton.jit
    def triton_prefill_direct_conv_kernel(
        raw,
        state_in,
        weight,
        out,
        state_out,
        raw_stride_t: tl.constexpr,
        raw_stride_c: tl.constexpr,
        tokens: tl.constexpr,
        channels: tl.constexpr,
        state_len: tl.constexpr,
        kernel_dim: tl.constexpr,
        block_t: tl.constexpr,
        block_c: tl.constexpr,
    ) -> None:
        pid_t = tl.program_id(0)
        pid_c = tl.program_id(1)
        offs_t = pid_t * block_t + tl.arange(0, block_t)
        offs_c = pid_c * block_c + tl.arange(0, block_c)
        mask_t = offs_t < tokens
        mask_c = offs_c < channels
        acc = tl.zeros((block_t, block_c), tl.float32)
        for idx in range(0, kernel_dim):
            conv_input_idx = offs_t + idx
            use_state = conv_input_idx < state_len
            raw_t = conv_input_idx - state_len
            state_values = tl.load(
                state_in + offs_c[None, :] * state_len + conv_input_idx[:, None],
                mask=mask_c[None, :] & use_state[:, None],
                other=0.0,
            ).to(tl.float32)
            raw_values = tl.load(
                raw + raw_t[:, None] * raw_stride_t + offs_c[None, :] * raw_stride_c,
                mask=mask_t[:, None] & mask_c[None, :] & ~use_state[:, None],
                other=0.0,
            ).to(tl.float32)
            weight_values = tl.load(weight + offs_c * kernel_dim + idx, mask=mask_c, other=0.0).to(tl.float32)
            acc += (state_values + raw_values) * weight_values[None, :]
        activated = acc / (1.0 + tl.exp(-acc))
        tl.store(out + offs_t[:, None] * channels + offs_c[None, :], activated, mask=mask_t[:, None] & mask_c[None, :])

        if pid_t == 0:
            for idx in range(0, state_len):
                conv_input_idx = tokens + idx
                if conv_input_idx < state_len:
                    state_tail = tl.load(
                        state_in + offs_c * state_len + conv_input_idx,
                        mask=mask_c,
                        other=0.0,
                    )
                else:
                    state_tail = tl.load(
                        raw + (conv_input_idx - state_len) * raw_stride_t + offs_c * raw_stride_c,
                        mask=mask_c,
                        other=0.0,
                    )
                tl.store(state_out + offs_c * state_len + idx, state_tail, mask=mask_c)

    @triton.jit
    def triton_prefill_conv_post_prep_kernel(
        raw,
        state_in,
        conv_weight,
        a_ptr,
        b_ptr,
        A_log_ptr,
        dt_bias_ptr,
        q_ptr,
        k_ptr,
        v_ptr,
        g_ptr,
        beta_ptr,
        state_out,
        raw_stride_t: tl.constexpr,
        raw_stride_c: tl.constexpr,
        stride_a_tok: tl.constexpr,
        stride_b_tok: tl.constexpr,
        stride_q_tok: tl.constexpr,
        stride_k_tok: tl.constexpr,
        stride_v_tok: tl.constexpr,
        tokens: tl.constexpr,
        state_len: tl.constexpr,
        kernel_dim: tl.constexpr,
        H: tl.constexpr,
        HV: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        L2NORM_EPS: tl.constexpr,
        SOFTPLUS_THRESHOLD: tl.constexpr,
        block_t: tl.constexpr,
        BK: tl.constexpr,
        BV: tl.constexpr,
    ) -> None:
        pid_t = tl.program_id(0)
        pid_head = tl.program_id(1)

        HK: tl.constexpr = H * K
        V_OFFSET: tl.constexpr = 2 * H * K

        offs_t = pid_t * block_t + tl.arange(0, block_t)
        mask_t = offs_t < tokens

        if pid_head < H:
            i_h = pid_head
            offs_k = tl.arange(0, BK)
            mask_k = offs_k < K
            mask_2d = mask_t[:, None] & mask_k[None, :]

            q_c = i_h * K + offs_k
            k_c = HK + i_h * K + offs_k
            q_acc = tl.zeros((block_t, BK), tl.float32)
            k_acc = tl.zeros((block_t, BK), tl.float32)
            for idx in range(0, kernel_dim):
                conv_input_idx = offs_t + idx
                use_state = conv_input_idx < state_len
                raw_t = conv_input_idx - state_len

                q_state = tl.load(
                    state_in + q_c[None, :] * state_len + conv_input_idx[:, None],
                    mask=mask_k[None, :] & use_state[:, None],
                    other=0.0,
                ).to(tl.float32)
                q_raw = tl.load(
                    raw + raw_t[:, None] * raw_stride_t + q_c[None, :] * raw_stride_c,
                    mask=mask_2d & ~use_state[:, None],
                    other=0.0,
                ).to(tl.float32)
                q_w = tl.load(conv_weight + q_c * kernel_dim + idx, mask=mask_k, other=0.0).to(tl.float32)
                q_acc += (q_state + q_raw) * q_w[None, :]

                k_state = tl.load(
                    state_in + k_c[None, :] * state_len + conv_input_idx[:, None],
                    mask=mask_k[None, :] & use_state[:, None],
                    other=0.0,
                ).to(tl.float32)
                k_raw = tl.load(
                    raw + raw_t[:, None] * raw_stride_t + k_c[None, :] * raw_stride_c,
                    mask=mask_2d & ~use_state[:, None],
                    other=0.0,
                ).to(tl.float32)
                k_w = tl.load(conv_weight + k_c * kernel_dim + idx, mask=mask_k, other=0.0).to(tl.float32)
                k_acc += (k_state + k_raw) * k_w[None, :]

            q_bf16_f32 = (q_acc / (1.0 + tl.exp(-q_acc))).to(tl.bfloat16).to(tl.float32)
            k_bf16_f32 = (k_acc / (1.0 + tl.exp(-k_acc))).to(tl.bfloat16).to(tl.float32)
            q_sq_sum = tl.sum(q_bf16_f32 * q_bf16_f32, axis=1)
            k_sq_sum = tl.sum(k_bf16_f32 * k_bf16_f32, axis=1)
            q_norm = q_bf16_f32 * (1.0 / tl.sqrt(q_sq_sum + L2NORM_EPS))[:, None]
            k_norm = k_bf16_f32 * (1.0 / tl.sqrt(k_sq_sum + L2NORM_EPS))[:, None]

            q_out = offs_t[:, None] * stride_q_tok + i_h * K + offs_k[None, :]
            k_out = offs_t[:, None] * stride_k_tok + i_h * K + offs_k[None, :]
            tl.store(q_ptr + q_out, q_norm.to(tl.bfloat16), mask=mask_2d)
            tl.store(k_ptr + k_out, k_norm.to(tl.bfloat16), mask=mask_2d)

            if pid_t == 0:
                for idx in range(0, state_len):
                    tail_raw_t = tokens - state_len + idx
                    q_tail = tl.load(raw + tail_raw_t * raw_stride_t + q_c * raw_stride_c, mask=mask_k, other=0.0)
                    k_tail = tl.load(raw + tail_raw_t * raw_stride_t + k_c * raw_stride_c, mask=mask_k, other=0.0)
                    tl.store(state_out + q_c * state_len + idx, q_tail, mask=mask_k)
                    tl.store(state_out + k_c * state_len + idx, k_tail, mask=mask_k)
        else:
            i_hv = pid_head - H
            offs_v = tl.arange(0, BV)
            mask_v = offs_v < V
            mask_2d = mask_t[:, None] & mask_v[None, :]
            v_c = V_OFFSET + i_hv * V + offs_v
            v_acc = tl.zeros((block_t, BV), tl.float32)

            for idx in range(0, kernel_dim):
                conv_input_idx = offs_t + idx
                use_state = conv_input_idx < state_len
                raw_t = conv_input_idx - state_len
                v_state = tl.load(
                    state_in + v_c[None, :] * state_len + conv_input_idx[:, None],
                    mask=mask_v[None, :] & use_state[:, None],
                    other=0.0,
                ).to(tl.float32)
                v_raw = tl.load(
                    raw + raw_t[:, None] * raw_stride_t + v_c[None, :] * raw_stride_c,
                    mask=mask_2d & ~use_state[:, None],
                    other=0.0,
                ).to(tl.float32)
                v_w = tl.load(conv_weight + v_c * kernel_dim + idx, mask=mask_v, other=0.0).to(tl.float32)
                v_acc += (v_state + v_raw) * v_w[None, :]

            v_vals = (v_acc / (1.0 + tl.exp(-v_acc))).to(tl.bfloat16)
            v_out = offs_t[:, None] * stride_v_tok + i_hv * V + offs_v[None, :]
            tl.store(v_ptr + v_out, v_vals, mask=mask_2d)

            A_log_val = tl.load(A_log_ptr + i_hv).to(tl.float32)
            dt_bias_val = tl.load(dt_bias_ptr + i_hv).to(tl.float32)
            a_offsets = offs_t * stride_a_tok + i_hv
            b_offsets = offs_t * stride_b_tok + i_hv
            a_vals = tl.load(a_ptr + a_offsets, mask=mask_t, other=0.0).to(tl.float32)
            b_vals = tl.load(b_ptr + b_offsets, mask=mask_t, other=0.0).to(tl.float32)

            x = a_vals + dt_bias_val
            sp = tl.where(x > 0, x + tl.log(1.0 + tl.exp(-x)), tl.log(1.0 + tl.exp(x)))
            sp = tl.where(x <= SOFTPLUS_THRESHOLD, sp, x)
            g_vals = -tl.exp(A_log_val) * sp
            beta_vals = tl.sigmoid(b_vals)
            gb_offsets = offs_t * HV + i_hv
            tl.store(g_ptr + gb_offsets, g_vals, mask=mask_t)
            tl.store(beta_ptr + gb_offsets, beta_vals, mask=mask_t)

            if pid_t == 0:
                for idx in range(0, state_len):
                    tail_raw_t = tokens - state_len + idx
                    v_tail = tl.load(raw + tail_raw_t * raw_stride_t + v_c * raw_stride_c, mask=mask_v, other=0.0)
                    tl.store(state_out + v_c * state_len + idx, v_tail, mask=mask_v)

    @triton.jit
    def triton_fused_input_proj_conv_kernel(
        x,
        weight_t,
        state_in,
        conv_weight,
        out,
        state_out,
        hidden_size: tl.constexpr,
        out_features: tl.constexpr,
        qkv_features: tl.constexpr,
        state_len: tl.constexpr,
        kernel_dim: tl.constexpr,
        block_n: tl.constexpr,
        block_k: tl.constexpr,
    ) -> None:
        pid = tl.program_id(0)
        offs_n = pid * block_n + tl.arange(0, block_n)
        offs_k = tl.arange(0, block_k)
        valid_n = offs_n < out_features
        acc = tl.zeros((block_n,), tl.float32)
        for start in range(0, hidden_size, block_k):
            k = start + offs_k
            x_values = tl.load(x + k, mask=k < hidden_size, other=0.0).to(tl.float32)
            weight_values = tl.load(
                weight_t + k[:, None] * out_features + offs_n[None, :],
                mask=(k[:, None] < hidden_size) & valid_n[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(weight_values * x_values[:, None], axis=0)

        raw_bf16 = acc.to(tl.bfloat16)
        is_qkv = offs_n < qkv_features
        conv_acc = tl.zeros((block_n,), tl.float32)
        for idx in range(0, state_len):
            state_values = tl.load(state_in + offs_n * state_len + idx, mask=is_qkv, other=0.0).to(tl.float32)
            weight_values = tl.load(conv_weight + offs_n * kernel_dim + idx, mask=is_qkv, other=0.0).to(tl.float32)
            conv_acc += state_values * weight_values
            if idx + 1 < state_len:
                next_state = tl.load(state_in + offs_n * state_len + idx + 1, mask=is_qkv, other=0.0)
            else:
                next_state = raw_bf16
            tl.store(state_out + offs_n * state_len + idx, next_state, mask=is_qkv)
        last_weight = tl.load(conv_weight + offs_n * kernel_dim + state_len, mask=is_qkv, other=0.0).to(tl.float32)
        conv_acc += raw_bf16.to(tl.float32) * last_weight
        activated = conv_acc / (1.0 + tl.exp(-conv_acc))
        result = tl.where(is_qkv, activated, raw_bf16.to(tl.float32))
        tl.store(out + offs_n, result, mask=valid_n)

    @triton.jit
    def triton_fused_input_proj_conv_qkv_kernel(
        x,
        weight_t,
        state_in,
        conv_weight,
        q_out,
        k_out,
        v_out,
        z_out,
        a_out,
        b_out,
        state_out,
        hidden_size: tl.constexpr,
        out_features: tl.constexpr,
        key_dim: tl.constexpr,
        value_dim: tl.constexpr,
        qkv_features: tl.constexpr,
        value_heads: tl.constexpr,
        state_len: tl.constexpr,
        kernel_dim: tl.constexpr,
        block_n: tl.constexpr,
        block_k: tl.constexpr,
    ) -> None:
        pid = tl.program_id(0)
        offs_n = pid * block_n + tl.arange(0, block_n)
        offs_k = tl.arange(0, block_k)
        valid_n = offs_n < out_features
        acc = tl.zeros((block_n,), tl.float32)
        for start in range(0, hidden_size, block_k):
            k = start + offs_k
            x_values = tl.load(x + k, mask=k < hidden_size, other=0.0).to(tl.float32)
            weight_values = tl.load(
                weight_t + k[:, None] * out_features + offs_n[None, :],
                mask=(k[:, None] < hidden_size) & valid_n[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(weight_values * x_values[:, None], axis=0)

        raw_bf16 = acc.to(tl.bfloat16)
        is_qkv = offs_n < qkv_features
        conv_acc = tl.zeros((block_n,), tl.float32)
        for idx in range(0, state_len):
            state_values = tl.load(state_in + offs_n * state_len + idx, mask=is_qkv, other=0.0).to(tl.float32)
            weight_values = tl.load(conv_weight + offs_n * kernel_dim + idx, mask=is_qkv, other=0.0).to(tl.float32)
            conv_acc += state_values * weight_values
            if idx + 1 < state_len:
                next_state = tl.load(state_in + offs_n * state_len + idx + 1, mask=is_qkv, other=0.0)
            else:
                next_state = raw_bf16
            tl.store(state_out + offs_n * state_len + idx, next_state, mask=is_qkv)
        last_weight = tl.load(conv_weight + offs_n * kernel_dim + state_len, mask=is_qkv, other=0.0).to(tl.float32)
        conv_acc += raw_bf16.to(tl.float32) * last_weight
        activated = conv_acc / (1.0 + tl.exp(-conv_acc))
        qkv_result = activated

        is_q = offs_n < key_dim
        is_k = (offs_n >= key_dim) & (offs_n < 2 * key_dim)
        is_v = (offs_n >= 2 * key_dim) & (offs_n < qkv_features)
        z_start = qkv_features
        a_start = qkv_features + value_dim
        b_start = a_start + value_heads
        is_z = (offs_n >= z_start) & (offs_n < a_start)
        is_a = (offs_n >= a_start) & (offs_n < b_start)
        is_b = (offs_n >= b_start) & valid_n

        tl.store(q_out + offs_n, qkv_result, mask=is_q)
        tl.store(k_out + (offs_n - key_dim), qkv_result, mask=is_k)
        tl.store(v_out + (offs_n - 2 * key_dim), qkv_result, mask=is_v)
        tl.store(z_out + (offs_n - z_start), raw_bf16, mask=is_z)
        tl.store(a_out + (offs_n - a_start), raw_bf16, mask=is_a)
        tl.store(b_out + (offs_n - b_start), raw_bf16, mask=is_b)

    @triton.jit
    def triton_rmsnorm_kernel(
        x,
        weight,
        out,
        hidden_size: tl.constexpr,
        eps: tl.constexpr,
        block_h: tl.constexpr,
    ) -> None:
        row = tl.program_id(0)
        offs = tl.arange(0, block_h)
        mask = offs < hidden_size
        values = tl.load(x + row * hidden_size + offs, mask=mask, other=0.0).to(tl.float32)
        weights = tl.load(weight + offs, mask=mask, other=0.0).to(tl.float32) + 1.0
        variance = tl.sum(values * values, axis=0) / hidden_size
        normalized = values * tl.rsqrt(variance + eps) * weights
        tl.store(out + row * hidden_size + offs, normalized, mask=mask)

    @triton.jit
    def triton_prefill_fused_add_rmsnorm_kernel(
        x,
        residual,
        weight,
        residual_out,
        norm_out,
        hidden_size: tl.constexpr,
        eps: tl.constexpr,
        block_h: tl.constexpr,
    ) -> None:
        row = tl.program_id(0)
        offs = tl.arange(0, block_h)
        mask = offs < hidden_size
        x_values = tl.load(x + row * hidden_size + offs, mask=mask, other=0.0).to(tl.float32)
        residual_values = tl.load(
            residual + row * hidden_size + offs, mask=mask, other=0.0
        ).to(tl.float32)
        summed_bf16 = (x_values + residual_values).to(tl.bfloat16)
        values = summed_bf16.to(tl.float32)
        weights = tl.load(weight + offs, mask=mask, other=0.0).to(tl.float32) + 1.0
        variance = tl.sum(values * values, axis=0) / hidden_size
        normalized = values * tl.rsqrt(variance + eps) * weights
        tl.store(residual_out + row * hidden_size + offs, summed_bf16, mask=mask)
        tl.store(norm_out + row * hidden_size + offs, normalized, mask=mask)

    @triton.jit
    def triton_head_norm_rope_kernel(
        inp,
        weight,
        cos,
        sin,
        out,
        head_dim: tl.constexpr,
        rotary_dim: tl.constexpr,
        heads_per_token: tl.constexpr,
        input_token_stride: tl.constexpr,
        input_head_stride: tl.constexpr,
        block_h: tl.constexpr,
    ) -> None:
        row = tl.program_id(0)
        token_index = row // heads_per_token
        head_index = row - token_index * heads_per_token
        input_base = token_index * input_token_stride + head_index * input_head_stride
        offs = tl.arange(0, block_h)
        mask = offs < head_dim
        values = tl.load(inp + input_base + offs, mask=mask, other=0.0).to(tl.float32)
        weights = tl.load(weight + offs, mask=mask, other=0.0).to(tl.float32) + 1.0
        variance = tl.sum(values * values, axis=0) / head_dim
        inv_rms = tl.rsqrt(variance + 1.0e-6)
        normalized = (values * inv_rms * weights).to(tl.bfloat16).to(tl.float32)

        half_rotary = rotary_dim // 2
        first_half = offs < half_rotary
        mate_offs = tl.where(first_half, offs + half_rotary, offs - half_rotary)
        mate_mask = mate_offs < head_dim
        mate_values = tl.load(inp + input_base + mate_offs, mask=mate_mask, other=0.0).to(tl.float32)
        mate_weights = tl.load(weight + mate_offs, mask=mate_mask, other=0.0).to(tl.float32) + 1.0
        mate_normalized = (mate_values * inv_rms * mate_weights).to(tl.bfloat16).to(tl.float32)

        in_rotary = offs < rotary_dim
        pair = tl.where(first_half, offs, offs - half_rotary)
        rotary_base = token_index * half_rotary
        cos_values = tl.load(cos + rotary_base + pair, mask=in_rotary, other=1.0).to(tl.float32)
        sin_values = tl.load(sin + rotary_base + pair, mask=in_rotary, other=0.0).to(tl.float32)
        first_rotated = normalized * cos_values - mate_normalized * sin_values
        second_rotated = normalized * cos_values + mate_normalized * sin_values
        rotated = tl.where(first_half, first_rotated, second_rotated)
        result = tl.where(in_rotary, rotated, normalized)
        tl.store(out + row * head_dim + offs, result, mask=mask)

    @triton.jit
    def triton_full_attention_norm_rope_kv_write_kernel(
        q,
        k,
        v,
        q_weight,
        k_weight,
        cos,
        sin,
        q_out,
        k_cache,
        v_cache,
        cache_position,
        head_dim: tl.constexpr,
        rotary_dim: tl.constexpr,
        heads: tl.constexpr,
        kv_heads: tl.constexpr,
        q_head_stride: tl.constexpr,
        k_head_stride: tl.constexpr,
        v_head_stride: tl.constexpr,
        cache_token_stride: tl.constexpr,
        cache_head_stride: tl.constexpr,
        block_h: tl.constexpr,
    ) -> None:
        row = tl.program_id(0)
        offs = tl.arange(0, block_h)
        mask = offs < head_dim

        is_q = row < heads
        is_k = (row >= heads) & (row < heads + kv_heads)
        is_v = row >= heads + kv_heads
        q_head = tl.where(is_q, row, 0)
        k_head = tl.where(is_k, row - heads, 0)
        v_head = tl.where(is_v, row - heads - kv_heads, 0)

        q_values = tl.load(q + q_head * q_head_stride + offs, mask=is_q & mask, other=0.0).to(tl.float32)
        k_values = tl.load(k + k_head * k_head_stride + offs, mask=is_k & mask, other=0.0).to(tl.float32)
        v_values = tl.load(v + v_head * v_head_stride + offs, mask=is_v & mask, other=0.0)

        q_weights = tl.load(q_weight + offs, mask=mask, other=0.0).to(tl.float32) + 1.0
        k_weights = tl.load(k_weight + offs, mask=mask, other=0.0).to(tl.float32) + 1.0
        q_variance = tl.sum(q_values * q_values, axis=0) / head_dim
        k_variance = tl.sum(k_values * k_values, axis=0) / head_dim
        q_inv_rms = tl.rsqrt(q_variance + 1.0e-6)
        k_inv_rms = tl.rsqrt(k_variance + 1.0e-6)
        q_normalized = (q_values * q_inv_rms * q_weights).to(tl.bfloat16).to(tl.float32)
        k_normalized = (k_values * k_inv_rms * k_weights).to(tl.bfloat16).to(tl.float32)

        half_rotary = rotary_dim // 2
        first_half = offs < half_rotary
        mate_offs = tl.where(first_half, offs + half_rotary, offs - half_rotary)
        mate_mask = mate_offs < head_dim
        q_mate_values = tl.load(
            q + q_head * q_head_stride + mate_offs,
            mask=is_q & mate_mask,
            other=0.0,
        ).to(tl.float32)
        k_mate_values = tl.load(
            k + k_head * k_head_stride + mate_offs,
            mask=is_k & mate_mask,
            other=0.0,
        ).to(tl.float32)
        q_mate_weights = tl.load(q_weight + mate_offs, mask=mate_mask, other=0.0).to(tl.float32) + 1.0
        k_mate_weights = tl.load(k_weight + mate_offs, mask=mate_mask, other=0.0).to(tl.float32) + 1.0
        q_mate_normalized = (q_mate_values * q_inv_rms * q_mate_weights).to(tl.bfloat16).to(tl.float32)
        k_mate_normalized = (k_mate_values * k_inv_rms * k_mate_weights).to(tl.bfloat16).to(tl.float32)

        in_rotary = offs < rotary_dim
        pair = tl.where(first_half, offs, offs - half_rotary)
        cos_values = tl.load(cos + pair, mask=in_rotary, other=1.0).to(tl.float32)
        sin_values = tl.load(sin + pair, mask=in_rotary, other=0.0).to(tl.float32)
        q_first_rotated = q_normalized * cos_values - q_mate_normalized * sin_values
        q_second_rotated = q_normalized * cos_values + q_mate_normalized * sin_values
        k_first_rotated = k_normalized * cos_values - k_mate_normalized * sin_values
        k_second_rotated = k_normalized * cos_values + k_mate_normalized * sin_values
        q_rotated = tl.where(first_half, q_first_rotated, q_second_rotated)
        k_rotated = tl.where(first_half, k_first_rotated, k_second_rotated)
        q_result = tl.where(in_rotary, q_rotated, q_normalized)
        k_result = tl.where(in_rotary, k_rotated, k_normalized)

        cache_base = cache_position * cache_token_stride
        tl.store(q_out + q_head * head_dim + offs, q_result, mask=is_q & mask)
        tl.store(k_cache + cache_base + k_head * cache_head_stride + offs, k_result, mask=is_k & mask)
        tl.store(v_cache + cache_base + v_head * cache_head_stride + offs, v_values, mask=is_v & mask)

    @triton.jit
    def triton_linear_gated_norm_kernel(
        core,
        z,
        weight,
        out,
        head_dim: tl.constexpr,
        eps: tl.constexpr,
        block_h: tl.constexpr,
    ) -> None:
        row = tl.program_id(0)
        offs = tl.arange(0, block_h)
        mask = offs < head_dim
        core_values = tl.load(core + row * head_dim + offs, mask=mask, other=0.0).to(tl.float32)
        z_values = tl.load(z + row * head_dim + offs, mask=mask, other=0.0).to(tl.float32)
        weights = tl.load(weight + offs, mask=mask, other=0.0).to(tl.float32)
        variance = tl.sum(core_values * core_values, axis=0) / head_dim
        normalized = core_values * tl.rsqrt(variance + eps) * weights
        z_gate = z_values / (1.0 + tl.exp(-z_values))
        tl.store(out + row * head_dim + offs, normalized.to(tl.bfloat16).to(tl.float32) * z_gate, mask=mask)



    @triton.jit
    def triton_linear_gated_norm_from_invstd_kernel(
        core,
        z,
        weight,
        invstd,
        out,
        head_dim: tl.constexpr,
        block_h: tl.constexpr,
    ) -> None:
        row = tl.program_id(0)
        offs = tl.arange(0, block_h)
        mask = offs < head_dim
        core_values = tl.load(core + row * head_dim + offs, mask=mask, other=0.0).to(tl.float32)
        z_values = tl.load(z + row * head_dim + offs, mask=mask, other=0.0).to(tl.float32)
        weights = tl.load(weight + offs, mask=mask, other=0.0).to(tl.float32)
        invstd_value = tl.load(invstd + row).to(tl.float32)
        normalized = core_values * invstd_value * weights
        z_gate = z_values / (1.0 + tl.exp(-z_values))
        tl.store(
            out + row * head_dim + offs,
            normalized.to(tl.bfloat16).to(tl.float32) * z_gate,
            mask=mask,
        )

    @triton.jit
    def triton_native_moe_gate_up_activation_kernel(
        hidden_state,
        gate_up_native,
        topk_ids,
        act_out,
        hidden_size: tl.constexpr,
        intermediate_size: tl.constexpr,
        num_i_tiles: tl.constexpr,
        block_i: tl.constexpr,
        block_h: tl.constexpr,
    ) -> None:
        rank = tl.program_id(0)
        pid_i = tl.program_id(1)
        expert = tl.load(topk_ids + rank)
        offs_i_inner = tl.arange(0, block_i)
        offs_i = pid_i * block_i + offs_i_inner
        offs_h = tl.arange(0, block_h)
        valid_i = offs_i < intermediate_size

        gate = tl.zeros((block_i,), tl.float32)
        up = tl.zeros((block_i,), tl.float32)
        gate_base = ((expert * 2 + 0) * num_i_tiles + pid_i) * hidden_size
        up_base = ((expert * 2 + 1) * num_i_tiles + pid_i) * hidden_size

        for start in range(0, hidden_size, block_h):
            h = start + offs_h
            valid_h = h < hidden_size
            x = tl.load(hidden_state + h, mask=valid_h, other=0.0).to(tl.float32)
            gate_w = tl.load(
                gate_up_native + (gate_base + h[:, None]) * block_i + offs_i_inner[None, :],
                mask=valid_h[:, None] & valid_i[None, :],
                other=0.0,
            ).to(tl.float32)
            up_w = tl.load(
                gate_up_native + (up_base + h[:, None]) * block_i + offs_i_inner[None, :],
                mask=valid_h[:, None] & valid_i[None, :],
                other=0.0,
            ).to(tl.float32)
            gate += tl.sum(gate_w * x[:, None], axis=0)
            up += tl.sum(up_w * x[:, None], axis=0)

        silu_gate = gate / (1.0 + tl.exp(-gate))
        tl.store(act_out + rank * intermediate_size + offs_i, silu_gate * up, mask=valid_i)

    @triton.jit
    def triton_native_moe_down_sum_kernel(
        act,
        down_native,
        topk_ids,
        topk_weights,
        out,
        hidden_size: tl.constexpr,
        intermediate_size: tl.constexpr,
        top_k: tl.constexpr,
        hidden_tiles: tl.constexpr,
        block_h: tl.constexpr,
        block_i: tl.constexpr,
    ) -> None:
        pid_h = tl.program_id(0)
        offs_h_inner = tl.arange(0, block_h)
        offs_h = pid_h * block_h + offs_h_inner
        offs_i_inner = tl.arange(0, block_i)
        valid_h = offs_h < hidden_size
        acc = tl.zeros((block_h,), tl.float32)

        for rank in range(0, top_k):
            expert = tl.load(topk_ids + rank)
            rank_weight = tl.load(topk_weights + rank).to(tl.float32)
            rank_acc = tl.zeros((block_h,), tl.float32)
            rank_down_base = (expert * hidden_tiles + pid_h) * intermediate_size
            rank_act_base = rank * intermediate_size
            for start in range(0, intermediate_size, block_i):
                i = start + offs_i_inner
                valid_i = i < intermediate_size
                act_values = tl.load(act + rank_act_base + i, mask=valid_i, other=0.0).to(tl.float32)
                down_values = tl.load(
                    down_native + (rank_down_base + i[:, None]) * block_h + offs_h_inner[None, :],
                    mask=valid_i[:, None] & valid_h[None, :],
                    other=0.0,
                ).to(tl.float32)
                rank_acc += tl.sum(down_values * act_values[:, None], axis=0)
            acc += rank_weight * rank_acc

        tl.store(out + offs_h, acc, mask=valid_h)

    @triton.jit
    def raw_row_gate_up_activation_kernel(
        hidden_state,
        gate_up_raw,
        topk_ids,
        act_out,
        hidden_size: tl.constexpr,
        intermediate_size: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        rank = tl.program_id(0)
        out_i = tl.program_id(1)
        expert = tl.load(topk_ids + rank)
        offs_h = tl.arange(0, BLOCK_H)
        gate = 0.0
        up = 0.0
        gate_row = expert * (2 * intermediate_size) + out_i
        up_row = gate_row + intermediate_size
        for start in range(0, hidden_size, BLOCK_H):
            h = start + offs_h
            mask = h < hidden_size
            x = tl.load(hidden_state + h, mask=mask, other=0.0).to(tl.float32)
            gate_w = tl.load(
                gate_up_raw + gate_row * hidden_size + h,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            up_w = tl.load(
                gate_up_raw + up_row * hidden_size + h,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            gate += tl.sum(gate_w * x, axis=0)
            up += tl.sum(up_w * x, axis=0)
        silu_gate = gate / (1.0 + tl.exp(-gate))
        tl.store(act_out + rank * intermediate_size + out_i, silu_gate * up)

    @triton.jit
    def raw_row_down_sum_kernel(
        act,
        down_raw,
        topk_ids,
        topk_weights,
        out,
        hidden_size: tl.constexpr,
        intermediate_size: tl.constexpr,
        top_k: tl.constexpr,
        BLOCK_I: tl.constexpr,
    ):
        out_h = tl.program_id(0)
        offs_i = tl.arange(0, BLOCK_I)
        acc = 0.0
        for rank in range(0, top_k):
            expert = tl.load(topk_ids + rank)
            rank_weight = tl.load(topk_weights + rank).to(tl.float32)
            rank_acc = 0.0
            row_base = (expert * hidden_size + out_h) * intermediate_size
            act_base = rank * intermediate_size
            for start in range(0, intermediate_size, BLOCK_I):
                i = start + offs_i
                mask = i < intermediate_size
                act_values = tl.load(act + act_base + i, mask=mask, other=0.0).to(tl.float32)
                down_values = tl.load(
                    down_raw + row_base + i,
                    mask=mask,
                    other=0.0,
                ).to(tl.float32)
                rank_acc += tl.sum(down_values * act_values, axis=0)
            acc += rank_weight * rank_acc
        tl.store(out + out_h, acc)

    @triton.jit
    def triton_lm_head_rowwise_int8_gemv_kernel(
        q_ptr,
        x_ptr,
        scale_ptr,
        out_ptr,
        rows: tl.constexpr,
        cols: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ) -> None:
        pid = tl.program_id(0)
        row_offsets = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = row_offsets < rows
        acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for k_start in tl.static_range(0, cols, BLOCK_K):
            col_offsets = k_start + tl.arange(0, BLOCK_K)
            col_mask = col_offsets < cols
            weights = tl.load(
                q_ptr + row_offsets[:, None] * cols + col_offsets[None, :],
                mask=row_mask[:, None] & col_mask[None, :],
                other=0,
            ).to(tl.float32)
            hidden_values = tl.load(
                x_ptr + col_offsets,
                mask=col_mask,
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(weights * hidden_values[None, :], axis=1)
        row_scales = tl.load(
            scale_ptr + row_offsets,
            mask=row_mask,
            other=0.0,
        ).to(tl.float32)
        tl.store(out_ptr + row_offsets, acc * row_scales, mask=row_mask)

else:
    triton_fla_fused_chunk_h_o_kernel = None
    triton_fla_recompute_w_only_kernel = None
    triton_fla_fused_u_h_o_kernel = None
    triton_matvec_kernel = None
    triton_full_attention_gated_o_proj_kernel = None
    triton_router_topk_stage1_kernel = None
    triton_router_topk_stage2_kernel = None
    triton_router_topk_stage2_softmax_kernel = None
    triton_decode_direct_conv_kernel = None
    triton_prefill_direct_conv_kernel = None
    triton_prefill_conv_post_prep_kernel = None
    triton_fused_input_proj_conv_kernel = None
    triton_fused_input_proj_conv_qkv_kernel = None
    triton_fused_shared_down_kernel = None
    triton_rmsnorm_kernel = None
    triton_prefill_fused_add_rmsnorm_kernel = None
    triton_head_norm_rope_kernel = None
    triton_full_attention_norm_rope_kv_write_kernel = None
    triton_linear_gated_norm_kernel = None
    triton_linear_gated_norm_from_invstd_kernel = None
    triton_native_moe_gate_up_activation_kernel = None
    triton_native_moe_down_sum_kernel = None
    raw_row_gate_up_activation_kernel = None
    raw_row_down_sum_kernel = None
    triton_lm_head_rowwise_int8_gemv_kernel = None


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return data


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(root: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def source_diff_scope(root: Path) -> str:
    diff_paths = [
        line
        for line in (git_value(root, "diff", "--name-only") or "").splitlines()
        if line and not line.startswith("output/")
    ]
    diff_stat = git_value(root, "diff", "--stat", "--", *diff_paths) if diff_paths else ""
    diff_stat = diff_stat or ""
    untracked = git_value(root, "ls-files", "--others", "--exclude-standard") or ""
    untracked = "\n".join(line for line in untracked.splitlines() if not line.startswith("output/"))
    parts: list[str] = []
    if diff_stat:
        parts.append(diff_stat)
    if untracked:
        parts.append("untracked:\n" + untracked)
    return "\n".join(parts) if parts else "clean"


def csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


VLLM_MOE_CONFIG_KEYS = {
    "BLOCK_SIZE_M",
    "BLOCK_SIZE_N",
    "BLOCK_SIZE_K",
    "GROUP_SIZE_M",
    "SPLIT_K",
    "num_warps",
    "num_stages",
    "waves_per_eu",
    "matrix_instr_nonkdim",
    "kpack",
}


def normalize_moe_override_config(parsed: Any, option_name: str) -> dict[str, Any]:
    if not isinstance(parsed, dict):
        raise ValueError(f"{option_name} must be a JSON object")
    unknown = sorted(set(parsed) - VLLM_MOE_CONFIG_KEYS)
    if unknown:
        raise ValueError(f"{option_name} contains unknown keys: {unknown}")
    normalized: dict[str, Any] = {}
    for key, raw in parsed.items():
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise ValueError(f"{option_name} value for {key} must be an integer")
        normalized[key] = int(raw)
    return normalized


def parse_moe_override_config(value: str | None) -> dict[str, Any] | None:
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--moe-override-config-json must be valid JSON: {exc}") from exc
    return normalize_moe_override_config(parsed, "--moe-override-config-json")


def normalize_moe_override_config_by_layer(value: Any) -> dict[int, dict[str, Any]] | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"--moe-override-config-by-layer-json must be valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("--moe-override-config-by-layer-json must be a JSON object keyed by model layer id")
    normalized: dict[int, dict[str, Any]] = {}
    for raw_layer, raw_config in value.items():
        if isinstance(raw_layer, bool):
            raise ValueError("--moe-override-config-by-layer-json layer ids must be integers")
        try:
            layer = int(raw_layer)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"--moe-override-config-by-layer-json layer id {raw_layer!r} is not an integer"
            ) from exc
        if layer < 0:
            raise ValueError("--moe-override-config-by-layer-json layer ids must be non-negative")
        normalized[layer] = normalize_moe_override_config(
            raw_config,
            f"--moe-override-config-by-layer-json[{layer}]",
        )
    return normalized or None


def token_ids_digest(token_ids: list[int]) -> str:
    payload = ",".join(str(item) for item in token_ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


_TOKENIZER_CACHE: dict[str, Any] = {}
_ENGINE_TENSOR_CACHE: dict[str, Any] = {}
_ENGINE_STRIPED_IMAGE_STATE: dict[str, Any] = {}
EXACT_PREFIX_CACHE_MIN_MATCH_FRACTION = 0.5
_ENGINE_EXACT_PREFIX_CACHE: dict[str, dict[str, Any]] = {}
_ENGINE_AUXILIARY_STREAM_CACHE: dict[tuple[str, int, int | None], Any] = {}


def _tensor_tree_bytes(value: Any) -> int:
    if isinstance(value, dict):
        return sum(_tensor_tree_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_tree_bytes(item) for item in value)
    numel = getattr(value, "numel", None)
    element_size = getattr(value, "element_size", None)
    if callable(numel) and callable(element_size):
        return int(numel()) * int(element_size())
    return 0


def engine_exact_prefix_cache_stats() -> dict[str, Any]:
    return {
        "entries": len(_ENGINE_EXACT_PREFIX_CACHE),
        "retained_bytes": sum(
            int(entry.get("retained_bytes") or 0)
            for entry in _ENGINE_EXACT_PREFIX_CACHE.values()
        ),
        "keys": list(_ENGINE_EXACT_PREFIX_CACHE),
        "prompt_tokens": [
            int(entry.get("prompt_tokens") or 0)
            for entry in _ENGINE_EXACT_PREFIX_CACHE.values()
        ],
    }


def evict_engine_exact_prefix_cache(*, reason: str | None = None) -> dict[str, Any]:
    before = engine_exact_prefix_cache_stats()
    evicted_keys = list(_ENGINE_EXACT_PREFIX_CACHE)
    _ENGINE_EXACT_PREFIX_CACHE.clear()
    return {
        "schema_version": 1,
        "reason": reason,
        "entries_before": before["entries"],
        "retained_bytes_before": before["retained_bytes"],
        "evicted_keys": evicted_keys,
        "entries_after": 0,
        "retained_bytes_after": 0,
    }


def _trim_engine_exact_prefix_cache(max_entries: int) -> dict[str, Any]:
    evicted_keys: list[str] = []
    evicted_bytes = 0
    while len(_ENGINE_EXACT_PREFIX_CACHE) > max_entries:
        key = next(iter(_ENGINE_EXACT_PREFIX_CACHE))
        entry = _ENGINE_EXACT_PREFIX_CACHE.pop(key)
        evicted_keys.append(key)
        evicted_bytes += int(entry.get("retained_bytes") or 0)
    return {
        "evicted_keys": evicted_keys,
        "evicted_entries": len(evicted_keys),
        "evicted_bytes": evicted_bytes,
        "max_entries": max_entries,
    }


def evict_engine_tensor_cache(
    *,
    native_moe: bool = False,
    reason: str | None = None,
) -> dict[str, Any]:
    entries_before = len(_ENGINE_TENSOR_CACHE)
    removed_keys: list[str] = []
    if native_moe:
        for key in list(_ENGINE_TENSOR_CACHE):
            if ":native_moe_gate_up" in key or ":native_moe_down" in key:
                _ENGINE_TENSOR_CACHE.pop(key, None)
                removed_keys.append(key)
    return {
        "schema_version": 1,
        "reason": reason,
        "entries_before": entries_before,
        "entries_after": len(_ENGINE_TENSOR_CACHE),
        "removed_entries": len(removed_keys),
        "removed_key_examples": removed_keys[:8],
        "policy": {"native_moe": bool(native_moe), "raw_moe": False},
    }


def cached_tokenizer(model_dir: Path, feature: str) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit(f"transformers is required for {feature}") from exc
    key = str(model_dir.resolve())
    tokenizer = _TOKENIZER_CACHE.get(key)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=True)
        _TOKENIZER_CACHE[key] = tokenizer
    return tokenizer


def engine_cache_key(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def tokenize_text(model_dir: Path, text: str) -> list[int]:
    tokenizer = cached_tokenizer(model_dir, "--input-text")
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        raise ValueError("--input-text produced zero tokens")
    return [int(item) for item in token_ids]


def decode_token_ids(model_dir: Path, token_ids: list[int]) -> list[str]:
    tokenizer = cached_tokenizer(model_dir, "--decode-output-token")
    return [tokenizer.decode([item]) for item in token_ids]


def decode_token_sequence(model_dir: Path, token_ids: list[int]) -> str:
    tokenizer = cached_tokenizer(model_dir, "--decode-output-token")
    return tokenizer.decode(token_ids, skip_special_tokens=False)


def text_config(config: dict[str, Any]) -> dict[str, Any]:
    nested = config.get("text_config")
    if isinstance(nested, dict):
        return nested
    return config


def layer_tensor_names(layer: int) -> dict[str, str]:
    prefix = f"model.language_model.layers.{layer}"
    mlp = f"{prefix}.mlp"
    return {
        "input_layernorm": f"{prefix}.input_layernorm.weight",
        "post_attention_layernorm": f"{prefix}.post_attention_layernorm.weight",
        "q_proj": f"{prefix}.self_attn.q_proj.weight",
        "k_proj": f"{prefix}.self_attn.k_proj.weight",
        "v_proj": f"{prefix}.self_attn.v_proj.weight",
        "o_proj": f"{prefix}.self_attn.o_proj.weight",
        "q_norm": f"{prefix}.self_attn.q_norm.weight",
        "k_norm": f"{prefix}.self_attn.k_norm.weight",
        "linear_A_log": f"{prefix}.linear_attn.A_log",
        "linear_conv1d": f"{prefix}.linear_attn.conv1d.weight",
        "linear_dt_bias": f"{prefix}.linear_attn.dt_bias",
        "linear_in_proj_a": f"{prefix}.linear_attn.in_proj_a.weight",
        "linear_in_proj_b": f"{prefix}.linear_attn.in_proj_b.weight",
        "linear_in_proj_qkv": f"{prefix}.linear_attn.in_proj_qkv.weight",
        "linear_in_proj_z": f"{prefix}.linear_attn.in_proj_z.weight",
        "linear_norm": f"{prefix}.linear_attn.norm.weight",
        "linear_out_proj": f"{prefix}.linear_attn.out_proj.weight",
        "router": f"{mlp}.gate.weight",
        "expert_gate_up": f"{mlp}.experts.gate_up_proj",
        "expert_down": f"{mlp}.experts.down_proj",
        "shared_gate": f"{mlp}.shared_expert_gate.weight",
        "shared_gate_proj": f"{mlp}.shared_expert.gate_proj.weight",
        "shared_up_proj": f"{mlp}.shared_expert.up_proj.weight",
        "shared_down_proj": f"{mlp}.shared_expert.down_proj.weight",
    }


def global_tensor_names() -> dict[str, str]:
    return {
        "embed_tokens": "model.language_model.embed_tokens.weight",
        "final_norm": "model.language_model.norm.weight",
        "lm_head": "lm_head.weight",
    }


def finite(value: float) -> bool:
    return math.isfinite(value)


def full_attention_enabled(attention_mode: str) -> bool:
    return attention_mode in {"full_for_full_attention", "linear_and_full_attention"}


def linear_attention_enabled(attention_mode: str) -> bool:
    return attention_mode == "linear_and_full_attention"


def linear_attention_variants() -> set[str]:
    return {
        "torch_ref",
        "vllm_fla_chunk",
        "vllm_fla_auto",
        "vllm_fla_auto_prestates",
        "vllm_fla_auto_prestates_chunk16",
        "vllm_fla_auto_prestates_chunk32",
        "vllm_fla_auto_prestates_native_chunk16",
        "vllm_fla_auto_prestates_native_refswap_chunk16",
        "vllm_fla_auto_prestates_native_refswap_chunk32",
        "vllm_fla_packed_decode",
        "vllm_fla_packed_refswap_decode",
        "vllm_fla_packed_refswap_decode_chunk16",
    }


def linear_attention_base_variant(linear_attention_variant: str) -> str:
    for suffix in ("_chunk16", "_chunk32"):
        if linear_attention_variant.endswith(suffix):
            return linear_attention_variant[: -len(suffix)]
    return linear_attention_variant


def linear_attention_chunk_size(linear_attention_variant: str) -> int | None:
    if linear_attention_variant.endswith("_chunk16"):
        return 16
    if linear_attention_variant.endswith("_chunk32"):
        return 32
    return None


def linear_attention_uses_vllm_fla(linear_attention_variant: str) -> bool:
    return linear_attention_base_variant(linear_attention_variant) in {
        "vllm_fla_chunk",
        "vllm_fla_auto",
        "vllm_fla_auto_prestates",
        "vllm_fla_auto_prestates_native",
        "vllm_fla_auto_prestates_native_refswap",
        "vllm_fla_packed_decode",
        "vllm_fla_packed_refswap_decode",
    }


def linear_attention_uses_vllm_auto_decode(linear_attention_variant: str) -> bool:
    return linear_attention_base_variant(linear_attention_variant) in {
        "vllm_fla_auto",
        "vllm_fla_auto_prestates",
        "vllm_fla_auto_prestates_native",
        "vllm_fla_auto_prestates_native_refswap",
    }


def linear_attention_uses_vllm_prestates(linear_attention_variant: str) -> bool:
    return linear_attention_base_variant(linear_attention_variant) in {
        "vllm_fla_auto_prestates",
        "vllm_fla_auto_prestates_native",
        "vllm_fla_auto_prestates_native_refswap",
    }


def linear_attention_uses_native_vllm_decode_state(linear_attention_variant: str) -> bool:
    return linear_attention_base_variant(linear_attention_variant) in {
        "vllm_fla_auto_prestates_native",
        "vllm_fla_auto_prestates_native_refswap",
    }


def linear_attention_uses_native_vllm_decode_state_refswap(linear_attention_variant: str) -> bool:
    return linear_attention_base_variant(linear_attention_variant) == "vllm_fla_auto_prestates_native_refswap"


def linear_attention_uses_packed_state_refswap(linear_attention_variant: str) -> bool:
    return linear_attention_base_variant(linear_attention_variant) == "vllm_fla_packed_refswap_decode"


def moe_variants() -> set[str]:
    return {
        "native_selected_expert_consumer",
        "resident_dispatch",
        "padded_batched",
        "count_batched",
        "vllm_fused",
        "vllm_fused_inplace",
        "vllm_fused_m32_n16_k512",
        "vllm_fused_prefill_m32_n32_decode_m32_n16_k512",
        "vllm_fused_prefill_m32_n32_decode_m32_n16_k512_prealloc",
    }


def vllm_fused_moe_variants() -> set[str]:
    return {
        "vllm_fused",
        "vllm_fused_inplace",
        "vllm_fused_m32_n16_k512",
        "vllm_fused_prefill_m32_n32_decode_m32_n16_k512",
        "vllm_fused_prefill_m32_n32_decode_m32_n16_k512_prealloc",
    }


def native_moe_consumer_variants() -> set[str]:
    return {"native_selected_expert_consumer"}


NATIVE_MOE_CONSUMER_CONFIG = {
    "layout_block_i": 32,
    "layout_block_h": 64,
    "gate_block_h": 512,
    "gate_num_warps": 8,
    "down_block_i": 128,
    "down_num_warps": 4,
}


def native_moe_consumer_config(moe_variant: str) -> dict[str, int]:
    if moe_variant != "native_selected_expert_consumer":
        raise ValueError(f"{moe_variant} is not a native MoE consumer variant")
    return dict(NATIVE_MOE_CONSUMER_CONFIG)


VLLM_MOE_M32_N16_K512_CONFIG = {
    "BLOCK_SIZE_M": 32,
    "BLOCK_SIZE_N": 16,
    "BLOCK_SIZE_K": 512,
    "GROUP_SIZE_M": 1,
    "SPLIT_K": 1,
    "num_warps": 4,
    "num_stages": 2,
    "waves_per_eu": 2,
    "matrix_instr_nonkdim": 16,
    "kpack": 1,
}


VLLM_MOE_M32_N32_K512_CONFIG = {
    "BLOCK_SIZE_M": 32,
    "BLOCK_SIZE_N": 32,
    "BLOCK_SIZE_K": 512,
    "GROUP_SIZE_M": 1,
    "SPLIT_K": 1,
    "num_warps": 4,
    "num_stages": 2,
    "waves_per_eu": 2,
    "matrix_instr_nonkdim": 16,
    "kpack": 1,
}


def vllm_moe_override_config(
    moe_variant: str,
    mode: str | None = None,
    moe_override_config: dict[str, Any] | None = None,
    moe_override_config_by_layer: dict[int, dict[str, Any]] | None = None,
    layer: int | None = None,
) -> dict[str, Any] | None:
    if mode == "decode" and layer is not None and moe_override_config_by_layer is not None:
        layer_config = moe_override_config_by_layer.get(int(layer))
        if layer_config is not None:
            return dict(layer_config)
    if moe_override_config is not None:
        return dict(moe_override_config)
    if moe_variant == "vllm_fused_m32_n16_k512":
        return dict(VLLM_MOE_M32_N16_K512_CONFIG)
    if moe_variant in {
        "native_selected_expert_consumer",
        "vllm_fused_prefill_m32_n32_decode_m32_n16_k512",
        "vllm_fused_prefill_m32_n32_decode_m32_n16_k512_prealloc",
    }:
        if mode == "prefill":
            return dict(VLLM_MOE_M32_N32_K512_CONFIG)
        if mode == "decode":
            return dict(VLLM_MOE_M32_N16_K512_CONFIG)
        raise ValueError(
            "mode must be prefill or decode for "
            f"{moe_variant}"
        )
    return None


def router_variants() -> set[str]:
    return {"torch", "torch_out", "triton_topk", "triton_topk_softmax"}


def linear_attention_input_proj_variants() -> set[str]:
    return {
        "separate",
        "decode_fused",
        "decode_fused_t",
        "decode_fused_t_triton",
        "decode_fused_t_conv_triton",
        "decode_fused_t_conv_qkv_triton",
        "prefill_fused_t_decode_fused_t_conv_triton",
        "prefill_fused_t_decode_fused_t_conv_qkv_triton",
    }


def linear_attention_output_proj_variants() -> set[str]:
    return {"torch", "triton_matvec"}


def linear_attention_conv_variants() -> set[str]:
    return {"conv1d", "decode_direct", "decode_direct_triton"}


def linear_attention_gated_norm_variants() -> set[str]:
    return {"torch", "triton"}


def rmsnorm_variants() -> set[str]:
    return {"torch", "triton"}


def full_attention_variants() -> set[str]:
    return {"sdpa", "decode_grouped_bmm", "decode_grouped_bmm_bf16"}


def full_attention_proj_variants() -> set[str]:
    return {"torch", "triton_matvec", "triton_fused_qkv_matvec"}


def full_attention_norm_rope_variants() -> set[str]:
    return {"torch", "triton"}


def full_attention_kv_cache_layouts() -> set[str]:
    return {"seq", "grouped"}


def lm_head_variants() -> set[str]:
    return {"view", "pretransposed", "pretransposed_out", "int8_certified_global_tie"}


def shared_expert_proj_variants() -> set[str]:
    return {"torch", "triton_matvec", "triton_fused_in_matvec", "triton_fused_in_down_matvec"}


def planned_contract(
    manifest: dict[str, Any],
    model_dir: Path,
    layers: list[int],
    mode: str,
    seq_len: int,
    tokens: int,
    attention_mode: str,
    moe_variant: str,
    moe_override_config: dict[str, Any] | None,
    moe_override_config_by_layer: dict[int, dict[str, Any]] | None,
    overlap_shared_expert_moe: bool,
    overlap_shared_expert_router_moe: bool,
    router_variant: str,
    linear_attention_variant: str,
    linear_attention_input_proj_variant: str,
    linear_attention_output_proj_variant: str,
    linear_attention_conv_variant: str,
    linear_attention_conv_state_refswap: bool,
    linear_attention_gated_norm_variant: str,
    linear_attention_post_conv_prep_block_t: int | None,
    linear_attention_prefill_conv_block_t: int | None,
    linear_attention_prefill_conv_block_c: int | None,
    linear_attention_prefill_conv_num_warps: int | None,
    linear_attention_prefill_conv_post_prep_fusion: bool,
    linear_attention_prefill_vllm_state_handoff: bool,
    linear_attention_prefill_fused_h_o: bool,
    linear_attention_prefill_fused_u_h_o: bool,
    linear_attention_chunk_gdn_internal_timing: bool,
    rmsnorm_variant: str,
    full_attention_variant: str,
    full_attention_proj_variant: str,
    full_attention_norm_rope_variant: str,
    full_attention_kv_cache_layout: str,
    full_attention_fused_gate_o_proj: bool,
    lm_head_variant: str,
    shared_expert_proj_variant: str,
    retained_attention_fast_path: bool,
    skip_layer_dispatch_metadata: bool,
    include_shared_expert: bool,
    input_source: str,
    include_lm_head: bool,
    full_attention_fused_norm_rope_kv_write: bool = False,
    shared_expert_overlap_stream_priority: int | None = None,
) -> dict[str, Any]:
    core = manifest["core_shapes"]
    moe = manifest["moe"]
    layer_types = manifest["layers"]["layer_types"]
    for layer in layers:
        if layer < 0 or layer >= len(layer_types):
            raise ValueError(f"layer {layer} is outside manifest layer range 0..{len(layer_types) - 1}")
    return {
        "model_dir": str(model_dir),
        "layers": layers,
        "layer_types": [layer_types[layer] for layer in layers],
        "mode": mode,
        "logical_seq_len": seq_len,
        "context_len": seq_len if mode == "decode" else tokens,
        "tokens": tokens,
        "attention_mode": attention_mode,
        "moe_variant": moe_variant,
        "moe_override_config": (
            vllm_moe_override_config(moe_variant, mode, moe_override_config)
            if moe_variant in vllm_fused_moe_variants()
            else None
        ),
        "moe_override_config_by_layer": (
            {
                str(layer): vllm_moe_override_config(
                    moe_variant, "decode", moe_override_config, moe_override_config_by_layer, layer
                )
                for layer in layers
                if vllm_moe_override_config(
                    moe_variant, "decode", moe_override_config, moe_override_config_by_layer, layer
                )
                != vllm_moe_override_config(moe_variant, "decode", moe_override_config)
            }
            if moe_variant in vllm_fused_moe_variants() and moe_override_config_by_layer is not None
            else None
        ),
        "overlap_shared_expert_moe": overlap_shared_expert_moe,
        "overlap_shared_expert_router_moe": overlap_shared_expert_router_moe,
        "shared_expert_overlap_stream_priority": shared_expert_overlap_stream_priority,
        "router_variant": router_variant,
        "linear_attention_variant": linear_attention_variant,
        "linear_attention_input_proj_variant": linear_attention_input_proj_variant,
        "linear_attention_output_proj_variant": linear_attention_output_proj_variant,
        "linear_attention_conv_variant": linear_attention_conv_variant,
        "linear_attention_conv_state_refswap": linear_attention_conv_state_refswap,
        "linear_attention_gated_norm_variant": linear_attention_gated_norm_variant,
        "linear_attention_post_conv_prep_block_t": linear_attention_post_conv_prep_block_t,
        "linear_attention_prefill_conv_block_t": linear_attention_prefill_conv_block_t,
        "linear_attention_prefill_conv_block_c": linear_attention_prefill_conv_block_c,
        "linear_attention_prefill_conv_num_warps": linear_attention_prefill_conv_num_warps,
        "linear_attention_prefill_conv_effective_block_t": linear_attention_prefill_conv_block_t or 16,
        "linear_attention_prefill_conv_effective_block_c": linear_attention_prefill_conv_block_c or 32,
        "linear_attention_prefill_conv_effective_num_warps": linear_attention_prefill_conv_num_warps or 4,
        "linear_attention_prefill_conv_post_prep_fusion": linear_attention_prefill_conv_post_prep_fusion,
        "linear_attention_prefill_vllm_state_handoff": linear_attention_prefill_vllm_state_handoff,
        "linear_attention_prefill_fused_h_o": linear_attention_prefill_fused_h_o,
        "linear_attention_prefill_fused_u_h_o": linear_attention_prefill_fused_u_h_o,
        "linear_attention_chunk_gdn_internal_timing": linear_attention_chunk_gdn_internal_timing,
        "rmsnorm_variant": rmsnorm_variant,
        "full_attention_variant": full_attention_variant,
        "full_attention_proj_variant": full_attention_proj_variant,
        "full_attention_norm_rope_variant": full_attention_norm_rope_variant,
        "full_attention_kv_cache_layout": full_attention_kv_cache_layout,
        "full_attention_fused_gate_o_proj": full_attention_fused_gate_o_proj,
        "full_attention_fused_norm_rope_kv_write": full_attention_fused_norm_rope_kv_write,
        "lm_head_variant": lm_head_variant,
        "shared_expert_proj_variant": shared_expert_proj_variant,
        "retained_attention_fast_path": retained_attention_fast_path,
        "skip_layer_dispatch_metadata": skip_layer_dispatch_metadata,
        "input_source": input_source,
        "include_lm_head": include_lm_head,
        "batch_size": 1,
        "dtype": manifest["slice_lab"]["dtype"],
        "hidden_size": int(core["hidden_size"]),
        "num_experts": int(moe["num_experts"]),
        "top_k": int(moe["top_k"]),
        "expert_intermediate_size": int(moe["expert_intermediate_size"]),
        "approximations": [
            "real target layernorm and MoE weights for the selected layers",
            (
                {
                    "sdpa": "full-attention layers use real q/k/v/o, q/k norm, RoPE, torch SDPA, output gate, and KV buffers",
                    "decode_grouped_bmm": "full-attention layers use real q/k/v/o, q/k norm, RoPE, and a decode-only grouped-BMM attention candidate before output gate/projection",
                    "decode_grouped_bmm_bf16": "full-attention layers use real q/k/v/o, q/k norm, RoPE, and a decode-only grouped-BMM attention candidate with BF16 QK scores before output gate/projection",
                }[full_attention_variant]
                if full_attention_enabled(attention_mode)
                else "attention is a zero-output stub"
            ),
            (
                {
                    "torch": "full-attention q/k/v/o projections use torch matmul",
                    "triton_matvec": "full-attention one-token decode q/k/v/o projections use a guarded Triton matvec kernel and fall back otherwise",
                    "triton_fused_qkv_matvec": "full-attention one-token decode q/k/v projections use one fused Triton matvec before the guarded Triton output projection",
                }[full_attention_proj_variant]
                if full_attention_enabled(attention_mode)
                else "full-attention projection variant is inactive because full attention is stubbed"
            ),
            (
                "full-attention one-token decode fuses output gate sigmoid/multiply into the Triton o_proj matvec"
                if full_attention_fused_gate_o_proj and full_attention_enabled(attention_mode)
                else "full-attention output gate and o_proj remain separate"
            ),
            (
                "full-attention one-token decode fuses q/k head norm+RoPE with k/v cache writes"
                if full_attention_fused_norm_rope_kv_write and full_attention_enabled(attention_mode)
                else "full-attention norm+RoPE and KV cache writes remain separate"
            ),
            (
                {
                    "torch": "full-attention q/k head RMSNorm and RoPE use the torch reference path",
                    "triton": "full-attention one-token decode q/k head RMSNorm+RoPE use a guarded fused Triton kernel and fall back otherwise",
                }[full_attention_norm_rope_variant]
                if full_attention_enabled(attention_mode)
                else "full-attention norm+RoPE variant is inactive because full attention is stubbed"
            ),
            (
                "linear-attention layers use real Gated Delta Net projections, conv state, recurrent state, gated norm, and output projection"
                if linear_attention_enabled(attention_mode)
                else "linear-attention layers still use a zero-output attention stub"
            ),
            (
                {
                    "torch_ref": "linear-attention core uses the local PyTorch reference",
                    "vllm_fla_chunk": "linear-attention core uses vLLM Triton/FLA chunk_gated_delta_rule after PyTorch grouped causal conv",
                    "vllm_fla_auto": "linear-attention core uses vLLM Triton/FLA chunk_gated_delta_rule for prefill and fused recurrent GDN update for decode",
                    "vllm_fla_auto_prestates": "linear-attention core uses the vLLM auto path and precomputes vLLM-layout initial SSM states for one-token decode",
                    "vllm_fla_auto_prestates_chunk16": "linear-attention core uses the vLLM auto-prestates path but runs prefill chunk_gated_delta_rule with a 16-token chunk size",
                    "vllm_fla_auto_prestates_chunk32": "linear-attention core uses the vLLM auto-prestates path but runs prefill chunk_gated_delta_rule with a 32-token chunk size",
                    "vllm_fla_auto_prestates_native_chunk16": "linear-attention core uses the vLLM auto-prestates path, keeps decode SSM state in vLLM layout, and runs prefill with a 16-token chunk size",
                    "vllm_fla_auto_prestates_native_refswap_chunk16": "linear-attention core uses the vLLM auto-prestates path, keeps decode SSM state in vLLM layout, promotes decode state by tensor reference, and runs prefill with a 16-token chunk size",
                    "vllm_fla_auto_prestates_native_refswap_chunk32": "linear-attention core uses the vLLM auto-prestates path, keeps decode SSM state in vLLM layout, promotes decode state by tensor reference, and runs prefill with a 32-token chunk size",
                    "vllm_fla_packed_decode": "linear-attention core uses vLLM Triton/FLA chunk_gated_delta_rule for prefill and a packed recurrent GDN update for one-token decode",
                    "vllm_fla_packed_refswap_decode": "linear-attention core uses vLLM Triton/FLA chunk_gated_delta_rule for prefill and a packed recurrent GDN update that mutates packed decode SSM state in place for serving-like one-token decode",
                    "vllm_fla_packed_refswap_decode_chunk16": "linear-attention core uses vLLM Triton/FLA chunk_gated_delta_rule with a 16-token chunk size for prefill and a packed recurrent GDN update that mutates packed decode SSM state in place for serving-like one-token decode",
                }[linear_attention_variant]
            ),
            (
                {
                    "separate": "linear-attention input projections run as separate qkv/z/a/b matmuls",
                    "decode_fused": "linear-attention decode uses a pre-concatenated qkv/z/a/b projection weight for one-token decode and falls back to separate matmuls otherwise",
                    "decode_fused_t": "linear-attention decode uses a pre-concatenated and pretransposed qkv/z/a/b projection weight for one-token decode and falls back to separate matmuls otherwise",
                    "decode_fused_t_triton": "linear-attention decode uses a pre-concatenated and pretransposed qkv/z/a/b projection weight with a guarded Triton matvec kernel for one-token decode and falls back otherwise",
                    "decode_fused_t_conv_triton": "linear-attention decode uses a guarded Triton kernel that fuses pretransposed qkv/z/a/b projection with the decode direct causal-conv update for one-token decode and falls back otherwise",
                    "decode_fused_t_conv_qkv_triton": "linear-attention decode uses a guarded Triton projection+conv kernel that writes q/k/v directly into vLLM recurrent-GDN layout buffers for one-token decode and falls back otherwise",
                    "prefill_fused_t_decode_fused_t_conv_triton": "linear-attention prefill uses one pretransposed fused qkv/z/a/b projection matmul; one-token decode uses the retained guarded Triton projection+conv kernel",
                    "prefill_fused_t_decode_fused_t_conv_qkv_triton": "linear-attention prefill uses one pretransposed fused qkv/z/a/b projection matmul; one-token decode uses the guarded projection+conv q/k/v layout kernel",
                }[linear_attention_input_proj_variant]
                if linear_attention_enabled(attention_mode)
                else "linear-attention input projection variant is inactive because linear attention is stubbed"
            ),
            (
                {
                    "torch": "linear-attention output projection uses torch matmul",
                    "triton_matvec": "linear-attention one-token decode output projection uses a guarded Triton matvec kernel and falls back otherwise",
                }[linear_attention_output_proj_variant]
                if linear_attention_enabled(attention_mode)
                else "linear-attention output projection variant is inactive because linear attention is stubbed"
            ),
            (
                {
                    "conv1d": "linear-attention causal conv uses grouped torch conv1d",
                    "decode_direct": "linear-attention prefill uses a direct depthwise shift-sum; one-token decode uses a preallocated depthwise window and direct weighted sum",
                    "decode_direct_triton": "linear-attention prefill uses a guarded Triton direct causal conv when available; one-token decode uses a guarded Triton depthwise update kernel",
                }[linear_attention_conv_variant]
                if linear_attention_enabled(attention_mode)
                else "linear-attention conv variant is inactive because linear attention is stubbed"
            ),
            (
                {
                    "torch": "linear-attention gated norm and z gate use the torch elementwise path",
                    "triton": "linear-attention one-token decode gated norm and z gate use a guarded Triton row kernel and fall back otherwise",
                }[linear_attention_gated_norm_variant]
                if linear_attention_enabled(attention_mode)
                else "linear-attention gated norm variant is inactive because linear attention is stubbed"
            ),
            {
                "torch": "RMSNorm uses the torch elementwise reference path",
                "triton": "one-token decode RMSNorm uses a guarded Triton kernel for layer and final RMSNorm and falls back otherwise",
            }[rmsnorm_variant],
            (
                "input hidden states come from target embed_tokens for supplied token ids"
                if input_source != "synthetic_random"
                else "synthetic random hidden states, not real model activations"
            ),
            (
                {
                    "view": "last selected-layer token runs through target final RMSNorm and lm_head using lm_head.weight.T view; partial-layer slices are not text-quality claims",
                    "pretransposed": "last selected-layer token runs through target final RMSNorm and a pretransposed contiguous lm_head RHS; partial-layer slices are not text-quality claims",
                    "pretransposed_out": "one-token decode uses target final RMSNorm and torch.mm into a preallocated logits buffer with a pretransposed lm_head RHS; other modes fall back to pretransposed logits",
                    "int8_certified_global_tie": "final RMSNorm feeds the fixed row1932 BM8/BK128/W8 rowwise-int8 full-logit kernel and exposes a 1024-row exact BF16 certificate ceiling, with an async cutoff proof before full global argmax",
                }[lm_head_variant]
                if include_lm_head
                else "lm_head is not evaluated"
            ),
            "decode mode validates synthetic previous state/cache lifetimes, not real prompt history",
            {
                "native_selected_expert_consumer": "strict vLLM fused prefill over resident raw BF16 experts plus a parameterized leading-layer native-layout decode hotset and row-contiguous selected-expert Triton decode elsewhere",
                "resident_dispatch": "candidate MoE uses resident dispatch per populated expert",
                "padded_batched": "candidate MoE uses padded batched torch.bmm over populated expert buckets",
                "count_batched": "candidate MoE groups populated experts by identical active-row count and runs grouped torch.bmm without padded rows",
                "vllm_fused": "candidate MoE calls vLLM fused_experts as an external fused segmented MoE implementation",
                "vllm_fused_inplace": "candidate MoE calls vLLM fused_experts with inplace=True after computing the shared expert from the unmodified post-attention hidden state",
                "vllm_fused_m32_n16_k512": "candidate MoE calls vLLM fused_experts with a fixed AMD395 one-token override config using BLOCK_SIZE_M=32, BLOCK_SIZE_N=16, and BLOCK_SIZE_K=512",
                "vllm_fused_prefill_m32_n32_decode_m32_n16_k512": "candidate MoE calls vLLM fused_experts with BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_K=512 for prefill and the retained M32/N16/K512 override for decode",
                "vllm_fused_prefill_m32_n32_decode_m32_n16_k512_prealloc": "candidate MoE uses the same vLLM M32/N32 prefill and M32/N16 decode configs, but one-token decode decomposes fused_experts into preallocated vLLM dispatch, activation, dispatch, and moe_sum workspaces",
            }[moe_variant],
            (
                f"custom vLLM fused MoE override config is active: {vllm_moe_override_config(moe_variant, mode, moe_override_config)}"
                if moe_override_config is not None
                else "no custom vLLM fused MoE override config was supplied"
            ),
            (
                f"decode-only per-layer vLLM fused MoE override configs are active for layers {sorted(moe_override_config_by_layer)}"
                if moe_override_config_by_layer is not None
                else "no per-layer vLLM fused MoE override configs were supplied"
            ),
            (
                "one-token decode overlaps the shared expert with non-inplace vLLM fused MoE on a side CUDA stream"
                if overlap_shared_expert_moe
                else "shared expert executes as a separate post-routed-MoE stage unless an overlap probe is enabled"
            ),
            (
                "one-token decode starts the shared expert side stream before router/top-k so router and routed MoE can overlap with shared expert work"
                if overlap_shared_expert_router_moe
                else "shared expert side-stream overlap starts after router/top-k unless the router-overlap probe is enabled"
            ),
            (
                f"shared-expert overlap uses torch.cuda.Stream(priority={shared_expert_overlap_stream_priority})"
                if shared_expert_overlap_stream_priority is not None
                else "shared-expert overlap uses the default CUDA stream priority"
            ),
            {
                "torch": "router logits and top-k use the existing torch matmul and torch.topk path",
                "torch_out": "one-token decode router logits use a pretransposed RHS and torch.mm into a preallocated logits buffer before torch.topk; other shapes fall back",
                "triton_topk": "one-token decode router matvec and top-k use a guarded two-stage Triton kernel; other shapes fall back",
                "triton_topk_softmax": "one-token decode router matvec, top-k, and top-k softmax weights use guarded Triton kernels; other shapes fall back",
            }[router_variant],
            (
                "one-token decode uses a retained-route attention fast path that inlines the current full/linear attention parent branches"
                if retained_attention_fast_path
                else "attention uses the generic retained parent functions"
            ),
            (
                {
                    "torch": "shared expert projections use torch matmul",
                    "triton_matvec": "shared expert one-token decode gate/up/down projections use a guarded Triton matvec kernel and fall back otherwise",
                    "triton_fused_in_matvec": "shared expert one-token decode gate/gate-proj/up-proj use one fused Triton matvec before the guarded Triton down projection",
                    "triton_fused_in_down_matvec": "shared expert one-token decode uses one fused Triton input matvec plus a fused activation/down/final-gate Triton matvec",
                }[shared_expert_proj_variant]
                if include_shared_expert
                else "shared expert projection variant is inactive because shared expert is disabled"
            ),
        ],
    }


def run_with_torch(
    *,
    manifest: dict[str, Any],
    model_dir: Path,
    layers: list[int],
    mode: str,
    seq_len: int,
    tokens: int,
    device: str,
    warmup: int,
    iters: int,
    seed: int,
    moe_chunk_size: int,
    attention_mode: str,
    moe_variant: str,
    moe_override_config: dict[str, Any] | None,
    moe_override_config_by_layer: dict[int, dict[str, Any]] | None,
    overlap_shared_expert_moe: bool,
    overlap_shared_expert_router_moe: bool,
    router_variant: str,
    linear_attention_variant: str,
    linear_attention_input_proj_variant: str,
    linear_attention_output_proj_variant: str,
    linear_attention_conv_variant: str,
    linear_attention_conv_state_refswap: bool,
    linear_attention_gated_norm_variant: str,
    linear_attention_post_conv_prep_block_t: int | None,
    linear_attention_prefill_conv_block_t: int | None,
    linear_attention_prefill_conv_block_c: int | None,
    linear_attention_prefill_conv_num_warps: int | None,
    linear_attention_prefill_conv_post_prep_fusion: bool,
    linear_attention_prefill_vllm_state_handoff: bool,
    linear_attention_prefill_fused_h_o: bool,
    linear_attention_prefill_fused_u_h_o: bool,
    linear_attention_chunk_gdn_internal_timing: bool,
    rmsnorm_variant: str,
    full_attention_variant: str,
    full_attention_proj_variant: str,
    full_attention_norm_rope_variant: str,
    full_attention_kv_cache_layout: str,
    full_attention_fused_gate_o_proj: bool,
    lm_head_variant: str,
    shared_expert_proj_variant: str,
    include_shared_expert: bool,
    input_token_ids: list[int] | None,
    input_text: str | None,
    include_lm_head: bool,
    decode_output_token: bool,
    logit_topk: int,
    attention_substage_timing: bool,
    moe_substage_timing: bool,
    collect_resident_stage_timeline: bool,
    measurement_mode: str,
    decode_loop_steps: int,
    decode_sampling: str,
    sampling_temperature: float,
    sampling_top_k: int,
    decode_stop_token_ids: set[int],
    prefill_seed_output: bool = False,
    decode_loop_fast_housekeeping: bool = False,
    decode_loop_defer_token_cpu_sync: bool = False,
    decode_loop_token_cpu_sync_interval: int = 1,
    decode_loop_diagnostic: bool = False,
    overlap_decode_state_promotion_lm_head: bool = False,
    attention_cluster_timing: bool = False,
    attention_event_timing: bool = False,
    retained_attention_fast_path: bool = False,
    skip_layer_dispatch_metadata: bool = False,
    full_attention_fused_norm_rope_kv_write: bool = False,
    cuda_graph_replay_timing: bool = False,
    moe_overlap_event_timing: bool = False,
    shared_expert_overlap_stream_priority: int | None = None,
    reuse_tensor_cache: bool = False,
    resident_native_decode_hotset_layers: int = 0,
    exact_prefix_cache: bool = False,
    exact_prefix_cache_max_entries: int = 1,
    exact_prefix_cache_max_tokens: int = 8192,
    load_only: bool = False,
) -> dict[str, Any]:
    engine_wall_start = time.perf_counter()
    python_import_wall_start = engine_wall_start
    try:
        import torch
        import torch.nn.functional as F
        from safetensors import safe_open
    except ImportError as exc:
        raise SystemExit("torch and safetensors are required for --execute") from exc
    try:
        from torch.nn.attention.bias import causal_lower_right
    except Exception:
        causal_lower_right = None
    python_import_wall_time_ms = (time.perf_counter() - python_import_wall_start) * 1000.0
    runtime_setup_wall_start = time.perf_counter()
    striped_manifest_env = os.environ.get("AMD395_STRIPED_IMAGE_MANIFEST")
    striped_library_env = os.environ.get("AMD395_STRIPED_IMAGE_LOADER")
    striped_native_report_env = os.environ.get("AMD395_STRIPED_IMAGE_NATIVE_REPORT")
    striped_expected_xor_env = os.environ.get("AMD395_STRIPED_IMAGE_EXPECTED_XOR")
    striped_expected_sum_env = os.environ.get("AMD395_STRIPED_IMAGE_EXPECTED_SUM")
    striped_lane_sha256_env = [
        os.environ.get("AMD395_STRIPED_IMAGE_LANE0_SHA256"),
        os.environ.get("AMD395_STRIPED_IMAGE_LANE1_SHA256"),
    ]
    striped_chunk_bytes = int(os.environ.get("AMD395_STRIPED_IMAGE_CHUNK_BYTES", str(512 << 20)))
    striped_image_loader_report: dict[str, Any] | None = None

    if moe_override_config is not None:
        moe_override_config = normalize_moe_override_config(moe_override_config, "moe_override_config")
    moe_override_config_by_layer = normalize_moe_override_config_by_layer(moe_override_config_by_layer)

    if mode not in {"prefill", "decode"}:
        raise ValueError("mode must be prefill or decode")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if iters <= 0:
        raise ValueError("iters must be positive")
    if tokens <= 0:
        raise ValueError("tokens must be positive")
    if moe_chunk_size <= 0:
        raise ValueError("moe_chunk_size must be positive")
    if logit_topk <= 0:
        raise ValueError("logit_topk must be positive")
    if decode_loop_steps < 0:
        raise ValueError("decode_loop_steps must be non-negative")
    if decode_loop_fast_housekeeping and not decode_loop_steps:
        decode_loop_fast_housekeeping = False
    if decode_loop_defer_token_cpu_sync and not decode_loop_steps:
        decode_loop_defer_token_cpu_sync = False
    if decode_loop_token_cpu_sync_interval < 0:
        raise ValueError("decode_loop_token_cpu_sync_interval must be non-negative")
    if decode_loop_defer_token_cpu_sync:
        decode_loop_token_cpu_sync_interval = 0
    if not decode_loop_steps:
        decode_loop_token_cpu_sync_interval = 1
        decode_loop_diagnostic = False
    if overlap_decode_state_promotion_lm_head and not decode_loop_steps:
        overlap_decode_state_promotion_lm_head = False
    if decode_sampling not in {"argmax", "top_k"}:
        raise ValueError("decode_sampling must be argmax or top_k")
    if sampling_temperature <= 0:
        raise ValueError("sampling_temperature must be positive")
    if sampling_top_k <= 0:
        raise ValueError("sampling_top_k must be positive")
    if decode_loop_steps and mode == "decode" and tokens != 1:
        raise ValueError("--decode-loop-steps with decode mode requires exactly one input token")
    if decode_loop_steps and mode not in {"prefill", "decode"}:
        raise ValueError("--decode-loop-steps requires prefill or decode mode")
    if decode_loop_steps and (input_token_ids is None or not include_lm_head):
        raise ValueError("--decode-loop-steps requires --input-token-ids and --include-lm-head")
    if prefill_seed_output and (mode != "prefill" or input_token_ids is None or not include_lm_head):
        raise ValueError(
            "prefill seed output requires prefill mode, input token ids, and the LM head"
        )
    if exact_prefix_cache_max_entries < 0:
        raise ValueError("exact_prefix_cache_max_entries must be non-negative")
    if exact_prefix_cache_max_tokens < 0:
        raise ValueError("exact_prefix_cache_max_tokens must be non-negative")
    if exact_prefix_cache and exact_prefix_cache_max_entries == 0:
        raise ValueError("exact prefix cache requires at least one cache entry")
    if exact_prefix_cache and not reuse_tensor_cache:
        raise ValueError("exact prefix cache requires the resident tensor cache")
    if exact_prefix_cache and (
        mode != "prefill"
        or measurement_mode != "resident_only"
        or input_token_ids is None
        or (
            not decode_loop_steps
            and not prefill_seed_output
            and len(input_token_ids or []) <= exact_prefix_cache_max_tokens
        )
        or decode_sampling != "argmax"
        or attention_substage_timing
        or moe_substage_timing
        or collect_resident_stage_timeline
        or attention_cluster_timing
        or attention_event_timing
        or moe_overlap_event_timing
        or cuda_graph_replay_timing
    ):
        raise ValueError(
            "exact prefix cache currently requires deterministic resident prefill+decode "
            "without diagnostic replay passes"
        )
    if attention_mode not in {"stub", "full_for_full_attention", "linear_and_full_attention"}:
        raise ValueError("attention_mode must be stub, full_for_full_attention, or linear_and_full_attention")
    if measurement_mode not in {"correctness", "resident_only"}:
        raise ValueError("measurement_mode must be correctness or resident_only")
    if moe_variant not in moe_variants():
        raise ValueError(
            "moe_variant must be resident_dispatch, padded_batched, count_batched, "
            "vllm_fused, vllm_fused_inplace, vllm_fused_m32_n16_k512, "
            "vllm_fused_prefill_m32_n32_decode_m32_n16_k512, or "
            "vllm_fused_prefill_m32_n32_decode_m32_n16_k512_prealloc"
        )
    if (
        (moe_override_config is not None or moe_override_config_by_layer is not None)
        and moe_variant not in vllm_fused_moe_variants()
    ):
        raise ValueError("MoE override configs require a vllm_fused MoE variant")
    if router_variant not in router_variants():
        raise ValueError("router_variant must be torch, torch_out, triton_topk, or triton_topk_softmax")
    if linear_attention_variant not in linear_attention_variants():
        raise ValueError(
            "linear_attention_variant must be torch_ref, vllm_fla_chunk, "
            "vllm_fla_auto, vllm_fla_auto_prestates, "
            "vllm_fla_auto_prestates_chunk16, vllm_fla_auto_prestates_chunk32, "
            "vllm_fla_auto_prestates_native_chunk16, "
            "vllm_fla_auto_prestates_native_refswap_chunk16, "
            "vllm_fla_auto_prestates_native_refswap_chunk32, "
            "vllm_fla_packed_decode, vllm_fla_packed_refswap_decode, "
            "or vllm_fla_packed_refswap_decode_chunk16"
        )
    if linear_attention_uses_packed_state_refswap(linear_attention_variant):
        if mode == "decode" and decode_loop_steps:
            raise ValueError(
                "vllm_fla_packed_refswap_decode mutates packed decode state in-place; "
                "use prefill mode for decode-loop gates"
            )
        if mode == "decode" and (warmup != 0 or iters != 1):
            raise ValueError(
                "vllm_fla_packed_refswap_decode requires serving-like one-token decode timing "
                "with --warmup 0 --iters 1"
            )
        if mode == "decode" and (
            attention_substage_timing
            or moe_substage_timing
            or moe_overlap_event_timing
            or attention_event_timing
            or collect_resident_stage_timeline
        ):
            raise ValueError(
                "vllm_fla_packed_refswap_decode mutates packed decode state in-place; "
                "disable diagnostic passes with --no-attention-substage-timing "
                "--no-moe-substage-timing --no-moe-overlap-event-timing "
                "--no-attention-event-timing "
                "--no-collect-resident-stage-timeline for direct decode"
            )
    if linear_attention_input_proj_variant not in linear_attention_input_proj_variants():
        raise ValueError(
            "linear_attention_input_proj_variant must be separate, decode_fused, "
            "decode_fused_t, decode_fused_t_triton, decode_fused_t_conv_triton, "
            "decode_fused_t_conv_qkv_triton, prefill_fused_t_decode_fused_t_conv_triton, "
            "or prefill_fused_t_decode_fused_t_conv_qkv_triton"
        )
    if linear_attention_output_proj_variant not in linear_attention_output_proj_variants():
        raise ValueError("linear_attention_output_proj_variant must be torch or triton_matvec")
    if linear_attention_conv_variant not in linear_attention_conv_variants():
        raise ValueError("linear_attention_conv_variant must be conv1d, decode_direct, or decode_direct_triton")
    if linear_attention_gated_norm_variant not in linear_attention_gated_norm_variants():
        raise ValueError("linear_attention_gated_norm_variant must be torch or triton")
    if linear_attention_post_conv_prep_block_t is not None and linear_attention_post_conv_prep_block_t not in {
        8,
        16,
        32,
        64,
        128,
        256,
    }:
        raise ValueError("--linear-attention-post-conv-prep-block-t must be one of 8, 16, 32, 64, 128, or 256")
    if linear_attention_prefill_conv_block_t is not None and linear_attention_prefill_conv_block_t not in {
        8,
        16,
        32,
        64,
    }:
        raise ValueError("--linear-attention-prefill-conv-block-t must be one of 8, 16, 32, or 64")
    if linear_attention_prefill_conv_block_c is not None and linear_attention_prefill_conv_block_c not in {
        16,
        32,
        64,
    }:
        raise ValueError("--linear-attention-prefill-conv-block-c must be one of 16, 32, or 64")
    if linear_attention_prefill_conv_num_warps is not None and linear_attention_prefill_conv_num_warps not in {
        4,
        8,
    }:
        raise ValueError("--linear-attention-prefill-conv-num-warps must be one of 4 or 8")
    prefill_conv_block_t = linear_attention_prefill_conv_block_t or 16
    prefill_conv_block_c = linear_attention_prefill_conv_block_c or 32
    prefill_conv_num_warps = linear_attention_prefill_conv_num_warps or 4
    if full_attention_variant not in full_attention_variants():
        raise ValueError("full_attention_variant must be sdpa, decode_grouped_bmm, or decode_grouped_bmm_bf16")
    if full_attention_proj_variant not in full_attention_proj_variants():
        raise ValueError("full_attention_proj_variant must be torch, triton_matvec, or triton_fused_qkv_matvec")
    if full_attention_fused_gate_o_proj and full_attention_proj_variant not in {
        "triton_matvec",
        "triton_fused_qkv_matvec",
    }:
        raise ValueError("--full-attention-fused-gate-o-proj requires a Triton full-attention projection variant")
    if full_attention_norm_rope_variant not in full_attention_norm_rope_variants():
        raise ValueError("full_attention_norm_rope_variant must be torch or triton")
    if full_attention_kv_cache_layout not in full_attention_kv_cache_layouts():
        raise ValueError("full_attention_kv_cache_layout must be seq or grouped")
    if full_attention_fused_norm_rope_kv_write and full_attention_norm_rope_variant != "triton":
        raise ValueError("--full-attention-fused-norm-rope-kv-write requires --full-attention-norm-rope-variant triton")
    if lm_head_variant not in lm_head_variants():
        raise ValueError(
            "lm_head_variant must be view, pretransposed, pretransposed_out, "
            "or int8_certified_global_tie"
        )
    if shared_expert_proj_variant not in shared_expert_proj_variants():
        raise ValueError(
            "shared_expert_proj_variant must be torch, triton_matvec, "
            "triton_fused_in_matvec, or triton_fused_in_down_matvec"
        )
    if rmsnorm_variant not in rmsnorm_variants():
        raise ValueError("rmsnorm_variant must be torch or triton")
    triton_linear_output_proj_decode = False
    triton_linear_input_proj_decode = False
    triton_fused_linear_input_proj_conv_decode = False
    triton_fused_linear_input_proj_conv_qkv_decode = False
    torch_out_router_decode = False
    triton_router_topk_decode = False
    triton_router_topk_softmax_decode = False
    triton_linear_conv_decode = False
    triton_linear_conv_prefill = False
    triton_linear_gated_norm_decode = False
    triton_linear_gated_norm_prefill = False
    triton_full_attention_proj_decode = False
    triton_full_attention_fused_qkv_decode = False
    triton_full_attention_norm_rope_decode = False
    triton_full_attention_fused_gate_o_proj_decode = False
    triton_full_attention_fused_norm_rope_kv_write_decode = False
    triton_shared_expert_proj_decode = False
    triton_shared_expert_fused_input_decode = False
    triton_shared_expert_fused_down_decode = False
    triton_rmsnorm_decode = False
    triton_rmsnorm_prefill = False
    native_moe_consumer_decode = False

    def refresh_runtime_flags() -> None:
        nonlocal triton_linear_output_proj_decode, triton_linear_input_proj_decode
        nonlocal triton_fused_linear_input_proj_conv_decode, triton_fused_linear_input_proj_conv_qkv_decode
        nonlocal torch_out_router_decode
        nonlocal triton_router_topk_decode, triton_router_topk_softmax_decode
        nonlocal triton_linear_conv_decode, triton_linear_conv_prefill
        nonlocal triton_linear_gated_norm_decode, triton_linear_gated_norm_prefill
        nonlocal triton_full_attention_proj_decode
        nonlocal triton_full_attention_fused_qkv_decode, triton_full_attention_norm_rope_decode
        nonlocal triton_full_attention_fused_gate_o_proj_decode
        nonlocal triton_full_attention_fused_norm_rope_kv_write_decode
        nonlocal triton_shared_expert_proj_decode, triton_shared_expert_fused_input_decode
        nonlocal triton_shared_expert_fused_down_decode, triton_rmsnorm_decode
        nonlocal triton_rmsnorm_prefill, native_moe_consumer_decode
        triton_linear_output_proj_decode = (
            linear_attention_output_proj_variant == "triton_matvec" and mode == "decode" and tokens == 1
        )
        triton_linear_input_proj_decode = (
            linear_attention_input_proj_variant == "decode_fused_t_triton" and mode == "decode" and tokens == 1
        )
        triton_fused_linear_input_proj_conv_decode = (
            linear_attention_input_proj_variant
            in {
                "decode_fused_t_conv_triton",
                "decode_fused_t_conv_qkv_triton",
                "prefill_fused_t_decode_fused_t_conv_triton",
                "prefill_fused_t_decode_fused_t_conv_qkv_triton",
            }
            and linear_attention_enabled(attention_mode)
            and mode == "decode"
            and tokens == 1
        )
        triton_fused_linear_input_proj_conv_qkv_decode = (
            linear_attention_input_proj_variant
            in {
                "decode_fused_t_conv_qkv_triton",
                "prefill_fused_t_decode_fused_t_conv_qkv_triton",
            }
            and linear_attention_enabled(attention_mode)
            and mode == "decode"
            and tokens == 1
        )
        torch_out_router_decode = router_variant == "torch_out" and mode == "decode" and tokens == 1
        triton_router_topk_decode = (
            router_variant in {"triton_topk", "triton_topk_softmax"} and mode == "decode" and tokens == 1
        )
        triton_router_topk_softmax_decode = (
            router_variant == "triton_topk_softmax" and mode == "decode" and tokens == 1
        )
        triton_linear_conv_decode = (
            linear_attention_conv_variant == "decode_direct_triton"
            and linear_attention_enabled(attention_mode)
            and mode == "decode"
            and tokens == 1
        )
        triton_linear_conv_prefill = (
            linear_attention_conv_variant == "decode_direct_triton"
            and linear_attention_enabled(attention_mode)
            and mode == "prefill"
        )
        triton_linear_gated_norm_decode = (
            linear_attention_gated_norm_variant == "triton"
            and linear_attention_enabled(attention_mode)
            and mode == "decode"
            and tokens == 1
        )
        triton_linear_gated_norm_prefill = (
            linear_attention_enabled(attention_mode)
            and mode == "prefill"
            and decode_loop_steps > 0
            and moe_variant in native_moe_consumer_variants()
        )
        triton_full_attention_proj_decode = (
            full_attention_proj_variant in {"triton_matvec", "triton_fused_qkv_matvec"}
            and full_attention_enabled(attention_mode)
            and mode == "decode"
            and tokens == 1
        )
        triton_full_attention_fused_qkv_decode = (
            full_attention_proj_variant == "triton_fused_qkv_matvec"
            and full_attention_enabled(attention_mode)
            and mode == "decode"
            and tokens == 1
        )
        triton_full_attention_norm_rope_decode = (
            full_attention_norm_rope_variant == "triton"
            and full_attention_enabled(attention_mode)
            and mode == "decode"
            and tokens == 1
        )
        triton_full_attention_fused_gate_o_proj_decode = (
            full_attention_fused_gate_o_proj
            and triton_full_attention_proj_decode
            and full_attention_enabled(attention_mode)
            and mode == "decode"
            and tokens == 1
        )
        triton_full_attention_fused_norm_rope_kv_write_decode = (
            full_attention_fused_norm_rope_kv_write
            and triton_full_attention_norm_rope_decode
            and full_attention_enabled(attention_mode)
            and mode == "decode"
            and tokens == 1
        )
        triton_shared_expert_proj_decode = (
            shared_expert_proj_variant in {"triton_matvec", "triton_fused_in_matvec", "triton_fused_in_down_matvec"}
            and include_shared_expert
            and mode == "decode"
            and tokens == 1
        )
        triton_shared_expert_fused_input_decode = (
            shared_expert_proj_variant in {"triton_fused_in_matvec", "triton_fused_in_down_matvec"}
            and include_shared_expert
            and mode == "decode"
            and tokens == 1
        )
        triton_shared_expert_fused_down_decode = (
            shared_expert_proj_variant == "triton_fused_in_down_matvec"
            and include_shared_expert
            and mode == "decode"
            and tokens == 1
        )
        triton_rmsnorm_decode = rmsnorm_variant == "triton" and mode == "decode" and tokens == 1
        triton_rmsnorm_prefill = (
            rmsnorm_variant == "triton"
            and mode == "prefill"
            and tokens > 1
            and moe_variant in native_moe_consumer_variants()
        )
        native_moe_consumer_decode = (
            moe_variant in native_moe_consumer_variants()
            and mode == "decode"
            and tokens == 1
        )

    def check_runtime_kernel_requirements() -> None:
        if triton_full_attention_proj_decode and triton_matvec_kernel is None:
            raise SystemExit("triton is required for one-token decode --full-attention-proj-variant triton_matvec")
        if triton_full_attention_norm_rope_decode and triton_head_norm_rope_kernel is None:
            raise SystemExit("triton is required for one-token decode --full-attention-norm-rope-variant triton")
        if (
            triton_full_attention_fused_norm_rope_kv_write_decode
            and triton_full_attention_norm_rope_kv_write_kernel is None
        ):
            raise SystemExit("triton is required for one-token decode --full-attention-fused-norm-rope-kv-write")
        if (
            triton_full_attention_fused_gate_o_proj_decode
            and triton_full_attention_gated_o_proj_kernel is None
        ):
            raise SystemExit("triton is required for one-token decode --full-attention-fused-gate-o-proj")
        if triton_shared_expert_proj_decode and triton_matvec_kernel is None:
            raise SystemExit("triton is required for one-token decode --shared-expert-proj-variant triton_matvec")
        if triton_shared_expert_fused_down_decode and triton_fused_shared_down_kernel is None:
            raise SystemExit(
                "triton is required for one-token decode --shared-expert-proj-variant triton_fused_in_down_matvec"
            )
        if triton_linear_input_proj_decode and triton_matvec_kernel is None:
            raise SystemExit(
                "triton is required for one-token decode --linear-attention-input-proj-variant decode_fused_t_triton"
            )
        if triton_fused_linear_input_proj_conv_decode and linear_attention_conv_variant != "decode_direct_triton":
            raise SystemExit(
                f"--linear-attention-input-proj-variant {linear_attention_input_proj_variant} requires "
                "--linear-attention-conv-variant decode_direct_triton for one-token decode"
            )
        if triton_fused_linear_input_proj_conv_qkv_decode and not linear_attention_uses_vllm_auto_decode(
            linear_attention_variant
        ):
            raise SystemExit(
                "--linear-attention-input-proj-variant qkv-layout decode variants require a "
                "vllm_fla_auto* decode linear-attention variant"
            )
        if triton_fused_linear_input_proj_conv_qkv_decode and triton_fused_input_proj_conv_qkv_kernel is None:
            raise SystemExit(
                "triton is required for one-token decode --linear-attention-input-proj-variant "
                "qkv-layout variants"
            )
        if triton_fused_linear_input_proj_conv_decode and triton_fused_input_proj_conv_kernel is None:
            raise SystemExit(
                "triton is required for one-token decode --linear-attention-input-proj-variant "
                f"{linear_attention_input_proj_variant}"
            )
        if triton_linear_output_proj_decode and triton_matvec_kernel is None:
            raise SystemExit("triton is required for one-token decode --linear-attention-output-proj-variant triton_matvec")
        if (
            triton_router_topk_decode
            and (
                triton_router_topk_stage1_kernel is None
                or triton_router_topk_stage2_kernel is None
                or (triton_router_topk_softmax_decode and triton_router_topk_stage2_softmax_kernel is None)
            )
        ):
            raise SystemExit(f"triton is required for one-token decode --router-variant {router_variant}")
        if triton_linear_conv_decode and triton_decode_direct_conv_kernel is None:
            raise SystemExit("triton is required for one-token decode --linear-attention-conv-variant decode_direct_triton")
        if triton_linear_conv_prefill and triton_prefill_direct_conv_kernel is None:
            raise SystemExit("triton is required for prefill --linear-attention-conv-variant decode_direct_triton")
        if linear_attention_prefill_conv_post_prep_fusion and triton_prefill_conv_post_prep_kernel is None:
            raise SystemExit("triton is required for --linear-attention-prefill-conv-post-prep-fusion")
        if triton_linear_gated_norm_decode and triton_linear_gated_norm_kernel is None:
            raise SystemExit("triton is required for one-token decode --linear-attention-gated-norm-variant triton")
        if (
            triton_linear_gated_norm_prefill
            and triton_linear_gated_norm_from_invstd_kernel is None
        ):
            raise SystemExit("triton is required for exact-staged prefill gated norm")
        if triton_rmsnorm_decode and triton_rmsnorm_kernel is None:
            raise SystemExit("triton is required for one-token decode --rmsnorm-variant triton")
        if triton_rmsnorm_prefill and (
            triton_rmsnorm_kernel is None
            or triton_prefill_fused_add_rmsnorm_kernel is None
        ):
            raise SystemExit("triton is required for prefill residual normalization")
        if native_moe_consumer_decode and (
            raw_row_gate_up_activation_kernel is None
            or raw_row_down_sum_kernel is None
        ):
            raise SystemExit("triton is required for raw-row selected-expert one-token decode")
        if native_moe_consumer_decode and resident_native_decode_hotset_layers and (
            triton_native_moe_gate_up_activation_kernel is None
            or triton_native_moe_down_sum_kernel is None
        ):
            raise SystemExit("triton is required for native-layout selected-expert one-token decode")

    refresh_runtime_flags()
    check_runtime_kernel_requirements()
    native_moe_config = (
        native_moe_consumer_config(moe_variant)
        if moe_variant in native_moe_consumer_variants()
        else native_moe_consumer_config("native_selected_expert_consumer")
    )
    if resident_native_decode_hotset_layers < 0 or resident_native_decode_hotset_layers > len(layers):
        raise ValueError("resident_native_decode_hotset_layers must be between 0 and the layer count")
    if resident_native_decode_hotset_layers and not (
        moe_variant in native_moe_consumer_variants()
        and decode_loop_steps > 0
        and measurement_mode == "resident_only"
        and reuse_tensor_cache
    ):
        raise ValueError(
            "resident native decode hotset requires the resident native MoE decode loop and tensor cache"
        )
    resident_native_decode_hotset_layer_indices = tuple(
        layers[:resident_native_decode_hotset_layers]
    )
    native_moe_consumer_decode_layout_required = bool(
        resident_native_decode_hotset_layer_indices
    )
    native_moe_consumer_memory_safe = False
    native_moe_consumer_prefill_official_linear_surface = (
        moe_variant in native_moe_consumer_variants()
        and (
            decode_loop_steps > 0
            or prefill_seed_output
            or (
                exact_prefix_cache
                and input_token_ids is not None
                and len(input_token_ids) > exact_prefix_cache_max_tokens
            )
        )
    )
    fused_experts = None
    fused_experts_override_config_fn = None
    vllm_moe_dispatch_kernel = None
    vllm_moe_apply_activation = None
    vllm_moe_activation_silu = None
    vllm_moe_ops = None
    wvsplitk_cu_count = None
    if moe_variant in vllm_fused_moe_variants() or moe_variant in native_moe_consumer_variants():
        try:
            from vllm.model_executor.layers.fused_moe import override_config as vllm_override_config
            from vllm.model_executor.layers.fused_moe.activation import MoEActivation as vllm_moe_activation
            from vllm.model_executor.layers.fused_moe.activation import apply_moe_activation as vllm_apply_moe_activation
            from vllm.model_executor.layers.fused_moe.fused_moe import (
                dispatch_fused_moe_kernel as vllm_dispatch_fused_moe_kernel,
            )
            from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts as vllm_fused_experts
            from vllm import _custom_ops as vllm_custom_ops
        except ImportError as exc:
            raise SystemExit(f"vLLM fused_experts is required for --moe-variant {moe_variant}") from exc
        if moe_variant == "vllm_fused_inplace":
            disable_inplace = vllm_fused_experts.__globals__.get("disable_inplace")
            if callable(disable_inplace) and disable_inplace():
                raise SystemExit(
                    "vLLM fused_experts disables inplace=True for this Torch build; "
                    "--moe-variant vllm_fused_inplace is unavailable"
                )
        fused_experts = vllm_fused_experts
        fused_experts_override_config_fn = vllm_override_config
        vllm_moe_dispatch_kernel = vllm_dispatch_fused_moe_kernel
        vllm_moe_apply_activation = vllm_apply_moe_activation
        vllm_moe_activation_silu = vllm_moe_activation.SILU
        vllm_moe_ops = vllm_custom_ops
        wvsplitk_cu_count = int(torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count)
    vllm_fused_post_conv_prep = None
    vllm_fused_post_conv_kernel = None
    vllm_chunk_gated_delta_rule = None
    vllm_chunk_local_cumsum = None
    vllm_chunk_scaled_dot_kkt_fwd = None
    vllm_solve_tril = None
    vllm_recompute_w_u_fwd = None
    vllm_chunk_gated_delta_rule_fwd_h = None
    vllm_chunk_fwd_o = None
    vllm_l2norm_fwd = None
    vllm_fused_recurrent_gdn_update = None
    vllm_fused_recurrent_gdn_packed_decode = None
    if linear_attention_uses_vllm_fla(linear_attention_variant):
        try:
            from vllm.model_executor.layers.fla.ops import (
                chunk_gated_delta_rule as vllm_chunk_gated_delta_rule_fn,
            )
            from vllm.model_executor.layers.mamba.gdn_linear_attn import (
                fused_recurrent_gated_delta_rule_packed_decode as vllm_fused_recurrent_gdn_packed_decode_fn,
            )
            from vllm.model_executor.layers.mamba.gdn_linear_attn import (
                fused_post_conv_prep as vllm_fused_post_conv_prep_fn,
            )
            from vllm.model_executor.layers.mamba.gdn_linear_attn import (
                fused_sigmoid_gating_delta_rule_update as vllm_fused_recurrent_gdn_update_fn,
            )
        except ImportError as exc:
            raise SystemExit(
                f"vLLM FLA GDN kernels are required for --linear-attention-variant {linear_attention_variant}"
            ) from exc
        vllm_fused_post_conv_prep = vllm_fused_post_conv_prep_fn
        vllm_fused_post_conv_kernel = vllm_fused_post_conv_prep_fn.__globals__.get("_fused_post_conv_kernel")
        vllm_chunk_gated_delta_rule = vllm_chunk_gated_delta_rule_fn
        vllm_fused_recurrent_gdn_update = vllm_fused_recurrent_gdn_update_fn
        vllm_fused_recurrent_gdn_packed_decode = vllm_fused_recurrent_gdn_packed_decode_fn
        if linear_attention_chunk_size(linear_attention_variant) is not None:
            try:
                from vllm.model_executor.layers.fla.ops.chunk_delta_h import (
                    chunk_gated_delta_rule_fwd_h as vllm_chunk_gated_delta_rule_fwd_h_fn,
                )
                from vllm.model_executor.layers.fla.ops.chunk_o import (
                    chunk_fwd_o as vllm_chunk_fwd_o_fn,
                )
                from vllm.model_executor.layers.fla.ops.chunk_scaled_dot_kkt import (
                    chunk_scaled_dot_kkt_fwd as vllm_chunk_scaled_dot_kkt_fwd_fn,
                )
                from vllm.model_executor.layers.fla.ops.cumsum import (
                    chunk_local_cumsum as vllm_chunk_local_cumsum_fn,
                )
                from vllm.model_executor.layers.fla.ops.l2norm import (
                    l2norm_fwd as vllm_l2norm_fwd_fn,
                )
                from vllm.model_executor.layers.fla.ops.solve_tril import (
                    solve_tril as vllm_solve_tril_fn,
                )
                from vllm.model_executor.layers.fla.ops.wy_fast import (
                    recompute_w_u_fwd as vllm_recompute_w_u_fwd_fn,
                )
            except ImportError as exc:
                raise SystemExit(
                    f"vLLM internal FLA kernels are required for --linear-attention-variant {linear_attention_variant}"
                ) from exc
            vllm_chunk_local_cumsum = vllm_chunk_local_cumsum_fn
            vllm_chunk_scaled_dot_kkt_fwd = vllm_chunk_scaled_dot_kkt_fwd_fn
            vllm_solve_tril = vllm_solve_tril_fn
            vllm_recompute_w_u_fwd = vllm_recompute_w_u_fwd_fn
            vllm_chunk_gated_delta_rule_fwd_h = vllm_chunk_gated_delta_rule_fwd_h_fn
            vllm_chunk_fwd_o = vllm_chunk_fwd_o_fn
            vllm_l2norm_fwd = vllm_l2norm_fwd_fn

    config_path = model_dir / "config.json"
    index_path = model_dir / "model.safetensors.index.json"
    config_sha256 = sha256_file(config_path)
    index_sha256 = sha256_file(index_path)
    config = text_config(load_json(config_path))
    index_data = load_json(index_path)
    weight_map = index_data["weight_map"]
    layer_types = manifest["layers"]["layer_types"]

    hidden = int(config["hidden_size"])
    vocab = int(config["vocab_size"])
    heads = int(config["num_attention_heads"])
    kv_heads = int(config["num_key_value_heads"])
    head_dim = int(config["head_dim"])
    q_dim = heads * head_dim
    kv_dim = kv_heads * head_dim
    rotary_dim = int(head_dim * float(config["partial_rotary_factor"]))
    rope_theta = float(config["rope_parameters"]["rope_theta"])
    linear = manifest["attention"]["linear_attention"]
    linear_key_heads = int(linear["linear_num_key_heads"])
    linear_value_heads = int(linear["linear_num_value_heads"])
    linear_key_head_dim = int(linear["linear_key_head_dim"])
    linear_value_head_dim = int(linear["linear_value_head_dim"])
    linear_key_dim = linear_key_heads * linear_key_head_dim
    linear_value_dim = linear_value_heads * linear_value_head_dim
    linear_conv_kernel_dim = int(linear["linear_conv_kernel_dim"])
    linear_conv_dim = 2 * linear_key_dim + linear_value_dim
    linear_conv_state_len = linear_conv_kernel_dim - 1
    if linear_attention_prefill_conv_post_prep_fusion and not (
        mode == "prefill"
        and linear_attention_conv_variant == "decode_direct_triton"
        and linear_attention_enabled(attention_mode)
        and linear_attention_uses_vllm_fla(linear_attention_variant)
        and linear_key_heads == 16
        and linear_value_heads == 32
        and linear_key_head_dim == 128
        and linear_value_head_dim == 128
    ):
        raise ValueError(
            "--linear-attention-prefill-conv-post-prep-fusion requires retained Qwen prefill "
            "with vLLM FLA and decode_direct_triton causal conv"
        )
    experts = int(config["num_experts"])
    top_k = int(config["num_experts_per_tok"])
    if triton_router_topk_decode and top_k != 8:
        raise ValueError(f"--router-variant {router_variant} currently requires top_k=8")
    intermediate = int(config["moe_intermediate_size"])
    shared_intermediate = int(config.get("shared_expert_intermediate_size", 0))
    rms_norm_eps = float(config.get("rms_norm_eps", 1e-6))
    dtype = torch.bfloat16
    exact_prefix_cache_entry: dict[str, Any] | None = None
    exact_prefix_cache_entry_key: str | None = None
    exact_prefix_cache_key: str | None = None
    exact_prefix_runtime_contract_key: str | None = None
    exact_prefix_cache_record: dict[str, Any] = {
        "enabled": bool(exact_prefix_cache),
        "lookup": "disabled",
        "hit": False,
        "match_kind": "none",
        "stored": False,
        "matched_tokens": 0,
        "suffix_tokens": int(len(input_token_ids or [])),
        "request_tokens": int(len(input_token_ids or [])),
        "max_entries": int(exact_prefix_cache_max_entries),
        "max_tokens": int(exact_prefix_cache_max_tokens),
        "minimum_match_fraction": EXACT_PREFIX_CACHE_MIN_MATCH_FRACTION,
        "minimum_reusable_tokens": math.ceil(
            len(input_token_ids or []) * EXACT_PREFIX_CACHE_MIN_MATCH_FRACTION
        ),
        "matching_rule": "longest_exact_token_prefix_and_runtime_contract",
    }
    if exact_prefix_cache:
        trim_record = _trim_engine_exact_prefix_cache(exact_prefix_cache_max_entries)
        exact_prefix_cache_record["pre_lookup_eviction"] = trim_record
        if len(input_token_ids or []) > exact_prefix_cache_max_tokens:
            exact_prefix_cache_record["lookup"] = "bypass_prompt_too_long"
        else:
            exact_prefix_runtime_contract = {
                "schema": "exact-prefix-cache-runtime-contract/v2",
                "model_dir": str(model_dir.resolve()),
                "config_sha256": config_sha256,
                "index_sha256": index_sha256,
                "device": device,
                "dtype": str(dtype),
                "layers": [int(layer) for layer in layers],
                "mode": mode,
                "seq_len": int(seq_len),
                "attention_mode": attention_mode,
                "moe_variant": moe_variant,
                "moe_override_config": moe_override_config,
                "moe_override_config_by_layer": moe_override_config_by_layer,
                "overlap_shared_expert_moe": bool(overlap_shared_expert_moe),
                "overlap_shared_expert_router_moe": bool(overlap_shared_expert_router_moe),
                "shared_expert_overlap_stream_priority": shared_expert_overlap_stream_priority,
                "router_variant": router_variant,
                "linear_attention_variant": linear_attention_variant,
                "linear_attention_input_proj_variant": linear_attention_input_proj_variant,
                "linear_attention_output_proj_variant": linear_attention_output_proj_variant,
                "linear_attention_conv_variant": linear_attention_conv_variant,
                "linear_attention_conv_state_refswap": bool(linear_attention_conv_state_refswap),
                "linear_attention_gated_norm_variant": linear_attention_gated_norm_variant,
                "linear_attention_post_conv_prep_block_t": linear_attention_post_conv_prep_block_t,
                "linear_attention_prefill_conv_block_t": linear_attention_prefill_conv_block_t,
                "linear_attention_prefill_conv_block_c": linear_attention_prefill_conv_block_c,
                "linear_attention_prefill_conv_num_warps": linear_attention_prefill_conv_num_warps,
                "linear_attention_prefill_conv_post_prep_fusion": bool(
                    linear_attention_prefill_conv_post_prep_fusion
                ),
                "linear_attention_prefill_vllm_state_handoff": bool(
                    linear_attention_prefill_vllm_state_handoff
                ),
                "linear_attention_prefill_fused_h_o": bool(linear_attention_prefill_fused_h_o),
                "linear_attention_prefill_fused_u_h_o": bool(linear_attention_prefill_fused_u_h_o),
                "rmsnorm_variant": rmsnorm_variant,
                "full_attention_variant": full_attention_variant,
                "full_attention_proj_variant": full_attention_proj_variant,
                "full_attention_norm_rope_variant": full_attention_norm_rope_variant,
                "full_attention_kv_cache_layout": full_attention_kv_cache_layout,
                "full_attention_fused_gate_o_proj": bool(full_attention_fused_gate_o_proj),
                "full_attention_fused_norm_rope_kv_write": bool(
                    full_attention_fused_norm_rope_kv_write
                ),
                "lm_head_variant": lm_head_variant,
                "shared_expert_proj_variant": shared_expert_proj_variant,
                "include_shared_expert": bool(include_shared_expert),
                "include_lm_head": bool(include_lm_head),
                "prefill_seed_output": bool(prefill_seed_output),
                "resident_native_decode_hotset_layers": int(resident_native_decode_hotset_layers),
                "decode_sampling": decode_sampling,
            }
            exact_prefix_runtime_contract_key = engine_cache_key(exact_prefix_runtime_contract)
            exact_tokens = tuple(int(item) for item in input_token_ids or [])
            exact_prefix_cache_key = engine_cache_key(
                {
                    "schema": "exact-prefix-cache-entry-key/v2",
                    "runtime_contract_key": exact_prefix_runtime_contract_key,
                    "prompt_token_ids": exact_tokens,
                }
            )
            exact_prefix_cache_record["key"] = exact_prefix_cache_key
            exact_prefix_cache_record["runtime_contract_key"] = exact_prefix_runtime_contract_key
            lookup_started = time.perf_counter()
            candidate_entry = _ENGINE_EXACT_PREFIX_CACHE.get(exact_prefix_cache_key)
            if (
                isinstance(candidate_entry, dict)
                and candidate_entry.get("runtime_contract_key") == exact_prefix_runtime_contract_key
                and candidate_entry.get("token_ids") == exact_tokens
                and int(candidate_entry.get("prompt_tokens") or 0) == len(exact_tokens)
            ):
                exact_prefix_cache_entry_key = exact_prefix_cache_key
                exact_prefix_cache_entry = _ENGINE_EXACT_PREFIX_CACHE.pop(exact_prefix_cache_entry_key)
                _ENGINE_EXACT_PREFIX_CACHE[exact_prefix_cache_entry_key] = exact_prefix_cache_entry
                exact_prefix_cache_record.update(
                    {
                        "lookup": "exact_hit_pending_state_restore",
                        "hit": True,
                        "match_kind": "exact_hit",
                        "matched_tokens": len(exact_tokens),
                        "suffix_tokens": 0,
                        "retained_bytes": int(exact_prefix_cache_entry.get("retained_bytes") or 0),
                    }
                )
            else:
                longest_prefix: tuple[str, dict[str, Any], int] | None = None
                minimum_reusable_tokens = math.ceil(
                    len(exact_tokens) * EXACT_PREFIX_CACHE_MIN_MATCH_FRACTION
                )
                for cached_key, cached_entry in _ENGINE_EXACT_PREFIX_CACHE.items():
                    if not isinstance(cached_entry, dict):
                        continue
                    if cached_entry.get("runtime_contract_key") != exact_prefix_runtime_contract_key:
                        continue
                    cached_tokens = cached_entry.get("token_ids")
                    cached_count = int(cached_entry.get("prompt_tokens") or 0)
                    if (
                        not isinstance(cached_tokens, tuple)
                        or cached_count <= 0
                        or cached_count < minimum_reusable_tokens
                        or cached_count >= len(exact_tokens)
                        or len(cached_tokens) != cached_count
                        or cached_tokens != exact_tokens[:cached_count]
                    ):
                        continue
                    if longest_prefix is None or cached_count > longest_prefix[2]:
                        longest_prefix = (cached_key, cached_entry, cached_count)
                if longest_prefix is None:
                    exact_prefix_cache_record["lookup"] = "prefix_miss"
                else:
                    exact_prefix_cache_entry_key, _candidate_entry, matched_tokens = longest_prefix
                    exact_prefix_cache_entry = _ENGINE_EXACT_PREFIX_CACHE.pop(exact_prefix_cache_entry_key)
                    _ENGINE_EXACT_PREFIX_CACHE[exact_prefix_cache_entry_key] = exact_prefix_cache_entry
                    exact_prefix_cache_record.update(
                        {
                            "lookup": "strict_prefix_hit_pending_state_restore",
                            "hit": True,
                            "match_kind": "strict_prefix_hit",
                            "matched_tokens": matched_tokens,
                            "suffix_tokens": len(exact_tokens) - matched_tokens,
                            "matched_entry_key": exact_prefix_cache_entry_key,
                            "retained_bytes": int(exact_prefix_cache_entry.get("retained_bytes") or 0),
                        }
                    )
            exact_prefix_cache_record["lookup_wall_time_ms"] = (
                time.perf_counter() - lookup_started
            ) * 1000.0
    runtime_setup_wall_time_ms = (time.perf_counter() - runtime_setup_wall_start) * 1000.0
    tensor_cache_hits = 0
    tensor_cache_misses = 0
    tensor_cache_by_scope = {
        "raw_weights": {"hits": 0, "misses": 0},
        "derived_layouts": {"hits": 0, "misses": 0},
    }
    # The resident tensor cache stores only raw model weights and derived weight
    # layouts. Runtime buffers still depend on prompt length, but weight cache
    # reuse must survive independent requests with different token counts and
    # candidate flags. Keep raw weights scoped only by the immutable model files;
    # derived layouts are keyed by their transform name.
    raw_tensor_cache_prefix = engine_cache_key(
        {
            "cache_scope": "raw_model_weights",
            "model_dir": str(model_dir.resolve()),
            "config_sha256": config_sha256,
            "index_sha256": index_sha256,
            "device": device,
        }
    )
    derived_tensor_cache_prefix = engine_cache_key(
        {
            "cache_scope": "derived_weight_layouts",
            "model_dir": str(model_dir.resolve()),
            "config_sha256": config_sha256,
            "index_sha256": index_sha256,
            "device": device,
        }
    )
    runtime_tensor_cache_prefix = engine_cache_key(
        {
            "cache_scope": "runtime_contract",
            "model_dir": str(model_dir.resolve()),
            "config_sha256": config_sha256,
            "index_sha256": index_sha256,
            "device": device,
            "layers": layers,
            "mode": mode,
            "seq_len": seq_len,
            "attention_mode": attention_mode,
            "moe_variant": moe_variant,
            "moe_override_config": moe_override_config,
            "moe_override_config_by_layer": moe_override_config_by_layer,
            "overlap_shared_expert_moe": overlap_shared_expert_moe,
            "overlap_shared_expert_router_moe": overlap_shared_expert_router_moe,
            "shared_expert_overlap_stream_priority": shared_expert_overlap_stream_priority,
            "router_variant": router_variant,
            "linear_attention_variant": linear_attention_variant,
            "linear_attention_input_proj_variant": linear_attention_input_proj_variant,
            "linear_attention_output_proj_variant": linear_attention_output_proj_variant,
            "linear_attention_conv_variant": linear_attention_conv_variant,
            "linear_attention_conv_state_refswap": linear_attention_conv_state_refswap,
            "linear_attention_gated_norm_variant": linear_attention_gated_norm_variant,
            "linear_attention_post_conv_prep_block_t": linear_attention_post_conv_prep_block_t,
            "linear_attention_prefill_conv_block_t": linear_attention_prefill_conv_block_t,
            "linear_attention_prefill_conv_block_c": linear_attention_prefill_conv_block_c,
            "linear_attention_prefill_conv_num_warps": linear_attention_prefill_conv_num_warps,
            "linear_attention_prefill_conv_effective_block_t": prefill_conv_block_t,
            "linear_attention_prefill_conv_effective_block_c": prefill_conv_block_c,
            "linear_attention_prefill_conv_effective_num_warps": prefill_conv_num_warps,
            "attention_cluster_timing": attention_cluster_timing,
            "attention_event_timing": attention_event_timing,
            "retained_attention_fast_path": retained_attention_fast_path,
            "rmsnorm_variant": rmsnorm_variant,
            "full_attention_variant": full_attention_variant,
            "full_attention_proj_variant": full_attention_proj_variant,
            "full_attention_norm_rope_variant": full_attention_norm_rope_variant,
            "full_attention_kv_cache_layout": full_attention_kv_cache_layout,
            "full_attention_fused_gate_o_proj": full_attention_fused_gate_o_proj,
            "full_attention_fused_norm_rope_kv_write": full_attention_fused_norm_rope_kv_write,
            "decode_projection_owner_providers": {
                "full_attention_qkv": "row2100_selected_triton_matvec",
                "shared_expert_fused_input": "row2100_selected_triton_matvec",
                "shared_expert_down": "direct_wvsplitk",
                "full_attention_output": "rocm_unquantized_gemm",
                "linear_attention_output": "direct_wvsplitk",
            },
            "lm_head_variant": lm_head_variant,
            "shared_expert_proj_variant": shared_expert_proj_variant,
            "include_shared_expert": include_shared_expert,
            "needs_embed_tokens": input_token_ids is not None,
            "include_lm_head": include_lm_head,
        }
    )

    def sync() -> None:
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

    shared_expert_overlap_stream = None
    shared_expert_overlap_stream_cache_hit = False
    shared_expert_overlap_stream_cache_key: tuple[str, int, int | None] | None = None
    shared_expert_overlap_stream_identifier_sha256: str | None = None
    if (
        (overlap_shared_expert_moe or overlap_shared_expert_router_moe)
        and device.startswith("cuda")
        and torch.cuda.is_available()
    ):
        stream_priority = (
            None
            if shared_expert_overlap_stream_priority is None
            else int(shared_expert_overlap_stream_priority)
        )
        shared_expert_overlap_stream_cache_key = (
            "shared_expert_overlap",
            int(torch.cuda.current_device()),
            stream_priority,
        )
        shared_expert_overlap_stream = _ENGINE_AUXILIARY_STREAM_CACHE.get(
            shared_expert_overlap_stream_cache_key
        )
        shared_expert_overlap_stream_cache_hit = shared_expert_overlap_stream is not None
        if shared_expert_overlap_stream is None:
            if stream_priority is None:
                shared_expert_overlap_stream = torch.cuda.Stream()
            else:
                shared_expert_overlap_stream = torch.cuda.Stream(priority=stream_priority)
            _ENGINE_AUXILIARY_STREAM_CACHE[
                shared_expert_overlap_stream_cache_key
            ] = shared_expert_overlap_stream
        shared_expert_overlap_stream_identifier_sha256 = hashlib.sha256(
            str(int(shared_expert_overlap_stream.cuda_stream)).encode("ascii")
        ).hexdigest()
    decode_state_promotion_stream = (
        torch.cuda.Stream()
        if overlap_decode_state_promotion_lm_head
        and device.startswith("cuda")
        and torch.cuda.is_available()
        else None
    )

    def should_overlap_shared_expert_moe() -> bool:
        return (
            overlap_shared_expert_moe
            and shared_expert_overlap_stream is not None
            and include_shared_expert
            and mode == "decode"
            and tokens == 1
            and (
                moe_variant in vllm_fused_moe_variants()
                or moe_variant in native_moe_consumer_variants()
            )
            and moe_variant != "vllm_fused_inplace"
        )

    def should_overlap_shared_expert_router_moe() -> bool:
        return (
            overlap_shared_expert_router_moe
            and shared_expert_overlap_stream is not None
            and include_shared_expert
            and mode == "decode"
            and tokens == 1
            and (
                moe_variant in vllm_fused_moe_variants()
                or moe_variant in native_moe_consumer_variants()
            )
            and moe_variant != "vllm_fused_inplace"
        )

    def vllm_moe_override_config_for_layer_index(layer_index: int | None) -> dict[str, Any] | None:
        model_layer = None
        if layer_index is not None:
            model_layer = int(layer_weights[layer_index]["layer"])
        return vllm_moe_override_config(
            moe_variant,
            mode,
            moe_override_config,
            moe_override_config_by_layer,
            model_layer,
        )

    def vllm_moe_effective_override_config_by_layer() -> dict[str, Any] | None:
        if moe_override_config_by_layer is None or moe_variant not in vllm_fused_moe_variants():
            return None
        fallback = vllm_moe_override_config(moe_variant, "decode", moe_override_config)
        effective = {
            str(entry["layer"]): vllm_moe_override_config(
                moe_variant,
                "decode",
                moe_override_config,
                moe_override_config_by_layer,
                int(entry["layer"]),
            )
            for layer_index, entry in enumerate(layer_weights)
            if vllm_moe_override_config(
                moe_variant,
                "decode",
                moe_override_config,
                moe_override_config_by_layer,
                int(entry["layer"]),
            )
            != fallback
        }
        return effective or None

    def cache_key_for_tensor(name: str) -> tuple[str, str]:
        if name.startswith("raw:"):
            return "raw_weights", f"{raw_tensor_cache_prefix}:{name}"
        if name.startswith("derived:"):
            return "derived_layouts", f"{derived_tensor_cache_prefix}:{name}"
        return "derived_layouts", f"{runtime_tensor_cache_prefix}:{name}"

    def cached_tensor(name: str, build: Callable[[], Any]) -> Any:
        nonlocal tensor_cache_hits, tensor_cache_misses
        scope, key = cache_key_for_tensor(name)
        if reuse_tensor_cache and key in _ENGINE_TENSOR_CACHE:
            tensor_cache_hits += 1
            tensor_cache_by_scope[scope]["hits"] += 1
            return _ENGINE_TENSOR_CACHE[key]
        tensor_cache_misses += 1
        tensor_cache_by_scope[scope]["misses"] += 1
        tensor = build()
        if reuse_tensor_cache:
            _ENGINE_TENSOR_CACHE[key] = tensor
        return tensor

    def required_cached_tensor(name: str) -> Any:
        nonlocal tensor_cache_hits
        scope, key = cache_key_for_tensor(name)
        if not reuse_tensor_cache or key not in _ENGINE_TENSOR_CACHE:
            raise RuntimeError(f"required resident tensor cache entry is absent: {name}")
        tensor_cache_hits += 1
        tensor_cache_by_scope[scope]["hits"] += 1
        return _ENGINE_TENSOR_CACHE[key]

    def native_moe_pack_gate_up_layout(gate_up: Any) -> Any:
        expert_count, two_intermediate, hidden_features = gate_up.shape
        if two_intermediate != 2 * intermediate:
            raise ValueError("native MoE gate/up layout requires 2x intermediate")
        block_i = native_moe_config["layout_block_i"]
        padded_intermediate = math.ceil(intermediate / block_i) * block_i
        if padded_intermediate == intermediate:
            split_gate_up = gate_up.reshape(expert_count, 2, intermediate, hidden_features)
        else:
            split_gate_up = torch.zeros(
                expert_count, 2, padded_intermediate, hidden_features,
                device=gate_up.device, dtype=gate_up.dtype,
            )
            split_gate_up[:, 0, :intermediate, :] = gate_up[:, :intermediate, :]
            split_gate_up[:, 1, :intermediate, :] = gate_up[:, intermediate:, :]
        return (
            split_gate_up.reshape(
                expert_count, 2, padded_intermediate // block_i, block_i, hidden_features
            )
            .permute(0, 1, 2, 4, 3)
            .contiguous()
        )

    def native_moe_pack_down_layout(down: Any) -> Any:
        expert_count, hidden_features, down_intermediate = down.shape
        if down_intermediate != intermediate:
            raise ValueError("native MoE down layout requires intermediate match")
        block_h = native_moe_config["layout_block_h"]
        padded_hidden = math.ceil(hidden_features / block_h) * block_h
        if padded_hidden == hidden_features:
            down_padded = down
        else:
            down_padded = torch.zeros(
                expert_count, padded_hidden, down_intermediate,
                device=down.device, dtype=down.dtype,
            )
            down_padded[:, :hidden_features, :] = down
        return (
            down_padded.reshape(
                expert_count, padded_hidden // block_h, block_h, down_intermediate
            )
            .permute(0, 1, 3, 2)
            .contiguous()
        )

    def native_moe_unpack_gate_up_layout(gate_up_native: Any) -> Any:
        expert_count, two, block_count, hidden_features, block_i = gate_up_native.shape
        if two != 2 or block_i != native_moe_config["layout_block_i"]:
            raise ValueError("native MoE gate/up layout does not match the active layout contract")
        split = gate_up_native.permute(0, 1, 2, 4, 3).contiguous()
        split = split.reshape(expert_count, 2, block_count * block_i, hidden_features)
        return split[:, :, :intermediate, :].reshape(
            expert_count, 2 * intermediate, hidden_features
        ).contiguous()

    def native_moe_unpack_down_layout(down_native: Any) -> Any:
        expert_count, block_count, down_intermediate, block_h = down_native.shape
        if down_intermediate != intermediate or block_h != native_moe_config["layout_block_h"]:
            raise ValueError("native MoE down layout does not match the active layout contract")
        down = down_native.permute(0, 1, 3, 2).contiguous()
        return down.reshape(expert_count, block_count * block_h, down_intermediate)[
            :, :hidden, :
        ].contiguous()

    def ensure_striped_image_loaded() -> dict[str, Any]:
        nonlocal striped_image_loader_report
        required_env = {
            "manifest": striped_manifest_env,
            "library": striped_library_env,
            "native_report": striped_native_report_env,
            "expected_xor": striped_expected_xor_env,
            "expected_sum": striped_expected_sum_env,
            "lane0_sha256": striped_lane_sha256_env[0],
            "lane1_sha256": striped_lane_sha256_env[1],
        }
        missing_env = [name for name, value in required_env.items() if not value]
        if missing_env:
            raise RuntimeError(f"striped image environment is incomplete: {missing_env}")
        manifest_path = Path(striped_manifest_env).resolve()
        library_path = Path(striped_library_env).resolve()
        native_report_path = Path(striped_native_report_env).resolve()
        striped = load_json(manifest_path)
        lanes = striped.get("lanes", [])
        entries = striped.get("entries", [])
        if striped.get("complete") is not True or len(lanes) != 2 or len(entries) != 693:
            raise RuntimeError("striped image manifest is incomplete")
        if striped.get("inputs", {}).get("checkpoint_index", {}).get("sha256") != index_sha256:
            raise RuntimeError("striped image checkpoint index drift")
        lane_paths = [Path(item["image_path"]).resolve() for item in lanes]
        lane_bytes = [int(item["image_bytes"]) for item in lanes]
        total_bytes = int(striped["layout"]["aligned_bytes"])
        payload_total = int(striped["layout"]["payload_bytes"])
        if total_bytes != sum(lane_bytes) or total_bytes != 69321650176:
            raise RuntimeError("striped image byte geometry drift")
        if payload_total != 69321221376:
            raise RuntimeError("striped image payload geometry drift")
        if any(not path.is_file() for path in lane_paths):
            raise RuntimeError("striped image lane is absent")
        if [path.stat().st_size for path in lane_paths] != lane_bytes:
            raise RuntimeError("striped image lane size drift")
        manifest_sha256 = sha256_file(manifest_path)
        state_key = f"scatter:{manifest_sha256}:{sha256_file(library_path)}:{device}"
        retained = _ENGINE_STRIPED_IMAGE_STATE.get(state_key)
        if retained is not None:
            striped_image_loader_report = dict(retained["report"])
            striped_image_loader_report["reused"] = True
            return striped_image_loader_report
        raw_cache_entries_before = sum(
            1 for key in _ENGINE_TENSOR_CACHE if key.startswith(f"{raw_tensor_cache_prefix}:")
        )
        if raw_cache_entries_before != 0:
            raise RuntimeError("striped scatter loader requires an empty raw cache")

        preload_started = time.perf_counter()
        memory_before = int(torch.cuda.memory_allocated())
        allocation_started = time.perf_counter()
        tensors = []
        cache_keys = []
        aggregate_offsets = []
        payload_sizes = []
        destination_pointers = []
        previous_offset = -1
        all_shapes_exact = True
        all_payload_sizes_exact = True
        for entry in entries:
            if entry["dtype"] != "BF16":
                raise RuntimeError(f"non-BF16 striped entry: {entry['name']}")
            offset = int(entry["aggregate_offset_bytes"])
            payload_bytes = int(entry["payload_bytes"])
            shape = tuple(int(value) for value in entry["shape"])
            if offset <= previous_offset or offset % 4096 != 0:
                raise RuntimeError("striped entry order or alignment drift")
            typed = torch.empty(shape, dtype=torch.bfloat16, device=device)
            actual_payload_bytes = int(typed.numel()) * int(typed.element_size())
            all_shapes_exact = all_shapes_exact and tuple(typed.shape) == shape
            all_payload_sizes_exact = (
                all_payload_sizes_exact and actual_payload_bytes == payload_bytes
            )
            cache_key = cache_key_for_tensor(f"raw:{entry['name']}")[1]
            if cache_key in _ENGINE_TENSOR_CACHE or cache_key in cache_keys:
                raise RuntimeError(f"duplicate striped cache key: {entry['name']}")
            tensors.append(typed)
            cache_keys.append(cache_key)
            aggregate_offsets.append(offset)
            payload_sizes.append(payload_bytes)
            destination_pointers.append(int(typed.data_ptr()))
            previous_offset = offset
        sync()
        allocation_ms = (time.perf_counter() - allocation_started) * 1000.0
        memory_after_tensors = int(torch.cuda.memory_allocated())
        unique_data_pointers = len(set(destination_pointers))
        storage_pointers = [int(tensor.untyped_storage().data_ptr()) for tensor in tensors]
        unique_storage_pointers = len(set(storage_pointers))
        all_data_pointer_matches_storage = all(
            data_pointer == storage_pointer
            for data_pointer, storage_pointer in zip(destination_pointers, storage_pointers)
        )
        all_storage_offsets_zero = all(int(tensor.storage_offset()) == 0 for tensor in tensors)
        allocation_checks = {
            "tensors_exact": len(tensors) == len(entries) == 693,
            "payload_sum_exact": sum(payload_sizes) == payload_total,
            "shapes_exact": all_shapes_exact,
            "payload_sizes_exact": all_payload_sizes_exact,
            "unique_data_pointers": unique_data_pointers == len(entries),
            "unique_storage_pointers": unique_storage_pointers == len(entries),
            "data_pointer_matches_storage": all_data_pointer_matches_storage,
            "storage_offsets_zero": all_storage_offsets_zero,
        }
        if not all(allocation_checks.values()):
            raise RuntimeError(f"independent Torch allocation failed: {allocation_checks}")

        offset_array = (ctypes.c_uint64 * len(entries))(*aggregate_offsets)
        payload_array = (ctypes.c_uint64 * len(entries))(*payload_sizes)
        pointer_array = (ctypes.c_uint64 * len(entries))(*destination_pointers)
        library = ctypes.CDLL(str(library_path))
        function = library.torch_owned_striped_tensor_scatter_ingest
        function.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_char_p,
        ]
        function.restype = ctypes.c_int
        native_started = time.perf_counter()
        native_return_code = int(
            function(
                str(lane_paths[0]).encode(),
                str(lane_paths[1]).encode(),
                offset_array,
                payload_array,
                pointer_array,
                ctypes.c_size_t(len(entries)),
                ctypes.c_size_t(total_bytes),
                ctypes.c_size_t(lane_bytes[0]),
                ctypes.c_size_t(lane_bytes[1]),
                ctypes.c_size_t(striped_chunk_bytes),
                ctypes.c_uint64(int(striped_expected_xor_env, 0)),
                ctypes.c_uint64(int(striped_expected_sum_env, 0)),
                str(native_report_path).encode(),
            )
        )
        sync()
        native_elapsed_ms = (time.perf_counter() - native_started) * 1000.0
        native = load_json(native_report_path)
        native_checks = {
            "return_zero": native_return_code == 0,
            "complete": native.get("complete") is True,
            "image_bytes_exact": native.get("image_bytes") == total_bytes,
            "payload_bytes_exact": native.get("payload_bytes") == payload_total,
            "lane_bytes_exact": native.get("lane_bytes") == lane_bytes,
            "tensor_count_exact": native.get("tensor_count") == len(entries),
            "unique_destinations_exact": native.get("unique_destination_pointers") == len(entries),
            "pointer_types_device": native.get("all_pointer_types_device") is True,
            "pointers_match": native.get("all_device_pointers_match") is True,
            "gpu_checksum_exact": native.get("gpu_payload_checksum_equal") is True,
            "destination_not_freed": native.get("destination_freed_by_native") is False,
            "cleanup_complete": native.get("cleanup_complete") is True,
        }
        if not all(native_checks.values()):
            raise RuntimeError(f"striped scatter native loader failed: {native_checks}")

        bind_started = time.perf_counter()
        for cache_key, typed in zip(cache_keys, tensors):
            _ENGINE_TENSOR_CACHE[cache_key] = typed
        bind_ms = (time.perf_counter() - bind_started) * 1000.0
        raw_cache_entries_after = sum(
            1 for key in _ENGINE_TENSOR_CACHE if key.startswith(f"{raw_tensor_cache_prefix}:")
        )
        report = {
            "active": True,
            "complete": True,
            "reused": False,
            "provider": "dual_physical_nvme_odirect_pinned_h2d_into_independent_torch_storages",
            "manifest": str(manifest_path),
            "manifest_sha256": manifest_sha256,
            "library": str(library_path),
            "library_sha256": sha256_file(library_path),
            "lane_paths": [str(path) for path in lane_paths],
            "lane_sha256": striped_lane_sha256_env,
            "lane_bytes": lane_bytes,
            "aligned_image_bytes": total_bytes,
            "payload_bytes": payload_total,
            "planned_tensors": len(entries),
            "raw_cache_entries_before": raw_cache_entries_before,
            "raw_cache_entries_after": raw_cache_entries_after,
            "fallback_loads": 0,
            "allocation_ms": allocation_ms,
            "native_elapsed_ms": native_elapsed_ms,
            "bind_ms": bind_ms,
            "preload_wall_time_ms": (time.perf_counter() - preload_started) * 1000.0,
            "unique_data_pointers": unique_data_pointers,
            "unique_storage_pointers": unique_storage_pointers,
            "memory_before": memory_before,
            "memory_after_tensors": memory_after_tensors,
            "native_return_code": native_return_code,
            "allocation_checks": allocation_checks,
            "native_checks": native_checks,
            "native": native,
        }
        report_checks = {
            "raw_cache_exact": raw_cache_entries_after == len(entries) == 693,
            "independent_storages_exact": unique_storage_pointers == len(entries),
            "allocation_checks": all(allocation_checks.values()),
            "native_checks": all(native_checks.values()),
            "no_fallback": report["fallback_loads"] == 0,
        }
        if not all(report_checks.values()):
            raise RuntimeError(f"striped scatter binding failed: {report_checks}")
        report["checks"] = report_checks
        _ENGINE_STRIPED_IMAGE_STATE[state_key] = {
            "tensors": tensors,
            "report": report,
        }
        striped_image_loader_report = report
        return report

    def load_tensor(name: str) -> Any:
        return cached_tensor(
            f"raw:{name}",
            lambda: _load_tensor_uncached(name),
        )

    def _load_tensor_uncached(name: str) -> Any:
        if striped_manifest_env:
            raise RuntimeError(f"striped image cache missed required tensor: {name}")
        shard = model_dir / weight_map[name]
        with safe_open(shard, framework="pt", device="cpu") as file:
            return file.get_tensor(name).to(device=device)

    def quantize_certified_lm_head(weight: Any) -> tuple[Any, Any, Any]:
        rows, _ = weight.shape
        q_weight = torch.empty_like(weight, dtype=torch.int8)
        scales = torch.empty(rows, device=weight.device, dtype=torch.float32)
        residual_l2 = torch.empty(rows, device=weight.device, dtype=torch.float32)
        chunk_rows = 8192
        for start in range(0, rows, chunk_rows):
            end = min(rows, start + chunk_rows)
            source = weight[start:end].float()
            chunk_scales = source.abs().amax(dim=1).clamp_min(1e-12) / 127.0
            quantized = torch.round(source / chunk_scales[:, None]).clamp(-127, 127).to(torch.int8)
            residual = source - quantized.float() * chunk_scales[:, None]
            q_weight[start:end].copy_(quantized)
            scales[start:end].copy_(chunk_scales)
            residual_l2[start:end].copy_(torch.linalg.vector_norm(residual, dim=1))
        return q_weight, scales, residual_l2

    def launch_certified_lm_head_int8(
        q_weight: Any,
        hidden_state: Any,
        scales: Any,
        output: Any,
    ) -> None:
        if triton is None or triton_lm_head_rowwise_int8_gemv_kernel is None:
            raise RuntimeError("int8_certified_global_tie requires Triton")
        rows, cols = q_weight.shape
        grid = (triton.cdiv(rows, 8),)
        triton_lm_head_rowwise_int8_gemv_kernel[grid](
            q_weight,
            hidden_state,
            scales,
            output,
            rows=rows,
            cols=cols,
            BLOCK_M=8,
            BLOCK_K=128,
            num_warps=8,
        )

    native_layout_cache_names = [
        f"derived:layer{layer}:native_moe_{kind}_{block_name}{block_value}"
        for layer in resident_native_decode_hotset_layer_indices
        for kind, block_name, block_value in (
            ("gate_up", "i", native_moe_config["layout_block_i"]),
            ("down", "h", native_moe_config["layout_block_h"]),
        )
    ]
    native_layout_cache_keys = [cache_key_for_tensor(name)[1] for name in native_layout_cache_names]
    resident_native_layout_cache_entries_at_load = sum(
        key in _ENGINE_TENSOR_CACHE for key in native_layout_cache_keys
    )
    resident_native_layout_partial_cache_evicted = 0
    if (
        reuse_tensor_cache
        and 0 < resident_native_layout_cache_entries_at_load < len(native_layout_cache_keys)
    ):
        for key in native_layout_cache_keys:
            if _ENGINE_TENSOR_CACHE.pop(key, None) is not None:
                resident_native_layout_partial_cache_evicted += 1
        resident_native_layout_cache_entries_at_load = 0
    resident_native_layout_cache_complete_at_load = (
        bool(native_layout_cache_keys)
        and reuse_tensor_cache
        and resident_native_layout_cache_entries_at_load == len(native_layout_cache_keys)
    )
    resident_native_prefill_reconstruction_calls = 0
    resident_native_prefill_reconstructed_layers: set[int] = set()

    if input_token_ids is not None:
        if len(input_token_ids) != tokens:
            raise ValueError(f"input token count {len(input_token_ids)} must match --tokens {tokens}")
        out_of_range = [item for item in input_token_ids if item < 0 or item >= vocab]
        if out_of_range:
            raise ValueError(f"input token ids outside vocab range 0..{vocab - 1}: {out_of_range[:8]}")

    torch.manual_seed(seed)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()

    layer_weights: list[dict[str, Any]] = []
    tensor_metadata: dict[str, Any] = {}
    layer_tensor_wall_start = time.perf_counter()
    if striped_manifest_env:
        striped_image_loader_report = ensure_striped_image_loaded()
    for layer in layers:
        names = layer_tensor_names(layer)
        required = [
            "input_layernorm",
            "post_attention_layernorm",
            "router",
            "expert_gate_up",
            "expert_down",
        ]
        if full_attention_enabled(attention_mode) and layer_types[layer] == "full_attention":
            required.extend(["q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"])
        if linear_attention_enabled(attention_mode) and layer_types[layer] == "linear_attention":
            required.extend(
                [
                    "linear_A_log",
                    "linear_conv1d",
                    "linear_dt_bias",
                    "linear_in_proj_a",
                    "linear_in_proj_b",
                    "linear_in_proj_qkv",
                    "linear_in_proj_z",
                    "linear_norm",
                    "linear_out_proj",
                ]
            )
        if include_shared_expert:
            required.extend(["shared_gate", "shared_gate_proj", "shared_up_proj", "shared_down_proj"])
        missing = [names[key] for key in required if names[key] not in weight_map]
        if missing:
            raise KeyError(f"missing tensors in safetensors index: {missing}")

        load_keys = list(required)
        tensors = {key: load_tensor(names[key]) for key in load_keys}
        if (
            resident_native_layout_cache_complete_at_load
            and layer in resident_native_decode_hotset_layer_indices
        ):
            tensors["native_moe_gate_up"] = required_cached_tensor(
                f"derived:layer{layer}:native_moe_gate_up_i{native_moe_config['layout_block_i']}"
            )
            tensors["native_moe_down"] = required_cached_tensor(
                f"derived:layer{layer}:native_moe_down_h{native_moe_config['layout_block_h']}"
            )
        expected_shapes = {
            "input_layernorm": (hidden,),
            "post_attention_layernorm": (hidden,),
            "router": (experts, hidden),
            "expert_gate_up": (experts, 2 * intermediate, hidden),
            "expert_down": (experts, hidden, intermediate),
        }
        if (
            resident_native_layout_cache_complete_at_load
            and layer in resident_native_decode_hotset_layer_indices
        ):
            expected_shapes.update(
                {
                    "native_moe_gate_up": (
                        experts,
                        2,
                        math.ceil(intermediate / native_moe_config["layout_block_i"]),
                        hidden,
                        native_moe_config["layout_block_i"],
                    ),
                    "native_moe_down": (
                        experts,
                        math.ceil(hidden / native_moe_config["layout_block_h"]),
                        intermediate,
                        native_moe_config["layout_block_h"],
                    ),
                }
            )
        if full_attention_enabled(attention_mode) and layer_types[layer] == "full_attention":
            expected_shapes.update(
                {
                    "q_proj": (2 * q_dim, hidden),
                    "k_proj": (kv_dim, hidden),
                    "v_proj": (kv_dim, hidden),
                    "o_proj": (hidden, q_dim),
                    "q_norm": (head_dim,),
                    "k_norm": (head_dim,),
                }
            )
        if linear_attention_enabled(attention_mode) and layer_types[layer] == "linear_attention":
            expected_shapes.update(
                {
                    "linear_A_log": (linear_value_heads,),
                    "linear_conv1d": (linear_conv_dim, 1, linear_conv_kernel_dim),
                    "linear_dt_bias": (linear_value_heads,),
                    "linear_in_proj_a": (linear_value_heads, hidden),
                    "linear_in_proj_b": (linear_value_heads, hidden),
                    "linear_in_proj_qkv": (linear_conv_dim, hidden),
                    "linear_in_proj_z": (linear_value_dim, hidden),
                    "linear_norm": (linear_value_head_dim,),
                    "linear_out_proj": (hidden, linear_value_dim),
                }
            )
        if include_shared_expert:
            expected_shapes.update(
                {
                    "shared_gate": (1, hidden),
                    "shared_gate_proj": (shared_intermediate, hidden),
                    "shared_up_proj": (shared_intermediate, hidden),
                    "shared_down_proj": (hidden, shared_intermediate),
                }
            )
        shape_errors = {
            key: {"expected": list(expected), "actual": list(tensors[key].shape)}
            for key, expected in expected_shapes.items()
            if tuple(tensors[key].shape) != expected
        }
        if shape_errors:
            raise ValueError(f"unexpected tensor shapes for layer {layer}: {shape_errors}")

        tensor_metadata[str(layer)] = {}
        for key, tensor in tensors.items():
            source_name = names.get(key)
            tensor_metadata[str(layer)][key] = {
                "name": source_name if source_name is not None else f"derived:{key}",
                "shard": weight_map[source_name] if source_name is not None else None,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype).replace("torch.", ""),
            }
        if torch_out_router_decode or triton_router_topk_decode:
            router_t = cached_tensor(
                f"derived:layer{layer}:router_t",
                lambda tensors=tensors: tensors["router"].t().contiguous(),
            )
            tensors["router_t"] = router_t
            tensor_metadata[str(layer)]["router_t"] = {
                "name": "derived:transpose(router)",
                "shard": None,
                "shape": list(router_t.shape),
                "dtype": str(router_t.dtype).replace("torch.", ""),
            }
        if (
            linear_attention_conv_variant in {"decode_direct", "decode_direct_triton"}
            and linear_attention_enabled(attention_mode)
            and layer_types[layer] == "linear_attention"
        ):
            tensors["linear_conv1d_direct_weight"] = cached_tensor(
                f"derived:layer{layer}:linear_conv1d_direct_weight",
                lambda tensors=tensors: tensors["linear_conv1d"].squeeze(1).contiguous(),
            )
            tensor_metadata[str(layer)]["linear_conv1d_direct_weight"] = {
                "name": "derived:squeeze(linear_conv1d, dim=1)",
                "shard": None,
                "shape": list(tensors["linear_conv1d_direct_weight"].shape),
                "dtype": str(tensors["linear_conv1d_direct_weight"].dtype).replace("torch.", ""),
            }
        if (
            linear_attention_input_proj_variant
            in {
                "decode_fused",
                "decode_fused_t",
                "decode_fused_t_triton",
                "decode_fused_t_conv_triton",
                "decode_fused_t_conv_qkv_triton",
                "prefill_fused_t_decode_fused_t_conv_triton",
                "prefill_fused_t_decode_fused_t_conv_qkv_triton",
            }
            and linear_attention_enabled(attention_mode)
            and layer_types[layer] == "linear_attention"
        ):
            fused_input_proj = cached_tensor(
                f"derived:layer{layer}:linear_input_proj_fused",
                lambda tensors=tensors: torch.cat(
                    [
                        tensors["linear_in_proj_qkv"],
                        tensors["linear_in_proj_z"],
                        tensors["linear_in_proj_a"],
                        tensors["linear_in_proj_b"],
                    ],
                    dim=0,
                ).contiguous(),
            )
            tensors["linear_input_proj_fused"] = fused_input_proj
            tensor_metadata[str(layer)]["linear_input_proj_fused"] = {
                "name": "derived:cat(linear_in_proj_qkv,linear_in_proj_z,linear_in_proj_a,linear_in_proj_b)",
                "shard": None,
                "shape": list(fused_input_proj.shape),
                "dtype": str(fused_input_proj.dtype).replace("torch.", ""),
            }
            if linear_attention_input_proj_variant in {
                "decode_fused_t",
                "decode_fused_t_triton",
                "decode_fused_t_conv_triton",
                "decode_fused_t_conv_qkv_triton",
                "prefill_fused_t_decode_fused_t_conv_triton",
                "prefill_fused_t_decode_fused_t_conv_qkv_triton",
            }:
                fused_input_proj_t = cached_tensor(
                    f"derived:layer{layer}:linear_input_proj_fused_t",
                    lambda fused_input_proj=fused_input_proj: fused_input_proj.t().contiguous(),
                )
                tensors["linear_input_proj_fused_t"] = fused_input_proj_t
                tensor_metadata[str(layer)]["linear_input_proj_fused_t"] = {
                    "name": "derived:transpose(linear_input_proj_fused)",
                    "shard": None,
                    "shape": list(fused_input_proj_t.shape),
                    "dtype": str(fused_input_proj_t.dtype).replace("torch.", ""),
                }
        if (
            triton_linear_output_proj_decode
            and linear_attention_enabled(attention_mode)
            and layer_types[layer] == "linear_attention"
        ):
            linear_out_proj_t = cached_tensor(
                f"derived:layer{layer}:linear_out_proj_t",
                lambda tensors=tensors: tensors["linear_out_proj"].t().contiguous(),
            )
            tensors["linear_out_proj_t"] = linear_out_proj_t
            tensor_metadata[str(layer)]["linear_out_proj_t"] = {
                "name": "derived:transpose(linear_out_proj)",
                "shard": None,
                "shape": list(linear_out_proj_t.shape),
                "dtype": str(linear_out_proj_t.dtype).replace("torch.", ""),
            }
        if (
            triton_full_attention_proj_decode
            and not triton_full_attention_fused_qkv_decode
            and layer_types[layer] == "full_attention"
        ):
            for source_key, derived_key in (
                ("q_proj", "q_proj_t"),
                ("k_proj", "k_proj_t"),
                ("v_proj", "v_proj_t"),
                ("o_proj", "o_proj_t"),
            ):
                transposed = cached_tensor(
                    f"derived:layer{layer}:{derived_key}",
                    lambda tensors=tensors, source_key=source_key: tensors[source_key].t().contiguous(),
                )
                tensors[derived_key] = transposed
                tensor_metadata[str(layer)][derived_key] = {
                    "name": f"derived:transpose({source_key})",
                    "shard": None,
                    "shape": list(transposed.shape),
                    "dtype": str(transposed.dtype).replace("torch.", ""),
                }
        if triton_full_attention_fused_qkv_decode and layer_types[layer] == "full_attention":
            full_qkv_proj_fused = cached_tensor(
                f"derived:layer{layer}:full_qkv_proj_fused",
                lambda tensors=tensors: torch.cat(
                    [
                        tensors["q_proj"],
                        tensors["k_proj"],
                        tensors["v_proj"],
                    ],
                    dim=0,
                ).contiguous(),
            )
            tensors["full_qkv_proj_fused"] = full_qkv_proj_fused
            tensor_metadata[str(layer)]["full_qkv_proj_fused"] = {
                "name": "derived:cat(q_proj,k_proj,v_proj)",
                "shard": None,
                "shape": list(full_qkv_proj_fused.shape),
                "dtype": str(full_qkv_proj_fused.dtype).replace("torch.", ""),
            }
            for source_key, derived_key in (
                ("full_qkv_proj_fused", "full_qkv_proj_fused_t"),
                ("o_proj", "o_proj_t"),
            ):
                transposed = cached_tensor(
                    f"derived:layer{layer}:{derived_key}",
                    lambda tensors=tensors, source_key=source_key: tensors[source_key].t().contiguous(),
                )
                tensors[derived_key] = transposed
                tensor_metadata[str(layer)][derived_key] = {
                    "name": f"derived:transpose({source_key})",
                    "shard": None,
                    "shape": list(transposed.shape),
                    "dtype": str(transposed.dtype).replace("torch.", ""),
                }
        if triton_shared_expert_proj_decode and not triton_shared_expert_fused_input_decode:
            for source_key, derived_key in (
                ("shared_gate", "shared_gate_t"),
                ("shared_gate_proj", "shared_gate_proj_t"),
                ("shared_up_proj", "shared_up_proj_t"),
                ("shared_down_proj", "shared_down_proj_t"),
            ):
                transposed = cached_tensor(
                    f"derived:layer{layer}:{derived_key}",
                    lambda tensors=tensors, source_key=source_key: tensors[source_key].t().contiguous(),
                )
                tensors[derived_key] = transposed
                tensor_metadata[str(layer)][derived_key] = {
                    "name": f"derived:transpose({source_key})",
                    "shard": None,
                    "shape": list(transposed.shape),
                    "dtype": str(transposed.dtype).replace("torch.", ""),
                }
        if triton_shared_expert_fused_input_decode:
            shared_input_proj_fused = cached_tensor(
                f"derived:layer{layer}:shared_input_proj_fused",
                lambda tensors=tensors: torch.cat(
                    [
                        tensors["shared_gate"],
                        tensors["shared_gate_proj"],
                        tensors["shared_up_proj"],
                    ],
                    dim=0,
                ).contiguous(),
            )
            tensors["shared_input_proj_fused"] = shared_input_proj_fused
            tensor_metadata[str(layer)]["shared_input_proj_fused"] = {
                "name": "derived:cat(shared_gate,shared_gate_proj,shared_up_proj)",
                "shard": None,
                "shape": list(shared_input_proj_fused.shape),
                "dtype": str(shared_input_proj_fused.dtype).replace("torch.", ""),
            }
            for source_key, derived_key in (
                ("shared_input_proj_fused", "shared_input_proj_fused_t"),
                ("shared_down_proj", "shared_down_proj_t"),
            ):
                transposed = cached_tensor(
                    f"derived:layer{layer}:{derived_key}",
                    lambda tensors=tensors, source_key=source_key: tensors[source_key].t().contiguous(),
                )
                tensors[derived_key] = transposed
                tensor_metadata[str(layer)][derived_key] = {
                    "name": f"derived:transpose({source_key})",
                    "shard": None,
                    "shape": list(transposed.shape),
                    "dtype": str(transposed.dtype).replace("torch.", ""),
                }
        layer_weights.append({"layer": layer, "layer_type": layer_types[layer], "tensors": tensors})

    def ensure_native_moe_decode_layouts() -> None:
        if not native_moe_consumer_decode_layout_required:
            return
        for entry in layer_weights:
            layer = int(entry["layer"])
            if layer not in resident_native_decode_hotset_layer_indices:
                continue
            names = layer_tensor_names(layer)
            tensors = entry["tensors"]
            if "native_moe_gate_up" not in tensors:
                gate_up_native = cached_tensor(
                    (
                        f"derived:layer{layer}:native_moe_gate_up"
                        f"_i{native_moe_config['layout_block_i']}"
                    ),
                    lambda tensors=tensors: native_moe_pack_gate_up_layout(tensors["expert_gate_up"]),
                )
                tensors["native_moe_gate_up"] = gate_up_native
                tensor_metadata[str(layer)]["native_moe_gate_up"] = {
                    "name": (
                        "derived:native_moe_gate_up"
                        f"(block_i={native_moe_config['layout_block_i']})"
                    ),
                    "shard": None,
                    "shape": list(gate_up_native.shape),
                    "dtype": str(gate_up_native.dtype).replace("torch.", ""),
                }
            if "native_moe_down" not in tensors:
                down_native = cached_tensor(
                    (
                        f"derived:layer{layer}:native_moe_down"
                        f"_h{native_moe_config['layout_block_h']}"
                    ),
                    lambda tensors=tensors: native_moe_pack_down_layout(tensors["expert_down"]),
                )
                tensors["native_moe_down"] = down_native
                tensor_metadata[str(layer)]["native_moe_down"] = {
                    "name": (
                        "derived:native_moe_down"
                        f"(block_h={native_moe_config['layout_block_h']})"
                    ),
                    "shard": None,
                    "shape": list(down_native.shape),
                    "dtype": str(down_native.dtype).replace("torch.", ""),
                }
            if native_moe_consumer_memory_safe:
                tensors.pop("expert_gate_up", None)
                tensors.pop("expert_down", None)
                if reuse_tensor_cache:
                    for key in ("expert_gate_up", "expert_down"):
                        _ENGINE_TENSOR_CACHE.pop(
                            cache_key_for_tensor(f"raw:{names[key]}")[1],
                            None,
                        )
        sync()

    if mode == "decode" and tokens == 1:
        ensure_native_moe_decode_layouts()
    sync()
    layer_tensor_load_derive_wall_time_ms = (time.perf_counter() - layer_tensor_wall_start) * 1000.0

    global_tensors: dict[str, Any] = {}
    global_tensor_metadata: dict[str, Any] = {}
    global_required: list[str] = []
    if input_token_ids is not None:
        global_required.append("embed_tokens")
    if include_lm_head:
        global_required.extend(["final_norm", "lm_head"])
    global_tensor_wall_start = time.perf_counter()
    if global_required:
        names = global_tensor_names()
        missing = [names[key] for key in global_required if names[key] not in weight_map]
        if missing:
            raise KeyError(f"missing global tensors in safetensors index: {missing}")
        global_tensors = {key: load_tensor(names[key]) for key in global_required}
        expected_global_shapes = {
            "embed_tokens": (vocab, hidden),
            "final_norm": (hidden,),
            "lm_head": (vocab, hidden),
        }
        shape_errors = {
            key: {
                "expected": list(expected_global_shapes[key]),
                "actual": list(global_tensors[key].shape),
            }
            for key in global_required
            if tuple(global_tensors[key].shape) != expected_global_shapes[key]
        }
        if shape_errors:
            raise ValueError(f"unexpected global tensor shapes: {shape_errors}")
        for key, tensor in global_tensors.items():
            global_tensor_metadata[key] = {
                "name": names[key],
                "shard": weight_map[names[key]],
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype).replace("torch.", ""),
            }
        if include_lm_head and lm_head_variant in {"pretransposed", "pretransposed_out"}:
            global_tensors["lm_head_t"] = cached_tensor(
                "derived:global:lm_head_t",
                lambda: global_tensors["lm_head"].t().contiguous(),
            )
            global_tensor_metadata["lm_head_t"] = {
                "name": "derived:transpose(lm_head.weight)",
                "shard": None,
                "shape": list(global_tensors["lm_head_t"].shape),
                "dtype": str(global_tensors["lm_head_t"].dtype).replace("torch.", ""),
            }
        if include_lm_head and lm_head_variant == "int8_certified_global_tie":
            lm_head_bundle = cached_tensor(
                "derived:global:lm_head_int8_certified_global_tie",
                lambda: quantize_certified_lm_head(global_tensors["lm_head"]),
            )
            (
                global_tensors["lm_head_int8"],
                global_tensors["lm_head_int8_scales"],
                global_tensors["lm_head_int8_residual_l2"],
            ) = lm_head_bundle
            for key, name in (
                ("lm_head_int8", "derived:rowwise_int8(lm_head.weight)"),
                ("lm_head_int8_scales", "derived:rowwise_int8_scales(lm_head.weight)"),
                ("lm_head_int8_residual_l2", "derived:rowwise_int8_residual_l2(lm_head.weight)"),
            ):
                global_tensor_metadata[key] = {
                    "name": name,
                    "shard": None,
                    "shape": list(global_tensors[key].shape),
                    "dtype": str(global_tensors[key].dtype).replace("torch.", ""),
                }
    sync()
    global_tensor_load_derive_wall_time_ms = (time.perf_counter() - global_tensor_wall_start) * 1000.0

    workspace_wall_start = time.perf_counter()
    if input_token_ids is None:
        x = torch.randn(tokens, hidden, device=device, dtype=dtype)
    else:
        token_tensor = torch.tensor(input_token_ids, device=device, dtype=torch.long)
        x = global_tensors["embed_tokens"].index_select(0, token_tensor)
    resident_workspace = torch.empty(tokens, hidden, device=device, dtype=dtype)
    resident_workspaces = [resident_workspace for _ in layers]
    native_moe_activation_outputs = {
        layer_index: torch.empty(top_k, intermediate, device=device, dtype=dtype)
        for layer_index, _ in enumerate(layer_weights)
    } if moe_variant in native_moe_consumer_variants() else {}
    native_moe_outputs = {
        layer_index: torch.empty(1, hidden, device=device, dtype=dtype)
        for layer_index, _ in enumerate(layer_weights)
    } if moe_variant in native_moe_consumer_variants() else {}
    attention_stub_buffer = torch.zeros(tokens, hidden, device=device, dtype=dtype)
    attention_stub_buffers = [attention_stub_buffer for _ in layers]
    if mode == "decode":
        full_attention_cache_len = seq_len + max(tokens, decode_loop_steps)
    elif decode_loop_steps:
        full_attention_cache_len = tokens + decode_loop_steps
    else:
        full_attention_cache_len = tokens
    full_attention_kv_caches: dict[int, tuple[Any, Any]] = {}
    full_attention_kv_cache_shape = (
        [kv_heads, full_attention_cache_len, head_dim]
        if full_attention_kv_cache_layout == "grouped"
        else [full_attention_cache_len, kv_heads, head_dim]
    )
    full_attention_proj_outputs: dict[int, dict[str, Any]] = {}
    full_attention_norm_rope_outputs: dict[int, dict[str, Any]] = {}
    full_attention_norm_rope_shared_outputs = (
        {
            "q": torch.empty(tokens, heads, head_dim, device=device, dtype=dtype),
            "k": torch.empty(tokens, kv_heads, head_dim, device=device, dtype=dtype),
        }
        if triton_full_attention_norm_rope_decode
        and full_attention_enabled(attention_mode)
        and any(entry["layer_type"] == "full_attention" for entry in layer_weights)
        else None
    )
    shared_expert_proj_outputs: dict[int, dict[str, Any]] = {}
    rmsnorm_input_outputs: dict[int, Any] = {}
    rmsnorm_post_outputs: dict[int, Any] = {}
    rmsnorm_final_output = torch.empty(1, hidden, device=device, dtype=dtype) if triton_rmsnorm_decode else None
    rmsnorm_prefill_output = (
        torch.empty(tokens, hidden, device=device, dtype=dtype)
        if triton_rmsnorm_prefill
        else None
    )
    residual_prefill_output = (
        torch.empty(tokens, hidden, device=device, dtype=dtype)
        if triton_rmsnorm_prefill
        else None
    )
    lm_head_logits_output = (
        torch.empty(1, vocab, device=device, dtype=dtype)
        if include_lm_head and lm_head_variant == "pretransposed_out" and mode == "decode" and tokens == 1
        else None
    )
    certified_lm_head_logits_output = (
        torch.empty(vocab, device=device, dtype=torch.float32)
        if include_lm_head and lm_head_variant == "int8_certified_global_tie"
        else None
    )
    for layer_index, entry in enumerate(layer_weights):
        if triton_rmsnorm_decode and mode == "decode" and tokens == 1:
            rmsnorm_input_outputs[layer_index] = torch.empty(tokens, hidden, device=device, dtype=dtype)
            rmsnorm_post_outputs[layer_index] = torch.empty(tokens, hidden, device=device, dtype=dtype)
        if full_attention_enabled(attention_mode) and entry["layer_type"] == "full_attention":
            if mode == "decode":
                if full_attention_kv_cache_layout == "grouped":
                    k_cache = torch.randn(
                        full_attention_cache_len,
                        kv_heads,
                        head_dim,
                        device=device,
                        dtype=dtype,
                    ).transpose(0, 1).contiguous()
                    v_cache = torch.randn(
                        full_attention_cache_len,
                        kv_heads,
                        head_dim,
                        device=device,
                        dtype=dtype,
                    ).transpose(0, 1).contiguous()
                else:
                    k_cache = torch.randn(*full_attention_kv_cache_shape, device=device, dtype=dtype)
                    v_cache = torch.randn(*full_attention_kv_cache_shape, device=device, dtype=dtype)
            else:
                k_cache = torch.empty(*full_attention_kv_cache_shape, device=device, dtype=dtype)
                v_cache = torch.empty(*full_attention_kv_cache_shape, device=device, dtype=dtype)
            full_attention_kv_caches[layer_index] = (k_cache, v_cache)
            if triton_full_attention_proj_decode:
                if triton_full_attention_fused_qkv_decode:
                    full_attention_proj_outputs[layer_index] = {
                        "qkv": torch.empty(2 * q_dim + 2 * kv_dim, device=device, dtype=dtype),
                        "o": torch.empty(hidden, device=device, dtype=dtype),
                    }
                else:
                    full_attention_proj_outputs[layer_index] = {
                        "q_gate": torch.empty(2 * q_dim, device=device, dtype=dtype),
                        "k": torch.empty(kv_dim, device=device, dtype=dtype),
                        "v": torch.empty(kv_dim, device=device, dtype=dtype),
                        "o": torch.empty(hidden, device=device, dtype=dtype),
                    }
            if triton_full_attention_norm_rope_decode:
                if full_attention_norm_rope_shared_outputs is None:
                    raise RuntimeError("shared full-attention norm/RoPE workspace is not allocated")
                full_attention_norm_rope_outputs[layer_index] = (
                    full_attention_norm_rope_shared_outputs
                )
        if triton_shared_expert_proj_decode:
            if triton_shared_expert_fused_input_decode:
                shared_expert_proj_outputs[layer_index] = {
                    "input": torch.empty(1 + 2 * shared_intermediate, device=device, dtype=dtype),
                    "down_proj": torch.empty(hidden, device=device, dtype=dtype),
                }
            else:
                shared_expert_proj_outputs[layer_index] = {
                    "gate": torch.empty(1, device=device, dtype=dtype),
                    "gate_proj": torch.empty(shared_intermediate, device=device, dtype=dtype),
                    "up_proj": torch.empty(shared_intermediate, device=device, dtype=dtype),
                    "down_proj": torch.empty(hidden, device=device, dtype=dtype),
                }

    linear_attention_initial_conv_states: dict[int, Any] = {}
    linear_attention_initial_ssm_states: dict[int, Any] = {}
    linear_attention_initial_ssm_states_vllm: dict[int, Any] = {}
    linear_attention_conv_states: dict[int, Any] = {}
    linear_attention_ssm_states: dict[int, Any] = {}
    linear_attention_ssm_states_vllm: dict[int, Any] = {}
    linear_attention_conv_windows: dict[int, Any] = {}
    linear_attention_packed_initial_ssm_states: dict[int, Any] = {}
    linear_attention_packed_ssm_states: dict[int, Any] = {}
    linear_attention_packed_outputs: dict[int, Any] = {}
    linear_attention_packed_state_indices: dict[int, Any] = {}
    router_logits_outputs: dict[int, Any] = {}
    router_topk_partial_values: dict[int, Any] = {}
    router_topk_partial_ids: dict[int, Any] = {}
    router_topk_score_outputs: dict[int, Any] = {}
    router_topk_index_outputs: dict[int, Any] = {}
    linear_attention_input_proj_outputs: dict[int, Any] = {}
    linear_attention_qkv_layout_outputs: dict[int, dict[str, Any]] = {}
    linear_attention_conv_outputs: dict[int, Any] = {}
    linear_attention_gated_norm_outputs: dict[int, Any] = {}
    linear_attention_output_proj_outputs: dict[int, Any] = {}
    vllm_moe_cache1_outputs: dict[int, Any] = {}
    vllm_moe_cache2_outputs: dict[int, Any] = {}
    vllm_moe_cache3_outputs: dict[int, Any] = {}
    vllm_moe_prealloc_outputs: dict[int, Any] = {}
    vllm_moe_num_tokens_post_padded: dict[int, Any] = {}

    def should_use_packed_linear_gdn() -> bool:
        return (
            linear_attention_base_variant(linear_attention_variant)
            in {"vllm_fla_packed_decode", "vllm_fla_packed_refswap_decode"}
            and mode == "decode"
            and tokens == 1
        )

    linear_attention_prefill_conv_output = (
        torch.empty(tokens, linear_conv_dim, device=device, dtype=dtype)
        if triton_linear_conv_prefill and triton_prefill_direct_conv_kernel is not None
        else None
    )
    linear_attention_gated_norm_prefill_output = (
        torch.empty(tokens, linear_value_dim, device=device, dtype=dtype)
        if triton_linear_gated_norm_prefill
        else None
    )
    for layer_index, entry in enumerate(layer_weights):
        if linear_attention_enabled(attention_mode) and entry["layer_type"] == "linear_attention":
            if mode == "decode":
                initial_conv = torch.randn(
                    linear_conv_dim,
                    linear_conv_state_len,
                    device=device,
                    dtype=dtype,
                )
                initial_ssm = torch.randn(
                    linear_value_heads,
                    linear_key_head_dim,
                    linear_value_head_dim,
                    device=device,
                    dtype=torch.float32,
                )
            else:
                initial_conv = torch.zeros(
                    linear_conv_dim,
                    linear_conv_state_len,
                    device=device,
                    dtype=dtype,
                )
                initial_ssm = torch.zeros(
                    linear_value_heads,
                    linear_key_head_dim,
                    linear_value_head_dim,
                    device=device,
                    dtype=torch.float32,
                )
            linear_attention_initial_conv_states[layer_index] = initial_conv
            linear_attention_initial_ssm_states[layer_index] = initial_ssm
            if linear_attention_uses_vllm_prestates(linear_attention_variant) and mode == "decode":
                linear_attention_initial_ssm_states_vllm[layer_index] = (
                    initial_ssm.unsqueeze(0).transpose(-1, -2).contiguous()
                )
                if linear_attention_uses_native_vllm_decode_state(linear_attention_variant):
                    linear_attention_ssm_states_vllm[layer_index] = torch.empty_like(
                        linear_attention_initial_ssm_states_vllm[layer_index]
                    )
            linear_attention_conv_states[layer_index] = torch.empty_like(initial_conv)
            linear_attention_ssm_states[layer_index] = torch.empty_like(initial_ssm)
            if should_use_packed_linear_gdn():
                packed_initial = torch.zeros(
                    2,
                    linear_value_heads,
                    linear_value_head_dim,
                    linear_key_head_dim,
                    device=device,
                    dtype=torch.float32,
                )
                packed_initial[1].copy_(initial_ssm.transpose(-1, -2).contiguous())
                linear_attention_packed_initial_ssm_states[layer_index] = packed_initial
                if not linear_attention_uses_packed_state_refswap(linear_attention_variant):
                    linear_attention_packed_ssm_states[layer_index] = torch.empty_like(packed_initial)
                linear_attention_packed_outputs[layer_index] = torch.empty(
                    tokens,
                    1,
                    linear_value_heads,
                    linear_value_head_dim,
                    device=device,
                    dtype=dtype,
                )
                linear_attention_packed_state_indices[layer_index] = torch.ones(
                    tokens,
                    device=device,
                    dtype=torch.int32,
                )
            if linear_attention_conv_variant == "decode_direct":
                linear_attention_conv_windows[layer_index] = torch.empty(
                    linear_conv_dim,
                    linear_conv_kernel_dim,
                    device=device,
                    dtype=dtype,
                )
            if triton_linear_conv_decode and not triton_fused_linear_input_proj_conv_decode:
                linear_attention_conv_outputs[layer_index] = torch.empty(
                    linear_conv_dim,
                    device=device,
                    dtype=dtype,
                )
            if triton_linear_gated_norm_decode and mode == "decode" and tokens == 1:
                linear_attention_gated_norm_outputs[layer_index] = torch.empty(
                    tokens,
                    linear_value_dim,
                    device=device,
                    dtype=dtype,
                )
            if triton_fused_linear_input_proj_conv_qkv_decode:
                linear_attention_qkv_layout_outputs[layer_index] = {
                    "q": torch.empty(
                        1,
                        tokens,
                        linear_key_heads,
                        linear_key_head_dim,
                        device=device,
                        dtype=dtype,
                    ),
                    "k": torch.empty(
                        1,
                        tokens,
                        linear_key_heads,
                        linear_key_head_dim,
                        device=device,
                        dtype=dtype,
                    ),
                    "v": torch.empty(
                        1,
                        tokens,
                        linear_value_heads,
                        linear_value_head_dim,
                        device=device,
                        dtype=dtype,
                    ),
                    "z": torch.empty(
                        tokens,
                        linear_value_heads,
                        linear_value_head_dim,
                        device=device,
                        dtype=dtype,
                    ),
                    "a": torch.empty(tokens, linear_value_heads, device=device, dtype=dtype),
                    "b": torch.empty(tokens, linear_value_heads, device=device, dtype=dtype),
                }
            elif triton_linear_input_proj_decode or triton_fused_linear_input_proj_conv_decode:
                linear_attention_input_proj_outputs[layer_index] = torch.empty(
                    int(entry["tensors"]["linear_input_proj_fused"].shape[0]),
                    device=device,
                    dtype=dtype,
                )
            if triton_linear_output_proj_decode:
                linear_attention_output_proj_outputs[layer_index] = torch.empty(
                    hidden,
                    device=device,
                    dtype=dtype,
                )
        if torch_out_router_decode:
            router_logits_outputs[layer_index] = torch.empty(
                tokens,
                experts,
                device=device,
                dtype=dtype,
            )
        if triton_router_topk_decode:
            partial_count = math.ceil(experts / 64) * top_k
            router_topk_partial_values[layer_index] = torch.empty(
                partial_count,
                device=device,
                dtype=torch.float32,
            )
            router_topk_partial_ids[layer_index] = torch.empty(
                partial_count,
                device=device,
                dtype=torch.int32,
            )
            router_topk_score_outputs[layer_index] = torch.empty(
                tokens,
                top_k,
                device=device,
                dtype=dtype if triton_router_topk_softmax_decode else torch.float32,
            )
            router_topk_index_outputs[layer_index] = torch.empty(
                tokens,
                top_k,
                device=device,
                dtype=torch.int32,
            )
        if (
            moe_variant == "vllm_fused_prefill_m32_n32_decode_m32_n16_k512_prealloc"
            and mode == "decode"
            and tokens == 1
        ):
            gate_up = entry["tensors"]["expert_gate_up"]
            down = entry["tensors"]["expert_down"]
            gate_up_features = int(gate_up.shape[1])
            hidden_features = int(down.shape[1])
            intermediate_features = int(down.shape[2])
            cache13 = torch.empty(
                tokens * top_k * max(gate_up_features, hidden_features),
                device=device,
                dtype=dtype,
            )
            vllm_moe_cache1_outputs[layer_index] = cache13[: tokens * top_k * gate_up_features].view(
                tokens,
                top_k,
                gate_up_features,
            )
            vllm_moe_cache3_outputs[layer_index] = cache13[: tokens * top_k * hidden_features].view(
                tokens,
                top_k,
                hidden_features,
            )
            vllm_moe_cache2_outputs[layer_index] = torch.empty(
                tokens * top_k,
                intermediate_features,
                device=device,
                dtype=dtype,
            )
            vllm_moe_prealloc_outputs[layer_index] = torch.empty(tokens, hidden_features, device=device, dtype=dtype)
            vllm_moe_num_tokens_post_padded[layer_index] = torch.empty((1), dtype=torch.int32, device=device)

    freq_seq = torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (rope_theta ** (freq_seq / rotary_dim))
    position_start = seq_len if mode == "decode" else 0
    position_end = position_start + tokens
    positions = torch.empty(0, device=device, dtype=torch.float32)
    freqs = torch.empty(0, device=device, dtype=torch.float32)
    cos = torch.empty(0, device=device, dtype=torch.float32)
    sin = torch.empty(0, device=device, dtype=torch.float32)
    cos_flat = torch.empty(0, device=device, dtype=torch.float32)
    sin_flat = torch.empty(0, device=device, dtype=torch.float32)
    empty_rotary_tensor = torch.empty(0, device=device, dtype=torch.float32)
    cross_owner_prefill_stats: dict[str, Any] = {
        "eligible_request": False,
        "hipblas_projection_calls": 0,
        "beta_decay_bmm_calls": 0,
        "activation_windows": [],
    }
    cross_owner_lower_mask_cache: dict[tuple[int, str], Any] = {}

    def set_position_window(start: int, count: int) -> None:
        nonlocal position_start, position_end, positions, freqs, cos, sin, cos_flat, sin_flat
        position_start = start
        position_end = start + count
        positions = torch.arange(position_start, position_end, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        cos = torch.cos(freqs).unsqueeze(1)
        sin = torch.sin(freqs).unsqueeze(1)
        cos_flat = cos.reshape(-1).contiguous()
        sin_flat = sin.reshape(-1).contiguous()

    def build_decode_rotary_cache(start: int, count: int) -> tuple[Any, Any]:
        cache_positions = torch.arange(start, start + count, device=device, dtype=torch.float32)
        cache_freqs = torch.outer(cache_positions, inv_freq)
        return torch.cos(cache_freqs).contiguous(), torch.sin(cache_freqs).contiguous()

    def set_decode_position_window_from_cache(start: int, offset: int, cos_cache: Any, sin_cache: Any) -> None:
        nonlocal position_start, position_end, positions, freqs, cos, sin, cos_flat, sin_flat
        position_start = start
        position_end = start + 1
        positions = empty_rotary_tensor
        freqs = empty_rotary_tensor
        cos_flat = cos_cache[offset]
        sin_flat = sin_cache[offset]
        cos = cos_flat.view(1, 1, -1)
        sin = sin_flat.view(1, 1, -1)

    set_position_window(position_start, tokens)
    compound_provider_exact_stack = tuple(int(layer) for layer in layers) == tuple(range(40))
    initial_request_tokens = int(tokens)

    def use_q8192_compound_provider() -> bool:
        return compound_provider_exact_stack and q8192_compound_provider().exact_request_active(
            mode=mode,
            tokens=tokens,
            position_start=position_start,
        )

    def use_q16_hybrid_attention(model_layer: int) -> bool:
        return (
            compound_provider_exact_stack
            and mode == "prefill"
            and initial_request_tokens == 16384
            and int(tokens) == 8192
            and int(position_start) == 8192
            and int(position_end) == 16384
            and int(model_layer) < 39
            and full_attention_kv_cache_layout == "seq"
        )


    def use_persistent_tilequeue_attention(model_layer: int) -> bool:
        q32_active = (
            initial_request_tokens == 32768
            and int(position_start) in {16384, 24576}
            and int(position_end) in {24576, 32768}
        )
        q64_active = (
            initial_request_tokens == 65536
            and int(position_start) in {16384, 24576, 32768, 40960, 49152, 57344}
            and int(position_end) in {24576, 32768, 40960, 49152, 57344, 65536}
        )
        q128_active = (
            initial_request_tokens == 131072
            and int(position_start) in {16384, 24576, 32768, 40960, 49152, 57344, 65536, 73728, 81920, 90112, 98304, 106496, 114688, 122880}
            and int(position_end) in {24576, 32768, 40960, 49152, 57344, 65536, 73728, 81920, 90112, 98304, 106496, 114688, 122880, 131072}
        )
        q256_active = (
            initial_request_tokens in {261120, 261632, 262143}
            and 16384 <= int(position_start) < 253952
            and int(position_start) % 8192 == 0
            and int(position_end) == int(position_start) + 8192
        )
        return (
            compound_provider_exact_stack
            and mode == "prefill"
            and int(tokens) == 8192
            and int(model_layer) < 39
            and full_attention_kv_cache_layout == "seq"
            and (q32_active or q64_active or q128_active or q256_active)
        )

    def use_partial_persistent_tilequeue_attention(model_layer: int) -> bool:
        return (
            compound_provider_exact_stack
            and mode == "prefill"
            and int(model_layer) < 39
            and full_attention_kv_cache_layout == "seq"
            and (
                (
                    initial_request_tokens == 261120
                    and int(tokens) == 7168
                    and int(position_start) == 253952
                    and int(position_end) == 261120
                )
                or (
                    initial_request_tokens == 261632
                    and int(tokens) == 7680
                    and int(position_start) == 253952
                    and int(position_end) == 261632
                )
            )
        )

    def use_cross_owner_prefill_composition() -> bool:
        request_window_active = (
            initial_request_tokens == 16384
            and int(position_start) in {0, 8192}
            and int(position_end) in {8192, 16384}
        ) or (
            initial_request_tokens == 15360
            and int(position_start) == 0
            and int(position_end) == 8192
        )
        active = (
            compound_provider_exact_stack
            and mode == "prefill"
            and int(tokens) == 8192
            and request_window_active
        )
        if active:
            cross_owner_prefill_stats["eligible_request"] = True
            window = [int(position_start), int(position_end)]
            if window not in cross_owner_prefill_stats["activation_windows"]:
                cross_owner_prefill_stats["activation_windows"].append(window)
        return active

    sync()
    workspace_alloc_init_wall_time_ms = (time.perf_counter() - workspace_wall_start) * 1000.0

    if load_only:
        native_layout_wall_start = time.perf_counter()
        ensure_native_moe_decode_layouts()
        sync()
        native_layout_load_derive_wall_time_ms = (
            time.perf_counter() - native_layout_wall_start
        ) * 1000.0
        service_kernel_priming_started = time.perf_counter()
        prefix_entries_before_priming = len(_ENGINE_EXACT_PREFIX_CACHE)

        q8192_priming_started = time.perf_counter()
        q8192_warm_q = torch.zeros(
            8192, heads, head_dim, device=device, dtype=dtype
        )
        q8192_warm_kv = torch.zeros(
            8192, kv_heads, head_dim, device=device, dtype=dtype
        )
        q8192_compound_provider().launch_ck_fmha(
            q=q8192_warm_q,
            k=q8192_warm_kv,
            value=q8192_warm_kv,
            model_layer=3,
        )
        sync()
        q8192_priming_wall_time_ms = (
            time.perf_counter() - q8192_priming_started
        ) * 1000.0
        del q8192_warm_q, q8192_warm_kv

        certified_lm_head_priming_started = time.perf_counter()
        if (
            not include_lm_head
            or lm_head_variant != "int8_certified_global_tie"
            or certified_lm_head_logits_output is None
            or triton is None
            or triton_rmsnorm_kernel is None
        ):
            raise RuntimeError(
                "load-only service priming requires the certified LM-head and Triton RMSNorm"
            )
        warm_rmsnorm_output = (
            rmsnorm_final_output
            if rmsnorm_final_output is not None
            else torch.empty(1, hidden, device=device, dtype=dtype)
        )
        triton_rmsnorm_kernel[(1,)](
            x[-1:].reshape(hidden),
            global_tensors["final_norm"],
            warm_rmsnorm_output.reshape(hidden),
            hidden,
            rms_norm_eps,
            block_h=triton.next_power_of_2(hidden),
            num_warps=8,
        )
        launch_certified_lm_head_int8(
            global_tensors["lm_head_int8"],
            warm_rmsnorm_output,
            global_tensors["lm_head_int8_scales"],
            certified_lm_head_logits_output,
        )
        warm_hidden_l2 = torch.linalg.vector_norm(warm_rmsnorm_output.float())
        warm_error_bound = (
            global_tensors["lm_head_int8_residual_l2"] * warm_hidden_l2
        )
        warm_lower_max = (certified_lm_head_logits_output - warm_error_bound).max()
        warm_upper_values, warm_upper_indices = torch.topk(
            certified_lm_head_logits_output + warm_error_bound,
            k=1024,
        )
        warm_shortlist_weight = global_tensors["lm_head"].index_select(
            0, warm_upper_indices
        )
        warm_exact_logits = torch.mm(
            warm_rmsnorm_output, warm_shortlist_weight.t()
        ).view(-1)
        torch._assert_async(
            warm_lower_max > warm_upper_values[-1],
            "load-only LM-head warmup certificate failed",
        )
        certified_lm_head_logits_output.scatter_(
            0,
            warm_upper_indices,
            warm_exact_logits.float(),
        )
        sync()
        certified_lm_head_priming_wall_time_ms = (
            time.perf_counter() - certified_lm_head_priming_started
        ) * 1000.0
        service_kernel_priming_wall_time_ms = (
            time.perf_counter() - service_kernel_priming_started
        ) * 1000.0
        prefix_entries_after_priming = len(_ENGINE_EXACT_PREFIX_CACHE)
        if prefix_entries_after_priming != prefix_entries_before_priming:
            raise RuntimeError("load-only kernel priming must not create prefix state")
        service_kernel_priming = {
            "enabled": True,
            "policy": "q8192 CK-FMHA plus certified final-logit first-launch only",
            "full_model_inference": False,
            "prefix_entries_before": prefix_entries_before_priming,
            "prefix_entries_after": prefix_entries_after_priming,
            "q8192_ck_fmha": {
                "model_layer": 3,
                "q_shape": [8192, heads, head_dim],
                "kv_shape": [8192, kv_heads, head_dim],
                "wall_time_ms": q8192_priming_wall_time_ms,
            },
            "certified_lm_head": {
                "independent_rmsnorm_output": rmsnorm_final_output is None,
                "shortlist_rows": 1024,
                "wall_time_ms": certified_lm_head_priming_wall_time_ms,
            },
            "wall_time_ms": service_kernel_priming_wall_time_ms,
        }
        return {
            "load_only": True,
            "model_loaded": True,
            "striped_image_loader": striped_image_loader_report,
            "engine_wall_time_ms": (time.perf_counter() - engine_wall_start) * 1000.0,
            "engine_stage_wall_time_ms": {
                "python_imports": python_import_wall_time_ms,
                "runtime_setup": runtime_setup_wall_time_ms,
                "layer_tensor_load_derive": layer_tensor_load_derive_wall_time_ms,
                "global_tensor_load_derive": global_tensor_load_derive_wall_time_ms,
                "workspace_alloc_init": workspace_alloc_init_wall_time_ms,
                "native_layout_load_derive": native_layout_load_derive_wall_time_ms,
                "service_kernel_priming": service_kernel_priming_wall_time_ms,
            },
            "service_kernel_priming": service_kernel_priming,
            "engine_tensor_cache": {
                "enabled": reuse_tensor_cache,
                "hits": tensor_cache_hits,
                "misses": tensor_cache_misses,
                "entries_after_load": len(_ENGINE_TENSOR_CACHE),
                "raw_weight_entries_after_load": sum(
                    1
                    for key in _ENGINE_TENSOR_CACHE
                    if key.startswith(f"{raw_tensor_cache_prefix}:")
                ),
                "derived_layout_entries_after_load": sum(
                    1
                    for key in _ENGINE_TENSOR_CACHE
                    if key.startswith(f"{derived_tensor_cache_prefix}:")
                ),
                "native_decode_hotset_layers_requested": resident_native_decode_hotset_layers,
                "native_decode_hotset_entries_expected": len(native_layout_cache_keys),
                "native_decode_hotset_entries_after_load": sum(
                    key in _ENGINE_TENSOR_CACHE for key in native_layout_cache_keys
                ),
            },
            "workspace": {
                "request_tokens": tokens,
                "full_attention_kv_cache_shape": full_attention_kv_cache_shape,
                "linear_attention_state_buffers": (
                    len(linear_attention_conv_states) + len(linear_attention_ssm_states)
                ),
            },
            "peak_memory_bytes": (
                int(torch.cuda.max_memory_allocated())
                if device.startswith("cuda") and torch.cuda.is_available()
                else None
            ),
            "device": device,
            "moe_variant": moe_variant,
        }

    def set_runtime_context(new_mode: str, new_tokens: int, new_x: Any) -> None:
        nonlocal mode, tokens, x
        mode = new_mode
        tokens = new_tokens
        x = new_x
        refresh_runtime_flags()
        check_runtime_kernel_requirements()

    def clone_tensor_dict(source: dict[int, Any]) -> dict[int, Any]:
        return {
            int(key): tensor.detach().clone()
            for key, tensor in source.items()
        }

    def copy_cached_tensor_dict(
        *,
        label: str,
        target: dict[int, Any],
        source: Any,
    ) -> tuple[bool, str | None, int]:
        if not isinstance(source, dict):
            return False, f"missing_{label}", 0
        if set(int(key) for key in source) != set(int(key) for key in target):
            return False, f"key_mismatch_{label}", 0
        copied_bytes = 0
        for raw_key, cached_tensor in source.items():
            key = int(raw_key)
            target_tensor = target[key]
            if (
                list(target_tensor.shape) != list(cached_tensor.shape)
                or target_tensor.dtype != cached_tensor.dtype
                or target_tensor.device != cached_tensor.device
            ):
                return False, f"tensor_contract_mismatch_{label}_{key}", copied_bytes
            target_tensor.copy_(cached_tensor)
            copied_bytes += int(cached_tensor.numel()) * int(cached_tensor.element_size())
        return True, None, copied_bytes

    def restore_exact_prefix_state(
        entry: dict[str, Any],
        *,
        for_suffix_prefill: bool = False,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        record: dict[str, Any] = {
            "restored": False,
            "status": "not_attempted",
            "restored_bytes": 0,
        }
        cached_kv = entry.get("full_attention_kv")
        if not isinstance(cached_kv, dict):
            record["status"] = "missing_full_attention_kv"
            return record
        expected_layers = set(int(key) for key in full_attention_kv_caches)
        if set(int(key) for key in cached_kv) != expected_layers:
            record["status"] = "full_attention_layer_mismatch"
            return record
        restored_bytes = 0
        for raw_layer_index, cached_pair in cached_kv.items():
            layer_index = int(raw_layer_index)
            if not isinstance(cached_pair, tuple) or len(cached_pair) != 2:
                record["status"] = f"invalid_full_attention_pair_{layer_index}"
                return record
            cached_k, cached_v = cached_pair
            target_k, target_v = full_attention_kv_caches[layer_index]
            target_k_prefix = (
                target_k[:, : int(entry["prompt_tokens"])]
                if full_attention_kv_cache_layout == "grouped"
                else target_k[: int(entry["prompt_tokens"])]
            )
            target_v_prefix = (
                target_v[:, : int(entry["prompt_tokens"])]
                if full_attention_kv_cache_layout == "grouped"
                else target_v[: int(entry["prompt_tokens"])]
            )
            if (
                list(target_k_prefix.shape) != list(cached_k.shape)
                or list(target_v_prefix.shape) != list(cached_v.shape)
                or target_k_prefix.dtype != cached_k.dtype
                or target_v_prefix.dtype != cached_v.dtype
                or target_k_prefix.device != cached_k.device
                or target_v_prefix.device != cached_v.device
            ):
                record["status"] = f"full_attention_tensor_contract_mismatch_{layer_index}"
                return record
            target_k_prefix.copy_(cached_k)
            target_v_prefix.copy_(cached_v)
            restored_bytes += int(cached_k.numel()) * int(cached_k.element_size())
            restored_bytes += int(cached_v.numel()) * int(cached_v.element_size())
        for label, target in (
            ("linear_conv", linear_attention_conv_states),
            ("linear_ssm", linear_attention_ssm_states),
        ):
            ok, failure, copied = copy_cached_tensor_dict(
                label=label,
                target=target,
                source=entry.get(label),
            )
            restored_bytes += copied
            if not ok:
                record["status"] = failure
                return record
        if for_suffix_prefill:
            for label, target in (
                ("linear_conv_initial", linear_attention_initial_conv_states),
                ("linear_ssm_initial", linear_attention_initial_ssm_states),
            ):
                source_label = "linear_conv" if label.startswith("linear_conv") else "linear_ssm"
                ok, failure, copied = copy_cached_tensor_dict(
                    label=label,
                    target=target,
                    source=entry.get(source_label),
                )
                restored_bytes += copied
                if not ok:
                    record["status"] = failure
                    return record
        sync()
        record.update(
            {
                "restored": True,
                "status": (
                    "strict_token_prefix_state_restored_for_suffix_prefill"
                    if for_suffix_prefill
                    else "exact_token_prefix_state_restored"
                ),
                "restore_scope": (
                    "prefix_kv_plus_linear_current_and_initial_state"
                    if for_suffix_prefill
                    else "full_prompt_kv_plus_linear_current_state"
                ),
                "restored_bytes": restored_bytes,
                "wall_time_ms": (time.perf_counter() - started) * 1000.0,
                "full_attention_layers": len(cached_kv),
                "linear_attention_layers": len(linear_attention_ssm_states),
            }
        )
        return record

    def store_exact_prefix_state(resident_out: Any, resident_logits: Any) -> dict[str, Any]:
        if not exact_prefix_cache or exact_prefix_cache_key is None or input_token_ids is None:
            return {"stored": False, "status": "disabled_or_unkeyed"}
        if len(input_token_ids) > exact_prefix_cache_max_tokens:
            return {"stored": False, "status": "prompt_too_long"}
        started = time.perf_counter()
        full_attention_kv: dict[int, tuple[Any, Any]] = {}
        for layer_index, (k_cache, v_cache) in full_attention_kv_caches.items():
            k_prefix = (
                k_cache[:, : len(input_token_ids)]
                if full_attention_kv_cache_layout == "grouped"
                else k_cache[: len(input_token_ids)]
            )
            v_prefix = (
                v_cache[:, : len(input_token_ids)]
                if full_attention_kv_cache_layout == "grouped"
                else v_cache[: len(input_token_ids)]
            )
            full_attention_kv[int(layer_index)] = (
                k_prefix.detach().clone(),
                v_prefix.detach().clone(),
            )
        aligned_state_checkpoint: dict[str, Any] | None = None
        if (
            0 < canonical_state_checkpoint_tokens < len(input_token_ids)
            and len(canonical_state_checkpoint_conv_states) == len(linear_attention_conv_states)
            and len(canonical_state_checkpoint_ssm_states) == len(linear_attention_ssm_states)
        ):
            checkpoint_full_attention_kv: dict[int, tuple[Any, Any]] = {}
            for layer_index, (cached_k, cached_v) in full_attention_kv.items():
                checkpoint_k = (
                    cached_k[:, :canonical_state_checkpoint_tokens]
                    if full_attention_kv_cache_layout == "grouped"
                    else cached_k[:canonical_state_checkpoint_tokens]
                )
                checkpoint_v = (
                    cached_v[:, :canonical_state_checkpoint_tokens]
                    if full_attention_kv_cache_layout == "grouped"
                    else cached_v[:canonical_state_checkpoint_tokens]
                )
                checkpoint_full_attention_kv[int(layer_index)] = (
                    checkpoint_k.detach().clone(),
                    checkpoint_v.detach().clone(),
                )
            aligned_state_checkpoint = {
                "schema": "exact-token-prefix-aligned-state-checkpoint/v1",
                "prompt_tokens": canonical_state_checkpoint_tokens,
                "full_attention_kv": checkpoint_full_attention_kv,
                "linear_conv": canonical_state_checkpoint_conv_states,
                "linear_ssm": canonical_state_checkpoint_ssm_states,
            }
        entry: dict[str, Any] = {
            "schema": "exact-token-prefix-state/v2",
            "key": exact_prefix_cache_key,
            "runtime_contract_key": exact_prefix_runtime_contract_key,
            "token_ids": tuple(int(item) for item in input_token_ids),
            "token_ids_sha256": token_ids_digest(input_token_ids),
            "prompt_tokens": len(input_token_ids),
            "full_attention_kv": full_attention_kv,
            "linear_conv": clone_tensor_dict(linear_attention_conv_states),
            "linear_ssm": clone_tensor_dict(linear_attention_ssm_states),
            "aligned_state_checkpoint": aligned_state_checkpoint,
            "resident_out": resident_out.detach().clone(),
            "resident_logits": resident_logits.detach().clone(),
            "created_monotonic_ns": time.monotonic_ns(),
        }
        entry["retained_bytes"] = _tensor_tree_bytes(entry)
        _ENGINE_EXACT_PREFIX_CACHE.pop(exact_prefix_cache_key, None)
        _ENGINE_EXACT_PREFIX_CACHE[exact_prefix_cache_key] = entry
        eviction = _trim_engine_exact_prefix_cache(exact_prefix_cache_max_entries)
        sync()
        return {
            "stored": exact_prefix_cache_key in _ENGINE_EXACT_PREFIX_CACHE,
            "status": "exact_token_prefix_state_stored",
            "key": exact_prefix_cache_key,
            "prompt_tokens": len(input_token_ids),
            "retained_bytes": int(entry["retained_bytes"]),
            "full_attention_layers": len(full_attention_kv),
            "linear_attention_layers": len(linear_attention_ssm_states),
            "stored_state_checkpoint_tokens": (
                canonical_state_checkpoint_tokens
                if aligned_state_checkpoint is not None
                else len(input_token_ids)
            ),
            "stored_cached_tail_tokens": (
                len(input_token_ids) - canonical_state_checkpoint_tokens
                if aligned_state_checkpoint is not None
                else 0
            ),
            "eviction": eviction,
            "wall_time_ms": (time.perf_counter() - started) * 1000.0,
        }

    def timed(name: str, fn: Callable[[], Any]) -> tuple[Any, dict[str, Any]]:
        for _ in range(warmup):
            fn()
        sync()
        start = time.perf_counter()
        out = None
        for _ in range(iters):
            out = fn()
        sync()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return out, {
            "name": name,
            "elapsed_ms": elapsed_ms,
            "iters": iters,
            "ms_per_iter": elapsed_ms / iters,
        }

    def cuda_graph_timed(name: str, fn: Callable[[], Any]) -> tuple[Any | None, dict[str, Any]]:
        if not (device.startswith("cuda") and torch.cuda.is_available() and hasattr(torch.cuda, "CUDAGraph")):
            return None, {
                "name": name,
                "status": "unavailable",
                "reason": "torch.cuda.CUDAGraph is not available for this device",
            }
        try:
            side_stream = torch.cuda.Stream()
            side_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side_stream):
                for _ in range(max(1, warmup)):
                    fn()
            torch.cuda.current_stream().wait_stream(side_stream)
            sync()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = fn()
            sync()
            for _ in range(warmup):
                graph.replay()
            sync()
            start = time.perf_counter()
            for _ in range(iters):
                graph.replay()
            sync()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return out, {
                "name": name,
                "status": "ok",
                "elapsed_ms": elapsed_ms,
                "iters": iters,
                "ms_per_iter": elapsed_ms / iters,
            }
        except Exception as exc:  # pragma: no cover - records remote graph capture failures.
            try:
                sync()
            except Exception:
                pass
            return None, {
                "name": name,
                "status": "error",
                "error": repr(exc),
            }

    def diagnostic_timed(
        *,
        layer: int,
        layer_type: str,
        attention: str,
        stage: str,
        fn: Callable[[], Any],
        records: list[dict[str, Any]],
    ) -> Any:
        for _ in range(warmup):
            fn()
        sync()
        start = time.perf_counter()
        out = None
        for _ in range(iters):
            out = fn()
        sync()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        records.append(
            {
                "layer": layer,
                "layer_type": layer_type,
                "attention": attention,
                "stage": stage,
                "name": f"layer{layer}_{attention}_{stage}",
                "elapsed_ms": elapsed_ms,
                "iters": iters,
                "ms_per_iter": elapsed_ms / iters,
            }
        )
        return out

    def rmsnorm(inp: Any, weight: Any, out: Any | None = None) -> Any:
        rows = int(inp.numel() // hidden)
        use_prefill_kernel = triton_rmsnorm_prefill and rows > 1
        if (triton_rmsnorm_decode and out is not None) or use_prefill_kernel:
            if triton is None or triton_rmsnorm_kernel is None:
                raise RuntimeError("Triton RMSNorm kernel is not available")
            target_out = out if not use_prefill_kernel else rmsnorm_prefill_output[:rows]
            triton_rmsnorm_kernel[(rows,)](
                inp.reshape(rows * hidden),
                weight,
                target_out.reshape(rows * hidden),
                hidden,
                rms_norm_eps,
                block_h=triton.next_power_of_2(hidden),
                num_warps=8,
            )
            return target_out.view(rows, hidden)
        variance = inp.float().pow(2).mean(dim=-1, keepdim=True)
        return (inp.float() * torch.rsqrt(variance + rms_norm_eps) * (weight.float() + 1.0)).to(dtype)

    def prefill_fused_add_rmsnorm(inp: Any, residual: Any, weight: Any) -> tuple[Any, Any]:
        if triton is None or triton_prefill_fused_add_rmsnorm_kernel is None:
            raise RuntimeError("Triton prefill fused residual normalization is unavailable")
        rows = int(inp.numel() // hidden)
        if not triton_rmsnorm_prefill or rows <= 1:
            raise RuntimeError("prefill fused residual normalization requires a multirow prefill")
        residual_out = residual_prefill_output[:rows]
        norm_out = rmsnorm_prefill_output[:rows]
        triton_prefill_fused_add_rmsnorm_kernel[(rows,)](
            inp.reshape(rows * hidden),
            residual.reshape(rows * hidden),
            weight,
            residual_out.reshape(rows * hidden),
            norm_out.reshape(rows * hidden),
            hidden,
            rms_norm_eps,
            block_h=triton.next_power_of_2(hidden),
            num_warps=8,
        )
        return residual_out.view(rows, hidden), norm_out.view(rows, hidden)

    def head_rmsnorm(inp: Any, weight: Any) -> Any:
        variance = inp.float().pow(2).mean(dim=-1, keepdim=True)
        return (inp.float() * torch.rsqrt(variance + 1e-6) * (weight.float() + 1.0)).to(dtype)

    def apply_rope(tensor: Any) -> Any:
        rotated = tensor[..., :rotary_dim].float()
        rest = tensor[..., rotary_dim:]
        half_rotary = rotary_dim // 2
        first = rotated[..., :half_rotary]
        second = rotated[..., half_rotary:]
        rope = torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)
        return torch.cat((rope.to(dtype), rest), dim=-1)

    def l2norm(inp: Any) -> Any:
        return inp.float() * torch.rsqrt(inp.float().pow(2).sum(dim=-1, keepdim=True) + 1e-6)

    def torch_chunk_gated_delta_rule(
        query: Any,
        key: Any,
        value: Any,
        g: Any,
        beta: Any,
        initial_state: Any,
        chunk_size: int = 64,
    ) -> tuple[Any, Any]:
        query = l2norm(query)
        key = l2norm(key)
        query, key, value, beta, g = [
            item.transpose(1, 2).contiguous().to(torch.float32) for item in (query, key, value, beta, g)
        ]

        batch_size, num_heads, sequence_length, k_head_dim = key.shape
        v_head_dim = value.shape[-1]
        pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
        query = F.pad(query, (0, 0, 0, pad_size))
        key = F.pad(key, (0, 0, 0, pad_size))
        value = F.pad(value, (0, 0, 0, pad_size))
        beta = F.pad(beta, (0, pad_size))
        g = F.pad(g, (0, pad_size))
        total_sequence_length = sequence_length + pad_size
        scale = 1.0 / math.sqrt(k_head_dim)
        query = query * scale

        v_beta = value * beta.unsqueeze(-1)
        k_beta = key * beta.unsqueeze(-1)
        query, key, value, k_beta, v_beta = [
            item.reshape(item.shape[0], item.shape[1], -1, chunk_size, item.shape[-1])
            for item in (query, key, value, k_beta, v_beta)
        ]
        g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
        mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

        g = g.cumsum(dim=-1)
        decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
        attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
        for idx in range(1, chunk_size):
            row = attn[..., idx, :idx].clone()
            sub = attn[..., :idx, :idx].clone()
            attn[..., idx, :idx] = row + (row.unsqueeze(-1) * sub).sum(-2)
        attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
        value = attn @ v_beta
        k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
        last_recurrent_state = initial_state.to(value)
        core_attn_out = torch.zeros(
            batch_size,
            num_heads,
            total_sequence_length // chunk_size,
            chunk_size,
            v_head_dim,
            device=value.device,
            dtype=value.dtype,
        )
        causal_mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

        for idx in range(0, total_sequence_length // chunk_size):
            q_i, k_i, v_i = query[:, :, idx], key[:, :, idx], value[:, :, idx]
            attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, idx]).masked_fill_(causal_mask, 0)
            v_prime = k_cumdecay[:, :, idx] @ last_recurrent_state
            v_new = v_i - v_prime
            attn_inter = (q_i * g[:, :, idx, :, None].exp()) @ last_recurrent_state
            core_attn_out[:, :, idx] = attn_inter + attn @ v_new
            last_recurrent_state = (
                last_recurrent_state * g[:, :, idx, -1, None, None].exp()
                + (k_i * (g[:, :, idx, -1, None] - g[:, :, idx]).exp()[..., None]).transpose(-1, -2) @ v_new
            )

        core_attn_out = core_attn_out.reshape(batch_size, num_heads, -1, v_head_dim)
        core_attn_out = core_attn_out[:, :, :sequence_length]
        core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(dtype)
        return core_attn_out, last_recurrent_state

    def torch_recurrent_gated_delta_rule(
        query: Any,
        key: Any,
        value: Any,
        g: Any,
        beta: Any,
        initial_state: Any,
    ) -> tuple[Any, Any]:
        query = l2norm(query)
        key = l2norm(key)
        query, key, value, beta, g = [
            item.transpose(1, 2).contiguous().to(torch.float32) for item in (query, key, value, beta, g)
        ]

        batch_size, num_heads, sequence_length, k_head_dim = key.shape
        v_head_dim = value.shape[-1]
        query = query * (1.0 / math.sqrt(k_head_dim))
        core_attn_out = torch.zeros(batch_size, num_heads, sequence_length, v_head_dim, device=value.device, dtype=value.dtype)
        last_recurrent_state = initial_state.to(value)

        for idx in range(sequence_length):
            q_t = query[:, :, idx]
            k_t = key[:, :, idx]
            v_t = value[:, :, idx]
            g_t = g[:, :, idx].exp().unsqueeze(-1).unsqueeze(-1)
            beta_t = beta[:, :, idx].unsqueeze(-1)

            last_recurrent_state = last_recurrent_state * g_t
            kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
            delta = (v_t - kv_mem) * beta_t
            last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
            core_attn_out[:, :, idx] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

        core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(dtype)
        return core_attn_out, last_recurrent_state

    def should_use_direct_linear_conv() -> bool:
        return linear_attention_conv_variant == "decode_direct" and mode == "decode" and tokens == 1

    def should_use_prefill_direct_linear_conv() -> bool:
        return linear_attention_conv_variant in {"decode_direct", "decode_direct_triton"} and mode == "prefill"

    def should_use_triton_direct_linear_conv() -> bool:
        return triton_linear_conv_decode

    def should_use_triton_prefill_linear_conv() -> bool:
        return triton_linear_conv_prefill and triton_prefill_direct_conv_kernel is not None

    official_prefill_conv_calls = 0
    official_prefill_conv_initial_state_true_calls = 0

    def linear_causal_conv(mixed_qkv: Any, layer_index: int, weights: dict[str, Any]) -> Any:
        nonlocal official_prefill_conv_calls
        nonlocal official_prefill_conv_initial_state_true_calls
        if native_moe_consumer_prefill_official_linear_surface and mode == "prefill":
            try:
                from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
            except Exception as exc:
                raise RuntimeError("formal prefill alignment requires vLLM causal_conv1d_fn") from exc
            qkv_bf16 = mixed_qkv.to(dtype=torch.bfloat16).contiguous()
            weight = weights["linear_conv1d_direct_weight"].to(
                device=qkv_bf16.device,
                dtype=torch.bfloat16,
            ).contiguous()
            state = linear_attention_initial_conv_states[layer_index].unsqueeze(0).to(
                dtype=torch.bfloat16
            ).contiguous()
            has_initial_state = int(position_start) > 0
            official_prefill_conv_calls += 1
            official_prefill_conv_initial_state_true_calls += int(has_initial_state)
            out_t_dim = causal_conv1d_fn(
                qkv_bf16.T,
                weight,
                None,
                conv_states=state,
                query_start_loc=torch.tensor([0, tokens], dtype=torch.int32, device=device),
                cache_indices=torch.tensor([0], dtype=torch.int32, device=device),
                has_initial_state=torch.tensor([has_initial_state], dtype=torch.bool, device=device),
                activation="silu",
                null_block_id=-1,
                validate_data=False,
            )
            linear_attention_conv_states[layer_index].copy_(state[0])
            return out_t_dim.T.contiguous().to(dtype)
        if should_use_triton_direct_linear_conv():
            if triton is None or triton_decode_direct_conv_kernel is None:
                raise RuntimeError("Triton decode direct conv kernel is not available")
            out = linear_attention_conv_outputs[layer_index]
            triton_decode_direct_conv_kernel[(triton.cdiv(linear_conv_dim, 256),)](
                mixed_qkv.reshape(linear_conv_dim),
                linear_attention_initial_conv_states[layer_index],
                weights["linear_conv1d_direct_weight"],
                out,
                linear_attention_conv_states[layer_index],
                linear_conv_dim,
                linear_conv_state_len,
                linear_conv_kernel_dim,
                block_c=256,
                num_warps=8,
            )
            return out.view(tokens, linear_conv_dim)
        conv_weight = weights["linear_conv1d"]
        raw = mixed_qkv.t().unsqueeze(0)
        if should_use_triton_prefill_linear_conv():
            if triton is None or triton_prefill_direct_conv_kernel is None or linear_attention_prefill_conv_output is None:
                raise RuntimeError("Triton prefill direct conv kernel is not available")
            out = linear_attention_prefill_conv_output[:tokens]
            triton_prefill_direct_conv_kernel[
                (triton.cdiv(tokens, prefill_conv_block_t), triton.cdiv(linear_conv_dim, prefill_conv_block_c))
            ](
                mixed_qkv,
                linear_attention_initial_conv_states[layer_index],
                weights["linear_conv1d_direct_weight"],
                out,
                linear_attention_conv_states[layer_index],
                mixed_qkv.stride(0),
                mixed_qkv.stride(1),
                tokens,
                linear_conv_dim,
                linear_conv_state_len,
                linear_conv_kernel_dim,
                block_t=prefill_conv_block_t,
                block_c=prefill_conv_block_c,
                num_warps=prefill_conv_num_warps,
            )
            return out
        if should_use_prefill_direct_linear_conv():
            raw_2d = raw.squeeze(0)
            conv_input = torch.cat((linear_attention_initial_conv_states[layer_index], raw_2d), dim=-1)
            direct_weight = weights["linear_conv1d_direct_weight"]
            conv_out = conv_input[:, :tokens] * direct_weight[:, :1]
            for kernel_index in range(1, linear_conv_kernel_dim):
                conv_out = conv_out + conv_input[:, kernel_index : kernel_index + tokens] * direct_weight[
                    :, kernel_index : kernel_index + 1
                ]
            linear_attention_conv_states[layer_index].copy_(conv_input[:, -linear_conv_state_len:])
            return F.silu(conv_out).t().contiguous().to(dtype)
        if should_use_direct_linear_conv():
            raw_2d = raw.squeeze(0)
            conv_window = linear_attention_conv_windows[layer_index]
            conv_window[:, :linear_conv_state_len].copy_(linear_attention_initial_conv_states[layer_index])
            conv_window[:, linear_conv_state_len:].copy_(raw_2d)
            linear_attention_conv_states[layer_index].copy_(conv_window[:, -linear_conv_state_len:])
            conv_out = (conv_window * weights["linear_conv1d_direct_weight"]).sum(dim=-1)
            return F.silu(conv_out).unsqueeze(0).to(dtype)
        if mode == "decode":
            conv_input = torch.cat((linear_attention_initial_conv_states[layer_index].unsqueeze(0), raw), dim=-1)
            conv_out = F.conv1d(conv_input.to(conv_weight.dtype), conv_weight, bias=None, groups=linear_conv_dim)
        else:
            conv_out = F.conv1d(
                raw.to(conv_weight.dtype),
                conv_weight,
                bias=None,
                padding=linear_conv_state_len,
                groups=linear_conv_dim,
            )[:, :, :tokens]
            if tokens < linear_conv_state_len:
                conv_input = F.pad(raw, (linear_conv_state_len - tokens, 0))
            else:
                conv_input = raw
        linear_attention_conv_states[layer_index].copy_(conv_input[:, :, -linear_conv_state_len:].squeeze(0))
        return F.silu(conv_out[:, :, -tokens:]).squeeze(0).t().contiguous().to(dtype)

    def should_use_fused_linear_input_proj() -> bool:
        if (
            linear_attention_input_proj_variant
            in {
                "prefill_fused_t_decode_fused_t_conv_triton",
                "prefill_fused_t_decode_fused_t_conv_qkv_triton",
            }
            and mode == "prefill"
        ):
            return True
        return (
            linear_attention_input_proj_variant
            in {
                "decode_fused",
                "decode_fused_t",
                "decode_fused_t_triton",
                "decode_fused_t_conv_triton",
                "decode_fused_t_conv_qkv_triton",
                "prefill_fused_t_decode_fused_t_conv_triton",
                "prefill_fused_t_decode_fused_t_conv_qkv_triton",
            }
            and mode == "decode"
            and tokens == 1
        )

    def vllm_linear_initial_state(layer_index: int) -> Any:
        if linear_attention_uses_vllm_prestates(linear_attention_variant):
            return linear_attention_initial_ssm_states_vllm[layer_index]
        return linear_attention_initial_ssm_states[layer_index].unsqueeze(0).transpose(-1, -2).contiguous()

    def use_prefill_vllm_state_handoff() -> bool:
        return (
            linear_attention_prefill_vllm_state_handoff
            and mode == "prefill"
            and linear_attention_uses_native_vllm_decode_state(linear_attention_variant)
        )

    def use_prefill_conv_post_prep_fusion() -> bool:
        return (
            linear_attention_prefill_conv_post_prep_fusion
            and mode == "prefill"
            and linear_attention_uses_vllm_fla(linear_attention_variant)
            and should_use_triton_prefill_linear_conv()
            and linear_key_heads == 16
            and linear_value_heads == 32
            and linear_key_head_dim == 128
            and linear_value_head_dim == 128
            and triton is not None
            and tl is not None
        )

    def use_prefill_fused_h_o(chunk_size: int) -> bool:
        return (
            linear_attention_prefill_fused_h_o
            and mode == "prefill"
            and chunk_size == 16
            and linear_key_heads == 16
            and linear_value_heads == 32
            and linear_key_head_dim == 128
            and linear_value_head_dim == 128
            and triton is not None
            and tl is not None
        )

    def use_prefill_fused_u_h_o(chunk_size: int) -> bool:
        return (
            linear_attention_prefill_fused_u_h_o
            and mode == "prefill"
            and chunk_size == 16
            and linear_key_heads == 16
            and linear_value_heads == 32
            and linear_key_head_dim == 128
            and linear_value_head_dim == 128
            and triton is not None
            and tl is not None
        )

    def prefill_chunk_initial_state(layer_index: int) -> Any:
        if use_prefill_vllm_state_handoff():
            if layer_index not in linear_attention_initial_ssm_states_vllm:
                linear_attention_initial_ssm_states_vllm[layer_index] = (
                    linear_attention_initial_ssm_states[layer_index].unsqueeze(0).transpose(-1, -2).contiguous()
                )
            if layer_index not in linear_attention_ssm_states_vllm:
                linear_attention_ssm_states_vllm[layer_index] = torch.empty_like(
                    linear_attention_initial_ssm_states_vllm[layer_index]
                )
            return vllm_linear_initial_state(layer_index)
        return linear_attention_initial_ssm_states[layer_index].unsqueeze(0).transpose(-1, -2).contiguous()

    def store_prefill_chunk_final_state(layer_index: int, final_state: Any) -> None:
        if use_prefill_vllm_state_handoff():
            if linear_attention_uses_native_vllm_decode_state_refswap(linear_attention_variant):
                linear_attention_ssm_states_vllm[layer_index] = final_state.contiguous()
            else:
                linear_attention_ssm_states_vllm[layer_index].copy_(final_state)
            return
        linear_attention_ssm_states[layer_index].copy_(final_state.squeeze(0).transpose(-1, -2))

    def cross_owner_beta_decay_bmm(k: Any, g_cumsum: Any, beta: Any) -> Any:
        batch, token_count, key_heads, head_dim = [int(value) for value in k.shape]
        _, beta_tokens, value_heads = [int(value) for value in beta.shape]
        if [batch, token_count, key_heads, head_dim] != [1, 8192, 16, 128]:
            raise RuntimeError("cross-owner beta/decay BMM received an unexpected K shape")
        if [batch, beta_tokens, value_heads] != [1, 8192, 32]:
            raise RuntimeError("cross-owner beta/decay BMM received an unexpected beta shape")
        if list(g_cumsum.shape) != [1, 8192, 32]:
            raise RuntimeError("cross-owner beta/decay BMM received an unexpected decay shape")
        block = 32
        chunks = token_count // block
        head_ratio = value_heads // key_heads
        matrices = batch * chunks * value_heads
        k_blocks = k.view(batch, chunks, block, key_heads, head_dim).permute(0, 1, 3, 2, 4)
        k_expanded = (
            k_blocks.unsqueeze(3)
            .expand(batch, chunks, key_heads, head_ratio, block, head_dim)
            .reshape(matrices, block, head_dim)
            .contiguous()
        )
        beta_flat = (
            beta.view(batch, chunks, block, value_heads)
            .permute(0, 1, 3, 2)
            .reshape(matrices, block, 1)
            .contiguous()
        )
        g_flat = (
            g_cumsum.view(batch, chunks, block, value_heads)
            .permute(0, 1, 3, 2)
            .reshape(matrices, block)
            .contiguous()
        )
        lower_mask_key = (block, str(k.device))
        lower_mask = cross_owner_lower_mask_cache.get(lower_mask_key)
        if lower_mask is None:
            lower_mask = torch.ones(
                (block, block), device=k.device, dtype=torch.bool
            ).tril(diagonal=-1)
            cross_owner_lower_mask_cache[lower_mask_key] = lower_mask
        weighted = (k_expanded.float() * beta_flat).to(torch.bfloat16)
        original_backend = torch.backends.cuda.preferred_blas_library()
        try:
            torch.backends.cuda.preferred_blas_library("hipblas")
            gram = torch.bmm(
                weighted,
                k_expanded.transpose(1, 2),
                out_dtype=torch.float32,
            )
        finally:
            torch.backends.cuda.preferred_blas_library(original_backend)
        decay = torch.exp(g_flat[:, :, None] - g_flat[:, None, :])
        gram.mul_(decay)
        gram.masked_fill_(~lower_mask, 0.0)
        result = (
            gram.view(batch, chunks, key_heads, head_ratio, block, block)
            .reshape(batch, chunks, value_heads, block, block)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
            .view(batch, token_count, value_heads, block)
        )
        cross_owner_prefill_stats["beta_decay_bmm_calls"] += 1
        return result

    def tuned_chunk_gated_delta_rule(
        q: Any,
        k: Any,
        value: Any,
        g: Any,
        beta: Any,
        *,
        initial_state: Any,
        output_final_state: bool,
        chunk_size: int,
        scale: float | None = None,
        cu_seqlens: Any = None,
        chunk_indices: Any = None,
        chunk_offsets: Any = None,
        use_qk_l2norm_in_kernel: bool = False,
        profile_stage: Callable[[str, Callable[[], Any]], Any] | None = None,
    ) -> tuple[Any, Any]:
        if (
            vllm_chunk_local_cumsum is None
            or vllm_chunk_scaled_dot_kkt_fwd is None
            or vllm_solve_tril is None
            or vllm_recompute_w_u_fwd is None
            or vllm_chunk_gated_delta_rule_fwd_h is None
            or vllm_chunk_fwd_o is None
            or vllm_l2norm_fwd is None
        ):
            raise RuntimeError("vLLM chunk-size-tuned FLA kernels were not loaded")
        run_stage = (lambda stage, fn: profile_stage(stage, fn)) if profile_stage is not None else (lambda _stage, fn: fn())
        if use_qk_l2norm_in_kernel:
            q = run_stage("l2norm_q", lambda: vllm_l2norm_fwd(q))
            k = run_stage("l2norm_k", lambda: vllm_l2norm_fwd(k))
        if scale is None:
            scale = k.shape[-1] ** -0.5
        g_cumsum = run_stage(
            "chunk_local_cumsum",
            lambda: vllm_chunk_local_cumsum(
                g,
                chunk_size=chunk_size,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
            ),
        )
        if use_cross_owner_prefill_composition():
            a_matrix = run_stage(
                "chunk_scaled_dot_kkt",
                lambda: cross_owner_beta_decay_bmm(k, g_cumsum, beta),
            )
        else:
            a_matrix = run_stage(
                "chunk_scaled_dot_kkt",
                lambda: vllm_chunk_scaled_dot_kkt_fwd(
                    k=k,
                    beta=beta,
                    g=g_cumsum,
                    cu_seqlens=cu_seqlens,
                    chunk_indices=chunk_indices,
                    chunk_size=chunk_size,
                    output_dtype=torch.float32,
                ),
            )
        a_matrix = run_stage(
            "solve_tril",
            lambda: vllm_solve_tril(
                A=a_matrix,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                output_dtype=k.dtype,
            ),
        )
        if use_prefill_fused_u_h_o(chunk_size):
            if (
                triton_fla_recompute_w_only_kernel is None
                or triton_fla_fused_u_h_o_kernel is None
                or cu_seqlens is not None
                or chunk_indices is not None
                or chunk_offsets is not None
                or output_final_state is not True
                or use_qk_l2norm_in_kernel
            ):
                raise RuntimeError("prefill fused u+h/o only supports retained equal-length chunk16 prefill")

            def recompute_w_only() -> Any:
                batch, token_count, key_heads, head_dim = k.shape
                _, _, value_heads, _value_dim = value.shape
                if batch != 1:
                    raise RuntimeError("prefill fused u+h/o only supports batch size 1")
                w_only = torch.empty(
                    (batch, token_count, value_heads, head_dim),
                    device=k.device,
                    dtype=k.dtype,
                )
                grid = (triton.cdiv(token_count, chunk_size), value_heads)
                triton_fla_recompute_w_only_kernel[grid](
                    k,
                    beta,
                    w_only,
                    a_matrix,
                    g_cumsum,
                    T=token_count,
                    H=value_heads,
                    Hg=key_heads,
                    K=head_dim,
                    BT=chunk_size,
                    BK=64,
                    num_warps=4,
                    num_stages=4,
                )
                return w_only

            w = run_stage("recompute_w_only", recompute_w_only)

            def fused_u_h_o() -> tuple[Any, Any]:
                batch, token_count, q_heads, head_dim = q.shape
                _, _, value_heads, value_dim = value.shape
                if batch != 1:
                    raise RuntimeError("prefill fused u+h/o only supports batch size 1")
                out = torch.empty_like(value)
                final_state = torch.empty_like(initial_state, dtype=torch.float32)
                grid = (triton.cdiv(value_dim, 32), value_heads)
                triton_fla_fused_u_h_o_kernel[grid](
                    q,
                    k,
                    value,
                    beta,
                    w,
                    a_matrix,
                    g_cumsum,
                    initial_state,
                    out,
                    final_state,
                    scale,
                    T=token_count,
                    H=value_heads,
                    Hg=q_heads,
                    K=head_dim,
                    V=value_dim,
                    BT=chunk_size,
                    NT=triton.cdiv(token_count, chunk_size),
                    BV=32,
                    num_warps=2,
                    num_stages=2,
                )
                return out, final_state

            out, final_state = run_stage("fused_u_chunk_gated_delta_rule_fwd_h_chunk_fwd_o", fused_u_h_o)
            return out.to(q.dtype), final_state

        w, u = run_stage(
            "recompute_w_u",
            lambda: vllm_recompute_w_u_fwd(
                k=k,
                v=value,
                beta=beta,
                A=a_matrix,
                g_cumsum=g_cumsum,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
            ),
        )
        if use_prefill_fused_h_o(chunk_size):
            if (
                triton_fla_fused_chunk_h_o_kernel is None
                or cu_seqlens is not None
                or chunk_indices is not None
                or chunk_offsets is not None
                or output_final_state is not True
                or use_qk_l2norm_in_kernel
            ):
                raise RuntimeError("prefill fused h/o only supports retained equal-length chunk16 prefill")

            def fused_h_o() -> tuple[Any, Any]:
                batch, token_count, q_heads, head_dim = q.shape
                _, _, value_heads, value_dim = u.shape
                if batch != 1:
                    raise RuntimeError("prefill fused h/o only supports batch size 1")
                out = torch.empty_like(u)
                final_state = torch.empty_like(initial_state, dtype=torch.float32)
                grid = (triton.cdiv(value_dim, 32), value_heads)
                triton_fla_fused_chunk_h_o_kernel[grid](
                    q,
                    k,
                    w,
                    u,
                    g_cumsum,
                    initial_state,
                    out,
                    final_state,
                    scale,
                    T=token_count,
                    H=value_heads,
                    Hg=q_heads,
                    K=head_dim,
                    V=value_dim,
                    BT=chunk_size,
                    NT=triton.cdiv(token_count, chunk_size),
                    BV=32,
                    num_warps=2,
                    num_stages=2,
                )
                return out, final_state

            out, final_state = run_stage("fused_chunk_gated_delta_rule_fwd_h_chunk_fwd_o", fused_h_o)
            return out.to(q.dtype), final_state
        h, v_new, final_state = run_stage(
            "chunk_gated_delta_rule_fwd_h",
            lambda: vllm_chunk_gated_delta_rule_fwd_h(
                k=k,
                w=w,
                u=u,
                g=g_cumsum,
                initial_state=initial_state,
                output_final_state=output_final_state,
                chunk_size=chunk_size,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                chunk_offsets=chunk_offsets,
            ),
        )
        out = run_stage(
            "chunk_fwd_o",
            lambda: vllm_chunk_fwd_o(
                q=q,
                k=k,
                v=v_new,
                h=h,
                g=g_cumsum,
                scale=scale,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                chunk_size=chunk_size,
            ),
        )
        return out.to(q.dtype), final_state

    def should_use_grouped_bmm_full_attention() -> bool:
        return full_attention_variant in {"decode_grouped_bmm", "decode_grouped_bmm_bf16"} and mode == "decode" and tokens == 1

    def should_use_triton_full_attention_proj() -> bool:
        return triton_full_attention_proj_decode

    def should_use_triton_full_attention_fused_qkv() -> bool:
        return triton_full_attention_fused_qkv_decode

    def should_use_triton_full_attention_norm_rope() -> bool:
        return triton_full_attention_norm_rope_decode

    def should_use_triton_full_attention_fused_gate_o_proj() -> bool:
        return triton_full_attention_fused_gate_o_proj_decode

    def should_use_triton_full_attention_fused_norm_rope_kv_write() -> bool:
        return triton_full_attention_fused_norm_rope_kv_write_decode

    def should_use_triton_linear_output_proj() -> bool:
        return triton_linear_output_proj_decode

    def should_use_triton_linear_gated_norm() -> bool:
        return triton_linear_gated_norm_decode

    def should_use_triton_linear_input_proj() -> bool:
        return triton_linear_input_proj_decode

    def should_use_triton_fused_linear_input_proj_conv() -> bool:
        return triton_fused_linear_input_proj_conv_decode and not triton_fused_linear_input_proj_conv_qkv_decode

    def should_use_triton_fused_linear_input_proj_conv_qkv_layout() -> bool:
        return triton_fused_linear_input_proj_conv_qkv_decode

    def should_use_triton_shared_expert_proj() -> bool:
        return triton_shared_expert_proj_decode

    def should_use_triton_shared_expert_fused_input() -> bool:
        return triton_shared_expert_fused_input_decode

    def should_use_triton_shared_expert_fused_down() -> bool:
        return triton_shared_expert_fused_down_decode

    def triton_full_attention_projection(
        inp: Any,
        layer_index: int,
        weights: dict[str, Any],
        weight_key: str,
        output_key: str,
        input_features: int,
        output_features: int,
        *,
        block_n: int = 64,
        block_k: int = 256,
    ) -> Any:
        if triton is None or triton_matvec_kernel is None:
            raise RuntimeError("Triton matvec kernel is not available")
        if weight_key == "o_proj_t":
            if vllm_moe_ops is None:
                raise RuntimeError("ROCm projection provider is unavailable")
            return torch.ops.vllm.rocm_unquantized_gemm(
                inp.reshape(1, input_features), weights["o_proj"], None
            ).view(tokens, output_features)
        out = full_attention_proj_outputs[layer_index][output_key]
        launch_warps = 8
        triton_matvec_kernel[(triton.cdiv(output_features, block_n),)](
            inp.reshape(input_features),
            weights[weight_key],
            out,
            input_features,
            output_features,
            block_n=block_n,
            block_k=block_k,
            num_warps=launch_warps,
        )
        return out.view(tokens, output_features)

    def triton_full_attention_qkv_projection(inp: Any, layer_index: int, weights: dict[str, Any]) -> Any:
        return triton_full_attention_projection(
            inp,
            layer_index,
            weights,
            "full_qkv_proj_fused_t",
            "qkv",
            hidden,
            2 * q_dim + 2 * kv_dim,
        )

    def triton_full_attention_gated_o_projection(
        attn_out: Any,
        gate: Any,
        layer_index: int,
        weights: dict[str, Any],
    ) -> Any:
        if triton is None or triton_full_attention_gated_o_proj_kernel is None:
            raise RuntimeError("Triton full-attention gated o_proj kernel is not available")
        out = full_attention_proj_outputs[layer_index]["o"]
        triton_full_attention_gated_o_proj_kernel[(triton.cdiv(hidden, 64),)](
            attn_out.reshape(q_dim),
            gate.reshape(q_dim),
            weights["o_proj_t"],
            out,
            q_dim,
            hidden,
            block_n=64,
            block_k=256,
            num_warps=4,
        )
        return out.view(tokens, hidden)

    def triton_full_attention_head_norm_rope(
        inp: Any,
        layer_index: int,
        weights: dict[str, Any],
        weight_key: str,
        output_key: str,
        rows: int,
    ) -> Any:
        if triton is None or triton_head_norm_rope_kernel is None:
            raise RuntimeError("Triton head norm+RoPE kernel is not available")
        out = full_attention_norm_rope_outputs[layer_index][output_key]
        triton_head_norm_rope_kernel[(tokens * rows,)](
            inp,
            weights[weight_key],
            cos_flat,
            sin_flat,
            out,
            head_dim,
            rotary_dim,
            rows,
            inp.stride(0),
            inp.stride(1),
            block_h=triton.next_power_of_2(head_dim),
            num_warps=4,
        )
        return out

    def triton_full_attention_norm_rope_kv_write(
        q: Any,
        k: Any,
        v: Any,
        layer_index: int,
        weights: dict[str, Any],
    ) -> Any:
        if triton is None or triton_full_attention_norm_rope_kv_write_kernel is None:
            raise RuntimeError("Triton full-attention norm+RoPE KV-write kernel is not available")
        q_out = full_attention_norm_rope_outputs[layer_index]["q"]
        k_cache, v_cache = full_attention_kv_caches[layer_index]
        triton_full_attention_norm_rope_kv_write_kernel[(heads + 2 * kv_heads,)](
            q,
            k,
            v,
            weights["q_norm"],
            weights["k_norm"],
            cos_flat,
            sin_flat,
            q_out,
            k_cache,
            v_cache,
            int(position_start),
            head_dim,
            rotary_dim,
            heads,
            kv_heads,
            q.stride(1),
            k.stride(1),
            v.stride(1),
            full_attention_cache_token_stride(k_cache),
            full_attention_cache_head_stride(k_cache),
            block_h=triton.next_power_of_2(head_dim),
            num_warps=2,
        )
        return q_out

    def full_attention_cache_token_stride(cache: Any) -> int:
        return int(cache.stride(1) if full_attention_kv_cache_layout == "grouped" else cache.stride(0))

    def full_attention_cache_head_stride(cache: Any) -> int:
        return int(cache.stride(0) if full_attention_kv_cache_layout == "grouped" else cache.stride(1))

    def write_full_attention_kv_cache(k_cache: Any, v_cache: Any, k: Any, v: Any) -> None:
        if full_attention_kv_cache_layout == "grouped":
            k_cache[:, position_start:position_end].copy_(k.transpose(0, 1))
            v_cache[:, position_start:position_end].copy_(v.transpose(0, 1))
            return
        k_cache[position_start:position_end].copy_(k)
        v_cache[position_start:position_end].copy_(v)

    def grouped_full_attention_cache_views(k_cache: Any, v_cache: Any, cache_end: int) -> tuple[Any, Any]:
        if full_attention_kv_cache_layout == "grouped":
            return k_cache[:, :cache_end], v_cache[:, :cache_end]
        return k_cache[:cache_end].transpose(0, 1), v_cache[:cache_end].transpose(0, 1)

    def triton_linear_input_projection(inp: Any, layer_index: int, weights: dict[str, Any]) -> Any:
        if triton is None or triton_matvec_kernel is None:
            raise RuntimeError("Triton matvec kernel is not available")
        out = linear_attention_input_proj_outputs[layer_index]
        out_features = int(weights["linear_input_proj_fused"].shape[0])
        triton_matvec_kernel[(triton.cdiv(out_features, 32),)](
            inp.reshape(hidden),
            weights["linear_input_proj_fused_t"],
            out,
            hidden,
            out_features,
            block_n=32,
            block_k=256,
            num_warps=8,
        )
        return out.view(tokens, out_features)

    def triton_fused_linear_input_projection_conv(inp: Any, layer_index: int, weights: dict[str, Any]) -> Any:
        if triton is None or triton_fused_input_proj_conv_kernel is None:
            raise RuntimeError("Triton fused input projection and conv kernel is not available")
        out = linear_attention_input_proj_outputs[layer_index]
        out_features = int(weights["linear_input_proj_fused"].shape[0])
        triton_fused_input_proj_conv_kernel[(triton.cdiv(out_features, 32),)](
            inp.reshape(hidden),
            weights["linear_input_proj_fused_t"],
            linear_attention_initial_conv_states[layer_index],
            weights["linear_conv1d_direct_weight"],
            out,
            linear_attention_conv_states[layer_index],
            hidden,
            out_features,
            linear_conv_dim,
            linear_conv_state_len,
            linear_conv_kernel_dim,
            block_n=32,
            block_k=256,
            num_warps=8,
        )
        return out.view(tokens, out_features)

    def triton_fused_linear_input_projection_conv_qkv_layout(
        inp: Any,
        layer_index: int,
        weights: dict[str, Any],
    ) -> tuple[Any, Any, Any, Any, Any, Any]:
        if triton is None or triton_fused_input_proj_conv_qkv_kernel is None:
            raise RuntimeError("Triton fused input projection, conv, and qkv-layout kernel is not available")
        outputs = linear_attention_qkv_layout_outputs[layer_index]
        out_features = int(weights["linear_input_proj_fused"].shape[0])
        triton_fused_input_proj_conv_qkv_kernel[(triton.cdiv(out_features, 32),)](
            inp.reshape(hidden),
            weights["linear_input_proj_fused_t"],
            linear_attention_initial_conv_states[layer_index],
            weights["linear_conv1d_direct_weight"],
            outputs["q"],
            outputs["k"],
            outputs["v"],
            outputs["z"],
            outputs["a"],
            outputs["b"],
            linear_attention_conv_states[layer_index],
            hidden,
            out_features,
            linear_key_dim,
            linear_value_dim,
            linear_conv_dim,
            linear_value_heads,
            linear_conv_state_len,
            linear_conv_kernel_dim,
            block_n=32,
            block_k=256,
            num_warps=8,
        )
        return outputs["q"], outputs["k"], outputs["v"], outputs["z"], outputs["a"], outputs["b"]

    def triton_shared_expert_projection(
        inp: Any,
        layer_index: int,
        weights: dict[str, Any],
        weight_key: str,
        output_key: str,
        input_features: int,
        output_features: int,
        *,
        block_n: int = 64,
        block_k: int = 256,
    ) -> Any:
        if triton is None or triton_matvec_kernel is None:
            raise RuntimeError("Triton matvec kernel is not available")
        if weight_key == "shared_down_proj_t":
            if vllm_moe_ops is None or wvsplitk_cu_count is None:
                raise RuntimeError("wvSplitK projection provider is unavailable")
            return vllm_moe_ops.wvSplitK(
                weights["shared_down_proj"],
                inp.reshape(1, input_features),
                wvsplitk_cu_count,
                None,
            ).view(tokens, output_features)
        out = shared_expert_proj_outputs[layer_index][output_key]
        launch_warps = 8
        if weight_key == "shared_input_proj_fused_t":
            block_n = 16
            launch_warps = 2
        triton_matvec_kernel[(triton.cdiv(output_features, block_n),)](
            inp.reshape(input_features),
            weights[weight_key],
            out,
            input_features,
            output_features,
            block_n=block_n,
            block_k=block_k,
            num_warps=launch_warps,
        )
        return out.view(tokens, output_features)

    def triton_shared_expert_fused_down(shared_input: Any, layer_index: int, weights: dict[str, Any]) -> Any:
        if triton is None or triton_fused_shared_down_kernel is None:
            raise RuntimeError("Triton fused shared expert down kernel is not available")
        out = shared_expert_proj_outputs[layer_index]["down_proj"]
        triton_fused_shared_down_kernel[(triton.cdiv(hidden, 64),)](
            shared_input.reshape(1 + 2 * shared_intermediate),
            weights["shared_down_proj_t"],
            out,
            shared_intermediate,
            hidden,
            block_n=64,
            block_k=256,
            num_warps=4,
        )
        return out.view(tokens, hidden)

    def triton_linear_output_projection(inp: Any, layer_index: int, weights: dict[str, Any]) -> Any:
        if vllm_moe_ops is None or wvsplitk_cu_count is None:
            raise RuntimeError("wvSplitK projection provider is unavailable")
        return vllm_moe_ops.wvSplitK(
            weights["linear_out_proj"],
            inp.reshape(1, linear_value_dim),
            wvsplitk_cu_count,
            None,
        ).view(tokens, hidden)

    def triton_linear_gated_norm(core_attn_out: Any, z: Any, layer_index: int, weights: dict[str, Any]) -> Any:
        if triton is None or triton_linear_gated_norm_kernel is None:
            raise RuntimeError("Triton linear gated norm kernel is not available")
        rows = tokens * linear_value_heads
        out = linear_attention_gated_norm_outputs[layer_index]
        triton_linear_gated_norm_kernel[(rows,)](
            core_attn_out.reshape(rows, linear_value_head_dim),
            z.reshape(rows, linear_value_head_dim),
            weights["linear_norm"],
            out.reshape(rows, linear_value_head_dim),
            linear_value_head_dim,
            1e-6,
            block_h=triton.next_power_of_2(linear_value_head_dim),
            num_warps=4,
        )
        return out.view(tokens, linear_value_dim)


    def triton_linear_gated_norm_from_invstd(
        core_attn_out: Any,
        z: Any,
        layer_index: int,
        weights: dict[str, Any],
    ) -> Any:
        if triton is None or triton_linear_gated_norm_from_invstd_kernel is None:
            raise RuntimeError("Triton exact-staged linear gated norm kernel is not available")
        rows = tokens * linear_value_heads
        core_flat = core_attn_out.reshape(rows, linear_value_head_dim)
        variance = core_flat.float().pow(2).mean(dim=-1, keepdim=True)
        invstd = torch.rsqrt(variance + 1e-6)
        out = (
            linear_attention_gated_norm_prefill_output[:tokens]
            if linear_attention_gated_norm_prefill_output is not None
            else None
        )
        if out is None:
            raise RuntimeError("exact-staged prefill gated norm shared output is unavailable")
        triton_linear_gated_norm_from_invstd_kernel[(rows,)](
            core_flat,
            z.reshape(rows, linear_value_head_dim),
            weights["linear_norm"],
            invstd,
            out.reshape(rows, linear_value_head_dim),
            linear_value_head_dim,
            block_h=triton.next_power_of_2(linear_value_head_dim),
            num_warps=4,
        )
        return out.view(tokens, linear_value_dim)

    def vllm_fused_post_conv_prep_with_block_t(
        *,
        conv_output: Any,
        a: Any,
        b: Any,
        weights: dict[str, Any],
        block_t: int,
    ) -> tuple[Any, Any, Any, Any, Any]:
        if triton is None or vllm_fused_post_conv_kernel is None:
            raise RuntimeError("vLLM fused_post_conv_prep kernel is not available")
        conv_output = conv_output.contiguous()
        a = a.contiguous()
        b = b.contiguous()
        length = int(conv_output.shape[0])
        qkv_dim = int(conv_output.shape[1])
        expected_qkv_dim = 2 * linear_key_heads * linear_key_head_dim + linear_value_heads * linear_value_head_dim
        if qkv_dim != expected_qkv_dim:
            raise RuntimeError(f"qkv_dim={qkv_dim} != expected {expected_qkv_dim}")
        q = torch.empty(length, linear_key_heads, linear_key_head_dim, dtype=conv_output.dtype, device=conv_output.device)
        k = torch.empty_like(q)
        value = torch.empty(
            length,
            linear_value_heads,
            linear_value_head_dim,
            dtype=conv_output.dtype,
            device=conv_output.device,
        )
        g = torch.empty(length, linear_value_heads, dtype=torch.float32, device=conv_output.device)
        beta = torch.empty_like(g)
        vllm_fused_post_conv_kernel[(triton.cdiv(length, block_t), linear_key_heads + linear_value_heads)](
            mixed_qkv_ptr=conv_output,
            a_ptr=a,
            b_ptr=b,
            A_log_ptr=weights["linear_A_log"].contiguous(),
            dt_bias_ptr=weights["linear_dt_bias"].contiguous(),
            q_ptr=q,
            k_ptr=k,
            v_ptr=value,
            g_ptr=g,
            beta_ptr=beta,
            stride_x_tok=conv_output.stride(0),
            stride_a_tok=a.stride(0),
            stride_b_tok=b.stride(0),
            stride_q_tok=q.stride(0),
            stride_k_tok=k.stride(0),
            stride_v_tok=value.stride(0),
            L=length,
            H=linear_key_heads,
            HV=linear_value_heads,
            K=linear_key_head_dim,
            V=linear_value_head_dim,
            APPLY_L2NORM=True,
            L2NORM_EPS=1e-6,
            OUTPUT_G_EXP=False,
            SOFTPLUS_THRESHOLD=20.0,
            BLOCK_T=block_t,
            BK=triton.next_power_of_2(linear_key_head_dim),
            BV=triton.next_power_of_2(linear_value_head_dim),
            num_warps=4,
            num_stages=2,
        )
        return q, k, value, g, beta

    def run_vllm_fused_post_conv_prep(conv_qkv: Any, a: Any, b: Any, weights: dict[str, Any]) -> tuple[Any, Any, Any, Any, Any]:
        if linear_attention_post_conv_prep_block_t is not None:
            return vllm_fused_post_conv_prep_with_block_t(
                conv_output=conv_qkv,
                a=a,
                b=b,
                weights=weights,
                block_t=linear_attention_post_conv_prep_block_t,
            )
        if vllm_fused_post_conv_prep is None:
            raise RuntimeError("vLLM fused_post_conv_prep kernel was not loaded")
        return vllm_fused_post_conv_prep(
            conv_output=conv_qkv.contiguous(),
            a=a.contiguous(),
            b=b.contiguous(),
            A_log=weights["linear_A_log"].contiguous(),
            dt_bias=weights["linear_dt_bias"].contiguous(),
            num_k_heads=linear_key_heads,
            head_k_dim=linear_key_head_dim,
            head_v_dim=linear_value_head_dim,
            apply_l2norm=True,
            output_g_exp=False,
        )

    def run_triton_prefill_conv_post_prep(
        mixed_qkv: Any,
        a: Any,
        b: Any,
        layer_index: int,
        weights: dict[str, Any],
    ) -> tuple[Any, Any, Any, Any, Any]:
        if triton is None or triton_prefill_conv_post_prep_kernel is None:
            raise RuntimeError("Triton prefill conv+post-prep kernel is not available")
        if mixed_qkv is None:
            raise RuntimeError("prefill conv+post-prep fusion requires raw mixed_qkv input")
        raw = mixed_qkv.contiguous()
        a = a.contiguous()
        b = b.contiguous()
        length = int(raw.shape[0])
        q = torch.empty(length, linear_key_heads, linear_key_head_dim, dtype=raw.dtype, device=raw.device)
        k = torch.empty_like(q)
        value = torch.empty(
            length,
            linear_value_heads,
            linear_value_head_dim,
            dtype=raw.dtype,
            device=raw.device,
        )
        g = torch.empty(length, linear_value_heads, dtype=torch.float32, device=raw.device)
        beta = torch.empty_like(g)
        triton_prefill_conv_post_prep_kernel[(triton.cdiv(length, 16), linear_key_heads + linear_value_heads)](
            raw,
            linear_attention_initial_conv_states[layer_index],
            weights["linear_conv1d_direct_weight"],
            a,
            b,
            weights["linear_A_log"].contiguous(),
            weights["linear_dt_bias"].contiguous(),
            q,
            k,
            value,
            g,
            beta,
            linear_attention_conv_states[layer_index],
            raw.stride(0),
            raw.stride(1),
            a.stride(0),
            b.stride(0),
            q.stride(0),
            k.stride(0),
            value.stride(0),
            length,
            linear_conv_state_len,
            linear_conv_kernel_dim,
            linear_key_heads,
            linear_value_heads,
            linear_key_head_dim,
            linear_value_head_dim,
            1e-6,
            20.0,
            block_t=16,
            BK=triton.next_power_of_2(linear_key_head_dim),
            BV=triton.next_power_of_2(linear_value_head_dim),
            num_warps=8,
            num_stages=2,
        )
        return q, k, value, g, beta

    def split_fused_linear_input_proj(combined: Any, weights: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
        split_sizes = [
            int(weights["linear_in_proj_qkv"].shape[0]),
            int(weights["linear_in_proj_z"].shape[0]),
            int(weights["linear_in_proj_a"].shape[0]),
            int(weights["linear_in_proj_b"].shape[0]),
        ]
        mixed_qkv, z_raw, a, b = combined.split(split_sizes, dim=-1)
        z = z_raw.view(tokens, linear_value_heads, linear_value_head_dim)
        return mixed_qkv, z, a, b

    def linear_input_projection(inp: Any, layer_index: int, weights: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
        if should_use_triton_fused_linear_input_proj_conv_qkv_layout():
            raise RuntimeError("qkv-layout fused input projection is handled directly by linear_attention()")
        if should_use_fused_linear_input_proj():
            if should_use_triton_fused_linear_input_proj_conv():
                combined = triton_fused_linear_input_projection_conv(inp, layer_index, weights)
            elif should_use_triton_linear_input_proj():
                combined = triton_linear_input_projection(inp, layer_index, weights)
            elif linear_attention_input_proj_variant in {
                "decode_fused_t",
                "prefill_fused_t_decode_fused_t_conv_triton",
                "prefill_fused_t_decode_fused_t_conv_qkv_triton",
            }:
                fused_weight = weights["linear_input_proj_fused_t"]
                if use_cross_owner_prefill_composition():
                    original_backend = torch.backends.cuda.preferred_blas_library()
                    try:
                        torch.backends.cuda.preferred_blas_library("hipblas")
                        combined = inp @ fused_weight
                    finally:
                        torch.backends.cuda.preferred_blas_library(original_backend)
                    cross_owner_prefill_stats["hipblas_projection_calls"] += 1
                else:
                    combined = inp @ fused_weight
            else:
                combined = inp @ weights["linear_input_proj_fused"].t()
            return split_fused_linear_input_proj(combined, weights)
        mixed_qkv = inp @ weights["linear_in_proj_qkv"].t()
        z = (inp @ weights["linear_in_proj_z"].t()).view(tokens, linear_value_heads, linear_value_head_dim)
        a = inp @ weights["linear_in_proj_a"].t()
        b = inp @ weights["linear_in_proj_b"].t()
        return mixed_qkv, z, a, b

    def linear_attention(inp: Any, layer_index: int, weights: dict[str, Any]) -> Any:
        prepacked_qkv: tuple[Any, Any, Any] | None = None
        if should_use_triton_fused_linear_input_proj_conv_qkv_layout():
            q, k, value, z, a, b = triton_fused_linear_input_projection_conv_qkv_layout(inp, layer_index, weights)
            prepacked_qkv = (q, k, value)
            conv_qkv = None
        else:
            mixed_qkv, z, a, b = linear_input_projection(inp, layer_index, weights)
            conv_qkv = (
                None
                if use_prefill_conv_post_prep_fusion()
                else (
                    mixed_qkv
                    if should_use_triton_fused_linear_input_proj_conv()
                    else linear_causal_conv(mixed_qkv, layer_index, weights)
                )
            )
        if should_use_packed_linear_gdn():
            if vllm_fused_recurrent_gdn_packed_decode is None:
                raise RuntimeError("vLLM packed recurrent GDN kernel was not loaded")
            if linear_attention_uses_packed_state_refswap(linear_attention_variant):
                packed_state = linear_attention_packed_initial_ssm_states[layer_index]
            else:
                packed_state = linear_attention_packed_ssm_states[layer_index]
                packed_state.copy_(linear_attention_packed_initial_ssm_states[layer_index])
            core_attn_out, final_state = vllm_fused_recurrent_gdn_packed_decode(
                mixed_qkv=conv_qkv.contiguous(),
                a=a.contiguous(),
                b=b.contiguous(),
                A_log=weights["linear_A_log"].contiguous(),
                dt_bias=weights["linear_dt_bias"].contiguous(),
                scale=linear_key_head_dim**-0.5,
                initial_state=packed_state,
                out=linear_attention_packed_outputs[layer_index],
                ssm_state_indices=linear_attention_packed_state_indices[layer_index],
                use_qk_l2norm_in_kernel=True,
            )
            if not linear_attention_uses_packed_state_refswap(linear_attention_variant):
                linear_attention_ssm_states[layer_index].copy_(final_state[1].transpose(-1, -2))
        elif linear_attention_uses_vllm_auto_decode(linear_attention_variant) and mode == "decode":
            if vllm_fused_recurrent_gdn_update is None:
                raise RuntimeError("vLLM recurrent GDN kernel was not loaded")
            if prepacked_qkv is None:
                if conv_qkv is None:
                    raise RuntimeError("linear-attention decode qkv source is missing")
                q_raw, k_raw, value = conv_qkv.split([linear_key_dim, linear_key_dim, linear_value_dim], dim=-1)
                q = q_raw.view(tokens, linear_key_heads, linear_key_head_dim).unsqueeze(0).contiguous()
                k = k_raw.view(tokens, linear_key_heads, linear_key_head_dim).unsqueeze(0).contiguous()
                value = value.view(tokens, linear_value_heads, linear_value_head_dim).unsqueeze(0).contiguous()
            else:
                q, k, value = prepacked_qkv
            core_attn_out, final_state = vllm_fused_recurrent_gdn_update(
                A_log=weights["linear_A_log"].contiguous(),
                a=a.contiguous(),
                b=b.contiguous(),
                dt_bias=weights["linear_dt_bias"].contiguous(),
                q=q,
                k=k,
                v=value,
                initial_state=vllm_linear_initial_state(layer_index),
                inplace_final_state=False,
                use_qk_l2norm_in_kernel=True,
            )
            if linear_attention_uses_native_vllm_decode_state(linear_attention_variant):
                if linear_attention_uses_native_vllm_decode_state_refswap(linear_attention_variant):
                    linear_attention_ssm_states_vllm[layer_index] = final_state
                else:
                    linear_attention_ssm_states_vllm[layer_index].copy_(final_state)
            else:
                linear_attention_ssm_states[layer_index].copy_(final_state.squeeze(0).transpose(-1, -2))
        elif linear_attention_uses_vllm_fla(linear_attention_variant):
            if vllm_fused_post_conv_prep is None or vllm_chunk_gated_delta_rule is None:
                raise RuntimeError("vLLM FLA kernels were not loaded")
            if use_prefill_conv_post_prep_fusion():
                q, k, value, g, beta = run_triton_prefill_conv_post_prep(mixed_qkv, a, b, layer_index, weights)
            else:
                q, k, value, g, beta = run_vllm_fused_post_conv_prep(conv_qkv, a, b, weights)
            model_layer = int(layer_weights[layer_index]["layer"])
            if (
                use_q8192_compound_provider()
                and q8192_compound_provider().component_enabled("fused_gdn")
                and model_layer != 0
            ):
                core_attn_out, provider_final_state = q8192_compound_provider().launch_gdn(
                    q=q,
                    k=k,
                    value=value,
                    g=g,
                    beta=beta,
                    model_layer=model_layer,
                )
                store_prefill_chunk_final_state(layer_index, provider_final_state.unsqueeze(0))
            else:
                chunk_size = linear_attention_chunk_size(linear_attention_variant)
                initial_state = prefill_chunk_initial_state(layer_index)
                if chunk_size is None:
                    core_attn_out, final_state = vllm_chunk_gated_delta_rule(
                        q=q.unsqueeze(0),
                        k=k.unsqueeze(0),
                        v=value.unsqueeze(0),
                        g=g.unsqueeze(0),
                        beta=beta.unsqueeze(0),
                        initial_state=initial_state,
                        output_final_state=True,
                        use_qk_l2norm_in_kernel=False,
                    )
                else:
                    core_attn_out, final_state = tuned_chunk_gated_delta_rule(
                        q=q.unsqueeze(0),
                        k=k.unsqueeze(0),
                        value=value.unsqueeze(0),
                        g=g.unsqueeze(0),
                        beta=beta.unsqueeze(0),
                        initial_state=initial_state,
                        output_final_state=True,
                        chunk_size=chunk_size,
                        use_qk_l2norm_in_kernel=False,
                    )
                store_prefill_chunk_final_state(layer_index, final_state)
        else:
            q, k, value = conv_qkv.split([linear_key_dim, linear_key_dim, linear_value_dim], dim=-1)
            q = q.view(tokens, linear_key_heads, linear_key_head_dim)
            k = k.view(tokens, linear_key_heads, linear_key_head_dim)
            value = value.view(tokens, linear_value_heads, linear_value_head_dim)
            repeat_factor = linear_value_heads // linear_key_heads
            if repeat_factor > 1:
                q = q.repeat_interleave(repeat_factor, dim=1)
                k = k.repeat_interleave(repeat_factor, dim=1)

            beta = torch.sigmoid(b).view(1, tokens, linear_value_heads)
            g = (
                -weights["linear_A_log"].float().exp().unsqueeze(0)
                * F.softplus(a.float() + weights["linear_dt_bias"].float().unsqueeze(0))
            ).view(1, tokens, linear_value_heads)
            initial_state = linear_attention_initial_ssm_states[layer_index].unsqueeze(0)
            q = q.unsqueeze(0)
            k = k.unsqueeze(0)
            value = value.unsqueeze(0)
            if mode == "decode":
                core_attn_out, final_state = torch_recurrent_gated_delta_rule(q, k, value, g, beta, initial_state)
            else:
                core_attn_out, final_state = torch_chunk_gated_delta_rule(q, k, value, g, beta, initial_state)
            linear_attention_ssm_states[layer_index].copy_(final_state.squeeze(0))

        if should_use_triton_linear_gated_norm():
            gated_out = triton_linear_gated_norm(core_attn_out, z, layer_index, weights)
        elif triton_linear_gated_norm_prefill:
            gated_out = triton_linear_gated_norm_from_invstd(
                core_attn_out,
                z,
                layer_index,
                weights,
            )
        else:
            core_flat = core_attn_out.reshape(-1, linear_value_head_dim)
            z_flat = z.reshape(-1, linear_value_head_dim)
            variance = core_flat.float().pow(2).mean(dim=-1, keepdim=True)
            gated = (
                core_flat.float()
                * torch.rsqrt(variance + 1e-6)
                * weights["linear_norm"].float()
            ).to(dtype)
            gated = (gated.float() * F.silu(z_flat.float())).to(dtype)
            gated_out = gated.view(tokens, linear_value_dim)
        if should_use_triton_linear_output_proj():
            return triton_linear_output_projection(gated_out, layer_index, weights)
        return gated_out @ weights["linear_out_proj"].t()

    def full_attention(inp: Any, layer_index: int, weights: dict[str, Any]) -> Any:
        if should_use_triton_full_attention_fused_qkv():
            qkv = triton_full_attention_qkv_projection(inp, layer_index, weights)
            q_gate, k, v = qkv.split([2 * q_dim, kv_dim, kv_dim], dim=-1)
        elif should_use_triton_full_attention_proj():
            q_gate = triton_full_attention_projection(
                inp, layer_index, weights, "q_proj_t", "q_gate", hidden, 2 * q_dim
            )
            k = triton_full_attention_projection(inp, layer_index, weights, "k_proj_t", "k", hidden, kv_dim)
            v = triton_full_attention_projection(
                inp, layer_index, weights, "v_proj_t", "v", hidden, kv_dim, block_n=32, block_k=512
            )
        else:
            q_gate = inp @ weights["q_proj"].t()
            k = inp @ weights["k_proj"].t()
            v = inp @ weights["v_proj"].t()
        q_gate = q_gate.view(tokens, heads, 2 * head_dim)
        q, gate = q_gate.chunk(2, dim=-1)
        gate = gate.reshape(tokens, q_dim)
        k = k.view(tokens, kv_heads, head_dim)
        v = v.view(tokens, kv_heads, head_dim)
        k_cache, v_cache = full_attention_kv_caches[layer_index]
        if should_use_triton_full_attention_fused_norm_rope_kv_write():
            q = triton_full_attention_norm_rope_kv_write(q, k, v, layer_index, weights)
        elif should_use_triton_full_attention_norm_rope():
            q = triton_full_attention_head_norm_rope(q, layer_index, weights, "q_norm", "q", heads)
            k = triton_full_attention_head_norm_rope(k, layer_index, weights, "k_norm", "k", kv_heads)
            write_full_attention_kv_cache(k_cache, v_cache, k, v)
        else:
            q = apply_rope(head_rmsnorm(q, weights["q_norm"]))
            k = apply_rope(head_rmsnorm(k, weights["k_norm"]))
            write_full_attention_kv_cache(k_cache, v_cache, k, v)
        cache_end = position_end if mode == "decode" or (mode == "prefill" and position_start > 0) else tokens
        model_layer = int(layer_weights[layer_index]["layer"])
        if (
            use_q8192_compound_provider()
            and q8192_compound_provider().component_enabled("ck_fmha")
            and model_layer < 39
        ):
            attn_out = q8192_compound_provider().launch_ck_fmha(
                q=q.contiguous(),
                k=k.contiguous(),
                value=v.contiguous(),
                model_layer=model_layer,
            ).to(dtype=dtype)
        elif use_q16_hybrid_attention(model_layer):
            if int(cache_end) != 16384:
                raise RuntimeError(f"q16 hybrid cache-end drift: {cache_end}")
            attn_out = q8192_compound_provider().launch_q16_hybrid_attention(
                q=q.contiguous(),
                k=k_cache[:cache_end].contiguous(),
                value=v_cache[:cache_end].contiguous(),
                model_layer=model_layer,
            )
        elif use_persistent_tilequeue_attention(model_layer):
            attn_out = q8192_compound_provider().launch_persistent_tilequeue_attention(
                q=q.contiguous(),
                k=k_cache[:cache_end].contiguous(),
                value=v_cache[:cache_end].contiguous(),
                model_layer=model_layer,
            )
        elif use_partial_persistent_tilequeue_attention(model_layer):
            attn_out = q8192_compound_provider().launch_partial_persistent_tilequeue_attention(
                q=q.contiguous(),
                k=k_cache[:cache_end].contiguous(),
                value=v_cache[:cache_end].contiguous(),
                model_layer=model_layer,
            )
        elif should_use_grouped_bmm_full_attention():
            q_grouped = q.view(kv_heads, heads // kv_heads, head_dim)
            k_grouped, v_grouped = grouped_full_attention_cache_views(k_cache, v_cache, cache_end)
            if full_attention_variant == "decode_grouped_bmm_bf16":
                scores = torch.matmul(q_grouped, k_grouped.transpose(-1, -2))
            else:
                scores = torch.matmul(q_grouped.float(), k_grouped.float().transpose(-1, -2))
            scores = scores * (1.0 / math.sqrt(head_dim))
            probs = torch.softmax(scores, dim=-1)
            if probs.dtype != dtype:
                probs = probs.to(dtype)
            attn_grouped = torch.matmul(probs, v_grouped)
            attn_out = attn_grouped.reshape(tokens, q_dim)
        else:
            k_grouped, v_grouped = grouped_full_attention_cache_views(k_cache, v_cache, cache_end)
            attn_mask = None
            if mode == "prefill" and position_start > 0:
                if causal_lower_right is not None and int(position_end) == int(cache_end):
                    attn_mask = causal_lower_right(int(tokens), int(cache_end))
                else:
                    query_positions = torch.arange(position_start, position_end, device=device)
                    key_positions = torch.arange(int(cache_end), device=device)
                    attn_mask = (
                        key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
                    ).view(1, 1, tokens, int(cache_end))
            attn = F.scaled_dot_product_attention(
                q.transpose(0, 1).unsqueeze(0),
                k_grouped.unsqueeze(0),
                v_grouped.unsqueeze(0),
                attn_mask=attn_mask,
                is_causal=(mode == "prefill" and attn_mask is None),
                enable_gqa=True,
            )
            attn_out = attn.squeeze(0).transpose(0, 1).reshape(tokens, q_dim)
        if should_use_triton_full_attention_fused_gate_o_proj():
            return triton_full_attention_gated_o_projection(attn_out, gate, layer_index, weights)
        attn_out = attn_out * torch.sigmoid(gate).to(dtype)
        if should_use_triton_full_attention_proj():
            return triton_full_attention_projection(attn_out, layer_index, weights, "o_proj_t", "o", q_dim, hidden)
        return attn_out @ weights["o_proj"].t()

    def should_use_retained_linear_attention_fast_path() -> bool:
        return (
            retained_attention_fast_path
            and mode == "decode"
            and tokens == 1
            and linear_attention_uses_vllm_auto_decode(linear_attention_variant)
            and linear_attention_uses_native_vllm_decode_state(linear_attention_variant)
            and should_use_triton_fused_linear_input_proj_conv()
            and should_use_triton_linear_gated_norm()
            and should_use_triton_linear_output_proj()
        )

    def should_use_retained_full_attention_fast_path() -> bool:
        return (
            retained_attention_fast_path
            and mode == "decode"
            and tokens == 1
            and should_use_triton_full_attention_fused_qkv()
            and should_use_triton_full_attention_norm_rope()
            and should_use_grouped_bmm_full_attention()
            and full_attention_variant == "decode_grouped_bmm_bf16"
        )

    def retained_linear_attention_fast(inp: Any, layer_index: int, weights: dict[str, Any]) -> Any:
        mixed_qkv, z, a, b = linear_input_projection(inp, layer_index, weights)
        q_raw, k_raw, value_raw = mixed_qkv.split([linear_key_dim, linear_key_dim, linear_value_dim], dim=-1)
        q = q_raw.view(tokens, linear_key_heads, linear_key_head_dim).unsqueeze(0).contiguous()
        k = k_raw.view(tokens, linear_key_heads, linear_key_head_dim).unsqueeze(0).contiguous()
        value = value_raw.view(tokens, linear_value_heads, linear_value_head_dim).unsqueeze(0).contiguous()
        if vllm_fused_recurrent_gdn_update is None:
            raise RuntimeError("vLLM recurrent GDN kernel was not loaded")
        core_attn_out, final_state = vllm_fused_recurrent_gdn_update(
            A_log=weights["linear_A_log"].contiguous(),
            a=a.contiguous(),
            b=b.contiguous(),
            dt_bias=weights["linear_dt_bias"].contiguous(),
            q=q,
            k=k,
            v=value,
            initial_state=vllm_linear_initial_state(layer_index),
            inplace_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )
        if linear_attention_uses_native_vllm_decode_state_refswap(linear_attention_variant):
            linear_attention_ssm_states_vllm[layer_index] = final_state
        else:
            linear_attention_ssm_states_vllm[layer_index].copy_(final_state)
        gated_out = triton_linear_gated_norm(core_attn_out, z, layer_index, weights)
        return triton_linear_output_projection(gated_out, layer_index, weights)

    def retained_full_attention_fast(inp: Any, layer_index: int, weights: dict[str, Any]) -> Any:
        qkv = triton_full_attention_qkv_projection(inp, layer_index, weights)
        q_gate, k, v = qkv.split([2 * q_dim, kv_dim, kv_dim], dim=-1)
        q_gate = q_gate.view(tokens, heads, 2 * head_dim)
        q, gate = q_gate.chunk(2, dim=-1)
        gate = gate.reshape(tokens, q_dim)
        k = k.view(tokens, kv_heads, head_dim)
        v = v.view(tokens, kv_heads, head_dim)
        k_cache, v_cache = full_attention_kv_caches[layer_index]
        if should_use_triton_full_attention_fused_norm_rope_kv_write():
            q = triton_full_attention_norm_rope_kv_write(q, k, v, layer_index, weights)
        else:
            q = triton_full_attention_head_norm_rope(q, layer_index, weights, "q_norm", "q", heads)
            k = triton_full_attention_head_norm_rope(k, layer_index, weights, "k_norm", "k", kv_heads)
            write_full_attention_kv_cache(k_cache, v_cache, k, v)
        cache_end = position_end
        q_grouped = q.view(kv_heads, heads // kv_heads, head_dim)
        k_grouped, v_grouped = grouped_full_attention_cache_views(k_cache, v_cache, cache_end)
        scores = torch.matmul(q_grouped, k_grouped.transpose(-1, -2))
        scores = scores * (1.0 / math.sqrt(head_dim))
        probs = torch.softmax(scores, dim=-1)
        if probs.dtype != dtype:
            probs = probs.to(dtype)
        attn_out = torch.matmul(probs, v_grouped).reshape(tokens, q_dim)
        if should_use_triton_full_attention_fused_gate_o_proj():
            return triton_full_attention_gated_o_projection(attn_out, gate, layer_index, weights)
        attn_out = attn_out * torch.sigmoid(gate).to(dtype)
        return triton_full_attention_projection(attn_out, layer_index, weights, "o_proj_t", "o", q_dim, hidden)

    def attention(inp: Any, layer_index: int, weights: dict[str, Any]) -> Any:
        if full_attention_enabled(attention_mode) and layer_weights[layer_index]["layer_type"] == "full_attention":
            if should_use_retained_full_attention_fast_path():
                return retained_full_attention_fast(inp, layer_index, weights)
            return full_attention(inp, layer_index, weights)
        if linear_attention_enabled(attention_mode) and layer_weights[layer_index]["layer_type"] == "linear_attention":
            if should_use_retained_linear_attention_fast_path():
                return retained_linear_attention_fast(inp, layer_index, weights)
            return linear_attention(inp, layer_index, weights)
        return attention_stub_buffers[layer_index]

    def profiled_full_attention(
        inp: Any,
        layer_index: int,
        weights: dict[str, Any],
        records: list[dict[str, Any]],
    ) -> Any:
        entry = layer_weights[layer_index]
        profile = lambda stage, fn: diagnostic_timed(
            layer=entry["layer"],
            layer_type=entry["layer_type"],
            attention="full_attention",
            stage=stage,
            fn=fn,
            records=records,
        )
        if should_use_triton_full_attention_fused_qkv():
            qkv = profile(
                "qkv_proj_triton_matvec",
                lambda: triton_full_attention_qkv_projection(inp, layer_index, weights),
            )
            q_gate, k, v = profile(
                "qkv_proj_split",
                lambda: qkv.split([2 * q_dim, kv_dim, kv_dim], dim=-1),
            )
        elif should_use_triton_full_attention_proj():
            q_gate = profile(
                "q_proj_gate_triton_matvec",
                lambda: triton_full_attention_projection(
                    inp, layer_index, weights, "q_proj_t", "q_gate", hidden, 2 * q_dim
                ),
            )
            k = profile(
                "k_proj_triton_matvec",
                lambda: triton_full_attention_projection(inp, layer_index, weights, "k_proj_t", "k", hidden, kv_dim),
            )
            v = profile(
                "v_proj_triton_matvec",
                lambda: triton_full_attention_projection(
                    inp, layer_index, weights, "v_proj_t", "v", hidden, kv_dim, block_n=32, block_k=512
                ),
            )
        else:
            q_gate = profile("q_proj_gate", lambda: inp @ weights["q_proj"].t())
            k = profile("k_proj", lambda: inp @ weights["k_proj"].t())
            v = profile("v_proj", lambda: inp @ weights["v_proj"].t())
        q, gate = q_gate.chunk(2, dim=-1)
        q = profile("q_view", lambda: q.view(tokens, heads, head_dim))
        k = profile("k_view", lambda: k.view(tokens, kv_heads, head_dim))
        v = profile("v_view", lambda: v.view(tokens, kv_heads, head_dim))
        k_cache, v_cache = full_attention_kv_caches[layer_index]
        if should_use_triton_full_attention_fused_norm_rope_kv_write():
            q = profile(
                "qk_head_norm_rope_kv_write_triton_fused",
                lambda: triton_full_attention_norm_rope_kv_write(q, k, v, layer_index, weights),
            )
        elif should_use_triton_full_attention_norm_rope():
            q = profile(
                "q_head_norm_rope_triton",
                lambda: triton_full_attention_head_norm_rope(q, layer_index, weights, "q_norm", "q", heads),
            )
            k = profile(
                "k_head_norm_rope_triton",
                lambda: triton_full_attention_head_norm_rope(k, layer_index, weights, "k_norm", "k", kv_heads),
            )
            def write_kv_cache() -> None:
                write_full_attention_kv_cache(k_cache, v_cache, k, v)

            profile("kv_cache_write", write_kv_cache)
        else:
            q = profile("q_head_rmsnorm", lambda: head_rmsnorm(q, weights["q_norm"]))
            k = profile("k_head_rmsnorm", lambda: head_rmsnorm(k, weights["k_norm"]))
            q = profile("q_rope", lambda: apply_rope(q))
            k = profile("k_rope", lambda: apply_rope(k))
            def write_kv_cache() -> None:
                write_full_attention_kv_cache(k_cache, v_cache, k, v)

            profile("kv_cache_write", write_kv_cache)
        cache_end = position_end if mode == "decode" else tokens
        if should_use_grouped_bmm_full_attention():
            q_grouped = profile("decode_group_q", lambda: q.view(kv_heads, heads // kv_heads, head_dim))
            k_grouped = profile("decode_group_k", lambda: grouped_full_attention_cache_views(k_cache, v_cache, cache_end)[0])
            v_grouped = profile("decode_group_v", lambda: grouped_full_attention_cache_views(k_cache, v_cache, cache_end)[1])
            if full_attention_variant == "decode_grouped_bmm_bf16":
                scores = profile(
                    "decode_grouped_qk_bmm_bf16",
                    lambda: torch.matmul(q_grouped, k_grouped.transpose(-1, -2))
                    * (1.0 / math.sqrt(head_dim)),
                )
            else:
                scores = profile(
                    "decode_grouped_qk_bmm",
                    lambda: torch.matmul(q_grouped.float(), k_grouped.float().transpose(-1, -2))
                    * (1.0 / math.sqrt(head_dim)),
                )
            probs = profile("decode_grouped_softmax", lambda: torch.softmax(scores, dim=-1).to(dtype))
            attn_grouped = profile("decode_grouped_pv_bmm", lambda: torch.matmul(probs, v_grouped))
            attn_out = profile("decode_grouped_reshape", lambda: attn_grouped.reshape(tokens, q_dim))
        else:
            k_grouped, v_grouped = grouped_full_attention_cache_views(k_cache, v_cache, cache_end)
            attn = profile(
                "sdpa",
                lambda: F.scaled_dot_product_attention(
                    q.transpose(0, 1).unsqueeze(0),
                    k_grouped.unsqueeze(0),
                    v_grouped.unsqueeze(0),
                    is_causal=(mode == "prefill"),
                    enable_gqa=True,
                ),
            )
            attn_out = profile("reshape_attention_output", lambda: attn.squeeze(0).transpose(0, 1).reshape(tokens, q_dim))
        if should_use_triton_full_attention_fused_gate_o_proj():
            return profile(
                "output_gate_o_proj_triton_fused",
                lambda: triton_full_attention_gated_o_projection(attn_out, gate, layer_index, weights),
            )
        gated = profile("output_gate", lambda: attn_out * torch.sigmoid(gate).to(dtype))
        if should_use_triton_full_attention_proj():
            return profile(
                "o_proj_triton_matvec",
                lambda: triton_full_attention_projection(gated, layer_index, weights, "o_proj_t", "o", q_dim, hidden),
            )
        return profile("o_proj", lambda: gated @ weights["o_proj"].t())

    def profiled_linear_attention(
        inp: Any,
        layer_index: int,
        weights: dict[str, Any],
        records: list[dict[str, Any]],
        chunk_gdn_internal_records: list[dict[str, Any]],
    ) -> Any:
        entry = layer_weights[layer_index]
        profile = (
            (
                lambda stage, fn: diagnostic_timed(
                    layer=entry["layer"],
                    layer_type=entry["layer_type"],
                    attention="linear_attention",
                    stage=stage,
                    fn=fn,
                    records=records,
                )
            )
            if attention_substage_timing
            else (lambda stage, fn: fn())
        )
        profile_chunk_gdn = (
            lambda stage, fn: diagnostic_timed(
                layer=entry["layer"],
                layer_type=entry["layer_type"],
                attention="linear_attention_chunk_gdn",
                stage=stage,
                fn=fn,
                records=chunk_gdn_internal_records,
            )
            if linear_attention_chunk_gdn_internal_timing
            else None
        )
        prepacked_qkv: tuple[Any, Any, Any] | None = None
        if should_use_triton_fused_linear_input_proj_conv_qkv_layout():
            q, k, value, z, a, b = profile(
                "input_proj_fused_conv_qkv_triton",
                lambda: triton_fused_linear_input_projection_conv_qkv_layout(inp, layer_index, weights),
            )
            prepacked_qkv = (q, k, value)
            conv_qkv = None
        elif should_use_fused_linear_input_proj():
            input_proj_weight = (
                weights["linear_input_proj_fused_t"]
                if linear_attention_input_proj_variant
                in {
                    "decode_fused_t",
                    "decode_fused_t_triton",
                    "decode_fused_t_conv_triton",
                    "decode_fused_t_conv_qkv_triton",
                    "prefill_fused_t_decode_fused_t_conv_triton",
                    "prefill_fused_t_decode_fused_t_conv_qkv_triton",
                }
                else weights["linear_input_proj_fused"].t()
            )
            if should_use_triton_fused_linear_input_proj_conv():
                input_proj_stage = "input_proj_fused_conv_triton"
            elif should_use_triton_linear_input_proj():
                input_proj_stage = "input_proj_fused_triton_matvec"
            elif (
                linear_attention_input_proj_variant
                in {
                    "prefill_fused_t_decode_fused_t_conv_triton",
                    "prefill_fused_t_decode_fused_t_conv_qkv_triton",
                }
                and mode == "prefill"
            ):
                input_proj_stage = "input_proj_prefill_fused_t"
            else:
                input_proj_stage = "input_proj_fused"
            mixed_qkv, z, a, b = profile(
                input_proj_stage,
                lambda: split_fused_linear_input_proj(
                    (
                        triton_fused_linear_input_projection_conv(inp, layer_index, weights)
                        if should_use_triton_fused_linear_input_proj_conv()
                        else (
                            triton_linear_input_projection(inp, layer_index, weights)
                            if should_use_triton_linear_input_proj()
                            else inp @ input_proj_weight
                        )
                    ),
                    weights,
                ),
            )
            conv_qkv = (
                None
                if use_prefill_conv_post_prep_fusion()
                else (
                    mixed_qkv
                    if should_use_triton_fused_linear_input_proj_conv()
                    else profile("causal_conv", lambda: linear_causal_conv(mixed_qkv, layer_index, weights))
                )
            )
        else:
            mixed_qkv = profile("in_proj_qkv", lambda: inp @ weights["linear_in_proj_qkv"].t())
            z = profile(
                "in_proj_z",
                lambda: (inp @ weights["linear_in_proj_z"].t()).view(tokens, linear_value_heads, linear_value_head_dim),
            )
            a = profile("in_proj_a", lambda: inp @ weights["linear_in_proj_a"].t())
            b = profile("in_proj_b", lambda: inp @ weights["linear_in_proj_b"].t())
            conv_qkv = profile("causal_conv", lambda: linear_causal_conv(mixed_qkv, layer_index, weights))
        if should_use_packed_linear_gdn():
            if vllm_fused_recurrent_gdn_packed_decode is None:
                raise RuntimeError("vLLM packed recurrent GDN kernel was not loaded")
            if linear_attention_uses_packed_state_refswap(linear_attention_variant):
                packed_state = linear_attention_packed_initial_ssm_states[layer_index]
            else:
                packed_state = linear_attention_packed_ssm_states[layer_index]
                profile(
                    "packed_ssm_state_reset",
                    lambda: packed_state.copy_(linear_attention_packed_initial_ssm_states[layer_index]),
                )
            core_attn_out, final_state = profile(
                (
                    "packed_recurrent_gdn_update_refswap"
                    if linear_attention_uses_packed_state_refswap(linear_attention_variant)
                    else "packed_recurrent_gdn_update"
                ),
                lambda: vllm_fused_recurrent_gdn_packed_decode(
                    mixed_qkv=conv_qkv.contiguous(),
                    a=a.contiguous(),
                    b=b.contiguous(),
                    A_log=weights["linear_A_log"].contiguous(),
                    dt_bias=weights["linear_dt_bias"].contiguous(),
                    scale=linear_key_head_dim**-0.5,
                    initial_state=packed_state,
                    out=linear_attention_packed_outputs[layer_index],
                    ssm_state_indices=linear_attention_packed_state_indices[layer_index],
                    use_qk_l2norm_in_kernel=True,
                ),
            )
            if not linear_attention_uses_packed_state_refswap(linear_attention_variant):
                profile(
                    "packed_ssm_state_write",
                    lambda: linear_attention_ssm_states[layer_index].copy_(final_state[1].transpose(-1, -2)),
                )
        elif linear_attention_uses_vllm_auto_decode(linear_attention_variant) and mode == "decode":
            if vllm_fused_recurrent_gdn_update is None:
                raise RuntimeError("vLLM recurrent GDN kernel was not loaded")

            def split_decode_qkv() -> tuple[Any, Any, Any]:
                if conv_qkv is None:
                    raise RuntimeError("linear-attention decode qkv source is missing")
                q_raw, k_raw, value = conv_qkv.split([linear_key_dim, linear_key_dim, linear_value_dim], dim=-1)
                return (
                    q_raw.view(tokens, linear_key_heads, linear_key_head_dim).unsqueeze(0).contiguous(),
                    k_raw.view(tokens, linear_key_heads, linear_key_head_dim).unsqueeze(0).contiguous(),
                    value.view(tokens, linear_value_heads, linear_value_head_dim).unsqueeze(0).contiguous(),
                )

            if prepacked_qkv is None:
                q, k, value = profile("split_decode_qkv", split_decode_qkv)
            else:
                q, k, value = prepacked_qkv
            core_attn_out, final_state = profile(
                "recurrent_gdn_update",
                lambda: vllm_fused_recurrent_gdn_update(
                    A_log=weights["linear_A_log"].contiguous(),
                    a=a.contiguous(),
                    b=b.contiguous(),
                    dt_bias=weights["linear_dt_bias"].contiguous(),
                    q=q,
                    k=k,
                    v=value,
                    initial_state=vllm_linear_initial_state(layer_index),
                    inplace_final_state=False,
                    use_qk_l2norm_in_kernel=True,
                ),
            )
            if linear_attention_uses_native_vllm_decode_state(linear_attention_variant):
                if linear_attention_uses_native_vllm_decode_state_refswap(linear_attention_variant):
                    profile(
                        "ssm_vllm_state_refswap",
                        lambda: linear_attention_ssm_states_vllm.__setitem__(layer_index, final_state),
                    )
                else:
                    profile(
                        "ssm_vllm_state_write",
                        lambda: linear_attention_ssm_states_vllm[layer_index].copy_(final_state),
                    )
            else:
                profile(
                    "ssm_state_write",
                    lambda: linear_attention_ssm_states[layer_index].copy_(final_state.squeeze(0).transpose(-1, -2)),
                )
        elif linear_attention_uses_vllm_fla(linear_attention_variant):
            if vllm_fused_post_conv_prep is None or vllm_chunk_gated_delta_rule is None:
                raise RuntimeError("vLLM FLA kernels were not loaded")
            if use_prefill_conv_post_prep_fusion():
                q, k, value, g, beta = profile(
                    "conv_post_prep_fusion",
                    lambda: run_triton_prefill_conv_post_prep(mixed_qkv, a, b, layer_index, weights),
                )
            else:
                q, k, value, g, beta = profile(
                    "post_conv_prep",
                    lambda: run_vllm_fused_post_conv_prep(conv_qkv, a, b, weights),
                )
            chunk_size = linear_attention_chunk_size(linear_attention_variant)
            initial_state = profile("prefill_initial_state", lambda: prefill_chunk_initial_state(layer_index))
            if chunk_size is None:
                core_attn_out, final_state = profile(
                    "chunk_gated_delta_rule",
                    lambda: vllm_chunk_gated_delta_rule(
                        q=q.unsqueeze(0),
                        k=k.unsqueeze(0),
                        v=value.unsqueeze(0),
                        g=g.unsqueeze(0),
                        beta=beta.unsqueeze(0),
                        initial_state=initial_state,
                        output_final_state=True,
                        use_qk_l2norm_in_kernel=False,
                    ),
                )
            else:
                if profile_chunk_gdn is None:
                    core_attn_out, final_state = profile(
                        "chunk_gated_delta_rule",
                        lambda: tuned_chunk_gated_delta_rule(
                            q=q.unsqueeze(0),
                            k=k.unsqueeze(0),
                            value=value.unsqueeze(0),
                            g=g.unsqueeze(0),
                            beta=beta.unsqueeze(0),
                            initial_state=initial_state,
                            output_final_state=True,
                            chunk_size=chunk_size,
                            use_qk_l2norm_in_kernel=False,
                        ),
                    )
                else:
                    core_attn_out, final_state = tuned_chunk_gated_delta_rule(
                        q=q.unsqueeze(0),
                        k=k.unsqueeze(0),
                        value=value.unsqueeze(0),
                        g=g.unsqueeze(0),
                        beta=beta.unsqueeze(0),
                        initial_state=initial_state,
                        output_final_state=True,
                        chunk_size=chunk_size,
                        use_qk_l2norm_in_kernel=False,
                        profile_stage=profile_chunk_gdn,
                    )
            profile(
                "ssm_vllm_state_handoff" if use_prefill_vllm_state_handoff() else "ssm_state_write",
                lambda: store_prefill_chunk_final_state(layer_index, final_state),
            )
        else:
            q, k, value = profile("split_torch_qkv", lambda: conv_qkv.split([linear_key_dim, linear_key_dim, linear_value_dim], dim=-1))
            q = profile("reshape_torch_q", lambda: q.view(tokens, linear_key_heads, linear_key_head_dim))
            k = profile("reshape_torch_k", lambda: k.view(tokens, linear_key_heads, linear_key_head_dim))
            value = profile("reshape_torch_v", lambda: value.view(tokens, linear_value_heads, linear_value_head_dim))
            repeat_factor = linear_value_heads // linear_key_heads
            if repeat_factor > 1:
                q = profile("repeat_torch_q", lambda: q.repeat_interleave(repeat_factor, dim=1))
                k = profile("repeat_torch_k", lambda: k.repeat_interleave(repeat_factor, dim=1))
            beta = profile("torch_beta", lambda: torch.sigmoid(b).view(1, tokens, linear_value_heads))
            g = profile(
                "torch_decay",
                lambda: (
                    -weights["linear_A_log"].float().exp().unsqueeze(0)
                    * F.softplus(a.float() + weights["linear_dt_bias"].float().unsqueeze(0))
                ).view(1, tokens, linear_value_heads),
            )
            initial_state = linear_attention_initial_ssm_states[layer_index].unsqueeze(0)
            q = q.unsqueeze(0)
            k = k.unsqueeze(0)
            value = value.unsqueeze(0)
            if mode == "decode":
                core_attn_out, final_state = profile(
                    "torch_recurrent_gated_delta_rule",
                    lambda: torch_recurrent_gated_delta_rule(q, k, value, g, beta, initial_state),
                )
            else:
                core_attn_out, final_state = profile(
                    "torch_chunk_gated_delta_rule",
                    lambda: torch_chunk_gated_delta_rule(q, k, value, g, beta, initial_state),
                )
            profile("ssm_state_write", lambda: linear_attention_ssm_states[layer_index].copy_(final_state.squeeze(0)))

        if native_moe_consumer_prefill_official_linear_surface and mode == "prefill":
            try:
                from vllm.model_executor.layers.fla.ops.layernorm_guard import rmsnorm_fn
            except Exception as exc:
                raise RuntimeError("formal prefill alignment requires vLLM rmsnorm_fn") from exc
            gated_out = profile(
                "gated_norm_z_vllm_official",
                lambda: rmsnorm_fn(
                    core_attn_out.to(dtype=torch.bfloat16),
                    weights["linear_norm"].to(device=device, dtype=torch.bfloat16),
                    None,
                    z=z.to(dtype=torch.bfloat16),
                    eps=1e-6,
                    group_size=None,
                    norm_before_gate=True,
                    activation="swish",
                ).to(dtype),
            )
        elif should_use_triton_linear_gated_norm():
            gated_out = profile(
                "gated_norm_z_triton",
                lambda: triton_linear_gated_norm(core_attn_out, z, layer_index, weights),
            )
        else:
            core_flat = profile("core_flatten", lambda: core_attn_out.reshape(-1, linear_value_head_dim))
            z_flat = profile("z_flatten", lambda: z.reshape(-1, linear_value_head_dim))
            variance = profile("gated_norm_variance", lambda: core_flat.float().pow(2).mean(dim=-1, keepdim=True))
            gated = profile(
                "gated_norm_scale",
                lambda: (core_flat.float() * torch.rsqrt(variance + 1e-6) * weights["linear_norm"].float()).to(dtype),
            )
            gated = profile("z_gate", lambda: (gated.float() * F.silu(z_flat.float())).to(dtype))
            gated_out = profile("out_proj_input_view", lambda: gated.view(tokens, linear_value_dim))
        if should_use_triton_linear_output_proj():
            return profile(
                "out_proj_triton_matvec",
                lambda: triton_linear_output_projection(gated_out, layer_index, weights),
            )
        return profile("out_proj", lambda: gated_out @ weights["linear_out_proj"].t())

    def profiled_full_attention_clusters(
        inp: Any,
        layer_index: int,
        weights: dict[str, Any],
        records: list[dict[str, Any]],
    ) -> Any:
        entry = layer_weights[layer_index]
        profile = lambda stage, fn: diagnostic_timed(
            layer=entry["layer"],
            layer_type=entry["layer_type"],
            attention="full_attention",
            stage=stage,
            fn=fn,
            records=records,
        )
        parent_out = profile("parent_replay", lambda: full_attention(inp, layer_index, weights))

        def qkv_projection_layout() -> tuple[Any, Any, Any, Any]:
            if should_use_triton_full_attention_fused_qkv():
                qkv = triton_full_attention_qkv_projection(inp, layer_index, weights)
                q_gate, k_raw, v_raw = qkv.split([2 * q_dim, kv_dim, kv_dim], dim=-1)
            elif should_use_triton_full_attention_proj():
                q_gate = triton_full_attention_projection(
                    inp, layer_index, weights, "q_proj_t", "q_gate", hidden, 2 * q_dim
                )
                k_raw = triton_full_attention_projection(inp, layer_index, weights, "k_proj_t", "k", hidden, kv_dim)
                v_raw = triton_full_attention_projection(
                    inp, layer_index, weights, "v_proj_t", "v", hidden, kv_dim, block_n=32, block_k=512
                )
            else:
                q_gate = inp @ weights["q_proj"].t()
                k_raw = inp @ weights["k_proj"].t()
                v_raw = inp @ weights["v_proj"].t()
            q_gate = q_gate.view(tokens, heads, 2 * head_dim)
            q_raw, gate_raw = q_gate.chunk(2, dim=-1)
            return (
                q_raw,
                gate_raw.reshape(tokens, q_dim),
                k_raw.view(tokens, kv_heads, head_dim),
                v_raw.view(tokens, kv_heads, head_dim),
            )

        q, gate, k, v = profile("qkv_projection_layout", qkv_projection_layout)

        def norm_rope_kv_write() -> tuple[Any, Any]:
            if should_use_triton_full_attention_fused_norm_rope_kv_write():
                q_normed = triton_full_attention_norm_rope_kv_write(q, k, v, layer_index, weights)
                return q_normed, None
            if should_use_triton_full_attention_norm_rope():
                q_normed = triton_full_attention_head_norm_rope(q, layer_index, weights, "q_norm", "q", heads)
                k_normed = triton_full_attention_head_norm_rope(k, layer_index, weights, "k_norm", "k", kv_heads)
            else:
                q_normed = apply_rope(head_rmsnorm(q, weights["q_norm"]))
                k_normed = apply_rope(head_rmsnorm(k, weights["k_norm"]))
            k_cache, v_cache = full_attention_kv_caches[layer_index]
            write_full_attention_kv_cache(k_cache, v_cache, k_normed, v)
            return q_normed, k_normed

        q, _ = profile("norm_rope_kv_write", norm_rope_kv_write)

        def grouped_attention_cluster() -> Any:
            k_cache, v_cache = full_attention_kv_caches[layer_index]
            cache_end = position_end if mode == "decode" else tokens
            if should_use_grouped_bmm_full_attention():
                q_grouped = q.view(kv_heads, heads // kv_heads, head_dim)
                k_grouped, v_grouped = grouped_full_attention_cache_views(k_cache, v_cache, cache_end)
                if full_attention_variant == "decode_grouped_bmm_bf16":
                    scores = torch.matmul(q_grouped, k_grouped.transpose(-1, -2))
                else:
                    scores = torch.matmul(q_grouped.float(), k_grouped.float().transpose(-1, -2))
                scores = scores * (1.0 / math.sqrt(head_dim))
                probs = torch.softmax(scores, dim=-1)
                if probs.dtype != dtype:
                    probs = probs.to(dtype)
                return torch.matmul(probs, v_grouped).reshape(tokens, q_dim)
            k_grouped, v_grouped = grouped_full_attention_cache_views(k_cache, v_cache, cache_end)
            attn = F.scaled_dot_product_attention(
                q.transpose(0, 1).unsqueeze(0),
                k_grouped.unsqueeze(0),
                v_grouped.unsqueeze(0),
                is_causal=(mode == "prefill"),
                enable_gqa=True,
            )
            return attn.squeeze(0).transpose(0, 1).reshape(tokens, q_dim)

        attn_out = profile("grouped_attention", grouped_attention_cluster)

        def output_gate_o_proj() -> Any:
            if should_use_triton_full_attention_fused_gate_o_proj():
                return triton_full_attention_gated_o_projection(attn_out, gate, layer_index, weights)
            gated = attn_out * torch.sigmoid(gate).to(dtype)
            if should_use_triton_full_attention_proj():
                return triton_full_attention_projection(gated, layer_index, weights, "o_proj_t", "o", q_dim, hidden)
            return gated @ weights["o_proj"].t()

        output_stage = (
            "output_gate_o_proj_triton_fused"
            if should_use_triton_full_attention_fused_gate_o_proj()
            else "output_gate_o_proj"
        )
        profile(output_stage, output_gate_o_proj)
        return parent_out

    def profiled_linear_attention_clusters(
        inp: Any,
        layer_index: int,
        weights: dict[str, Any],
        records: list[dict[str, Any]],
    ) -> Any:
        entry = layer_weights[layer_index]
        profile = lambda stage, fn: diagnostic_timed(
            layer=entry["layer"],
            layer_type=entry["layer_type"],
            attention="linear_attention",
            stage=stage,
            fn=fn,
            records=records,
        )
        parent_out = profile("parent_replay", lambda: linear_attention(inp, layer_index, weights))
        if not (linear_attention_uses_vllm_auto_decode(linear_attention_variant) and mode == "decode"):
            return parent_out

        def input_proj_conv_qkv_layout() -> tuple[Any, Any, Any, Any, Any, Any]:
            if should_use_triton_fused_linear_input_proj_conv_qkv_layout():
                return triton_fused_linear_input_projection_conv_qkv_layout(inp, layer_index, weights)
            mixed_qkv, z, a, b = linear_input_projection(inp, layer_index, weights)
            conv_qkv = (
                mixed_qkv
                if should_use_triton_fused_linear_input_proj_conv()
                else linear_causal_conv(mixed_qkv, layer_index, weights)
            )
            q_raw, k_raw, value = conv_qkv.split([linear_key_dim, linear_key_dim, linear_value_dim], dim=-1)
            return (
                q_raw.view(tokens, linear_key_heads, linear_key_head_dim).unsqueeze(0).contiguous(),
                k_raw.view(tokens, linear_key_heads, linear_key_head_dim).unsqueeze(0).contiguous(),
                value.view(tokens, linear_value_heads, linear_value_head_dim).unsqueeze(0).contiguous(),
                z,
                a,
                b,
            )

        q, k, value, z, a, b = profile("input_proj_conv_qkv_layout", input_proj_conv_qkv_layout)

        def recurrent_gdn_state() -> Any:
            if vllm_fused_recurrent_gdn_update is None:
                raise RuntimeError("vLLM recurrent GDN kernel was not loaded")
            core_attn_out, final_state = vllm_fused_recurrent_gdn_update(
                A_log=weights["linear_A_log"].contiguous(),
                a=a.contiguous(),
                b=b.contiguous(),
                dt_bias=weights["linear_dt_bias"].contiguous(),
                q=q,
                k=k,
                v=value,
                initial_state=vllm_linear_initial_state(layer_index),
                inplace_final_state=False,
                use_qk_l2norm_in_kernel=True,
            )
            if linear_attention_uses_native_vllm_decode_state(linear_attention_variant):
                if linear_attention_uses_native_vllm_decode_state_refswap(linear_attention_variant):
                    linear_attention_ssm_states_vllm[layer_index] = final_state
                else:
                    linear_attention_ssm_states_vllm[layer_index].copy_(final_state)
            else:
                linear_attention_ssm_states[layer_index].copy_(final_state.squeeze(0).transpose(-1, -2))
            return core_attn_out

        core_attn_out = profile("recurrent_gdn_state", recurrent_gdn_state)

        def gated_norm_out_proj() -> Any:
            if should_use_triton_linear_gated_norm():
                gated_out = triton_linear_gated_norm(core_attn_out, z, layer_index, weights)
            else:
                core_flat = core_attn_out.reshape(-1, linear_value_head_dim)
                z_flat = z.reshape(-1, linear_value_head_dim)
                variance = core_flat.float().pow(2).mean(dim=-1, keepdim=True)
                gated = (
                    core_flat.float()
                    * torch.rsqrt(variance + 1e-6)
                    * weights["linear_norm"].float()
                ).to(dtype)
                gated = (gated.float() * F.silu(z_flat.float())).to(dtype)
                gated_out = gated.view(tokens, linear_value_dim)
            if should_use_triton_linear_output_proj():
                return triton_linear_output_projection(gated_out, layer_index, weights)
            return gated_out @ weights["linear_out_proj"].t()

        profile("gated_norm_out_proj", gated_norm_out_proj)
        return parent_out

    def append_attention_event_records(
        *,
        entry: dict[str, Any],
        attention: str,
        event_pairs: dict[str, list[tuple[Any, Any]]],
        records: list[dict[str, Any]],
    ) -> None:
        for stage, pairs in event_pairs.items():
            if not pairs:
                continue
            elapsed_ms = sum(float(start.elapsed_time(end)) for start, end in pairs)
            records.append(
                {
                    "layer": entry["layer"],
                    "layer_type": entry["layer_type"],
                    "attention": attention,
                    "stage": stage,
                    "name": f"layer{entry['layer']}_{attention}_{stage}",
                    "elapsed_ms": elapsed_ms,
                    "iters": len(pairs),
                    "ms_per_iter": elapsed_ms / len(pairs),
                    "timing_source": "cuda_event",
                }
            )

    def profiled_full_attention_events(
        inp: Any,
        layer_index: int,
        weights: dict[str, Any],
        records: list[dict[str, Any]],
    ) -> Any:
        if not (device.startswith("cuda") and torch.cuda.is_available()):
            return full_attention(inp, layer_index, weights)
        entry = layer_weights[layer_index]
        event_pairs: dict[str, list[tuple[Any, Any]]] = {}

        def once(record_events: bool) -> Any:
            def stage(name: str, fn: Callable[[], Any]) -> Any:
                if not record_events:
                    return fn()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record(torch.cuda.current_stream())
                out = fn()
                end_event.record(torch.cuda.current_stream())
                event_pairs.setdefault(name, []).append((start_event, end_event))
                return out

            total_start = torch.cuda.Event(enable_timing=True) if record_events else None
            total_end = torch.cuda.Event(enable_timing=True) if record_events else None
            if record_events and total_start is not None:
                total_start.record(torch.cuda.current_stream())

            def qkv_projection_layout() -> tuple[Any, Any, Any, Any]:
                if should_use_triton_full_attention_fused_qkv():
                    qkv = triton_full_attention_qkv_projection(inp, layer_index, weights)
                    q_gate, k_raw, v_raw = qkv.split([2 * q_dim, kv_dim, kv_dim], dim=-1)
                elif should_use_triton_full_attention_proj():
                    q_gate = triton_full_attention_projection(
                        inp, layer_index, weights, "q_proj_t", "q_gate", hidden, 2 * q_dim
                    )
                    k_raw = triton_full_attention_projection(
                        inp, layer_index, weights, "k_proj_t", "k", hidden, kv_dim
                    )
                    v_raw = triton_full_attention_projection(
                        inp, layer_index, weights, "v_proj_t", "v", hidden, kv_dim, block_n=32, block_k=512
                    )
                else:
                    q_gate = inp @ weights["q_proj"].t()
                    k_raw = inp @ weights["k_proj"].t()
                    v_raw = inp @ weights["v_proj"].t()
                q_gate_view = q_gate.view(tokens, heads, 2 * head_dim)
                q_raw, gate_raw = q_gate_view.chunk(2, dim=-1)
                return (
                    q_raw,
                    gate_raw.reshape(tokens, q_dim),
                    k_raw.view(tokens, kv_heads, head_dim),
                    v_raw.view(tokens, kv_heads, head_dim),
                )

            q, gate, k, v = stage("qkv_projection_layout", qkv_projection_layout)

            def norm_rope_kv_write() -> Any:
                if should_use_triton_full_attention_fused_norm_rope_kv_write():
                    return triton_full_attention_norm_rope_kv_write(q, k, v, layer_index, weights)
                if should_use_triton_full_attention_norm_rope():
                    q_normed = triton_full_attention_head_norm_rope(q, layer_index, weights, "q_norm", "q", heads)
                    k_normed = triton_full_attention_head_norm_rope(k, layer_index, weights, "k_norm", "k", kv_heads)
                else:
                    q_normed = apply_rope(head_rmsnorm(q, weights["q_norm"]))
                    k_normed = apply_rope(head_rmsnorm(k, weights["k_norm"]))
                k_cache, v_cache = full_attention_kv_caches[layer_index]
                write_full_attention_kv_cache(k_cache, v_cache, k_normed, v)
                return q_normed

            q = stage("norm_rope_kv_write", norm_rope_kv_write)

            def grouped_attention_cluster() -> Any:
                k_cache, v_cache = full_attention_kv_caches[layer_index]
                cache_end = position_end if mode == "decode" else tokens
                if should_use_grouped_bmm_full_attention():
                    q_grouped = q.view(kv_heads, heads // kv_heads, head_dim)
                    k_grouped, v_grouped = grouped_full_attention_cache_views(k_cache, v_cache, cache_end)
                    if full_attention_variant == "decode_grouped_bmm_bf16":
                        scores = torch.matmul(q_grouped, k_grouped.transpose(-1, -2))
                    else:
                        scores = torch.matmul(q_grouped.float(), k_grouped.float().transpose(-1, -2))
                    scores = scores * (1.0 / math.sqrt(head_dim))
                    probs = torch.softmax(scores, dim=-1)
                    if probs.dtype != dtype:
                        probs = probs.to(dtype)
                    return torch.matmul(probs, v_grouped).reshape(tokens, q_dim)
                k_grouped, v_grouped = grouped_full_attention_cache_views(k_cache, v_cache, cache_end)
                attn = F.scaled_dot_product_attention(
                    q.transpose(0, 1).unsqueeze(0),
                    k_grouped.unsqueeze(0),
                    v_grouped.unsqueeze(0),
                    is_causal=(mode == "prefill"),
                    enable_gqa=True,
                )
                return attn.squeeze(0).transpose(0, 1).reshape(tokens, q_dim)

            attn_out = stage("grouped_attention", grouped_attention_cluster)

            def output_gate_o_proj() -> Any:
                if should_use_triton_full_attention_fused_gate_o_proj():
                    return triton_full_attention_gated_o_projection(attn_out, gate, layer_index, weights)
                gated = attn_out * torch.sigmoid(gate).to(dtype)
                if should_use_triton_full_attention_proj():
                    return triton_full_attention_projection(gated, layer_index, weights, "o_proj_t", "o", q_dim, hidden)
                return gated @ weights["o_proj"].t()

            output_stage = (
                "output_gate_o_proj_triton_fused"
                if should_use_triton_full_attention_fused_gate_o_proj()
                else "output_gate_o_proj"
            )
            out = stage(output_stage, output_gate_o_proj)
            if record_events and total_start is not None and total_end is not None:
                total_end.record(torch.cuda.current_stream())
                event_pairs.setdefault("total", []).append((total_start, total_end))
            return out

        for _ in range(warmup):
            once(False)
        sync()
        out = None
        for _ in range(iters):
            out = once(True)
        sync()
        append_attention_event_records(
            entry=entry,
            attention="full_attention_cuda_event",
            event_pairs=event_pairs,
            records=records,
        )
        return out

    def profiled_linear_attention_events(
        inp: Any,
        layer_index: int,
        weights: dict[str, Any],
        records: list[dict[str, Any]],
    ) -> Any:
        if not (device.startswith("cuda") and torch.cuda.is_available()):
            return linear_attention(inp, layer_index, weights)
        if not (linear_attention_uses_vllm_auto_decode(linear_attention_variant) and mode == "decode"):
            return linear_attention(inp, layer_index, weights)
        entry = layer_weights[layer_index]
        event_pairs: dict[str, list[tuple[Any, Any]]] = {}

        def once(record_events: bool) -> Any:
            def stage(name: str, fn: Callable[[], Any]) -> Any:
                if not record_events:
                    return fn()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record(torch.cuda.current_stream())
                out = fn()
                end_event.record(torch.cuda.current_stream())
                event_pairs.setdefault(name, []).append((start_event, end_event))
                return out

            total_start = torch.cuda.Event(enable_timing=True) if record_events else None
            total_end = torch.cuda.Event(enable_timing=True) if record_events else None
            if record_events and total_start is not None:
                total_start.record(torch.cuda.current_stream())

            def input_proj_conv_qkv_layout() -> tuple[Any, Any, Any, Any, Any, Any]:
                if should_use_triton_fused_linear_input_proj_conv_qkv_layout():
                    return triton_fused_linear_input_projection_conv_qkv_layout(inp, layer_index, weights)
                mixed_qkv, z, a, b = linear_input_projection(inp, layer_index, weights)
                conv_qkv = (
                    mixed_qkv
                    if should_use_triton_fused_linear_input_proj_conv()
                    else linear_causal_conv(mixed_qkv, layer_index, weights)
                )
                q_raw, k_raw, value_raw = conv_qkv.split([linear_key_dim, linear_key_dim, linear_value_dim], dim=-1)
                return (
                    q_raw.view(tokens, linear_key_heads, linear_key_head_dim).unsqueeze(0).contiguous(),
                    k_raw.view(tokens, linear_key_heads, linear_key_head_dim).unsqueeze(0).contiguous(),
                    value_raw.view(tokens, linear_value_heads, linear_value_head_dim).unsqueeze(0).contiguous(),
                    z,
                    a,
                    b,
                )

            q, k, value, z, a, b = stage("input_proj_conv_qkv_layout", input_proj_conv_qkv_layout)

            def recurrent_gdn_state() -> Any:
                if vllm_fused_recurrent_gdn_update is None:
                    raise RuntimeError("vLLM recurrent GDN kernel was not loaded")
                core_attn_out, final_state = vllm_fused_recurrent_gdn_update(
                    A_log=weights["linear_A_log"].contiguous(),
                    a=a.contiguous(),
                    b=b.contiguous(),
                    dt_bias=weights["linear_dt_bias"].contiguous(),
                    q=q,
                    k=k,
                    v=value,
                    initial_state=vllm_linear_initial_state(layer_index),
                    inplace_final_state=False,
                    use_qk_l2norm_in_kernel=True,
                )
                if linear_attention_uses_native_vllm_decode_state(linear_attention_variant):
                    if linear_attention_uses_native_vllm_decode_state_refswap(linear_attention_variant):
                        linear_attention_ssm_states_vllm[layer_index] = final_state
                    else:
                        linear_attention_ssm_states_vllm[layer_index].copy_(final_state)
                else:
                    linear_attention_ssm_states[layer_index].copy_(final_state.squeeze(0).transpose(-1, -2))
                return core_attn_out

            core_attn_out = stage("recurrent_gdn_state", recurrent_gdn_state)

            def gated_norm_out_proj() -> Any:
                if should_use_triton_linear_gated_norm():
                    gated_out = triton_linear_gated_norm(core_attn_out, z, layer_index, weights)
                else:
                    core_flat = core_attn_out.reshape(-1, linear_value_head_dim)
                    z_flat = z.reshape(-1, linear_value_head_dim)
                    variance = core_flat.float().pow(2).mean(dim=-1, keepdim=True)
                    gated = (
                        core_flat.float()
                        * torch.rsqrt(variance + 1e-6)
                        * weights["linear_norm"].float()
                    ).to(dtype)
                    gated = (gated.float() * F.silu(z_flat.float())).to(dtype)
                    gated_out = gated.view(tokens, linear_value_dim)
                if should_use_triton_linear_output_proj():
                    return triton_linear_output_projection(gated_out, layer_index, weights)
                return gated_out @ weights["linear_out_proj"].t()

            out = stage("gated_norm_out_proj", gated_norm_out_proj)
            if record_events and total_start is not None and total_end is not None:
                total_end.record(torch.cuda.current_stream())
                event_pairs.setdefault("total", []).append((total_start, total_end))
            return out

        for _ in range(warmup):
            once(False)
        sync()
        out = None
        for _ in range(iters):
            out = once(True)
        sync()
        append_attention_event_records(
            entry=entry,
            attention="linear_attention_cuda_event",
            event_pairs=event_pairs,
            records=records,
        )
        return out

    def summarize_attention_substages(records: list[dict[str, Any]]) -> dict[str, Any]:
        by_attention: dict[str, float] = {}
        by_attention_stage: dict[str, float] = {}
        by_layer: dict[str, dict[str, Any]] = {}
        for record in records:
            attention = str(record["attention"])
            stage = str(record["stage"])
            layer_key = str(record["layer"])
            ms = float(record["ms_per_iter"])
            by_attention[attention] = by_attention.get(attention, 0.0) + ms
            stage_key = f"{attention}.{stage}"
            by_attention_stage[stage_key] = by_attention_stage.get(stage_key, 0.0) + ms
            layer_entry = by_layer.setdefault(
                layer_key,
                {
                    "layer": record["layer"],
                    "layer_type": record["layer_type"],
                    "attention": attention,
                    "ms_per_iter_sum": 0.0,
                    "stages": {},
                },
            )
            layer_entry["ms_per_iter_sum"] += ms
            layer_entry["stages"][stage] = layer_entry["stages"].get(stage, 0.0) + ms
        return {
            "record_count": len(records),
            "by_attention_ms": dict(sorted(by_attention.items())),
            "by_attention_stage_ms": dict(sorted(by_attention_stage.items())),
            "by_layer": [by_layer[key] for key in sorted(by_layer, key=int)],
        }

    def summarize_moe_substages(records: list[dict[str, Any]]) -> dict[str, Any]:
        by_stage: dict[str, float] = {}
        by_layer: dict[str, dict[str, Any]] = {}
        for record in records:
            stage = str(record["stage"])
            layer_key = str(record["layer"])
            ms = float(record["ms_per_iter"])
            by_stage[stage] = by_stage.get(stage, 0.0) + ms
            layer_entry = by_layer.setdefault(
                layer_key,
                {
                    "layer": record["layer"],
                    "layer_type": record["layer_type"],
                    "ms_per_iter_sum": 0.0,
                    "stages": {},
                },
            )
            layer_entry["ms_per_iter_sum"] += ms
            layer_entry["stages"][stage] = layer_entry["stages"].get(stage, 0.0) + ms
        return {
            "record_count": len(records),
            "by_stage_ms": dict(sorted(by_stage.items())),
            "by_layer": [by_layer[key] for key in sorted(by_layer, key=int)],
        }

    def summarize_moe_overlap_events(records: list[dict[str, Any]]) -> dict[str, Any]:
        summary = summarize_moe_substages(records)
        by_stage = summary["by_stage_ms"]
        main_total = float(by_stage.get("main_total", 0.0))
        main_pre_wait = float(by_stage.get("main_pre_wait_window", 0.0))
        wait_for_shared = float(by_stage.get("wait_for_shared_expert", 0.0))
        shared_side = float(by_stage.get("shared_expert_side_stream", 0.0))
        summary["derived"] = {
            "main_total_ms": main_total,
            "main_pre_wait_window_ms": main_pre_wait,
            "shared_expert_side_stream_ms": shared_side,
            "wait_for_shared_expert_ms": wait_for_shared,
            "wait_share_of_main_total_pct": (
                wait_for_shared * 100.0 / main_total
                if main_total > 0.0
                else None
            ),
            "shared_expert_hidden_by_main_window_ms": max(0.0, shared_side - wait_for_shared),
        }
        return summary

    def triton_router_topk(inp: Any, layer_index: int, weights: dict[str, Any]) -> tuple[Any, Any]:
        if (
            triton is None
            or triton_router_topk_stage1_kernel is None
            or triton_router_topk_stage2_kernel is None
            or (triton_router_topk_softmax_decode and triton_router_topk_stage2_softmax_kernel is None)
        ):
            raise RuntimeError("Triton router top-k kernels are not available")
        partial_values = router_topk_partial_values[layer_index]
        partial_ids = router_topk_partial_ids[layer_index]
        out_values = router_topk_score_outputs[layer_index]
        out_ids = router_topk_index_outputs[layer_index]
        block_e = 64
        block_k = 128
        num_blocks = triton.cdiv(experts, block_e)
        triton_router_topk_stage1_kernel[(num_blocks,)](
            inp.reshape(hidden),
            weights["router_t"],
            partial_values,
            partial_ids,
            hidden,
            experts,
            block_e=block_e,
            block_k=block_k,
            num_warps=4,
        )
        num_candidates = num_blocks * top_k
        stage2_kernel = (
            triton_router_topk_stage2_softmax_kernel
            if triton_router_topk_softmax_decode
            else triton_router_topk_stage2_kernel
        )
        stage2_kernel[(1,)](
            partial_values,
            partial_ids,
            out_values.reshape(top_k),
            out_ids.reshape(top_k),
            num_candidates,
            block_c=triton.next_power_of_2(num_candidates),
            num_warps=4,
        )
        return out_values, out_ids

    def router_topk(inp: Any, layer_index: int, weights: dict[str, Any]) -> tuple[Any, Any]:
        if triton_router_topk_decode:
            return triton_router_topk(inp, layer_index, weights)
        if torch_out_router_decode:
            logits = router_logits_outputs[layer_index]
            torch.mm(inp.view(tokens, hidden), weights["router_t"], out=logits)
        else:
            logits = inp @ weights["router"].t()
        scores, indices = torch.topk(logits.float(), k=top_k, dim=-1)
        return scores, indices

    def routing_weights(scores: Any) -> Any:
        if triton_router_topk_softmax_decode:
            return scores.to(dtype)
        return torch.softmax(scores, dim=-1).to(dtype)

    def expanded_routed_moe(inp: Any, scores: Any, indices: Any, weights: dict[str, Any]) -> Any:
        expanded = inp.repeat_interleave(top_k, dim=0)
        flat_indices = indices.reshape(-1)
        chunks = []
        for start in range(0, expanded.shape[0], moe_chunk_size):
            end = min(start + moe_chunk_size, expanded.shape[0])
            selected_gate_up = weights["expert_gate_up"][flat_indices[start:end]].transpose(1, 2)
            projected = torch.bmm(expanded[start:end].unsqueeze(1), selected_gate_up).squeeze(1)
            gate, up = projected.chunk(2, dim=-1)
            activated = (F.silu(gate.float()) * up.float()).to(dtype)
            selected_down = weights["expert_down"][flat_indices[start:end]].transpose(1, 2)
            chunks.append(torch.bmm(activated.unsqueeze(1), selected_down).squeeze(1))
        expert_out = torch.cat(chunks, dim=0)
        return (expert_out.view(tokens, top_k, hidden) * routing_weights(scores).unsqueeze(-1)).sum(dim=1)

    def build_dispatch_table(indices: Any) -> dict[str, Any]:
        flat_indices = indices.reshape(-1)
        sorted_experts, order = torch.sort(flat_indices)
        token_positions = torch.div(order, top_k, rounding_mode="floor")
        unique_experts, counts = torch.unique_consecutive(sorted_experts, return_counts=True)
        segments = []
        start = 0
        for expert_id, count in zip(unique_experts.tolist(), counts.tolist(), strict=True):
            end = start + int(count)
            segments.append((int(expert_id), start, end))
            start = end
        return {
            "active_positions": order,
            "token_positions": token_positions,
            "segments": segments,
            "populated_experts": len(segments),
            "active_rows": int(order.numel()),
        }

    def build_padded_dispatch_table(dispatch_table: dict[str, Any]) -> dict[str, Any]:
        segments = dispatch_table["segments"]
        if not segments:
            raise ValueError("dispatch table has no populated experts")
        max_rows_per_expert = max(end - start for _, start, end in segments)
        segment_count = len(segments)
        expert_ids = torch.empty(segment_count, device=device, dtype=torch.long)
        padded_active_positions = torch.zeros(segment_count, max_rows_per_expert, device=device, dtype=torch.long)
        padded_token_positions = torch.zeros(segment_count, max_rows_per_expert, device=device, dtype=torch.long)
        valid_mask = torch.zeros(segment_count, max_rows_per_expert, device=device, dtype=torch.bool)
        active_positions_all = dispatch_table["active_positions"]
        token_positions_all = dispatch_table["token_positions"]
        for row, (expert_id, start, end) in enumerate(segments):
            count = end - start
            expert_ids[row] = expert_id
            padded_active_positions[row, :count] = active_positions_all[start:end]
            padded_token_positions[row, :count] = token_positions_all[start:end]
            valid_mask[row, :count] = True
        return {
            "expert_ids": expert_ids,
            "padded_active_positions": padded_active_positions,
            "padded_token_positions": padded_token_positions,
            "valid_mask": valid_mask,
            "segment_count": segment_count,
            "max_rows_per_expert": max_rows_per_expert,
            "padded_rows": segment_count * max_rows_per_expert,
            "active_rows": dispatch_table["active_rows"],
            "padding_overhead_rows": segment_count * max_rows_per_expert - dispatch_table["active_rows"],
        }

    def build_count_batched_dispatch_table(dispatch_table: dict[str, Any]) -> dict[str, Any]:
        groups_by_count: dict[int, dict[str, list[Any]]] = {}
        active_positions_all = dispatch_table["active_positions"]
        token_positions_all = dispatch_table["token_positions"]
        for expert_id, start, end in dispatch_table["segments"]:
            count = end - start
            bucket = groups_by_count.setdefault(
                count,
                {
                    "expert_ids": [],
                    "active_positions": [],
                    "token_positions": [],
                },
            )
            bucket["expert_ids"].append(expert_id)
            bucket["active_positions"].append(active_positions_all[start:end])
            bucket["token_positions"].append(token_positions_all[start:end])
        groups = []
        row_count_groups = []
        for count in sorted(groups_by_count):
            bucket = groups_by_count[count]
            expert_ids = torch.tensor(bucket["expert_ids"], device=device, dtype=torch.long)
            active_positions = torch.stack(bucket["active_positions"], dim=0)
            token_positions = torch.stack(bucket["token_positions"], dim=0)
            groups.append(
                {
                    "rows_per_expert": count,
                    "expert_ids": expert_ids,
                    "active_positions": active_positions,
                    "token_positions": token_positions,
                }
            )
            row_count_groups.append(
                {
                    "rows_per_expert": count,
                    "segments": int(expert_ids.numel()),
                }
            )
        return {
            "groups": groups,
            "row_count_groups": row_count_groups,
            "row_count_group_count": len(groups),
            "max_segments_per_count": max((item["segments"] for item in row_count_groups), default=0),
            "segment_count": dispatch_table["populated_experts"],
            "active_rows": dispatch_table["active_rows"],
            "padded_rows": dispatch_table["active_rows"],
            "padding_overhead_rows": 0,
        }

    def resident_routed_moe(
        inp: Any,
        scores: Any,
        dispatch_table: dict[str, Any],
        weights: dict[str, Any],
        workspace: Any,
    ) -> Any:
        flat_weights = routing_weights(scores).reshape(-1)
        active_positions_all = dispatch_table["active_positions"]
        token_positions_all = dispatch_table["token_positions"]
        result = workspace.zero_()
        for expert_id, start, end in dispatch_table["segments"]:
            active_positions = active_positions_all[start:end]
            token_positions = token_positions_all[start:end]
            selected = inp[token_positions]
            projected = selected @ weights["expert_gate_up"][expert_id].t()
            gate, up = projected.chunk(2, dim=-1)
            activated = (F.silu(gate.float()) * up.float()).to(dtype)
            expert_out = activated @ weights["expert_down"][expert_id].t()
            weighted = expert_out * flat_weights[active_positions].unsqueeze(-1)
            result.index_add_(0, token_positions, weighted)
        return result

    def padded_batched_routed_moe(
        inp: Any,
        scores: Any,
        padded_dispatch_table: dict[str, Any],
        weights: dict[str, Any],
        workspace: Any,
    ) -> Any:
        flat_weights = routing_weights(scores).reshape(-1)
        expert_ids = padded_dispatch_table["expert_ids"]
        active_positions = padded_dispatch_table["padded_active_positions"]
        token_positions = padded_dispatch_table["padded_token_positions"]
        valid_mask = padded_dispatch_table["valid_mask"]

        selected = inp[token_positions]
        selected_gate_up = weights["expert_gate_up"][expert_ids].transpose(1, 2)
        projected = torch.bmm(selected, selected_gate_up)
        gate, up = projected.chunk(2, dim=-1)
        activated = (F.silu(gate.float()) * up.float()).to(dtype)
        selected_down = weights["expert_down"][expert_ids].transpose(1, 2)
        expert_out = torch.bmm(activated, selected_down)
        weighted = expert_out * flat_weights[active_positions].unsqueeze(-1)

        result = workspace.zero_()
        result.index_add_(0, token_positions[valid_mask], weighted[valid_mask])
        return result

    def count_batched_routed_moe(
        inp: Any,
        scores: Any,
        count_dispatch_table: dict[str, Any],
        weights: dict[str, Any],
        workspace: Any,
    ) -> Any:
        flat_weights = routing_weights(scores).reshape(-1)
        result = workspace.zero_()
        for group in count_dispatch_table["groups"]:
            expert_ids = group["expert_ids"]
            active_positions = group["active_positions"]
            token_positions = group["token_positions"]
            selected = inp[token_positions]
            selected_gate_up = weights["expert_gate_up"][expert_ids].transpose(1, 2)
            projected = torch.bmm(selected, selected_gate_up)
            gate, up = projected.chunk(2, dim=-1)
            activated = (F.silu(gate.float()) * up.float()).to(dtype)
            selected_down = weights["expert_down"][expert_ids].transpose(1, 2)
            expert_out = torch.bmm(activated, selected_down)
            weighted = expert_out * flat_weights[active_positions].unsqueeze(-1)
            result.index_add_(0, token_positions.reshape(-1), weighted.reshape(-1, hidden))
        return result

    def vllm_fused_routed_moe(
        inp: Any,
        scores: Any,
        indices: Any,
        weights: dict[str, Any],
        *,
        layer_index: int | None = None,
        inplace: bool = False,
    ) -> Any:
        topk_ids = indices.contiguous() if indices.dtype == torch.int32 else indices.to(torch.int32).contiguous()
        if (
            moe_variant == "vllm_fused_prefill_m32_n32_decode_m32_n16_k512_prealloc"
            and mode == "decode"
            and tokens == 1
            and not inplace
        ):
            if layer_index is None:
                raise RuntimeError("preallocated vLLM MoE requires layer_index")
            if (
                vllm_moe_dispatch_kernel is None
                or vllm_moe_apply_activation is None
                or vllm_moe_activation_silu is None
                or vllm_moe_ops is None
                or tl is None
            ):
                raise RuntimeError("vLLM MoE prealloc dependencies are unavailable")
            override_config = vllm_moe_override_config_for_layer_index(layer_index)
            if override_config is None:
                raise RuntimeError(f"{moe_variant} requires an explicit vLLM MoE override config")
            if dtype == torch.bfloat16:
                compute_type = tl.bfloat16
            elif dtype == torch.float16:
                compute_type = tl.float16
            elif dtype == torch.float32:
                compute_type = tl.float32
            else:
                raise RuntimeError(f"unsupported vLLM MoE prealloc dtype: {dtype}")
            moe_input = inp.contiguous()
            routing = routing_weights(scores).contiguous()
            cache1 = vllm_moe_cache1_outputs[layer_index]
            cache2 = vllm_moe_cache2_outputs[layer_index]
            cache3 = vllm_moe_cache3_outputs[layer_index]
            out = vllm_moe_prealloc_outputs[layer_index]
            num_tokens_post_padded = vllm_moe_num_tokens_post_padded[layer_index]
            num_tokens_post_padded.fill_(topk_ids.numel() * int(override_config["BLOCK_SIZE_M"]))
            expert_ids = topk_ids.view(-1)
            vllm_moe_dispatch_kernel(
                moe_input,
                weights["expert_gate_up"],
                cache1,
                None,
                None,
                None,
                routing,
                None,
                expert_ids,
                num_tokens_post_padded,
                False,
                top_k,
                override_config,
                compute_type=compute_type,
                use_fp8_w8a8=False,
                use_int8_w8a8=False,
                use_int8_w8a16=False,
                use_int4_w4a16=False,
                per_channel_quant=False,
                block_shape=None,
                B_bias=None,
            )
            vllm_moe_apply_activation(vllm_moe_activation_silu, cache2, cache1.view(-1, int(cache1.shape[-1])))
            vllm_moe_dispatch_kernel(
                cache2,
                weights["expert_down"],
                cache3,
                None,
                None,
                None,
                routing,
                None,
                expert_ids,
                num_tokens_post_padded,
                True,
                1,
                override_config,
                compute_type=compute_type,
                use_fp8_w8a8=False,
                use_int8_w8a8=False,
                use_int8_w8a16=False,
                use_int4_w4a16=False,
                per_channel_quant=False,
                block_shape=None,
                B_bias=None,
            )
            vllm_moe_ops.moe_sum(cache3.view(*cache3.size()), out)
            return out

        args = (
            inp.contiguous(),
            weights["expert_gate_up"],
            weights["expert_down"],
            routing_weights(scores).contiguous(),
            topk_ids,
        )
        override_config = vllm_moe_override_config_for_layer_index(layer_index)
        if override_config is None:
            return fused_experts(*args, inplace=inplace)
        with fused_experts_override_config_fn(override_config):
            return fused_experts(*args, inplace=inplace)

    def native_moe_consumer_routed_moe(
        inp: Any,
        scores: Any,
        indices: Any,
        weights: dict[str, Any],
        *,
        layer_index: int,
    ) -> Any:
        nonlocal resident_native_prefill_reconstruction_calls
        if not native_moe_consumer_decode:
            prefill_weights = weights
            if "expert_gate_up" not in weights or "expert_down" not in weights:
                if not native_moe_consumer_memory_safe:
                    raise RuntimeError("raw prefill expert weights are absent outside resident memory-safe mode")
                if "native_moe_gate_up" not in weights or "native_moe_down" not in weights:
                    raise RuntimeError("resident native expert layouts are incomplete")
                prefill_weights = dict(weights)
                prefill_weights["expert_gate_up"] = native_moe_unpack_gate_up_layout(
                    weights["native_moe_gate_up"]
                )
                prefill_weights["expert_down"] = native_moe_unpack_down_layout(
                    weights["native_moe_down"]
                )
                resident_native_prefill_reconstruction_calls += 1
                resident_native_prefill_reconstructed_layers.add(layer_index)
            return vllm_fused_routed_moe(
                inp, scores, indices, prefill_weights, layer_index=layer_index
            )
        if dtype != torch.bfloat16:
            raise RuntimeError(f"{moe_variant} currently supports BF16 only")
        topk_ids = (
            indices.contiguous()
            if indices.dtype == torch.int32
            else indices.to(torch.int32).contiguous()
        )
        routing = routing_weights(scores).contiguous()
        moe_input = inp.contiguous()
        act = native_moe_activation_outputs[layer_index]
        out = native_moe_outputs[layer_index]
        if layer_index in resident_native_decode_hotset_layer_indices:
            gate_up_native = weights["native_moe_gate_up"]
            down_native = weights["native_moe_down"]
            block_i = native_moe_config["layout_block_i"]
            block_h = native_moe_config["layout_block_h"]
            num_i_tiles = math.ceil(intermediate / block_i)
            hidden_tiles = math.ceil(hidden / block_h)
            triton_native_moe_gate_up_activation_kernel[(top_k, num_i_tiles)](
                moe_input.reshape(-1),
                gate_up_native,
                topk_ids.reshape(-1),
                act,
                hidden,
                intermediate,
                num_i_tiles,
                block_i=block_i,
                block_h=native_moe_config["gate_block_h"],
                num_warps=native_moe_config["gate_num_warps"],
            )
            triton_native_moe_down_sum_kernel[(hidden_tiles,)](
                act,
                down_native,
                topk_ids.reshape(-1),
                routing.reshape(-1),
                out.reshape(-1),
                hidden,
                intermediate,
                top_k,
                hidden_tiles,
                block_h=block_h,
                block_i=native_moe_config["down_block_i"],
                num_warps=native_moe_config["down_num_warps"],
            )
            return out
        gate_up_raw = weights["expert_gate_up"]
        down_raw = weights["expert_down"]
        raw_row_gate_up_activation_kernel[(top_k, intermediate)](
            moe_input.reshape(-1),
            gate_up_raw,
            topk_ids.reshape(-1),
            act,
            hidden,
            intermediate,
            BLOCK_H=2048,
            num_warps=4,
        )
        raw_row_down_sum_kernel[(hidden,)](
            act,
            down_raw,
            topk_ids.reshape(-1),
            routing.reshape(-1),
            out.reshape(-1),
            hidden,
            intermediate,
            top_k,
            BLOCK_I=256,
            num_warps=1,
        )
        return out

    def profiled_vllm_fused_routed_moe(
        inp: Any,
        scores: Any,
        indices: Any,
        weights: dict[str, Any],
        layer_index: int,
        records: list[dict[str, Any]],
    ) -> Any:
        entry = layer_weights[layer_index]
        profile = lambda stage, fn: diagnostic_timed(
            layer=entry["layer"],
            layer_type=entry["layer_type"],
            attention="moe",
            stage=stage,
            fn=fn,
            records=records,
        )
        moe_input = profile("input_contiguous", lambda: inp.contiguous())
        routing = profile("routing_weights", lambda: routing_weights(scores).contiguous())
        topk_ids = profile(
            "topk_ids_int32_contiguous",
            lambda: indices.contiguous() if indices.dtype == torch.int32 else indices.to(torch.int32).contiguous(),
        )
        override_config = vllm_moe_override_config_for_layer_index(layer_index)

        def run_fused_experts() -> Any:
            if moe_variant == "vllm_fused_prefill_m32_n32_decode_m32_n16_k512_prealloc":
                return vllm_fused_routed_moe(
                    moe_input,
                    scores,
                    topk_ids,
                    weights,
                    layer_index=layer_index,
                )
            if override_config is None:
                return fused_experts(
                    moe_input,
                    weights["expert_gate_up"],
                    weights["expert_down"],
                    routing,
                    topk_ids,
                    inplace=False,
                )
            with fused_experts_override_config_fn(override_config):
                return fused_experts(
                    moe_input,
                    weights["expert_gate_up"],
                    weights["expert_down"],
                    routing,
                    topk_ids,
                    inplace=False,
                )

        return profile("vllm_fused_experts", run_fused_experts)

    def shared_expert(inp: Any, layer_index: int, weights: dict[str, Any]) -> Any:
        if not include_shared_expert:
            return torch.zeros(tokens, hidden, device=device, dtype=dtype)
        if should_use_triton_shared_expert_fused_input():
            shared_input = triton_shared_expert_projection(
                inp,
                layer_index,
                weights,
                "shared_input_proj_fused_t",
                "input",
                hidden,
                1 + 2 * shared_intermediate,
            )
            if should_use_triton_shared_expert_fused_down():
                return triton_shared_expert_fused_down(shared_input, layer_index, weights)
            shared_gate, gate, up = shared_input.split([1, shared_intermediate, shared_intermediate], dim=-1)
        elif should_use_triton_shared_expert_proj():
            shared_gate = triton_shared_expert_projection(
                inp,
                layer_index,
                weights,
                "shared_gate_t",
                "gate",
                hidden,
                1,
                block_n=16,
                block_k=256,
            )
            gate = triton_shared_expert_projection(
                inp,
                layer_index,
                weights,
                "shared_gate_proj_t",
                "gate_proj",
                hidden,
                shared_intermediate,
            )
            up = triton_shared_expert_projection(
                inp,
                layer_index,
                weights,
                "shared_up_proj_t",
                "up_proj",
                hidden,
                shared_intermediate,
            )
        else:
            shared_gate = inp @ weights["shared_gate"].t()
            gate = inp @ weights["shared_gate_proj"].t()
            up = inp @ weights["shared_up_proj"].t()
        activated = (F.silu(gate.float()) * up.float()).to(dtype)
        if should_use_triton_shared_expert_proj():
            shared_out = triton_shared_expert_projection(
                activated,
                layer_index,
                weights,
                "shared_down_proj_t",
                "down_proj",
                shared_intermediate,
                hidden,
            )
        else:
            shared_out = activated @ weights["shared_down_proj"].t()
        return torch.sigmoid(shared_gate).to(dtype) * shared_out

    def vllm_fused_moe_with_shared_overlap(
        inp: Any,
        scores: Any,
        indices: Any,
        layer_index: int,
        weights: dict[str, Any],
    ) -> Any:
        if shared_expert_overlap_stream is None:
            raise RuntimeError("shared expert overlap stream is not initialized")
        current_stream = torch.cuda.current_stream()
        shared_expert_overlap_stream.wait_stream(current_stream)
        with torch.cuda.stream(shared_expert_overlap_stream):
            shared_out = shared_expert(inp, layer_index, weights)
        moe_out = vllm_fused_routed_moe(inp, scores, indices, weights, layer_index=layer_index)
        current_stream.wait_stream(shared_expert_overlap_stream)
        return moe_out + shared_out

    def native_moe_consumer_with_shared_overlap(
        inp: Any,
        scores: Any,
        indices: Any,
        layer_index: int,
        weights: dict[str, Any],
    ) -> Any:
        if shared_expert_overlap_stream is None:
            raise RuntimeError("shared expert overlap stream is not initialized")
        current_stream = torch.cuda.current_stream()
        shared_expert_overlap_stream.wait_stream(current_stream)
        with torch.cuda.stream(shared_expert_overlap_stream):
            shared_out = shared_expert(inp, layer_index, weights)
        moe_out = native_moe_consumer_routed_moe(
            inp, scores, indices, weights, layer_index=layer_index
        )
        current_stream.wait_stream(shared_expert_overlap_stream)
        return moe_out + shared_out

    def start_shared_expert_overlap(inp: Any, layer_index: int, weights: dict[str, Any]) -> Any:
        if shared_expert_overlap_stream is None:
            raise RuntimeError("shared expert overlap stream is not initialized")
        current_stream = torch.cuda.current_stream()
        shared_expert_overlap_stream.wait_stream(current_stream)
        with torch.cuda.stream(shared_expert_overlap_stream):
            return shared_expert(inp, layer_index, weights)

    def finish_shared_expert_overlap(shared_out: Any) -> Any:
        if shared_expert_overlap_stream is None:
            raise RuntimeError("shared expert overlap stream is not initialized")
        torch.cuda.current_stream().wait_stream(shared_expert_overlap_stream)
        return shared_out

    def profile_router_shared_overlap_events(
        inp: Any,
        layer_index: int,
        weights: dict[str, Any],
        records: list[dict[str, Any]],
    ) -> None:
        if not (device.startswith("cuda") and torch.cuda.is_available()):
            return
        if shared_expert_overlap_stream is None:
            return
        if not (should_overlap_shared_expert_router_moe() and moe_variant in vllm_fused_moe_variants()):
            return

        entry = layer_weights[layer_index]

        def run_once() -> Any:
            profile_shared_out = start_shared_expert_overlap(inp, layer_index, weights)
            profile_scores, profile_indices = router_topk(inp, layer_index, weights)
            profile_moe_out = vllm_fused_routed_moe(
                inp,
                profile_scores,
                profile_indices,
                weights,
                layer_index=layer_index,
            )
            profile_shared_out = finish_shared_expert_overlap(profile_shared_out)
            return profile_moe_out + profile_shared_out

        for _ in range(warmup):
            run_once()
        sync()

        rows: list[dict[str, Any]] = []
        for _ in range(iters):
            current_stream = torch.cuda.current_stream()
            events = {
                key: torch.cuda.Event(enable_timing=True)
                for key in (
                    "main_start",
                    "router_start",
                    "router_end",
                    "moe_start",
                    "moe_end",
                    "wait_start",
                    "wait_end",
                    "add_start",
                    "add_end",
                    "main_end",
                    "shared_start",
                    "shared_end",
                )
            }

            events["main_start"].record(current_stream)
            shared_expert_overlap_stream.wait_stream(current_stream)
            with torch.cuda.stream(shared_expert_overlap_stream):
                events["shared_start"].record(shared_expert_overlap_stream)
                shared_out = shared_expert(inp, layer_index, weights)
                events["shared_end"].record(shared_expert_overlap_stream)

            events["router_start"].record(current_stream)
            scores, indices = router_topk(inp, layer_index, weights)
            events["router_end"].record(current_stream)

            events["moe_start"].record(current_stream)
            moe_out = vllm_fused_routed_moe(
                inp,
                scores,
                indices,
                weights,
                layer_index=layer_index,
            )
            events["moe_end"].record(current_stream)

            events["wait_start"].record(current_stream)
            current_stream.wait_stream(shared_expert_overlap_stream)
            events["wait_end"].record(current_stream)

            events["add_start"].record(current_stream)
            _ = moe_out + shared_out
            events["add_end"].record(current_stream)
            events["main_end"].record(current_stream)
            rows.append(events)

        sync()

        stage_values = {
            "router_topk": [],
            "routed_moe_with_prep": [],
            "shared_expert_side_stream": [],
            "wait_for_shared_expert": [],
            "moe_shared_add": [],
            "main_pre_wait_window": [],
            "main_total": [],
        }
        for events in rows:
            stage_values["router_topk"].append(events["router_start"].elapsed_time(events["router_end"]))
            stage_values["routed_moe_with_prep"].append(events["moe_start"].elapsed_time(events["moe_end"]))
            stage_values["shared_expert_side_stream"].append(
                events["shared_start"].elapsed_time(events["shared_end"])
            )
            stage_values["wait_for_shared_expert"].append(events["wait_start"].elapsed_time(events["wait_end"]))
            stage_values["moe_shared_add"].append(events["add_start"].elapsed_time(events["add_end"]))
            stage_values["main_pre_wait_window"].append(events["main_start"].elapsed_time(events["wait_start"]))
            stage_values["main_total"].append(events["main_start"].elapsed_time(events["main_end"]))

        for stage, values in stage_values.items():
            ms_per_iter = sum(float(value) for value in values) / len(values) if values else 0.0
            records.append(
                {
                    "layer": entry["layer"],
                    "layer_type": entry["layer_type"],
                    "attention": "moe",
                    "stage": stage,
                    "name": f"layer{entry['layer']}_moe_overlap_{stage}",
                    "iters": iters,
                    "ms_per_iter": ms_per_iter,
                    "elapsed_ms": ms_per_iter * iters,
                }
            )

    def layer_body(inp: Any, layer_index: int, variant: str) -> tuple[Any, dict[str, Any]]:
        entry = layer_weights[layer_index]
        weights = entry["tensors"]
        h1 = rmsnorm(inp, weights["input_layernorm"], rmsnorm_input_outputs.get(layer_index))
        attn_out = attention(h1, layer_index, weights)
        if triton_rmsnorm_prefill and int(inp.shape[0]) > 1:
            after_attn, h2 = prefill_fused_add_rmsnorm(
                inp,
                attn_out,
                weights["post_attention_layernorm"],
            )
        else:
            after_attn = inp + attn_out
            h2 = rmsnorm(after_attn, weights["post_attention_layernorm"], rmsnorm_post_outputs.get(layer_index))
        model_layer = int(entry["layer"])
        if (
            variant == "resident"
            and use_q8192_compound_provider()
            and q8192_compound_provider().component_enabled("selected_moe")
            and model_layer < 39
        ):
            provider_hidden = q8192_compound_provider().launch_selected_moe(
                post_attention=h2,
                residual_hidden=after_attn,
                weights=weights,
                model_layer=model_layer,
            )
            if skip_layer_dispatch_metadata:
                return provider_hidden, None
            return provider_hidden, {
                "layer": model_layer,
                "layer_type": entry["layer_type"],
                "dispatch": {
                    "moe_variant": "q8192_triton_selected_moe_full_v3_async",
                    "populated_experts": None,
                    "active_rows": 8192 * top_k,
                    "backend": "row2553_exact_linux_source_port",
                    "inplace": True,
                    "override_config": None,
                    "padded_rows": None,
                    "padding_overhead_rows": None,
                },
            }
        shared_out = None
        shared_router_overlap_started = variant == "resident" and should_overlap_shared_expert_router_moe()
        if shared_router_overlap_started:
            shared_out = start_shared_expert_overlap(h2, layer_index, weights)
        scores, indices = router_topk(h2, layer_index, weights)
        moe_out_includes_shared = False
        if variant != "expanded" and include_shared_expert and moe_variant == "vllm_fused_inplace":
            shared_out = shared_expert(h2, layer_index, weights)
        dispatch_info = None
        if variant == "expanded":
            moe_out = expanded_routed_moe(h2, scores, indices, weights)
        elif variant == "resident":
            if moe_variant == "resident_dispatch":
                dispatch = build_dispatch_table(indices)
                moe_out = resident_routed_moe(h2, scores, dispatch, weights, resident_workspaces[layer_index])
                if not skip_layer_dispatch_metadata:
                    dispatch_info = {
                        "moe_variant": moe_variant,
                        "populated_experts": dispatch["populated_experts"],
                        "active_rows": dispatch["active_rows"],
                    }
            elif moe_variant in native_moe_consumer_variants():
                if shared_router_overlap_started:
                    moe_out = native_moe_consumer_routed_moe(
                        h2, scores, indices, weights, layer_index=layer_index
                    )
                    shared_out = finish_shared_expert_overlap(shared_out)
                    moe_out = moe_out + shared_out
                    moe_out_includes_shared = True
                elif should_overlap_shared_expert_moe():
                    moe_out = native_moe_consumer_with_shared_overlap(
                        h2, scores, indices, layer_index, weights
                    )
                    moe_out_includes_shared = True
                else:
                    moe_out = native_moe_consumer_routed_moe(
                        h2, scores, indices, weights, layer_index=layer_index
                    )
                if not skip_layer_dispatch_metadata:
                    dispatch_info = {
                        "moe_variant": moe_variant,
                        "populated_experts": None,
                        "active_rows": int(indices.numel()),
                        "backend": (
                            "native_selected_expert_consumer"
                            if layer_index in resident_native_decode_hotset_layer_indices
                            else "raw_row_contiguous_selected_expert_consumer"
                        ),
                        "inplace": False,
                        "override_config": None,
                        "padded_rows": None,
                        "padding_overhead_rows": None,
                    }
            elif moe_variant in vllm_fused_moe_variants():
                if shared_router_overlap_started:
                    moe_out = vllm_fused_routed_moe(h2, scores, indices, weights, layer_index=layer_index)
                    shared_out = finish_shared_expert_overlap(shared_out)
                    moe_out = moe_out + shared_out
                    moe_out_includes_shared = True
                elif should_overlap_shared_expert_moe():
                    moe_out = vllm_fused_moe_with_shared_overlap(h2, scores, indices, layer_index, weights)
                    moe_out_includes_shared = True
                else:
                    moe_out = vllm_fused_routed_moe(
                        h2,
                        scores,
                        indices,
                        weights,
                        layer_index=layer_index,
                        inplace=moe_variant == "vllm_fused_inplace",
                    )
                if not skip_layer_dispatch_metadata:
                    dispatch_info = {
                        "moe_variant": moe_variant,
                        "populated_experts": None,
                        "active_rows": int(indices.numel()),
                        "backend": "vllm.fused_experts",
                        "inplace": moe_variant == "vllm_fused_inplace",
                        "override_config": vllm_moe_override_config_for_layer_index(layer_index),
                        "padded_rows": None,
                        "padding_overhead_rows": None,
                    }
            else:
                dispatch = build_dispatch_table(indices)
                if moe_variant == "padded_batched":
                    candidate_dispatch = build_padded_dispatch_table(dispatch)
                    moe_out = padded_batched_routed_moe(
                        h2,
                        scores,
                        candidate_dispatch,
                        weights,
                        resident_workspaces[layer_index],
                    )
                else:
                    candidate_dispatch = build_count_batched_dispatch_table(dispatch)
                    moe_out = count_batched_routed_moe(
                        h2,
                        scores,
                        candidate_dispatch,
                        weights,
                        resident_workspaces[layer_index],
                    )
                if not skip_layer_dispatch_metadata:
                    dispatch_info = {
                        "moe_variant": moe_variant,
                        "populated_experts": dispatch["populated_experts"],
                        "active_rows": dispatch["active_rows"],
                        "row_count_group_count": candidate_dispatch.get("row_count_group_count"),
                        "max_segments_per_count": candidate_dispatch.get("max_segments_per_count"),
                        "max_rows_per_expert": candidate_dispatch.get("max_rows_per_expert"),
                        "padded_rows": candidate_dispatch["padded_rows"],
                        "padding_overhead_rows": candidate_dispatch["padding_overhead_rows"],
                    }
        else:
            raise ValueError(f"unsupported variant {variant}")
        if include_shared_expert and not moe_out_includes_shared:
            if shared_out is None:
                shared_out = shared_expert(h2, layer_index, weights)
            moe_out = moe_out + shared_out
        if skip_layer_dispatch_metadata:
            return after_attn + moe_out, None
        return after_attn + moe_out, {
            "layer": entry["layer"],
            "layer_type": entry["layer_type"],
            "dispatch": dispatch_info,
        }

    def run_engine(variant: str, start_hidden: Any | None = None) -> Any:
        hidden_state = x if start_hidden is None else start_hidden
        for layer_index in range(len(layer_weights)):
            hidden_state, _ = layer_body(hidden_state, layer_index, variant)
        return hidden_state

    def final_logits(inp: Any) -> Any:
        if not include_lm_head:
            raise RuntimeError("lm_head requested but not allocated")
        final_hidden = rmsnorm(inp[-1:], global_tensors["final_norm"], rmsnorm_final_output)
        if lm_head_variant == "int8_certified_global_tie":
            if certified_lm_head_logits_output is None:
                raise RuntimeError("certified lm_head logits buffer is not allocated")
            launch_certified_lm_head_int8(
                global_tensors["lm_head_int8"],
                final_hidden,
                global_tensors["lm_head_int8_scales"],
                certified_lm_head_logits_output,
            )
            hidden_l2 = torch.linalg.vector_norm(final_hidden.float())
            error_bound = global_tensors["lm_head_int8_residual_l2"] * hidden_l2
            lower_max = (certified_lm_head_logits_output - error_bound).max()
            upper = certified_lm_head_logits_output + error_bound
            upper_values, upper_indices = torch.topk(upper, k=1024)
            shortlist_weight = global_tensors["lm_head"].index_select(0, upper_indices)
            exact_logits = torch.mm(final_hidden, shortlist_weight.t()).view(-1)
            torch._assert_async(lower_max > upper_values[-1], "lm_head certificate failed")
            certified_lm_head_logits_output.scatter_(
                0,
                upper_indices,
                exact_logits.float(),
            )
            return certified_lm_head_logits_output.view(1, -1)
        if lm_head_variant == "pretransposed_out":
            if mode == "decode" and tokens == 1 and measurement_mode != "correctness":
                if lm_head_logits_output is None:
                    raise RuntimeError("pretransposed_out logits buffer is not allocated")
                torch.mm(final_hidden, global_tensors["lm_head_t"], out=lm_head_logits_output)
                return lm_head_logits_output
            return final_hidden @ global_tensors["lm_head_t"]
        if lm_head_variant == "pretransposed":
            return final_hidden @ global_tensors["lm_head_t"]
        return final_hidden @ global_tensors["lm_head"].t()

    def ensure_decode_buffers() -> None:
        nonlocal rmsnorm_final_output, lm_head_logits_output
        nonlocal full_attention_norm_rope_shared_outputs
        if mode != "decode" or tokens != 1:
            raise RuntimeError("decode buffers require decode mode with one token")
        if triton_rmsnorm_decode and rmsnorm_final_output is None:
            rmsnorm_final_output = torch.empty(1, hidden, device=device, dtype=dtype)
        if include_lm_head and lm_head_variant == "pretransposed_out" and lm_head_logits_output is None:
            lm_head_logits_output = torch.empty(1, vocab, device=device, dtype=dtype)
        if triton_full_attention_norm_rope_decode:
            full_attention_norm_rope_shared_outputs = {
                "q": torch.empty(tokens, heads, head_dim, device=device, dtype=dtype),
                "k": torch.empty(tokens, kv_heads, head_dim, device=device, dtype=dtype),
            }
        for layer_index, entry in enumerate(layer_weights):
            layer = entry["layer"]
            weights = entry["tensors"]
            if (torch_out_router_decode or triton_router_topk_decode) and "router_t" not in weights:
                weights["router_t"] = cached_tensor(
                    f"derived:layer{layer}:router_t",
                    lambda weights=weights: weights["router"].t().contiguous(),
                )
            if triton_full_attention_proj_decode and entry["layer_type"] == "full_attention":
                if triton_full_attention_fused_qkv_decode:
                    if "full_qkv_proj_fused" not in weights:
                        weights["full_qkv_proj_fused"] = cached_tensor(
                            f"derived:layer{layer}:full_qkv_proj_fused",
                            lambda weights=weights: torch.cat(
                                [
                                    weights["q_proj"],
                                    weights["k_proj"],
                                    weights["v_proj"],
                                ],
                                dim=0,
                            ).contiguous(),
                        )
                    if "full_qkv_proj_fused_t" not in weights:
                        weights["full_qkv_proj_fused_t"] = cached_tensor(
                            f"derived:layer{layer}:full_qkv_proj_fused_t",
                            lambda weights=weights: weights["full_qkv_proj_fused"].t().contiguous(),
                        )
                    if "o_proj_t" not in weights:
                        weights["o_proj_t"] = cached_tensor(
                            f"derived:layer{layer}:o_proj_t",
                            lambda weights=weights: weights["o_proj"].t().contiguous(),
                        )
                else:
                    for source_key, derived_key in (
                        ("q_proj", "q_proj_t"),
                        ("k_proj", "k_proj_t"),
                        ("v_proj", "v_proj_t"),
                        ("o_proj", "o_proj_t"),
                    ):
                        if derived_key not in weights:
                            weights[derived_key] = cached_tensor(
                                f"derived:layer{layer}:{derived_key}",
                                lambda weights=weights, source_key=source_key: weights[source_key].t().contiguous(),
                            )
            if triton_shared_expert_proj_decode:
                if triton_shared_expert_fused_input_decode:
                    if "shared_input_proj_fused" not in weights:
                        weights["shared_input_proj_fused"] = cached_tensor(
                            f"derived:layer{layer}:shared_input_proj_fused",
                            lambda weights=weights: torch.cat(
                                [
                                    weights["shared_gate"],
                                    weights["shared_gate_proj"],
                                    weights["shared_up_proj"],
                                ],
                                dim=0,
                            ).contiguous(),
                        )
                    if "shared_input_proj_fused_t" not in weights:
                        weights["shared_input_proj_fused_t"] = cached_tensor(
                            f"derived:layer{layer}:shared_input_proj_fused_t",
                            lambda weights=weights: weights["shared_input_proj_fused"].t().contiguous(),
                        )
                    if "shared_down_proj_t" not in weights:
                        weights["shared_down_proj_t"] = cached_tensor(
                            f"derived:layer{layer}:shared_down_proj_t",
                            lambda weights=weights: weights["shared_down_proj"].t().contiguous(),
                        )
                else:
                    for source_key, derived_key in (
                        ("shared_gate", "shared_gate_t"),
                        ("shared_gate_proj", "shared_gate_proj_t"),
                        ("shared_up_proj", "shared_up_proj_t"),
                        ("shared_down_proj", "shared_down_proj_t"),
                    ):
                        if derived_key not in weights:
                            weights[derived_key] = cached_tensor(
                                f"derived:layer{layer}:{derived_key}",
                                lambda weights=weights, source_key=source_key: weights[source_key].t().contiguous(),
                            )
            if triton_rmsnorm_decode:
                rmsnorm_input_outputs[layer_index] = torch.empty(tokens, hidden, device=device, dtype=dtype)
                rmsnorm_post_outputs[layer_index] = torch.empty(tokens, hidden, device=device, dtype=dtype)
            if full_attention_enabled(attention_mode) and entry["layer_type"] == "full_attention":
                if triton_full_attention_proj_decode:
                    if triton_full_attention_fused_qkv_decode:
                        full_attention_proj_outputs[layer_index] = {
                            "qkv": torch.empty(2 * q_dim + 2 * kv_dim, device=device, dtype=dtype),
                            "o": torch.empty(hidden, device=device, dtype=dtype),
                        }
                    else:
                        full_attention_proj_outputs[layer_index] = {
                            "q_gate": torch.empty(2 * q_dim, device=device, dtype=dtype),
                            "k": torch.empty(kv_dim, device=device, dtype=dtype),
                            "v": torch.empty(kv_dim, device=device, dtype=dtype),
                            "o": torch.empty(hidden, device=device, dtype=dtype),
                        }
                if triton_full_attention_norm_rope_decode:
                    if full_attention_norm_rope_shared_outputs is None:
                        raise RuntimeError("shared decode full-attention norm/RoPE workspace is not allocated")
                    full_attention_norm_rope_outputs[layer_index] = (
                        full_attention_norm_rope_shared_outputs
                    )
            if triton_shared_expert_proj_decode:
                if triton_shared_expert_fused_input_decode:
                    shared_expert_proj_outputs[layer_index] = {
                        "input": torch.empty(1 + 2 * shared_intermediate, device=device, dtype=dtype),
                        "down_proj": torch.empty(hidden, device=device, dtype=dtype),
                    }
                else:
                    shared_expert_proj_outputs[layer_index] = {
                        "gate": torch.empty(1, device=device, dtype=dtype),
                        "gate_proj": torch.empty(shared_intermediate, device=device, dtype=dtype),
                        "up_proj": torch.empty(shared_intermediate, device=device, dtype=dtype),
                        "down_proj": torch.empty(hidden, device=device, dtype=dtype),
                    }
            if linear_attention_enabled(attention_mode) and entry["layer_type"] == "linear_attention":
                if triton_linear_output_proj_decode and "linear_out_proj_t" not in weights:
                    weights["linear_out_proj_t"] = cached_tensor(
                        f"derived:layer{layer}:linear_out_proj_t",
                        lambda weights=weights: weights["linear_out_proj"].t().contiguous(),
                    )
                if linear_attention_uses_vllm_prestates(linear_attention_variant):
                    linear_attention_initial_ssm_states_vllm[layer_index] = (
                        linear_attention_initial_ssm_states[layer_index].unsqueeze(0).transpose(-1, -2).contiguous()
                    )
                    if linear_attention_uses_native_vllm_decode_state(linear_attention_variant):
                        linear_attention_ssm_states_vllm[layer_index] = torch.empty_like(
                            linear_attention_initial_ssm_states_vllm[layer_index]
                        )
                if should_use_packed_linear_gdn():
                    packed_initial = torch.zeros(
                        2,
                        linear_value_heads,
                        linear_value_head_dim,
                        linear_key_head_dim,
                        device=device,
                        dtype=torch.float32,
                    )
                    packed_initial[1].copy_(linear_attention_initial_ssm_states[layer_index].transpose(-1, -2).contiguous())
                    linear_attention_packed_initial_ssm_states[layer_index] = packed_initial
                    if not linear_attention_uses_packed_state_refswap(linear_attention_variant):
                        linear_attention_packed_ssm_states[layer_index] = torch.empty_like(packed_initial)
                    linear_attention_packed_outputs[layer_index] = torch.empty(
                        tokens,
                        1,
                        linear_value_heads,
                        linear_value_head_dim,
                        device=device,
                        dtype=dtype,
                    )
                    linear_attention_packed_state_indices[layer_index] = torch.ones(
                        tokens,
                        device=device,
                        dtype=torch.int32,
                    )
                if linear_attention_conv_variant == "decode_direct":
                    linear_attention_conv_windows[layer_index] = torch.empty(
                        linear_conv_dim,
                        linear_conv_kernel_dim,
                        device=device,
                        dtype=dtype,
                    )
                if triton_linear_conv_decode and not triton_fused_linear_input_proj_conv_decode:
                    linear_attention_conv_outputs[layer_index] = torch.empty(
                        linear_conv_dim,
                        device=device,
                        dtype=dtype,
                    )
                if triton_linear_gated_norm_decode:
                    linear_attention_gated_norm_outputs[layer_index] = torch.empty(
                        tokens,
                        linear_value_dim,
                        device=device,
                        dtype=dtype,
                    )
                if triton_fused_linear_input_proj_conv_qkv_decode:
                    linear_attention_qkv_layout_outputs[layer_index] = {
                        "q": torch.empty(
                            1,
                            tokens,
                            linear_key_heads,
                            linear_key_head_dim,
                            device=device,
                            dtype=dtype,
                        ),
                        "k": torch.empty(
                            1,
                            tokens,
                            linear_key_heads,
                            linear_key_head_dim,
                            device=device,
                            dtype=dtype,
                        ),
                        "v": torch.empty(
                            1,
                            tokens,
                            linear_value_heads,
                            linear_value_head_dim,
                            device=device,
                            dtype=dtype,
                        ),
                        "z": torch.empty(
                            tokens,
                            linear_value_heads,
                            linear_value_head_dim,
                            device=device,
                            dtype=dtype,
                        ),
                        "a": torch.empty(tokens, linear_value_heads, device=device, dtype=dtype),
                        "b": torch.empty(tokens, linear_value_heads, device=device, dtype=dtype),
                    }
                elif triton_linear_input_proj_decode or triton_fused_linear_input_proj_conv_decode:
                    linear_attention_input_proj_outputs[layer_index] = torch.empty(
                        int(entry["tensors"]["linear_input_proj_fused"].shape[0]),
                        device=device,
                        dtype=dtype,
                    )
                if triton_linear_output_proj_decode:
                    linear_attention_output_proj_outputs[layer_index] = torch.empty(
                        hidden,
                        device=device,
                        dtype=dtype,
                    )
            if torch_out_router_decode:
                router_logits_outputs[layer_index] = torch.empty(
                    tokens,
                    experts,
                    device=device,
                    dtype=dtype,
                )
            if triton_router_topk_decode:
                partial_count = math.ceil(experts / 64) * top_k
                router_topk_partial_values[layer_index] = torch.empty(
                    partial_count,
                    device=device,
                    dtype=torch.float32,
                )
                router_topk_partial_ids[layer_index] = torch.empty(
                    partial_count,
                    device=device,
                    dtype=torch.int32,
                )
                router_topk_score_outputs[layer_index] = torch.empty(
                    tokens,
                    top_k,
                    device=device,
                    dtype=dtype if triton_router_topk_softmax_decode else torch.float32,
                )
                router_topk_index_outputs[layer_index] = torch.empty(
                    tokens,
                    top_k,
                    device=device,
                    dtype=torch.int32,
                )

    def promote_decode_state_outputs_to_inputs() -> None:
        for layer_index, entry in enumerate(layer_weights):
            if not (linear_attention_enabled(attention_mode) and entry["layer_type"] == "linear_attention"):
                continue
            if linear_attention_conv_state_refswap:
                (
                    linear_attention_initial_conv_states[layer_index],
                    linear_attention_conv_states[layer_index],
                ) = (
                    linear_attention_conv_states[layer_index],
                    linear_attention_initial_conv_states[layer_index],
                )
            else:
                linear_attention_initial_conv_states[layer_index].copy_(linear_attention_conv_states[layer_index])
            if (
                mode == "decode"
                and linear_attention_uses_packed_state_refswap(linear_attention_variant)
                and layer_index in linear_attention_packed_initial_ssm_states
            ):
                continue
            if (
                mode == "decode"
                and linear_attention_uses_native_vllm_decode_state(linear_attention_variant)
                and layer_index in linear_attention_ssm_states_vllm
            ):
                if linear_attention_uses_native_vllm_decode_state_refswap(linear_attention_variant):
                    linear_attention_initial_ssm_states_vllm[layer_index] = linear_attention_ssm_states_vllm[
                        layer_index
                    ]
                    continue
                linear_attention_initial_ssm_states_vllm[layer_index].copy_(
                    linear_attention_ssm_states_vllm[layer_index]
                )
                continue
            if use_prefill_vllm_state_handoff() and layer_index in linear_attention_ssm_states_vllm:
                if linear_attention_uses_native_vllm_decode_state_refswap(linear_attention_variant):
                    linear_attention_initial_ssm_states_vllm[layer_index] = linear_attention_ssm_states_vllm[
                        layer_index
                    ]
                    continue
                linear_attention_initial_ssm_states_vllm[layer_index].copy_(
                    linear_attention_ssm_states_vllm[layer_index]
                )
                continue
            linear_attention_initial_ssm_states[layer_index].copy_(linear_attention_ssm_states[layer_index])
            if layer_index in linear_attention_initial_ssm_states_vllm:
                linear_attention_initial_ssm_states_vllm[layer_index].copy_(
                    linear_attention_ssm_states[layer_index].unsqueeze(0).transpose(-1, -2).contiguous()
                )
            if layer_index in linear_attention_packed_initial_ssm_states:
                packed_initial = linear_attention_packed_initial_ssm_states[layer_index]
                packed_initial.zero_()
                packed_initial[1].copy_(linear_attention_ssm_states[layer_index].transpose(-1, -2).contiguous())

    def promote_prefill_layer_state_output_to_input(layer_index: int) -> None:
        entry = layer_weights[layer_index]
        if not (
            linear_attention_enabled(attention_mode)
            and entry["layer_type"] == "linear_attention"
        ):
            return
        if linear_attention_conv_state_refswap:
            (
                linear_attention_initial_conv_states[layer_index],
                linear_attention_conv_states[layer_index],
            ) = (
                linear_attention_conv_states[layer_index],
                linear_attention_initial_conv_states[layer_index],
            )
        else:
            linear_attention_initial_conv_states[layer_index].copy_(
                linear_attention_conv_states[layer_index]
            )
        if use_prefill_vllm_state_handoff() and layer_index in linear_attention_ssm_states_vllm:
            if linear_attention_uses_native_vllm_decode_state_refswap(linear_attention_variant):
                linear_attention_initial_ssm_states_vllm[layer_index] = (
                    linear_attention_ssm_states_vllm[layer_index]
                )
            else:
                linear_attention_initial_ssm_states_vllm[layer_index].copy_(
                    linear_attention_ssm_states_vllm[layer_index]
                )
            return
        linear_attention_initial_ssm_states[layer_index].copy_(
            linear_attention_ssm_states[layer_index]
        )
        if layer_index in linear_attention_initial_ssm_states_vllm:
            linear_attention_initial_ssm_states_vllm[layer_index].copy_(
                linear_attention_ssm_states[layer_index]
                .unsqueeze(0)
                .transpose(-1, -2)
                .contiguous()
            )
        if layer_index in linear_attention_packed_initial_ssm_states:
            packed_initial = linear_attention_packed_initial_ssm_states[layer_index]
            packed_initial.zero_()
            packed_initial[1].copy_(
                linear_attention_ssm_states[layer_index].transpose(-1, -2).contiguous()
            )

    def select_decode_token(logits: Any) -> Any:
        if decode_sampling == "argmax":
            return torch.argmax(logits.float(), dim=-1).to(torch.long)
        logits_1d = logits.float().reshape(-1)
        top_count = min(sampling_top_k, int(logits_1d.numel()))
        top_values, top_indices = torch.topk(logits_1d, k=top_count, dim=-1)
        probs = torch.softmax(top_values / sampling_temperature, dim=-1)
        sampled_index = torch.multinomial(probs, num_samples=1)
        return top_indices.index_select(0, sampled_index).to(torch.long)

    resident_logits_for_decode = None

    def tensor_stat_digest(tensor: Any) -> dict[str, Any]:
        data = tensor.detach().float()
        numel = int(data.numel())
        digest: dict[str, Any] = {
            "shape": [int(item) for item in tensor.shape],
            "dtype": str(tensor.dtype).replace("torch.", ""),
            "numel": numel,
        }
        if numel == 0:
            digest.update({"finite": True, "sum": None, "abs_sum": None, "mean": None, "max_abs": None})
            return digest
        finite = torch.isfinite(data)
        safe = torch.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        abs_safe = safe.abs()
        digest.update(
            {
                "finite": bool(finite.all().item()),
                "sum": float(safe.sum().item()),
                "abs_sum": float(abs_safe.sum().item()),
                "mean": float(safe.mean().item()),
                "max_abs": float(abs_safe.max().item()),
            }
        )
        return digest

    def linear_attention_state_digest() -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for layer_index, entry in enumerate(layer_weights):
            if not (linear_attention_enabled(attention_mode) and entry["layer_type"] == "linear_attention"):
                continue
            record: dict[str, Any] = {
                "layer_index": layer_index,
                "model_layer": int(entry["layer"]),
            }
            state_sources = [
                ("initial_conv", linear_attention_initial_conv_states),
                ("current_conv", linear_attention_conv_states),
                ("initial_ssm", linear_attention_initial_ssm_states),
                ("current_ssm", linear_attention_ssm_states),
                ("initial_vllm_ssm", linear_attention_initial_ssm_states_vllm),
                ("current_vllm_ssm", linear_attention_ssm_states_vllm),
            ]
            for name, source in state_sources:
                tensor = source.get(layer_index)
                if tensor is not None:
                    record[name] = tensor_stat_digest(tensor)
            records.append(record)
        return records

    def logits_topk_digest(logits: Any) -> dict[str, Any]:
        logits_1d = logits.detach().float().reshape(-1)
        top_count = min(logit_topk, int(logits_1d.numel()))
        values, indices = torch.topk(logits_1d, k=top_count, dim=-1)
        return {
            "topk": top_count,
            "topk_token_ids": [int(item) for item in indices.cpu().tolist()],
            "topk_logits": [float(item) for item in values.cpu().tolist()],
        }

    def run_decode_loop(steps: int) -> dict[str, Any] | None:
        if steps <= 0:
            return None
        if input_token_ids is None:
            raise RuntimeError("decode loop requires a seed input token")
        original_mode = mode
        original_tokens = tokens
        original_x = x
        original_position_start = position_start
        original_position_tokens = tokens
        generated_token_ids: list[int] = []
        visible_generated_token_ids: list[int] = []
        pending_generated_token_tensors: list[Any] = []
        requested_steps = steps
        model_steps = 0
        seed_token_id = None
        seed_token_text = None
        seed_source = "decode_input_token"
        prefill_state_source = None
        stop_reason = "length"
        stopped_on_token_id = None
        initial_context_len = seq_len
        current_hidden = x
        current_decode_token_id = int(input_token_ids[-1]) if input_token_ids else None
        diagnostic_records: list[dict[str, Any]] = []
        token_cpu_sync_interval = decode_loop_token_cpu_sync_interval
        batch_token_cpu_sync = token_cpu_sync_interval != 1

        def append_diagnostic_record(
            *,
            stage: str,
            step: int | None,
            position: int,
            input_token_id: int | None,
            selected_token_id: int | None,
            logits: Any,
        ) -> None:
            if not decode_loop_diagnostic:
                return
            diagnostic_records.append(
                {
                    "stage": stage,
                    "step": step,
                    "position": int(position),
                    "input_token_id": input_token_id,
                    "selected_token_id": selected_token_id,
                    "logits": logits_topk_digest(logits),
                    "linear_attention_states": linear_attention_state_digest(),
                }
            )

        def flush_pending_generated_tokens() -> bool:
            nonlocal stop_reason, stopped_on_token_id
            if not pending_generated_token_tensors:
                return False
            pending_token_ids = [
                int(item)
                for item in torch.cat(pending_generated_token_tensors).reshape(-1).to(torch.long).cpu().tolist()
            ]
            pending_generated_token_tensors.clear()
            for token_id in pending_token_ids:
                generated_token_ids.append(token_id)
                if token_id in decode_stop_token_ids:
                    stop_reason = "stop_token"
                    stopped_on_token_id = token_id
                    return True
                visible_generated_token_ids.append(token_id)
            return False

        try:
            if original_mode == "prefill":
                if not collect_resident_stage_timeline and resident_logits_for_decode is not None:
                    set_runtime_context("prefill", original_tokens, original_x)
                    set_position_window(0, original_tokens)
                    prefill_logits = resident_logits_for_decode
                    prefill_state_source = (
                        "exact_token_prefix_cache"
                        if exact_prefix_cache_exact_hit
                        else (
                            "strict_token_prefix_cache_plus_suffix_prefill"
                            if exact_prefix_cache_strict_prefix_hit
                            else "measured_resident_prefill"
                        )
                    )
                else:
                    set_runtime_context("prefill", original_tokens, original_x)
                    set_position_window(0, original_tokens)
                    prefill_state_out = run_engine("resident", original_x)
                    prefill_logits = final_logits(prefill_state_out)
                    prefill_state_source = "rerun_prefill"
                seed_token = select_decode_token(prefill_logits)
                seed_token_id = int(seed_token.item())
                if decode_output_token:
                    seed_token_text = decode_token_ids(model_dir, [seed_token_id])[0]
                promote_decode_state_outputs_to_inputs()
                append_diagnostic_record(
                    stage="prefill_seed",
                    step=None,
                    position=original_tokens,
                    input_token_id=current_decode_token_id,
                    selected_token_id=seed_token_id,
                    logits=prefill_logits,
                )
                current_hidden = global_tensors["embed_tokens"].index_select(0, seed_token)
                initial_context_len = original_tokens
                current_decode_token_id = seed_token_id
                seed_source = "prefill_top1" if decode_sampling == "argmax" else "prefill_top_k_sample"
                if seed_token_id in decode_stop_token_ids:
                    stop_reason = "stop_token"
                    stopped_on_token_id = seed_token_id
                    steps = 0

            ensure_native_moe_decode_layouts()
            set_runtime_context("decode", 1, current_hidden)
            ensure_decode_buffers()
            sync()
            start = time.perf_counter()
            decode_cos_cache = None
            decode_sin_cache = None
            next_hidden_buffer = None
            if decode_loop_fast_housekeeping and steps > 0:
                decode_cos_cache, decode_sin_cache = build_decode_rotary_cache(initial_context_len, steps)
                next_hidden_buffer = torch.empty_like(current_hidden)
            for step in range(steps):
                if decode_cos_cache is not None and decode_sin_cache is not None:
                    set_decode_position_window_from_cache(initial_context_len + step, step, decode_cos_cache, decode_sin_cache)
                else:
                    set_position_window(initial_context_len + step, 1)
                hidden_out = run_engine("resident", current_hidden)
                state_promotion_started = False
                state_ready = None
                if decode_state_promotion_stream is not None:
                    state_ready = torch.cuda.Event()
                    state_ready.record(torch.cuda.current_stream())
                logits = final_logits(hidden_out)
                if decode_state_promotion_stream is not None and state_ready is not None:
                    with torch.cuda.stream(decode_state_promotion_stream):
                        decode_state_promotion_stream.wait_event(state_ready)
                        promote_decode_state_outputs_to_inputs()
                    state_promotion_started = True
                next_token = select_decode_token(logits)
                next_token_id_for_diagnostic = int(next_token.item()) if decode_loop_diagnostic else None
                append_diagnostic_record(
                    stage="decode_step",
                    step=step,
                    position=initial_context_len + step,
                    input_token_id=current_decode_token_id,
                    selected_token_id=next_token_id_for_diagnostic,
                    logits=logits,
                )
                model_steps += 1
                if batch_token_cpu_sync:
                    pending_generated_token_tensors.append(next_token.reshape(1))
                    if token_cpu_sync_interval > 0 and len(pending_generated_token_tensors) >= token_cpu_sync_interval:
                        if flush_pending_generated_tokens():
                            break
                else:
                    next_token_id = (
                        next_token_id_for_diagnostic
                        if next_token_id_for_diagnostic is not None
                        else int(next_token.item())
                    )
                    generated_token_ids.append(next_token_id)
                    if next_token_id in decode_stop_token_ids:
                        stop_reason = "stop_token"
                        stopped_on_token_id = next_token_id
                        break
                    visible_generated_token_ids.append(next_token_id)
                if state_promotion_started:
                    torch.cuda.current_stream().wait_stream(decode_state_promotion_stream)
                else:
                    promote_decode_state_outputs_to_inputs()
                if next_hidden_buffer is not None:
                    torch.index_select(global_tensors["embed_tokens"], 0, next_token, out=next_hidden_buffer)
                    current_hidden, next_hidden_buffer = next_hidden_buffer, current_hidden
                else:
                    current_hidden = global_tensors["embed_tokens"].index_select(0, next_token)
                if decode_loop_diagnostic:
                    current_decode_token_id = next_token_id_for_diagnostic
            if batch_token_cpu_sync:
                flush_pending_generated_tokens()
            sync()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
        finally:
            set_runtime_context(original_mode, original_tokens, original_x)
            set_position_window(original_position_start, original_position_tokens)
        actual_steps = len(generated_token_ids)
        generated_text = decode_token_ids(model_dir, generated_token_ids) if decode_output_token else None
        generated_text_joined = decode_token_sequence(model_dir, generated_token_ids) if decode_output_token else None
        visible_generated_text = (
            decode_token_ids(model_dir, visible_generated_token_ids) if decode_output_token else None
        )
        visible_generated_text_joined = (
            decode_token_sequence(model_dir, visible_generated_token_ids) if decode_output_token else None
        )
        prefill_state_reused = original_mode == "prefill"
        return {
            "requested_steps": requested_steps,
            "steps": actual_steps,
            "model_steps": model_steps,
            "visible_steps": len(visible_generated_token_ids),
            "initial_context_len": initial_context_len,
            "final_context_len": initial_context_len + actual_steps,
            "elapsed_ms": elapsed_ms,
            "ms_per_token": elapsed_ms / actual_steps if actual_steps > 0 else None,
            "tokens_per_s": actual_steps * 1000.0 / elapsed_ms if elapsed_ms > 0 else None,
            "stop_reason": stop_reason,
            "stop_token_ids": sorted(decode_stop_token_ids),
            "stopped_on_token_id": stopped_on_token_id,
            "seed_source": seed_source,
            "prefill_state_source": prefill_state_source,
            "prefill_seed_token_id": seed_token_id,
            "prefill_seed_token_text": seed_token_text,
            "generated_token_ids": generated_token_ids,
            "visible_generated_token_ids": visible_generated_token_ids,
            "generated_token_text": generated_text,
            "generated_text": generated_text_joined,
            "visible_generated_token_text": visible_generated_text,
            "visible_generated_text": visible_generated_text_joined,
            "generated_token_ids_sha256": token_ids_digest(generated_token_ids),
            "visible_generated_token_ids_sha256": token_ids_digest(visible_generated_token_ids),
            "decode_sampling": decode_sampling,
            "sampling_temperature": sampling_temperature,
            "sampling_top_k": sampling_top_k,
            "fast_housekeeping": decode_loop_fast_housekeeping,
            "defer_token_cpu_sync": decode_loop_defer_token_cpu_sync,
            "token_cpu_sync_interval": token_cpu_sync_interval,
            "overlap_state_promotion_lm_head": decode_state_promotion_stream is not None,
            "diagnostic": (
                {
                    "enabled": True,
                    "scope": (
                        "per-step top-k logits plus linear-attention conv/SSM state "
                        "statistics; timing includes diagnostic reductions and CPU syncs"
                    ),
                    "records": diagnostic_records,
                }
                if decode_loop_diagnostic
                else None
            ),
            "state_scope": (
                "real prompt prefill KV/linear state seeds this loop"
                if prefill_state_reused
                else (
                    "synthetic initial decode KV/linear state; generated tokens advance "
                    "full-attention KV and linear-attention conv/SSM state within this loop"
                )
            ),
            "timing_scope": (
                "resident pipeline plus final norm, LM-head, argmax, next-token embedding lookup, "
                "state promotion, and generated-token CPU materialization"
            ),
            "prefill_state_reused": prefill_state_reused,
        }

    measurement_wall_start = time.perf_counter()
    expanded_out = None
    expanded_rec = None
    if measurement_mode == "correctness":
        expanded_out, expanded_rec = timed("expanded_four_layer_pipeline", lambda: run_engine("expanded"))
    full_request_tokens = int(tokens)
    full_request_x = x
    canonical_prefill_chunk_tokens = 8192
    canonical_state_checkpoint_tokens = (
        full_request_tokens
        - (full_request_tokens % canonical_prefill_chunk_tokens)
    )
    canonical_state_checkpoint_conv_states: dict[int, Any] = {}
    canonical_state_checkpoint_ssm_states: dict[int, Any] = {}
    canonical_chunked_cold_prefill_record: dict[str, Any] = {
        "enabled": False,
        "chunk_tokens": canonical_prefill_chunk_tokens,
        "request_tokens": full_request_tokens,
        "chunks": [],
        "state_promotion_after_each_chunk": True,
        "operator_shape_contract": "fixed_1024_except_final_remainder",
    }
    canonical_chunked_strict_suffix_record: dict[str, Any] = {
        "enabled": False,
        "chunk_tokens": canonical_prefill_chunk_tokens,
        "request_tokens": full_request_tokens,
        "chunks": [],
        "state_promotion_after_each_chunk": True,
        "operator_shape_contract": "fixed_1024_except_final_remainder",
    }
    exact_prefix_cache_state_hit = False
    exact_prefix_cache_exact_hit = False
    exact_prefix_cache_strict_prefix_hit = False
    exact_prefix_restore_record: dict[str, Any] = {
        "restored": False,
        "status": "cache_miss_or_disabled",
        "restored_bytes": 0,
    }

    def run_canonical_fixed_1k_chunked_cold_prefill() -> tuple[Any, dict[str, Any]]:
        if warmup != 0 or iters != 1:
            raise RuntimeError(
                "canonical fixed-1k stateful cold prefill requires warmup=0 and iters=1"
            )
        chunks: list[dict[str, Any]] = []
        chunk_out = None
        sync()
        started = time.perf_counter()
        try:
            for chunk_start in range(0, full_request_tokens, canonical_prefill_chunk_tokens):
                chunk_end = min(
                    chunk_start + canonical_prefill_chunk_tokens,
                    full_request_tokens,
                )
                chunk_count = chunk_end - chunk_start
                chunk_x = full_request_x[chunk_start:chunk_end]
                set_position_window(chunk_start, chunk_count)
                set_runtime_context("prefill", chunk_count, chunk_x)
                chunk_started = time.perf_counter()
                chunk_out = run_engine("resident")
                promote_decode_state_outputs_to_inputs()
                sync()
                chunks.append(
                    {
                        "start": chunk_start,
                        "end": chunk_end,
                        "tokens": chunk_count,
                        "state_promoted": True,
                        "elapsed_ms": (time.perf_counter() - chunk_started) * 1000.0,
                    }
                )
        finally:
            set_runtime_context("prefill", full_request_tokens, full_request_x)
            set_position_window(0, full_request_tokens)
        if chunk_out is None:
            raise RuntimeError("canonical fixed-1k cold prefill produced no chunks")
        sync()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        canonical_chunked_cold_prefill_record.update(
            {
                "enabled": True,
                "status": "complete",
                "chunks": chunks,
                "chunk_count": len(chunks),
                "call_partitions": [int(item["tokens"]) for item in chunks],
                "elapsed_ms": elapsed_ms,
            }
        )
        return chunk_out, {
            "name": "resident_four_layer_pipeline_canonical_fixed_1k_chunked_cold_prefill",
            "elapsed_ms": elapsed_ms,
            "iters": 1,
            "ms_per_iter": elapsed_ms,
        }

    def run_layer_major_fixed_1k_chunked_cold_prefill() -> tuple[Any, dict[str, Any]]:
        if warmup != 0 or iters != 1:
            raise RuntimeError(
                "layer-major fixed-1k stateful cold prefill requires warmup=0 and iters=1"
            )
        ranges = [
            (start, min(start + canonical_prefill_chunk_tokens, full_request_tokens))
            for start in range(0, full_request_tokens, canonical_prefill_chunk_tokens)
        ]
        hidden_chunks = [full_request_x[start:end] for start, end in ranges]
        chunks = [
            {
                "start": start,
                "end": end,
                "tokens": end - start,
                "state_promoted": True,
                "elapsed_ms": 0.0,
                "timing_scope": "layer-major total only",
            }
            for start, end in ranges
        ]
        state_promotions = 0
        sync()
        started = time.perf_counter()
        try:
            for layer_index in range(len(layer_weights)):
                for chunk_index, (chunk_start, chunk_end) in enumerate(ranges):
                    chunk_count = chunk_end - chunk_start
                    chunk_hidden = hidden_chunks[chunk_index]
                    set_position_window(chunk_start, chunk_count)
                    set_runtime_context("prefill", chunk_count, chunk_hidden)
                    hidden_chunks[chunk_index], _ = layer_body(
                        chunk_hidden, layer_index, "resident"
                    )
                    promote_prefill_layer_state_output_to_input(layer_index)
                    if layer_weights[layer_index]["layer_type"] == "linear_attention":
                        state_promotions += 1
                        if chunk_end == canonical_state_checkpoint_tokens:
                            canonical_state_checkpoint_conv_states[layer_index] = (
                                linear_attention_conv_states[layer_index].detach().clone()
                            )
                            canonical_state_checkpoint_ssm_states[layer_index] = (
                                linear_attention_ssm_states[layer_index].detach().clone()
                            )
        finally:
            set_runtime_context("prefill", full_request_tokens, full_request_x)
            set_position_window(0, full_request_tokens)
        sync()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        canonical_chunked_cold_prefill_record.update(
            {
                "enabled": True,
                "status": "complete",
                "schedule": "layer-major across fixed q8192 chunks",
                "chunks": chunks,
                "chunk_count": len(chunks),
                "call_partitions": [int(item["tokens"]) for item in chunks],
                "state_promotions": state_promotions,
                "elapsed_ms": elapsed_ms,
            }
        )
        return hidden_chunks[-1], {
            "name": "resident_pipeline_layer_major_fixed_q8192_chunked_cold_prefill",
            "elapsed_ms": elapsed_ms,
            "iters": 1,
            "ms_per_iter": elapsed_ms,
        }

    def run_canonical_fixed_1k_chunked_strict_suffix_prefill(
        matched_tokens: int,
    ) -> tuple[Any, dict[str, Any]]:
        if warmup != 0 or iters != 1:
            raise RuntimeError(
                "canonical fixed-1k stateful strict suffix prefill requires warmup=0 and iters=1"
            )
        chunks: list[dict[str, Any]] = []
        chunk_out = None
        sync()
        started = time.perf_counter()
        try:
            for chunk_start in range(
                matched_tokens,
                full_request_tokens,
                canonical_prefill_chunk_tokens,
            ):
                chunk_end = min(
                    chunk_start + canonical_prefill_chunk_tokens,
                    full_request_tokens,
                )
                chunk_count = chunk_end - chunk_start
                chunk_x = full_request_x[chunk_start:chunk_end]
                set_position_window(chunk_start, chunk_count)
                set_runtime_context("prefill", chunk_count, chunk_x)
                chunk_started = time.perf_counter()
                chunk_out = run_engine("resident")
                promote_decode_state_outputs_to_inputs()
                sync()
                chunks.append(
                    {
                        "start": chunk_start,
                        "end": chunk_end,
                        "tokens": chunk_count,
                        "state_promoted": True,
                        "elapsed_ms": (time.perf_counter() - chunk_started) * 1000.0,
                    }
                )
        finally:
            set_runtime_context("prefill", full_request_tokens, full_request_x)
            set_position_window(0, full_request_tokens)
        if chunk_out is None:
            raise RuntimeError("canonical fixed-1k strict suffix prefill produced no chunks")
        sync()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        canonical_chunked_strict_suffix_record.update(
            {
                "enabled": True,
                "status": "complete",
                "matched_tokens": matched_tokens,
                "suffix_tokens": full_request_tokens - matched_tokens,
                "chunks": chunks,
                "chunk_count": len(chunks),
                "call_partitions": [int(item["tokens"]) for item in chunks],
                "elapsed_ms": elapsed_ms,
            }
        )
        return chunk_out, {
            "name": "resident_four_layer_pipeline_canonical_fixed_1k_chunked_strict_suffix_prefill",
            "elapsed_ms": elapsed_ms,
            "iters": 1,
            "ms_per_iter": elapsed_ms,
        }

    if exact_prefix_cache_entry is not None:
        matched_tokens = int(exact_prefix_cache_record.get("matched_tokens") or 0)
        strict_prefix_hit = 0 < matched_tokens < full_request_tokens
        novel_suffix_tokens = full_request_tokens - matched_tokens
        direct_full_state_reuse = strict_prefix_hit and (
            matched_tokens % canonical_prefill_chunk_tokens == 0
            or novel_suffix_tokens <= 1
        )
        restore_entry = exact_prefix_cache_entry
        state_reuse_tokens = matched_tokens
        replayed_prefix_tokens = 0
        replay_policy = "full_cached_state"
        if strict_prefix_hit and not direct_full_state_reuse:
            checkpoint = exact_prefix_cache_entry.get("aligned_state_checkpoint")
            checkpoint_tokens = (
                int(checkpoint.get("prompt_tokens") or 0)
                if isinstance(checkpoint, dict)
                else 0
            )
            if 0 < checkpoint_tokens <= matched_tokens:
                restore_entry = checkpoint
                state_reuse_tokens = checkpoint_tokens
                replayed_prefix_tokens = matched_tokens - checkpoint_tokens
                replay_policy = "aligned_checkpoint_plus_cached_tail"
            else:
                if exact_prefix_cache_entry_key is not None:
                    _ENGINE_EXACT_PREFIX_CACHE.pop(exact_prefix_cache_entry_key, None)
                exact_prefix_cache_entry = None
                exact_prefix_cache_record.update(
                    {
                        "lookup": "prefix_entry_rejected_missing_aligned_state_checkpoint",
                        "hit": False,
                        "match_kind": "none",
                        "matched_tokens": 0,
                        "suffix_tokens": full_request_tokens,
                    }
                )
        if exact_prefix_cache_entry is not None:
            exact_prefix_restore_record = restore_exact_prefix_state(
                restore_entry,
                for_suffix_prefill=strict_prefix_hit,
            )
            if exact_prefix_restore_record.get("restored") is True:
                exact_prefix_cache_state_hit = True
                if strict_prefix_hit:
                    exact_prefix_cache_strict_prefix_hit = True
                    state_prefill_tokens = full_request_tokens - state_reuse_tokens
                    if state_prefill_tokens > canonical_prefill_chunk_tokens:
                        resident_out, resident_rec = run_canonical_fixed_1k_chunked_strict_suffix_prefill(
                            state_reuse_tokens
                        )
                    else:
                        suffix_x = global_tensors["embed_tokens"].index_select(
                            0,
                            token_tensor[state_reuse_tokens:full_request_tokens],
                        )
                        try:
                            set_position_window(state_reuse_tokens, state_prefill_tokens)
                            set_runtime_context("prefill", state_prefill_tokens, suffix_x)
                            resident_out, resident_rec = timed(
                                (
                                    "resident_four_layer_pipeline_strict_token_prefix_suffix_prefill"
                                    if direct_full_state_reuse
                                    else "resident_pipeline_q8192_checkpoint_tail_replay_plus_novel_suffix_prefill"
                                ),
                                lambda: run_engine("resident"),
                            )
                            promote_decode_state_outputs_to_inputs()
                        finally:
                            set_runtime_context("prefill", full_request_tokens, full_request_x)
                            set_position_window(0, full_request_tokens)
                    suffix_pipeline_ms = float(resident_rec.get("ms_per_iter") or 0.0)
                    exact_prefix_cache_record.update(
                        {
                            "lookup": (
                                "strict_prefix_hit_state_restored_suffix_prefill_complete"
                                if direct_full_state_reuse
                                else "strict_prefix_hit_aligned_checkpoint_restored_tail_replayed_suffix_prefill_complete"
                            ),
                            "hit": True,
                            "match_kind": "strict_prefix_hit",
                            "state_source": (
                                "strict_token_prefix_cache_plus_suffix_prefill"
                                if direct_full_state_reuse
                                else "q8192_aligned_checkpoint_plus_cached_tail_replay_plus_novel_suffix"
                            ),
                            "matched_tokens": matched_tokens,
                            "suffix_tokens": novel_suffix_tokens,
                            "state_reuse_tokens": state_reuse_tokens,
                            "replayed_prefix_tokens": replayed_prefix_tokens,
                            "state_prefill_tokens": state_prefill_tokens,
                            "replay_policy": replay_policy,
                            "direct_full_state_reuse": direct_full_state_reuse,
                            "restore": exact_prefix_restore_record,
                            "suffix_pipeline_ms": suffix_pipeline_ms,
                            "prefix_restore_plus_suffix_prefill_ms": (
                                float(exact_prefix_restore_record.get("wall_time_ms") or 0.0)
                                + suffix_pipeline_ms
                            ),
                        }
                    )
                else:
                    exact_prefix_cache_exact_hit = True
                    resident_out = exact_prefix_cache_entry["resident_out"]
                    restore_ms = float(exact_prefix_restore_record.get("wall_time_ms") or 0.0)
                    resident_rec = {
                        "name": "resident_four_layer_pipeline_exact_token_prefix_cache_hit",
                        "elapsed_ms": restore_ms,
                        "iters": 1,
                        "ms_per_iter": restore_ms,
                    }
                    exact_prefix_cache_record.update(
                        {
                            "lookup": "exact_hit_state_restored",
                            "hit": True,
                            "match_kind": "exact_hit",
                            "state_source": "exact_token_prefix_cache",
                            "restore": exact_prefix_restore_record,
                        }
                    )
            else:
                if exact_prefix_cache_entry_key is not None:
                    _ENGINE_EXACT_PREFIX_CACHE.pop(exact_prefix_cache_entry_key, None)
                exact_prefix_cache_entry = None
                exact_prefix_cache_record.update(
                    {
                        "lookup": "prefix_entry_rejected_restore_contract",
                        "hit": False,
                        "match_kind": "none",
                        "matched_tokens": 0,
                        "suffix_tokens": full_request_tokens,
                        "restore": exact_prefix_restore_record,
                    }
                )
    if not exact_prefix_cache_state_hit:
        if (
            exact_prefix_cache
            and full_request_tokens > canonical_prefill_chunk_tokens
            and not use_q8192_compound_provider()
        ):
            resident_out, resident_rec = run_layer_major_fixed_1k_chunked_cold_prefill()
        else:
            pipeline_name = (
                "resident_full_q8192_compound_provider_pipeline"
                if use_q8192_compound_provider()
                else "resident_four_layer_pipeline"
            )
            resident_out, resident_rec = timed(pipeline_name, lambda: run_engine("resident"))

    resident_stage_timeline: list[dict[str, Any]] = []
    moe_substage_timeline: list[dict[str, Any]] = []
    moe_overlap_event_timeline: list[dict[str, Any]] = []
    resident_layer_metadata: list[dict[str, Any]] = []
    resident_graph_out = None
    resident_graph_rec = None
    if cuda_graph_replay_timing:
        resident_graph_out, resident_graph_rec = cuda_graph_timed(
            "resident_four_layer_pipeline_cuda_graph_replay",
            lambda: run_engine("resident"),
        )
    if collect_resident_stage_timeline or moe_substage_timing or moe_overlap_event_timing:
        stage_hidden = x
        for layer_index, entry in enumerate(layer_weights):
            weights = entry["tensors"]
            h1, rec = timed(
                f"layer{entry['layer']}_input_rmsnorm",
                lambda weights=weights, layer_index=layer_index: rmsnorm(
                    stage_hidden,
                    weights["input_layernorm"],
                    rmsnorm_input_outputs.get(layer_index),
                ),
            )
            resident_stage_timeline.append(rec)
            if full_attention_enabled(attention_mode) and entry["layer_type"] == "full_attention":
                attention_stage_name = f"layer{entry['layer']}_full_attention"
            elif linear_attention_enabled(attention_mode) and entry["layer_type"] == "linear_attention":
                attention_stage_name = f"layer{entry['layer']}_linear_attention_{linear_attention_variant}"
            else:
                attention_stage_name = f"layer{entry['layer']}_attention_stub"
            attn_out, rec = timed(
                attention_stage_name,
                lambda weights=weights, layer_index=layer_index: attention(h1, layer_index, weights),
            )
            resident_stage_timeline.append(rec)
            after_attn, rec = timed(f"layer{entry['layer']}_attention_residual", lambda: stage_hidden + attn_out)
            resident_stage_timeline.append(rec)
            h2, rec = timed(
                f"layer{entry['layer']}_post_attention_rmsnorm",
                lambda weights=weights, layer_index=layer_index: rmsnorm(
                    after_attn,
                    weights["post_attention_layernorm"],
                    rmsnorm_post_outputs.get(layer_index),
                ),
            )
            resident_stage_timeline.append(rec)
            moe_out_includes_shared = False
            shared_router_overlap_profiled = should_overlap_shared_expert_router_moe()
            if shared_router_overlap_profiled:
                if moe_substage_timing:
                    profile_scores, profile_indices = router_topk(h2, layer_index, weights)
                    profiled_vllm_fused_routed_moe(
                        h2,
                        profile_scores,
                        profile_indices,
                        weights,
                        layer_index,
                        moe_substage_timeline,
                    )
                if moe_overlap_event_timing:
                    profile_router_shared_overlap_events(
                        h2,
                        layer_index,
                        weights,
                        moe_overlap_event_timeline,
                    )

                def router_moe_shared_overlap(
                    weights: dict[str, Any] = weights,
                    layer_index: int = layer_index,
                ) -> tuple[Any, Any, Any]:
                    profile_shared_out = start_shared_expert_overlap(h2, layer_index, weights)
                    profile_scores, profile_indices = router_topk(h2, layer_index, weights)
                    profile_moe_out = vllm_fused_routed_moe(
                        h2,
                        profile_scores,
                        profile_indices,
                        weights,
                        layer_index=layer_index,
                    )
                    profile_shared_out = finish_shared_expert_overlap(profile_shared_out)
                    return profile_moe_out + profile_shared_out, profile_scores, profile_indices

                (moe_out, scores, indices), rec = timed(
                    f"layer{entry['layer']}_{moe_variant}_router_shared_expert_overlap",
                    router_moe_shared_overlap,
                )
                resident_stage_timeline.append(rec)
                moe_out_includes_shared = True
            else:
                (scores, indices), rec = timed(
                    f"layer{entry['layer']}_router_topk",
                    lambda weights=weights, layer_index=layer_index: router_topk(h2, layer_index, weights),
                )
                resident_stage_timeline.append(rec)
            dispatch = None
            if moe_variant not in vllm_fused_moe_variants():
                dispatch, rec = timed(
                    f"layer{entry['layer']}_build_dispatch_table",
                    lambda: build_dispatch_table(indices),
                )
                resident_stage_timeline.append(rec)
            candidate_dispatch = None
            shared_out = None
            if include_shared_expert and moe_variant == "vllm_fused_inplace":
                shared_out, rec = timed(
                    f"layer{entry['layer']}_shared_expert_pre_inplace_moe",
                    lambda weights=weights, layer_index=layer_index: shared_expert(h2, layer_index, weights),
                )
                resident_stage_timeline.append(rec)
            if shared_router_overlap_profiled:
                pass
            elif moe_variant == "padded_batched":
                candidate_dispatch, rec = timed(
                    f"layer{entry['layer']}_build_padded_dispatch_table",
                    lambda: build_padded_dispatch_table(dispatch),
                )
                resident_stage_timeline.append(rec)
                moe_out, rec = timed(
                    f"layer{entry['layer']}_padded_batched_routed_moe",
                    lambda weights=weights, layer_index=layer_index: padded_batched_routed_moe(
                        h2,
                        scores,
                        candidate_dispatch,
                        weights,
                        resident_workspaces[layer_index],
                    ),
                )
            elif moe_variant == "count_batched":
                candidate_dispatch, rec = timed(
                    f"layer{entry['layer']}_build_count_batched_dispatch_table",
                    lambda: build_count_batched_dispatch_table(dispatch),
                )
                resident_stage_timeline.append(rec)
                moe_out, rec = timed(
                    f"layer{entry['layer']}_count_batched_routed_moe",
                    lambda weights=weights, layer_index=layer_index: count_batched_routed_moe(
                        h2,
                        scores,
                        candidate_dispatch,
                        weights,
                        resident_workspaces[layer_index],
                    ),
                )
            elif moe_variant in vllm_fused_moe_variants():
                if moe_variant == "vllm_fused_inplace":
                    def inplace_moe_from_reset(
                        weights: dict[str, Any] = weights,
                        layer_index: int = layer_index,
                    ) -> Any:
                        moe_input = resident_workspaces[layer_index]
                        moe_input.copy_(h2)
                        return vllm_fused_routed_moe(
                            moe_input,
                            scores,
                            indices,
                            weights,
                            layer_index=layer_index,
                            inplace=True,
                        )

                    moe_out, rec = timed(
                        f"layer{entry['layer']}_vllm_fused_inplace_routed_moe_input_reset",
                        inplace_moe_from_reset,
                    )
                else:
                    if moe_substage_timing:
                        profiled_vllm_fused_routed_moe(
                            h2,
                            scores,
                            indices,
                            weights,
                            layer_index,
                            moe_substage_timeline,
                        )
                    moe_stage_name = (
                        f"layer{entry['layer']}_{moe_variant}_routed_moe"
                        if moe_variant != "vllm_fused"
                        else f"layer{entry['layer']}_vllm_fused_routed_moe"
                    )
                    if should_overlap_shared_expert_moe():
                        moe_out, rec = timed(
                            f"layer{entry['layer']}_{moe_variant}_shared_expert_overlap",
                            lambda weights=weights, layer_index=layer_index: vllm_fused_moe_with_shared_overlap(
                                h2,
                                scores,
                                indices,
                                layer_index,
                                weights,
                            ),
                        )
                        moe_out_includes_shared = True
                    else:
                        moe_out, rec = timed(
                            moe_stage_name,
                            lambda weights=weights, layer_index=layer_index: vllm_fused_routed_moe(
                                h2,
                                scores,
                                indices,
                                weights,
                                layer_index=layer_index,
                            ),
                        )
            else:
                moe_out, rec = timed(
                    f"layer{entry['layer']}_resident_routed_moe",
                    lambda weights=weights, layer_index=layer_index: resident_routed_moe(
                        h2, scores, dispatch, weights, resident_workspaces[layer_index]
                    ),
                )
            if not shared_router_overlap_profiled:
                resident_stage_timeline.append(rec)
            if include_shared_expert and not moe_out_includes_shared:
                if shared_out is None:
                    shared_out, rec = timed(
                        f"layer{entry['layer']}_shared_expert",
                        lambda weights=weights, layer_index=layer_index: shared_expert(h2, layer_index, weights),
                    )
                    resident_stage_timeline.append(rec)
                moe_out = moe_out + shared_out
            stage_hidden, rec = timed(f"layer{entry['layer']}_moe_residual", lambda: after_attn + moe_out)
            resident_stage_timeline.append(rec)
            resident_layer_metadata.append(
                {
                    "layer": entry["layer"],
                    "layer_type": entry["layer_type"],
                    "attention": (
                        "full_attention"
                        if full_attention_enabled(attention_mode) and entry["layer_type"] == "full_attention"
                        else (
                            "linear_attention"
                            if linear_attention_enabled(attention_mode) and entry["layer_type"] == "linear_attention"
                            else "stub"
                        )
                    ),
                    "populated_experts": dispatch["populated_experts"] if dispatch is not None else None,
                    "active_rows": dispatch["active_rows"] if dispatch is not None else int(indices.numel()),
                    "moe_variant": moe_variant,
                    "backend": "vllm.fused_experts" if moe_variant in vllm_fused_moe_variants() else None,
                    "inplace": moe_variant == "vllm_fused_inplace" if moe_variant.startswith("vllm_fused") else None,
                    "override_config": vllm_moe_override_config_for_layer_index(layer_index),
                    "row_count_group_count": (
                        candidate_dispatch.get("row_count_group_count") if candidate_dispatch is not None else None
                    ),
                    "max_segments_per_count": (
                        candidate_dispatch.get("max_segments_per_count") if candidate_dispatch is not None else None
                    ),
                    "max_rows_per_expert": (
                        candidate_dispatch.get("max_rows_per_expert") if candidate_dispatch is not None else None
                    ),
                    "padded_rows": candidate_dispatch["padded_rows"] if candidate_dispatch is not None else None,
                    "padding_overhead_rows": (
                        candidate_dispatch["padding_overhead_rows"] if candidate_dispatch is not None else None
                    ),
                }
            )

    sync()
    diff = None
    expanded_abs = None
    max_reference_abs = None
    max_abs_diff = None
    if measurement_mode == "correctness":
        diff = (resident_out.float() - expanded_out.float()).abs()
        expanded_abs = expanded_out.float().abs()
        max_reference_abs = float(expanded_abs.max().item())
        max_abs_diff = float(diff.max().item())

    text_smoke: dict[str, Any] = {
        "input_source": "synthetic_random" if input_token_ids is None else ("input_text" if input_text else "token_ids"),
        "input_token_count": tokens,
        "input_text": input_text,
        "input_text_sha256": hashlib.sha256(input_text.encode("utf-8")).hexdigest() if input_text is not None else None,
        "input_token_ids": input_token_ids if input_token_ids is not None and len(input_token_ids) <= 128 else None,
        "input_token_ids_preview": input_token_ids[:16] if input_token_ids is not None else None,
        "input_token_ids_sha256": token_ids_digest(input_token_ids) if input_token_ids is not None else None,
        "include_lm_head": include_lm_head,
        "decode_output_token": decode_output_token,
        "text_quality_claim": False,
        "note": "Selected-layer slice smoke only; token identity is a wiring sanity check, not model quality evidence.",
    }
    if include_lm_head:
        expanded_logits = None
        expanded_logits_rec = None
        if measurement_mode == "correctness":
            expanded_logits, expanded_logits_rec = timed("expanded_final_norm_lm_head", lambda: final_logits(expanded_out))
        if exact_prefix_cache_exact_hit and exact_prefix_cache_entry is not None:
            logits_started = time.perf_counter()
            resident_logits = exact_prefix_cache_entry["resident_logits"]
            logits_elapsed_ms = (time.perf_counter() - logits_started) * 1000.0
            resident_logits_rec = {
                "name": "resident_final_norm_lm_head_exact_token_prefix_cache_hit",
                "elapsed_ms": logits_elapsed_ms,
                "iters": 1,
                "ms_per_iter": logits_elapsed_ms,
            }
        else:
            resident_logits, resident_logits_rec = timed(
                "resident_final_norm_lm_head",
                lambda: final_logits(resident_out),
            )
        resident_logits_for_decode = resident_logits
        topk_count = min(logit_topk, vocab)
        top_values, top_indices = torch.topk(resident_logits.float(), k=topk_count, dim=-1)
        top_ids = [int(item) for item in top_indices[0].tolist()]
        decoded = decode_token_ids(model_dir, top_ids) if decode_output_token else None
        logits_checksum_values = [float(resident_logits.float().mean().item())]
        logits_comparison = None
        if measurement_mode == "correctness":
            logit_diff = (resident_logits.float() - expanded_logits.float()).abs()
            expanded_logits_abs = expanded_logits.float().abs()
            logits_checksum_values.append(float(expanded_logits.float().mean().item()))
            logits_comparison = {
                "reference_variant": "expanded_final_norm_lm_head",
                "candidate_variant": "resident_final_norm_lm_head",
                "max_abs_diff": float(logit_diff.max().item()),
                "mean_abs_diff": float(logit_diff.mean().item()),
                "max_reference_abs": float(expanded_logits_abs.max().item()),
                "max_relative_to_reference_max": (
                    float(logit_diff.max().item()) / float(expanded_logits_abs.max().item())
                    if float(expanded_logits_abs.max().item())
                    else None
                ),
            }
        text_smoke.update(
            {
                "expanded_final_norm_lm_head": expanded_logits_rec,
                "resident_final_norm_lm_head": resident_logits_rec,
                "generated_token_id": top_ids[0],
                "generated_token_text": decoded[0] if decoded else None,
                "topk_token_ids": top_ids,
                "topk_token_text": decoded,
                "topk_logits": [float(item) for item in top_values[0].tolist()],
                "logits_checksum_finite": all(finite(value) for value in logits_checksum_values),
                "logits_comparison": logits_comparison,
            }
        )

    if exact_prefix_cache and not exact_prefix_cache_exact_hit and include_lm_head:
        store_record = store_exact_prefix_state(resident_out, resident_logits_for_decode)
        exact_prefix_cache_record.update(store_record)
    exact_prefix_cache_record["cache_after_prefill"] = engine_exact_prefix_cache_stats()

    decode_loop = run_decode_loop(decode_loop_steps)

    attention_substage_timeline: list[dict[str, Any]] = []
    linear_attention_chunk_gdn_internal_timeline: list[dict[str, Any]] = []
    if attention_substage_timing or linear_attention_chunk_gdn_internal_timing:
        profile_hidden = x
        for layer_index, entry in enumerate(layer_weights):
            weights = entry["tensors"]
            h1 = rmsnorm(profile_hidden, weights["input_layernorm"], rmsnorm_input_outputs.get(layer_index))
            if attention_substage_timing and full_attention_enabled(attention_mode) and entry["layer_type"] == "full_attention":
                profiled_full_attention(h1, layer_index, weights, attention_substage_timeline)
            elif linear_attention_enabled(attention_mode) and entry["layer_type"] == "linear_attention":
                profiled_linear_attention(
                    h1,
                    layer_index,
                    weights,
                    attention_substage_timeline,
                    linear_attention_chunk_gdn_internal_timeline,
                )
            profile_hidden, _ = layer_body(profile_hidden, layer_index, "resident")
    attention_substage_summary = summarize_attention_substages(attention_substage_timeline)
    linear_attention_chunk_gdn_internal_summary = summarize_attention_substages(
        linear_attention_chunk_gdn_internal_timeline
    )

    attention_cluster_timeline: list[dict[str, Any]] = []
    if attention_cluster_timing:
        cluster_hidden = x
        for layer_index, entry in enumerate(layer_weights):
            weights = entry["tensors"]
            h1 = rmsnorm(cluster_hidden, weights["input_layernorm"], rmsnorm_input_outputs.get(layer_index))
            if full_attention_enabled(attention_mode) and entry["layer_type"] == "full_attention":
                profiled_full_attention_clusters(h1, layer_index, weights, attention_cluster_timeline)
            elif linear_attention_enabled(attention_mode) and entry["layer_type"] == "linear_attention":
                profiled_linear_attention_clusters(h1, layer_index, weights, attention_cluster_timeline)
            cluster_hidden, _ = layer_body(cluster_hidden, layer_index, "resident")
    attention_cluster_summary = summarize_attention_substages(attention_cluster_timeline)

    attention_event_timeline: list[dict[str, Any]] = []
    if attention_event_timing:
        event_hidden = x
        for layer_index, entry in enumerate(layer_weights):
            weights = entry["tensors"]
            h1 = rmsnorm(event_hidden, weights["input_layernorm"], rmsnorm_input_outputs.get(layer_index))
            if full_attention_enabled(attention_mode) and entry["layer_type"] == "full_attention":
                profiled_full_attention_events(h1, layer_index, weights, attention_event_timeline)
            elif linear_attention_enabled(attention_mode) and entry["layer_type"] == "linear_attention":
                profiled_linear_attention_events(h1, layer_index, weights, attention_event_timeline)
            event_hidden, _ = layer_body(event_hidden, layer_index, "resident")
    attention_event_summary = summarize_attention_substages(attention_event_timeline)

    moe_substage_summary = summarize_moe_substages(moe_substage_timeline)
    moe_overlap_event_summary = summarize_moe_overlap_events(moe_overlap_event_timeline)
    measurement_and_reporting_wall_time_ms = (time.perf_counter() - measurement_wall_start) * 1000.0
    engine_wall_time_ms = (time.perf_counter() - engine_wall_start) * 1000.0

    return {
        "config_sha256": sha256_file(config_path),
        "index_sha256": sha256_file(index_path),
        "engine_wall_time_ms": engine_wall_time_ms,
        "engine_stage_wall_time_ms": {
            "python_imports": python_import_wall_time_ms,
            "runtime_setup": runtime_setup_wall_time_ms,
            "layer_tensor_load_derive": layer_tensor_load_derive_wall_time_ms,
            "global_tensor_load_derive": global_tensor_load_derive_wall_time_ms,
            "workspace_alloc_init": workspace_alloc_init_wall_time_ms,
            "measurement_and_reporting": measurement_and_reporting_wall_time_ms,
        },
        "cross_owner_prefill_composition": {
            **cross_owner_prefill_stats,
            "initial_request_tokens": initial_request_tokens,
            "lower_mask_cache_entries": len(cross_owner_lower_mask_cache),
            "activation_policy": "q16384 two q8192 chunks plus q15360 seed checkpoint first q8192",
        },
        "engine_tensor_cache": {
            "enabled": reuse_tensor_cache,
            "key_scope": "split_raw_weights_and_derived_layouts",
            "excluded_from_key": ["input_token_ids", "token_count"],
            "raw_weight_key_excludes": [
                "mode",
                "seq_len",
                "layers",
                "attention_mode",
                "variant_flags",
                "include_lm_head",
                "include_shared_expert",
            ],
            "derived_layout_key_excludes": [
                "mode",
                "seq_len",
                "layers",
                "attention_mode",
                "variant_flags",
                "include_lm_head",
                "include_shared_expert",
            ],
            "hits": tensor_cache_hits,
            "misses": tensor_cache_misses,
            "hits_by_scope": {
                key: value["hits"]
                for key, value in tensor_cache_by_scope.items()
            },
            "misses_by_scope": {
                key: value["misses"]
                for key, value in tensor_cache_by_scope.items()
            },
            "entries_after_run": len(_ENGINE_TENSOR_CACHE),
            "raw_weight_entries_after_run": sum(
                1
                for key in _ENGINE_TENSOR_CACHE
                if key.startswith(f"{raw_tensor_cache_prefix}:")
            ),
            "derived_layout_entries_after_run": sum(
                1
                for key in _ENGINE_TENSOR_CACHE
                if key.startswith(f"{derived_tensor_cache_prefix}:")
            ),
            "resident_native_prefill": {
                "weight_ownership": "raw_full_plus_native_decode_hotset",
                "decode_provider": "native_selected_expert_hotset_plus_raw_row_remainder",
                "decode_schedule": {"gate_block_h": 1024, "gate_num_warps": 4, "down_block_i": 512, "down_num_warps": 4},
                "native_decode_hotset_layers_requested": resident_native_decode_hotset_layers,
                "native_decode_hotset_layer_indices": sorted(
                    resident_native_decode_hotset_layer_indices
                ),
                "native_decode_hotset_entries_expected": 2 * resident_native_decode_hotset_layers,
                "native_decode_hotset_cache_entries_after_run": sum(
                    key in _ENGINE_TENSOR_CACHE for key in native_layout_cache_keys
                ),
                "native_decode_hotset_layers_available": sorted(
                    int(entry["layer"])
                    for entry in layer_weights
                    if "native_moe_gate_up" in entry["tensors"]
                    and "native_moe_down" in entry["tensors"]
                ),
                "layout_entries_expected": len(native_layout_cache_keys),
                "layout_entries_at_load": resident_native_layout_cache_entries_at_load,
                "layout_cache_complete_at_load": resident_native_layout_cache_complete_at_load,
                "partial_layout_entries_evicted": resident_native_layout_partial_cache_evicted,
                "reconstruction_calls": resident_native_prefill_reconstruction_calls,
                "reconstructed_layer_indices": sorted(resident_native_prefill_reconstructed_layers),
                "operational_transient_bytes_per_layer": (
                    experts * 2 * intermediate * hidden * torch.empty((), dtype=dtype).element_size()
                    + experts * hidden * intermediate * torch.empty((), dtype=dtype).element_size()
                ),
            },
        },
        "exact_prefix_cache": {
            **exact_prefix_cache_record,
            "canonical_chunked_cold_prefill": canonical_chunked_cold_prefill_record,
            "canonical_chunked_strict_suffix_prefill": canonical_chunked_strict_suffix_record,
            "cache_after_decode": engine_exact_prefix_cache_stats(),
            "decode_path_unchanged": True,
            "cold_prefill_smoothness_transfer": False,
        },
        "tensor_metadata": tensor_metadata,
        "global_tensor_metadata": global_tensor_metadata,
        "layer_metadata": resident_layer_metadata,
        "resident_stage_timeline_collected": (
            collect_resident_stage_timeline or moe_substage_timing or moe_overlap_event_timing
        ),
        "workspace": {
            "resident_moe_buffers": len(resident_workspaces),
            "resident_moe_unique_buffers": len(
                {int(buffer.data_ptr()) for buffer in resident_workspaces}
            ),
            "resident_moe_buffer_shape": [tokens, hidden],
            "attention_stub_buffers": len(attention_stub_buffers),
            "attention_stub_unique_buffers": len(
                {int(buffer.data_ptr()) for buffer in attention_stub_buffers}
            ),
            "attention_stub_buffer_shape": [tokens, hidden],
            "rmsnorm_input_buffers": len(rmsnorm_input_outputs),
            "rmsnorm_post_buffers": len(rmsnorm_post_outputs),
            "rmsnorm_final_buffers": 1 if rmsnorm_final_output is not None else 0,
            "lm_head_logits_buffers": 1 if lm_head_logits_output is not None else 0,
            "certified_lm_head_logits_buffers": (
                1 if certified_lm_head_logits_output is not None else 0
            ),
            "certified_lm_head_shortlist_limit": (
                1024 if lm_head_variant == "int8_certified_global_tie" else None
            ),
            "lm_head_logits_buffer_shape": [1, vocab],
            "rmsnorm_buffer_shape": [tokens, hidden],
            "router_logits_buffers": len(router_logits_outputs),
            "router_logits_buffer_shape": [tokens, experts],
            "router_topk_buffers": len(router_topk_score_outputs) + len(router_topk_index_outputs),
            "router_topk_partial_buffers": len(router_topk_partial_values) + len(router_topk_partial_ids),
            "router_topk_shape": [tokens, top_k],
            "vllm_moe_prealloc_cache1_buffers": len(vllm_moe_cache1_outputs),
            "vllm_moe_prealloc_cache2_buffers": len(vllm_moe_cache2_outputs),
            "vllm_moe_prealloc_cache3_buffers": len(vllm_moe_cache3_outputs),
            "vllm_moe_prealloc_output_buffers": len(vllm_moe_prealloc_outputs),
            "full_attention_kv_cache_buffers": len(full_attention_kv_caches) * 2,
            "full_attention_kv_cache_layout": full_attention_kv_cache_layout,
            "full_attention_kv_cache_shape": full_attention_kv_cache_shape,
            "full_attention_proj_buffers": sum(len(buffers) for buffers in full_attention_proj_outputs.values()),
            "full_attention_proj_buffer_shapes": (
                {
                    "qkv": [2 * q_dim + 2 * kv_dim],
                    "o": [hidden],
                }
                if triton_full_attention_fused_qkv_decode
                else {
                    "q_gate": [2 * q_dim],
                    "k": [kv_dim],
                    "v": [kv_dim],
                    "o": [hidden],
                }
            ),
            "full_attention_norm_rope_buffers": sum(
                len(buffers) for buffers in full_attention_norm_rope_outputs.values()
            ),
            "full_attention_norm_rope_unique_buffers": len(
                {
                    int(buffer.data_ptr())
                    for buffers in full_attention_norm_rope_outputs.values()
                    for buffer in buffers.values()
                }
            ),
            "full_attention_norm_rope_buffer_shapes": {
                "q": [tokens, heads, head_dim],
                "k": [tokens, kv_heads, head_dim],
            },
            "full_attention_kv_cache_validated": full_attention_enabled(attention_mode) and bool(full_attention_kv_caches),
            "linear_attention_state_buffers": len(linear_attention_conv_states) + len(linear_attention_ssm_states),
            "linear_attention_initial_vllm_ssm_state_buffers": len(linear_attention_initial_ssm_states_vllm),
            "linear_attention_output_vllm_ssm_state_buffers": len(linear_attention_ssm_states_vllm),
            "linear_attention_conv_window_buffers": len(linear_attention_conv_windows),
            "linear_attention_packed_ssm_state_buffers": len(linear_attention_packed_ssm_states),
            "linear_attention_packed_output_buffers": len(linear_attention_packed_outputs),
            "linear_attention_input_proj_buffers": len(linear_attention_input_proj_outputs),
            "linear_attention_qkv_layout_buffers": sum(
                len(buffers) for buffers in linear_attention_qkv_layout_outputs.values()
            ),
            "linear_attention_conv_output_buffers": len(linear_attention_conv_outputs),
            "linear_attention_gated_norm_buffers": len(linear_attention_gated_norm_outputs),
            "linear_attention_gated_norm_prefill_shared_output": (
                linear_attention_gated_norm_prefill_output is not None
            ),
            "linear_attention_output_proj_buffers": len(linear_attention_output_proj_outputs),
            "shared_expert_proj_buffers": sum(len(buffers) for buffers in shared_expert_proj_outputs.values()),
            "shared_expert_proj_buffer_shapes": (
                {
                    "input": [1 + 2 * shared_intermediate],
                    "down_proj": [hidden],
                }
                if triton_shared_expert_fused_input_decode
                else {
                    "gate": [1],
                    "gate_proj": [shared_intermediate],
                    "up_proj": [shared_intermediate],
                    "down_proj": [hidden],
                }
            ),
            "linear_attention_conv_output_shape": [linear_conv_dim],
            "linear_attention_qkv_layout_buffer_shapes": {
                "q": [1, tokens, linear_key_heads, linear_key_head_dim],
                "k": [1, tokens, linear_key_heads, linear_key_head_dim],
                "v": [1, tokens, linear_value_heads, linear_value_head_dim],
                "z": [tokens, linear_value_heads, linear_value_head_dim],
                "a": [tokens, linear_value_heads],
                "b": [tokens, linear_value_heads],
            },
            "linear_attention_conv_state_shape": [linear_conv_dim, linear_conv_state_len],
            "linear_attention_ssm_state_shape": [linear_value_heads, linear_key_head_dim, linear_value_head_dim],
            "linear_attention_initial_vllm_ssm_state_shape": [
                1,
                linear_value_heads,
                linear_value_head_dim,
                linear_key_head_dim,
            ],
            "linear_attention_packed_ssm_state_shape": [2, linear_value_heads, linear_value_head_dim, linear_key_head_dim],
            "linear_attention_state_validated": linear_attention_enabled(attention_mode) and bool(linear_attention_ssm_states),
        },
        "orientation": {
            "full_attention": "q_proj contains per-head q and output-gate halves; q/k use head RMSNorm, NeoX-style partial RoPE, SDPA, output gate, then o_proj",
            "rmsnorm_weight_semantics": "Qwen3.5 Gemma-style RMSNorm applies (1 + weight) for layer, q/k, and final norms",
            "full_attention_variant": full_attention_variant,
            "full_attention_proj_variant": full_attention_proj_variant,
            "full_attention_norm_rope_variant": full_attention_norm_rope_variant,
            "full_attention_kv_cache_layout": full_attention_kv_cache_layout,
            "linear_attention": "in_proj_qkv is q,k,v; in_proj_z is gated RMSNorm input; in_proj_b/a form beta and GDN decay; conv state is raw qkv history; SSM state maps key to value",
            "linear_attention_variant": linear_attention_variant,
            "linear_attention_input_proj_variant": linear_attention_input_proj_variant,
            "linear_attention_output_proj_variant": linear_attention_output_proj_variant,
            "linear_attention_conv_variant": linear_attention_conv_variant,
            "linear_attention_conv_state_refswap": linear_attention_conv_state_refswap,
            "linear_attention_gated_norm_variant": linear_attention_gated_norm_variant,
            "linear_attention_post_conv_prep_block_t": linear_attention_post_conv_prep_block_t,
            "linear_attention_prefill_conv_block_t": linear_attention_prefill_conv_block_t,
            "linear_attention_prefill_conv_block_c": linear_attention_prefill_conv_block_c,
            "linear_attention_prefill_conv_num_warps": linear_attention_prefill_conv_num_warps,
            "linear_attention_prefill_conv_effective_block_t": prefill_conv_block_t,
            "linear_attention_prefill_conv_effective_block_c": prefill_conv_block_c,
            "linear_attention_prefill_conv_effective_num_warps": prefill_conv_num_warps,
            "linear_attention_prefill_conv_post_prep_fusion": linear_attention_prefill_conv_post_prep_fusion,
            "linear_attention_prefill_fused_h_o": linear_attention_prefill_fused_h_o,
            "linear_attention_prefill_fused_u_h_o": linear_attention_prefill_fused_u_h_o,
            "linear_attention_chunk_gdn_internal_timing": linear_attention_chunk_gdn_internal_timing,
            "rmsnorm_variant": rmsnorm_variant,
            "lm_head_variant": lm_head_variant,
            "shared_expert_proj_variant": shared_expert_proj_variant,
            "padded_batched_moe": "active rows are sorted by expert, padded to max rows per populated expert, then executed with torch.bmm and scattered back with index_add",
            "count_batched_moe": "active rows are sorted by expert, grouped by identical rows per expert, executed with torch.bmm without padded rows, and scattered back with index_add",
            "vllm_fused_moe": "hidden states, top-k weights, top-k ids, and real expert tensors are passed to vLLM fused_experts",
            "vllm_fused_moe_override_config": vllm_moe_override_config(moe_variant, mode, moe_override_config),
            "vllm_fused_moe_override_config_by_layer": vllm_moe_effective_override_config_by_layer(),
            "vllm_fused_inplace_moe": (
                "in-place candidate computes the shared expert before fused_experts mutates post-attention hidden; "
                "stage timing resets a scratch input buffer, so pipeline timing is the promotion signal"
            ),
            "router_variant": router_variant,
            "router": (
                (
                    "router matvec, top-k, and top-k softmax weights use Triton kernels "
                    "over pretransposed gate.weight"
                )
                if triton_router_topk_softmax_decode
                else "router matvec and top-k use a two-stage Triton kernel over pretransposed gate.weight"
                if triton_router_topk_decode
                else (
                    "router logits use pretransposed gate.weight and a preallocated logits buffer"
                    if torch_out_router_decode
                    else "router_logits = hidden @ gate.weight.T"
                )
            ),
            "expert_gate_up": "projected = hidden @ experts.gate_up_proj[expert].T; chunk order is gate, up",
            "expert_down": "expert_out = silu(gate) * up @ experts.down_proj[expert].T",
            "shared_expert": "sigmoid(hidden @ shared_expert_gate.weight.T) * down(silu(gate) * up)",
        },
        "pipeline": {
            "expanded": expanded_rec,
            "resident": resident_rec,
            "resident_cuda_graph_replay": resident_graph_rec,
            "expanded_ms_per_iter": (
                float(expanded_rec["ms_per_iter"]) if expanded_rec is not None else None
            ),
            "resident_ms_per_iter": float(resident_rec["ms_per_iter"]),
            "resident_cuda_graph_replay_ms_per_iter": (
                float(resident_graph_rec["ms_per_iter"])
                if resident_graph_rec is not None and resident_graph_rec.get("status") == "ok"
                else None
            ),
            "resident_cuda_graph_replay_vs_eager_speedup": (
                float(resident_rec["ms_per_iter"] / resident_graph_rec["ms_per_iter"])
                if resident_graph_rec is not None and resident_graph_rec.get("status") == "ok"
                else None
            ),
            "resident_cuda_graph_replay_max_abs_diff": (
                float((resident_graph_out.float() - resident_out.float()).abs().max().item())
                if resident_graph_out is not None
                else None
            ),
            "resident_vs_expanded_speedup": (
                float(expanded_rec["ms_per_iter"] / resident_rec["ms_per_iter"])
                if expanded_rec is not None
                else None
            ),
            "moe_variant": moe_variant,
            "moe_override_config": vllm_moe_override_config(moe_variant, mode, moe_override_config),
            "moe_override_config_by_layer": vllm_moe_effective_override_config_by_layer(),
            "router_variant": router_variant,
        },
        "resident_stage_timeline": resident_stage_timeline,
        "attention_substage_timing": {
            "enabled": attention_substage_timing,
            "note": (
                "Separate diagnostic pass over resident layer inputs; use for substage attribution, "
                "not as the top-level pipeline latency."
            ),
            "summary": attention_substage_summary,
            "timeline": attention_substage_timeline,
        },
        "linear_attention_chunk_gdn_internal_timing": {
            "enabled": linear_attention_chunk_gdn_internal_timing,
            "note": (
                "Separate diagnostic pass over retained prefill chunk-GDN internals; "
                "use to locate the dominant vLLM FLA subkernel before attempting a structural rewrite."
            ),
            "summary": linear_attention_chunk_gdn_internal_summary,
            "timeline": linear_attention_chunk_gdn_internal_timeline,
        },
        "attention_cluster_timing": {
            "enabled": attention_cluster_timing,
            "note": (
                "Separate diagnostic pass that times parent attention replay plus coarse clusters; "
                "use to decide whether parent/substage gaps are worth fusion work."
            ),
            "summary": attention_cluster_summary,
            "timeline": attention_cluster_timeline,
        },
        "attention_event_timing": {
            "enabled": attention_event_timing,
            "note": (
                "Separate CUDA-event diagnostic pass over retained full/linear attention coarse stages. "
                "Use this to separate GPU-stage elapsed time from wall-clock parent/substage gaps."
            ),
            "summary": attention_event_summary,
            "timeline": attention_event_timeline,
        },
        "moe_substage_timing": {
            "enabled": moe_substage_timing,
            "note": (
                "Separate diagnostic pass over resident layer inputs; use for MoE substage attribution, "
                "not as the top-level pipeline latency."
            ),
            "summary": moe_substage_summary,
            "timeline": moe_substage_timeline,
        },
        "moe_overlap_event_timing": {
            "enabled": moe_overlap_event_timing,
            "note": (
                "CUDA-event diagnostic for router-early shared-expert/MoE overlap. "
                "It reports main-stream router, routed MoE, exposed wait, add, and "
                "side-stream shared-expert durations; use this to classify overlap "
                "headroom, not as a serving benchmark."
            ),
            "summary": moe_overlap_event_summary,
            "timeline": moe_overlap_event_timeline,
        },
        "cuda_graph_replay_timing": {
            "enabled": cuda_graph_replay_timing,
            "scope": "resident_four_layer_pipeline only; decode-loop state promotion and token selection are not captured",
            "record": resident_graph_rec,
        },
        "comparison": {
            "enabled": measurement_mode == "correctness",
            "reference_variant": "expanded_four_layer_pipeline",
            "candidate_variant": "resident_four_layer_pipeline",
            "candidate_moe_variant": moe_variant,
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": float(diff.mean().item()) if diff is not None else None,
            "max_reference_abs": max_reference_abs,
            "max_relative_to_reference_max": (
                max_abs_diff / max_reference_abs if max_reference_abs else None
            ),
        },
        "text_smoke": text_smoke,
        "decode_loop": decode_loop,
        "checksums": {
            "expanded_mean": (
                float(expanded_out.float().mean().item()) if expanded_out is not None else None
            ),
            "resident_mean": float(resident_out.float().mean().item()),
        },
        "checksum_finite": all(
            finite(value)
            for value in [
                value
                for value in [
                    float(expanded_out.float().mean().item()) if expanded_out is not None else None,
                    float(resident_out.float().mean().item()),
                ]
                if value is not None
            ]
        ),
        "peak_memory_bytes": (
            int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") and torch.cuda.is_available() else None
        ),
        "torch_version": torch.__version__,
        "device": device,
        "attention_mode": attention_mode,
        "full_attention_variant": full_attention_variant,
        "full_attention_proj_variant": full_attention_proj_variant,
        "full_attention_norm_rope_variant": full_attention_norm_rope_variant,
        "full_attention_kv_cache_layout": full_attention_kv_cache_layout,
        "full_attention_fused_gate_o_proj": full_attention_fused_gate_o_proj,
        "full_attention_fused_norm_rope_kv_write": full_attention_fused_norm_rope_kv_write,
        "moe_variant": moe_variant,
        "router_variant": router_variant,
        "linear_attention_variant": linear_attention_variant,
        "linear_attention_input_proj_variant": linear_attention_input_proj_variant,
        "linear_attention_output_proj_variant": linear_attention_output_proj_variant,
        "linear_attention_conv_variant": linear_attention_conv_variant,
        "linear_attention_conv_state_refswap": linear_attention_conv_state_refswap,
        "official_prefill_conv_calls": official_prefill_conv_calls,
        "official_prefill_conv_initial_state_true_calls": official_prefill_conv_initial_state_true_calls,
        "linear_attention_gated_norm_variant": linear_attention_gated_norm_variant,
        "linear_attention_post_conv_prep_block_t": linear_attention_post_conv_prep_block_t,
        "linear_attention_prefill_conv_block_t": linear_attention_prefill_conv_block_t,
        "linear_attention_prefill_conv_block_c": linear_attention_prefill_conv_block_c,
        "linear_attention_prefill_conv_num_warps": linear_attention_prefill_conv_num_warps,
        "linear_attention_prefill_conv_effective_block_t": prefill_conv_block_t,
        "linear_attention_prefill_conv_effective_block_c": prefill_conv_block_c,
        "linear_attention_prefill_conv_effective_num_warps": prefill_conv_num_warps,
        "rmsnorm_variant": rmsnorm_variant,
        "lm_head_variant": lm_head_variant,
        "shared_expert_proj_variant": shared_expert_proj_variant,
        "shared_expert_overlap_stream_residency": {
            "enabled": shared_expert_overlap_stream is not None,
            "module_cache_hit": shared_expert_overlap_stream_cache_hit,
            "module_cache_entries": len(_ENGINE_AUXILIARY_STREAM_CACHE),
            "cache_key": (
                list(shared_expert_overlap_stream_cache_key)
                if shared_expert_overlap_stream_cache_key is not None
                else None
            ),
            "stream_identifier_sha256": shared_expert_overlap_stream_identifier_sha256,
        },
        "measurement_mode": measurement_mode,
        "seed": seed,
        "warmup": warmup,
        "iters": iters,
        "decode_loop_steps": decode_loop_steps,
        "prefill_seed_output": prefill_seed_output,
        "decode_loop_fast_housekeeping": decode_loop_fast_housekeeping,
        "decode_loop_defer_token_cpu_sync": decode_loop_defer_token_cpu_sync,
        "decode_loop_token_cpu_sync_interval": decode_loop_token_cpu_sync_interval,
        "decode_loop_diagnostic": decode_loop_diagnostic,
        "overlap_decode_state_promotion_lm_head": decode_state_promotion_stream is not None,
        "skip_layer_dispatch_metadata": skip_layer_dispatch_metadata,
        "moe_chunk_size": moe_chunk_size,
        "env": {
            "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": os.environ.get(
                "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"
            ),
        },
    }


def main() -> None:
    root = repo_root()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default=str(root / "doc/reference/amd395-qwen36-35b-a3b-bf16/qwen36-shape-manifest.json"),
    )
    parser.add_argument("--model-dir")
    parser.add_argument("--layers", default="0,1,2,3")
    parser.add_argument("--mode", choices=["prefill", "decode"], default="prefill")
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--tokens", type=int)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--moe-chunk-size", type=int, default=64)
    parser.add_argument(
        "--moe-variant",
        choices=sorted(moe_variants()),
        default="resident_dispatch",
    )
    parser.add_argument(
        "--moe-override-config-json",
        help=(
            "Optional JSON object passed to vLLM fused_moe.override_config for "
            "bounded config probes; retained named variants are unchanged when omitted."
        ),
    )
    parser.add_argument(
        "--moe-override-config-by-layer-json",
        help=(
            "Optional JSON object mapping model layer id to a vLLM fused_moe.override_config "
            "object. The map only overrides one-token decode; prefill keeps the retained config."
        ),
    )
    parser.add_argument(
        "--overlap-shared-expert-moe",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Probe CUDA-stream overlap of shared expert with non-inplace vLLM "
            "fused MoE for one-token decode."
        ),
    )
    parser.add_argument(
        "--overlap-shared-expert-router-moe",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Probe starting the shared-expert side stream before router/top-k "
            "so router and non-inplace vLLM fused MoE can overlap with it."
        ),
    )
    parser.add_argument(
        "--shared-expert-overlap-stream-priority",
        type=int,
        help=(
            "Optional torch.cuda.Stream priority for the shared-expert overlap "
            "side stream; unset preserves the default stream priority."
        ),
    )
    parser.add_argument(
        "--router-variant",
        choices=sorted(router_variants()),
        default="torch",
    )
    parser.add_argument(
        "--attention-mode",
        choices=["stub", "full_for_full_attention", "linear_and_full_attention"],
        default="stub",
    )
    parser.add_argument(
        "--linear-attention-variant",
        choices=sorted(linear_attention_variants()),
        default="torch_ref",
    )
    parser.add_argument(
        "--linear-attention-input-proj-variant",
        choices=sorted(linear_attention_input_proj_variants()),
        default="separate",
    )
    parser.add_argument(
        "--linear-attention-output-proj-variant",
        choices=sorted(linear_attention_output_proj_variants()),
        default="torch",
    )
    parser.add_argument(
        "--linear-attention-conv-variant",
        choices=sorted(linear_attention_conv_variants()),
        default="conv1d",
    )
    parser.add_argument(
        "--linear-attention-conv-state-refswap",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Promote one-token decode linear-attention causal-conv state by tensor reference instead of copying.",
    )
    parser.add_argument(
        "--linear-attention-gated-norm-variant",
        choices=sorted(linear_attention_gated_norm_variants()),
        default="torch",
    )
    parser.add_argument(
        "--linear-attention-post-conv-prep-block-t",
        type=int,
        help="Opt-in vLLM fused_post_conv_prep BLOCK_T override for focused prefill probes.",
    )
    parser.add_argument(
        "--linear-attention-prefill-conv-block-t",
        type=int,
        choices=[8, 16, 32, 64],
        help="Opt-in Triton prefill causal-conv BLOCK_T override for focused prefill probes.",
    )
    parser.add_argument(
        "--linear-attention-prefill-conv-block-c",
        type=int,
        choices=[16, 32, 64],
        help="Opt-in Triton prefill causal-conv BLOCK_C override for focused prefill probes.",
    )
    parser.add_argument(
        "--linear-attention-prefill-conv-num-warps",
        type=int,
        choices=[4, 8],
        help="Opt-in Triton prefill causal-conv num_warps override for focused prefill probes.",
    )
    parser.add_argument(
        "--linear-attention-prefill-conv-post-prep-fusion",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Default-off prefill candidate that fuses Triton causal-conv with vLLM post-conv prep.",
    )
    parser.add_argument(
        "--linear-attention-prefill-vllm-state-handoff",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Keep prefill chunk-GDN final state in vLLM layout for native-vLLM "
            "decode state handoff instead of round-tripping through engine layout."
        ),
    )
    parser.add_argument(
        "--linear-attention-chunk-gdn-internal-timing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Collect a separate diagnostic timeline for retained prefill chunk-GDN internal vLLM FLA kernels.",
    )
    parser.add_argument(
        "--linear-attention-prefill-fused-h-o",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the experimental fused prefill chunk-GDN h/o boundary for retained chunk16 Qwen shapes.",
    )
    parser.add_argument(
        "--linear-attention-prefill-fused-u-h-o",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the experimental W-only plus fused U+h/o prefill chunk-GDN boundary.",
    )
    parser.add_argument(
        "--rmsnorm-variant",
        choices=sorted(rmsnorm_variants()),
        default="torch",
    )
    parser.add_argument(
        "--full-attention-variant",
        choices=sorted(full_attention_variants()),
        default="sdpa",
    )
    parser.add_argument(
        "--full-attention-proj-variant",
        choices=sorted(full_attention_proj_variants()),
        default="torch",
    )
    parser.add_argument(
        "--full-attention-norm-rope-variant",
        choices=sorted(full_attention_norm_rope_variants()),
        default="torch",
    )
    parser.add_argument(
        "--full-attention-kv-cache-layout",
        choices=sorted(full_attention_kv_cache_layouts()),
        default="seq",
        help=(
            "Full-attention KV cache memory layout: seq=[seq,kv_head,head_dim] "
            "or grouped=[kv_head,seq,head_dim] for grouped-BMM decode probes."
        ),
    )
    parser.add_argument(
        "--full-attention-fused-gate-o-proj",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Fuse full-attention one-token decode output-gate sigmoid/multiply "
            "into the Triton o_proj matvec."
        ),
    )
    parser.add_argument(
        "--full-attention-fused-norm-rope-kv-write",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Fuse full-attention one-token decode q/k head norm+RoPE with "
            "k/v cache writes."
        ),
    )
    parser.add_argument(
        "--lm-head-variant",
        choices=sorted(lm_head_variants()),
        default="view",
    )
    parser.add_argument(
        "--shared-expert-proj-variant",
        choices=sorted(shared_expert_proj_variants()),
        default="torch",
    )
    parser.add_argument("--include-shared-expert", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--input-text")
    parser.add_argument("--input-token-ids")
    parser.add_argument("--include-lm-head", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--decode-output-token", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--logit-topk", type=int, default=5)
    parser.add_argument("--decode-sampling", choices=["argmax", "top_k"], default="argmax")
    parser.add_argument("--sampling-temperature", type=float, default=0.8)
    parser.add_argument("--sampling-top-k", type=int, default=50)
    parser.add_argument("--decode-stop-token-ids", default="")
    parser.add_argument(
        "--decode-loop-steps",
        type=int,
        default=0,
        help=(
            "For decode/tokens=1, run a state-carrying generated-token loop after "
            "the normal resident timing."
        ),
    )
    parser.add_argument(
        "--decode-loop-fast-housekeeping",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "For decode-loop generation, precompute RoPE rows once per loop and "
            "reuse the next-token embedding buffer."
        ),
    )
    parser.add_argument(
        "--defer-decode-token-cpu-sync",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "For decode-loop generation, keep selected token ids on GPU during "
            "the loop and materialize them on CPU once after the loop."
        ),
    )
    parser.add_argument(
        "--decode-token-cpu-sync-interval",
        type=int,
        default=1,
        help=(
            "For decode-loop generation, materialize generated token ids on CPU every N "
            "tokens; 1 preserves the current per-token sync path and 0 defers until loop end."
        ),
    )
    parser.add_argument(
        "--decode-loop-diagnostic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "For decode-loop correctness probes, record per-step top-k logits and "
            "linear-attention state statistics. Timings include diagnostic sync overhead."
        ),
    )
    parser.add_argument(
        "--overlap-decode-state-promotion-lm-head",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Overlap decode state promotion with final RMSNorm/LM-head during decode loops.",
    )
    parser.add_argument(
        "--attention-substage-timing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run an extra diagnostic pass that times full/linear attention internals by layer.",
    )
    parser.add_argument(
        "--attention-cluster-timing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run an extra diagnostic pass that times parent attention replay and coarse clusters by layer.",
    )
    parser.add_argument(
        "--attention-event-timing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run an extra CUDA-event diagnostic pass over retained full/linear "
            "attention coarse stages."
        ),
    )
    parser.add_argument(
        "--retained-attention-fast-path",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use a decode-only retained-route fast path that inlines full/linear attention parent branches.",
    )
    parser.add_argument(
        "--skip-layer-dispatch-metadata",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip per-layer dispatch metadata dict construction in the main resident pipeline.",
    )
    parser.add_argument(
        "--cuda-graph-replay-timing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Diagnostic only: capture and replay the resident one-token pipeline "
            "with torch.cuda.CUDAGraph to estimate launch/runtime-boundary headroom."
        ),
    )
    parser.add_argument(
        "--moe-substage-timing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run an extra diagnostic pass that times vLLM fused-MoE prep and kernel call by layer.",
    )
    parser.add_argument(
        "--moe-overlap-event-timing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run an extra CUDA-event diagnostic pass for router-early shared-expert/MoE overlap, "
            "including exposed wait time on the main stream."
        ),
    )
    parser.add_argument(
        "--collect-resident-stage-timeline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Collect the extra per-layer resident stage timeline diagnostic pass.",
    )
    parser.add_argument(
        "--measurement-mode",
        choices=["correctness", "resident_only"],
        default="correctness",
        help="Use resident_only for performance-only timing after a separate correctness gate.",
    )
    parser.add_argument(
        "--reuse-tensor-cache",
        action="store_true",
        help="Reuse raw and derived tensors across in-process run_with_torch calls.",
    )
    parser.add_argument(
        "--resident-cache-probe-repeats",
        type=int,
        default=1,
        help="Run the same execute path repeatedly in one process with tensor-cache reuse enabled.",
    )
    args = parser.parse_args()
    if args.resident_cache_probe_repeats <= 0:
        raise SystemExit("--resident-cache-probe-repeats must be positive")

    manifest = load_json(Path(args.manifest))
    model_dir = Path(args.model_dir or manifest["source"]["model_dir"])
    if args.input_text and args.input_token_ids:
        raise SystemExit("--input-text and --input-token-ids are mutually exclusive")
    input_token_ids = csv_ints(args.input_token_ids) if args.input_token_ids else None
    if args.input_text:
        input_token_ids = tokenize_text(model_dir, args.input_text)
    tokens = args.tokens if args.tokens is not None else (len(input_token_ids) if input_token_ids is not None else 128)
    if input_token_ids is not None and args.tokens is not None and len(input_token_ids) != args.tokens:
        raise SystemExit(f"--tokens {args.tokens} must match input token count {len(input_token_ids)}")
    if args.decode_token_cpu_sync_interval < 0:
        raise SystemExit("--decode-token-cpu-sync-interval must be non-negative")
    decode_loop_token_cpu_sync_interval = (
        0 if args.defer_decode_token_cpu_sync else args.decode_token_cpu_sync_interval
    )
    try:
        moe_override_config = parse_moe_override_config(args.moe_override_config_json)
        moe_override_config_by_layer = normalize_moe_override_config_by_layer(args.moe_override_config_by_layer_json)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if (
        (moe_override_config is not None or moe_override_config_by_layer is not None)
        and args.moe_variant not in vllm_fused_moe_variants()
    ):
        raise SystemExit("--moe-override-config-json requires a vllm_fused MoE variant")
    layers = csv_ints(args.layers)
    contract = planned_contract(
        manifest,
        model_dir,
        layers,
        args.mode,
        args.seq_len,
        tokens,
        args.attention_mode,
        args.moe_variant,
        moe_override_config,
        moe_override_config_by_layer,
        args.overlap_shared_expert_moe,
        args.overlap_shared_expert_router_moe,
        args.router_variant,
        args.linear_attention_variant,
        args.linear_attention_input_proj_variant,
        args.linear_attention_output_proj_variant,
        args.linear_attention_conv_variant,
        args.linear_attention_conv_state_refswap,
        args.linear_attention_gated_norm_variant,
        args.linear_attention_post_conv_prep_block_t,
        args.linear_attention_prefill_conv_block_t,
        args.linear_attention_prefill_conv_block_c,
        args.linear_attention_prefill_conv_num_warps,
        args.linear_attention_prefill_conv_post_prep_fusion,
        args.linear_attention_prefill_vllm_state_handoff,
        args.linear_attention_prefill_fused_h_o,
        args.linear_attention_prefill_fused_u_h_o,
        args.linear_attention_chunk_gdn_internal_timing,
        args.rmsnorm_variant,
        args.full_attention_variant,
        args.full_attention_proj_variant,
        args.full_attention_norm_rope_variant,
        args.full_attention_kv_cache_layout,
        args.full_attention_fused_gate_o_proj,
        args.lm_head_variant,
        args.shared_expert_proj_variant,
        args.retained_attention_fast_path,
        args.skip_layer_dispatch_metadata,
        args.include_shared_expert,
        "synthetic_random" if input_token_ids is None else ("input_text" if args.input_text else "token_ids"),
        args.include_lm_head,
        full_attention_fused_norm_rope_kv_write=args.full_attention_fused_norm_rope_kv_write,
        shared_expert_overlap_stream_priority=args.shared_expert_overlap_stream_priority,
    )
    contract["include_shared_expert"] = args.include_shared_expert
    contract["decode_output_token"] = args.decode_output_token
    contract["logit_topk"] = args.logit_topk
    contract["decode_loop_steps"] = args.decode_loop_steps
    contract["decode_sampling"] = args.decode_sampling
    contract["sampling_temperature"] = args.sampling_temperature
    contract["sampling_top_k"] = args.sampling_top_k
    decode_stop_token_ids = set(csv_ints(args.decode_stop_token_ids)) if args.decode_stop_token_ids else set()
    contract["decode_stop_token_ids"] = sorted(decode_stop_token_ids)
    contract["decode_loop_fast_housekeeping"] = args.decode_loop_fast_housekeeping
    contract["decode_loop_defer_token_cpu_sync"] = args.defer_decode_token_cpu_sync
    contract["decode_loop_token_cpu_sync_interval"] = decode_loop_token_cpu_sync_interval
    contract["decode_loop_diagnostic"] = args.decode_loop_diagnostic
    contract["overlap_decode_state_promotion_lm_head"] = args.overlap_decode_state_promotion_lm_head
    contract["shared_expert_overlap_stream_priority"] = args.shared_expert_overlap_stream_priority
    contract["attention_substage_timing"] = args.attention_substage_timing
    contract["attention_cluster_timing"] = args.attention_cluster_timing
    contract["attention_event_timing"] = args.attention_event_timing
    contract["retained_attention_fast_path"] = args.retained_attention_fast_path
    contract["skip_layer_dispatch_metadata"] = args.skip_layer_dispatch_metadata
    contract["cuda_graph_replay_timing"] = args.cuda_graph_replay_timing
    contract["linear_attention_post_conv_prep_block_t"] = args.linear_attention_post_conv_prep_block_t
    contract["linear_attention_prefill_conv_block_t"] = args.linear_attention_prefill_conv_block_t
    contract["linear_attention_prefill_conv_block_c"] = args.linear_attention_prefill_conv_block_c
    contract["linear_attention_prefill_conv_num_warps"] = args.linear_attention_prefill_conv_num_warps
    contract["linear_attention_prefill_conv_effective_block_t"] = args.linear_attention_prefill_conv_block_t or 16
    contract["linear_attention_prefill_conv_effective_block_c"] = args.linear_attention_prefill_conv_block_c or 32
    contract["linear_attention_prefill_conv_effective_num_warps"] = args.linear_attention_prefill_conv_num_warps or 4
    contract["linear_attention_prefill_conv_post_prep_fusion"] = (
        args.linear_attention_prefill_conv_post_prep_fusion
    )
    contract["linear_attention_prefill_vllm_state_handoff"] = args.linear_attention_prefill_vllm_state_handoff
    contract["linear_attention_prefill_fused_h_o"] = args.linear_attention_prefill_fused_h_o
    contract["linear_attention_prefill_fused_u_h_o"] = args.linear_attention_prefill_fused_u_h_o
    contract["linear_attention_chunk_gdn_internal_timing"] = args.linear_attention_chunk_gdn_internal_timing
    contract["moe_substage_timing"] = args.moe_substage_timing
    contract["moe_overlap_event_timing"] = args.moe_overlap_event_timing
    contract["collect_resident_stage_timeline"] = args.collect_resident_stage_timeline
    contract["measurement_mode"] = args.measurement_mode
    contract["reuse_tensor_cache"] = args.reuse_tensor_cache or args.resident_cache_probe_repeats > 1
    contract["resident_cache_probe_repeats"] = args.resident_cache_probe_repeats
    if input_token_ids is not None:
        contract["input_token_ids_sha256"] = token_ids_digest(input_token_ids)
        contract["input_token_count"] = len(input_token_ids)
    if args.input_text:
        contract["input_text"] = args.input_text
        contract["input_text_sha256"] = hashlib.sha256(args.input_text.encode("utf-8")).hexdigest()

    result: dict[str, Any] = {
        "schema_version": 1,
        "mode": "four-layer-mini-engine",
        "contract": contract,
        "executed": False,
        "source": {
            "commit": git_value(root, "rev-parse", "HEAD") or "unknown",
            "diff_scope": source_diff_scope(root),
        },
    }
    if args.execute:
        result["executed"] = True
        captured_stdout = io.StringIO()
        with contextlib.redirect_stdout(captured_stdout):
            measurements: list[dict[str, Any]] = []
            for _ in range(args.resident_cache_probe_repeats):
                measurements.append(
                    run_with_torch(
                        manifest=manifest,
                        model_dir=model_dir,
                        layers=layers,
                        mode=args.mode,
                        seq_len=args.seq_len,
                        tokens=tokens,
                        device=args.device,
                        warmup=args.warmup,
                        iters=args.iters,
                        seed=args.seed,
                        moe_chunk_size=args.moe_chunk_size,
                        attention_mode=args.attention_mode,
                        moe_variant=args.moe_variant,
                        moe_override_config=moe_override_config,
                        moe_override_config_by_layer=moe_override_config_by_layer,
                        overlap_shared_expert_moe=args.overlap_shared_expert_moe,
                        overlap_shared_expert_router_moe=args.overlap_shared_expert_router_moe,
                        router_variant=args.router_variant,
                        linear_attention_variant=args.linear_attention_variant,
                        linear_attention_input_proj_variant=args.linear_attention_input_proj_variant,
                        linear_attention_output_proj_variant=args.linear_attention_output_proj_variant,
                        linear_attention_conv_variant=args.linear_attention_conv_variant,
                        linear_attention_conv_state_refswap=args.linear_attention_conv_state_refswap,
                        linear_attention_gated_norm_variant=args.linear_attention_gated_norm_variant,
                        linear_attention_post_conv_prep_block_t=args.linear_attention_post_conv_prep_block_t,
                        linear_attention_prefill_conv_block_t=args.linear_attention_prefill_conv_block_t,
                        linear_attention_prefill_conv_block_c=args.linear_attention_prefill_conv_block_c,
                        linear_attention_prefill_conv_num_warps=args.linear_attention_prefill_conv_num_warps,
                        linear_attention_prefill_conv_post_prep_fusion=(
                            args.linear_attention_prefill_conv_post_prep_fusion
                        ),
                        linear_attention_prefill_vllm_state_handoff=args.linear_attention_prefill_vllm_state_handoff,
                        linear_attention_prefill_fused_h_o=args.linear_attention_prefill_fused_h_o,
                        linear_attention_prefill_fused_u_h_o=args.linear_attention_prefill_fused_u_h_o,
                        linear_attention_chunk_gdn_internal_timing=args.linear_attention_chunk_gdn_internal_timing,
                        rmsnorm_variant=args.rmsnorm_variant,
                        full_attention_variant=args.full_attention_variant,
                        full_attention_proj_variant=args.full_attention_proj_variant,
                        full_attention_norm_rope_variant=args.full_attention_norm_rope_variant,
                        full_attention_kv_cache_layout=args.full_attention_kv_cache_layout,
                        full_attention_fused_gate_o_proj=args.full_attention_fused_gate_o_proj,
                        full_attention_fused_norm_rope_kv_write=args.full_attention_fused_norm_rope_kv_write,
                        shared_expert_overlap_stream_priority=args.shared_expert_overlap_stream_priority,
                        lm_head_variant=args.lm_head_variant,
                        shared_expert_proj_variant=args.shared_expert_proj_variant,
                        include_shared_expert=args.include_shared_expert,
                        input_token_ids=input_token_ids,
                        input_text=args.input_text,
                        include_lm_head=args.include_lm_head,
                        decode_output_token=args.decode_output_token,
                        logit_topk=args.logit_topk,
                        attention_substage_timing=args.attention_substage_timing,
                        moe_substage_timing=args.moe_substage_timing,
                        collect_resident_stage_timeline=args.collect_resident_stage_timeline,
                        measurement_mode=args.measurement_mode,
                        decode_loop_steps=args.decode_loop_steps,
                        decode_sampling=args.decode_sampling,
                        sampling_temperature=args.sampling_temperature,
                        sampling_top_k=args.sampling_top_k,
                        decode_stop_token_ids=decode_stop_token_ids,
                        decode_loop_fast_housekeeping=args.decode_loop_fast_housekeeping,
                        decode_loop_defer_token_cpu_sync=args.defer_decode_token_cpu_sync,
                        decode_loop_token_cpu_sync_interval=decode_loop_token_cpu_sync_interval,
                        decode_loop_diagnostic=args.decode_loop_diagnostic,
                        overlap_decode_state_promotion_lm_head=args.overlap_decode_state_promotion_lm_head,
                        attention_cluster_timing=args.attention_cluster_timing,
                        attention_event_timing=args.attention_event_timing,
                        retained_attention_fast_path=args.retained_attention_fast_path,
                        skip_layer_dispatch_metadata=args.skip_layer_dispatch_metadata,
                        cuda_graph_replay_timing=args.cuda_graph_replay_timing,
                        moe_overlap_event_timing=args.moe_overlap_event_timing,
                        reuse_tensor_cache=contract["reuse_tensor_cache"],
                    )
                )
            result["measurement"] = measurements[-1]
            if args.resident_cache_probe_repeats > 1:
                result["resident_cache_probe"] = {
                    "repeats": args.resident_cache_probe_repeats,
                    "runs": [
                        {
                            "run_index": index,
                            "engine_wall_time_ms": measurement.get("engine_wall_time_ms"),
                            "engine_stage_wall_time_ms": measurement.get("engine_stage_wall_time_ms"),
                            "engine_tensor_cache": measurement.get("engine_tensor_cache"),
                            "pipeline": measurement.get("pipeline"),
                            "decode_loop": measurement.get("decode_loop"),
                            "text_smoke": measurement.get("text_smoke"),
                        }
                        for index, measurement in enumerate(measurements)
                    ],
                }
        captured = captured_stdout.getvalue()
        if captured:
            result["captured_stdout"] = captured
            print(captured, file=sys.stderr, end="")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
