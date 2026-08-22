#!/usr/bin/env python3
"""Define the two pinned kernels for exact-hybrid routed gate/up."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def hybrid_scalar_block_flag_kernel(
    hidden_ptr,
    weight_ptr,
    expert_ids_ptr,
    output_ptr,
    flagged_indices_ptr,
    flagged_count_ptr,
    HIDDEN_SIZE: tl.constexpr,
    OUTPUT_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    CHUNK_K: tl.constexpr,
    ERROR_COEFFICIENT: tl.constexpr,
    CORRECT_SUBNORMALS: tl.constexpr = False,
):
    rank = tl.program_id(0)
    pid_m = tl.program_id(1)
    expert = tl.load(expert_ids_ptr + rank).to(tl.int64)
    outputs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_k = tl.arange(0, CHUNK_K)
    hidden = tl.load(hidden_ptr + offsets_k).to(tl.float32)
    weight = tl.load(
        weight_ptr
        + (expert * OUTPUT_SIZE + outputs[:, None]) * HIDDEN_SIZE
        + offsets_k[None, :]
    ).to(tl.float32)
    products = hidden[None, :] * weight
    accumulator = tl.sum(products, axis=1)
    absolute_sum = tl.sum(tl.abs(products), axis=1)
    rounded_bf16 = accumulator.to(tl.bfloat16)
    rounded = rounded_bf16.to(tl.float32)
    magnitude = tl.abs(rounded)
    rounded_bits = rounded_bf16.to(tl.uint16, bitcast=True)
    negative = (rounded_bits & 0x8000) != 0
    previous_bits = tl.where(
        negative, rounded_bits + 1, rounded_bits - 1
    ).to(tl.uint16)
    following_bits = tl.where(
        negative, rounded_bits - 1, rounded_bits + 1
    ).to(tl.uint16)
    previous = previous_bits.to(tl.bfloat16, bitcast=True).to(tl.float32)
    following = following_bits.to(tl.bfloat16, bitcast=True).to(tl.float32)
    rounding_margin = tl.minimum(
        accumulator - (previous + rounded) * 0.5,
        (rounded + following) * 0.5 - accumulator,
    )
    may_cross_boundary = (
        ERROR_COEFFICIENT * absolute_sum >= rounding_margin
    ) | (CORRECT_SUBNORMALS & (magnitude < 1.0e-30))
    linear = rank * OUTPUT_SIZE + outputs
    tl.store(output_ptr + linear, rounded_bf16)
    slots = tl.atomic_add(
        flagged_count_ptr + outputs * 0,
        may_cross_boundary.to(tl.int32),
    )
    tl.store(
        flagged_indices_ptr + slots,
        linear,
        mask=may_cross_boundary,
    )


@triton.jit
def sparse_wmma_correction_kernel(
    hidden_ptr,
    weight_ptr,
    expert_ids_ptr,
    output_ptr,
    flagged_indices_ptr,
    flagged_count_ptr,
    HIDDEN_SIZE: tl.constexpr,
    OUTPUT_SIZE: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    flagged_count = tl.load(flagged_count_ptr)
    if pid * BLOCK_N >= flagged_count:
        return
    offsets_m = tl.arange(0, 16)
    offsets_n = tl.arange(0, BLOCK_N)
    list_offsets = pid * BLOCK_N + offsets_n
    valid = list_offsets < flagged_count
    encoded = tl.load(flagged_indices_ptr + list_offsets, mask=valid, other=0)
    ranks = encoded // OUTPUT_SIZE
    output_indices = encoded % OUTPUT_SIZE
    experts = tl.load(expert_ids_ptr + ranks, mask=valid, other=0).to(tl.int64)
    offsets_k = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((16, BLOCK_N), tl.float32)
    for block in range(0, tl.cdiv(HIDDEN_SIZE, BLOCK_K)):
        k = block * BLOCK_K + offsets_k
        hidden = tl.load(
            hidden_ptr + k[None, :],
            mask=(offsets_m[:, None] == 0) & (k[None, :] < HIDDEN_SIZE),
            other=0.0,
        )
        rows = experts * OUTPUT_SIZE + output_indices
        weight = tl.load(
            weight_ptr + rows[None, :] * HIDDEN_SIZE + k[:, None],
            mask=valid[None, :] & (k[:, None] < HIDDEN_SIZE),
            other=0.0,
        )
        accumulator += tl.dot(hidden, weight)
    store_mask = (offsets_m[:, None] == 0) & valid[None, :]
    tl.store(
        output_ptr
        + encoded[None, :]
        + offsets_m[:, None] * 0,
        accumulator,
        mask=store_mask,
    )
