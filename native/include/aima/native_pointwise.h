#pragma once

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include <cstddef>
#include <cstdint>

namespace aima {

// Uploads host token ids into a resident device buffer and materializes their
// BF16 embedding rows directly into the admitted layer-0 input.  The upload
// storage is owned by NativePrefillWorkspace, so requests do not allocate.
void launch_prompt_embeddings(const void* embedding_bf16,
                              const std::uint32_t* host_token_ids,
                              void* device_token_ids,
                              void* output_bf16,
                              std::size_t token_count,
                              void* stream = nullptr);

void launch_bf16_add(const void* left, const void* right, void* output,
                     std::size_t count, void* stream = nullptr);

// Writes the BF16-rounded left+right intermediate and then the BF16-rounded
// residual+intermediate result in one launch, preserving the two-add oracle.
void launch_bf16_add_pair(const void* left, const void* right,
                          const void* residual, void* intermediate,
                          void* output, std::size_t count,
                          void* stream = nullptr);

// Native fixed-width Gemma RMSNorm used by language prefill and the current
// vLLM linear-decode input boundary.  The checkpoint stores RMS weights as
// (scale - 1), matching the frozen `weight + 1` rule.
void launch_prefill_rmsnorm_2048(const void* input_bf16,
                                 const void* weight_bf16,
                                 void* output_bf16,
                                 std::size_t token_count,
                                 void* stream = nullptr);

// BF16-rounded residual add followed by the same fixed-width RMSNorm.
void launch_prefill_add_rmsnorm_2048(const void* input_bf16,
                                     const void* residual_bf16,
                                     const void* weight_bf16,
                                     void* residual_output_bf16,
                                     void* norm_output_bf16,
                                     std::size_t token_count,
                                     void* stream = nullptr);

void launch_shared_silu_multiply(const void* fused_shared_input,
                                 void* activated_512,
                                 void* stream = nullptr);

void launch_shared_sigmoid_scale(const void* fused_shared_input,
                                 const void* shared_down_2048,
                                 void* output_2048,
                                 void* stream = nullptr);

// The fused Q/Gate projection stores 16 rows of [Q(256), Gate(256)].
void launch_full_attention_sigmoid_gate(const void* attention_4096,
                                        const void* fused_q_gate_storage,
                                        void* gated_4096,
                                        void* stream = nullptr);

// Builds the exact q8192/position-0 64-dimensional rotary table used by the
// fixed Qwen3.6 full-attention prefill path.  cos/sin are FP32 [8192,32].
void launch_q8192_rotary_table(void* cosine_fp32, void* sine_fp32,
                               void* stream = nullptr);

void launch_prefill_rotary_table(void* cosine_fp32, void* sine_fp32,
                                 std::size_t token_count,
                                 std::size_t position_start = 0,
                                 void* stream = nullptr);

// Builds the Qwen3.6 language M-RoPE table from resident row-major
// int64 positions [3, position_row_stride].  Each selected T/H/W cache value
// is rounded through BF16 before being stored in the existing FP32 rotary
// workspace, matching vLLM's cache-to-model-dtype boundary without changing
// the already-qualified scalar-position text path.
void launch_prefill_mrope_rotary_table(
    void* cosine_fp32, void* sine_fp32,
    const void* positions_i64, std::size_t token_count,
    std::size_t position_row_stride = 0, void* stream = nullptr);

// q_gate is BF16 [8192,16,512] with Q in each head's first 256 values;
// k_raw is BF16 [8192,2,256].  Outputs are contiguous BF16 Q/K after the
// checkpoint's head RMSNorm and partial (64-dimension) RoPE.
void launch_q8192_full_attention_head_norm_rope(
    const void* q_gate, const void* k_raw,
    const void* q_norm_weight, const void* k_norm_weight,
    const void* cosine_fp32, const void* sine_fp32,
    void* q_output, void* k_output, void* stream = nullptr);

// Generic fixed-context form. q/k row strides are in BF16 elements, allowing
// q32768 to consume the single derived [tokens,9216] QKV projection directly.
void launch_full_attention_head_norm_rope_prefill(
    const void* q_gate, const void* k_raw, const void* v_raw,
    const void* q_norm_weight, const void* k_norm_weight,
    const void* cosine_fp32, const void* sine_fp32,
    void* q_output, void* k_output, void* v_output,
    std::size_t token_count, std::size_t q_row_stride,
    std::size_t k_row_stride, std::size_t v_row_stride,
    void* stream = nullptr);

// M-RoPE consumer for the pinned vLLM Triton arithmetic: each BF16 rotary
// product is truncated toward zero before the FP32 add/subtract and final
// BF16 RNE store.  The scalar-position text consumer above remains unchanged.
void launch_full_attention_head_norm_mrope_prefill(
    const void* q_gate, const void* k_raw, const void* v_raw,
    const void* q_norm_weight, const void* k_norm_weight,
    const void* cosine_fp32, const void* sine_fp32,
    void* q_output, void* k_output, void* v_output,
    std::size_t token_count, std::size_t q_row_stride,
    std::size_t k_row_stride, std::size_t v_row_stride,
    void* stream = nullptr);

// CK returns FP32 [8192,4096].  Preserve the frozen engine's two BF16
// boundaries: attention.to(BF16), then BF16 sigmoid(gate) multiplication.
void launch_q8192_full_attention_sigmoid_gate_f32(
    const void* attention_f32, const void* fused_q_gate_storage,
    void* attention_bf16, void* gated_bf16, void* stream = nullptr);

void launch_full_attention_sigmoid_gate_f32_prefill(
    const void* attention_f32, const void* fused_q_gate_storage,
    void* attention_bf16, void* gated_bf16, std::size_t token_count,
    std::size_t q_row_stride, void* stream = nullptr);

// The exact vLLM unified-attention AOT path already returns BF16. Preserve the
// same separately-rounded BF16 sigmoid boundary without converting through an
// artificial FP32 attention buffer.
void launch_full_attention_sigmoid_gate_bf16_prefill(
    const void* attention_bf16, const void* fused_q_gate_storage,
    void* gated_bf16, std::size_t token_count,
    std::size_t q_row_stride, void* stream = nullptr);

// Extracts the two 32-wide A/B projections from the fused q32768 linear input
// projection [QKV(8192), Z(4096), A(32), B(32)].
void launch_extract_linear_ab_fused(const void* fused_input_bf16,
                                    void* a_bf16, void* b_bf16,
                                    std::size_t token_count,
                                    std::size_t fused_row_stride = 12352,
                                    void* stream = nullptr);

// Reproduces the pinned vLLM RMSNormGated FP32 normalization and SiLU product,
// with a single BF16 boundary at the final output store.
void launch_linear_gated_norm_fused(
    const void* core_bf16, const void* fused_input_bf16,
    const void* norm_weight_bf16, void* output_bf16,
    std::size_t token_count, std::size_t fused_row_stride = 12352,
    void* stream = nullptr);

void launch_linear_gated_norm_separate(
    const void* core_bf16, const void* gate_bf16,
    const void* norm_weight_bf16, void* output_bf16,
    std::size_t token_count, void* stream = nullptr);

// Qualification-only mirror of the frozen PyTorch contiguous-width-128
// `pow(2).mean(-1)` reduction. The output is one FP32 variance per row.
void launch_bf16_rowwise_variance_128_pytorch(
    const void* rows_bf16, void* variance_fp32,
    std::size_t row_count, void* stream = nullptr);

// Computes rsqrt(mean(row^2) + epsilon) for BF16 rows of width 128.  This is
// the sole non-AOT reduction directly preceding the captured linear gated-norm
// kernel in q8192 prefill.
void launch_bf16_rowwise_invstd_128(const void* rows_bf16,
                                    void* invstd_fp32,
                                    std::size_t row_count,
                                    float epsilon = 1.0e-6f,
                                    void* stream = nullptr);

}  // namespace aima
