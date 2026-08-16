// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_decode_runner.h"

#include "aima/native_full_layer.h"
#include "aima/native_linear_layer.h"
#include "aima/native_lm_head_certificate.h"

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <chrono>
#include <stdexcept>
#include <string>

namespace aima {
namespace {

constexpr std::size_t kVocabulary = 248320;
constexpr std::size_t kHidden = 2048;
constexpr std::size_t kRotaryHalfDimension = 32;
constexpr std::size_t kRotaryDimension = 64;
constexpr std::size_t kMaximumPosition = 262144;
constexpr float kRopeTheta = 10000000.0f;

void check_hip(hipError_t status, const char* operation) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(status));
  }
}

__global__ void next_token_embedding_kernel(
    const __hip_bfloat16* embedding, std::uint32_t token_id,
    __hip_bfloat16* hidden) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= kHidden) return;
  hidden[index] = embedding[static_cast<std::size_t>(token_id) * kHidden +
                            index];
}

__global__ void decode_rotary_kernel(std::size_t position, float* cosine,
                                     float* sine, bool round_through_bf16) {
  const std::size_t index = threadIdx.x;
  if (index >= kRotaryHalfDimension) return;
  const float exponent =
      static_cast<float>(2 * index) / static_cast<float>(kRotaryDimension);
  const float inverse_frequency = 1.0f / powf(kRopeTheta, exponent);
  const float angle = static_cast<float>(position) * inverse_frequency;
  const float cosine_value = cosf(angle);
  const float sine_value = sinf(angle);
  // VL current-vLLM converts its FP32 cache to the BF16 query dtype before
  // the captured Triton consumer promotes it back to FP32. Text keeps the
  // pre-existing scalar FP32 table until the independent G3 lane qualifies
  // any change to that path.
  cosine[index] = round_through_bf16
                      ? __bfloat162float(__float2bfloat16(cosine_value))
                      : cosine_value;
  sine[index] = round_through_bf16
                    ? __bfloat162float(__float2bfloat16(sine_value))
                    : sine_value;
}

NativeDecodePrepareMetrics prepare_native_decode_step_impl(
    std::size_t position, std::size_t rotary_position,
    std::uint32_t input_token_id, bool round_rotary_through_bf16,
    const NativeWeightStore& weights,
    const NativeDecodeInvocations& invocations, void* stream_value) {
  const NativeTensorView* embedding =
      weights.find("model.language_model.embed_tokens.weight");
  if (position >= kMaximumPosition || rotary_position >= kMaximumPosition ||
      input_token_id >= kVocabulary ||
      embedding == nullptr || embedding->device_pointer == nullptr ||
      embedding->payload_bytes !=
          kVocabulary * kHidden * sizeof(__hip_bfloat16)) {
    throw std::invalid_argument("native decode step preparation is invalid");
  }
  void* hidden = invocations.tensor_pointer(0, "x");
  void* cosine = invocations.tensor_pointer(32, "cos");
  void* sine = invocations.tensor_pointer(32, "sin");
  hipStream_t stream = static_cast<hipStream_t>(stream_value);
  hipLaunchKernelGGL(
      next_token_embedding_kernel, dim3(kHidden / 256), dim3(256), 0, stream,
      static_cast<const __hip_bfloat16*>(embedding->device_pointer),
      input_token_id, static_cast<__hip_bfloat16*>(hidden));
  check_hip(hipGetLastError(), "next_token_embedding_kernel");
  hipLaunchKernelGGL(
      decode_rotary_kernel, dim3(1), dim3(32), 0, stream, rotary_position,
      static_cast<float*>(cosine), static_cast<float*>(sine),
      round_rotary_through_bf16);
  check_hip(hipGetLastError(), "decode_rotary_kernel");
  return {2, position, rotary_position, input_token_id};
}

}  // namespace

NativeDecodePrepareMetrics prepare_native_decode_step(
    std::size_t position, std::uint32_t input_token_id,
    const NativeWeightStore& weights,
    const NativeDecodeInvocations& invocations, void* stream_value) {
  return prepare_native_decode_step_impl(
      position, position, input_token_id, false, weights, invocations,
      stream_value);
}

NativeDecodePrepareMetrics prepare_native_decode_step(
    std::size_t position, std::size_t rotary_position,
    std::uint32_t input_token_id, const NativeWeightStore& weights,
    const NativeDecodeInvocations& invocations, void* stream_value) {
  return prepare_native_decode_step_impl(
      position, rotary_position, input_token_id, true, weights, invocations,
      stream_value);
}

NativeLmHeadTop1Metrics run_native_lm_head_top1(
    const void* final_hidden_row, const NativeWeightStore& weights,
    const NativeLmHeadStore& lm_head,
    const NativeDecodeWorkspace& workspace,
    const NativeDecodeInvocations& invocations,
    NativeDecodeExecutor& executor, int cu_count,
    const std::uint8_t* allowed_token_mask, void* stream_value) {
  const auto& launches = invocations.launches();
  if (final_hidden_row == nullptr || launches.size() != 402 ||
      !executor.loaded() || !lm_head.built() || cu_count <= 0) {
    throw std::runtime_error("native LM-head ownership is incomplete");
  }
  hipStream_t stream = static_cast<hipStream_t>(stream_value);
  const auto started = std::chrono::steady_clock::now();
  void* final_norm_input = invocations.tensor_pointer(400, "x");
  if (final_norm_input != final_hidden_row) {
    check_hip(hipMemcpyAsync(final_norm_input, final_hidden_row,
                             kHidden * sizeof(__hip_bfloat16),
                             hipMemcpyDeviceToDevice, stream),
              "hipMemcpyAsync native LM-head input");
  }
  executor.launch(launches[400], stream);
  executor.launch(launches[401], stream);

  const char* logits_binding = launches[401].launch->arguments[3].binding;
  const NativeDecodeWorkspaceView* logits = workspace.find(logits_binding);
  const NativeDecodeWorkspaceView* final_hidden =
      workspace.find("rmsnorm_final_output");
  const NativeDecodeWorkspaceView* candidate_weights =
      workspace.find("native.lm_head.candidate_weights");
  const NativeDecodeWorkspaceView* candidate_logits =
      workspace.find("native.lm_head.candidate_logits");
  const NativeDecodeWorkspaceView* certificate_scratch =
      workspace.find("native.lm_head.certificate_scratch");
  const NativeTensorView* raw_lm_head = weights.find("lm_head.weight");
  if (logits == nullptr || logits->device_pointer == nullptr ||
      logits->payload_bytes < kVocabulary * sizeof(float) ||
      final_hidden == nullptr || candidate_weights == nullptr ||
      candidate_logits == nullptr || certificate_scratch == nullptr ||
      raw_lm_head == nullptr || final_hidden->device_pointer == nullptr ||
      raw_lm_head->device_pointer == nullptr) {
    throw std::runtime_error(
        "native LM-head certificate bindings are incomplete");
  }
  const NativeLmHeadCertificateLaunchMetrics certificate =
      launch_native_lm_head_certificate(
          raw_lm_head->device_pointer, lm_head.residual_l2(),
          final_hidden->device_pointer, logits->device_pointer,
          candidate_weights->device_pointer, candidate_weights->payload_bytes,
          candidate_logits->device_pointer, candidate_logits->payload_bytes,
          certificate_scratch->device_pointer,
          certificate_scratch->payload_bytes, cu_count, allowed_token_mask,
          stream);
  check_hip(hipStreamSynchronize(stream),
            "hipStreamSynchronize native LM-head");
  NativeLmHeadCertificateWire host_certificate{};
  check_hip(hipMemcpy(&host_certificate, certificate.device_wire,
                      sizeof(host_certificate), hipMemcpyDeviceToHost),
            "hipMemcpy native LM-head certificate");

  NativeLmHeadTop1Metrics metrics;
  metrics.aot_launches = 2;
  metrics.native_lm_head_certificate_launches =
      certificate.native_kernel_launches;
  metrics.candidate_count = host_certificate.candidate_count;
  metrics.certified = host_certificate.overflow == 0 &&
                      host_certificate.candidate_count != 0 &&
                      host_certificate.candidate_count <=
                          kNativeLmHeadCandidateCapacity;
  metrics.top1_token_id = host_certificate.exact_top1_token_id;
  metrics.top1_logit = host_certificate.exact_top1_logit;
  if (!metrics.certified) {
    throw std::runtime_error("native LM-head top-1 certificate failed");
  }
  metrics.synchronized_wall_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - started)
          .count();
  return metrics;
}

NativeDecodeRunMetrics run_native_decode_token(
    std::size_t position, std::size_t cache_end,
    const NativeWeightStore& weights, const NativeLmHeadStore& lm_head,
    const NativeDecodeWorkspace& workspace,
    NativeDecodeInvocations& invocations,
    NativeDecodeExecutor& executor, NativeFullAttentionState& attention_state,
    int cu_count, const std::uint8_t* allowed_token_mask,
    void* stream_value, const NativeDecodeLayerObserver* layer_observer,
    const NativeDecodeLinearLayer0Observer* linear_layer0_observer,
    const NativeDecodeLinearLayer0Observer* layer0_tail_observer,
    const NativeDecodeFullAttentionObserver* full_attention_observer,
    bool use_mrope) {
  const auto& launches = invocations.launches();
  if (launches.size() != 402 || !executor.loaded() || !lm_head.built() ||
      cu_count <= 0 ||
      cache_end != position + 1) {
    throw std::runtime_error("native decode runner ownership is incomplete");
  }
  hipStream_t stream = static_cast<hipStream_t>(stream_value);
  NativeDecodeRunMetrics metrics;
  const auto started = std::chrono::steady_clock::now();
  for (std::size_t layer_index = 0; layer_index < 40; ++layer_index) {
    const std::size_t base = layer_index * 10;
    if (std::string(launches[base + 1].launch->symbol) ==
        "triton_fused_input_proj_conv_kernel") {
      const NativeLinearLayerMetrics layer = run_native_linear_layer(
          layer_index, weights, workspace, invocations, executor, cu_count,
          stream, false,
          layer_index == 0 ? linear_layer0_observer : nullptr,
          layer_index == 0 ? layer0_tail_observer : nullptr);
      ++metrics.linear_layer_count;
      metrics.aot_launches += layer.aot_launches;
      metrics.native_projection_launches += layer.native_projection_launches;
      metrics.native_pointwise_launches += layer.native_pointwise_launches;
    } else {
      const NativeFullLayerMetrics layer = run_native_full_layer(
          layer_index, position, cache_end, weights, workspace, invocations,
          executor, attention_state, cu_count, stream, false,
          use_mrope,
          full_attention_observer);
      ++metrics.full_layer_count;
      metrics.aot_launches += layer.aot_launches;
      metrics.native_attention_launches += layer.native_attention_launches;
      metrics.native_projection_launches += layer.native_projection_launches;
      metrics.native_pointwise_launches += layer.native_pointwise_launches;
    }
    ++metrics.layer_count;
    if (layer_observer != nullptr) {
      (*layer_observer)(layer_index,
                        invocations.tensor_pointer(base + 10, "x"));
    }
  }
  metrics.layer_submission_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - started)
          .count();
  metrics.resident_state_pointer_swaps =
      invocations.swap_linear_decode_conv_state_buffers();
  const NativeLmHeadTop1Metrics lm_head_result = run_native_lm_head_top1(
      invocations.tensor_pointer(400, "x"), weights, lm_head, workspace,
      invocations, executor, cu_count, allowed_token_mask, stream);
  if (layer_observer != nullptr) {
    const NativeDecodeWorkspaceView* final_norm =
        workspace.find("rmsnorm_final_output");
    if (final_norm == nullptr || final_norm->device_pointer == nullptr ||
        final_norm->payload_bytes < kHidden * sizeof(__hip_bfloat16)) {
      throw std::runtime_error(
          "native decode final-norm observer binding is incomplete");
    }
    (*layer_observer)(40, final_norm->device_pointer);
  }
  metrics.aot_launches += lm_head_result.aot_launches;
  metrics.native_lm_head_certificate_launches =
      lm_head_result.native_lm_head_certificate_launches;
  metrics.lm_head_candidate_count = lm_head_result.candidate_count;
  metrics.lm_head_certified = lm_head_result.certified;
  metrics.top1_token_id = lm_head_result.top1_token_id;
  metrics.top1_logit = lm_head_result.top1_logit;
  metrics.synchronized_wall_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - started)
          .count();
  return metrics;
}

}  // namespace aima
