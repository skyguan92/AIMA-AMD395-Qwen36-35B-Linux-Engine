// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_vl_unified_attention.h"

#include "aima/aot_registry.h"
#include "aima/native_decode_executor.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace aima {
namespace {

constexpr char kPrefillKernelHash[] =
    "85618d461d690f5f7732dfd55b693df8c15642737aa2c1cf66b0674ffd4d7a30";
constexpr char kPrefillKernelSymbol[] = "kernel_unified_attention_2d";
constexpr char kDecodeAttentionKernelHash[] =
    "57514aea3981e5fba3e25d46b9dd62fb311a3acfef9ca9ad8fa99b0076c61402";
constexpr char kDecodeAttentionKernelSymbol[] =
    "kernel_unified_attention_3d";
constexpr char kDecodeReduceKernelHash[] =
    "6ecf435e2f5f8cfa2805d7433192f64a5e5e749e7a93c3ed4ae39e50921fe078";
constexpr char kDecodeReduceKernelSymbol[] = "reduce_segments";
constexpr std::size_t kCacheBlockTokens = 1056;
constexpr std::uint32_t kKvHeads = 2;
constexpr std::uint32_t kQueryHeads = 16;
constexpr std::uint32_t kBlockQ = 2;
constexpr std::uint32_t kDecodeSoftmaxSegments = 16;
constexpr std::uint32_t kDecodeSequenceThreshold3d = 64;
constexpr std::int64_t kQueryStride0 = 4096;
constexpr std::int64_t kQueryStride1 = 256;
constexpr std::int64_t kCacheStride0 =
    static_cast<std::int64_t>(kCacheBlockTokens) * 2 * 256;
constexpr std::int64_t kCacheStride1 = 2 * 256;
constexpr std::int64_t kCacheStride2 = 256;
constexpr std::size_t kAlignment = 256;
constexpr std::size_t kDecodeSegmentOutputBytes =
    kDecodeSequenceThreshold3d * kQueryHeads * kDecodeSoftmaxSegments * 256 *
    sizeof(float);
constexpr std::size_t kDecodeSegmentStatisticBytes =
    kDecodeSequenceThreshold3d * kQueryHeads * kDecodeSoftmaxSegments *
    sizeof(float);

void check_hip(hipError_t status, const char* operation) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorName(status) + " (" +
                             hipGetErrorString(status) + ")");
  }
}

std::size_t align_up(std::size_t value) {
  return (value + kAlignment - 1) / kAlignment * kAlignment;
}

}  // namespace

struct NativeVlUnifiedAttentionPlan::Impl {
  Impl(NativeDecodeExecutor& executor_value, std::size_t query_limit,
       std::size_t kv_limit, int device_value)
      : executor(&executor_value), device(device_value) {
    if (!executor_value.loaded() || query_limit == 0 || kv_limit == 0 ||
        query_limit > kv_limit ||
        kv_limit > 262144 ||
        kv_limit > static_cast<std::size_t>(
                       std::numeric_limits<std::int32_t>::max())) {
      throw std::invalid_argument(
          "native VL unified-attention limits are unsupported");
    }
    const EmbeddedAotImage* prefill_image =
        find_embedded_aot_image(kPrefillKernelHash);
    const EmbeddedAotImage* decode_attention_image =
        find_embedded_aot_image(kDecodeAttentionKernelHash);
    const EmbeddedAotImage* decode_reduce_image =
        find_embedded_aot_image(kDecodeReduceKernelHash);
    if (prefill_image == nullptr ||
        std::string(prefill_image->symbol) != kPrefillKernelSymbol ||
        prefill_image->num_warps != 4 || prefill_image->warp_size != 32 ||
        prefill_image->shared_memory_bytes != 32768 ||
        prefill_image->image_bytes != 34608 ||
        decode_attention_image == nullptr ||
        std::string(decode_attention_image->symbol) !=
            kDecodeAttentionKernelSymbol ||
        decode_attention_image->num_warps != 4 ||
        decode_attention_image->warp_size != 32 ||
        decode_attention_image->shared_memory_bytes != 16384 ||
        decode_attention_image->image_bytes != 18736 ||
        decode_reduce_image == nullptr ||
        std::string(decode_reduce_image->symbol) !=
            kDecodeReduceKernelSymbol ||
        decode_reduce_image->num_warps != 4 ||
        decode_reduce_image->warp_size != 32 ||
        decode_reduce_image->shared_memory_bytes != 2048 ||
        decode_reduce_image->image_bytes != 7968) {
      throw std::runtime_error(
          "native VL unified-attention embedded image is missing or changed");
    }

    metrics.image_bytes = prefill_image->image_bytes +
                          decode_attention_image->image_bytes +
                          decode_reduce_image->image_bytes;
    metrics.max_query_tokens = query_limit;
    metrics.max_kv_tokens = kv_limit;
    metrics.cache_blocks =
        (kv_limit + kCacheBlockTokens - 1) / kCacheBlockTokens;

    const std::size_t block_table_bytes =
        metrics.cache_blocks * sizeof(std::int32_t);
    const std::size_t seq_length_bytes =
        (kv_limit + 1) * sizeof(std::int32_t);
    const std::size_t query_start_bytes =
        2 * (query_limit + 1) * sizeof(std::int32_t);
    block_table_offset = 0;
    seq_length_offset = align_up(block_table_bytes);
    query_start_offset = align_up(seq_length_offset + seq_length_bytes);
    metrics.metadata_bytes = align_up(query_start_offset + query_start_bytes);
    segment_output_offset = 0;
    segment_max_offset = align_up(kDecodeSegmentOutputBytes);
    segment_expsum_offset =
        align_up(segment_max_offset + kDecodeSegmentStatisticBytes);
    metrics.decode_scratch_bytes = align_up(
        segment_expsum_offset + kDecodeSegmentStatisticBytes);

    std::vector<std::int32_t> metadata(
        metrics.metadata_bytes / sizeof(std::int32_t), 0);
    auto* block_table = metadata.data() +
                        block_table_offset / sizeof(std::int32_t);
    for (std::size_t block = 0; block < metrics.cache_blocks; ++block) {
      block_table[block] = static_cast<std::int32_t>(block);
    }
    auto* seq_lengths = metadata.data() +
                        seq_length_offset / sizeof(std::int32_t);
    for (std::size_t length = 0; length <= kv_limit; ++length) {
      seq_lengths[length] = static_cast<std::int32_t>(length);
    }
    auto* query_starts = metadata.data() +
                         query_start_offset / sizeof(std::int32_t);
    for (std::size_t length = 0; length <= query_limit; ++length) {
      query_starts[2 * length] = 0;
      query_starts[2 * length + 1] = static_cast<std::int32_t>(length);
    }

    check_hip(hipSetDevice(device),
              "hipSetDevice native VL unified attention");
    try {
      check_hip(hipMalloc(&metadata_device, metrics.metadata_bytes),
                "hipMalloc native VL unified attention metadata");
      check_hip(hipMemcpy(metadata_device, metadata.data(),
                          metrics.metadata_bytes, hipMemcpyHostToDevice),
                "hipMemcpy native VL unified attention metadata");
      const float descales[] = {1.0f, 1.0f};
      check_hip(hipMalloc(&decode_descale_device, sizeof(descales)),
                "hipMalloc native VL decode attention descale");
      check_hip(hipMemcpy(decode_descale_device, descales, sizeof(descales),
                          hipMemcpyHostToDevice),
                "hipMemcpy native VL decode attention descale");
      check_hip(hipMalloc(&decode_scratch_device,
                          metrics.decode_scratch_bytes),
                "hipMalloc native VL decode attention scratch");
      metrics.loaded = true;
    } catch (...) {
      release();
      throw;
    }
  }

  ~Impl() { release(); }

  void release() noexcept {
    if (metadata_device != nullptr || decode_descale_device != nullptr ||
        decode_scratch_device != nullptr) {
      const hipError_t set_status = hipSetDevice(device);
      static_cast<void>(set_status);
      if (decode_scratch_device != nullptr) {
        const hipError_t free_status = hipFree(decode_scratch_device);
        static_cast<void>(free_status);
      }
      if (decode_descale_device != nullptr) {
        const hipError_t free_status = hipFree(decode_descale_device);
        static_cast<void>(free_status);
      }
      if (metadata_device != nullptr) {
        const hipError_t free_status = hipFree(metadata_device);
        static_cast<void>(free_status);
      }
      metadata_device = nullptr;
      decode_descale_device = nullptr;
      decode_scratch_device = nullptr;
    }
    metrics.loaded = false;
  }

  NativeDecodeExecutor* executor = nullptr;
  int device = 0;
  std::size_t block_table_offset = 0;
  std::size_t seq_length_offset = 0;
  std::size_t query_start_offset = 0;
  std::size_t segment_output_offset = 0;
  std::size_t segment_max_offset = 0;
  std::size_t segment_expsum_offset = 0;
  void* metadata_device = nullptr;
  void* decode_descale_device = nullptr;
  void* decode_scratch_device = nullptr;
  NativeVlUnifiedAttentionMetrics metrics;
};

NativeVlUnifiedAttentionPlan::NativeVlUnifiedAttentionPlan(
    NativeDecodeExecutor& executor, std::size_t max_query_tokens,
    std::size_t max_kv_tokens, int device)
    : impl_(std::make_unique<Impl>(executor, max_query_tokens,
                                   max_kv_tokens, device)) {}

NativeVlUnifiedAttentionPlan::~NativeVlUnifiedAttentionPlan() = default;
NativeVlUnifiedAttentionPlan::NativeVlUnifiedAttentionPlan(
    NativeVlUnifiedAttentionPlan&&) noexcept = default;
NativeVlUnifiedAttentionPlan& NativeVlUnifiedAttentionPlan::operator=(
    NativeVlUnifiedAttentionPlan&&) noexcept = default;

void NativeVlUnifiedAttentionPlan::launch(
    const void* query_bf16, const void* key_cache_bf16,
    const void* value_cache_bf16, void* output_bf16,
    std::size_t query_tokens, std::size_t kv_tokens, void* stream_pointer) {
  if (!impl_ || !impl_->metrics.loaded || impl_->executor == nullptr ||
      query_bf16 == nullptr || key_cache_bf16 == nullptr ||
      value_cache_bf16 == nullptr || output_bf16 == nullptr ||
      output_bf16 == query_bf16 || output_bf16 == key_cache_bf16 ||
      output_bf16 == value_cache_bf16 || query_tokens == 0 ||
      query_tokens > impl_->metrics.max_query_tokens ||
      kv_tokens < query_tokens || kv_tokens > impl_->metrics.max_kv_tokens) {
    throw std::invalid_argument(
        "native VL unified-attention launch geometry is invalid");
  }

  auto* metadata = static_cast<unsigned char*>(impl_->metadata_device);
  void* output = output_bf16;
  void* query = const_cast<void*>(query_bf16);
  void* key_cache = const_cast<void*>(key_cache_bf16);
  void* value_cache = const_cast<void*>(value_cache_bf16);
  void* block_table = metadata + impl_->block_table_offset;
  void* seq_lengths =
      metadata + impl_->seq_length_offset +
      kv_tokens * sizeof(std::int32_t);
  void* query_starts =
      metadata + impl_->query_start_offset +
      2 * query_tokens * sizeof(std::int32_t);

  float scale = 0.0625f;
  float out_scale = 1.0f;
  float softcap = 0.0f;
  std::int64_t block_table_stride =
      static_cast<std::int64_t>(impl_->metrics.cache_blocks);
  std::int64_t query_stride_0 = kQueryStride0;
  std::int64_t query_stride_1 = kQueryStride1;
  std::int64_t output_stride_0 = kQueryStride0;
  std::int64_t output_stride_1 = kQueryStride1;
  std::int64_t qq_bias_stride_0 = 0;
  std::int64_t stride_k_cache_0 = kCacheStride0;
  std::int64_t stride_k_cache_1 = kCacheStride1;
  std::int64_t stride_k_cache_2 = kCacheStride2;
  std::int64_t stride_v_cache_0 = kCacheStride0;
  std::int64_t stride_v_cache_1 = kCacheStride1;
  std::int64_t stride_v_cache_2 = kCacheStride2;
  std::int32_t num_seqs = 1;
  std::int32_t zero_stride = 0;

  std::vector<void*> parameters{
      &output,
      &query,
      &key_cache,
      &value_cache,
      &block_table,
      &seq_lengths,
      &scale,
      &out_scale,
      &softcap,
      &block_table_stride,
      &query_stride_0,
      &query_stride_1,
      &output_stride_0,
      &output_stride_1,
      &qq_bias_stride_0,
      &stride_k_cache_0,
      &stride_k_cache_1,
      &stride_k_cache_2,
      &stride_v_cache_0,
      &stride_v_cache_1,
      &stride_v_cache_2,
      &query_starts,
      &num_seqs,
      &zero_stride,
      &zero_stride,
      &zero_stride,
      &zero_stride,
      &zero_stride,
      &zero_stride,
  };
  AotLaunchConfig config;
  config.grid_x = static_cast<std::uint32_t>(
      (query_tokens + kBlockQ - 1) / kBlockQ);
  config.grid_y = kKvHeads;
  config.grid_z = 1;
  config.num_warps = 4;
  config.warp_size = 32;
  config.shared_memory_bytes = 32768;
  impl_->executor->launch_embedded(kPrefillKernelHash, config, parameters,
                                   stream_pointer);
  ++impl_->metrics.launches;
}

void NativeVlUnifiedAttentionPlan::launch_decode(
    const void* query_bf16, const void* key_cache_bf16,
    const void* value_cache_bf16, void* output_bf16,
    std::size_t kv_tokens, void* stream_pointer) {
  if (!impl_ || !impl_->metrics.loaded || impl_->executor == nullptr ||
      query_bf16 == nullptr || key_cache_bf16 == nullptr ||
      value_cache_bf16 == nullptr || output_bf16 == nullptr ||
      output_bf16 == query_bf16 || output_bf16 == key_cache_bf16 ||
      output_bf16 == value_cache_bf16 || kv_tokens == 0 ||
      kv_tokens > impl_->metrics.max_kv_tokens) {
    throw std::invalid_argument(
        "native VL unified-attention decode geometry is invalid");
  }

  auto* metadata = static_cast<unsigned char*>(impl_->metadata_device);
  auto* scratch = static_cast<unsigned char*>(impl_->decode_scratch_device);
  void* segment_output = scratch + impl_->segment_output_offset;
  void* segment_max = scratch + impl_->segment_max_offset;
  void* segment_expsum = scratch + impl_->segment_expsum_offset;
  void* query = const_cast<void*>(query_bf16);
  void* key_cache = const_cast<void*>(key_cache_bf16);
  void* value_cache = const_cast<void*>(value_cache_bf16);
  void* output = output_bf16;
  void* block_table = metadata + impl_->block_table_offset;
  void* seq_lengths = metadata + impl_->seq_length_offset +
                      kv_tokens * sizeof(std::int32_t);
  void* query_starts = metadata + impl_->query_start_offset +
                       2 * sizeof(std::int32_t);
  void* k_descale = impl_->decode_descale_device;
  void* v_descale = impl_->decode_descale_device;

  float scale = 0.0625f;
  float softcap = 0.0f;
  float out_scale_inverse = 1.0f;
  const std::size_t logical_blocks =
      (kv_tokens + kCacheBlockTokens - 1) / kCacheBlockTokens;
  std::int64_t block_table_stride =
      static_cast<std::int64_t>(logical_blocks);
  std::int64_t query_stride_0 = kQueryStride0;
  std::int64_t query_stride_1 = kQueryStride1;
  std::int64_t output_stride_0 = kQueryStride0;
  std::int64_t output_stride_1 = kQueryStride1;
  std::int64_t qq_bias_stride_0 = 0;
  std::int64_t stride_k_cache_0 = kCacheStride0;
  std::int64_t stride_k_cache_1 = kCacheStride1;
  std::int64_t stride_k_cache_2 = kCacheStride2;
  std::int64_t stride_v_cache_0 = kCacheStride0;
  std::int64_t stride_v_cache_1 = kCacheStride1;
  std::int64_t stride_v_cache_2 = kCacheStride2;
  std::int32_t num_seqs = 1;
  std::int32_t zero_stride = 0;

  std::vector<void*> attention_parameters{
      &segment_output,
      &segment_max,
      &segment_expsum,
      &query,
      &key_cache,
      &value_cache,
      &block_table,
      &seq_lengths,
      &scale,
      &k_descale,
      &v_descale,
      &softcap,
      &block_table_stride,
      &query_stride_0,
      &query_stride_1,
      &qq_bias_stride_0,
      &stride_k_cache_0,
      &stride_k_cache_1,
      &stride_k_cache_2,
      &stride_v_cache_0,
      &stride_v_cache_1,
      &stride_v_cache_2,
      &query_starts,
      &num_seqs,
      &zero_stride,
      &zero_stride,
      &zero_stride,
      &zero_stride,
      &zero_stride,
      &zero_stride,
  };
  if (attention_parameters.size() != 30) {
    throw std::runtime_error(
        "native segmented attention regular ABI argument count changed");
  }
  impl_->executor->launch_embedded(
      kDecodeAttentionKernelHash,
      AotLaunchConfig{1, 2, 16, 4, 32, 16384}, attention_parameters,
      stream_pointer);

  std::vector<void*> reduce_parameters{
      &output,
      &segment_output,
      &segment_max,
      &segment_expsum,
      &seq_lengths,
      &out_scale_inverse,
      &output_stride_0,
      &output_stride_1,
      &block_table_stride,
      &query_starts,
  };
  if (reduce_parameters.size() != 10) {
    throw std::runtime_error(
        "native attention reduce regular ABI argument count changed");
  }
  impl_->executor->launch_embedded(
      kDecodeReduceKernelHash, AotLaunchConfig{1, 16, 1, 4, 32, 2048},
      reduce_parameters, stream_pointer);
  impl_->metrics.decode_launches += 2;
}

const NativeVlUnifiedAttentionMetrics&
NativeVlUnifiedAttentionPlan::metrics() const {
  if (!impl_) {
    throw std::runtime_error(
        "native VL unified-attention plan is not initialized");
  }
  return impl_->metrics;
}

}  // namespace aima
