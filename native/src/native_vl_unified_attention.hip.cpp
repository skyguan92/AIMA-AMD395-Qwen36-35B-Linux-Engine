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

constexpr char kKernelHash[] =
    "85618d461d690f5f7732dfd55b693df8c15642737aa2c1cf66b0674ffd4d7a30";
constexpr char kKernelSymbol[] = "kernel_unified_attention_2d";
constexpr std::size_t kCacheBlockTokens = 1056;
constexpr std::uint32_t kKvHeads = 2;
constexpr std::uint32_t kBlockQ = 2;
constexpr std::int64_t kQueryStride0 = 4096;
constexpr std::int64_t kQueryStride1 = 256;
constexpr std::int64_t kCacheStride0 =
    static_cast<std::int64_t>(kCacheBlockTokens) * 2 * 256;
constexpr std::int64_t kCacheStride1 = 2 * 256;
constexpr std::int64_t kCacheStride2 = 256;
constexpr std::size_t kAlignment = 256;

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
    const EmbeddedAotImage* image = find_embedded_aot_image(kKernelHash);
    if (image == nullptr || std::string(image->symbol) != kKernelSymbol ||
        image->num_warps != 4 || image->warp_size != 32 ||
        image->shared_memory_bytes != 32768 || image->image_bytes != 34608) {
      throw std::runtime_error(
          "native VL unified-attention embedded image is missing or changed");
    }

    metrics.image_bytes = image->image_bytes;
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
      metrics.loaded = true;
    } catch (...) {
      release();
      throw;
    }
  }

  ~Impl() { release(); }

  void release() noexcept {
    if (metadata_device != nullptr) {
      const hipError_t set_status = hipSetDevice(device);
      static_cast<void>(set_status);
      const hipError_t free_status = hipFree(metadata_device);
      static_cast<void>(free_status);
      metadata_device = nullptr;
    }
    metrics.loaded = false;
  }

  NativeDecodeExecutor* executor = nullptr;
  int device = 0;
  std::size_t block_table_offset = 0;
  std::size_t seq_length_offset = 0;
  std::size_t query_start_offset = 0;
  void* metadata_device = nullptr;
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
  config.grid_x = static_cast<std::uint32_t>(query_tokens / kBlockQ + 1);
  config.grid_y = kKvHeads;
  config.grid_z = 1;
  config.num_warps = 4;
  config.warp_size = 32;
  config.shared_memory_bytes = 32768;
  impl_->executor->launch_embedded(kKernelHash, config, parameters,
                                   stream_pointer);
  ++impl_->metrics.launches;
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
