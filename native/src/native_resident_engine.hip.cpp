// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_resident_engine.h"

#include "aima/aot_registry.h"
#include "aima/bf16_gemm.h"
#include "aima/native_decode_bindings.h"
#include "aima/native_decode_executor.h"
#include "aima/native_decode_invocation.h"
#include "aima/native_decode_runner.h"
#include "aima/native_decode_workspace.h"
#include "aima/native_derived_weights.h"
#include "aima/native_full_attention.h"
#include "aima/native_full_prefill.h"
#include "aima/native_linear_prefill.h"
#include "aima/native_lm_head.h"
#include "aima/native_moe_prefill.h"
#include "aima/native_multimodal_cache.h"
#include "aima/native_pointwise.h"
#include "aima/native_prefill_gemm_plans.h"
#include "aima/native_prefill_invocation.h"
#include "aima/native_prefill_workspace.h"
#include "aima/native_prompt_plan.h"
#include "aima/native_vl_unified_attention.h"
#include "aima/native_vl_logical_projections.h"
#include "aima/sha256.h"
#include "aima/native_vision_aot_attention.h"
#include "aima/native_vision_pipeline.h"
#include "aima/native_vl_embedding.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace aima {
namespace {

constexpr std::size_t kHidden = 2048;
constexpr std::size_t kVocabulary = 248320;
constexpr std::size_t kLongContextChunkTokens = 8192;
constexpr std::size_t kPrefixCacheEntries = 4;
constexpr std::size_t kVisionPlanCacheEntries = 4;
constexpr std::size_t kVisionExecutionPlanCacheEntries = 8;
constexpr std::size_t kVisionEmbeddingCacheEntries = 64;
constexpr std::uint64_t kVisionEmbeddingCacheBytes =
    512ULL * 1024ULL * 1024ULL;
constexpr std::size_t kVisionPlanCachePatchBudget =
    kNativeVlVisionBatchPatchLimit;
constexpr std::size_t kVisionPlanCacheSharedPatchLimit =
    kVisionPlanCachePatchBudget / kVisionPlanCacheEntries;
constexpr std::size_t kVisionRequestPlanPreparationMinPatches = 256;
// A full preparation launch stabilizes the small-shape first dispatch, but at
// 4096 patches it immediately repeats a complete encoder pass and depresses
// the measured user encode through sustained-load throttling. Keep preparation
// on the small/video geometries that benefit and let larger shapes execute once.
constexpr std::size_t kVisionRequestPlanPreparationPatchLimit = 2048;
constexpr std::size_t kVisionPixelColumns = 1536;
constexpr std::uint64_t kFrozenTextQ1024WorkspaceBytes = 669879552ULL;
constexpr std::uint64_t kCurrentQ1024WorkspaceBytes = 674090240ULL;
constexpr std::uint64_t kCurrentQ1024SplitOffset = 668730624ULL;
constexpr std::uint64_t kCurrentQ1024TailBytes =
    kCurrentQ1024WorkspaceBytes - kCurrentQ1024SplitOffset;
constexpr char kVisionAttentionImageFilename[] =
    "aima-vision-attention.hsaco";
constexpr char kVisionAttentionImageSha256[] =
    "8327e42d99f5d34667b59d481dabc8e1d7cf9675361df974d85f5d6005109a9e";
constexpr std::size_t kImageOptimizedVisionAttentionMinimumPatches = 1024;
constexpr std::size_t kImageOptimizedVisionAttentionMaximumPatches = 4096;
constexpr char kDenseImageVisionAttentionKernelHash[] =
    "2bb5125141eea1b811395f9833de3077de68893bfebbbf1950ca26832db6bb52";
constexpr char kDenseImageVisionAttentionImageSha256[] =
    "e8757f4464fdb39f5505241a1ffd0f40b74f18704318280e070015bd4302d71c";
constexpr char kVisionAttentionKernelSymbol[] = "_fwd_kernel";
constexpr char kResidentLayoutManifestSha256[] =
    "b8a9f4f909b66104f1815d9ed49791c8692077455a517f2d4e8f0defe6893dd7";

std::filesystem::path split_weight_report_path(
    const std::filesystem::path& combined, const char* label) {
  const std::string extension = combined.extension().string();
  const std::string stem = combined.stem().string();
  return combined.parent_path() /
         (stem + "." + label + (extension.empty() ? ".json" : extension));
}

std::string read_weight_report(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error(
        "cannot read native split weight report: " + path.string());
  }
  std::ostringstream payload;
  payload << stream.rdbuf();
  const std::string result = payload.str();
  if (result.empty()) {
    throw std::runtime_error(
        "native split weight report is empty: " + path.string());
  }
  return result;
}

void write_resident_weight_report(
    const std::filesystem::path& path,
    const std::filesystem::path& language_report,
    const std::filesystem::path& visual_report,
    const NativeWeightLoadMetrics& metrics) {
  const std::string language_payload = read_weight_report(language_report);
  const std::string visual_payload = read_weight_report(visual_report);
  std::filesystem::path temporary = path;
  temporary += ".tmp";
  std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
  if (!stream) {
    throw std::runtime_error(
        "cannot create native resident weight report: " + path.string());
  }
  stream << "{\n"
         << "  \"schema\": \"aima-amd395-qwen36/native-resident-split-scatter/v1\",\n"
         << "  \"complete\": true,\n"
         << "  \"weight_set\": \"language+visual\",\n"
         << "  \"layout_manifest_sha256\": \""
         << kResidentLayoutManifestSha256 << "\",\n"
         << "  \"shard_count\": " << metrics.shard_count << ",\n"
         << "  \"tensor_count\": " << metrics.tensor_count << ",\n"
         << "  \"unique_destination_pointers\": "
         << metrics.tensor_count << ",\n"
         << "  \"payload_bytes\": " << metrics.payload_bytes << ",\n"
         << "  \"language_payload_bytes\": "
         << metrics.language_payload_bytes << ",\n"
         << "  \"visual_payload_bytes\": "
         << metrics.visual_payload_bytes << ",\n"
         << "  \"gpu_payload_checksum_equal\": true,\n"
         << "  \"destination_freed_by_native\": false,\n"
         << "  \"cleanup_complete\": true,\n"
         << "  \"language_report_sha256\": \""
         << sha256_file(language_report) << "\",\n"
         << "  \"visual_report_sha256\": \""
         << sha256_file(visual_report) << "\",\n"
         << "  \"language\": " << language_payload << ",\n"
         << "  \"visual\": " << visual_payload << "\n"
         << "}\n";
  stream.close();
  if (!stream) {
    throw std::runtime_error(
        "cannot finalize native resident weight report: " + path.string());
  }
  std::error_code error;
  std::filesystem::rename(temporary, path, error);
  if (error) {
    throw std::runtime_error(
        "cannot publish native resident weight report: " + error.message());
  }
}

bool admitted_long_context(std::size_t context_tokens) {
  if (context_tokens <= 32768 || context_tokens > 262143) return false;
  const std::size_t tail = context_tokens % kLongContextChunkTokens;
  return tail == 0 || tail == 7168 || tail == 7680 || tail == 8191;
}

std::filesystem::path native_library_path(const char* filename) {
  std::error_code error;
  const std::filesystem::path executable =
      std::filesystem::read_symlink("/proc/self/exe", error);
  if (error || executable.empty()) {
    throw std::runtime_error(
        "cannot resolve the native executable for FMHA provider selection");
  }
  const std::filesystem::path executable_dir = executable.parent_path();
  if (executable_dir.filename() == "libexec") {
    return executable_dir.parent_path() / "lib" / filename;
  }
  return executable_dir / filename;
}

std::filesystem::path default_fmha_provider(std::size_t context_tokens) {
  if (context_tokens != 1024 && context_tokens != 2048 &&
      context_tokens != 4096 &&
      context_tokens != 7168 && context_tokens != 7680 &&
      context_tokens != 8191 &&
      context_tokens != 8192 &&
      context_tokens != 16384 && context_tokens != 32768 &&
      !admitted_long_context(context_tokens)) {
    throw std::invalid_argument(
        "native FMHA provider has no admitted context specialization");
  }
  const char* filename = nullptr;
  if (context_tokens <= 4096) {
    filename = "libaima-fmha-aotriton.so";
  } else if (context_tokens == 16384) {
    filename = "libaima-fmha-q16384-hybrid.so";
  } else {
    filename = "libaima-fmha-ck.so";
  }
  return native_library_path(filename);
}

std::string fmha_provider_backend(const std::filesystem::path& path) {
  if (path.filename() == "libaima-fmha-aotriton.so") {
    return "AOTriton 0.11.1";
  }
  if (path.filename() == "libaima-fmha-q16384-hybrid.so") {
    return "packed-GQA/CK-Tile hybrid";
  }
  return "CK-Tile";
}

double elapsed_ms(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - start)
      .count();
}

bool prefill_timeline_enabled() {
  const char* value = std::getenv("AIMA_NATIVE_PREFILL_TIMELINE");
  return value != nullptr && std::string(value) == "1";
}

bool contains(const std::vector<std::uint32_t>& values,
              std::uint32_t value) {
  return std::find(values.begin(), values.end(), value) != values.end();
}

void check_hip(hipError_t status, const char* operation) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(status));
  }
}

class NativeExactPrefixCache {
 public:
  ~NativeExactPrefixCache() {
    if (allocation_ != nullptr) {
      (void)hipSetDevice(device_);
      (void)hipFree(allocation_);
    }
  }
  NativeExactPrefixCache(const NativeExactPrefixCache&) = delete;
  NativeExactPrefixCache& operator=(const NativeExactPrefixCache&) = delete;
  NativeExactPrefixCache() = default;

  std::uint64_t build(const NativeDecodeWorkspace& decode_workspace,
                      NativeFullAttentionState& attention_state,
                      int device, std::size_t max_cache_tokens) {
    if (allocation_ != nullptr) {
      throw std::runtime_error("native exact-prefix cache is already built");
    }
    if (max_cache_tokens == 0 || max_cache_tokens > 262144) {
      throw std::invalid_argument(
          "native exact-prefix context is unsupported");
    }
    device_ = device;
    max_cache_tokens_ = max_cache_tokens;
    constexpr std::uint64_t kConvBytes =
        8192ULL * 3ULL * sizeof(std::uint16_t);
    constexpr std::uint64_t kRecurrentBytes =
        32ULL * 128ULL * 128ULL * sizeof(float);
    const std::uint64_t kKvBytes =
        max_cache_tokens_ * 2ULL * 256ULL * sizeof(std::uint16_t);
    constexpr std::uint64_t kKvBytesPerToken =
        2ULL * 256ULL * sizeof(std::uint16_t);
    for (std::size_t layer = 0; layer < 40; ++layer) {
      if (layer % 4 == 3) {
        add_slice(attention_state.k_cache(layer), kKvBytes,
                  kKvBytesPerToken);
        add_slice(attention_state.v_cache(layer), kKvBytes,
                  kKvBytesPerToken);
      } else {
        const std::string index = std::to_string(layer);
        const NativeDecodeWorkspaceView* conv = decode_workspace.find(
            "linear_attention_initial_conv_states." + index);
        const NativeDecodeWorkspaceView* recurrent = decode_workspace.find(
            "linear_attention_initial_ssm_states_vllm." + index);
        if (conv == nullptr || conv->device_pointer == nullptr ||
            conv->payload_bytes != kConvBytes || recurrent == nullptr ||
            recurrent->device_pointer == nullptr ||
            recurrent->payload_bytes != kRecurrentBytes) {
          throw std::runtime_error(
              "native exact-prefix linear state geometry is incomplete");
        }
        add_slice(conv->device_pointer, kConvBytes, 0);
        add_slice(recurrent->device_pointer, kRecurrentBytes, 0);
      }
    }
    terminal_offset_ = bytes_;
    bytes_ += kHidden * sizeof(std::uint16_t);
    check_hip(hipSetDevice(device_), "hipSetDevice exact-prefix cache");
    check_hip(hipMalloc(&allocation_, bytes_),
              "hipMalloc exact-prefix cache");
    return bytes_;
  }

  bool valid() const { return valid_; }

  std::size_t matched_prefix_tokens(
      const std::vector<std::uint32_t>& tokens,
      std::string_view multimodal_namespace) const {
    if (!valid_) return 0;
    return native_prefix_cache_matched_tokens(
        tokens_, multimodal_namespace_, tokens, multimodal_namespace);
  }

  std::uint64_t capture(const std::vector<std::uint32_t>& tokens,
                        std::string_view multimodal_namespace,
                        const void* terminal_hidden,
                        void* stream_value = nullptr) {
    if (allocation_ == nullptr || terminal_hidden == nullptr ||
        tokens.empty() || tokens.size() > max_cache_tokens_) {
      throw std::invalid_argument(
          "native exact-prefix capture is incomplete");
    }
    if (!valid_native_multimodal_cache_namespace(multimodal_namespace)) {
      throw std::invalid_argument(
          "native exact-prefix multimodal namespace is invalid");
    }
    hipStream_t stream = static_cast<hipStream_t>(stream_value);
    auto* destination = static_cast<unsigned char*>(allocation_);
    std::uint64_t transfer_bytes = 0;
    valid_ = false;
    for (const Slice& slice : slices_) {
      const std::uint64_t copy_bytes = slice.bytes_per_token == 0
                                           ? slice.capacity_bytes
                                           : tokens.size() *
                                                 slice.bytes_per_token;
      check_hip(hipMemcpyAsync(destination + slice.offset, slice.live,
                               copy_bytes, hipMemcpyDeviceToDevice, stream),
                "hipMemcpyAsync exact-prefix capture");
      transfer_bytes += copy_bytes;
    }
    check_hip(hipMemcpyAsync(destination + terminal_offset_, terminal_hidden,
                             kHidden * sizeof(std::uint16_t),
                             hipMemcpyDeviceToDevice, stream),
              "hipMemcpyAsync exact-prefix terminal capture");
    check_hip(hipStreamSynchronize(stream),
              "hipStreamSynchronize exact-prefix capture");
    tokens_ = tokens;
    multimodal_namespace_ = multimodal_namespace;
    valid_ = true;
    return transfer_bytes + kHidden * sizeof(std::uint16_t);
  }

  std::uint64_t restore(void* stream_value = nullptr) const {
    return restore_slices(true, stream_value);
  }

  std::uint64_t restore_linear_state(void* stream_value = nullptr) const {
    return restore_slices(false, stream_value);
  }

 private:
  std::uint64_t restore_slices(bool include_attention_kv,
                               void* stream_value) const {
    if (!valid_ || allocation_ == nullptr) {
      throw std::runtime_error("native exact-prefix cache is empty");
    }
    hipStream_t stream = static_cast<hipStream_t>(stream_value);
    const auto* source = static_cast<const unsigned char*>(allocation_);
    std::uint64_t transfer_bytes = 0;
    for (const Slice& slice : slices_) {
      if (!include_attention_kv && slice.bytes_per_token != 0) continue;
      const std::uint64_t copy_bytes = slice.bytes_per_token == 0
                                           ? slice.capacity_bytes
                                           : tokens_.size() *
                                                 slice.bytes_per_token;
      check_hip(hipMemcpyAsync(slice.live, source + slice.offset,
                               copy_bytes, hipMemcpyDeviceToDevice, stream),
                "hipMemcpyAsync exact-prefix restore");
      transfer_bytes += copy_bytes;
    }
    return transfer_bytes;
  }

 public:
  const void* terminal_hidden() const {
    if (!valid_) throw std::runtime_error("native exact-prefix cache is empty");
    return static_cast<const unsigned char*>(allocation_) + terminal_offset_;
  }

 private:
  struct Slice {
    void* live = nullptr;
    std::uint64_t capacity_bytes = 0;
    std::uint64_t bytes_per_token = 0;
    std::uint64_t offset = 0;
  };
  void add_slice(void* live, std::uint64_t capacity_bytes,
                 std::uint64_t bytes_per_token) {
    if (live == nullptr || capacity_bytes == 0 ||
        (bytes_per_token != 0 &&
         capacity_bytes != max_cache_tokens_ * bytes_per_token)) {
      throw std::invalid_argument("native exact-prefix slice is invalid");
    }
    slices_.push_back(
        {live, capacity_bytes, bytes_per_token, bytes_});
    bytes_ += capacity_bytes;
  }

  int device_ = 0;
  void* allocation_ = nullptr;
  std::uint64_t bytes_ = 0;
  std::uint64_t terminal_offset_ = 0;
  std::size_t max_cache_tokens_ = 0;
  std::vector<Slice> slices_;
  std::vector<std::uint32_t> tokens_;
  std::string multimodal_namespace_;
  bool valid_ = false;
};

}  // namespace

struct NativeResidentAuxPrefillBucket {
  explicit NativeResidentAuxPrefillBucket(std::size_t token_count)
      : tokens(token_count),
        gemm_plans(
            std::make_unique<NativeQ8192PrefillGemmPlans>(token_count)) {}

  std::size_t tokens = 0;
  NativePrefillWorkspace workspace;
  NativePrefillInvocations invocations;
  std::unique_ptr<NativeQ8192PrefillGemmPlans> gemm_plans;
  std::size_t start_sequence = 0;
};

struct NativeResidentPrefillOwner {
  NativePrefillWorkspace* workspace = nullptr;
  NativePrefillInvocations* invocations = nullptr;
  NativeQ8192PrefillGemmPlans* gemm_plans = nullptr;
  NativeQ8192CkProvider* fmha_provider = nullptr;
  std::size_t start_sequence = 0;
};

struct NativeResidentVisionPlanEntry {
  std::vector<NativeVlGrid> grids;
  std::unique_ptr<NativeVisionPipelinePlan> pipeline;
  std::uint64_t use = 0;
};

struct NativeResidentVisionExecutionPlanEntry {
  std::size_t patch_count = 0;
  std::vector<std::uint32_t> cu_seqlens;
  std::string attention_image_sha256;
  std::shared_ptr<NativeVisionPatchEmbedPlan> patch;
  std::shared_ptr<const NativeVisionAotAttentionPlan> attention;
  std::shared_ptr<NativeVisionAotBlockGemmPlans> block_gemms;
  std::shared_ptr<NativeVisionMergerPlan> merger;
  std::uint64_t use = 0;
};

struct NativeResidentVisionEmbeddingEntry {
  std::string namespace_sha256;
  void* embeddings = nullptr;
  std::uint64_t bytes = 0;
  std::uint64_t use = 0;
};

struct NativeResidentVisionWarmupMetrics {
  std::size_t patches = 0;
  std::size_t visual_tokens = 0;
  std::size_t image_count_patches = 0;
  std::size_t image_count_visual_tokens = 0;
  std::size_t plan_cache_entries = 0;
  double plan_build_wall_ms = 0.0;
  double encode_wall_ms = 0.0;
  double image_count_plan_build_wall_ms = 0.0;
  double image_count_encode_wall_ms = 0.0;
  bool completed = false;
};

bool same_vision_grids(const std::vector<NativeVlGrid>& left,
                       const std::vector<NativeVlGrid>& right) {
  if (left.size() != right.size()) return false;
  for (std::size_t index = 0; index < left.size(); ++index) {
    if (left[index].temporal != right[index].temporal ||
        left[index].height != right[index].height ||
        left[index].width != right[index].width) {
      return false;
    }
  }
  return true;
}

bool use_image_optimized_vision_attention(
    const std::vector<NativeVlGrid>& grids, bool image_only_request) {
  if (!image_only_request || grids.empty()) return false;
  std::size_t patch_count = 0;
  for (const NativeVlGrid& grid : grids) {
    if (grid.temporal != 1) return false;
    const std::size_t grid_patches = grid.patch_count();
    if (grid_patches >
        kImageOptimizedVisionAttentionMaximumPatches - patch_count) {
      return false;
    }
    patch_count += grid_patches;
  }
  return patch_count >= kImageOptimizedVisionAttentionMinimumPatches;
}

const char* vision_attention_image_sha256_for_grids(
    const std::vector<NativeVlGrid>& grids, bool image_only_request) {
  return use_image_optimized_vision_attention(grids, image_only_request)
             ? kDenseImageVisionAttentionImageSha256
             : kVisionAttentionImageSha256;
}

std::shared_ptr<const NativeVisionAotAttentionPlan>
make_vision_attention_plan(
    const std::filesystem::path& primary_image_path,
    std::string_view desired_image_sha256, std::size_t patch_count,
    const std::vector<std::uint32_t>& cu_seqlens) {
  if (desired_image_sha256 == kVisionAttentionImageSha256) {
    return std::make_shared<NativeVisionAotAttentionPlan>(
        primary_image_path, patch_count, cu_seqlens);
  }
  if (desired_image_sha256 != kDenseImageVisionAttentionImageSha256) {
    throw std::runtime_error(
        "native vision attention dispatch selected an unknown image");
  }
  const EmbeddedAotImage* image =
      find_embedded_aot_image(kDenseImageVisionAttentionKernelHash);
  if (image == nullptr || image->symbol == nullptr || image->image == nullptr ||
      image->image_bytes == 0 ||
      std::string_view(image->symbol) != kVisionAttentionKernelSymbol ||
      image->num_warps != 8 || image->warp_size != 32 ||
      image->shared_memory_bytes != 32768 ||
      sha256_bytes(image->image, image->image_bytes) !=
          kDenseImageVisionAttentionImageSha256) {
    throw std::runtime_error(
        "native dense-image vision attention registry entry is missing or "
        "changed");
  }
  auto result = NativeVisionAotAttentionPlan::from_embedded_dense_image(
      image->image, image->image_bytes, patch_count, cu_seqlens);
  if (!result ||
      result->image_sha256() != kDenseImageVisionAttentionImageSha256) {
    throw std::runtime_error(
        "native dense-image vision attention plan identity changed");
  }
  return result;
}

std::vector<std::uint32_t> vision_attention_cu_seqlens(
    const std::vector<NativeVlGrid>& grids) {
  std::vector<std::uint32_t> result{0};
  for (const NativeVlGrid& grid : grids) {
    const std::size_t frame_patches = grid.patch_count() / grid.temporal;
    for (std::size_t frame = 0; frame < grid.temporal; ++frame) {
      if (frame_patches >
          std::numeric_limits<std::uint32_t>::max() - result.back()) {
        throw std::invalid_argument(
            "native vision attention boundary overflows");
      }
      result.push_back(result.back() +
                       static_cast<std::uint32_t>(frame_patches));
    }
  }
  return result;
}

struct NativeResidentEngine::Impl {
  NativeWeightStore weights;
  NativeWeightStore visual_weights;
  NativeDerivedWeightStore derived;
  NativeLmHeadStore lm_head;
  NativeDecodeBindings bindings;
  NativePrefillWorkspace prefill_workspace;
  NativePrefillInvocations prefill_invocations;
  NativePrefillWorkspace frozen_text_q1024_workspace;
  NativePrefillInvocations frozen_text_q1024_invocations;
  std::unique_ptr<NativeQ8192PrefillGemmPlans> prefill_gemm_plans;
  NativePrefillWorkspace tail_prefill_workspace;
  NativePrefillInvocations tail_prefill_invocations;
  std::unique_ptr<NativeQ8192PrefillGemmPlans> tail_prefill_gemm_plans;
  std::unique_ptr<Bf16GemmPlan> decode_shared_gate_plan;
  NativeDecodeWorkspace decode_workspace;
  NativeDecodeInvocations decode_invocations;
  NativeDecodeCrossLayerNormBindings decode_cross_layer_norms;
  NativeDecodeExecutor executor;
  std::unique_ptr<NativeVlUnifiedAttentionPlan> vl_unified_attention;
  NativeVlLogicalProjectionState vl_logical_projections;
  NativeQ8192CkProvider ck_provider;
  NativeQ8192CkProvider secondary_fmha_provider;
  NativeQ8192CkProvider auxiliary_short_fmha_provider;
  NativeQ8192CkProvider auxiliary_q8192_fmha_provider;
  std::array<bool, 40> secondary_fmha_layers{};
  NativeFullAttentionState attention_state;
  std::array<NativeExactPrefixCache, kPrefixCacheEntries> prefix_caches;
  std::array<std::uint64_t, kPrefixCacheEntries> prefix_cache_use{};
  std::uint64_t prefix_cache_clock = 0;
  std::size_t prefix_cache_entries = 0;
  std::size_t active_kv_prefix_cache_index = kPrefixCacheEntries;
  NativeResidentLoadMetrics metrics;
  int device = 0;
  int cu_count = 0;
  std::size_t prompt_tokens = 0;
  std::size_t prefill_tokens = 0;
  std::size_t tail_prefill_tokens = 0;
  void* chunked_hidden = nullptr;
  std::uint64_t chunked_hidden_bytes = 0;
  void* padded_prefill_initial_conv_state = nullptr;
  std::uint64_t padded_prefill_initial_conv_state_bytes = 0;
  void* mrope_positions = nullptr;
  std::uint64_t mrope_position_state_bytes = 0;
  std::size_t mrope_position_row_stride = 0;
  void* vl_prompt_index_state = nullptr;
  std::uint64_t vl_prompt_index_state_bytes = 0;
  void* vl_prompt_token_ids = nullptr;
  void* vl_scatter_indices = nullptr;
  void* structured_token_mask = nullptr;
  std::uint64_t structured_token_mask_bytes = 0;
  std::vector<std::uint8_t> host_structured_token_mask;
  void* vision_pixel_values = nullptr;
  std::uint64_t vision_pixel_capacity_bytes = 0;
  void* vision_embeddings = nullptr;
  std::uint64_t vision_embedding_capacity_bytes = 0;
  void* vision_preparation_embeddings = nullptr;
  std::uint64_t vision_preparation_embedding_capacity_bytes = 0;
  void* vision_temporary = nullptr;
  std::uint64_t vision_temporary_capacity_bytes = 0;
  std::vector<NativeResidentVisionPlanEntry> vision_plans;
  std::vector<NativeResidentVisionExecutionPlanEntry>
      vision_warmed_execution_plans;
  std::uint64_t vision_execution_plan_clock = 0;
  std::uint64_t vision_plan_clock = 0;
  std::vector<NativeResidentVisionEmbeddingEntry> vision_embedding_cache;
  std::uint64_t vision_embedding_cache_resident_bytes = 0;
  std::uint64_t vision_embedding_cache_clock = 0;
  std::filesystem::path vision_attention_image;
  std::size_t prefill_start_sequence = 0;
  std::size_t frozen_text_q1024_start_sequence = 0;
  std::size_t tail_prefill_start_sequence = 0;
  std::vector<std::size_t> resident_prefill_buckets;
  std::vector<std::unique_ptr<NativeResidentAuxPrefillBucket>>
      auxiliary_prefill_buckets;
  std::size_t request_count = 0;
  std::size_t prefix_cache_hits = 0;
  std::size_t prefix_cache_misses = 0;
  bool ready = false;

  NativeResidentPrefillOwner prefill_owner(
      std::size_t tokens, bool use_frozen_text = false) {
    NativeResidentPrefillOwner owner;
    if (tokens == prefill_tokens) {
      owner = {&prefill_workspace, &prefill_invocations,
               prefill_gemm_plans.get(), &ck_provider,
               prefill_start_sequence};
    } else if (tokens == tail_prefill_tokens && tail_prefill_tokens != 0) {
      owner = {&tail_prefill_workspace, &tail_prefill_invocations,
               tail_prefill_gemm_plans.get(), &ck_provider,
               tail_prefill_start_sequence};
    } else {
      for (const auto& bucket : auxiliary_prefill_buckets) {
        if (bucket->tokens != tokens) continue;
        NativeQ8192CkProvider* provider = &ck_provider;
        if (tokens <= 4096) {
          provider = &auxiliary_short_fmha_provider;
        } else if (tokens == 8192) {
          provider = &auxiliary_q8192_fmha_provider;
        }
        owner = {&bucket->workspace, &bucket->invocations,
                 bucket->gemm_plans.get(), provider,
                 bucket->start_sequence};
        break;
      }
    }
    if (owner.workspace == nullptr || owner.invocations == nullptr) {
      throw std::runtime_error(
          "native resident prefill bucket owner is unavailable");
    }
    if (use_frozen_text && tokens == 1024) {
      owner.workspace = &frozen_text_q1024_workspace;
      owner.invocations = &frozen_text_q1024_invocations;
      owner.start_sequence = frozen_text_q1024_start_sequence;
    }
    return owner;
  }

  void ensure_vision_allocation(void** pointer, std::uint64_t* capacity,
                                std::uint64_t required,
                                const char* operation) {
    if (required == 0 || pointer == nullptr || capacity == nullptr) {
      throw std::invalid_argument(
          "native resident vision allocation request is invalid");
    }
    if (*pointer != nullptr && *capacity >= required) return;
    void* replacement = nullptr;
    check_hip(hipMalloc(&replacement, required), operation);
    if (*pointer != nullptr) {
      const hipError_t released = hipFree(*pointer);
      if (released != hipSuccess) {
        (void)hipFree(replacement);
        check_hip(released, "hipFree replaced native vision allocation");
      }
    }
    *pointer = replacement;
    *capacity = required;
  }

  double prepare_vision_pipeline_once(
      NativeVisionPipelinePlan& pipeline, const char* description) {
    const std::uint64_t pixel_bytes =
        pipeline.patch_count() * kVisionPixelColumns *
        sizeof(std::uint16_t);
    const std::uint64_t embedding_bytes =
        pipeline.merged_token_count() * kHidden * sizeof(std::uint16_t);
    ensure_vision_allocation(
        &vision_pixel_values, &vision_pixel_capacity_bytes, pixel_bytes,
        description);
    ensure_vision_allocation(
        &vision_preparation_embeddings,
        &vision_preparation_embedding_capacity_bytes,
        embedding_bytes, description);
    ensure_vision_allocation(
        &vision_temporary, &vision_temporary_capacity_bytes,
        pipeline.temporary_bytes(), description);
    check_hip(hipMemset(vision_pixel_values, 0, pixel_bytes), description);
    const auto started = std::chrono::steady_clock::now();
    pipeline.launch(vision_pixel_values, vision_preparation_embeddings,
                    vision_temporary, vision_temporary_capacity_bytes);
    check_hip(hipDeviceSynchronize(), description);
    return elapsed_ms(started);
  }

  bool restore_vision_embedding_cache(std::string_view namespace_sha256,
                                      std::uint64_t bytes) {
    if (namespace_sha256.empty() || bytes == 0 ||
        vision_embeddings == nullptr ||
        vision_embedding_capacity_bytes < bytes) {
      return false;
    }
    for (NativeResidentVisionEmbeddingEntry& entry :
         vision_embedding_cache) {
      if (entry.namespace_sha256 != namespace_sha256 ||
          entry.bytes != bytes || entry.embeddings == nullptr) {
        continue;
      }
      check_hip(hipMemcpyAsync(vision_embeddings, entry.embeddings, bytes,
                               hipMemcpyDeviceToDevice, nullptr),
                "hipMemcpyAsync cached vision embeddings");
      entry.use = ++vision_embedding_cache_clock;
      return true;
    }
    return false;
  }

  void insert_vision_embedding_cache(std::string_view namespace_sha256,
                                     std::uint64_t bytes) {
    if (namespace_sha256.empty() || bytes == 0 ||
        bytes > kVisionEmbeddingCacheBytes || vision_embeddings == nullptr) {
      return;
    }
    for (NativeResidentVisionEmbeddingEntry& entry :
         vision_embedding_cache) {
      if (entry.namespace_sha256 == namespace_sha256 &&
          entry.bytes == bytes && entry.embeddings != nullptr) {
        entry.use = ++vision_embedding_cache_clock;
        return;
      }
    }
    while (!vision_embedding_cache.empty() &&
           (vision_embedding_cache.size() >= kVisionEmbeddingCacheEntries ||
            vision_embedding_cache_resident_bytes >
                kVisionEmbeddingCacheBytes - bytes)) {
      const auto oldest = std::min_element(
          vision_embedding_cache.begin(), vision_embedding_cache.end(),
          [](const NativeResidentVisionEmbeddingEntry& left,
             const NativeResidentVisionEmbeddingEntry& right) {
            return left.use < right.use;
          });
      check_hip(hipFree(oldest->embeddings),
                "hipFree evicted vision embedding cache");
      vision_embedding_cache_resident_bytes -= oldest->bytes;
      vision_embedding_cache.erase(oldest);
    }
    void* cached = nullptr;
    if (hipMalloc(&cached, bytes) != hipSuccess || cached == nullptr) {
      return;
    }
    const hipError_t copied = hipMemcpyAsync(
        cached, vision_embeddings, bytes, hipMemcpyDeviceToDevice, nullptr);
    if (copied != hipSuccess) {
      (void)hipFree(cached);
      check_hip(copied, "hipMemcpyAsync insert vision embedding cache");
    }
    vision_embedding_cache.push_back(NativeResidentVisionEmbeddingEntry{
        std::string(namespace_sha256), cached, bytes,
        ++vision_embedding_cache_clock});
    vision_embedding_cache_resident_bytes += bytes;
  }

  void retain_warmed_vision_execution_plans(
      const NativeVisionPipelinePlan& pipeline) {
    const std::size_t patch_count = pipeline.patch_count();
    const std::string attention_image_sha256 =
        pipeline.attention_image_sha256();
    const auto existing = std::find_if(
        vision_warmed_execution_plans.begin(),
        vision_warmed_execution_plans.end(),
        [patch_count, &attention_image_sha256](
            const NativeResidentVisionExecutionPlanEntry& entry) {
          return entry.patch_count == patch_count &&
                 entry.attention_image_sha256 == attention_image_sha256;
        });
    if (existing != vision_warmed_execution_plans.end()) {
      existing->cu_seqlens = pipeline.cu_seqlens();
      existing->attention = pipeline.attention_plan();
      existing->use = ++vision_execution_plan_clock;
      return;
    }
    if (vision_warmed_execution_plans.size() >=
        kVisionExecutionPlanCacheEntries) {
      const auto oldest = std::min_element(
          vision_warmed_execution_plans.begin(),
          vision_warmed_execution_plans.end(),
          [](const NativeResidentVisionExecutionPlanEntry& left,
             const NativeResidentVisionExecutionPlanEntry& right) {
            return left.use < right.use;
          });
      vision_warmed_execution_plans.erase(oldest);
    }
    NativeResidentVisionExecutionPlanEntry entry;
    entry.patch_count = patch_count;
    entry.cu_seqlens = pipeline.cu_seqlens();
    entry.attention_image_sha256 = attention_image_sha256;
    entry.patch = pipeline.patch_plan();
    entry.attention = pipeline.attention_plan();
    entry.block_gemms = pipeline.block_gemm_plans();
    entry.merger = pipeline.merger_plan();
    if (entry.cu_seqlens.empty() || entry.attention_image_sha256.empty() ||
        !entry.patch || !entry.attention ||
        !entry.block_gemms || !entry.merger) {
      throw std::runtime_error(
          "native warmed vision execution plans are incomplete");
    }
    entry.use = ++vision_execution_plan_clock;
    vision_warmed_execution_plans.push_back(std::move(entry));
  }

  NativeVisionPipelinePlan& vision_plan(
      const std::vector<NativeVlGrid>& grids, bool image_only_request,
      bool* cache_hit, double* build_wall_ms,
      bool prepare_on_miss = true) {
    if (cache_hit == nullptr || build_wall_ms == nullptr || grids.empty()) {
      throw std::invalid_argument(
          "native resident vision plan lookup is invalid");
    }
    const std::string desired_attention_image_sha256 =
        vision_attention_image_sha256_for_grids(grids, image_only_request);
    for (NativeResidentVisionPlanEntry& entry : vision_plans) {
      if (!same_vision_grids(entry.grids, grids) ||
          entry.pipeline->attention_image_sha256() !=
              desired_attention_image_sha256) {
        continue;
      }
      entry.use = ++vision_plan_clock;
      *cache_hit = true;
      *build_wall_ms = 0.0;
      return *entry.pipeline;
    }
    std::size_t incoming_patches = 0;
    for (const NativeVlGrid& grid : grids) {
      const std::size_t grid_patches = grid.patch_count();
      if (grid_patches >
          kVisionPlanCachePatchBudget - incoming_patches) {
        throw std::invalid_argument(
            "native resident vision plan exceeds the cache patch budget");
      }
      incoming_patches += grid_patches;
    }
    std::size_t cached_patches = 0;
    for (const NativeResidentVisionPlanEntry& entry : vision_plans) {
      cached_patches += entry.pipeline->patch_count();
    }
    const bool exclusive_cache_required =
        incoming_patches >= kVisionPlanCacheSharedPatchLimit ||
        std::any_of(
            vision_plans.begin(), vision_plans.end(),
            [](const NativeResidentVisionPlanEntry& entry) {
              return entry.pipeline->patch_count() >=
                     kVisionPlanCacheSharedPatchLimit;
            });
    if (exclusive_cache_required) {
      vision_plans.clear();
      cached_patches = 0;
    }
    while (!vision_plans.empty() &&
           (vision_plans.size() >= kVisionPlanCacheEntries ||
            cached_patches >
                kVisionPlanCachePatchBudget - incoming_patches)) {
      const auto oldest = std::min_element(
          vision_plans.begin(), vision_plans.end(),
          [](const NativeResidentVisionPlanEntry& left,
             const NativeResidentVisionPlanEntry& right) {
            return left.use < right.use;
          });
      cached_patches -= oldest->pipeline->patch_count();
      vision_plans.erase(oldest);
    }
    const auto started = std::chrono::steady_clock::now();
    const std::vector<std::uint32_t> cu_seqlens =
        vision_attention_cu_seqlens(grids);
    const auto warmed_resources = std::find_if(
        vision_warmed_execution_plans.begin(),
        vision_warmed_execution_plans.end(),
        [incoming_patches](
            const NativeResidentVisionExecutionPlanEntry& entry) {
          return entry.patch_count == incoming_patches;
        });
    const auto exact_attention = std::find_if(
        vision_warmed_execution_plans.begin(),
        vision_warmed_execution_plans.end(),
        [incoming_patches, &cu_seqlens,
         &desired_attention_image_sha256](
            const NativeResidentVisionExecutionPlanEntry& entry) {
          return entry.patch_count == incoming_patches &&
                 entry.cu_seqlens == cu_seqlens &&
                 entry.attention_image_sha256 ==
                     desired_attention_image_sha256;
        });
    const auto warmed_attention = std::find_if(
        vision_warmed_execution_plans.begin(),
        vision_warmed_execution_plans.end(),
        [&desired_attention_image_sha256](
            const NativeResidentVisionExecutionPlanEntry& entry) {
          return entry.attention_image_sha256 ==
                 desired_attention_image_sha256;
        });
    if (warmed_resources != vision_warmed_execution_plans.end()) {
      warmed_resources->use = ++vision_execution_plan_clock;
    }
    if (warmed_attention != vision_warmed_execution_plans.end()) {
      warmed_attention->use = ++vision_execution_plan_clock;
    }
    if (exact_attention != vision_warmed_execution_plans.end()) {
      exact_attention->use = ++vision_execution_plan_clock;
    }
    std::unique_ptr<NativeVisionPipelinePlan> pipeline;
    bool attention_reused = false;
    std::shared_ptr<const NativeVisionAotAttentionPlan> attention;
    if (exact_attention != vision_warmed_execution_plans.end()) {
      attention = exact_attention->attention;
      attention_reused = true;
    } else if (warmed_attention != vision_warmed_execution_plans.end()) {
      attention = warmed_attention->attention->rebind(
          incoming_patches, cu_seqlens);
    } else {
      attention = make_vision_attention_plan(
          vision_attention_image, desired_attention_image_sha256,
          incoming_patches, cu_seqlens);
    }
    if (warmed_resources == vision_warmed_execution_plans.end()) {
      pipeline = std::make_unique<NativeVisionPipelinePlan>(
          visual_weights, vision_attention_image, grids, nullptr,
          std::move(attention), nullptr, nullptr);
    } else {
      pipeline = std::make_unique<NativeVisionPipelinePlan>(
          visual_weights, vision_attention_image, grids,
          warmed_resources->patch, std::move(attention),
          warmed_resources->block_gemms, warmed_resources->merger);
    }
    if (prepare_on_miss &&
        incoming_patches > kVisionRequestPlanPreparationMinPatches &&
        incoming_patches <= kVisionRequestPlanPreparationPatchLimit &&
        (warmed_resources == vision_warmed_execution_plans.end() ||
         !attention_reused)) {
      prepare_vision_pipeline_once(
          *pipeline, "native request vision plan preparation");
      retain_warmed_vision_execution_plans(*pipeline);
    }
    *build_wall_ms = elapsed_ms(started);
    *cache_hit = false;
    NativeResidentVisionPlanEntry candidate{
        grids, std::move(pipeline), ++vision_plan_clock};
    vision_plans.push_back(std::move(candidate));
    return *vision_plans.back().pipeline;
  }

  NativeResidentVisionWarmupMetrics warm_up_standard_vision() {
    // vLLM profiles the visual tower before publishing READY. Match that
    // boundary with the frozen standard portrait geometry. Retain one exact
    // maximum-image-count execution shape as well: its shared hipBLASLt plans
    // otherwise pay their first-launch cost immediately after the sustained
    // maximum-image workload. Preparing it here avoids both that measured
    // cold launch and a duplicate encoder pass inside the user request.
    const std::vector<NativeVlGrid> grids{{1, 64, 16}};
    bool cache_hit = false;
    double plan_build_wall_ms = 0.0;
    NativeVisionPipelinePlan& pipeline =
        vision_plan(grids, false, &cache_hit, &plan_build_wall_ms, false);
    if (cache_hit || pipeline.patch_count() != 1024 ||
        pipeline.merged_token_count() != 256) {
      throw std::runtime_error(
          "native standard vision warmup geometry is inconsistent");
    }
    const double encode_wall_ms = prepare_vision_pipeline_once(
        pipeline, "native standard vision warmup");
    retain_warmed_vision_execution_plans(pipeline);

    const std::vector<NativeVlGrid> image_count_grids(
        16, NativeVlGrid{1, 16, 16});
    bool image_count_cache_hit = false;
    double image_count_plan_build_wall_ms = 0.0;
    NativeVisionPipelinePlan& image_count_pipeline = vision_plan(
        image_count_grids, true, &image_count_cache_hit,
        &image_count_plan_build_wall_ms, false);
    if (image_count_cache_hit || image_count_pipeline.patch_count() != 4096 ||
        image_count_pipeline.merged_token_count() != 1024) {
      throw std::runtime_error(
          "native image-count vision warmup geometry is inconsistent");
    }
    const double image_count_encode_wall_ms = prepare_vision_pipeline_once(
        image_count_pipeline, "native image-count vision warmup");
    retain_warmed_vision_execution_plans(image_count_pipeline);

    NativeResidentVisionWarmupMetrics metrics;
    metrics.patches = pipeline.patch_count();
    metrics.visual_tokens = pipeline.merged_token_count();
    metrics.image_count_patches = image_count_pipeline.patch_count();
    metrics.image_count_visual_tokens =
        image_count_pipeline.merged_token_count();
    metrics.plan_cache_entries = vision_plans.size();
    metrics.plan_build_wall_ms = plan_build_wall_ms;
    metrics.encode_wall_ms = encode_wall_ms;
    metrics.image_count_plan_build_wall_ms = image_count_plan_build_wall_ms;
    metrics.image_count_encode_wall_ms = image_count_encode_wall_ms;
    metrics.completed = true;
    return metrics;
  }

  ~Impl() {
    // Logical-M plans borrow immutable hipBLASLt handles and operations from
    // their admitted prefill bucket. Release the derived cache while every
    // bucket owner is still alive; member destruction follows this body.
    vl_logical_projections.reset();
    for (NativeResidentVisionEmbeddingEntry& entry :
         vision_embedding_cache) {
      if (entry.embeddings != nullptr) {
        (void)hipSetDevice(device);
        (void)hipFree(entry.embeddings);
      }
    }
    if (chunked_hidden != nullptr) {
      (void)hipSetDevice(device);
      (void)hipFree(chunked_hidden);
    }
    if (padded_prefill_initial_conv_state != nullptr) {
      (void)hipSetDevice(device);
      (void)hipFree(padded_prefill_initial_conv_state);
    }
    if (mrope_positions != nullptr) {
      (void)hipSetDevice(device);
      (void)hipFree(mrope_positions);
    }
    if (vl_prompt_index_state != nullptr) {
      (void)hipSetDevice(device);
      (void)hipFree(vl_prompt_index_state);
    }
    if (structured_token_mask != nullptr) {
      (void)hipSetDevice(device);
      (void)hipFree(structured_token_mask);
    }
    if (vision_pixel_values != nullptr) {
      (void)hipSetDevice(device);
      (void)hipFree(vision_pixel_values);
    }
    if (vision_embeddings != nullptr) {
      (void)hipSetDevice(device);
      (void)hipFree(vision_embeddings);
    }
    if (vision_preparation_embeddings != nullptr) {
      (void)hipSetDevice(device);
      (void)hipFree(vision_preparation_embeddings);
    }
    if (vision_temporary != nullptr) {
      (void)hipSetDevice(device);
      (void)hipFree(vision_temporary);
    }
  }
};

NativeResidentEngine::NativeResidentEngine()
    : impl_(std::make_unique<Impl>()) {}
NativeResidentEngine::~NativeResidentEngine() = default;

NativeResidentLoadMetrics NativeResidentEngine::load(
    const NativeResidentEngineOptions& options) {
  if (impl_->ready || options.weights.model_dir.empty() ||
      options.prompt_tokens == 0 || options.prompt_tokens > 262144 ||
      options.cache_capacity < options.prompt_tokens + 1 ||
      options.cache_capacity > 262144) {
    throw std::invalid_argument(
        "native resident engine load options are incomplete");
  }
  if (options.secondary_fmha_provider.empty() !=
      options.secondary_fmha_layers.empty()) {
    throw std::invalid_argument(
        "native secondary FMHA provider and layer set must be specified together");
  }
  if (options.prompt_tokens > 32768 &&
      !admitted_long_context(options.prompt_tokens)) {
    throw std::invalid_argument(
        "native long-context prefill has no admitted tail specialization");
  }
  const bool automatic_provider_policy = options.ck_provider.empty();
  std::filesystem::path provider_path =
      automatic_provider_policy
          ? default_fmha_provider(options.prompt_tokens)
          : std::filesystem::absolute(options.ck_provider);
  std::filesystem::path secondary_provider_path =
      options.secondary_fmha_provider.empty()
          ? std::filesystem::path{}
          : std::filesystem::absolute(options.secondary_fmha_provider);
  std::vector<std::size_t> secondary_layers =
      options.secondary_fmha_layers;
  if (automatic_provider_policy && secondary_provider_path.empty()) {
    if (options.prompt_tokens == 16384) {
      secondary_provider_path = native_library_path("libaima-fmha-ck.so");
      secondary_layers = {39};
    } else if (options.prompt_tokens > 32768) {
      secondary_provider_path =
          native_library_path("libaima-fmha-aotriton.so");
      secondary_layers = {39};
    }
  }
  for (std::size_t layer : secondary_layers) {
    if (layer >= 40 || layer % 4 != 3 ||
        impl_->secondary_fmha_layers[layer]) {
      throw std::invalid_argument(
          "native secondary FMHA layers must be unique full-attention layers");
    }
    impl_->secondary_fmha_layers[layer] = true;
  }
  const auto started = std::chrono::steady_clock::now();
  impl_->vision_attention_image =
      options.vision_attention_image.empty()
          ? native_library_path(kVisionAttentionImageFilename)
          : std::filesystem::absolute(options.vision_attention_image);
  if (sha256_file(impl_->vision_attention_image) !=
      kVisionAttentionImageSha256) {
    throw std::runtime_error(
        "native resident vision attention image is missing or changed");
  }
  impl_->device = options.weights.device;
  impl_->prompt_tokens = options.prompt_tokens;
  impl_->prefill_tokens = impl_->prompt_tokens > 32768
                              ? kLongContextChunkTokens
                              : impl_->prompt_tokens;
  impl_->tail_prefill_tokens = impl_->prompt_tokens > 32768
                                   ? impl_->prompt_tokens %
                                         kLongContextChunkTokens
                                   : 0;
  for (const std::size_t tokens : {1024ULL, 2048ULL, 4096ULL, 8192ULL}) {
    if (tokens < impl_->prompt_tokens && tokens <= impl_->prefill_tokens) {
      impl_->resident_prefill_buckets.push_back(tokens);
    }
  }
  impl_->resident_prefill_buckets.push_back(impl_->prefill_tokens);
  if (impl_->tail_prefill_tokens != 0) {
    impl_->resident_prefill_buckets.push_back(impl_->tail_prefill_tokens);
  }
  std::sort(impl_->resident_prefill_buckets.begin(),
            impl_->resident_prefill_buckets.end());
  impl_->resident_prefill_buckets.erase(
      std::unique(impl_->resident_prefill_buckets.begin(),
                  impl_->resident_prefill_buckets.end()),
      impl_->resident_prefill_buckets.end());
  impl_->chunked_hidden_bytes =
      options.cache_capacity * kHidden * sizeof(std::uint16_t);
  impl_->padded_prefill_initial_conv_state_bytes =
      8192ULL * 3ULL * sizeof(std::uint16_t);
  impl_->mrope_position_row_stride = options.cache_capacity;
  impl_->mrope_position_state_bytes =
      3ULL * options.cache_capacity * sizeof(std::int64_t);
  const std::uint64_t vl_prompt_token_id_bytes =
      options.cache_capacity * sizeof(std::uint32_t);
  const std::uint64_t vl_scatter_index_bytes =
      2ULL * kNativeVlAggregateTokenLimit * sizeof(std::uint32_t);
  impl_->vl_prompt_index_state_bytes =
      vl_prompt_token_id_bytes + vl_scatter_index_bytes;
  impl_->structured_token_mask_bytes =
      kVocabulary * sizeof(std::uint8_t);
  check_hip(hipSetDevice(impl_->device),
            "hipSetDevice composed prefill state");
  check_hip(hipMalloc(&impl_->chunked_hidden,
                      impl_->chunked_hidden_bytes),
            "hipMalloc composed prefill hidden store");
  check_hip(hipMalloc(&impl_->padded_prefill_initial_conv_state,
                      impl_->padded_prefill_initial_conv_state_bytes),
            "hipMalloc padded prefill convolution snapshot");
  impl_->prefill_gemm_plans =
      std::make_unique<NativeQ8192PrefillGemmPlans>(impl_->prefill_tokens);
  if (impl_->tail_prefill_tokens != 0) {
    impl_->tail_prefill_gemm_plans =
        std::make_unique<NativeQ8192PrefillGemmPlans>(
            impl_->tail_prefill_tokens);
  }
  const std::filesystem::path combined_weight_report =
      options.weights.native_report.empty()
          ? std::filesystem::absolute("native-resident-weight-load.json")
          : options.weights.native_report;
  const std::filesystem::path language_weight_report =
      split_weight_report_path(combined_weight_report, "language");
  const std::filesystem::path visual_weight_report =
      split_weight_report_path(combined_weight_report, "visual");
  NativeWeightLoadOptions language_weight_options = options.weights;
  language_weight_options.native_report = language_weight_report;
  NativeWeightLoadMetrics weight_metrics =
      impl_->weights.load(language_weight_options);
  NativeVlLogicalProjectionLoadMetrics vl_logical_load_metrics;
  NativeResidentVisionWarmupMetrics vision_warmup_metrics;
  const NativeDerivedWeightMetrics derived_metrics =
      impl_->derived.build(impl_->weights, impl_->device);
  const NativeLmHeadMetrics lm_head_metrics =
      impl_->lm_head.build(impl_->weights, impl_->device);
  const NativeDecodeBindingMetrics binding_metrics =
      impl_->bindings.build(impl_->weights, impl_->derived, impl_->lm_head);
  NativePrefillWorkspaceMetrics prefill_workspace_metrics;
  NativePrefillInvocationMetrics prefill_invocation_metrics;
  NativePrefillWorkspaceMetrics tail_prefill_workspace_metrics;
  NativePrefillInvocationMetrics tail_prefill_invocation_metrics;
  NativePrefillWorkspaceMetrics auxiliary_prefill_workspace_metrics;
  NativePrefillInvocationMetrics auxiliary_prefill_invocation_metrics;
  NativePrefillWorkspaceMetrics frozen_text_q1024_workspace_metrics;
  NativePrefillInvocationMetrics frozen_text_q1024_invocation_metrics;
  const auto build_frozen_text_q1024 = [&]() {
    if (impl_->frozen_text_q1024_workspace.built()) {
      throw std::runtime_error(
          "native frozen text q1024 workspace was built more than once");
    }
    frozen_text_q1024_workspace_metrics =
        impl_->frozen_text_q1024_workspace.build(
            impl_->device, 1024, true);
    if (frozen_text_q1024_workspace_metrics.allocation_bytes !=
            kFrozenTextQ1024WorkspaceBytes ||
        frozen_text_q1024_workspace_metrics.physical_allocation_bytes !=
            kFrozenTextQ1024WorkspaceBytes ||
        !impl_->frozen_text_q1024_workspace.owns_primary_allocation()) {
      throw std::runtime_error(
          "native frozen q1024 primary backing contract changed");
    }
    frozen_text_q1024_invocation_metrics =
        impl_->frozen_text_q1024_invocations.build(
            impl_->bindings, impl_->frozen_text_q1024_workspace, 1024,
            NativePrefillScheduleKind::kFrozenText);
  };
  for (const std::size_t tokens : impl_->resident_prefill_buckets) {
    if (tokens == impl_->prefill_tokens ||
        (impl_->tail_prefill_tokens != 0 &&
         tokens == impl_->tail_prefill_tokens)) {
      continue;
    }
    auto bucket =
        std::make_unique<NativeResidentAuxPrefillBucket>(tokens);
    if (tokens == 1024) {
      // Occupy the q1024 slot with the exact v1.5.1-sized owner.  The current
      // VL view and its small divergent tail are materialized only after all
      // text-critical allocations and GEMM plans are resident.
      build_frozen_text_q1024();
    } else {
      const NativePrefillWorkspaceMetrics workspace_metrics =
          bucket->workspace.build(impl_->device, tokens, false);
      const NativePrefillInvocationMetrics invocation_metrics =
          bucket->invocations.build(impl_->bindings, bucket->workspace, tokens);
      auxiliary_prefill_workspace_metrics.allocation_bytes +=
          workspace_metrics.allocation_bytes;
      auxiliary_prefill_workspace_metrics.physical_allocation_bytes +=
          workspace_metrics.physical_allocation_bytes;
      auxiliary_prefill_invocation_metrics.launch_count +=
          invocation_metrics.launch_count;
    }
    impl_->auxiliary_prefill_buckets.push_back(std::move(bucket));
  }
  if (impl_->prefill_tokens == 1024) {
    build_frozen_text_q1024();
  } else {
    prefill_workspace_metrics = impl_->prefill_workspace.build(
        impl_->device, impl_->prefill_tokens, false);
    prefill_invocation_metrics = impl_->prefill_invocations.build(
        impl_->bindings, impl_->prefill_workspace, impl_->prefill_tokens);
  }
  if (impl_->tail_prefill_tokens != 0) {
    if (impl_->tail_prefill_tokens == 1024) {
      build_frozen_text_q1024();
    } else {
      tail_prefill_workspace_metrics = impl_->tail_prefill_workspace.build(
          impl_->device, impl_->tail_prefill_tokens, false);
      tail_prefill_invocation_metrics = impl_->tail_prefill_invocations.build(
          impl_->bindings, impl_->tail_prefill_workspace,
          impl_->tail_prefill_tokens);
    }
  }
  if (!impl_->frozen_text_q1024_workspace.built()) {
    throw std::runtime_error(
        "native resident topology has no frozen text q1024 owner");
  }
  const NativeDecodeWorkspaceMetrics decode_workspace_metrics =
      impl_->decode_workspace.build(impl_->device);
  const NativeDecodeInvocationMetrics decode_invocation_metrics =
      impl_->decode_invocations.build(impl_->bindings,
                                      impl_->decode_workspace);
  impl_->decode_cross_layer_norms = bind_native_decode_cross_layer_norms(
      impl_->weights, impl_->decode_invocations);
  const NativeDecodeExecutorMetrics executor_metrics = impl_->executor.load();
  const NativeQ8192CkProviderMetrics provider_metrics =
      impl_->ck_provider.load(provider_path, impl_->prefill_tokens);
  NativeQ8192CkProviderMetrics secondary_provider_metrics;
  if (!secondary_provider_path.empty()) {
    if (secondary_provider_path == provider_path) {
      throw std::invalid_argument(
          "native primary and secondary FMHA providers must differ");
    }
    secondary_provider_metrics = impl_->secondary_fmha_provider.load(
        secondary_provider_path, impl_->prefill_tokens);
  }
  const bool has_auxiliary_short = std::any_of(
      impl_->auxiliary_prefill_buckets.begin(),
      impl_->auxiliary_prefill_buckets.end(), [](const auto& bucket) {
        return bucket->tokens <= 4096;
      });
  if (has_auxiliary_short) {
    (void)impl_->auxiliary_short_fmha_provider.load(
        default_fmha_provider(4096), 4096);
  }
  const bool has_auxiliary_q8192 = std::any_of(
      impl_->auxiliary_prefill_buckets.begin(),
      impl_->auxiliary_prefill_buckets.end(), [](const auto& bucket) {
        return bucket->tokens == 8192;
      });
  if (has_auxiliary_q8192) {
    (void)impl_->auxiliary_q8192_fmha_provider.load(
        default_fmha_provider(8192), 8192);
  }
  const NativeFullAttentionStateMetrics attention_metrics =
      impl_->attention_state.build(options.cache_capacity, impl_->device);
  impl_->prefix_cache_entries =
      options.prefix_cache_enabled
          ? (options.cache_capacity <= 32768
                 ? 4
                 : (options.cache_capacity <= 131072 ? 2 : 1))
          : 0;
  std::uint64_t prefix_cache_bytes = 0;
  for (std::size_t index = 0; index < impl_->prefix_cache_entries; ++index) {
    prefix_cache_bytes += impl_->prefix_caches[index].build(
        impl_->decode_workspace, impl_->attention_state, impl_->device,
        options.cache_capacity);
  }

  const auto plan_started = std::chrono::steady_clock::now();
  NativeQ8192PrefillGemmPlans* prefill_gemm_plans =
      impl_->prefill_gemm_plans.get();
  prefill_gemm_plans->prepare_all();
  if (prefill_gemm_plans->token_count() == 1024) {
    prefill_gemm_plans->warm_up_q1024_text();
  }
  NativeQ8192PrefillGemmPlans* tail_prefill_gemm_plans =
      impl_->tail_prefill_gemm_plans.get();
  if (tail_prefill_gemm_plans != nullptr) {
    tail_prefill_gemm_plans->prepare_all();
    if (tail_prefill_gemm_plans->token_count() == 1024) {
      tail_prefill_gemm_plans->warm_up_q1024_text();
    }
  }
  for (const auto& bucket : impl_->auxiliary_prefill_buckets) {
    NativeQ8192PrefillGemmPlans* auxiliary_prefill_gemm_plans =
        bucket->gemm_plans.get();
    auxiliary_prefill_gemm_plans->prepare_all();
    if (auxiliary_prefill_gemm_plans->token_count() == 1024) {
      auxiliary_prefill_gemm_plans->warm_up_q1024_text();
    }
  }
  const double plan_wall_ms = elapsed_ms(plan_started);

  NativeResidentPrefillOwner current_q1024_owner =
      impl_->prefill_owner(1024);
  if (current_q1024_owner.workspace->built()) {
    throw std::runtime_error(
        "native current q1024 workspace was allocated before text topology");
  }
  const NativePrefillWorkspaceMetrics current_q1024_workspace_metrics =
      current_q1024_owner.workspace->build(
          impl_->device, 1024, false,
          impl_->frozen_text_q1024_workspace.allocation(),
          impl_->frozen_text_q1024_workspace.allocation_bytes(),
          kCurrentQ1024SplitOffset);
  if (current_q1024_workspace_metrics.allocation_bytes !=
          kCurrentQ1024WorkspaceBytes ||
      current_q1024_workspace_metrics.physical_allocation_bytes !=
          kCurrentQ1024TailBytes ||
      !current_q1024_owner.workspace->has_split_allocation() ||
      current_q1024_owner.workspace->owns_primary_allocation() ||
      current_q1024_owner.workspace->split_allocation_offset() !=
          kCurrentQ1024SplitOffset ||
      current_q1024_owner.workspace->allocation() !=
          impl_->frozen_text_q1024_workspace.allocation()) {
    throw std::runtime_error(
        "native current q1024 split backing contract changed");
  }
  const NativePrefillInvocationMetrics current_q1024_invocation_metrics =
      current_q1024_owner.invocations->build(
          impl_->bindings, *current_q1024_owner.workspace, 1024);
  if (impl_->prefill_tokens == 1024) {
    prefill_workspace_metrics = current_q1024_workspace_metrics;
    prefill_invocation_metrics = current_q1024_invocation_metrics;
  } else if (impl_->tail_prefill_tokens == 1024) {
    tail_prefill_workspace_metrics = current_q1024_workspace_metrics;
    tail_prefill_invocation_metrics = current_q1024_invocation_metrics;
  } else {
    auxiliary_prefill_workspace_metrics.allocation_bytes +=
        current_q1024_workspace_metrics.allocation_bytes;
    auxiliary_prefill_workspace_metrics.physical_allocation_bytes +=
        current_q1024_workspace_metrics.physical_allocation_bytes;
    auxiliary_prefill_invocation_metrics.launch_count +=
        current_q1024_invocation_metrics.launch_count;
  }

  // Preserve the v1.5.1 text allocation topology, then complete every VL
  // resident before READY.  Language and vision weights remain independent
  // native owners so late vision residency cannot move text-critical buffers.
  NativeWeightLoadOptions visual_weight_options = options.weights;
  visual_weight_options.native_report = visual_weight_report;
  const NativeWeightLoadMetrics visual_weight_metrics =
      impl_->visual_weights.load_visual(visual_weight_options);
  if (visual_weight_metrics.device_name != weight_metrics.device_name ||
      visual_weight_metrics.gpu_arch != weight_metrics.gpu_arch ||
      visual_weight_metrics.model_config_sha256 !=
          weight_metrics.model_config_sha256 ||
      visual_weight_metrics.checkpoint_index_sha256 !=
          weight_metrics.checkpoint_index_sha256) {
    throw std::runtime_error(
        "native language and visual resident owners are incompatible");
  }
  weight_metrics.weight_set = "language+visual";
  weight_metrics.layout_manifest_sha256 =
      kResidentLayoutManifestSha256;
  weight_metrics.free_bytes_after = visual_weight_metrics.free_bytes_after;
  weight_metrics.payload_bytes += visual_weight_metrics.payload_bytes;
  weight_metrics.tensor_count += visual_weight_metrics.tensor_count;
  weight_metrics.visual_layout_manifest_sha256 =
      visual_weight_metrics.visual_layout_manifest_sha256;
  weight_metrics.visual_payload_bytes =
      visual_weight_metrics.visual_payload_bytes;
  weight_metrics.visual_tensor_count =
      visual_weight_metrics.visual_tensor_count;
  weight_metrics.visual_shard_count =
      visual_weight_metrics.visual_shard_count;
  weight_metrics.allocation_ms += visual_weight_metrics.allocation_ms;
  weight_metrics.ingest_ms += visual_weight_metrics.ingest_ms;
  weight_metrics.load_wall_ms += visual_weight_metrics.load_wall_ms;
  vl_logical_load_metrics = impl_->vl_logical_projections.build(
      impl_->weights, kNativeVlLogicalProjectionMaximumTokens,
      impl_->device);
  check_hip(hipMalloc(&impl_->mrope_positions,
                      impl_->mrope_position_state_bytes),
            "hipMalloc resident M-RoPE positions");
  check_hip(hipMalloc(&impl_->vl_prompt_index_state,
                      impl_->vl_prompt_index_state_bytes),
            "hipMalloc resident VL prompt/index state");
  check_hip(hipMalloc(&impl_->structured_token_mask,
                      impl_->structured_token_mask_bytes),
            "hipMalloc resident structured token mask");
  impl_->host_structured_token_mask.resize(kVocabulary);
  impl_->vl_prompt_token_ids = impl_->vl_prompt_index_state;
  impl_->vl_scatter_indices =
      static_cast<unsigned char*>(impl_->vl_prompt_index_state) +
      vl_prompt_token_id_bytes;
  // vLLM's singleton shared-expert gate falls through to PyTorch hipBLASLt.
  // Build the matching N=1 plan once; decode only reuses resident pointers.
  impl_->decode_shared_gate_plan = std::make_unique<Bf16GemmPlan>(
      1, 1, kHidden, 76ULL * 1024ULL * 1024ULL, true);
  impl_->vl_unified_attention =
      std::make_unique<NativeVlUnifiedAttentionPlan>(
          impl_->executor, impl_->resident_prefill_buckets.back(),
          options.cache_capacity, impl_->device);
  impl_->attention_state.bind_decode_unified_attention(
      impl_->vl_unified_attention.get());
  vision_warmup_metrics = impl_->warm_up_standard_vision();
  write_resident_weight_report(
      combined_weight_report, language_weight_report,
      visual_weight_report, weight_metrics);

  hipDeviceProp_t properties{};
  if (hipGetDeviceProperties(&properties, impl_->device) != hipSuccess) {
    throw std::runtime_error(
        "hipGetDeviceProperties failed for native resident engine");
  }
  impl_->cu_count = properties.multiProcessorCount;
  if (impl_->cu_count <= 0) {
    throw std::runtime_error("native resident engine has no compute units");
  }

  const auto find_prefill_start = [](const NativePrefillInvocations& owner,
                                     std::size_t launch_count) {
    std::size_t result = launch_count;
    for (std::size_t sequence = 0; sequence < owner.launches().size();
         ++sequence) {
      const auto* launch = owner.launches()[sequence].launch;
      if (launch != nullptr && launch->layer_index == 0 &&
          std::string(launch->symbol) == "triton_rmsnorm_kernel") {
        result = sequence;
        break;
      }
    }
    return result;
  };
  impl_->prefill_start_sequence = find_prefill_start(
      impl_->prefill_invocations, prefill_invocation_metrics.launch_count);
  if (impl_->prefill_start_sequence ==
      prefill_invocation_metrics.launch_count) {
    throw std::runtime_error(
        "native resident prefill entry binding is missing");
  }
  impl_->frozen_text_q1024_start_sequence = find_prefill_start(
      impl_->frozen_text_q1024_invocations,
      frozen_text_q1024_invocation_metrics.launch_count);
  if (impl_->frozen_text_q1024_start_sequence ==
      frozen_text_q1024_invocation_metrics.launch_count) {
    throw std::runtime_error(
        "native frozen text q1024 prefill entry binding is missing");
  }
  if (impl_->tail_prefill_tokens != 0) {
    impl_->tail_prefill_start_sequence = find_prefill_start(
        impl_->tail_prefill_invocations,
        tail_prefill_invocation_metrics.launch_count);
    if (impl_->tail_prefill_start_sequence ==
        tail_prefill_invocation_metrics.launch_count) {
      throw std::runtime_error(
          "native resident tail prefill entry binding is missing");
    }
  }
  for (const auto& bucket : impl_->auxiliary_prefill_buckets) {
    bucket->start_sequence = find_prefill_start(
        bucket->invocations, bucket->invocations.launches().size());
    if (bucket->start_sequence == bucket->invocations.launches().size()) {
      throw std::runtime_error(
          "native resident auxiliary prefill entry binding is missing");
    }
  }

  impl_->metrics.device_name = weight_metrics.device_name;
  impl_->metrics.gpu_arch = weight_metrics.gpu_arch;
  impl_->metrics.model_payload_bytes = weight_metrics.payload_bytes;
  impl_->metrics.model_tensor_count = weight_metrics.tensor_count;
  impl_->metrics.model_shard_count = weight_metrics.shard_count;
  impl_->metrics.language_model_payload_bytes =
      weight_metrics.language_payload_bytes;
  impl_->metrics.language_model_tensor_count =
      weight_metrics.language_tensor_count;
  impl_->metrics.language_model_shard_count =
      weight_metrics.language_shard_count;
  impl_->metrics.language_layout_manifest_sha256 =
      weight_metrics.language_layout_manifest_sha256;
  impl_->metrics.visual_model_payload_bytes =
      weight_metrics.visual_payload_bytes;
  impl_->metrics.visual_model_tensor_count =
      weight_metrics.visual_tensor_count;
  impl_->metrics.visual_model_shard_count =
      weight_metrics.visual_shard_count;
  impl_->metrics.visual_layout_manifest_sha256 =
      weight_metrics.visual_layout_manifest_sha256;
  impl_->metrics.decode_weight_bindings = binding_metrics.unique_bindings;
  impl_->metrics.prefill_prepared_launches =
      prefill_invocation_metrics.launch_count +
      tail_prefill_invocation_metrics.launch_count +
      auxiliary_prefill_invocation_metrics.launch_count +
      frozen_text_q1024_invocation_metrics.launch_count;
  impl_->metrics.decode_prepared_launches =
      decode_invocation_metrics.launch_count;
  impl_->metrics.aot_loaded_modules = executor_metrics.loaded_modules;
  impl_->metrics.prefill_gemm_plans =
      impl_->prefill_gemm_plans->built_plan_count() +
      (impl_->tail_prefill_gemm_plans == nullptr
           ? 0
           : impl_->tail_prefill_gemm_plans->built_plan_count());
  for (const auto& bucket : impl_->auxiliary_prefill_buckets) {
    impl_->metrics.prefill_gemm_plans +=
        bucket->gemm_plans->built_plan_count();
  }
  impl_->metrics.prefill_workspace_bytes =
      prefill_workspace_metrics.physical_allocation_bytes +
      tail_prefill_workspace_metrics.physical_allocation_bytes +
      auxiliary_prefill_workspace_metrics.physical_allocation_bytes +
      frozen_text_q1024_workspace_metrics.physical_allocation_bytes +
      impl_->chunked_hidden_bytes +
      impl_->padded_prefill_initial_conv_state_bytes +
      impl_->mrope_position_state_bytes;
  impl_->metrics.mrope_position_state_bytes =
      impl_->mrope_position_state_bytes;
  impl_->metrics.vl_unified_attention_metadata_bytes =
      impl_->vl_unified_attention->metrics().metadata_bytes;
  impl_->metrics.vl_unified_attention_decode_scratch_bytes =
      impl_->vl_unified_attention->metrics().decode_scratch_bytes;
  impl_->metrics.vl_unified_attention_image_bytes =
      impl_->vl_unified_attention->metrics().image_bytes;
  impl_->metrics.vl_unified_attention_loaded =
      impl_->vl_unified_attention->metrics().loaded;
  impl_->metrics.vl_logical_projection_weight_bytes =
      vl_logical_load_metrics.weight_bytes;
  impl_->metrics.vl_logical_projection_output_scratch_bytes =
      vl_logical_load_metrics.output_scratch_bytes;
  impl_->metrics.vl_logical_projection_weights_loaded =
      vl_logical_load_metrics.loaded;
  impl_->metrics.vl_prompt_index_state_bytes =
      impl_->vl_prompt_index_state_bytes;
  impl_->metrics.structured_token_mask_bytes =
      impl_->structured_token_mask_bytes;
  impl_->metrics.vision_plan_cache_capacity =
      kVisionPlanCacheEntries;
  impl_->metrics.vision_attention_image_sha256 =
      kVisionAttentionImageSha256;
  impl_->metrics.vision_dense_image_attention_image_sha256 =
      kDenseImageVisionAttentionImageSha256;
  impl_->metrics.vision_warmup_patches = vision_warmup_metrics.patches;
  impl_->metrics.vision_warmup_visual_tokens =
      vision_warmup_metrics.visual_tokens;
  impl_->metrics.vision_image_count_warmup_patches =
      vision_warmup_metrics.image_count_patches;
  impl_->metrics.vision_image_count_warmup_visual_tokens =
      vision_warmup_metrics.image_count_visual_tokens;
  impl_->metrics.vision_plan_cache_entries_at_ready =
      vision_warmup_metrics.plan_cache_entries;
  impl_->metrics.vision_warmup_plan_build_wall_ms =
      vision_warmup_metrics.plan_build_wall_ms;
  impl_->metrics.vision_warmup_encode_wall_ms =
      vision_warmup_metrics.encode_wall_ms;
  impl_->metrics.vision_image_count_warmup_plan_build_wall_ms =
      vision_warmup_metrics.image_count_plan_build_wall_ms;
  impl_->metrics.vision_image_count_warmup_encode_wall_ms =
      vision_warmup_metrics.image_count_encode_wall_ms;
  impl_->metrics.vision_warmup_completed =
      vision_warmup_metrics.completed;
  impl_->metrics.decode_workspace_bytes =
      decode_workspace_metrics.allocation_bytes;
  impl_->metrics.attention_state_bytes = attention_metrics.allocation_bytes;
  impl_->metrics.exact_prefix_cache_bytes = prefix_cache_bytes;
  impl_->metrics.prefix_cache_entries = impl_->prefix_cache_entries;
  impl_->metrics.cache_capacity = options.cache_capacity;
  impl_->metrics.prompt_tokens = impl_->prompt_tokens;
  impl_->metrics.resident_prefill_buckets =
      impl_->resident_prefill_buckets;
  impl_->metrics.fmha_provider_backend =
      fmha_provider_backend(provider_path);
  impl_->metrics.fmha_provider_path =
      provider_metrics.library_path.string();
  impl_->metrics.fmha_provider_loaded = provider_metrics.loaded;
  impl_->metrics.secondary_fmha_provider_backend =
      secondary_provider_path.empty()
          ? std::string{}
          : fmha_provider_backend(secondary_provider_path);
  impl_->metrics.secondary_fmha_provider_path =
      secondary_provider_metrics.library_path.string();
  impl_->metrics.secondary_fmha_layers =
      secondary_layers;
  impl_->metrics.secondary_fmha_provider_loaded =
      secondary_provider_metrics.loaded;
  impl_->metrics.ck_provider_loaded = provider_metrics.loaded;
  impl_->metrics.raw_weight_load_wall_ms = weight_metrics.load_wall_ms;
  impl_->metrics.derived_weight_build_wall_ms =
      derived_metrics.build_wall_ms;
  impl_->metrics.lm_head_build_wall_ms = lm_head_metrics.build_wall_ms;
  impl_->metrics.vl_logical_projection_weight_build_wall_ms =
      vl_logical_load_metrics.build_wall_ms;
  impl_->metrics.prefill_gemm_plan_build_wall_ms = plan_wall_ms;
  impl_->metrics.command_to_ready_wall_ms = elapsed_ms(started);
  impl_->ready = true;
  return impl_->metrics;
}

NativeResidentRequestMetrics NativeResidentEngine::run(
    const NativeResidentRequestOptions& request) {
  if (!impl_->ready ||
      !native_request_fits_capacity(request.input_token_ids.size(),
                                    request.max_new_tokens,
                                    impl_->attention_state.cache_capacity())) {
    throw std::invalid_argument(
        "native resident request context or output length is not admitted");
  }
  for (const std::uint32_t token : request.input_token_ids) {
    if (token >= kVocabulary) {
      throw std::invalid_argument(
          "native resident prompt token is outside the vocabulary");
    }
  }
  for (const std::uint32_t token : request.stop_token_ids) {
    if (token >= kVocabulary) {
      throw std::invalid_argument(
          "native resident stop token is outside the vocabulary");
    }
  }
  if (!request.disable_prefix_cache && impl_->prefix_cache_entries == 0) {
    throw std::invalid_argument(
        "native resident prefix cache was disabled before READY");
  }
  const bool has_decode_observer =
      static_cast<bool>(request.decode_layer_observer) ||
      static_cast<bool>(request.decode_linear_layer0_observer) ||
      static_cast<bool>(request.decode_layer0_tail_observer) ||
      static_cast<bool>(request.decode_full_attention_observer);
  const bool has_linear_decode_observer =
      static_cast<bool>(request.decode_linear_layer0_observer) ||
      static_cast<bool>(request.decode_layer0_tail_observer);
  if (has_decode_observer !=
          request.decode_layer_observer_output_index.has_value() ||
      (request.decode_layer_observer_output_index.has_value() &&
       (*request.decode_layer_observer_output_index == 0 ||
        *request.decode_layer_observer_output_index >=
            request.max_new_tokens))) {
    throw std::invalid_argument(
        "native decode layer observer target is invalid");
  }
  if (has_linear_decode_observer &&
      (request.decode_linear_observer_layer_index >= 40 ||
       request.decode_linear_observer_layer_index % 4 == 3)) {
    throw std::invalid_argument(
        "native decode linear observer layer is invalid");
  }
  if (!valid_native_multimodal_cache_namespace(
          request.multimodal_cache_namespace)) {
    throw std::invalid_argument(
        "native resident multimodal cache namespace is invalid");
  }
  const NativeMropePlan* mrope_plan =
      request.mrope_plan.has_value() ? &request.mrope_plan.value() : nullptr;
  if (mrope_plan != nullptr) {
    if (request.multimodal_cache_namespace.empty() ||
        mrope_plan->prompt_token_count() != request.input_token_ids.size() ||
        mrope_plan->positions().size() !=
            3 * request.input_token_ids.size() ||
        mrope_plan->maximum_position() < 0 ||
        mrope_plan->maximum_position() >= 262144) {
      throw std::invalid_argument(
          "native resident M-RoPE request contract is invalid");
    }
    if (request.max_new_tokens > 1) {
      const std::int64_t final_rotary_position =
          native_mrope_decode_position(
              request.input_token_ids.size(),
              mrope_plan->position_delta(), request.max_new_tokens - 2);
      if (final_rotary_position < 0 || final_rotary_position >= 262144) {
        throw std::invalid_argument(
            "native resident M-RoPE decode position is unsupported");
      }
    }
  }
  const NativeResidentVlInput* vl_input =
      request.vl_input.has_value() ? &request.vl_input.value() : nullptr;
  std::optional<NativeVlEmbeddingPlan> vl_embedding_plan;
  std::vector<NativeVlVisionBatch> vl_vision_batches;
  std::size_t vl_vision_patches = 0;
  std::size_t vl_visual_tokens = 0;
  if (vl_input != nullptr) {
    if (mrope_plan == nullptr || request.multimodal_cache_namespace.empty() ||
        vl_input->grids.empty() ||
        vl_input->grids.size() != vl_input->embedding_spans.size() ||
        vl_input->media_count != vl_input->grids.size() ||
        vl_input->image_count + vl_input->video_count !=
            vl_input->media_count) {
      throw std::invalid_argument(
          "native resident VL request contract is incomplete");
    }
    if (!vl_input->vision_embedding_cache_namespace.empty() &&
        !valid_native_multimodal_cache_namespace(
            vl_input->vision_embedding_cache_namespace)) {
      throw std::invalid_argument(
          "native resident vision embedding cache identity is invalid");
    }
    vl_vision_batches = native_qwen36_vision_batches(vl_input->grids);
    const NativeVlVisionBatch& final_batch = vl_vision_batches.back();
    vl_vision_patches = final_batch.patch_offset + final_batch.patch_count;
    vl_visual_tokens = final_batch.visual_token_offset +
                       final_batch.visual_token_count;
    std::size_t derived_images = 0;
    std::size_t derived_videos = 0;
    for (std::size_t index = 0; index < vl_input->grids.size(); ++index) {
      const NativeVlGrid& grid = vl_input->grids[index];
      const NativeVlEmbeddingSpan& span = vl_input->embedding_spans[index];
      const std::size_t visual_tokens = grid.language_token_count();
      if (span.visual_embedding_count != visual_tokens) {
        throw std::invalid_argument(
            "native resident VL grid/span budget is invalid");
      }
      switch (span.kind) {
        case NativeMediaKind::kImage:
          if (visual_tokens > kNativeVlImageTokenLimit) {
            throw std::invalid_argument(
                "native resident VL image grid exceeds its token limit");
          }
          ++derived_images;
          break;
        case NativeMediaKind::kVideo:
          if (visual_tokens > kNativeVlVideoTokenLimit) {
            throw std::invalid_argument(
                "native resident VL video grid exceeds its token limit");
          }
          ++derived_videos;
          break;
      }
    }
    if (derived_images != vl_input->image_count ||
        derived_videos != vl_input->video_count ||
        vl_vision_patches >
            std::numeric_limits<std::size_t>::max() / kVisionPixelColumns ||
        vl_input->pixel_values_bf16.size() !=
            vl_vision_patches * kVisionPixelColumns) {
      throw std::invalid_argument(
          "native resident VL pixel/media shape is invalid");
    }
    vl_embedding_plan = build_native_vl_embedding_plan(
        request.input_token_ids, vl_input->embedding_spans,
        vl_visual_tokens);
  }
  std::size_t matched_prefix_tokens = 0;
  std::size_t matched_prefix_cache_index = impl_->prefix_cache_entries;
  if (!request.disable_prefix_cache) {
    for (std::size_t index = 0; index < impl_->prefix_cache_entries; ++index) {
      const std::size_t matched =
          impl_->prefix_caches[index].matched_prefix_tokens(
              request.input_token_ids,
              request.multimodal_cache_namespace);
      if (matched > matched_prefix_tokens) {
        matched_prefix_tokens = matched;
        matched_prefix_cache_index = index;
      }
    }
  }
  const NativePromptExecutionPlan prompt_plan = plan_native_prompt_execution(
      request.input_token_ids.size(), matched_prefix_tokens,
      impl_->resident_prefill_buckets);
  if (matched_prefix_tokens + prompt_plan.aot_bucket_tokens >
      impl_->attention_state.cache_capacity()) {
    throw std::invalid_argument(
        "native resident cache has no padded-prefill headroom");
  }
  const bool prefix_hit = prompt_plan.prefix_hit;
  const bool exact_prefix_hit = prompt_plan.exact_prefix_hit;
  const bool prefix_extension_hit = prompt_plan.prefix_extension_hit;
  const bool reuse_active_prefix_kv =
      exact_prefix_hit &&
      impl_->active_kv_prefix_cache_index == matched_prefix_cache_index;
  // A request may overwrite the live attention cache. Re-establish ownership
  // only after the selected cache state has been restored or captured.
  impl_->active_kv_prefix_cache_index = impl_->prefix_cache_entries;
  if (prompt_plan.prompt_decode_required(request.input_token_ids.size())) {
    throw std::runtime_error(
        "native prompt planner left an unexpected serial decode tail");
  }
  const bool exact_configured_prefill =
      !prefix_hit && request.input_token_ids.size() == impl_->prompt_tokens &&
      !prompt_plan.padded_aot();
  if (!exact_configured_prefill &&
      request.secondary_fmha_layers_override_provided) {
    throw std::invalid_argument(
        "native provider-mask diagnostics require the configured prefill "
        "endpoint");
  }
  const bool segmented_or_padded_prefill =
      prompt_plan.aot_segments.size() > 1 || prompt_plan.padded_aot();
  if (segmented_or_padded_prefill &&
      (!request.layer_tail_oracle_dir.empty() ||
       !request.layer_sequence_oracle_dir.empty())) {
    throw std::invalid_argument(
        "native composed prefill does not admit layer-oracle diagnostics");
  }
  std::size_t vl_logical_projection_tokens = 0;
  std::size_t vl_logical_projection_bucket_tokens = 0;
  if (mrope_plan != nullptr) {
    for (const NativePromptAotSegment& segment : prompt_plan.aot_segments) {
      if (segment.bucket_tokens > 2048 ||
          segment.input_tokens >
              kNativeVlLogicalProjectionMaximumTokens ||
          !segment.padded()) {
        continue;
      }
      if (vl_logical_projection_tokens != 0 &&
          (vl_logical_projection_tokens != segment.input_tokens ||
           vl_logical_projection_bucket_tokens !=
               segment.bucket_tokens)) {
        throw std::runtime_error(
            "native VL request has multiple logical prefill shapes");
      }
      vl_logical_projection_tokens = segment.input_tokens;
      vl_logical_projection_bucket_tokens = segment.bucket_tokens;
    }
  }

  NativeResidentRequestMetrics metrics;
  metrics.request_index = ++impl_->request_count;
  metrics.prompt_tokens = request.input_token_ids.size();
  if (vl_input != nullptr) {
    std::ostringstream prompt_token_payload;
    prompt_token_payload << '[';
    for (std::size_t index = 0; index < request.input_token_ids.size();
         ++index) {
      if (index != 0) prompt_token_payload << ',';
      prompt_token_payload << request.input_token_ids[index];
    }
    prompt_token_payload << ']';
    const std::string prompt_token_payload_value =
        prompt_token_payload.str();
    metrics.prompt_token_ids_sha256 = sha256_bytes(
        prompt_token_payload_value.data(), prompt_token_payload_value.size());
  }
  metrics.model_loads = 1;
  metrics.oracle_tensor_reads = 0;
  auto prepare_next_token_mask = [&]() -> const std::uint8_t* {
    if (!request.next_token_mask) return nullptr;
    if (impl_->structured_token_mask == nullptr ||
        impl_->structured_token_mask_bytes != kVocabulary ||
        impl_->host_structured_token_mask.size() != kVocabulary) {
      throw std::runtime_error(
          "native resident structured token-mask owner is incomplete");
    }
    request.next_token_mask(metrics.output_token_ids,
                            &impl_->host_structured_token_mask);
    if (impl_->host_structured_token_mask.size() != kVocabulary ||
        std::none_of(impl_->host_structured_token_mask.begin(),
                     impl_->host_structured_token_mask.end(),
                     [](std::uint8_t value) { return value != 0; })) {
      throw std::runtime_error(
          "native resident token grammar produced an invalid mask");
    }
    check_hip(
        hipMemcpyAsync(impl_->structured_token_mask,
                       impl_->host_structured_token_mask.data(),
                       impl_->structured_token_mask_bytes,
                       hipMemcpyHostToDevice, nullptr),
        "hipMemcpyAsync resident structured token mask");
    metrics.constrained_decoding = true;
    ++metrics.constrained_token_selections;
    metrics.constrained_token_mask_upload_bytes +=
        impl_->structured_token_mask_bytes;
    return static_cast<const std::uint8_t*>(
        impl_->structured_token_mask);
  };
  metrics.state_orientation_resets =
      impl_->decode_invocations.reset_linear_decode_conv_state_buffers();
  metrics.state_orientation_resets +=
      impl_->decode_invocations.reset_linear_decode_recurrent_state_buffers();
  metrics.request_state_reset_bytes =
      impl_->attention_state.clear_request_scratch();
  if (!prefix_hit && prompt_plan.aot_segments.empty()) {
    metrics.request_state_reset_bytes += impl_->decode_workspace.clear();
  }
  metrics.prompt_execution =
      native_prompt_execution_mode_name(prompt_plan.mode);
  metrics.aot_prefill_tokens = prompt_plan.cold_aot_tokens;
  metrics.aot_prefill_bucket_tokens = prompt_plan.aot_bucket_tokens;
  metrics.aot_prefill_segments = prompt_plan.aot_segments.size();
  metrics.padded_prefill_tokens =
      prompt_plan.aot_bucket_tokens - prompt_plan.cold_aot_tokens;
  metrics.mrope_enabled = mrope_plan != nullptr;
  metrics.vl_enabled = vl_input != nullptr;
  if (vl_input != nullptr) {
    metrics.vl_media_count = vl_input->media_count;
    metrics.vl_image_count = vl_input->image_count;
    metrics.vl_video_count = vl_input->video_count;
    metrics.vl_source_bytes = vl_input->source_bytes;
    metrics.vl_vision_patches = vl_vision_patches;
    metrics.vl_visual_tokens = vl_visual_tokens;
    metrics.vl_vision_batch_count = vl_vision_batches.size();
    for (const NativeVlVisionBatch& batch : vl_vision_batches) {
      metrics.vl_vision_max_batch_patches = std::max(
          metrics.vl_vision_max_batch_patches, batch.patch_count);
      metrics.vl_vision_max_batch_tokens = std::max(
          metrics.vl_vision_max_batch_tokens,
          batch.visual_token_count);
    }
    metrics.vl_media_cache_hits = vl_input->media_cache_hits;
    metrics.vl_media_cache_misses = vl_input->media_cache_misses;
    metrics.vl_media_cache_entries = vl_input->media_cache_entries;
    metrics.vl_media_cache_resident_bytes =
        vl_input->media_cache_resident_bytes;
    metrics.vl_media_load_wall_ms = vl_input->media_load_wall_ms;
    metrics.vl_media_decode_wall_ms = vl_input->media_decode_wall_ms;
    metrics.vl_media_load_decode_wall_ms =
        vl_input->media_load_decode_wall_ms;
    metrics.vl_processor_wall_ms = vl_input->processor_wall_ms;
    metrics.vl_vision_plan_cache_entries = impl_->vision_plans.size();
    metrics.vl_vision_embedding_cache_entries =
        impl_->vision_embedding_cache.size();
    metrics.vl_vision_embedding_cache_resident_bytes =
        impl_->vision_embedding_cache_resident_bytes;
    metrics.vl_vision_embedding_cache_capacity_bytes =
        kVisionEmbeddingCacheBytes;
  }
  if (mrope_plan != nullptr) {
    metrics.mrope_position_delta = mrope_plan->position_delta();
  }
  if (mrope_plan != nullptr && !prompt_plan.aot_segments.empty()) {
    const std::size_t padded_position_end =
        matched_prefix_tokens + prompt_plan.aot_bucket_tokens;
    if (impl_->mrope_positions == nullptr ||
        impl_->mrope_position_row_stride < padded_position_end ||
        impl_->mrope_position_state_bytes <
            3ULL * impl_->mrope_position_row_stride *
                sizeof(std::int64_t)) {
      throw std::runtime_error(
          "native resident M-RoPE position owner is incomplete");
    }
    auto* device_positions =
        static_cast<std::int64_t*>(impl_->mrope_positions);
    const std::vector<std::int64_t>& host_positions =
        mrope_plan->positions();
    const std::size_t prompt_tokens = request.input_token_ids.size();
    const std::size_t upload_tokens =
        prompt_tokens - matched_prefix_tokens;
    for (std::size_t axis = 0; axis < 3; ++axis) {
      check_hip(
          hipMemcpyAsync(
              device_positions +
                  axis * impl_->mrope_position_row_stride +
                  matched_prefix_tokens,
              host_positions.data() + axis * prompt_tokens +
                  matched_prefix_tokens,
              upload_tokens * sizeof(std::int64_t),
              hipMemcpyHostToDevice, nullptr),
          "hipMemcpyAsync resident M-RoPE positions");
      if (padded_position_end > prompt_tokens) {
        check_hip(
            hipMemsetAsync(
                device_positions +
                    axis * impl_->mrope_position_row_stride + prompt_tokens,
                0,
                (padded_position_end - prompt_tokens) *
                    sizeof(std::int64_t),
                nullptr),
            "hipMemsetAsync resident M-RoPE padding");
      }
    }
    metrics.mrope_position_upload_bytes =
        3ULL * upload_tokens * sizeof(std::int64_t);
  }
  std::array<bool, 40> request_secondary_fmha_layers =
      impl_->secondary_fmha_layers;
  if (!exact_configured_prefill) {
    request_secondary_fmha_layers.fill(false);
  }
  if (request.secondary_fmha_layers_override_provided) {
    request_secondary_fmha_layers.fill(false);
    for (std::size_t layer : request.secondary_fmha_layers_override) {
      if (layer >= 40 || layer % 4 != 3 ||
          request_secondary_fmha_layers[layer]) {
        throw std::invalid_argument(
            "native request secondary FMHA layers must be unique "
            "full-attention layers");
      }
      request_secondary_fmha_layers[layer] = true;
    }
  }
  const auto request_started = std::chrono::steady_clock::now();
  if (vl_logical_projection_tokens != 0) {
    NativeResidentPrefillOwner logical_owner = impl_->prefill_owner(
        vl_logical_projection_bucket_tokens, false);
    if (logical_owner.gemm_plans == nullptr) {
      throw std::runtime_error(
          "native VL logical prefill GEMM source is unavailable");
    }
    const NativeVlLogicalProjectionPrepareMetrics logical_metrics =
        impl_->vl_logical_projections.prepare(
            vl_logical_projection_tokens, *logical_owner.gemm_plans);
    metrics.vl_logical_projections_enabled = logical_metrics.prepared;
    metrics.vl_logical_projection_tokens = logical_metrics.tokens;
    metrics.vl_logical_projection_plan_count = logical_metrics.plan_count;
    metrics.vl_logical_projection_workspace_bytes =
        logical_metrics.workspace_bytes;
    metrics.vl_logical_projection_plan_build_wall_ms =
        logical_metrics.build_wall_ms;
    metrics.vl_logical_projection_plan_reused = logical_metrics.reused;
  }
  const bool timeline_enabled = prefill_timeline_enabled() && !prefix_hit &&
                                !prompt_plan.aot_segments.empty();
  std::vector<double> attention_wall_ms;
  std::vector<double> moe_wall_ms;
  double embedding_wall_ms = 0.0;
  double first_token_wall_ms = 0.0;
  const void* last_hidden = nullptr;
  bool exact_prefix_restore_pending = false;
  if (prefix_hit) {
    metrics.prefix_cache_lookup = exact_prefix_hit ? "exact" : "prefix";
    metrics.prefix_cache_matched_tokens = matched_prefix_tokens;
    metrics.prefix_cache_suffix_tokens =
        request.input_token_ids.size() - matched_prefix_tokens;
    ++impl_->prefix_cache_hits;
    if (matched_prefix_cache_index >= impl_->prefix_cache_entries) {
      throw std::runtime_error(
          "native prefix-cache lookup lost its resident entry");
    }
    impl_->prefix_cache_use[matched_prefix_cache_index] =
        ++impl_->prefix_cache_clock;
    if (exact_prefix_hit) {
      last_hidden =
          impl_->prefix_caches[matched_prefix_cache_index].terminal_hidden();
      if (reuse_active_prefix_kv) {
        // Decode only appends after the prompt, so a consecutive exact hit
        // still owns byte-identical prompt KV in the live attention cache.
        // Restore only the mutable linear recurrent/conv state. This avoids
        // copying the prompt KV a second time without changing cache state.
        const auto restore_started = std::chrono::steady_clock::now();
        metrics.prefix_cache_transfer_bytes =
            impl_->prefix_caches[matched_prefix_cache_index]
                .restore_linear_state();
        check_hip(hipStreamSynchronize(nullptr),
                  "hipStreamSynchronize active-KV exact-prefix restore");
        metrics.prefix_cache_restore_wall_ms = elapsed_ms(restore_started);
        metrics.prefix_cache_active_kv_reused = true;
        impl_->active_kv_prefix_cache_index = matched_prefix_cache_index;
      } else {
        // The cached terminal hidden is sufficient to publish the first
        // token. Restore a non-active KV owner before the next decode step.
        exact_prefix_restore_pending = true;
      }
    } else {
      metrics.prefix_cache_transfer_bytes =
          impl_->prefix_caches[matched_prefix_cache_index].restore();
      impl_->active_kv_prefix_cache_index = matched_prefix_cache_index;
    }
  } else {
    metrics.prefix_cache_lookup =
        request.disable_prefix_cache ? "disabled" : "miss";
    metrics.prefix_cache_matched_tokens = 0;
    metrics.prefix_cache_suffix_tokens = request.input_token_ids.size();
    if (!request.disable_prefix_cache) ++impl_->prefix_cache_misses;
  }

  if (!prompt_plan.aot_segments.empty()) {
    const bool composed_prefill = prompt_plan.aot_segments.size() > 1;
    const NativeTensorView* embedding =
        impl_->weights.find("model.language_model.embed_tokens.weight");
    if (embedding == nullptr || embedding->device_pointer == nullptr) {
      throw std::runtime_error(
          "native resident prompt embedding weight is missing");
    }
    if (timeline_enabled) {
      check_hip(hipDeviceSynchronize(),
                "hipDeviceSynchronize before prefill timeline");
    }
    const auto embedding_started = std::chrono::steady_clock::now();
    if (vl_input != nullptr && !prefix_hit) {
      if (!vl_embedding_plan.has_value() ||
          matched_prefix_tokens != 0 ||
          impl_->vl_prompt_token_ids == nullptr ||
          impl_->vl_scatter_indices == nullptr) {
        throw std::runtime_error(
            "native resident VL embedding owner is incomplete");
      }
      const std::uint64_t pixel_bytes =
          vl_input->pixel_values_bf16.size() * sizeof(std::uint16_t);
      const std::uint64_t embedding_bytes =
          vl_visual_tokens * kHidden * sizeof(std::uint16_t);
      impl_->ensure_vision_allocation(
          &impl_->vision_embeddings,
          &impl_->vision_embedding_capacity_bytes, embedding_bytes,
          "hipMalloc resident vision embeddings");
      const bool vision_embedding_cache_hit =
          impl_->restore_vision_embedding_cache(
              vl_input->vision_embedding_cache_namespace,
              embedding_bytes);
      metrics.vl_vision_embedding_cache_hit =
          vision_embedding_cache_hit;
      if (vision_embedding_cache_hit) {
        // The cached output was produced by a resident plan in this engine.
        // No plan or encoder execution is needed on the exact media hit.
        metrics.vl_vision_plan_cache_hit = true;
      } else {
        bool all_vision_plan_cache_hits = true;
        const bool image_only_request =
            vl_input->video_count == 0 &&
            vl_input->image_count == vl_input->media_count;
        for (const NativeVlVisionBatch& batch : vl_vision_batches) {
          const auto grid_begin =
              vl_input->grids.begin() +
              static_cast<std::ptrdiff_t>(batch.media_offset);
          const std::vector<NativeVlGrid> batch_grids(
              grid_begin,
              grid_begin + static_cast<std::ptrdiff_t>(batch.media_count));
          bool batch_cache_hit = false;
          double batch_plan_build_wall_ms = 0.0;
          NativeVisionPipelinePlan& pipeline = impl_->vision_plan(
              batch_grids, image_only_request, &batch_cache_hit,
              &batch_plan_build_wall_ms);
          if (pipeline.patch_count() != batch.patch_count ||
              pipeline.merged_token_count() !=
                  batch.visual_token_count) {
            throw std::runtime_error(
                "native resident vision batch shape is inconsistent");
          }
          metrics.vl_vision_attention_image_sha256s.push_back(
              pipeline.attention_image_sha256());
          const std::uint64_t batch_pixel_bytes =
              batch.patch_count * kVisionPixelColumns *
              sizeof(std::uint16_t);
          impl_->ensure_vision_allocation(
              &impl_->vision_pixel_values,
              &impl_->vision_pixel_capacity_bytes, batch_pixel_bytes,
              "hipMalloc resident vision pixel input");
          impl_->ensure_vision_allocation(
              &impl_->vision_temporary,
              &impl_->vision_temporary_capacity_bytes,
              pipeline.temporary_bytes(),
              "hipMalloc resident vision temporary arena");
          const auto* batch_pixels =
              vl_input->pixel_values_bf16.data() +
              batch.patch_offset * kVisionPixelColumns;
          auto* batch_embeddings =
              static_cast<unsigned char*>(impl_->vision_embeddings) +
              batch.visual_token_offset * kHidden *
                  sizeof(std::uint16_t);
          const auto upload_started = std::chrono::steady_clock::now();
          check_hip(
              hipMemcpyAsync(impl_->vision_pixel_values, batch_pixels,
                             batch_pixel_bytes, hipMemcpyHostToDevice,
                             nullptr),
              "hipMemcpyAsync resident vision batch pixels");
          check_hip(hipDeviceSynchronize(),
                    "hipDeviceSynchronize resident vision batch upload");
          metrics.vl_vision_input_upload_wall_ms +=
              elapsed_ms(upload_started);
          const auto vision_started = std::chrono::steady_clock::now();
          pipeline.launch(impl_->vision_pixel_values, batch_embeddings,
                          impl_->vision_temporary,
                          impl_->vision_temporary_capacity_bytes);
          check_hip(hipDeviceSynchronize(),
                    "hipDeviceSynchronize resident vision batch");
          metrics.vl_vision_encode_wall_ms +=
              elapsed_ms(vision_started);
          metrics.vl_vision_plan_build_wall_ms +=
              batch_plan_build_wall_ms;
          all_vision_plan_cache_hits =
              all_vision_plan_cache_hits && batch_cache_hit;
        }
        metrics.vl_vision_plan_cache_hit = all_vision_plan_cache_hits;
        impl_->insert_vision_embedding_cache(
            vl_input->vision_embedding_cache_namespace,
            embedding_bytes);
      }
      metrics.vl_vision_plan_cache_entries = impl_->vision_plans.size();
      metrics.vl_vision_embedding_cache_entries =
          impl_->vision_embedding_cache.size();
      metrics.vl_vision_embedding_cache_resident_bytes =
          impl_->vision_embedding_cache_resident_bytes;

      const NativePromptAotSegment& first_segment =
          prompt_plan.aot_segments.front();
      const NativeResidentPrefillOwner first_owner =
          impl_->prefill_owner(first_segment.bucket_tokens, false);
      void* embedding_output =
          composed_prefill
              ? impl_->chunked_hidden
              : first_owner.invocations->tensor_pointer(
                    first_owner.start_sequence, "x");
      const auto injection_started = std::chrono::steady_clock::now();
      launch_native_vl_embeddings(
          embedding->device_pointer, request.input_token_ids.data(),
          vl_embedding_plan.value(), impl_->vision_embeddings,
          impl_->vl_prompt_token_ids, impl_->vl_scatter_indices,
          embedding_output);
      check_hip(hipDeviceSynchronize(),
                "hipDeviceSynchronize resident VL embedding injection");
      metrics.vl_embedding_injection_wall_ms =
          elapsed_ms(injection_started);
      metrics.vl_host_to_device_bytes =
          (vision_embedding_cache_hit ? 0 : pixel_bytes) +
          request.input_token_ids.size() * sizeof(std::uint32_t) +
          vl_embedding_plan->device_index_bytes();
      metrics.prefill_native_pointwise_launches += 2;
    } else {
      for (const NativePromptAotSegment& segment :
           prompt_plan.aot_segments) {
        const NativeResidentPrefillOwner owner =
            impl_->prefill_owner(segment.bucket_tokens, true);
        const NativePrefillWorkspaceView* token_ids =
            owner.workspace->find("native.prompt_token_ids");
        if (token_ids == nullptr || token_ids->device_pointer == nullptr) {
          throw std::runtime_error(
              "native resident composed-prefill token owner is missing");
        }
        const std::size_t hidden_offset =
            segment.input_offset - matched_prefix_tokens;
        void* embedding_output =
            composed_prefill
                ? static_cast<unsigned char*>(impl_->chunked_hidden) +
                      hidden_offset * kHidden * sizeof(std::uint16_t)
                : owner.invocations->tensor_pointer(owner.start_sequence,
                                                    "x");
        launch_prompt_embeddings(
            embedding->device_pointer,
            request.input_token_ids.data() + segment.input_offset,
            token_ids->device_pointer, embedding_output,
            segment.input_tokens);
        ++metrics.prefill_native_pointwise_launches;
      }
    }
    if (timeline_enabled) {
      check_hip(hipDeviceSynchronize(),
                "hipDeviceSynchronize after prompt embeddings");
      embedding_wall_ms = elapsed_ms(embedding_started);
      attention_wall_ms.reserve(40);
      moe_wall_ms.reserve(40);
    }

    for (std::size_t layer_index = 0; layer_index < 40; ++layer_index) {
      double layer_attention_wall_ms = 0.0;
      double layer_moe_wall_ms = 0.0;
      for (std::size_t segment_index = 0;
           segment_index < prompt_plan.aot_segments.size(); ++segment_index) {
        const NativePromptAotSegment& segment =
            prompt_plan.aot_segments[segment_index];
        const NativeResidentPrefillOwner owner =
            impl_->prefill_owner(segment.bucket_tokens, vl_input == nullptr);
        NativePrefillWorkspace& chunk_workspace = *owner.workspace;
        NativePrefillInvocations& chunk_invocations = *owner.invocations;
        NativeQ8192PrefillGemmPlans* chunk_gemm_plans = owner.gemm_plans;
        const bool logical_vl_segment =
            metrics.vl_logical_projections_enabled &&
            segment.bucket_tokens <= 2048 &&
            segment.input_tokens <=
                kNativeVlLogicalProjectionMaximumTokens &&
            segment.padded() &&
            segment.input_tokens == metrics.vl_logical_projection_tokens;
        const std::size_t hidden_offset =
            segment.input_offset - matched_prefix_tokens;
        void* layer_input = native_prefill_layer_input_pointer(
            chunk_workspace, chunk_invocations, layer_index);
        if (composed_prefill) {
          check_hip(
              hipMemcpyAsync(
                  layer_input,
                  static_cast<const unsigned char*>(impl_->chunked_hidden) +
                      hidden_offset * kHidden * sizeof(std::uint16_t),
                  segment.input_tokens * kHidden * sizeof(std::uint16_t),
                  hipMemcpyDeviceToDevice, nullptr),
              "hipMemcpyAsync composed prefill layer input");
        }
        if (segment.padded()) {
          check_hip(
              hipMemsetAsync(
                  static_cast<unsigned char*>(layer_input) +
                      segment.input_tokens * kHidden * sizeof(std::uint16_t),
                  0,
                  (segment.bucket_tokens - segment.input_tokens) * kHidden *
                      sizeof(std::uint16_t),
                  nullptr),
              "hipMemsetAsync composed prefill padding");
        }
        const auto attention_started = std::chrono::steady_clock::now();
        const bool focused_tail_oracle =
            !request.layer_tail_oracle_dir.empty() &&
            request.layer_tail_oracle_index < 40 &&
            request.layer_tail_oracle_index == layer_index;
        const bool focused_sequence_oracle =
            !request.layer_sequence_oracle_dir.empty() &&
            request.layer_tail_oracle_index <= 40 &&
            (request.layer_tail_oracle_index == 40 ||
             request.layer_tail_oracle_index == layer_index);
        if (layer_index % 4 == 3) {
          NativeFullPrefillOracleOptions attention_options;
          attention_options.layer_index = layer_index;
          attention_options.active_tokens = segment.input_tokens;
          attention_options.comparison_tokens = segment.input_tokens;
          attention_options.seed_layer_input = false;
          attention_options.prepare_rotary_table = true;
          attention_options.collect_oracle_comparisons = false;
          attention_options.synchronize_substages = timeline_enabled;
          attention_options.decode_attention_state = &impl_->attention_state;
          attention_options.gemm_plans = chunk_gemm_plans;
          if (logical_vl_segment && segment.input_offset == 0) {
            attention_options.gemm_plans =
                &impl_->vl_logical_projections.router_gemm_plans();
          }
          attention_options.bindings = &impl_->bindings;
          attention_options.vl_unified_attention =
              impl_->vl_unified_attention.get();
          attention_options.cache_position_start = segment.input_offset;
          if (mrope_plan != nullptr) {
            attention_options.mrope_positions_i64 =
                static_cast<const std::int64_t*>(impl_->mrope_positions) +
                segment.input_offset;
            attention_options.mrope_position_row_stride =
                impl_->mrope_position_row_stride;
            ++metrics.mrope_full_attention_launches;
          }
          if (focused_tail_oracle) {
            std::ostringstream prefix;
            prefix << "layer-";
            prefix.width(3);
            prefix.fill('0');
            prefix << layer_index << '-';
            attention_options.tail_oracle_dir = request.layer_tail_oracle_dir;
            attention_options.tail_oracle_label_prefix = prefix.str();
          }
          if (focused_sequence_oracle) {
            std::ostringstream prefix;
            prefix << "layer-";
            prefix.width(3);
            prefix.fill('0');
            prefix << layer_index << '-';
            attention_options.sequence_oracle_dir =
                request.layer_sequence_oracle_dir;
            attention_options.sequence_oracle_label_prefix = prefix.str();
          }
          NativeQ8192CkProvider* segment_provider = owner.fmha_provider;
          if (mrope_plan != nullptr && segment.input_offset != 0) {
            // The short AOTriton owner is qualified for standalone q1024-
            // q4096 buckets, but its selected image rejects a short query
            // against a longer prefix. Continuation M-RoPE segments use the
            // generic rectangular CK owner shared by the long-context path.
            segment_provider = &impl_->ck_provider;
          }
          NativeQ8192CkProvider& attention_provider =
              request_secondary_fmha_layers[layer_index]
                  ? impl_->secondary_fmha_provider
                  : *segment_provider;
          const NativeFullPrefillOracleResult attention =
              probe_native_q8192_full_prefill_oracle(
                  {}, impl_->weights, chunk_workspace, chunk_invocations,
                  impl_->executor, attention_provider, attention_options);
          metrics.prefill_aot_launches += attention.layer.aot_launches;
          metrics.prefill_dense_gemm_launches +=
              attention.layer.dense_gemm_launches;
          metrics.prefill_native_pointwise_launches +=
              attention.layer.native_pointwise_launches;
          metrics.prefill_ck_fmha_launches +=
              attention.layer.native_ck_fmha_launches;
          metrics.prefill_vl_unified_attention_launches +=
              attention.layer.native_vl_unified_attention_launches;
          metrics.layer_tail_comparisons.insert(
              metrics.layer_tail_comparisons.end(),
              attention.boundary_comparisons.begin(),
              attention.boundary_comparisons.end());
        } else {
          const bool has_initial_state = prefix_hit || segment_index != 0;
          if (segment.padded()) {
            const NativeDecodeWorkspaceView* conv_state =
                impl_->decode_workspace.find(
                    "linear_attention_initial_conv_states." +
                    std::to_string(layer_index));
            if (conv_state == nullptr ||
                conv_state->device_pointer == nullptr ||
                conv_state->payload_bytes !=
                    impl_->padded_prefill_initial_conv_state_bytes) {
              throw std::runtime_error(
                  "native padded prefill convolution state is missing");
            }
            if (has_initial_state) {
              check_hip(
                  hipMemcpyAsync(impl_->padded_prefill_initial_conv_state,
                                 conv_state->device_pointer,
                                 impl_->padded_prefill_initial_conv_state_bytes,
                                 hipMemcpyDeviceToDevice, nullptr),
                  "hipMemcpyAsync padded prefill convolution snapshot");
            } else {
              check_hip(
                  hipMemsetAsync(impl_->padded_prefill_initial_conv_state, 0,
                                 impl_->padded_prefill_initial_conv_state_bytes,
                                 nullptr),
                  "hipMemsetAsync padded prefill convolution snapshot");
            }
          }
          NativeLinearPrefillOracleOptions attention_options;
          attention_options.layer_index = layer_index;
          attention_options.use_vl_rmsnorm_semantics = vl_input != nullptr;
          attention_options.seed_layer_input = false;
          attention_options.run_output_projection_diagnostic = false;
          attention_options.collect_oracle_comparisons = false;
          attention_options.comparison_tokens = segment.input_tokens;
          if (layer_index == 0 &&
              !request.multimodal_cache_namespace.empty() &&
              segment.bucket_tokens == 1024 && segment.input_tokens <= 64) {
            attention_options.exact_b_projection_tokens =
                segment.input_tokens;
          }
          attention_options.decode_state_workspace = &impl_->decode_workspace;
          attention_options.has_initial_state = has_initial_state;
          attention_options.gemm_plans = chunk_gemm_plans;
          if (logical_vl_segment) {
            attention_options.active_tokens = segment.input_tokens;
            attention_options.gemm_plans =
                &impl_->vl_logical_projections.router_gemm_plans();
            attention_options.logical_ab_gemm_plan =
                &impl_->vl_logical_projections.ab_plan();
            attention_options.logical_ab_weight =
                impl_->vl_logical_projections.ab_weight(layer_index);
            attention_options.logical_ab_output =
                impl_->vl_logical_projections.ab_output();
          }
          attention_options.bindings = &impl_->bindings;
          if (focused_tail_oracle) {
            std::ostringstream prefix;
            prefix << "layer-";
            prefix.width(3);
            prefix.fill('0');
            prefix << layer_index << '-';
            attention_options.tail_oracle_dir = request.layer_tail_oracle_dir;
            attention_options.tail_oracle_label_prefix = prefix.str();
          }
          if (focused_sequence_oracle) {
            std::ostringstream prefix;
            prefix << "layer-";
            prefix.width(3);
            prefix.fill('0');
            prefix << layer_index << '-';
            attention_options.sequence_oracle_dir =
                request.layer_sequence_oracle_dir;
            attention_options.sequence_oracle_label_prefix = prefix.str();
          }
          const NativeLinearPrefillOracleResult attention =
              probe_native_q8192_linear_prefill_layer0_oracle(
                  {}, impl_->weights, chunk_workspace, chunk_invocations,
                  impl_->executor, attention_options);
          metrics.prefill_aot_launches += attention.layer.aot_launches;
          metrics.prefill_dense_gemm_launches +=
              attention.layer.dense_gemm_launches;
          metrics.prefill_native_pointwise_launches +=
              attention.layer.native_pointwise_launches;
          if (segment.padded()) {
            const NativeLinearPrefillStateRepairMetrics repair =
                repair_native_linear_prefill_padded_state(
                    chunk_workspace, chunk_invocations, impl_->executor,
                    layer_index, segment.input_tokens,
                    impl_->padded_prefill_initial_conv_state);
            metrics.prefill_aot_launches += repair.aot_launches;
            metrics.prefill_native_pointwise_launches +=
                repair.native_pointwise_launches;
          }
          metrics.layer_tail_comparisons.insert(
              metrics.layer_tail_comparisons.end(),
              attention.boundary_comparisons.begin(),
              attention.boundary_comparisons.end());
        }
        if (timeline_enabled) {
          const std::string operation =
              "hipDeviceSynchronize after prefill attention layer " +
              std::to_string(layer_index) + " segment " +
              std::to_string(segment_index);
          check_hip(hipDeviceSynchronize(), operation.c_str());
          layer_attention_wall_ms += elapsed_ms(attention_started);
        }

        NativeMoePrefillOracleOptions moe_options;
        moe_options.layer_index = layer_index;
        moe_options.use_vl_shared_expert_semantics = vl_input != nullptr;
        moe_options.use_vl_router_semantics = vl_input != nullptr;
        moe_options.seed_post_attention = false;
        moe_options.run_routing_diagnostic = false;
        moe_options.collect_oracle_comparisons = false;
        moe_options.comparison_tokens = segment.input_tokens;
        moe_options.gemm_plans = chunk_gemm_plans;
        if (logical_vl_segment) {
          moe_options.active_tokens = segment.input_tokens;
          moe_options.gemm_plans =
              &impl_->vl_logical_projections.router_gemm_plans();
          moe_options.logical_router_gemm_plans =
              &impl_->vl_logical_projections.router_gemm_plans();
        }
        if (focused_sequence_oracle) {
          std::ostringstream label;
          label << "layer-";
          label.width(3);
          label.fill('0');
          label << layer_index << "-return-layer_body-output";
          moe_options.chain_output_oracle_dir =
              request.layer_sequence_oracle_dir;
          moe_options.chain_output_oracle_label = label.str();
          moe_options.chain_output_last_token_only = false;
        } else if (!request.layer_tail_oracle_dir.empty() &&
                   (request.layer_tail_oracle_index == 40 ||
                    request.layer_tail_oracle_index == layer_index)) {
          std::ostringstream label;
          label << "layer-";
          label.width(3);
          label.fill('0');
          label << layer_index << "-return-layer_body-output";
          moe_options.chain_output_oracle_dir = request.layer_tail_oracle_dir;
          moe_options.chain_output_oracle_label = label.str();
          moe_options.chain_output_last_token_only = true;
        }
        const auto moe_started = std::chrono::steady_clock::now();
        const NativeMoePrefillOracleResult moe =
            probe_native_q8192_moe_prefill_layer0_oracle(
                {}, impl_->weights, chunk_workspace, chunk_invocations,
                impl_->executor, moe_options);
        metrics.prefill_aot_launches += moe.layer.aot_launches;
        metrics.prefill_dense_gemm_launches += moe.layer.dense_gemm_launches;
        metrics.prefill_native_pointwise_launches +=
            moe.layer.native_pointwise_launches;
        metrics.layer_tail_comparisons.insert(
            metrics.layer_tail_comparisons.end(), moe.comparisons.begin(),
            moe.comparisons.end());
        if (moe.chain_output_comparison_provided) {
          metrics.layer_tail_comparisons.push_back(moe.chain_output_comparison);
        }
        if (composed_prefill) {
          check_hip(hipMemcpyAsync(
                        static_cast<unsigned char*>(impl_->chunked_hidden) +
                            hidden_offset * kHidden * sizeof(std::uint16_t),
                        native_prefill_layer_output_pointer(
                            chunk_workspace, chunk_invocations, layer_index),
                        segment.input_tokens * kHidden * sizeof(std::uint16_t),
                        hipMemcpyDeviceToDevice, nullptr),
                    "hipMemcpyAsync composed prefill layer output");
        }
        if (timeline_enabled) {
          const std::string operation =
              "hipDeviceSynchronize after prefill MoE layer " +
              std::to_string(layer_index) + " segment " +
              std::to_string(segment_index);
          check_hip(hipDeviceSynchronize(), operation.c_str());
          layer_moe_wall_ms += elapsed_ms(moe_started);
        }
      }
      if (timeline_enabled) {
        attention_wall_ms.push_back(layer_attention_wall_ms);
        moe_wall_ms.push_back(layer_moe_wall_ms);
      }
    }

    if (composed_prefill) {
      last_hidden =
          static_cast<const unsigned char*>(impl_->chunked_hidden) +
          (prompt_plan.cold_aot_tokens - 1) * kHidden * sizeof(std::uint16_t);
    } else {
      const NativePromptAotSegment& segment = prompt_plan.aot_segments.front();
      const NativeResidentPrefillOwner owner =
          impl_->prefill_owner(segment.bucket_tokens, vl_input == nullptr);
      last_hidden =
          static_cast<const unsigned char*>(native_prefill_layer_output_pointer(
              *owner.workspace, *owner.invocations, 39)) +
          (segment.input_tokens - 1) * kHidden * sizeof(std::uint16_t);
    }
  }
  if (request.prefill_linear_state_observer) {
    for (std::size_t layer_index = 0; layer_index < 40; ++layer_index) {
      if (layer_index % 4 == 3) continue;
      const std::string layer = std::to_string(layer_index);
      const NativeDecodeWorkspaceView* conv = impl_->decode_workspace.find(
          "linear_attention_initial_conv_states." + layer);
      const NativeDecodeWorkspaceView* recurrent =
          impl_->decode_workspace.find(
              "linear_attention_initial_ssm_states_vllm." + layer);
      if (conv == nullptr || conv->device_pointer == nullptr ||
          conv->payload_bytes != 8192ULL * 3ULL * sizeof(std::uint16_t) ||
          recurrent == nullptr || recurrent->device_pointer == nullptr ||
          recurrent->payload_bytes != 32ULL * 128ULL * 128ULL * sizeof(float)) {
        throw std::runtime_error(
            "native prefill state observer binding is incomplete");
      }
      request.prefill_linear_state_observer(
          layer_index, conv->device_pointer, conv->payload_bytes,
          recurrent->device_pointer, recurrent->payload_bytes);
    }
  }
  std::uint32_t first_token_id = 0;
  const void* prompt_terminal_hidden = last_hidden;
  const auto first_token_started = std::chrono::steady_clock::now();
  const std::uint8_t* first_allowed_token_mask =
      prepare_next_token_mask();
  const NativeLmHeadTop1Metrics first = run_native_lm_head_top1(
      last_hidden, impl_->weights, impl_->lm_head, impl_->decode_workspace,
      impl_->decode_invocations, impl_->executor, impl_->cu_count,
      first_allowed_token_mask);
  metrics.first_token_certified = first.certified;
  first_token_id = first.top1_token_id;
  if (timeline_enabled) {
    check_hip(hipDeviceSynchronize(),
              "hipDeviceSynchronize after first-token LM head");
    first_token_wall_ms = elapsed_ms(first_token_started);
  }
  if (!request.disable_prefix_cache && !exact_prefix_hit) {
    std::size_t capture_index = impl_->prefix_cache_entries;
    for (std::size_t index = 0; index < impl_->prefix_cache_entries; ++index) {
      if (!impl_->prefix_caches[index].valid()) {
        capture_index = index;
        break;
      }
    }
    if (capture_index == impl_->prefix_cache_entries) {
      capture_index = 0;
      for (std::size_t index = 1; index < impl_->prefix_cache_entries;
           ++index) {
        if (impl_->prefix_cache_use[index] <
            impl_->prefix_cache_use[capture_index]) {
          capture_index = index;
        }
      }
    }
    metrics.prefix_cache_transfer_bytes +=
        impl_->prefix_caches[capture_index].capture(
            request.input_token_ids, request.multimodal_cache_namespace,
            prompt_terminal_hidden);
    impl_->prefix_cache_use[capture_index] = ++impl_->prefix_cache_clock;
    impl_->active_kv_prefix_cache_index = capture_index;
  }
  metrics.output_token_ids.push_back(first_token_id);
  metrics.prefill_wall_ms = elapsed_ms(request_started);
  metrics.prefill_tokens_per_second =
      request.input_token_ids.size() * 1000.0 / metrics.prefill_wall_ms;
  if (prefix_extension_hit) {
    metrics.prefix_cache_suffix_aot_launches = metrics.prefill_aot_launches;
    metrics.prefix_cache_suffix_native_launches =
        metrics.prefill_dense_gemm_launches +
        metrics.prefill_native_pointwise_launches +
        metrics.prefill_ck_fmha_launches;
    metrics.prefix_cache_suffix_wall_ms = metrics.prefill_wall_ms;
  }
  if (request.token_callback && !request.token_callback(first_token_id, 0)) {
    metrics.client_cancelled = true;
  }
  if (exact_prefix_restore_pending) {
    const auto restore_started = std::chrono::steady_clock::now();
    metrics.prefix_cache_transfer_bytes =
        impl_->prefix_caches[matched_prefix_cache_index].restore();
    check_hip(hipStreamSynchronize(nullptr),
              "hipStreamSynchronize deferred exact-prefix restore");
    metrics.prefix_cache_restore_wall_ms = elapsed_ms(restore_started);
    impl_->active_kv_prefix_cache_index = matched_prefix_cache_index;
  }
  if (timeline_enabled) {
    double linear_attention_ms = 0.0;
    double full_attention_ms = 0.0;
    double total_moe_ms = 0.0;
    for (std::size_t layer_index = 0; layer_index < attention_wall_ms.size();
         ++layer_index) {
      if (layer_index % 4 == 3) {
        full_attention_ms += attention_wall_ms[layer_index];
      } else {
        linear_attention_ms += attention_wall_ms[layer_index];
      }
    }
    for (const double value : moe_wall_ms) total_moe_ms += value;
    std::cerr << std::setprecision(17)
              << "{\"event\":\"native_prefill_timeline\""
              << ",\"prompt_tokens\":" << metrics.prompt_tokens
              << ",\"embedding_wall_ms\":" << embedding_wall_ms
              << ",\"linear_attention_wall_ms\":" << linear_attention_ms
              << ",\"full_attention_wall_ms\":" << full_attention_ms
              << ",\"moe_wall_ms\":" << total_moe_ms
              << ",\"first_token_wall_ms\":" << first_token_wall_ms
              << ",\"prefill_wall_ms\":" << metrics.prefill_wall_ms
              << ",\"attention_layers\":[";
    for (std::size_t index = 0; index < attention_wall_ms.size(); ++index) {
      if (index != 0) std::cerr << ',';
      std::cerr << attention_wall_ms[index];
    }
    std::cerr << "],\"moe_layers\":[";
    for (std::size_t index = 0; index < moe_wall_ms.size(); ++index) {
      if (index != 0) std::cerr << ',';
      std::cerr << moe_wall_ms[index];
    }
    std::cerr << "]}\n";
  }
  metrics.prefix_cache_hits = impl_->prefix_cache_hits;
  metrics.prefix_cache_misses = impl_->prefix_cache_misses;
  metrics.stopped = contains(request.stop_token_ids, first_token_id);
  if (metrics.stopped) metrics.stop_token_id = first_token_id;

  metrics.all_decode_tokens_certified = true;
  while (!metrics.stopped && !metrics.client_cancelled &&
         metrics.output_token_ids.size() < request.max_new_tokens) {
    const std::size_t position =
        request.input_token_ids.size() + metrics.output_token_ids.size() - 1;
    std::size_t rotary_position = position;
    if (mrope_plan != nullptr) {
      const std::int64_t value = native_mrope_decode_position(
          request.input_token_ids.size(), mrope_plan->position_delta(),
          metrics.output_token_ids.size() - 1);
      if (value < 0 || value >= 262144) {
        throw std::runtime_error(
            "native resident M-RoPE decode position escaped validation");
      }
      rotary_position = static_cast<std::size_t>(value);
      ++metrics.mrope_decode_steps;
    }
    const NativeDecodePrepareMetrics prepared =
        mrope_plan != nullptr
            ? prepare_native_decode_step(
                  position, rotary_position, metrics.output_token_ids.back(),
                  impl_->weights, impl_->decode_invocations)
            : prepare_native_decode_step(
                  position, metrics.output_token_ids.back(), impl_->weights,
                  impl_->decode_invocations);
    const std::uint8_t* allowed_token_mask = prepare_next_token_mask();
    const bool observe_decode_layers =
        request.decode_layer_observer_output_index.has_value() &&
        *request.decode_layer_observer_output_index ==
            metrics.output_token_ids.size();
    const NativeDecodeLayerObserver* layer_observer =
        observe_decode_layers && request.decode_layer_observer
            ? &request.decode_layer_observer
            : nullptr;
    const NativeDecodeLinearLayer0Observer* linear_layer0_observer =
        observe_decode_layers && request.decode_linear_layer0_observer
            ? &request.decode_linear_layer0_observer
            : nullptr;
    const NativeDecodeLinearLayer0Observer* layer0_tail_observer =
        observe_decode_layers && request.decode_layer0_tail_observer
            ? &request.decode_layer0_tail_observer
            : nullptr;
    const NativeDecodeFullAttentionObserver* full_attention_observer =
        observe_decode_layers && request.decode_full_attention_observer
            ? &request.decode_full_attention_observer
            : nullptr;
    const NativeDecodeRunMetrics token = run_native_decode_token(
        position, position + 1, impl_->weights, impl_->lm_head,
        impl_->decode_workspace, impl_->decode_invocations, impl_->executor,
        impl_->attention_state, impl_->cu_count, allowed_token_mask, nullptr,
        layer_observer, request.decode_linear_observer_layer_index,
        linear_layer0_observer, layer0_tail_observer, full_attention_observer,
        mrope_plan != nullptr, impl_->decode_shared_gate_plan.get(),
        mrope_plan != nullptr ? &impl_->decode_cross_layer_norms : nullptr);
    ++metrics.decode_tokens_executed;
    metrics.decode_aot_launches += token.aot_launches;
    metrics.decode_native_launches +=
        token.native_attention_launches + token.native_projection_launches +
        token.native_pointwise_launches +
        token.native_lm_head_certificate_launches +
        prepared.native_kernel_launches;
    metrics.decode_wall_ms += token.synchronized_wall_ms;
    metrics.all_decode_tokens_certified =
        metrics.all_decode_tokens_certified && token.lm_head_certified;
    metrics.output_token_ids.push_back(token.top1_token_id);
    if (request.token_callback &&
        !request.token_callback(token.top1_token_id,
                                metrics.output_token_ids.size() - 1)) {
      metrics.client_cancelled = true;
    }
    metrics.stopped = contains(request.stop_token_ids, token.top1_token_id);
    if (metrics.stopped) metrics.stop_token_id = token.top1_token_id;
  }

  metrics.completion_tokens = metrics.output_token_ids.size();
  metrics.oracle_tensor_reads = metrics.layer_tail_comparisons.size();
  std::ostringstream token_payload;
  for (std::size_t index = 0; index < metrics.output_token_ids.size();
       ++index) {
    if (index != 0) token_payload << ',';
    token_payload << metrics.output_token_ids[index];
  }
  const std::string token_payload_value = token_payload.str();
  metrics.output_token_ids_sha256 =
      sha256_bytes(token_payload_value.data(), token_payload_value.size());
  if (metrics.vl_enabled) {
    const std::string canonical_token_payload =
        '[' + token_payload_value + ']';
    metrics.output_token_ids_canonical_sha256 =
        sha256_bytes(canonical_token_payload.data(),
                     canonical_token_payload.size());
  }
  if (metrics.decode_tokens_executed != 0 && metrics.decode_wall_ms > 0.0) {
    metrics.decode_tokens_per_second =
        metrics.decode_tokens_executed * 1000.0 / metrics.decode_wall_ms;
  }
  metrics.request_wall_ms = elapsed_ms(request_started);
  return metrics;
}

NativeLogitsComparison NativeResidentEngine::compare_current_logits(
    const std::filesystem::path& reference_path) const {
  if (!impl_->ready || impl_->request_count == 0 || reference_path.empty()) {
    throw std::invalid_argument(
        "native resident logits comparison requires a completed request and reference");
  }
  const NativeDecodeWorkspaceView* logits =
      impl_->decode_workspace.find("certified_lm_head_logits_output");
  if (logits == nullptr || logits->device_pointer == nullptr ||
      logits->payload_bytes < kVocabulary * sizeof(float)) {
    throw std::runtime_error(
        "native resident full-vocabulary logits buffer is missing");
  }
  return compare_native_logits_fp32(logits->device_pointer, kVocabulary,
                                    reference_path);
}

bool NativeResidentEngine::loaded() const { return impl_->ready; }
std::size_t NativeResidentEngine::request_count() const {
  return impl_->request_count;
}
const NativeResidentLoadMetrics& NativeResidentEngine::load_metrics() const {
  if (!impl_->ready) {
    throw std::runtime_error("native resident engine is not loaded");
  }
  return impl_->metrics;
}

}  // namespace aima
