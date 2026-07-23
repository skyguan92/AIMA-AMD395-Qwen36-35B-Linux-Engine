// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_resident_engine.h"

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
#include "aima/native_pointwise.h"
#include "aima/native_prefill_gemm_plans.h"
#include "aima/native_prefill_invocation.h"
#include "aima/native_prefill_workspace.h"
#include "aima/sha256.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sstream>
#include <utility>

namespace aima {
namespace {

constexpr std::size_t kHidden = 2048;
constexpr std::size_t kVocabulary = 248320;
constexpr std::size_t kLongContextChunkTokens = 8192;

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
                      int device, std::size_t prompt_tokens) {
    if (allocation_ != nullptr) {
      throw std::runtime_error("native exact-prefix cache is already built");
    }
    if (prompt_tokens == 0 || prompt_tokens > 262144) {
      throw std::invalid_argument(
          "native exact-prefix context is unsupported");
    }
    device_ = device;
    prompt_tokens_ = prompt_tokens;
    constexpr std::uint64_t kConvBytes =
        8192ULL * 3ULL * sizeof(std::uint16_t);
    constexpr std::uint64_t kRecurrentBytes =
        32ULL * 128ULL * 128ULL * sizeof(float);
    const std::uint64_t kKvBytes =
        prompt_tokens_ * 2ULL * 256ULL * sizeof(std::uint16_t);
    for (std::size_t layer = 0; layer < 40; ++layer) {
      if (layer % 4 == 3) {
        add_slice(attention_state.k_cache(layer), kKvBytes);
        add_slice(attention_state.v_cache(layer), kKvBytes);
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
        add_slice(conv->device_pointer, kConvBytes);
        add_slice(recurrent->device_pointer, kRecurrentBytes);
      }
    }
    terminal_offset_ = bytes_;
    bytes_ += kHidden * sizeof(std::uint16_t);
    check_hip(hipSetDevice(device_), "hipSetDevice exact-prefix cache");
    check_hip(hipMalloc(&allocation_, bytes_),
              "hipMalloc exact-prefix cache");
    return bytes_;
  }

  bool matches(const std::vector<std::uint32_t>& tokens) const {
    return valid_ && tokens == tokens_;
  }

  std::size_t matched_prefix_tokens(
      const std::vector<std::uint32_t>& tokens) const {
    if (!valid_ || tokens.size() < tokens_.size() ||
        !std::equal(tokens_.begin(), tokens_.end(), tokens.begin())) {
      return 0;
    }
    return tokens_.size();
  }

  std::uint64_t capture(const std::vector<std::uint32_t>& tokens,
                        const void* terminal_hidden,
                        void* stream_value = nullptr) {
    if (allocation_ == nullptr || terminal_hidden == nullptr ||
        tokens.size() != prompt_tokens_) {
      throw std::invalid_argument(
          "native exact-prefix capture is incomplete");
    }
    hipStream_t stream = static_cast<hipStream_t>(stream_value);
    auto* destination = static_cast<unsigned char*>(allocation_);
    for (const Slice& slice : slices_) {
      check_hip(hipMemcpyAsync(destination + slice.offset, slice.live,
                               slice.bytes, hipMemcpyDeviceToDevice, stream),
                "hipMemcpyAsync exact-prefix capture");
    }
    check_hip(hipMemcpyAsync(destination + terminal_offset_, terminal_hidden,
                             kHidden * sizeof(std::uint16_t),
                             hipMemcpyDeviceToDevice, stream),
              "hipMemcpyAsync exact-prefix terminal capture");
    check_hip(hipStreamSynchronize(stream),
              "hipStreamSynchronize exact-prefix capture");
    tokens_ = tokens;
    valid_ = true;
    return bytes_;
  }

  std::uint64_t restore(void* stream_value = nullptr) const {
    if (!valid_ || allocation_ == nullptr) {
      throw std::runtime_error("native exact-prefix cache is empty");
    }
    hipStream_t stream = static_cast<hipStream_t>(stream_value);
    const auto* source = static_cast<const unsigned char*>(allocation_);
    for (const Slice& slice : slices_) {
      check_hip(hipMemcpyAsync(slice.live, source + slice.offset,
                               slice.bytes, hipMemcpyDeviceToDevice, stream),
                "hipMemcpyAsync exact-prefix restore");
    }
    return bytes_;
  }

  const void* terminal_hidden() const {
    if (!valid_) throw std::runtime_error("native exact-prefix cache is empty");
    return static_cast<const unsigned char*>(allocation_) + terminal_offset_;
  }

 private:
  struct Slice {
    void* live = nullptr;
    std::uint64_t bytes = 0;
    std::uint64_t offset = 0;
  };
  void add_slice(void* live, std::uint64_t bytes) {
    if (live == nullptr || bytes == 0) {
      throw std::invalid_argument("native exact-prefix slice is invalid");
    }
    slices_.push_back({live, bytes, bytes_});
    bytes_ += bytes;
  }

  int device_ = 0;
  void* allocation_ = nullptr;
  std::uint64_t bytes_ = 0;
  std::uint64_t terminal_offset_ = 0;
  std::size_t prompt_tokens_ = 0;
  std::vector<Slice> slices_;
  std::vector<std::uint32_t> tokens_;
  bool valid_ = false;
};

}  // namespace

struct NativeResidentEngine::Impl {
  NativeWeightStore weights;
  NativeDerivedWeightStore derived;
  NativeLmHeadStore lm_head;
  NativeDecodeBindings bindings;
  NativePrefillWorkspace prefill_workspace;
  NativePrefillInvocations prefill_invocations;
  std::unique_ptr<NativeQ8192PrefillGemmPlans> prefill_gemm_plans;
  NativePrefillWorkspace tail_prefill_workspace;
  NativePrefillInvocations tail_prefill_invocations;
  std::unique_ptr<NativeQ8192PrefillGemmPlans> tail_prefill_gemm_plans;
  NativeDecodeWorkspace decode_workspace;
  NativeDecodeInvocations decode_invocations;
  NativeDecodeExecutor executor;
  NativeQ8192CkProvider ck_provider;
  NativeQ8192CkProvider secondary_fmha_provider;
  std::array<bool, 40> secondary_fmha_layers{};
  NativeFullAttentionState attention_state;
  NativeExactPrefixCache prefix_cache;
  NativeResidentLoadMetrics metrics;
  int device = 0;
  int cu_count = 0;
  std::size_t prompt_tokens = 0;
  std::size_t prefill_tokens = 0;
  std::size_t tail_prefill_tokens = 0;
  void* chunked_hidden = nullptr;
  std::uint64_t chunked_hidden_bytes = 0;
  std::size_t prefill_start_sequence = 0;
  std::size_t tail_prefill_start_sequence = 0;
  std::size_t request_count = 0;
  std::size_t prefix_cache_hits = 0;
  std::size_t prefix_cache_misses = 0;
  bool ready = false;

  ~Impl() {
    if (chunked_hidden != nullptr) {
      (void)hipSetDevice(device);
      (void)hipFree(chunked_hidden);
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
  impl_->device = options.weights.device;
  impl_->prompt_tokens = options.prompt_tokens;
  impl_->prefill_tokens = impl_->prompt_tokens > 32768
                              ? kLongContextChunkTokens
                              : impl_->prompt_tokens;
  impl_->tail_prefill_tokens = impl_->prompt_tokens > 32768
                                   ? impl_->prompt_tokens %
                                         kLongContextChunkTokens
                                   : 0;
  if (impl_->prefill_tokens != impl_->prompt_tokens) {
    impl_->chunked_hidden_bytes =
        impl_->prompt_tokens * kHidden * sizeof(std::uint16_t);
    check_hip(hipSetDevice(impl_->device),
              "hipSetDevice chunked prefill hidden store");
    check_hip(hipMalloc(&impl_->chunked_hidden,
                        impl_->chunked_hidden_bytes),
              "hipMalloc chunked prefill hidden store");
  }
  impl_->prefill_gemm_plans =
      std::make_unique<NativeQ8192PrefillGemmPlans>(impl_->prefill_tokens);
  if (impl_->tail_prefill_tokens != 0) {
    impl_->tail_prefill_gemm_plans =
        std::make_unique<NativeQ8192PrefillGemmPlans>(
            impl_->tail_prefill_tokens);
  }
  const NativeWeightLoadMetrics weight_metrics =
      impl_->weights.load(options.weights);
  const NativeDerivedWeightMetrics derived_metrics =
      impl_->derived.build(impl_->weights, impl_->device);
  const NativeLmHeadMetrics lm_head_metrics =
      impl_->lm_head.build(impl_->weights, impl_->device);
  const NativeDecodeBindingMetrics binding_metrics =
      impl_->bindings.build(impl_->weights, impl_->derived, impl_->lm_head);
  const NativePrefillWorkspaceMetrics prefill_workspace_metrics =
      impl_->prefill_workspace.build(impl_->device, impl_->prefill_tokens);
  const NativePrefillInvocationMetrics prefill_invocation_metrics =
      impl_->prefill_invocations.build(impl_->bindings,
                                       impl_->prefill_workspace,
                                       impl_->prefill_tokens);
  NativePrefillWorkspaceMetrics tail_prefill_workspace_metrics;
  NativePrefillInvocationMetrics tail_prefill_invocation_metrics;
  if (impl_->tail_prefill_tokens != 0) {
    tail_prefill_workspace_metrics = impl_->tail_prefill_workspace.build(
        impl_->device, impl_->tail_prefill_tokens);
    tail_prefill_invocation_metrics = impl_->tail_prefill_invocations.build(
        impl_->bindings, impl_->tail_prefill_workspace,
        impl_->tail_prefill_tokens);
  }
  const NativeDecodeWorkspaceMetrics decode_workspace_metrics =
      impl_->decode_workspace.build(impl_->device);
  const NativeDecodeInvocationMetrics decode_invocation_metrics =
      impl_->decode_invocations.build(impl_->bindings,
                                      impl_->decode_workspace);
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
  const NativeFullAttentionStateMetrics attention_metrics =
      impl_->attention_state.build(options.cache_capacity, impl_->device);
  const std::uint64_t prefix_cache_bytes = impl_->prefix_cache.build(
      impl_->decode_workspace, impl_->attention_state, impl_->device,
      impl_->prompt_tokens);

  const auto plan_started = std::chrono::steady_clock::now();
  impl_->prefill_gemm_plans->prepare_all();
  if (impl_->tail_prefill_gemm_plans != nullptr) {
    impl_->tail_prefill_gemm_plans->prepare_all();
  }
  const double plan_wall_ms = elapsed_ms(plan_started);

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

  impl_->metrics.device_name = weight_metrics.device_name;
  impl_->metrics.gpu_arch = weight_metrics.gpu_arch;
  impl_->metrics.model_payload_bytes = weight_metrics.payload_bytes;
  impl_->metrics.model_tensor_count = weight_metrics.tensor_count;
  impl_->metrics.model_shard_count = weight_metrics.shard_count;
  impl_->metrics.decode_weight_bindings = binding_metrics.unique_bindings;
  impl_->metrics.prefill_prepared_launches =
      prefill_invocation_metrics.launch_count +
      tail_prefill_invocation_metrics.launch_count;
  impl_->metrics.decode_prepared_launches =
      decode_invocation_metrics.launch_count;
  impl_->metrics.aot_loaded_modules = executor_metrics.loaded_modules;
  impl_->metrics.prefill_gemm_plans =
      impl_->prefill_gemm_plans->built_plan_count() +
      (impl_->tail_prefill_gemm_plans == nullptr
           ? 0
           : impl_->tail_prefill_gemm_plans->built_plan_count());
  impl_->metrics.prefill_workspace_bytes =
      prefill_workspace_metrics.allocation_bytes +
      tail_prefill_workspace_metrics.allocation_bytes +
      impl_->chunked_hidden_bytes;
  impl_->metrics.decode_workspace_bytes =
      decode_workspace_metrics.allocation_bytes;
  impl_->metrics.attention_state_bytes = attention_metrics.allocation_bytes;
  impl_->metrics.exact_prefix_cache_bytes = prefix_cache_bytes;
  impl_->metrics.cache_capacity = options.cache_capacity;
  impl_->metrics.prompt_tokens = impl_->prompt_tokens;
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
  impl_->metrics.prefill_gemm_plan_build_wall_ms = plan_wall_ms;
  impl_->metrics.command_to_ready_wall_ms = elapsed_ms(started);
  impl_->ready = true;
  return impl_->metrics;
}

NativeResidentRequestMetrics NativeResidentEngine::run(
    const NativeResidentRequestOptions& request) {
  if (!impl_->ready || request.input_token_ids.empty() ||
      request.max_new_tokens == 0 ||
      request.input_token_ids.size() + request.max_new_tokens - 1 >
          impl_->attention_state.cache_capacity()) {
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
  const std::size_t matched_prefix_tokens =
      request.disable_prefix_cache
          ? 0
          : impl_->prefix_cache.matched_prefix_tokens(
                request.input_token_ids);
  const bool prefix_hit = matched_prefix_tokens != 0;
  const bool exact_prefix_hit =
      prefix_hit && matched_prefix_tokens == request.input_token_ids.size();
  const bool prefix_extension_hit =
      prefix_hit && matched_prefix_tokens < request.input_token_ids.size();
  if (request.input_token_ids.size() != impl_->prompt_tokens &&
      !prefix_extension_hit) {
    throw std::invalid_argument(
        "native resident cold prefill requires the static context; a longer "
        "prompt must extend the cached static prefix");
  }
  const bool chunked_prefill =
      impl_->prefill_tokens != impl_->prompt_tokens;
  if (chunked_prefill &&
      (!request.layer_tail_oracle_dir.empty() ||
       !request.layer_sequence_oracle_dir.empty())) {
    throw std::invalid_argument(
        "native chunked prefill does not admit layer-oracle diagnostics");
  }

  NativeResidentRequestMetrics metrics;
  metrics.request_index = ++impl_->request_count;
  metrics.prompt_tokens = request.input_token_ids.size();
  metrics.model_loads = 1;
  metrics.oracle_tensor_reads = 0;
  metrics.state_orientation_resets =
      impl_->decode_invocations.reset_linear_decode_state_buffers();
  std::array<bool, 40> request_secondary_fmha_layers =
      impl_->secondary_fmha_layers;
  if (request.secondary_fmha_layers_override_provided) {
    request_secondary_fmha_layers.fill(false);
    for (std::size_t layer : request.secondary_fmha_layers_override) {
      if (layer >= 40 || layer % 4 != 3 ||
          request_secondary_fmha_layers[layer]) {
        throw std::invalid_argument(
            "native request secondary FMHA layers must be unique full-attention layers");
      }
      request_secondary_fmha_layers[layer] = true;
    }
  }
  const auto request_started = std::chrono::steady_clock::now();
  const bool timeline_enabled = prefill_timeline_enabled() && !prefix_hit;
  std::vector<double> attention_wall_ms;
  std::vector<double> moe_wall_ms;
  double embedding_wall_ms = 0.0;
  double first_token_wall_ms = 0.0;
  void* terminal = chunked_prefill
                       ? impl_->chunked_hidden
                       : native_prefill_terminal_hidden_pointer(
                             impl_->prefill_workspace,
                             impl_->prefill_invocations);
  const void* last_hidden = nullptr;
  if (prefix_hit) {
    metrics.prefix_cache_lookup = exact_prefix_hit ? "exact" : "prefix";
    metrics.prefix_cache_matched_tokens = matched_prefix_tokens;
    metrics.prefix_cache_suffix_tokens =
        request.input_token_ids.size() - matched_prefix_tokens;
    ++impl_->prefix_cache_hits;
    metrics.prefix_cache_transfer_bytes = impl_->prefix_cache.restore();
    if (exact_prefix_hit) {
      last_hidden = impl_->prefix_cache.terminal_hidden();
    }
  } else {
    metrics.prefix_cache_lookup =
        request.disable_prefix_cache ? "disabled" : "miss";
    metrics.prefix_cache_matched_tokens = 0;
    metrics.prefix_cache_suffix_tokens = impl_->prompt_tokens;
    if (!request.disable_prefix_cache) ++impl_->prefix_cache_misses;

    const NativeTensorView* embedding =
        impl_->weights.find("model.language_model.embed_tokens.weight");
    const NativePrefillWorkspaceView* token_ids =
        impl_->prefill_workspace.find("native.prompt_token_ids");
    if (embedding == nullptr || embedding->device_pointer == nullptr ||
        token_ids == nullptr || token_ids->device_pointer == nullptr) {
      throw std::runtime_error(
          "native resident prompt embedding owners are missing");
    }
    if (timeline_enabled) {
      check_hip(hipDeviceSynchronize(),
                "hipDeviceSynchronize before prefill timeline");
    }
    const auto embedding_started = std::chrono::steady_clock::now();
    if (chunked_prefill) {
      for (std::size_t offset = 0; offset < impl_->prompt_tokens;
           offset += impl_->prefill_tokens) {
        const std::size_t chunk_tokens = std::min(
            impl_->prefill_tokens, impl_->prompt_tokens - offset);
        const bool tail_chunk = chunk_tokens != impl_->prefill_tokens;
        const NativePrefillWorkspace& chunk_workspace =
            tail_chunk ? impl_->tail_prefill_workspace
                       : impl_->prefill_workspace;
        const NativePrefillWorkspaceView* chunk_token_ids =
            chunk_workspace.find("native.prompt_token_ids");
        if (chunk_token_ids == nullptr ||
            chunk_token_ids->device_pointer == nullptr) {
          throw std::runtime_error(
              "native resident chunk token owner is missing");
        }
        launch_prompt_embeddings(
            embedding->device_pointer,
            request.input_token_ids.data() + offset,
            chunk_token_ids->device_pointer,
            static_cast<unsigned char*>(impl_->chunked_hidden) +
                offset * kHidden * sizeof(std::uint16_t),
            chunk_tokens);
        ++metrics.prefill_native_pointwise_launches;
      }
    } else {
      launch_prompt_embeddings(
          embedding->device_pointer, request.input_token_ids.data(),
          token_ids->device_pointer,
          impl_->prefill_invocations.tensor_pointer(
              impl_->prefill_start_sequence, "x"),
          impl_->prompt_tokens);
      ++metrics.prefill_native_pointwise_launches;
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
      const std::size_t chunk_count = chunked_prefill
                                          ? (impl_->prompt_tokens +
                                             impl_->prefill_tokens - 1) /
                                                impl_->prefill_tokens
                                          : 1;
      for (std::size_t chunk_index = 0; chunk_index < chunk_count;
           ++chunk_index) {
        const std::size_t chunk_offset =
            chunk_index * impl_->prefill_tokens;
        const std::size_t chunk_tokens = chunked_prefill
                                             ? std::min(
                                                   impl_->prefill_tokens,
                                                   impl_->prompt_tokens -
                                                       chunk_offset)
                                             : impl_->prefill_tokens;
        const bool tail_chunk =
            chunk_tokens != impl_->prefill_tokens;
        NativePrefillWorkspace& chunk_workspace =
            tail_chunk ? impl_->tail_prefill_workspace
                       : impl_->prefill_workspace;
        NativePrefillInvocations& chunk_invocations =
            tail_chunk ? impl_->tail_prefill_invocations
                       : impl_->prefill_invocations;
        NativeQ8192PrefillGemmPlans* chunk_gemm_plans =
            tail_chunk ? impl_->tail_prefill_gemm_plans.get()
                       : impl_->prefill_gemm_plans.get();
        if (chunked_prefill) {
          check_hip(hipMemcpyAsync(
                        native_prefill_layer_input_pointer(
                            chunk_workspace, chunk_invocations,
                            layer_index),
                        static_cast<const unsigned char*>(
                            impl_->chunked_hidden) +
                            chunk_offset * kHidden * sizeof(std::uint16_t),
                        chunk_tokens * kHidden *
                            sizeof(std::uint16_t),
                        hipMemcpyDeviceToDevice, nullptr),
                    "hipMemcpyAsync chunked prefill layer input");
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
        attention_options.seed_layer_input = false;
        attention_options.prepare_rotary_table = true;
        attention_options.collect_oracle_comparisons = false;
        attention_options.decode_attention_state = &impl_->attention_state;
        attention_options.gemm_plans = chunk_gemm_plans;
        attention_options.bindings = &impl_->bindings;
        attention_options.cache_position_start = chunk_offset;
        if (focused_tail_oracle) {
          std::ostringstream prefix;
          prefix << "layer-";
          prefix.width(3);
          prefix.fill('0');
          prefix << layer_index << '-';
          attention_options.tail_oracle_dir =
              request.layer_tail_oracle_dir;
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
        NativeQ8192CkProvider& attention_provider =
            request_secondary_fmha_layers[layer_index]
                ? impl_->secondary_fmha_provider
                : impl_->ck_provider;
        const NativeFullPrefillOracleResult attention =
            probe_native_q8192_full_prefill_oracle(
                {}, impl_->weights, chunk_workspace,
                chunk_invocations, impl_->executor,
                attention_provider, attention_options);
        metrics.prefill_aot_launches += attention.layer.aot_launches;
        metrics.prefill_dense_gemm_launches +=
            attention.layer.dense_gemm_launches;
        metrics.prefill_native_pointwise_launches +=
            attention.layer.native_pointwise_launches;
        metrics.prefill_ck_fmha_launches +=
            attention.layer.native_ck_fmha_launches;
        metrics.layer_tail_comparisons.insert(
            metrics.layer_tail_comparisons.end(),
            attention.boundary_comparisons.begin(),
            attention.boundary_comparisons.end());
      } else {
        NativeLinearPrefillOracleOptions attention_options;
        attention_options.layer_index = layer_index;
        attention_options.seed_layer_input = false;
        attention_options.run_output_projection_diagnostic = false;
        attention_options.collect_oracle_comparisons = false;
        attention_options.decode_state_workspace = &impl_->decode_workspace;
        attention_options.has_initial_state = chunk_index != 0;
        attention_options.gemm_plans = chunk_gemm_plans;
        attention_options.bindings = &impl_->bindings;
        if (focused_tail_oracle) {
          std::ostringstream prefix;
          prefix << "layer-";
          prefix.width(3);
          prefix.fill('0');
          prefix << layer_index << '-';
          attention_options.tail_oracle_dir =
              request.layer_tail_oracle_dir;
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
                {}, impl_->weights, chunk_workspace,
                chunk_invocations, impl_->executor,
                attention_options);
        metrics.prefill_aot_launches += attention.layer.aot_launches;
        metrics.prefill_dense_gemm_launches +=
            attention.layer.dense_gemm_launches;
        metrics.prefill_native_pointwise_launches +=
            attention.layer.native_pointwise_launches;
        metrics.layer_tail_comparisons.insert(
            metrics.layer_tail_comparisons.end(),
            attention.boundary_comparisons.begin(),
            attention.boundary_comparisons.end());
      }
      if (timeline_enabled) {
        check_hip(hipDeviceSynchronize(),
                  "hipDeviceSynchronize after prefill attention");
        layer_attention_wall_ms += elapsed_ms(attention_started);
      }

      NativeMoePrefillOracleOptions moe_options;
      moe_options.layer_index = layer_index;
      moe_options.seed_post_attention = false;
      moe_options.run_routing_diagnostic = false;
      moe_options.collect_oracle_comparisons = false;
      moe_options.gemm_plans = chunk_gemm_plans;
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
        moe_options.chain_output_oracle_dir =
            request.layer_tail_oracle_dir;
        moe_options.chain_output_oracle_label = label.str();
        moe_options.chain_output_last_token_only = true;
      }
      const auto moe_started = std::chrono::steady_clock::now();
      const NativeMoePrefillOracleResult moe =
          probe_native_q8192_moe_prefill_layer0_oracle(
              {}, impl_->weights, chunk_workspace,
              chunk_invocations, impl_->executor, moe_options);
      metrics.prefill_aot_launches += moe.layer.aot_launches;
      metrics.prefill_dense_gemm_launches += moe.layer.dense_gemm_launches;
      metrics.prefill_native_pointwise_launches +=
          moe.layer.native_pointwise_launches;
      metrics.layer_tail_comparisons.insert(
          metrics.layer_tail_comparisons.end(),
          moe.comparisons.begin(), moe.comparisons.end());
      if (moe.chain_output_comparison_provided) {
        metrics.layer_tail_comparisons.push_back(
            moe.chain_output_comparison);
      }
      if (chunked_prefill) {
        check_hip(hipMemcpyAsync(
                      static_cast<unsigned char*>(impl_->chunked_hidden) +
                          chunk_offset * kHidden * sizeof(std::uint16_t),
                      native_prefill_layer_output_pointer(
                          chunk_workspace, chunk_invocations,
                          layer_index),
                      chunk_tokens * kHidden *
                          sizeof(std::uint16_t),
                      hipMemcpyDeviceToDevice, nullptr),
                  "hipMemcpyAsync chunked prefill layer output");
      }
      if (timeline_enabled) {
        check_hip(hipDeviceSynchronize(),
                  "hipDeviceSynchronize after prefill MoE");
        layer_moe_wall_ms += elapsed_ms(moe_started);
      }
      }
      if (timeline_enabled) {
        attention_wall_ms.push_back(layer_attention_wall_ms);
        moe_wall_ms.push_back(layer_moe_wall_ms);
      }
    }

    const auto* terminal_bytes = static_cast<const unsigned char*>(terminal);
    last_hidden = terminal_bytes +
                  (impl_->prompt_tokens - 1) * kHidden *
                      sizeof(std::uint16_t);
  }
  std::uint32_t first_token_id = 0;
  if (prefix_extension_hit) {
    NativeDecodeRunMetrics suffix_last;
    const auto suffix_started = std::chrono::steady_clock::now();
    for (std::size_t token_index = matched_prefix_tokens;
         token_index < request.input_token_ids.size(); ++token_index) {
      (void)prepare_native_decode_step(
          token_index, request.input_token_ids[token_index], impl_->weights,
          impl_->decode_invocations);
      suffix_last = run_native_decode_token(
          token_index, token_index + 1, impl_->weights, impl_->lm_head,
          impl_->decode_workspace, impl_->decode_invocations,
          impl_->executor, impl_->attention_state, impl_->cu_count);
      ++metrics.prefix_cache_suffix_decode_tokens;
      metrics.prefix_cache_suffix_aot_launches += suffix_last.aot_launches;
      metrics.prefix_cache_suffix_native_launches +=
          suffix_last.native_attention_launches +
          suffix_last.native_projection_launches +
          suffix_last.native_pointwise_launches +
          suffix_last.native_lm_head_certificate_launches + 2;
    }
    metrics.prefix_cache_suffix_wall_ms = elapsed_ms(suffix_started);
    metrics.first_token_certified = suffix_last.lm_head_certified;
    first_token_id = suffix_last.top1_token_id;
  } else {
    const auto first_token_started = std::chrono::steady_clock::now();
    const NativeLmHeadTop1Metrics first = run_native_lm_head_top1(
        last_hidden, impl_->weights, impl_->lm_head,
        impl_->decode_workspace, impl_->decode_invocations,
        impl_->executor, impl_->cu_count);
    metrics.first_token_certified = first.certified;
    first_token_id = first.top1_token_id;
    if (timeline_enabled) {
      check_hip(hipDeviceSynchronize(),
                "hipDeviceSynchronize after first-token LM head");
      first_token_wall_ms = elapsed_ms(first_token_started);
    }
  }
  metrics.output_token_ids.push_back(first_token_id);
  metrics.prefill_wall_ms = elapsed_ms(request_started);
  metrics.prefill_tokens_per_second =
      request.input_token_ids.size() * 1000.0 / metrics.prefill_wall_ms;
  if (timeline_enabled) {
    double linear_attention_ms = 0.0;
    double full_attention_ms = 0.0;
    double total_moe_ms = 0.0;
    for (std::size_t layer_index = 0;
         layer_index < attention_wall_ms.size(); ++layer_index) {
      if (layer_index % 4 == 3) {
        full_attention_ms += attention_wall_ms[layer_index];
      } else {
        linear_attention_ms += attention_wall_ms[layer_index];
      }
    }
    for (const double value : moe_wall_ms) total_moe_ms += value;
    std::cerr << std::setprecision(17)
              << "{\"event\":\"native_prefill_timeline\""
              << ",\"prompt_tokens\":" << impl_->prompt_tokens
              << ",\"embedding_wall_ms\":" << embedding_wall_ms
              << ",\"linear_attention_wall_ms\":"
              << linear_attention_ms
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
  if (!prefix_hit && !request.disable_prefix_cache) {
    metrics.prefix_cache_transfer_bytes = impl_->prefix_cache.capture(
        request.input_token_ids, last_hidden);
  }
  metrics.prefix_cache_hits = impl_->prefix_cache_hits;
  metrics.prefix_cache_misses = impl_->prefix_cache_misses;
  metrics.stopped = contains(request.stop_token_ids, first_token_id);
  if (metrics.stopped) metrics.stop_token_id = first_token_id;

  metrics.all_decode_tokens_certified = true;
  while (!metrics.stopped &&
         metrics.output_token_ids.size() < request.max_new_tokens) {
    const std::size_t position =
        request.input_token_ids.size() + metrics.output_token_ids.size() - 1;
    (void)prepare_native_decode_step(
        position, metrics.output_token_ids.back(), impl_->weights,
        impl_->decode_invocations);
    const NativeDecodeRunMetrics token = run_native_decode_token(
        position, position + 1, impl_->weights, impl_->lm_head,
        impl_->decode_workspace, impl_->decode_invocations,
        impl_->executor, impl_->attention_state, impl_->cu_count);
    ++metrics.decode_tokens_executed;
    metrics.decode_aot_launches += token.aot_launches;
    metrics.decode_native_launches +=
        token.native_attention_launches + token.native_projection_launches +
        token.native_pointwise_launches +
        token.native_lm_head_certificate_launches + 2;
    metrics.decode_wall_ms += token.synchronized_wall_ms;
    metrics.all_decode_tokens_certified =
        metrics.all_decode_tokens_certified && token.lm_head_certified;
    metrics.output_token_ids.push_back(token.top1_token_id);
    metrics.stopped = contains(request.stop_token_ids, token.top1_token_id);
    if (metrics.stopped) metrics.stop_token_id = token.top1_token_id;
  }

  metrics.completion_tokens = metrics.output_token_ids.size();
  metrics.oracle_tensor_reads = metrics.layer_tail_comparisons.size();
  std::ostringstream token_payload;
  for (std::size_t index = 0; index < metrics.output_token_ids.size(); ++index) {
    if (index != 0) token_payload << ',';
    token_payload << metrics.output_token_ids[index];
  }
  const std::string token_payload_value = token_payload.str();
  metrics.output_token_ids_sha256 = sha256_bytes(
      token_payload_value.data(), token_payload_value.size());
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
