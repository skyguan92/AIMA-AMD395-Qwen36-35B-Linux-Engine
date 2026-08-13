// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_full_prefill.h"

#include "aima/bf16_gemm.h"
#include "aima/native_decode_bindings.h"
#include "aima/native_pointwise.h"
#include "aima/native_prefill_gemm_plans.h"

#include <dlfcn.h>
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

namespace aima {
namespace {

constexpr std::size_t kHidden = 2048;
constexpr std::size_t kQueryHeads = 16;
constexpr std::size_t kKvHeads = 2;
constexpr std::size_t kHeadDimension = 256;
constexpr std::size_t kQueryGateDimension = 8192;
constexpr std::size_t kQueryDimension = 4096;
constexpr std::size_t kKvDimension = 512;
constexpr std::size_t kRotaryPairs = 32;

void check_hip(hipError_t status, const char* operation) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorName(status) + " (" +
                             hipGetErrorString(status) + ")");
  }
}

const NativeTensorView& require_weight(const NativeWeightStore& weights,
                                       const std::string& name,
                                       std::uint64_t bytes) {
  const NativeTensorView* view = weights.find(name);
  if (view == nullptr || view->device_pointer == nullptr ||
      view->payload_bytes != bytes) {
    throw std::runtime_error(
        "native full prefill weight mismatch: " + name);
  }
  return *view;
}

const NativePrefillWorkspaceView& require_workspace(
    const NativePrefillWorkspace& workspace, const char* name,
    std::uint64_t bytes) {
  const NativePrefillWorkspaceView* view = workspace.find(name);
  if (view == nullptr || view->device_pointer == nullptr ||
      view->payload_bytes < bytes) {
    throw std::runtime_error(
        "native full prefill workspace mismatch: " + std::string(name));
  }
  return *view;
}

void* require_invocation_tensor(const NativePrefillInvocations& invocations,
                                std::size_t sequence, const char* name,
                                std::uint64_t bytes) {
  void* pointer = invocations.tensor_pointer(sequence, name);
  if (pointer == nullptr ||
      invocations.tensor_storage_bytes(sequence, name) < bytes) {
    throw std::runtime_error(
        "native full prefill invocation scratch mismatch: " +
        std::string(name));
  }
  return pointer;
}

std::size_t find_direct_scratch_base(
    const std::vector<PreparedDecodeInvocation>& launches) {
  for (std::size_t sequence = 1; sequence < launches.size(); ++sequence) {
    const auto* launch = launches[sequence].launch;
    if (launch != nullptr && launch->layer_index == 0 &&
        std::string(launch->symbol) ==
            "triton_prefill_direct_conv_kernel") {
      return sequence - 1;
    }
  }
  throw std::runtime_error(
      "native direct prefill scratch schedule is absent");
}

void require_symbol(const std::vector<PreparedDecodeInvocation>& launches,
                    std::size_t sequence, const char* symbol) {
  if (sequence >= launches.size() || launches[sequence].launch == nullptr ||
      std::string(launches[sequence].launch->symbol) != symbol) {
    throw std::runtime_error(
        "native full prefill schedule symbol mismatch at sequence " +
        std::to_string(sequence));
  }
}

std::size_t find_full_layer_base(
    const std::vector<PreparedDecodeInvocation>& launches,
    std::size_t layer_index) {
  if (layer_index >= 40 || layer_index % 4 != 3) {
    throw std::invalid_argument(
        "native full prefill requires a full-attention layer index");
  }
  for (std::size_t sequence = 0; sequence < launches.size(); ++sequence) {
    const auto* launch = launches[sequence].launch;
    if (launch == nullptr ||
        launch->layer_index != static_cast<std::int16_t>(layer_index)) {
      continue;
    }
    if (sequence + 4 > launches.size()) break;
    for (std::size_t offset = 0; offset < 4; ++offset) {
      if (launches[sequence + offset].launch == nullptr ||
          launches[sequence + offset].launch->layer_index !=
              static_cast<std::int16_t>(layer_index)) {
        throw std::runtime_error(
            "native full prefill layer schedule is not contiguous");
      }
    }
    return sequence;
  }
  throw std::runtime_error(
      "native full prefill layer is absent from schedule");
}

bool gate(const NativeOracleComparison& value) {
  return value.finite_elements == value.elements &&
         value.relative_l2_error <= 0.005 &&
         value.cosine_similarity >= 0.999;
}

}  // namespace

NativeQ8192CkProvider::~NativeQ8192CkProvider() { reset(); }

NativeQ8192CkProviderMetrics NativeQ8192CkProvider::load(
    const std::filesystem::path& library_path,
    std::size_t context_tokens) {
  if (loaded()) {
    throw std::runtime_error("native q8192 CK provider is already loaded");
  }
  if (context_tokens == 0 || context_tokens > 262144) {
    throw std::invalid_argument("native CK provider context is unsupported");
  }
  const std::filesystem::path path =
      std::filesystem::absolute(library_path);
  if (!std::filesystem::is_regular_file(path)) {
    throw std::runtime_error(
        "native q8192 CK provider library is missing: " + path.string());
  }
  handle_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (handle_ == nullptr) {
    const char* error = dlerror();
    throw std::runtime_error(
        "dlopen native q8192 CK provider failed: " +
        std::string(error == nullptr ? "unknown" : error));
  }
  try {
    prepare_ = reinterpret_cast<PrepareFn>(
        dlsym(handle_, "qrt_ck_fmha_prepare"));
    launch_ = reinterpret_cast<LaunchFn>(
        dlsym(handle_, "qrt_ck_fmha_bf16_launch"));
    rectangular_launch_ = reinterpret_cast<RectangularLaunchFn>(
        dlsym(handle_, "qrt_ck_fmha_bf16_launch_ex"));
    release_ = reinterpret_cast<ReleaseFn>(
        dlsym(handle_, "qrt_ck_fmha_release"));
    int status = static_cast<int>(hipErrorInvalidValue);
    if (prepare_ != nullptr && launch_ != nullptr && release_ != nullptr) {
      status = prepare_(static_cast<unsigned int>(context_tokens));
      metrics_.generic_context_abi = true;
      metrics_.rectangular_context_abi = rectangular_launch_ != nullptr;
      // Older copies of the bundled CK provider used prepare() only as an
      // admitted-context predicate even though the generic launch is fully
      // stateless and owns no context-sized allocation.  The engine's exact
      // embedded schedule is the authoritative admission gate, so retain
      // compatibility with those providers while the source provider accepts
      // the complete native context range.
      if (status == static_cast<int>(hipErrorInvalidValue) &&
          path.filename() == "libaima-fmha-ck.so") {
        status = static_cast<int>(hipSuccess);
      }
    } else {
      prepare_ = nullptr;
      launch_ = nullptr;
      release_ = nullptr;
      if (context_tokens != 8192) {
        throw std::runtime_error(
            "native CK provider lacks the generic context ABI");
      }
      legacy_prepare_ = reinterpret_cast<LegacyPrepareFn>(
          dlsym(handle_, "qrt_ck_fmha_q8192_prepare"));
      legacy_launch_ = reinterpret_cast<LegacyLaunchFn>(
          dlsym(handle_, "qrt_ck_fmha_q8192_bf16_launch"));
      release_ = reinterpret_cast<ReleaseFn>(
          dlsym(handle_, "qrt_ck_fmha_q8192_release"));
      if (legacy_prepare_ == nullptr || legacy_launch_ == nullptr ||
          release_ == nullptr) {
        throw std::runtime_error(
            "native q8192 CK provider ABI symbols are incomplete");
      }
      status = legacy_prepare_();
    }
    if (status != static_cast<int>(hipSuccess)) {
      throw std::runtime_error(
          "native q8192 CK provider prepare failed: hip_status=" +
          std::to_string(status));
    }
    metrics_.library_path = path;
    metrics_.loaded = true;
    metrics_.prepared = true;
    metrics_.context_tokens = context_tokens;
    return metrics_;
  } catch (...) {
    reset();
    throw;
  }
}

void NativeQ8192CkProvider::launch(
    const void* q_bf16, const void* k_bf16, const void* v_bf16,
    void* output_f32, std::size_t query_tokens, std::size_t kv_tokens,
    void* stream) {
  if (!loaded() || (launch_ == nullptr && legacy_launch_ == nullptr) ||
      q_bf16 == nullptr ||
      k_bf16 == nullptr || v_bf16 == nullptr || output_f32 == nullptr) {
    throw std::invalid_argument(
        "native q8192 CK launch requires a loaded provider and tensors");
  }
  if (query_tokens == 0) query_tokens = metrics_.context_tokens;
  if (kv_tokens == 0) kv_tokens = query_tokens;
  if (query_tokens > metrics_.context_tokens || kv_tokens < query_tokens ||
      kv_tokens > 262144) {
    throw std::invalid_argument(
        "native FMHA launch context geometry is unsupported");
  }
  int status = static_cast<int>(hipErrorInvalidValue);
  if (query_tokens != kv_tokens) {
    if (rectangular_launch_ == nullptr) {
      throw std::runtime_error(
          "native FMHA provider lacks the rectangular context ABI");
    }
    status = rectangular_launch_(
        q_bf16, k_bf16, v_bf16, output_f32,
        static_cast<unsigned int>(query_tokens),
        static_cast<unsigned int>(kv_tokens), stream);
  } else if (launch_ != nullptr) {
    status = launch_(q_bf16, k_bf16, v_bf16, output_f32,
                     static_cast<unsigned int>(query_tokens), stream);
  } else {
    if (query_tokens != metrics_.context_tokens) {
      throw std::runtime_error(
          "legacy native FMHA provider cannot change query context");
    }
    status = legacy_launch_(q_bf16, k_bf16, v_bf16, output_f32, stream);
  }
  if (status != static_cast<int>(hipSuccess)) {
    throw std::runtime_error(
        "native q8192 CK launch failed: hip_status=" +
        std::to_string(status));
  }
  ++metrics_.launches;
}

void NativeQ8192CkProvider::reset() noexcept {
  if (release_ != nullptr && metrics_.prepared) {
    (void)release_();
  }
  release_ = nullptr;
  legacy_launch_ = nullptr;
  legacy_prepare_ = nullptr;
  rectangular_launch_ = nullptr;
  launch_ = nullptr;
  prepare_ = nullptr;
  if (handle_ != nullptr) {
    (void)dlclose(handle_);
  }
  handle_ = nullptr;
  metrics_ = {};
}

NativeFullPrefillOracleResult probe_native_q8192_full_prefill_oracle(
    const std::filesystem::path& oracle_dir,
    const NativeWeightStore& weights,
    const NativePrefillWorkspace& workspace,
    NativePrefillInvocations& invocations,
    NativeDecodeExecutor& executor,
    NativeQ8192CkProvider& provider,
    const NativeFullPrefillOracleOptions& options) {
  if (!weights.loaded() || !workspace.built() || !executor.loaded() ||
      !provider.loaded() ||
      (invocations.launches().size() != 431 &&
       invocations.launches().size() != 401)) {
    throw std::invalid_argument(
        "native full prefill oracle requires complete resident owners");
  }
  const std::size_t tokens = workspace.context_tokens();
  const std::size_t comparison_tokens =
      options.comparison_tokens == 0 ? tokens : options.comparison_tokens;
  if (tokens == 0 || tokens > 262144 ||
      comparison_tokens == 0 || comparison_tokens > tokens ||
      provider.metrics().context_tokens < tokens ||
      (tokens != 8192 && options.collect_oracle_comparisons)) {
    throw std::invalid_argument(
        "native full prefill context or oracle mode is unsupported");
  }
  const bool use_mrope = options.mrope_positions_i64 != nullptr;
  const std::size_t mrope_position_row_stride =
      options.mrope_position_row_stride == 0
          ? tokens
          : options.mrope_position_row_stride;
  if ((use_mrope && mrope_position_row_stride < tokens) ||
      (!use_mrope && options.mrope_position_row_stride != 0)) {
    throw std::invalid_argument(
        "native full prefill M-RoPE position geometry is invalid");
  }
  const auto& launches = invocations.launches();
  const std::size_t base = find_full_layer_base(
      launches, options.layer_index);
  const bool q8192 = tokens == 8192;
  const bool split_projection_tail =
      !q8192 && workspace.find("native.tail_full_q_gate") != nullptr;
  const bool split_projections = q8192 || split_projection_tail;
  require_symbol(launches, base, "triton_rmsnorm_kernel");
  require_symbol(launches, base + 1,
                 "triton_prefill_fused_add_rmsnorm_kernel");
  require_symbol(launches, base + 2, "fused_moe_kernel");
  require_symbol(launches, base + 3, "fused_moe_kernel");

  const std::string prefix = "model.language_model.layers." +
                             std::to_string(options.layer_index) + ".";
  const NativeTensorView* q_weight = nullptr;
  const NativeTensorView* k_weight = nullptr;
  const NativeTensorView* v_weight = nullptr;
  const NativeDecodeBindingView* fused_weight = nullptr;
  if (split_projections) {
    q_weight = &require_weight(
        weights, prefix + "self_attn.q_proj.weight", 33554432ULL);
    k_weight = &require_weight(
        weights, prefix + "self_attn.k_proj.weight", 2097152ULL);
    v_weight = &require_weight(
        weights, prefix + "self_attn.v_proj.weight", 2097152ULL);
  } else {
    if (options.bindings == nullptr) {
      throw std::invalid_argument(
          "native direct full prefill requires derived bindings");
    }
    const std::string binding = "layer_weights." +
        std::to_string(options.layer_index) +
        ".tensors.full_qkv_proj_fused_t";
    fused_weight = options.bindings->find(binding);
    if (fused_weight == nullptr || fused_weight->device_pointer == nullptr ||
        fused_weight->payload_bytes != 37748736ULL ||
        fused_weight->dtype != DecodeTensorDtype::kBfloat16) {
      throw std::runtime_error(
          "native direct fused full-attention weight is missing");
    }
  }
  const auto& q_norm_weight = require_weight(
      weights, prefix + "self_attn.q_norm.weight", 512ULL);
  const auto& k_norm_weight = require_weight(
      weights, prefix + "self_attn.k_norm.weight", 512ULL);
  const auto& output_weight = require_weight(
      weights, prefix + "self_attn.o_proj.weight", 16777216ULL);
  const auto& input_norm_weight = require_weight(
      weights, prefix + "input_layernorm.weight", 4096ULL);
  const auto& post_attention_norm_weight = require_weight(
      weights, prefix + "post_attention_layernorm.weight", 4096ULL);

  const std::size_t hidden_bytes =
      tokens * kHidden * sizeof(std::uint16_t);
  const std::size_t q_gate_bytes =
      tokens * kQueryGateDimension * sizeof(std::uint16_t);
  const std::size_t qkv_fused_bytes =
      tokens * 9216ULL * sizeof(std::uint16_t);
  const std::size_t q_bytes =
      tokens * kQueryDimension * sizeof(std::uint16_t);
  const std::size_t kv_bytes =
      tokens * kKvDimension * sizeof(std::uint16_t);
  const std::size_t attention_f32_bytes =
      tokens * kQueryDimension * sizeof(float);
  const std::size_t rotary_bytes =
      tokens * kRotaryPairs * sizeof(float);

  if (options.decode_attention_state != nullptr &&
      (!options.decode_attention_state->built() ||
       options.cache_position_start + tokens >
           options.decode_attention_state->cache_capacity())) {
    throw std::invalid_argument(
        "native full prefill decode cache geometry is invalid");
  }

  void* layer_input = invocations.tensor_pointer(base, "x");
  void* normalized_input = invocations.tensor_pointer(base, "out");
  void* projected_attention =
      invocations.tensor_pointer(base + 1, "residual");
  void* after_attention =
      invocations.tensor_pointer(base + 1, "residual_out");
  void* post_attention_norm =
      invocations.tensor_pointer(base + 1, "norm_out");

  void* q_gate = nullptr;
  void* q = nullptr;
  void* attention_f32 = nullptr;
  void* gated = nullptr;
  void* cosine = nullptr;
  void* sine = nullptr;
  void* raw_k = nullptr;
  void* raw_v = nullptr;
  void* normalized_k = nullptr;
  void* normalized_v = nullptr;
  if (q8192) {
    q_gate = require_workspace(
        workspace, "transient.20", q_gate_bytes).device_pointer;
    q = require_workspace(
        workspace, "transient.34", q_bytes).device_pointer;
    attention_f32 = require_workspace(
        workspace, "transient.0", attention_f32_bytes).device_pointer;
    gated = require_workspace(
        workspace, "transient.35", q_bytes).device_pointer;
    cosine = require_workspace(
        workspace, "transient.12", rotary_bytes).device_pointer;
    sine = require_workspace(
        workspace, "transient.13", rotary_bytes).device_pointer;
    const auto& raw_kv = require_workspace(
        workspace, "transient.16", 2 * kv_bytes);
    const auto& k = require_workspace(workspace, "transient.6", kv_bytes);
    raw_k = raw_kv.device_pointer;
    raw_v = static_cast<unsigned char*>(raw_kv.device_pointer) + kv_bytes;
    normalized_k = k.device_pointer;
    normalized_v = raw_v;
  } else if (split_projection_tail) {
    q_gate = require_workspace(
        workspace, "native.tail_full_q_gate", q_gate_bytes).device_pointer;
    q = require_workspace(
        workspace, "native.tail_full_q", q_bytes).device_pointer;
    attention_f32 = require_workspace(
        workspace, "native.tail_full_attention_f32",
        attention_f32_bytes).device_pointer;
    gated = q;
    cosine = require_workspace(
        workspace, "native.tail_rotary_cos", rotary_bytes).device_pointer;
    sine = require_workspace(
        workspace, "native.tail_rotary_sin", rotary_bytes).device_pointer;
    raw_k = require_workspace(
        workspace, "native.tail_full_raw_k", kv_bytes).device_pointer;
  } else {
    const std::size_t scratch_base = find_direct_scratch_base(launches);
    q_gate = require_invocation_tensor(
        invocations, scratch_base + 1, "raw", qkv_fused_bytes);
    q = require_invocation_tensor(
        invocations, scratch_base + 7, "v_new", q_bytes);
    attention_f32 = require_invocation_tensor(
        invocations, scratch_base + 7, "h", attention_f32_bytes);
    gated = q;
    cosine = require_invocation_tensor(
        invocations, scratch_base + 2, "g_ptr", rotary_bytes);
    sine = require_invocation_tensor(
        invocations, scratch_base + 2, "beta_ptr", rotary_bytes);
    raw_k = static_cast<unsigned char*>(q_gate) +
            8192 * sizeof(std::uint16_t);
    raw_v = static_cast<unsigned char*>(q_gate) +
            8704 * sizeof(std::uint16_t);
    normalized_k = require_invocation_tensor(
        invocations, scratch_base + 2, "q_ptr", kv_bytes);
    normalized_v = require_invocation_tensor(
        invocations, scratch_base + 2, "k_ptr", kv_bytes);
  }
  if (options.decode_attention_state != nullptr) {
    constexpr std::size_t kCacheTokenBytes =
        kKvHeads * kHeadDimension * sizeof(std::uint16_t);
    const std::size_t offset =
        options.cache_position_start * kCacheTokenBytes;
    normalized_k = static_cast<unsigned char*>(
                       options.decode_attention_state->k_cache(
                           options.layer_index)) +
                   offset;
    normalized_v = static_cast<unsigned char*>(
                       options.decode_attention_state->v_cache(
                           options.layer_index)) +
                   offset;
    if (split_projections) raw_v = normalized_v;
  }
  if (split_projection_tail && options.decode_attention_state == nullptr) {
    throw std::invalid_argument(
        "native split-tail full prefill requires resident KV state");
  }

  const std::filesystem::path fixture =
      oracle_dir.empty() ? std::filesystem::path{}
                         : std::filesystem::absolute(oracle_dir);
  const auto oracle_file = [&fixture, &options](const std::string& label) {
    return find_native_oracle_tensor_file(
        fixture, options.oracle_label_prefix + label);
  };
  const std::filesystem::path tail_fixture =
      options.tail_oracle_dir.empty()
          ? std::filesystem::path{}
          : std::filesystem::absolute(options.tail_oracle_dir);
  const auto optional_tail_file = [&tail_fixture, &options](
                                      const char* label) {
    return find_native_oracle_tensor_file_if_present(
        tail_fixture, options.tail_oracle_label_prefix + label);
  };
  const std::filesystem::path sequence_fixture =
      options.sequence_oracle_dir.empty()
          ? std::filesystem::path{}
          : std::filesystem::absolute(options.sequence_oracle_dir);
  const auto optional_sequence_file = [&sequence_fixture, &options](
                                          const char* label) {
    return find_native_oracle_tensor_file_if_present(
        sequence_fixture, options.sequence_oracle_label_prefix + label);
  };

  NativeFullPrefillOracleResult result;
  const auto compare_optional_tail = [&] (
      const char* comparison_label, const void* pointer,
      std::size_t row_elements, std::size_t row_stride_elements,
      std::size_t intra_row_offset_elements, const char* oracle_label) {
    if (tail_fixture.empty()) return;
    const std::filesystem::path expected = optional_tail_file(oracle_label);
    if (expected.empty()) return;
    const auto* bytes = static_cast<const unsigned char*>(pointer);
    const std::size_t offset =
        ((comparison_tokens - 1) * row_stride_elements +
         intra_row_offset_elements) *
        sizeof(std::uint16_t);
    result.boundary_comparisons.push_back(compare_native_oracle_tensor(
        options.tail_oracle_label_prefix + comparison_label, "bfloat16",
        bytes + offset, row_elements * sizeof(std::uint16_t), expected));
  };
  const auto compare_optional_sequence = [&] (
      const char* comparison_label, const void* pointer,
      std::size_t elements, const char* oracle_label) {
    if (sequence_fixture.empty()) return;
    const std::filesystem::path expected =
        optional_sequence_file(oracle_label);
    if (expected.empty()) return;
    result.boundary_comparisons.push_back(compare_native_oracle_tensor(
        options.sequence_oracle_label_prefix + comparison_label,
        "bfloat16", pointer, elements * sizeof(std::uint16_t), expected));
  };
  result.layer.layer_index = options.layer_index;
  result.layer.tokens = tokens;
  if (options.decode_attention_state != nullptr) {
    result.layer.resident_kv_direct_bindings = 2;
    result.layer.resident_kv_payload_bytes = 2 * kv_bytes;
  }
  const auto diagnostic_stage = [&options](const char* stage) {
    if (!options.synchronize_substages) return;
    std::fprintf(stderr,
                 "{\"event\":\"native_full_prefill_stage\","
                 "\"layer_index\":%zu,\"stage\":\"%s\"}\n",
                 options.layer_index, stage);
    std::fflush(stderr);
  };
  if (options.seed_layer_input) {
    result.seed_bytes += seed_native_oracle_tensor(
        oracle_file("launch-000-x"), layer_input, hidden_bytes);
    ++result.seed_tensors;
  }

  std::unique_ptr<NativeQ8192PrefillGemmPlans> local_gemm_plans;
  NativeQ8192PrefillGemmPlans* gemm_plans = options.gemm_plans;
  if (gemm_plans == nullptr) {
    local_gemm_plans =
        std::make_unique<NativeQ8192PrefillGemmPlans>(tokens);
    gemm_plans = local_gemm_plans.get();
  }
  if (gemm_plans->token_count() != tokens) {
    throw std::invalid_argument("native full prefill GEMM context mismatch");
  }
  diagnostic_stage("before_q_plan");
  Bf16GemmPlan* q_plan =
      split_projections ? &gemm_plans->full_q() : nullptr;
  diagnostic_stage("after_q_plan");
  Bf16GemmPlan* kv_plan =
      split_projections ? &gemm_plans->full_kv() : nullptr;
  Bf16GemmPlan* fused_plan =
      split_projections ? nullptr : &gemm_plans->full_qkv();
  diagnostic_stage("after_kv_plan");
  Bf16GemmPlan& output_plan = gemm_plans->full_output();
  diagnostic_stage("after_output_plan");
  result.layer.gemm_workspace_bytes =
      output_plan.workspace_bytes() +
      (split_projections
           ? q_plan->workspace_bytes() + kv_plan->workspace_bytes()
           : fused_plan->workspace_bytes());

  const auto started = std::chrono::steady_clock::now();
  diagnostic_stage("before_input_norm");
  executor.launch(launches[base]);
  ++result.layer.aot_launches;
  compare_optional_tail(
      "input_last_token", layer_input, kHidden, kHidden, 0,
      "launch-000-x");
  compare_optional_tail(
      "input_norm_last_token", normalized_input, kHidden, kHidden, 0,
      "launch-000-out");
  compare_optional_tail(
      "attention_input_last_token", normalized_input, kHidden, kHidden, 0,
      "return-full_attention-inp");
  compare_optional_sequence(
      "attention_input_full_sequence", normalized_input,
      comparison_tokens * kHidden, "return-full_attention-inp");
  diagnostic_stage("before_q_projection");
  if (split_projections) {
    q_plan->launch(normalized_input, q_weight->device_pointer,
                   q_gate);
  } else {
    fused_plan->launch(normalized_input, fused_weight->device_pointer,
                       q_gate);
  }
  if (options.synchronize_substages) {
    check_hip(hipDeviceSynchronize(),
              "hipDeviceSynchronize native full q projection");
  }
  diagnostic_stage("after_q_projection");
  if (split_projections) {
    kv_plan->launch(normalized_input, k_weight->device_pointer, raw_k);
  }
  if (options.synchronize_substages) {
    check_hip(hipDeviceSynchronize(),
              "hipDeviceSynchronize native full k projection");
  }
  diagnostic_stage("after_k_projection");
  if (split_projections) {
    kv_plan->launch(normalized_input, v_weight->device_pointer, raw_v);
  }
  if (options.synchronize_substages) {
    check_hip(hipDeviceSynchronize(),
              "hipDeviceSynchronize native full v projection");
  }
  diagnostic_stage("after_v_projection");
  result.layer.dense_gemm_launches += split_projections ? 3 : 1;
  if (options.prepare_rotary_table) {
    if (use_mrope) {
      launch_prefill_mrope_rotary_table(
          cosine, sine, options.mrope_positions_i64, tokens,
          mrope_position_row_stride);
    } else {
      launch_prefill_rotary_table(
          cosine, sine, tokens, options.cache_position_start);
    }
    ++result.layer.native_pointwise_launches;
  }
  if (use_mrope) {
    launch_full_attention_head_norm_mrope_prefill(
        q_gate, raw_k, split_projections ? nullptr : raw_v,
        q_norm_weight.device_pointer, k_norm_weight.device_pointer,
        cosine, sine, q,
        normalized_k, split_projections ? nullptr : normalized_v, tokens,
        split_projections ? 8192 : 9216,
        split_projections ? 512 : 9216,
        split_projections ? 0 : 9216);
  } else {
    launch_full_attention_head_norm_rope_prefill(
        q_gate, raw_k, split_projections ? nullptr : raw_v,
        q_norm_weight.device_pointer, k_norm_weight.device_pointer,
        cosine, sine, q,
        normalized_k, split_projections ? nullptr : normalized_v, tokens,
        split_projections ? 8192 : 9216,
        split_projections ? 512 : 9216,
        split_projections ? 0 : 9216);
  }
  ++result.layer.native_pointwise_launches;
  if (options.collect_oracle_comparisons ||
      options.synchronize_substages) {
    check_hip(hipDeviceSynchronize(),
              "hipDeviceSynchronize native full prefill projections");
  }
  compare_optional_tail(
      "q_gate_projection_last_token", q_gate,
      kQueryGateDimension,
      split_projections ? kQueryGateDimension : 9216, 0,
      "return-full_attention-q_gate");
  compare_optional_tail(
      "normalized_rotary_q_last_token", q,
      kQueryDimension, kQueryDimension, 0, "return-full_attention-q");
  compare_optional_tail(
      "normalized_rotary_k_last_token", normalized_k,
      kKvDimension, kKvDimension, 0, "return-full_attention-k");
  compare_optional_tail(
      "raw_v_last_token", normalized_v,
      kKvDimension, kKvDimension, 0, "return-full_attention-v");
  compare_optional_sequence(
      "normalized_rotary_q_full_sequence", q,
      comparison_tokens * kQueryDimension, "return-full_attention-q");
  compare_optional_sequence(
      "normalized_rotary_k_full_sequence", normalized_k,
      comparison_tokens * kKvDimension, "return-full_attention-k");
  compare_optional_sequence(
      "raw_v_full_sequence", normalized_v,
      comparison_tokens * kKvDimension, "return-full_attention-v");

  if (options.collect_oracle_comparisons) {
    result.comparisons = {
      compare_native_oracle_tensor(
          "layer_input", "bfloat16", layer_input, hidden_bytes,
          oracle_file("launch-000-x")),
      compare_native_oracle_tensor(
          "attention_input", "bfloat16", normalized_input, hidden_bytes,
          oracle_file("launch-000-out")),
      compare_native_oracle_tensor(
          "q_gate_projection", "bfloat16", q_gate,
          q_gate_bytes, oracle_file("return-full_attention-q_gate")),
      compare_native_oracle_tensor(
          "normalized_rotary_q", "bfloat16", q,
          q_bytes, oracle_file("return-full_attention-q")),
      compare_native_oracle_tensor(
          "normalized_rotary_k", "bfloat16", normalized_k,
          kv_bytes, oracle_file("return-full_attention-k")),
      compare_native_oracle_tensor(
          "raw_v", "bfloat16", normalized_v, kv_bytes,
          oracle_file("return-full_attention-v")),
    };
  }

  const void* attention_k = normalized_k;
  const void* attention_v = normalized_v;
  if (options.decode_attention_state != nullptr) {
    attention_k = options.decode_attention_state->k_cache(options.layer_index);
    attention_v = options.decode_attention_state->v_cache(options.layer_index);
  }
  provider.launch(q, attention_k, attention_v, attention_f32, tokens,
                  options.cache_position_start + tokens);
  ++result.layer.native_ck_fmha_launches;
  if (options.synchronize_substages) {
    check_hip(hipDeviceSynchronize(),
              "hipDeviceSynchronize native full CK attention core");
  }
  diagnostic_stage("after_ck_attention");
  const std::filesystem::path pre_gate_expected =
      tail_fixture.empty()
          ? std::filesystem::path{}
          : optional_tail_file(
                "intermediate-full_attention-attn_pre_gate");
  const std::filesystem::path pre_gate_sequence_expected =
      sequence_fixture.empty()
          ? std::filesystem::path{}
          : optional_sequence_file(
                "intermediate-full_attention-attn_pre_gate");
  void* diagnostic_attention_bf16 = nullptr;
  if ((!pre_gate_expected.empty() ||
       !pre_gate_sequence_expected.empty()) && !split_projections) {
    check_hip(hipMalloc(&diagnostic_attention_bf16, q_bytes),
              "hipMalloc native full attention pre-gate diagnostic");
  }
  // The q buffer is dead after the provider launch on the same stream.  Reuse
  // it for the BF16-rounded attention boundary before the gate.
  launch_full_attention_sigmoid_gate_f32_prefill(
      attention_f32, q_gate,
      diagnostic_attention_bf16 == nullptr
          ? q
      : diagnostic_attention_bf16,
      gated, tokens,
      split_projections ? 8192 : 9216);
  ++result.layer.native_pointwise_launches;
  if (options.collect_oracle_comparisons ||
      options.synchronize_substages) {
    check_hip(hipDeviceSynchronize(),
              "hipDeviceSynchronize native q8192 CK attention");
  }
  if (!pre_gate_expected.empty()) {
    const void* pre_gate_attention =
        diagnostic_attention_bf16 == nullptr
            ? q
            : diagnostic_attention_bf16;
    compare_optional_tail(
        "attention_pre_gate_last_token", pre_gate_attention,
        kQueryDimension, kQueryDimension, 0,
        "intermediate-full_attention-attn_pre_gate");
  }
  if (!pre_gate_sequence_expected.empty()) {
    const void* pre_gate_attention =
        diagnostic_attention_bf16 == nullptr
            ? q
            : diagnostic_attention_bf16;
    result.boundary_comparisons.push_back(compare_native_oracle_tensor(
        options.sequence_oracle_label_prefix +
            "attention_pre_gate_full_sequence",
        "bfloat16", pre_gate_attention,
        comparison_tokens * kQueryDimension * sizeof(std::uint16_t),
        pre_gate_sequence_expected));
  }
  if (diagnostic_attention_bf16 != nullptr) {
    check_hip(hipFree(diagnostic_attention_bf16),
              "hipFree native full attention pre-gate diagnostic");
  }
  // The frozen Python local named `attn_out` is reassigned to the
  // sigmoid-gated BF16 tensor immediately before o_proj.  Compare that live
  // semantic boundary rather than the earlier CK output retained in q.
  if (options.collect_oracle_comparisons) {
    result.comparisons.push_back(compare_native_oracle_tensor(
        "gated_attention", "bfloat16", gated, q_bytes,
        oracle_file("return-full_attention-attn_out")));
  }
  compare_optional_tail(
      "gated_attention_last_token", gated,
      kQueryDimension, kQueryDimension, 0,
      "return-full_attention-attn_out");
  compare_optional_sequence(
      "gated_attention_full_sequence", gated,
      comparison_tokens * kQueryDimension,
      "return-full_attention-attn_out");

  output_plan.launch(gated, output_weight.device_pointer,
                     projected_attention);
  ++result.layer.dense_gemm_launches;
  if (options.synchronize_substages) {
    check_hip(hipDeviceSynchronize(),
              "hipDeviceSynchronize native full output projection");
  }
  compare_optional_tail(
      "projected_attention_last_token", projected_attention,
      kHidden, kHidden, 0, "return-full_attention-output");
  compare_optional_sequence(
      "projected_attention_full_sequence", projected_attention,
      comparison_tokens * kHidden, "return-full_attention-output");
  diagnostic_stage("after_output_projection");
  executor.launch(launches[base + 1]);
  ++result.layer.aot_launches;
  if (options.collect_oracle_comparisons ||
      options.synchronize_substages) {
    check_hip(hipDeviceSynchronize(),
              "hipDeviceSynchronize native full prefill residual");
  }
  compare_optional_tail(
      "post_attention_residual_last_token", after_attention,
      kHidden, kHidden, 0, "launch-001-residual_out");
  compare_optional_tail(
      "post_attention_norm_last_token", post_attention_norm,
      kHidden, kHidden, 0, "launch-001-norm_out");
  compare_optional_sequence(
      "post_attention_residual_full_sequence", after_attention,
      comparison_tokens * kHidden, "launch-001-residual_out");
  compare_optional_sequence(
      "post_attention_norm_full_sequence", post_attention_norm,
      comparison_tokens * kHidden, "launch-001-norm_out");
  result.layer.wall_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - started)
          .count();

  if (!options.collect_oracle_comparisons) {
    result.all_finite = true;
    for (const NativeOracleComparison& comparison :
         result.boundary_comparisons) {
      result.all_finite = result.all_finite &&
                          comparison.finite_elements == comparison.elements;
    }
    return result;
  }
  result.comparisons.push_back(compare_native_oracle_tensor(
      "projected_attention", "bfloat16", projected_attention, hidden_bytes,
      oracle_file("return-full_attention-output")));
  result.comparisons.push_back(compare_native_oracle_tensor(
      "post_attention_residual", "bfloat16", after_attention, hidden_bytes,
      oracle_file("launch-001-residual_out")));
  result.comparisons.push_back(compare_native_oracle_tensor(
      "post_attention_norm", "bfloat16", post_attention_norm, hidden_bytes,
      oracle_file("launch-001-norm_out")));

  result.all_finite = true;
  for (const NativeOracleComparison& comparison : result.comparisons) {
    result.all_finite = result.all_finite &&
                        comparison.finite_elements == comparison.elements;
  }
  result.q_gate_passed = gate(result.comparisons[2]);
  result.q_passed = gate(result.comparisons[3]);
  result.k_passed = gate(result.comparisons[4]);
  result.v_passed = gate(result.comparisons[5]);
  result.attention_output_passed = gate(result.comparisons[6]);
  result.projected_output_passed = gate(result.comparisons[7]);
  result.post_attention_passed =
      gate(result.comparisons[8]) && gate(result.comparisons[9]);
  return result;
}

}  // namespace aima
