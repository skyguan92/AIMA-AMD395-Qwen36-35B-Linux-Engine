// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_vl_logical_projections.h"

#include "aima/bf16_gemm.h"
#include "aima/native_prefill_gemm_plans.h"
#include "aima/native_weight_store.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace aima {
namespace {

constexpr std::size_t kHidden = 2048;
constexpr std::size_t kProjectionColumns = 32;
constexpr std::size_t kMergedColumns = 2 * kProjectionColumns;
constexpr std::size_t kLanguageLayers = 40;
constexpr std::size_t kFullAttentionPeriod = 4;
constexpr std::size_t kLinearLayers = 30;
constexpr std::size_t kExactPlanCacheEntries = 16;
constexpr std::size_t kWorkspaceLimit = 128ULL * 1024ULL * 1024ULL;
constexpr std::size_t kProjectionBytes =
    kProjectionColumns * kHidden * sizeof(std::uint16_t);
constexpr std::size_t kLayerBytes = 2 * kProjectionBytes;

void check_hip(hipError_t status, const char* operation) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorName(status) + " (" +
                             hipGetErrorString(status) + ")");
  }
}

double elapsed_ms(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - start)
      .count();
}

std::size_t linear_slot(std::size_t layer_index) {
  if (layer_index >= kLanguageLayers ||
      layer_index % kFullAttentionPeriod == kFullAttentionPeriod - 1) {
    throw std::invalid_argument(
        "VL logical A/B weight requires a linear-attention layer");
  }
  return layer_index - layer_index / kFullAttentionPeriod;
}

}  // namespace

struct NativeVlLogicalProjectionState::Impl {
  struct PlanEntry {
    std::size_t tokens = 0;
    std::size_t algorithm_source_tokens = 0;
    std::unique_ptr<NativeQ8192PrefillGemmPlans> gemm_plans;
    std::unique_ptr<Bf16GemmPlan> ab_plan;
    std::uint64_t use = 0;
  };

  int device = 0;
  void* weights = nullptr;
  void* output = nullptr;
  std::size_t maximum_tokens = 0;
  std::size_t prepared_tokens = 0;
  std::vector<PlanEntry> resident_plans;
  PlanEntry* active_plan = nullptr;
  std::uint64_t plan_clock = 0;
  NativeVlLogicalProjectionLoadMetrics load_metrics;

  void reset() noexcept {
    (void)hipSetDevice(device);
    active_plan = nullptr;
    resident_plans.clear();
    prepared_tokens = 0;
    plan_clock = 0;
    if (output != nullptr) {
      (void)hipFree(output);
      output = nullptr;
    }
    if (weights != nullptr) {
      (void)hipFree(weights);
      weights = nullptr;
    }
    maximum_tokens = 0;
    load_metrics = {};
  }

  ~Impl() { reset(); }
};

NativeVlLogicalProjectionState::NativeVlLogicalProjectionState()
    : impl_(std::make_unique<Impl>()) {}
NativeVlLogicalProjectionState::~NativeVlLogicalProjectionState() = default;

NativeVlLogicalProjectionLoadMetrics NativeVlLogicalProjectionState::build(
    const NativeWeightStore& weights, std::size_t maximum_tokens, int device) {
  if (impl_->weights != nullptr || !weights.loaded() || maximum_tokens == 0 ||
      maximum_tokens > kNativeVlLogicalProjectionMaximumTokens) {
    throw std::invalid_argument(
        "VL logical projection resident build contract is invalid");
  }
  const auto started = std::chrono::steady_clock::now();
  impl_->device = device;
  impl_->maximum_tokens = maximum_tokens;
  const std::uint64_t weight_bytes = kLinearLayers * kLayerBytes;
  const std::uint64_t output_bytes =
      maximum_tokens * kMergedColumns * sizeof(std::uint16_t);
  check_hip(hipSetDevice(device), "hipSetDevice VL logical projections");
  try {
    check_hip(hipMalloc(&impl_->weights, weight_bytes),
              "hipMalloc VL logical A/B weights");
    check_hip(hipMalloc(&impl_->output, output_bytes),
              "hipMalloc VL logical A/B output");
    for (std::size_t layer_index = 0; layer_index < kLanguageLayers;
         ++layer_index) {
      if (layer_index % kFullAttentionPeriod == kFullAttentionPeriod - 1) {
        continue;
      }
      const std::string prefix = "model.language_model.layers." +
                                 std::to_string(layer_index) +
                                 ".linear_attn.in_proj_";
      const NativeTensorView* a_weight = weights.find(prefix + "a.weight");
      const NativeTensorView* b_weight = weights.find(prefix + "b.weight");
      if (a_weight == nullptr || b_weight == nullptr ||
          a_weight->device_pointer == nullptr ||
          b_weight->device_pointer == nullptr ||
          a_weight->payload_bytes != kProjectionBytes ||
          b_weight->payload_bytes != kProjectionBytes) {
        throw std::runtime_error(
            "VL logical A/B source weight geometry differs");
      }
      auto* destination = static_cast<unsigned char*>(impl_->weights) +
                          linear_slot(layer_index) * kLayerBytes;
      check_hip(hipMemcpy(destination, a_weight->device_pointer,
                          kProjectionBytes, hipMemcpyDeviceToDevice),
                "hipMemcpy VL logical A weight");
      check_hip(hipMemcpy(destination + kProjectionBytes,
                          b_weight->device_pointer, kProjectionBytes,
                          hipMemcpyDeviceToDevice),
                "hipMemcpy VL logical B weight");
    }
    check_hip(hipMemset(impl_->output, 0, output_bytes),
              "hipMemset VL logical A/B output");
    impl_->resident_plans.reserve(kExactPlanCacheEntries);
  } catch (...) {
    impl_->reset();
    throw;
  }
  impl_->load_metrics.weight_bytes = weight_bytes;
  impl_->load_metrics.output_scratch_bytes = output_bytes;
  impl_->load_metrics.linear_layer_count = kLinearLayers;
  impl_->load_metrics.maximum_tokens = maximum_tokens;
  impl_->load_metrics.build_wall_ms = elapsed_ms(started);
  impl_->load_metrics.loaded = true;
  return impl_->load_metrics;
}

NativeVlLogicalProjectionPrepareMetrics
NativeVlLogicalProjectionState::prepare(
    std::size_t tokens,
    NativeQ8192PrefillGemmPlans& algorithm_source) {
  if (!loaded() || tokens == 0 || tokens > impl_->maximum_tokens) {
    throw std::invalid_argument(
        "VL logical projection plan token count is invalid");
  }
  NativeVlLogicalProjectionPrepareMetrics metrics;
  metrics.tokens = tokens;
  const std::size_t source_tokens = algorithm_source.token_count();
  if (source_tokens < tokens || source_tokens > 2048) {
    throw std::invalid_argument(
        "VL logical projection algorithm source is invalid");
  }
  auto selected = std::find_if(
      impl_->resident_plans.begin(), impl_->resident_plans.end(),
      [tokens, source_tokens](const Impl::PlanEntry& entry) {
        return entry.tokens == tokens &&
               entry.algorithm_source_tokens == source_tokens &&
               entry.gemm_plans != nullptr &&
               entry.ab_plan != nullptr;
      });
  if (selected == impl_->resident_plans.end()) {
    const auto started = std::chrono::steady_clock::now();
    if (impl_->resident_plans.size() >= kExactPlanCacheEntries) {
      impl_->active_plan = nullptr;
      selected = std::min_element(
          impl_->resident_plans.begin(), impl_->resident_plans.end(),
          [](const Impl::PlanEntry& left, const Impl::PlanEntry& right) {
            return left.use < right.use;
          });
      impl_->resident_plans.erase(selected);
    }
    check_hip(hipSetDevice(impl_->device),
              "hipSetDevice VL logical exact plan build");
    Impl::PlanEntry entry;
    entry.tokens = tokens;
    entry.algorithm_source_tokens = source_tokens;
    entry.gemm_plans = std::make_unique<NativeQ8192PrefillGemmPlans>(
        tokens, &algorithm_source);
    entry.gemm_plans->prepare_logical_linear_and_moe();
    entry.ab_plan = std::make_unique<Bf16GemmPlan>(
        tokens, kMergedColumns, kHidden, kWorkspaceLimit, true);
    entry.use = ++impl_->plan_clock;
    impl_->resident_plans.push_back(std::move(entry));
    selected = std::prev(impl_->resident_plans.end());
    metrics.build_wall_ms = elapsed_ms(started);
  } else {
    selected->use = ++impl_->plan_clock;
    metrics.reused = true;
  }
  impl_->active_plan = &*selected;
  impl_->prepared_tokens = tokens;
  metrics.plan_count =
      impl_->active_plan->gemm_plans->built_plan_count() + 1;
  metrics.workspace_bytes =
      impl_->active_plan->ab_plan->workspace_bytes() +
      impl_->active_plan->gemm_plans->workspace_bytes();
  metrics.prepared = true;
  return metrics;
}

void NativeVlLogicalProjectionState::reset() noexcept { impl_->reset(); }

bool NativeVlLogicalProjectionState::loaded() const {
  return impl_->weights != nullptr && impl_->output != nullptr &&
         impl_->load_metrics.loaded;
}

bool NativeVlLogicalProjectionState::prepared() const {
  return loaded() && impl_->prepared_tokens != 0 &&
         impl_->active_plan != nullptr &&
         impl_->active_plan->ab_plan != nullptr &&
         impl_->active_plan->gemm_plans != nullptr;
}

std::size_t NativeVlLogicalProjectionState::prepared_tokens() const {
  return impl_->prepared_tokens;
}

const void* NativeVlLogicalProjectionState::ab_weight(
    std::size_t layer_index) const {
  if (!loaded()) {
    throw std::runtime_error("VL logical A/B weights are not loaded");
  }
  return static_cast<const unsigned char*>(impl_->weights) +
         linear_slot(layer_index) * kLayerBytes;
}

void* NativeVlLogicalProjectionState::ab_output() const {
  if (!loaded()) {
    throw std::runtime_error("VL logical A/B output owner is not loaded");
  }
  return impl_->output;
}

Bf16GemmPlan& NativeVlLogicalProjectionState::ab_plan() const {
  if (!prepared()) {
    throw std::runtime_error("VL logical A/B plan is not prepared");
  }
  return *impl_->active_plan->ab_plan;
}

NativeQ8192PrefillGemmPlans&
NativeVlLogicalProjectionState::router_gemm_plans() const {
  if (!prepared()) {
    throw std::runtime_error("VL logical router plan is not prepared");
  }
  return *impl_->active_plan->gemm_plans;
}

const NativeVlLogicalProjectionLoadMetrics&
NativeVlLogicalProjectionState::load_metrics() const {
  return impl_->load_metrics;
}

}  // namespace aima
