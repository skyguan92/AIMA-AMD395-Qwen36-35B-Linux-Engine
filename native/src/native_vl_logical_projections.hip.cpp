// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_vl_logical_projections.h"

#include "aima/bf16_gemm.h"
#include "aima/native_prefill_gemm_plans.h"
#include "aima/native_weight_store.h"

#include <hip/hip_runtime.h>

#include <chrono>
#include <stdexcept>
#include <string>

namespace aima {
namespace {

constexpr std::size_t kHidden = 2048;
constexpr std::size_t kProjectionColumns = 32;
constexpr std::size_t kMergedColumns = 2 * kProjectionColumns;
constexpr std::size_t kLanguageLayers = 40;
constexpr std::size_t kFullAttentionPeriod = 4;
constexpr std::size_t kLinearLayers = 30;
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
  int device = 0;
  void* weights = nullptr;
  void* output = nullptr;
  std::size_t maximum_tokens = 0;
  std::size_t prepared_tokens = 0;
  std::unique_ptr<NativeQ8192PrefillGemmPlans> router_gemm_plans;
  std::unique_ptr<Bf16GemmPlan> ab_plan;
  NativeVlLogicalProjectionLoadMetrics load_metrics;

  void reset() noexcept {
    ab_plan.reset();
    router_gemm_plans.reset();
    prepared_tokens = 0;
    if (output != nullptr) {
      (void)hipSetDevice(device);
      (void)hipFree(output);
      output = nullptr;
    }
    if (weights != nullptr) {
      (void)hipSetDevice(device);
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
      maximum_tokens > 1024) {
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
    check_hip(hipDeviceSynchronize(),
              "hipDeviceSynchronize VL logical projection build");
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
NativeVlLogicalProjectionState::prepare(std::size_t tokens) {
  if (!loaded() || tokens == 0 || tokens > impl_->maximum_tokens) {
    throw std::invalid_argument(
        "VL logical projection plan token count is invalid");
  }
  NativeVlLogicalProjectionPrepareMetrics metrics;
  metrics.tokens = tokens;
  if (impl_->prepared_tokens == tokens && impl_->ab_plan != nullptr &&
      impl_->router_gemm_plans != nullptr) {
    metrics.plan_count = 2;
    metrics.workspace_bytes =
        impl_->ab_plan->workspace_bytes() +
        impl_->router_gemm_plans->workspace_bytes();
    metrics.reused = true;
    metrics.prepared = true;
    return metrics;
  }

  const auto started = std::chrono::steady_clock::now();
  impl_->ab_plan.reset();
  impl_->router_gemm_plans.reset();
  impl_->prepared_tokens = 0;
  impl_->router_gemm_plans =
      std::make_unique<NativeQ8192PrefillGemmPlans>(tokens);
  (void)impl_->router_gemm_plans->moe_router();
  impl_->ab_plan = std::make_unique<Bf16GemmPlan>(
      tokens, kMergedColumns, kHidden, kWorkspaceLimit, true);
  impl_->prepared_tokens = tokens;
  metrics.plan_count = 2;
  metrics.workspace_bytes =
      impl_->ab_plan->workspace_bytes() +
      impl_->router_gemm_plans->workspace_bytes();
  metrics.build_wall_ms = elapsed_ms(started);
  metrics.prepared = true;
  return metrics;
}

void NativeVlLogicalProjectionState::reset() noexcept { impl_->reset(); }

bool NativeVlLogicalProjectionState::loaded() const {
  return impl_->weights != nullptr && impl_->output != nullptr &&
         impl_->load_metrics.loaded;
}

bool NativeVlLogicalProjectionState::prepared() const {
  return loaded() && impl_->prepared_tokens != 0 && impl_->ab_plan != nullptr &&
         impl_->router_gemm_plans != nullptr;
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
  return *impl_->ab_plan;
}

NativeQ8192PrefillGemmPlans&
NativeVlLogicalProjectionState::router_gemm_plans() const {
  if (!prepared()) {
    throw std::runtime_error("VL logical router plan is not prepared");
  }
  return *impl_->router_gemm_plans;
}

const NativeVlLogicalProjectionLoadMetrics&
NativeVlLogicalProjectionState::load_metrics() const {
  return impl_->load_metrics;
}

}  // namespace aima
