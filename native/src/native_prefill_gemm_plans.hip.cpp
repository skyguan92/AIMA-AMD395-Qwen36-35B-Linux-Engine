// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_prefill_gemm_plans.h"

#include "aima/bf16_gemm.h"

#include <hip/hip_runtime.h>

#include <memory>
#include <stdexcept>
#include <string>

namespace aima {
namespace {

constexpr std::size_t kHidden = 2048;
constexpr std::size_t kWorkspaceLimit = 128ULL * 1024ULL * 1024ULL;

void check_hip(hipError_t status, const char* operation) {
  if (status != hipSuccess) {
    throw std::runtime_error(
        std::string(operation) + ": " + hipGetErrorString(status));
  }
}

class WarmupAllocation {
 public:
  explicit WarmupAllocation(std::size_t bytes) {
    check_hip(hipMalloc(&pointer_, bytes), "hipMalloc prefill GEMM warmup");
    check_hip(hipMemset(pointer_, 0, bytes),
              "hipMemset prefill GEMM warmup");
  }
  ~WarmupAllocation() {
    if (pointer_ != nullptr) {
      const hipError_t ignored = hipFree(pointer_);
      static_cast<void>(ignored);
    }
  }
  WarmupAllocation(const WarmupAllocation&) = delete;
  WarmupAllocation& operator=(const WarmupAllocation&) = delete;
  void* get() const { return pointer_; }

 private:
  void* pointer_ = nullptr;
};

template <typename Factory>
Bf16GemmPlan& get_or_build(std::unique_ptr<Bf16GemmPlan>& value,
                           Factory&& factory) {
  if (!value) value = factory();
  return *value;
}

}  // namespace

struct NativeQ8192PrefillGemmPlans::Impl {
  explicit Impl(std::size_t value) : tokens(value) {}
  std::size_t tokens = 0;
  std::unique_ptr<Bf16GemmPlan> linear_qkv;
  std::unique_ptr<Bf16GemmPlan> linear_z;
  std::unique_ptr<Bf16GemmPlan> linear_ab;
  std::unique_ptr<Bf16GemmPlan> linear_fused_input;
  std::unique_ptr<Bf16GemmPlan> linear_output;
  std::unique_ptr<Bf16GemmPlan> full_q;
  std::unique_ptr<Bf16GemmPlan> full_kv;
  std::unique_ptr<Bf16GemmPlan> full_qkv;
  std::unique_ptr<Bf16GemmPlan> full_output;
  std::unique_ptr<Bf16GemmPlan> moe_shared_gate;
  std::unique_ptr<Bf16GemmPlan> moe_shared_projection;
  std::unique_ptr<Bf16GemmPlan> moe_shared_down;
  std::unique_ptr<Bf16GemmPlan> moe_router;
};

NativeQ8192PrefillGemmPlans::NativeQ8192PrefillGemmPlans(
    std::size_t token_count)
    : impl_(std::make_unique<Impl>(token_count)) {
  if (token_count == 0 || token_count > 262144) {
    throw std::invalid_argument("unsupported native prefill GEMM context");
  }
}
NativeQ8192PrefillGemmPlans::~NativeQ8192PrefillGemmPlans() = default;

Bf16GemmPlan& NativeQ8192PrefillGemmPlans::linear_qkv() {
  const std::size_t tokens = impl_->tokens;
  return get_or_build(impl_->linear_qkv, [tokens] {
    return std::make_unique<Bf16GemmPlan>(
        tokens, 8192, kHidden, kWorkspaceLimit, true);
  });
}
Bf16GemmPlan& NativeQ8192PrefillGemmPlans::linear_z() {
  const std::size_t tokens = impl_->tokens;
  return get_or_build(impl_->linear_z, [tokens] {
    return std::make_unique<Bf16GemmPlan>(
        tokens, 4096, kHidden, kWorkspaceLimit, true);
  });
}
Bf16GemmPlan& NativeQ8192PrefillGemmPlans::linear_ab() {
  const std::size_t tokens = impl_->tokens;
  return get_or_build(impl_->linear_ab, [tokens] {
    return std::make_unique<Bf16GemmPlan>(
        tokens, 32, kHidden, kWorkspaceLimit, true);
  });
}
Bf16GemmPlan& NativeQ8192PrefillGemmPlans::linear_fused_input() {
  const std::size_t tokens = impl_->tokens;
  return get_or_build(impl_->linear_fused_input, [tokens] {
    return std::make_unique<Bf16GemmPlan>(
        tokens, 12352, kHidden, kWorkspaceLimit, false);
  });
}
Bf16GemmPlan& NativeQ8192PrefillGemmPlans::linear_output() {
  const std::size_t tokens = impl_->tokens;
  return get_or_build(impl_->linear_output, [tokens] {
    return std::make_unique<Bf16GemmPlan>(
        tokens, kHidden, 4096, kWorkspaceLimit, true);
  });
}

Bf16GemmPlan& NativeQ8192PrefillGemmPlans::full_q() {
  const std::size_t tokens = impl_->tokens;
  return get_or_build(impl_->full_q, [tokens] {
    return std::make_unique<Bf16GemmPlan>(
        tokens, 8192, kHidden, kWorkspaceLimit, true);
  });
}
Bf16GemmPlan& NativeQ8192PrefillGemmPlans::full_kv() {
  const std::size_t tokens = impl_->tokens;
  return get_or_build(impl_->full_kv, [tokens] {
    return std::make_unique<Bf16GemmPlan>(
        tokens, 512, kHidden, kWorkspaceLimit, true);
  });
}
Bf16GemmPlan& NativeQ8192PrefillGemmPlans::full_qkv() {
  const std::size_t tokens = impl_->tokens;
  return get_or_build(impl_->full_qkv, [tokens] {
    return std::make_unique<Bf16GemmPlan>(
        tokens, 9216, kHidden, kWorkspaceLimit, false);
  });
}
Bf16GemmPlan& NativeQ8192PrefillGemmPlans::full_output() {
  const std::size_t tokens = impl_->tokens;
  return get_or_build(impl_->full_output, [tokens] {
    return std::make_unique<Bf16GemmPlan>(
        tokens, kHidden, 4096, kWorkspaceLimit, true);
  });
}

Bf16GemmPlan& NativeQ8192PrefillGemmPlans::moe_shared_gate() {
  const std::size_t tokens = impl_->tokens;
  return get_or_build(impl_->moe_shared_gate, [tokens] {
    return std::make_unique<Bf16GemmPlan>(
        tokens, 1, kHidden, 76ULL * 1024ULL * 1024ULL, true);
  });
}
Bf16GemmPlan& NativeQ8192PrefillGemmPlans::moe_shared_projection() {
  const std::size_t tokens = impl_->tokens;
  return get_or_build(impl_->moe_shared_projection, [tokens] {
    return std::make_unique<Bf16GemmPlan>(
        tokens, 512, kHidden, kWorkspaceLimit, true);
  });
}
Bf16GemmPlan& NativeQ8192PrefillGemmPlans::moe_shared_down() {
  const std::size_t tokens = impl_->tokens;
  return get_or_build(impl_->moe_shared_down, [tokens] {
    return std::make_unique<Bf16GemmPlan>(
        tokens, kHidden, 512, kWorkspaceLimit, true);
  });
}
Bf16GemmPlan& NativeQ8192PrefillGemmPlans::moe_router() {
  const std::size_t tokens = impl_->tokens;
  return get_or_build(impl_->moe_router, [tokens] {
    return std::make_unique<Bf16GemmPlan>(
        tokens, 256, kHidden, kWorkspaceLimit, true);
  });
}

void NativeQ8192PrefillGemmPlans::prepare_all() {
  if (impl_->tokens == 8192) {
    (void)linear_qkv();
    (void)linear_z();
    (void)linear_ab();
    (void)linear_output();
    (void)full_q();
    (void)full_kv();
    (void)full_output();
  } else {
    (void)linear_fused_input();
    (void)linear_output();
    (void)full_qkv();
    (void)full_output();
  }
  (void)moe_shared_gate();
  (void)moe_shared_projection();
  (void)moe_shared_down();
  (void)moe_router();
}

void NativeQ8192PrefillGemmPlans::prepare_logical_linear_and_moe() {
  (void)linear_fused_input();
  (void)linear_output();
  (void)full_qkv();
  (void)full_output();
  (void)moe_shared_gate();
  (void)moe_shared_projection();
  (void)moe_shared_down();
  (void)moe_router();
}

void NativeQ8192PrefillGemmPlans::warm_up_q1024_text() {
  if (impl_->tokens != 1024) {
    throw std::invalid_argument(
        "native q1024 text GEMM warmup requires the q1024 plan owner");
  }
  prepare_all();
  // linear_fused_input is the largest B/D geometry; linear_output requires
  // the largest A geometry.  Reuse three private aligned owners for both.
  WarmupAllocation a(1024ULL * 4096ULL * sizeof(std::uint16_t));
  WarmupAllocation b(2048ULL * 12352ULL * sizeof(std::uint16_t));
  WarmupAllocation d(1024ULL * 12352ULL * sizeof(std::uint16_t));
  linear_fused_input().launch(a.get(), b.get(), d.get());
  linear_output().launch(a.get(), b.get(), d.get());
  check_hip(hipDeviceSynchronize(),
            "hipDeviceSynchronize prefill GEMM warmup");
}

std::size_t NativeQ8192PrefillGemmPlans::built_plan_count() const {
  return static_cast<std::size_t>(!!impl_->linear_qkv) +
         static_cast<std::size_t>(!!impl_->linear_z) +
         static_cast<std::size_t>(!!impl_->linear_ab) +
         static_cast<std::size_t>(!!impl_->linear_fused_input) +
         static_cast<std::size_t>(!!impl_->linear_output) +
         static_cast<std::size_t>(!!impl_->full_q) +
         static_cast<std::size_t>(!!impl_->full_kv) +
         static_cast<std::size_t>(!!impl_->full_qkv) +
         static_cast<std::size_t>(!!impl_->full_output) +
         static_cast<std::size_t>(!!impl_->moe_shared_gate) +
         static_cast<std::size_t>(!!impl_->moe_shared_projection) +
         static_cast<std::size_t>(!!impl_->moe_shared_down) +
         static_cast<std::size_t>(!!impl_->moe_router);
}

std::size_t NativeQ8192PrefillGemmPlans::workspace_bytes() const {
  std::size_t total = 0;
  const auto add = [&total](const std::unique_ptr<Bf16GemmPlan>& plan) {
    if (plan) total += plan->workspace_bytes();
  };
  add(impl_->linear_qkv);
  add(impl_->linear_z);
  add(impl_->linear_ab);
  add(impl_->linear_fused_input);
  add(impl_->linear_output);
  add(impl_->full_q);
  add(impl_->full_kv);
  add(impl_->full_qkv);
  add(impl_->full_output);
  add(impl_->moe_shared_gate);
  add(impl_->moe_shared_projection);
  add(impl_->moe_shared_down);
  add(impl_->moe_router);
  return total;
}

std::size_t NativeQ8192PrefillGemmPlans::token_count() const {
  return impl_->tokens;
}

}  // namespace aima
