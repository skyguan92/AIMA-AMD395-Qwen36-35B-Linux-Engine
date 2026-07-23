#pragma once

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_decode_executor.h"
#include "aima/native_full_attention.h"
#include "aima/native_layer_oracle.h"
#include "aima/native_prefill_invocation.h"
#include "aima/native_prefill_workspace.h"
#include "aima/native_weight_store.h"

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace aima {

class NativeQ8192PrefillGemmPlans;
class NativeDecodeBindings;

struct NativeQ8192CkProviderMetrics {
  std::filesystem::path library_path;
  bool loaded = false;
  bool prepared = false;
  bool generic_context_abi = false;
  bool rectangular_context_abi = false;
  std::size_t context_tokens = 0;
  std::size_t launches = 0;
};

// Process-resident owner for the admitted fixed-shape CK-Tile FMHA
// component.  The shared object is a native HIP/CK artifact and has no Python,
// Torch, vLLM, or Triton runtime dependency.
class NativeQ8192CkProvider {
 public:
  NativeQ8192CkProvider() = default;
  ~NativeQ8192CkProvider();
  NativeQ8192CkProvider(const NativeQ8192CkProvider&) = delete;
  NativeQ8192CkProvider& operator=(const NativeQ8192CkProvider&) = delete;

  NativeQ8192CkProviderMetrics load(
      const std::filesystem::path& library_path,
      std::size_t context_tokens = 8192);
  void launch(const void* q_bf16, const void* k_bf16,
              const void* v_bf16, void* output_f32,
              std::size_t query_tokens = 0,
              std::size_t kv_tokens = 0,
              void* stream = nullptr);
  void reset() noexcept;
  bool loaded() const { return handle_ != nullptr; }
  const NativeQ8192CkProviderMetrics& metrics() const { return metrics_; }

 private:
  using PrepareFn = int (*)(unsigned int);
  using LaunchFn = int (*)(const void*, const void*, const void*, void*,
                           unsigned int, void*);
  using RectangularLaunchFn = int (*)(const void*, const void*, const void*,
                                      void*, unsigned int, unsigned int,
                                      void*);
  using LegacyPrepareFn = int (*)();
  using LegacyLaunchFn = int (*)(const void*, const void*, const void*, void*,
                                 void*);
  using ReleaseFn = int (*)();

  void* handle_ = nullptr;
  PrepareFn prepare_ = nullptr;
  LaunchFn launch_ = nullptr;
  RectangularLaunchFn rectangular_launch_ = nullptr;
  LegacyPrepareFn legacy_prepare_ = nullptr;
  LegacyLaunchFn legacy_launch_ = nullptr;
  ReleaseFn release_ = nullptr;
  NativeQ8192CkProviderMetrics metrics_;
};

struct NativeFullPrefillMetrics {
  std::size_t layer_index = 0;
  std::size_t tokens = 0;
  std::size_t dense_gemm_launches = 0;
  std::size_t native_pointwise_launches = 0;
  std::size_t native_ck_fmha_launches = 0;
  std::size_t resident_kv_direct_bindings = 0;
  std::size_t resident_kv_payload_bytes = 0;
  std::size_t aot_launches = 0;
  std::size_t gemm_workspace_bytes = 0;
  double wall_ms = 0.0;
};

struct NativeFullPrefillOracleOptions {
  std::size_t layer_index = 3;
  bool seed_layer_input = true;
  bool prepare_rotary_table = true;
  bool collect_oracle_comparisons = true;
  bool synchronize_substages = false;
  // Optional product state owner.  Normalized K and raw V are produced
  // directly into its sequence-major cache, avoiding a promotion copy.
  NativeFullAttentionState* decode_attention_state = nullptr;
  NativeQ8192PrefillGemmPlans* gemm_plans = nullptr;
  const NativeDecodeBindings* bindings = nullptr;
  std::size_t cache_position_start = 0;
  std::string oracle_label_prefix = "layer-003-";
  // Qualification-only last-token boundaries. Sparse fixtures are accepted
  // so long-context attribution does not require copying full activations.
  std::filesystem::path tail_oracle_dir;
  std::string tail_oracle_label_prefix;
  // Qualification-only full-sequence boundaries. Sparse fixtures are
  // accepted so one focused layer can localize a context-specific drift.
  std::filesystem::path sequence_oracle_dir;
  std::string sequence_oracle_label_prefix;
};

struct NativeFullPrefillOracleResult {
  NativeFullPrefillMetrics layer;
  std::size_t seed_tensors = 0;
  std::size_t seed_bytes = 0;
  std::vector<NativeOracleComparison> comparisons;
  std::vector<NativeOracleComparison> boundary_comparisons;
  bool all_finite = false;
  bool q_gate_passed = false;
  bool q_passed = false;
  bool k_passed = false;
  bool v_passed = false;
  bool attention_output_passed = false;
  bool projected_output_passed = false;
  bool post_attention_passed = false;
};

// Executes the complete attention half of one q8192 full-attention layer:
// captured RMSNorm, three native BF16 projection GEMMs, native head
// RMSNorm+RoPE, the admitted CK-Tile causal FMHA provider, native gate,
// native BF16 output projection, and captured fused residual+RMSNorm.
// Oracle reads and the optional entry seed are qualification-only.
NativeFullPrefillOracleResult probe_native_q8192_full_prefill_oracle(
    const std::filesystem::path& oracle_dir,
    const NativeWeightStore& weights,
    const NativePrefillWorkspace& workspace,
    NativePrefillInvocations& invocations,
    NativeDecodeExecutor& executor,
    NativeQ8192CkProvider& provider,
    const NativeFullPrefillOracleOptions& options = {});

}  // namespace aima
