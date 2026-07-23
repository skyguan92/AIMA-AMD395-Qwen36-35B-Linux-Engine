#pragma once

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_layer_oracle.h"
#include "aima/native_weight_store.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace aima {

struct NativeResidentEngineOptions {
  NativeWeightLoadOptions weights;
  // Optional native FMHA provider override. When empty, the engine selects
  // the provider shipped beside the executable for the admitted context.
  // The historical field name is retained as source compatibility for
  // qualification tools.
  std::filesystem::path ck_provider;
  // Optional second native FMHA provider used by an explicit subset of the
  // ten full-attention layers. This keeps one resident engine while allowing
  // correctness-sensitive layers to use a different qualified backend.
  std::filesystem::path secondary_fmha_provider;
  std::vector<std::size_t> secondary_fmha_layers;
  // Static prefill schedule selected for this resident process.
  std::size_t prompt_tokens = 8192;
  // Includes prompt and decoded cache entries.
  std::size_t cache_capacity = 9216;
};

struct NativeResidentLoadMetrics {
  std::string device_name;
  std::string gpu_arch;
  std::uint64_t model_payload_bytes = 0;
  std::size_t model_tensor_count = 0;
  std::size_t model_shard_count = 0;
  std::size_t decode_weight_bindings = 0;
  std::size_t prefill_prepared_launches = 0;
  std::size_t decode_prepared_launches = 0;
  std::size_t aot_loaded_modules = 0;
  std::size_t prefill_gemm_plans = 0;
  std::uint64_t prefill_workspace_bytes = 0;
  std::uint64_t decode_workspace_bytes = 0;
  std::uint64_t attention_state_bytes = 0;
  std::uint64_t exact_prefix_cache_bytes = 0;
  std::size_t cache_capacity = 0;
  std::size_t prompt_tokens = 0;
  std::string fmha_provider_backend;
  std::string fmha_provider_path;
  bool fmha_provider_loaded = false;
  std::string secondary_fmha_provider_backend;
  std::string secondary_fmha_provider_path;
  std::vector<std::size_t> secondary_fmha_layers;
  bool secondary_fmha_provider_loaded = false;
  // Compatibility alias; new consumers should use fmha_provider_loaded.
  bool ck_provider_loaded = false;
  double raw_weight_load_wall_ms = 0.0;
  double derived_weight_build_wall_ms = 0.0;
  double lm_head_build_wall_ms = 0.0;
  double prefill_gemm_plan_build_wall_ms = 0.0;
  double command_to_ready_wall_ms = 0.0;
};

struct NativeResidentRequestOptions {
  std::vector<std::uint32_t> input_token_ids;
  std::size_t max_new_tokens = 1;
  std::vector<std::uint32_t> stop_token_ids;
  // Invoked synchronously as soon as each generated token is available.
  // Returning false cancels the remaining decode work without invalidating
  // resident model or prefix-cache state.
  std::function<bool(std::uint32_t, std::size_t)> token_callback;
  // Qualification-only provider-mask override. The resident provider pair
  // stays loaded; only this cold request's full-attention dispatch changes.
  bool secondary_fmha_layers_override_provided = false;
  std::vector<std::size_t> secondary_fmha_layers_override;
  // Qualification-only cold-run control for provider-mask sweeps.
  bool disable_prefix_cache = false;
  // Qualification-only tail-row references, one layer-XXX file per layer.
  std::filesystem::path layer_tail_oracle_dir;
  // 0..39 focuses the five attention boundaries plus final layer output.
  // 40 keeps the compact all-layer final-output diagnostic.
  std::size_t layer_tail_oracle_index = 40;
  // Qualification-only full-sequence attribution for one linear layer.  This
  // is kept separate from tail fixtures because each tensor is 128 MiB.
  std::filesystem::path layer_sequence_oracle_dir;
};

struct NativeResidentRequestMetrics {
  std::size_t request_index = 0;
  std::size_t prompt_tokens = 0;
  std::size_t completion_tokens = 0;
  std::vector<std::uint32_t> output_token_ids;
  std::string output_token_ids_sha256;
  bool stopped = false;
  bool client_cancelled = false;
  std::uint32_t stop_token_id = 0;
  std::size_t oracle_tensor_reads = 0;
  std::vector<NativeOracleComparison> layer_tail_comparisons;
  std::size_t model_loads = 0;
  std::size_t prefill_aot_launches = 0;
  std::size_t prefill_dense_gemm_launches = 0;
  std::size_t prefill_native_pointwise_launches = 0;
  std::size_t prefill_ck_fmha_launches = 0;
  std::size_t decode_tokens_executed = 0;
  std::size_t decode_aot_launches = 0;
  std::size_t decode_native_launches = 0;
  std::size_t state_orientation_resets = 0;
  std::string prefix_cache_lookup = "miss";
  std::size_t prefix_cache_matched_tokens = 0;
  std::size_t prefix_cache_suffix_tokens = 0;
  std::size_t prefix_cache_hits = 0;
  std::size_t prefix_cache_misses = 0;
  std::uint64_t prefix_cache_transfer_bytes = 0;
  std::size_t prefix_cache_suffix_decode_tokens = 0;
  std::size_t prefix_cache_suffix_aot_launches = 0;
  std::size_t prefix_cache_suffix_native_launches = 0;
  double prefix_cache_suffix_wall_ms = 0.0;
  bool first_token_certified = false;
  bool all_decode_tokens_certified = false;
  double prefill_wall_ms = 0.0;
  double prefill_tokens_per_second = 0.0;
  double decode_wall_ms = 0.0;
  double decode_tokens_per_second = 0.0;
  double request_wall_ms = 0.0;
};

// One model/device owner. All model weights, derived layouts, AOT modules,
// hipBLASLt plans, KV/state buffers and transient workspaces survive between
// run() calls. The initial specialization is exact batch-1 q8192; broader
// context schedules are admitted independently by the product matrix.
class NativeResidentEngine {
 public:
  NativeResidentEngine();
  ~NativeResidentEngine();
  NativeResidentEngine(const NativeResidentEngine&) = delete;
  NativeResidentEngine& operator=(const NativeResidentEngine&) = delete;

  NativeResidentLoadMetrics load(const NativeResidentEngineOptions& options);
  NativeResidentRequestMetrics run(
      const NativeResidentRequestOptions& request);
  // Qualification-only: compare the logits left by the most recent request
  // with an external FP32 full-vocabulary reference.  Normal serving never
  // reads an oracle and does not call this method.
  NativeLogitsComparison compare_current_logits(
      const std::filesystem::path& reference_path) const;
  bool loaded() const;
  std::size_t request_count() const;
  const NativeResidentLoadMetrics& load_metrics() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace aima
