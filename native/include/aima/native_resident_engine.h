#pragma once

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_layer_oracle.h"
#include "aima/native_mrope.h"
#include "aima/native_vl_embedding.h"
#include "aima/native_vl_processor.h"
#include "aima/native_weight_store.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
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
  // Optional hash-locked native vision-attention code object. Empty resolves
  // the artifact shipped beside the executable (or in the bundle lib dir).
  std::filesystem::path vision_attention_image;
  // Preferred AOT prefill schedule selected for this resident process.  This
  // is not a mandatory request length; variable prompts fall back to the
  // resident token path around the specialized prefix.
  std::size_t prompt_tokens = 8192;
  // Includes prompt and decoded cache entries.
  std::size_t cache_capacity = 9216;
  // Qualification and diagnostic processes that disable prefix caching for
  // every request can omit the otherwise resident cache backing. Product
  // servers retain the default so READY continues to cover the full cache
  // surface.
  bool prefix_cache_enabled = true;
};

struct NativeResidentLoadMetrics {
  std::string device_name;
  std::string gpu_arch;
  std::uint64_t model_payload_bytes = 0;
  std::size_t model_tensor_count = 0;
  std::size_t model_shard_count = 0;
  std::uint64_t language_model_payload_bytes = 0;
  std::size_t language_model_tensor_count = 0;
  std::size_t language_model_shard_count = 0;
  std::string language_layout_manifest_sha256;
  std::uint64_t visual_model_payload_bytes = 0;
  std::size_t visual_model_tensor_count = 0;
  std::size_t visual_model_shard_count = 0;
  std::string visual_layout_manifest_sha256;
  std::size_t decode_weight_bindings = 0;
  std::size_t prefill_prepared_launches = 0;
  std::size_t decode_prepared_launches = 0;
  std::size_t aot_loaded_modules = 0;
  std::size_t prefill_gemm_plans = 0;
  std::uint64_t prefill_workspace_bytes = 0;
  std::uint64_t mrope_position_state_bytes = 0;
  std::uint64_t vl_unified_attention_metadata_bytes = 0;
  std::uint64_t vl_unified_attention_decode_scratch_bytes = 0;
  std::size_t vl_unified_attention_image_bytes = 0;
  bool vl_unified_attention_loaded = false;
  std::uint64_t vl_logical_projection_weight_bytes = 0;
  std::uint64_t vl_logical_projection_output_scratch_bytes = 0;
  bool vl_logical_projection_weights_loaded = false;
  std::uint64_t vl_prompt_index_state_bytes = 0;
  std::uint64_t structured_token_mask_bytes = 0;
  std::size_t vision_plan_cache_capacity = 0;
  std::size_t vision_warmup_patches = 0;
  std::size_t vision_warmup_visual_tokens = 0;
  std::size_t vision_image_count_warmup_patches = 0;
  std::size_t vision_image_count_warmup_visual_tokens = 0;
  std::size_t vision_plan_cache_entries_at_ready = 0;
  double vision_warmup_plan_build_wall_ms = 0.0;
  double vision_warmup_encode_wall_ms = 0.0;
  double vision_image_count_warmup_plan_build_wall_ms = 0.0;
  double vision_image_count_warmup_encode_wall_ms = 0.0;
  bool vision_warmup_completed = false;
  std::uint64_t decode_workspace_bytes = 0;
  std::uint64_t attention_state_bytes = 0;
  std::uint64_t exact_prefix_cache_bytes = 0;
  std::size_t prefix_cache_entries = 0;
  std::size_t cache_capacity = 0;
  std::size_t prompt_tokens = 0;
  // Sorted fixed-shape AOT buckets kept resident for variable-length requests.
  // Long configured endpoints compose repeated q8192 and an admitted tail.
  std::vector<std::size_t> resident_prefill_buckets;
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
  double vl_logical_projection_weight_build_wall_ms = 0.0;
  double prefill_gemm_plan_build_wall_ms = 0.0;
  double command_to_ready_wall_ms = 0.0;
};

struct NativeResidentVlInput {
  std::vector<NativeVlGrid> grids;
  // Contiguous processor BF16 bits in [sum(patches),1536] order.
  std::vector<std::uint16_t> pixel_values_bf16;
  std::vector<NativeVlEmbeddingSpan> embedding_spans;
  std::string vision_embedding_cache_namespace;
  std::size_t media_count = 0;
  std::size_t image_count = 0;
  std::size_t video_count = 0;
  std::uint64_t source_bytes = 0;
  std::size_t media_cache_hits = 0;
  std::size_t media_cache_misses = 0;
  std::size_t media_cache_entries = 0;
  std::uint64_t media_cache_resident_bytes = 0;
  double media_load_wall_ms = 0.0;
  double media_decode_wall_ms = 0.0;
  double media_load_decode_wall_ms = 0.0;
  double processor_wall_ms = 0.0;
};

// Qualification-only synchronous observer for the logical prefill state
// handed to the first decode token. Linear layers expose their normalized
// convolution cache followed by their recurrent cache. Product requests leave
// this callback empty and incur no synchronization or host transfer.
using NativePrefillLinearStateObserver = std::function<void(
    std::size_t layer_index, const void* conv_state,
    std::uint64_t conv_state_bytes, const void* recurrent_state,
    std::uint64_t recurrent_state_bytes)>;

struct NativeResidentRequestOptions {
  std::vector<std::uint32_t> input_token_ids;
  // Empty for text-only requests. Native VL supplies a versioned digest of
  // ordered media content/config/spans; the cache compares this in addition
  // to input_token_ids so identical placeholder tokens cannot alias media.
  std::string multimodal_cache_namespace;
  // Exact host-resident Qwen3-VL positions for this prompt. The synchronous
  // request copies them into process-resident device storage; no request-time
  // device allocation is performed. Empty preserves the scalar text path.
  std::optional<NativeMropePlan> mrope_plan;
  // Empty keeps the text-only embedding path allocation-free. A populated
  // request is already decoded/processed on the host and is encoded by the
  // resident visual tower before prompt embeddings enter layer 0.
  std::optional<NativeResidentVlInput> vl_input;
  std::size_t max_new_tokens = 1;
  std::vector<std::uint32_t> stop_token_ids;
  // Optional fail-closed token grammar. The callback must replace `mask`
  // with exactly kNativeModelVocabularySize bytes for the next token, where
  // nonzero entries are admitted. The engine uploads this into resident
  // device storage before its certified LM-head selection.
  std::function<void(const std::vector<std::uint32_t>&,
                     std::vector<std::uint8_t>*)>
      next_token_mask;
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
  // Qualification-only logical prefill state observer. It runs after padded
  // state repair and before first-token selection.
  NativePrefillLinearStateObserver prefill_linear_state_observer;
  // Qualification-only decode boundary observer. Output index 0 is produced
  // by prefill, so the observer target must name a later generated token.
  std::optional<std::size_t> decode_layer_observer_output_index;
  NativeDecodeLayerObserver decode_layer_observer;
  // Qualification-only selected linear-layer observer. Layer zero preserves
  // the promotion-oracle default; diagnostics may select any linear layer.
  std::size_t decode_linear_observer_layer_index = 0;
  NativeDecodeLinearLayer0Observer decode_linear_layer0_observer;
  // The residual/post-attention/MoE callback stays separate to preserve the
  // frozen linear-attention oracle component set.
  NativeDecodeLinearLayer0Observer decode_layer0_tail_observer;
  // Qualification-only singleton full-attention observer. It shares the
  // exact output-index target above and exposes resident cache state without
  // introducing oracle reads into product execution.
  NativeDecodeFullAttentionObserver decode_full_attention_observer;
};

struct NativeResidentRequestMetrics {
  std::size_t request_index = 0;
  std::size_t prompt_tokens = 0;
  std::size_t completion_tokens = 0;
  std::vector<std::uint32_t> output_token_ids;
  std::string prompt_token_ids_sha256;
  std::string output_token_ids_sha256;
  std::string output_token_ids_canonical_sha256;
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
  std::size_t prefill_vl_unified_attention_launches = 0;
  bool vl_logical_projections_enabled = false;
  std::size_t vl_logical_projection_tokens = 0;
  std::size_t vl_logical_projection_plan_count = 0;
  std::uint64_t vl_logical_projection_workspace_bytes = 0;
  double vl_logical_projection_plan_build_wall_ms = 0.0;
  bool vl_logical_projection_plan_reused = false;
  bool vl_enabled = false;
  std::size_t vl_media_count = 0;
  std::size_t vl_image_count = 0;
  std::size_t vl_video_count = 0;
  std::uint64_t vl_source_bytes = 0;
  std::size_t vl_vision_patches = 0;
  std::size_t vl_visual_tokens = 0;
  std::size_t vl_media_cache_hits = 0;
  std::size_t vl_media_cache_misses = 0;
  std::size_t vl_media_cache_entries = 0;
  std::uint64_t vl_media_cache_resident_bytes = 0;
  std::size_t vl_vision_batch_count = 0;
  std::size_t vl_vision_max_batch_patches = 0;
  std::size_t vl_vision_max_batch_tokens = 0;
  bool vl_vision_plan_cache_hit = false;
  std::size_t vl_vision_plan_cache_entries = 0;
  bool vl_vision_embedding_cache_hit = false;
  std::size_t vl_vision_embedding_cache_entries = 0;
  std::uint64_t vl_vision_embedding_cache_resident_bytes = 0;
  std::uint64_t vl_vision_embedding_cache_capacity_bytes = 0;
  std::uint64_t vl_host_to_device_bytes = 0;
  double vl_media_load_wall_ms = 0.0;
  double vl_media_decode_wall_ms = 0.0;
  double vl_media_load_decode_wall_ms = 0.0;
  double vl_processor_wall_ms = 0.0;
  double vl_vision_plan_build_wall_ms = 0.0;
  double vl_vision_encode_wall_ms = 0.0;
  double vl_embedding_injection_wall_ms = 0.0;
  std::size_t decode_tokens_executed = 0;
  bool constrained_decoding = false;
  std::size_t constrained_token_selections = 0;
  std::uint64_t constrained_token_mask_upload_bytes = 0;
  std::size_t decode_aot_launches = 0;
  std::size_t decode_native_launches = 0;
  std::size_t state_orientation_resets = 0;
  // How the input prompt reached resident state. Variable lengths compose the
  // fixed resident AOT buckets and pad only the final segment when needed.
  std::string prompt_execution = "cold-aot";
  std::size_t aot_prefill_tokens = 0;
  std::size_t aot_prefill_bucket_tokens = 0;
  std::size_t aot_prefill_segments = 0;
  std::size_t padded_prefill_tokens = 0;
  std::size_t cold_prompt_decode_tokens = 0;
  std::size_t cold_prompt_decode_aot_launches = 0;
  std::size_t cold_prompt_decode_native_launches = 0;
  std::uint64_t request_state_reset_bytes = 0;
  bool mrope_enabled = false;
  std::int64_t mrope_position_delta = 0;
  std::uint64_t mrope_position_upload_bytes = 0;
  std::size_t mrope_full_attention_launches = 0;
  std::size_t mrope_decode_steps = 0;
  double cold_prompt_decode_wall_ms = 0.0;
  std::string prefix_cache_lookup = "miss";
  std::size_t prefix_cache_matched_tokens = 0;
  std::size_t prefix_cache_suffix_tokens = 0;
  std::size_t prefix_cache_hits = 0;
  std::size_t prefix_cache_misses = 0;
  std::uint64_t prefix_cache_transfer_bytes = 0;
  bool prefix_cache_active_kv_reused = false;
  double prefix_cache_restore_wall_ms = 0.0;
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
// run() calls. The fast-path specialization is batch-1 q8192; variable prompt
// lengths compose the resident fixed-shape prefill schedules, while broader
// context schedules are qualified independently.
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
