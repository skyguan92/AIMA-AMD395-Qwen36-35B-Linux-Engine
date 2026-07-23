// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_decode_bindings.h"

#include <hip/hip_runtime.h>

#include <charconv>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace aima {
namespace {

enum class Owner {
  kRaw,
  kLayerDerived,
  kLmHeadDerived,
};

struct Resolved {
  void* pointer = nullptr;
  std::uint64_t bytes = 0;
  Owner owner = Owner::kRaw;
};

std::pair<int, std::string_view> parse_layer_binding(std::string_view binding) {
  constexpr std::string_view prefix = "layer_weights.";
  constexpr std::string_view marker = ".tensors.";
  if (binding.size() < prefix.size() ||
      binding.substr(0, prefix.size()) != prefix) {
    throw std::runtime_error("decode layer binding has an invalid prefix");
  }
  binding.remove_prefix(prefix.size());
  const std::size_t marker_position = binding.find(marker);
  if (marker_position == std::string_view::npos) {
    throw std::runtime_error("decode layer binding has no tensor marker");
  }
  int layer = -1;
  const std::string_view layer_text = binding.substr(0, marker_position);
  const auto parsed = std::from_chars(
      layer_text.data(), layer_text.data() + layer_text.size(), layer);
  if (parsed.ec != std::errc() || parsed.ptr != layer_text.data() + layer_text.size() ||
      layer < 0 || layer >= 40) {
    throw std::runtime_error("decode layer binding has an invalid layer index");
  }
  return {layer, binding.substr(marker_position + marker.size())};
}

std::string layer_prefix(int layer) {
  return "model.language_model.layers." + std::to_string(layer);
}

Resolved raw(const NativeWeightStore& weights, const std::string& name) {
  const NativeTensorView* view = weights.find(name);
  if (view == nullptr) {
    throw std::runtime_error("decode raw weight binding is absent: " + name);
  }
  return {view->device_pointer, view->payload_bytes, Owner::kRaw};
}

Resolved layer_derived(const NativeDerivedWeightStore& derived,
                       const std::string& name) {
  const NativeDerivedWeightView* view = derived.find(name);
  if (view == nullptr) {
    throw std::runtime_error("decode derived weight binding is absent: " + name);
  }
  return {view->device_pointer, view->payload_bytes, Owner::kLayerDerived};
}

Resolved resolve_layer(const NativeWeightStore& weights,
                       const NativeDerivedWeightStore& derived,
                       int layer, std::string_view key) {
  const std::string base = layer_prefix(layer);
  if (key == "input_layernorm") return raw(weights, base + ".input_layernorm.weight");
  if (key == "post_attention_layernorm") {
    return raw(weights, base + ".post_attention_layernorm.weight");
  }
  if (key == "expert_gate_up") return raw(weights, base + ".mlp.experts.gate_up_proj");
  if (key == "expert_down") return raw(weights, base + ".mlp.experts.down_proj");
  if (key == "q_norm") return raw(weights, base + ".self_attn.q_norm.weight");
  if (key == "k_norm") return raw(weights, base + ".self_attn.k_norm.weight");
  if (key == "linear_A_log") return raw(weights, base + ".linear_attn.A_log");
  if (key == "linear_conv1d_direct_weight") {
    return raw(weights, base + ".linear_attn.conv1d.weight");
  }
  if (key == "linear_dt_bias") return raw(weights, base + ".linear_attn.dt_bias");
  if (key == "linear_norm") return raw(weights, base + ".linear_attn.norm.weight");
  const std::string derived_prefix = "layer" + std::to_string(layer) + ".";
  if (key == "linear_input_proj_fused_t") {
    return layer_derived(derived, derived_prefix + "linear_input_fused_t");
  }
  if (key == "full_qkv_proj_fused_t") {
    return layer_derived(derived, derived_prefix + "full_qkv_fused_t");
  }
  if (key == "shared_input_proj_fused_t") {
    return layer_derived(derived, derived_prefix + "shared_input_fused_t");
  }
  if (key == "router_t") {
    return layer_derived(derived, derived_prefix + "router_t");
  }
  throw std::runtime_error("unsupported native decode layer binding key: " +
                           std::string(key));
}

Resolved resolve(const NativeWeightStore& weights,
                 const NativeDerivedWeightStore& derived,
                 const NativeLmHeadStore& lm_head,
                 std::string_view binding) {
  if (binding == "global_tensors.final_norm") {
    return raw(weights, "model.language_model.norm.weight");
  }
  if (binding == "global_tensors.lm_head_int8") {
    return {lm_head.q_weight(), 508559360ULL, Owner::kLmHeadDerived};
  }
  if (binding == "global_tensors.lm_head_int8_scales") {
    return {lm_head.scales(), 993280ULL, Owner::kLmHeadDerived};
  }
  const auto [layer, key] = parse_layer_binding(binding);
  return resolve_layer(weights, derived, layer, key);
}

void validate_device_pointer(void* pointer) {
  if (pointer == nullptr) {
    throw std::runtime_error("native decode binding resolved to null");
  }
  hipPointerAttribute_t attributes{};
  const hipError_t status = hipPointerGetAttributes(&attributes, pointer);
  if (status != hipSuccess || attributes.type != hipMemoryTypeDevice ||
      attributes.devicePointer != pointer) {
    throw std::runtime_error("native decode binding is not a HIP device pointer");
  }
}

}  // namespace

NativeDecodeBindingMetrics NativeDecodeBindings::build(
    const NativeWeightStore& weights,
    const NativeDerivedWeightStore& derived,
    const NativeLmHeadStore& lm_head) {
  if (!views_.empty() || !name_to_index_.empty() || !weights.loaded() ||
      !derived.built() || !lm_head.built()) {
    throw std::runtime_error("native decode bindings require fresh complete owners");
  }
  NativeDecodeBindingMetrics metrics;
  std::size_t launch_count = 0;
  const DecodeLaunch* launches = native_decode_schedule(&launch_count);
  views_.reserve(423);
  name_to_index_.reserve(423);
  for (std::size_t launch_index = 0; launch_index < launch_count; ++launch_index) {
    const DecodeLaunch& launch = launches[launch_index];
    for (std::size_t argument_index = 0;
         argument_index < launch.argument_count; ++argument_index) {
      const DecodeArgument& argument = launch.arguments[argument_index];
      if (argument.kind != DecodeArgumentKind::kTensor ||
          argument.binding_kind != DecodeBindingKind::kModelOrDerivedWeight) {
        continue;
      }
      ++metrics.schedule_weight_arguments;
      const std::string name(argument.binding);
      if (name_to_index_.find(name) != name_to_index_.end()) {
        throw std::runtime_error("duplicate qualified decode weight binding: " + name);
      }
      const Resolved value = resolve(weights, derived, lm_head, name);
      validate_device_pointer(value.pointer);
      ++metrics.device_pointer_checks;
      if (value.bytes != argument.storage_bytes || argument.byte_offset != 0) {
        throw std::runtime_error("decode weight binding payload geometry mismatch: " + name);
      }
      ++metrics.exact_payload_byte_checks;
      switch (value.owner) {
        case Owner::kRaw: ++metrics.raw_weight_bindings; break;
        case Owner::kLayerDerived: ++metrics.layer_derived_bindings; break;
        case Owner::kLmHeadDerived: ++metrics.lm_head_derived_bindings; break;
      }
      const std::size_t index = views_.size();
      views_.push_back({name, value.pointer, value.bytes, argument.tensor_dtype});
      name_to_index_.emplace(name, index);
    }
  }
  metrics.unique_bindings = views_.size();
  if (metrics.schedule_weight_arguments != 423 ||
      metrics.unique_bindings != 423 || metrics.raw_weight_bindings != 301 ||
      metrics.layer_derived_bindings != 120 ||
      metrics.lm_head_derived_bindings != 2 ||
      metrics.device_pointer_checks != 423 ||
      metrics.exact_payload_byte_checks != 423) {
    throw std::runtime_error("native decode binding closure count mismatch");
  }
  return metrics;
}

const NativeDecodeBindingView* NativeDecodeBindings::find(
    std::string_view name) const {
  const auto found = name_to_index_.find(std::string(name));
  return found == name_to_index_.end() ? nullptr : &views_[found->second];
}

}  // namespace aima
