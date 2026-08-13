// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_weight_store.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace {

constexpr std::size_t kTensorCount = 333;
constexpr std::uint64_t kPayloadBytes = 893142496ULL;
constexpr char kManifestSha256[] =
    "abc5b3a0cc0881ba2d3e815b472eebe3404a6e3bc6438a430faccfbe8093c0aa";
constexpr std::size_t kResidentTensorCount = 1026;
constexpr std::uint64_t kResidentPayloadBytes = 70214363872ULL;
constexpr char kResidentManifestSha256[] =
    "b8a9f4f909b66104f1815d9ed49791c8692077455a517f2d4e8f0defe6893dd7";

const aima::NativeTensorView& require_tensor(
    const aima::NativeWeightStore& weights, const char* name,
    std::uint8_t rank, std::uint32_t dimension0,
    std::uint32_t dimension1 = 1) {
  const aima::NativeTensorView* tensor = weights.find(name);
  if (tensor == nullptr || tensor->device_pointer == nullptr ||
      tensor->rank != rank || tensor->shape[0] != dimension0 ||
      tensor->shape[1] != dimension1) {
    throw std::runtime_error(std::string("visual tensor contract failed: ") +
                             name);
  }
  return *tensor;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3 && argc != 4) {
    std::cerr << "usage: native-visual-weight-probe MODEL_DIR REPORT "
                 "[--resident]\n";
    return 2;
  }
  try {
    const bool resident = argc == 4 && std::string(argv[3]) == "--resident";
    if (argc == 4 && !resident) {
      throw std::runtime_error("only --resident is accepted as a probe mode");
    }
    aima::NativeWeightLoadOptions options;
    options.model_dir = std::filesystem::absolute(argv[1]);
    options.native_report = std::filesystem::absolute(argv[2]);
    aima::NativeWeightStore weights;
    const aima::NativeWeightLoadMetrics metrics =
        resident ? weights.load_resident(options) : weights.load_visual(options);
    const std::size_t expected_tensors =
        resident ? kResidentTensorCount : kTensorCount;
    const std::uint64_t expected_payload =
        resident ? kResidentPayloadBytes : kPayloadBytes;
    const std::size_t expected_shards = resident ? 26 : 2;
    const char* expected_manifest =
        resident ? kResidentManifestSha256 : kManifestSha256;
    if (metrics.weight_set != (resident ? "language+visual" : "visual") ||
        metrics.layout_manifest_sha256 != expected_manifest ||
        metrics.tensor_count != expected_tensors ||
        metrics.payload_bytes != expected_payload ||
        metrics.shard_count != expected_shards ||
        metrics.visual_layout_manifest_sha256 != kManifestSha256 ||
        metrics.visual_tensor_count != kTensorCount ||
        metrics.visual_payload_bytes != kPayloadBytes ||
        weights.tensors().size() != expected_tensors ||
        !std::filesystem::is_regular_file(options.native_report)) {
      throw std::runtime_error("visual weight-store metrics are incomplete");
    }
    if (resident &&
        (metrics.language_tensor_count != 693 ||
         metrics.language_payload_bytes != 69321221376ULL ||
         metrics.language_shard_count != 26)) {
      throw std::runtime_error("resident language weight metrics are incomplete");
    }

    std::uint64_t payload_bytes = 0;
    std::unordered_set<void*> pointers;
    for (const aima::NativeTensorView& tensor : weights.tensors()) {
      payload_bytes += tensor.payload_bytes;
      pointers.insert(tensor.device_pointer);
    }
    if (payload_bytes != expected_payload ||
        pointers.size() != expected_tensors) {
      throw std::runtime_error("visual weight-store ownership is incomplete");
    }
    if (resident) {
      (void)require_tensor(
          weights, "model.language_model.embed_tokens.weight", 2,
          248320, 2048);
    }
    (void)require_tensor(weights, "model.visual.blocks.0.attn.qkv.weight",
                         2, 3456, 1152);
    (void)require_tensor(weights, "model.visual.blocks.26.mlp.linear_fc2.weight",
                         2, 1152, 4304);
    const aima::NativeTensorView& patch = require_tensor(
        weights, "model.visual.patch_embed.proj.weight", 5, 1152, 3);
    if (patch.shape[2] != 2 || patch.shape[3] != 16 ||
        patch.shape[4] != 16) {
      throw std::runtime_error("visual patch tensor rank-5 shape is invalid");
    }
    (void)require_tensor(weights, "model.visual.pos_embed.weight", 2,
                         2304, 1152);
    (void)require_tensor(weights,
                         "model.visual.merger.linear_fc1.weight", 2,
                         4608, 4608);

    std::cout << "{\"schema\":\"aima-amd395-qwen36/"
                 "native-visual-weight-probe/v1\","
              << "\"complete\":true,"
              << "\"mode\":\"" << (resident ? "resident" : "visual")
              << "\","
              << "\"device_name\":\"" << metrics.device_name << "\","
              << "\"gpu_arch\":\"" << metrics.gpu_arch << "\","
              << "\"layout_manifest_sha256\":\""
              << metrics.layout_manifest_sha256 << "\","
              << "\"tensor_count\":" << metrics.tensor_count << ','
              << "\"unique_device_pointers\":" << pointers.size() << ','
              << "\"payload_bytes\":" << metrics.payload_bytes << ','
              << "\"shard_count\":" << metrics.shard_count << ','
              << "\"allocation_ms\":" << metrics.allocation_ms << ','
              << "\"ingest_ms\":" << metrics.ingest_ms << ','
              << "\"load_wall_ms\":" << metrics.load_wall_ms << "}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "native visual weight probe: " << error.what() << '\n';
    return 1;
  }
}
