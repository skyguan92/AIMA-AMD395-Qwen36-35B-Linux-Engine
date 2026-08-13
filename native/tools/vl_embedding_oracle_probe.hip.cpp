// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_multimodal_cache.h"
#include "aima/native_vl_embedding.h"
#include "aima/native_weight_store.h"
#include "aima/sha256.h"

#include <hip/hip_runtime.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using json = nlohmann::json;
constexpr std::size_t kLanguageHidden = 2048;

void check_hip(hipError_t status, const char* operation) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(status));
  }
}

class DeviceAllocation {
 public:
  explicit DeviceAllocation(std::size_t bytes) {
    if (bytes == 0) {
      throw std::invalid_argument("native VL probe allocation is empty");
    }
    check_hip(hipMalloc(&pointer_, bytes), "hipMalloc VL embedding tensor");
  }
  ~DeviceAllocation() {
    if (pointer_ != nullptr) {
      const hipError_t ignored = hipFree(pointer_);
      static_cast<void>(ignored);
    }
  }
  DeviceAllocation(const DeviceAllocation&) = delete;
  DeviceAllocation& operator=(const DeviceAllocation&) = delete;
  void* get() const { return pointer_; }

 private:
  void* pointer_ = nullptr;
};

class Event {
 public:
  Event() { check_hip(hipEventCreate(&event_), "hipEventCreate"); }
  ~Event() {
    if (event_ != nullptr) {
      const hipError_t ignored = hipEventDestroy(event_);
      static_cast<void>(ignored);
    }
  }
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;
  operator hipEvent_t() const { return event_; }

 private:
  hipEvent_t event_ = nullptr;
};

json read_json(const std::filesystem::path& path) {
  std::ifstream stream(path);
  if (!stream) {
    throw std::runtime_error("JSON file is unavailable: " + path.string());
  }
  json value;
  stream >> value;
  if (!value.is_object()) {
    throw std::runtime_error("JSON root is not an object: " + path.string());
  }
  return value;
}

std::filesystem::path checked_oracle_path(
    const std::filesystem::path& root, std::string_view relative_text) {
  const std::filesystem::path relative(relative_text);
  if (relative.empty() || relative.is_absolute()) {
    throw std::runtime_error("oracle tensor path is not relative");
  }
  for (const auto& component : relative) {
    if (component == "..") {
      throw std::runtime_error("oracle tensor path escapes its root");
    }
  }
  return root / relative;
}

std::vector<unsigned char> read_tensor_record(
    const std::filesystem::path& root, const json& record) {
  const std::size_t expected_bytes = record.at("bytes").get<std::size_t>();
  const std::filesystem::path path = checked_oracle_path(
      root, record.at("path").get<std::string>());
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream || stream.tellg() < 0 ||
      static_cast<std::size_t>(stream.tellg()) != expected_bytes) {
    throw std::runtime_error("oracle tensor size mismatch: " + path.string());
  }
  std::vector<unsigned char> bytes(expected_bytes);
  stream.seekg(0);
  if (expected_bytes != 0 &&
      !stream.read(reinterpret_cast<char*>(bytes.data()),
                   static_cast<std::streamsize>(bytes.size()))) {
    throw std::runtime_error("oracle tensor read failed: " + path.string());
  }
  if (aima::sha256_bytes(bytes.data(), bytes.size()) !=
      record.at("sha256").get<std::string>()) {
    throw std::runtime_error("oracle tensor SHA-256 mismatch: " +
                             path.string());
  }
  return bytes;
}

void write_file(const std::filesystem::path& path,
                const std::vector<unsigned char>& bytes) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream ||
      !stream.write(reinterpret_cast<const char*>(bytes.data()),
                    static_cast<std::streamsize>(bytes.size()))) {
    throw std::runtime_error("VL embedding output write failed: " +
                             path.string());
  }
}

float bf16_to_float(std::uint16_t bits) {
  const std::uint32_t value = static_cast<std::uint32_t>(bits) << 16U;
  float result = 0.0f;
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

struct Comparison {
  std::size_t elements = 0;
  std::size_t exact_elements = 0;
  std::size_t finite_elements = 0;
  std::size_t first_mismatch_index = std::numeric_limits<std::size_t>::max();
  std::uint16_t first_expected_bits = 0;
  std::uint16_t first_actual_bits = 0;
  double maximum_absolute_error = 0.0;
  double relative_l2_error = 0.0;
  double cosine_similarity = 0.0;
  std::string expected_sha256;
  std::string actual_sha256;
};

Comparison compare_bf16(const std::vector<unsigned char>& actual,
                        const std::vector<unsigned char>& expected) {
  if (actual.size() != expected.size() || actual.size() % 2 != 0) {
    throw std::invalid_argument("VL embedding comparison sizes differ");
  }
  Comparison result;
  result.elements = actual.size() / sizeof(std::uint16_t);
  result.expected_sha256 = aima::sha256_bytes(expected.data(), expected.size());
  result.actual_sha256 = aima::sha256_bytes(actual.data(), actual.size());
  double squared_error = 0.0;
  double squared_expected = 0.0;
  double squared_actual = 0.0;
  double dot = 0.0;
  for (std::size_t index = 0; index < result.elements; ++index) {
    std::uint16_t expected_bits = 0;
    std::uint16_t actual_bits = 0;
    std::memcpy(&expected_bits,
                expected.data() + index * sizeof(expected_bits),
                sizeof(expected_bits));
    std::memcpy(&actual_bits, actual.data() + index * sizeof(actual_bits),
                sizeof(actual_bits));
    if (expected_bits == actual_bits) {
      ++result.exact_elements;
    } else if (result.first_mismatch_index ==
               std::numeric_limits<std::size_t>::max()) {
      result.first_mismatch_index = index;
      result.first_expected_bits = expected_bits;
      result.first_actual_bits = actual_bits;
    }
    const double expected_value = bf16_to_float(expected_bits);
    const double actual_value = bf16_to_float(actual_bits);
    if (std::isfinite(actual_value)) ++result.finite_elements;
    const double error = actual_value - expected_value;
    result.maximum_absolute_error =
        std::max(result.maximum_absolute_error, std::abs(error));
    squared_error += error * error;
    squared_expected += expected_value * expected_value;
    squared_actual += actual_value * actual_value;
    dot += expected_value * actual_value;
  }
  result.relative_l2_error =
      std::sqrt(squared_error / std::max(squared_expected, 1.0e-30));
  result.cosine_similarity =
      dot / std::sqrt(std::max(squared_expected * squared_actual, 1.0e-30));
  return result;
}

struct PendingSpan {
  aima::NativeMediaKind kind = aima::NativeMediaKind::kImage;
  std::size_t offset = 0;
  std::size_t length = 0;
  std::size_t embeddings = 0;
  const json* record = nullptr;
};

std::uint32_t placeholder_token(aima::NativeMediaKind kind) {
  return kind == aima::NativeMediaKind::kImage
             ? aima::kNativeImagePadTokenId
             : aima::kNativeVideoPadTokenId;
}

std::vector<std::uint32_t> prompt_token_ids(const json& case_record) {
  const json& values = case_record.at("processor").at("prompt_token_ids");
  if (!values.is_array() || values.empty()) {
    throw std::runtime_error("oracle prompt token ids are empty");
  }
  std::vector<std::uint32_t> result;
  result.reserve(values.size());
  for (const json& value : values) {
    const std::uint64_t token = value.get<std::uint64_t>();
    if (token > std::numeric_limits<std::uint32_t>::max()) {
      throw std::runtime_error("oracle prompt token id exceeds uint32");
    }
    result.push_back(static_cast<std::uint32_t>(token));
  }
  return result;
}

std::vector<aima::NativeVlEmbeddingSpan> embedding_spans(
    const json& case_record, const std::vector<std::uint32_t>& tokens,
    std::size_t* visual_embedding_count) {
  const json& placeholders = case_record.at("processor").at("placeholders");
  std::vector<PendingSpan> pending;
  for (const auto& descriptor :
       {std::pair<const char*, aima::NativeMediaKind>{
            "image", aima::NativeMediaKind::kImage},
        std::pair<const char*, aima::NativeMediaKind>{
            "video", aima::NativeMediaKind::kVideo}}) {
    if (!placeholders.contains(descriptor.first)) continue;
    for (const json& record : placeholders.at(descriptor.first)) {
      pending.push_back(PendingSpan{
          descriptor.second, record.at("offset").get<std::size_t>(),
          record.at("length").get<std::size_t>(),
          record.at("num_embeds").get<std::size_t>(), &record});
    }
  }
  std::sort(pending.begin(), pending.end(),
            [](const PendingSpan& left, const PendingSpan& right) {
              return left.offset < right.offset;
            });
  std::vector<aima::NativeVlEmbeddingSpan> spans;
  std::size_t source_offset = 0;
  for (const PendingSpan& value : pending) {
    if (value.record == nullptr || value.offset > tokens.size() ||
        value.length > tokens.size() - value.offset) {
      throw std::runtime_error("oracle placeholder span is invalid");
    }
    const std::uint32_t expected = placeholder_token(value.kind);
    if (value.record->contains("is_embed")) {
      const json& mask = value.record->at("is_embed");
      if (!mask.is_array() || mask.size() != value.length) {
        throw std::runtime_error("oracle is_embed mask length is invalid");
      }
      for (std::size_t index = 0; index < value.length; ++index) {
        const bool derived = tokens[value.offset + index] == expected;
        if (mask.at(index).get<bool>() != derived) {
          throw std::runtime_error(
              "oracle is_embed mask differs from placeholder token ids");
        }
      }
    } else {
      for (std::size_t index = 0; index < value.length; ++index) {
        if (tokens[value.offset + index] != expected) {
          throw std::runtime_error(
              "unmasked oracle span contains a non-placeholder token");
        }
      }
    }
    spans.push_back(aima::NativeVlEmbeddingSpan{
        value.kind, value.offset, value.length, source_offset,
        value.embeddings});
    source_offset += value.embeddings;
  }
  *visual_embedding_count = source_offset;
  return spans;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 7) {
    std::cerr <<
        "usage: native-vl-embedding-probe MODEL_DIR ORACLE_MANIFEST "
        "ORACLE_ROOT CASE_ID LOAD_REPORT ACTUAL_OUTPUT\n";
    return 2;
  }
  try {
    const std::filesystem::path manifest_path =
        std::filesystem::absolute(argv[2]);
    const std::filesystem::path oracle_root =
        std::filesystem::absolute(argv[3]);
    const std::string case_id = argv[4];
    const json manifest = read_json(manifest_path);
    if (manifest.value("schema", "") !=
            "aima-amd395-qwen36/vl-oracle-manifest/v1" ||
        !manifest.value("complete", false) ||
        !manifest.contains("cases") || !manifest.at("cases").is_array()) {
      throw std::runtime_error("VL oracle manifest is incomplete");
    }
    const json* case_record = nullptr;
    for (const json& value : manifest.at("cases")) {
      if (value.value("case_id", "") == case_id) {
        if (case_record != nullptr) {
          throw std::runtime_error("VL oracle case id is duplicated");
        }
        case_record = &value;
      }
    }
    if (case_record == nullptr) {
      throw std::runtime_error("VL oracle case id was not found");
    }

    const std::vector<std::uint32_t> tokens = prompt_token_ids(*case_record);
    std::size_t visual_embedding_count = 0;
    const std::vector<aima::NativeVlEmbeddingSpan> spans = embedding_spans(
        *case_record, tokens, &visual_embedding_count);
    const aima::NativeVlEmbeddingPlan plan =
        aima::build_native_vl_embedding_plan(tokens, spans,
                                             visual_embedding_count);
    const json& boundaries = case_record->at("boundaries");
    const json& visual_record = boundaries.at("vision_merger");
    const json& expected_record = boundaries.at("injected_embeddings");
    if (visual_record.at("shape") !=
            json::array({visual_embedding_count, kLanguageHidden}) ||
        expected_record.at("shape") !=
            json::array({tokens.size(), kLanguageHidden}) ||
        visual_record.value("dtype", "") != "torch.bfloat16" ||
        expected_record.value("dtype", "") != "torch.bfloat16") {
      throw std::runtime_error("VL embedding oracle shape/dtype is invalid");
    }
    const std::vector<unsigned char> visual =
        read_tensor_record(oracle_root, visual_record);
    const std::vector<unsigned char> expected =
        read_tensor_record(oracle_root, expected_record);
    if (visual.size() != visual_embedding_count * kLanguageHidden * 2 ||
        expected.size() != tokens.size() * kLanguageHidden * 2) {
      throw std::runtime_error("VL embedding oracle byte count is invalid");
    }

    aima::NativeWeightLoadOptions options;
    options.model_dir = std::filesystem::absolute(argv[1]);
    options.native_report = std::filesystem::absolute(argv[5]);
    aima::NativeWeightStore weights;
    const aima::NativeWeightLoadMetrics load = weights.load(options);
    const aima::NativeTensorView* embedding =
        weights.find("model.language_model.embed_tokens.weight");
    if (embedding == nullptr || embedding->device_pointer == nullptr ||
        embedding->rank != 2 || embedding->shape[0] != 248320 ||
        embedding->shape[1] != kLanguageHidden ||
        embedding->payload_bytes != 248320ULL * kLanguageHidden * 2) {
      throw std::runtime_error("language token embedding weight is invalid");
    }

    DeviceAllocation visual_device(visual.size());
    DeviceAllocation prompt_ids_device(tokens.size() * sizeof(std::uint32_t));
    DeviceAllocation scatter_indices(plan.device_index_bytes());
    DeviceAllocation output_device(expected.size());
    check_hip(hipMemcpy(visual_device.get(), visual.data(), visual.size(),
                        hipMemcpyHostToDevice),
              "hipMemcpy visual merger rows");
    aima::launch_native_vl_embeddings(
        embedding->device_pointer, tokens.data(), plan, visual_device.get(),
        prompt_ids_device.get(), scatter_indices.get(), output_device.get());
    check_hip(hipDeviceSynchronize(),
              "hipDeviceSynchronize VL embedding warmup");
    std::vector<unsigned char> first(expected.size());
    check_hip(hipMemcpy(first.data(), output_device.get(), first.size(),
                        hipMemcpyDeviceToHost),
              "hipMemcpy first VL embeddings");

    Event start;
    Event stop;
    std::vector<float> measured_ms;
    measured_ms.reserve(5);
    for (std::size_t repetition = 0; repetition < 5; ++repetition) {
      check_hip(hipEventRecord(start), "hipEventRecord VL embedding start");
      aima::launch_native_vl_embeddings(
          embedding->device_pointer, tokens.data(), plan, visual_device.get(),
          prompt_ids_device.get(), scatter_indices.get(), output_device.get());
      check_hip(hipEventRecord(stop), "hipEventRecord VL embedding stop");
      check_hip(hipEventSynchronize(stop),
                "hipEventSynchronize VL embedding stop");
      float milliseconds = 0.0f;
      check_hip(hipEventElapsedTime(&milliseconds, start, stop),
                "hipEventElapsedTime VL embedding");
      measured_ms.push_back(milliseconds);
    }
    std::vector<unsigned char> actual(expected.size());
    check_hip(hipMemcpy(actual.data(), output_device.get(), actual.size(),
                        hipMemcpyDeviceToHost),
              "hipMemcpy VL embeddings");
    write_file(std::filesystem::absolute(argv[6]), actual);
    const Comparison comparison = compare_bf16(actual, expected);
    std::vector<float> sorted_ms = measured_ms;
    std::sort(sorted_ms.begin(), sorted_ms.end());
    const bool deterministic = first == actual;
    const bool exact = comparison.exact_elements == comparison.elements &&
                       comparison.finite_elements == comparison.elements;

    json result = {
        {"schema",
         "aima-amd395-qwen36/native-vl-embedding-oracle/v1"},
        {"complete", exact && deterministic},
        {"case_id", case_id},
        {"prompt_tokens", plan.prompt_token_count()},
        {"visual_embeddings", plan.visual_embedding_count()},
        {"span_count", spans.size()},
        {"device_index_bytes", plan.device_index_bytes()},
        {"language_weight_payload_bytes", load.payload_bytes},
        {"measured_ms", measured_ms},
        {"median_ms", sorted_ms[sorted_ms.size() / 2]},
        {"elements", comparison.elements},
        {"exact_elements", comparison.exact_elements},
        {"finite_elements", comparison.finite_elements},
        {"first_mismatch_index",
         comparison.first_mismatch_index ==
                 std::numeric_limits<std::size_t>::max()
             ? -1LL
             : static_cast<long long>(comparison.first_mismatch_index)},
        {"first_expected_bits", comparison.first_expected_bits},
        {"first_actual_bits", comparison.first_actual_bits},
        {"maximum_absolute_error", comparison.maximum_absolute_error},
        {"relative_l2_error", comparison.relative_l2_error},
        {"cosine_similarity", comparison.cosine_similarity},
        {"expected_sha256", comparison.expected_sha256},
        {"actual_sha256", comparison.actual_sha256},
        {"repeat_actual_sha256",
         aima::sha256_bytes(first.data(), first.size())},
        {"repeat_deterministic", deterministic},
        {"bit_exact", exact},
        {"prompt_token_ids_sha256",
         case_record->at("processor").at("prompt_token_ids_sha256")},
        {"vision_merger_sha256", visual_record.at("sha256")},
    };
    std::cout << result.dump() << '\n';
    return exact && deterministic ? 0 : 3;
  } catch (const std::exception& error) {
    std::cerr << "native VL embedding probe: " << error.what() << '\n';
    return 1;
  }
}
