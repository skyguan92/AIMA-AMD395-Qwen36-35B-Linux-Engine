// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/aot_kernel.h"
#include "aima/sha256.h"

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#ifndef AIMA_SOURCE_COMMIT
#define AIMA_SOURCE_COMMIT "unknown"
#endif

namespace {

using json = nlohmann::json;

constexpr std::size_t kQueryHeads = 16;
constexpr std::size_t kKvHeads = 2;
constexpr std::size_t kHeadSize = 256;
constexpr std::size_t kCacheBlockTokens = 1056;
constexpr std::size_t kSoftmaxSegments = 16;
constexpr std::size_t kSequenceThreshold3d = 64;

void check_hip(hipError_t status, const char* operation) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(status));
  }
}

class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t bytes) : bytes_(bytes) {
    if (bytes_ == 0) {
      throw std::invalid_argument("unified-attention buffer cannot be empty");
    }
    check_hip(hipMalloc(&pointer_, bytes_),
              "hipMalloc unified-attention probe");
  }
  ~DeviceBuffer() {
    if (pointer_ != nullptr) static_cast<void>(hipFree(pointer_));
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  void* get() const { return pointer_; }
  std::size_t bytes() const { return bytes_; }

 private:
  void* pointer_ = nullptr;
  std::size_t bytes_ = 0;
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

std::filesystem::path checked_component_path(
    const std::filesystem::path& root, std::string_view relative_text) {
  const std::filesystem::path relative(relative_text);
  if (relative.empty() || relative.is_absolute()) {
    throw std::runtime_error(
        "unified-attention component path is not relative");
  }
  for (const auto& component : relative) {
    if (component == "..") {
      throw std::runtime_error(
          "unified-attention component path escapes its root");
    }
  }
  return root / relative;
}

std::vector<unsigned char> read_component(
    const std::filesystem::path& root, const json& record,
    const json& expected_shape, std::string_view expected_dtype,
    std::size_t expected_bytes) {
  if (record.value("shape", json::array()) != expected_shape ||
      record.value("dtype", "") != expected_dtype ||
      record.value("bytes", 0ULL) != expected_bytes) {
    throw std::runtime_error(
        "unified-attention captured component geometry changed");
  }
  const std::filesystem::path path = checked_component_path(
      root, record.at("path").get<std::string>());
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream || stream.tellg() !=
                     static_cast<std::streamoff>(expected_bytes)) {
    throw std::runtime_error(
        "unified-attention component size mismatch: " + path.string());
  }
  std::vector<unsigned char> bytes(expected_bytes);
  stream.seekg(0);
  if (!stream.read(reinterpret_cast<char*>(bytes.data()),
                   static_cast<std::streamsize>(bytes.size()))) {
    throw std::runtime_error(
        "unified-attention component read failed: " + path.string());
  }
  if (aima::sha256_bytes(bytes.data(), bytes.size()) !=
      record.at("sha256").get<std::string>()) {
    throw std::runtime_error(
        "unified-attention component SHA-256 mismatch: " + path.string());
  }
  return bytes;
}

void upload(const std::vector<unsigned char>& host, DeviceBuffer& device) {
  if (host.size() != device.bytes()) {
    throw std::invalid_argument(
        "unified-attention upload byte count changed");
  }
  check_hip(hipMemcpy(device.get(), host.data(), host.size(),
                      hipMemcpyHostToDevice),
            "hipMemcpy unified-attention input");
}

json compare_device(const DeviceBuffer& actual,
                    const std::vector<unsigned char>& expected) {
  if (actual.bytes() != expected.size() ||
      expected.size() % sizeof(std::uint16_t) != 0) {
    throw std::invalid_argument(
        "unified-attention comparison geometry changed");
  }
  std::vector<unsigned char> host(actual.bytes());
  check_hip(hipMemcpy(host.data(), actual.get(), host.size(),
                      hipMemcpyDeviceToHost),
            "hipMemcpy unified-attention output");
  const std::size_t elements = host.size() / sizeof(std::uint16_t);
  std::size_t exact = 0;
  std::size_t first = elements;
  for (std::size_t index = 0; index < elements; ++index) {
    if (std::memcmp(host.data() + index * sizeof(std::uint16_t),
                    expected.data() + index * sizeof(std::uint16_t),
                    sizeof(std::uint16_t)) == 0) {
      ++exact;
    } else if (first == elements) {
      first = index;
    }
  }
  json result = {
      {"elements", elements},
      {"exact_elements", exact},
      {"expected_sha256",
       aima::sha256_bytes(expected.data(), expected.size())},
      {"actual_sha256", aima::sha256_bytes(host.data(), host.size())},
      {"bit_exact", exact == elements},
  };
  result["first_mismatch_index"] =
      first == elements ? json(nullptr) : json(first);
  return result;
}

const json& find_case(const json& manifest, const std::string& case_id) {
  if (manifest.value("schema", "") !=
          "aima-amd395-qwen36/vl-generation-layer-diagnostic/v1" ||
      !manifest.value("complete", false) ||
      !manifest.value("qualified_for_decode_attribution", false) ||
      !manifest.contains("cases") || !manifest.at("cases").is_array()) {
    throw std::runtime_error(
        "unified-attention diagnostic manifest is incomplete");
  }
  for (const json& value : manifest.at("cases")) {
    if (value.value("case_id", "") == case_id) return value;
  }
  throw std::runtime_error(
      "unified-attention diagnostic case was not found");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 7) {
    std::cerr <<
        "usage: native-unified-attention-decode-aot-probe ATTENTION_IMAGE "
        "REDUCE_IMAGE CAPTURE_MANIFEST CAPTURE_ROOT CASE_ID ATTENTION_SET\n";
    return 2;
  }
  try {
    const std::filesystem::path attention_image_path =
        std::filesystem::absolute(argv[1]);
    const std::filesystem::path reduce_image_path =
        std::filesystem::absolute(argv[2]);
    const std::filesystem::path manifest_path =
        std::filesystem::absolute(argv[3]);
    const std::filesystem::path capture_root =
        std::filesystem::absolute(argv[4]);
    const std::string case_id = argv[5];
    const std::string attention_set = argv[6];
    if (attention_set != "full_attention" &&
        attention_set != "first_decode_full_attention") {
      throw std::runtime_error(
          "unified-attention capture set is unsupported");
    }

    const json manifest = read_json(manifest_path);
    const json& case_record = find_case(manifest, case_id);
    const json& captured = case_record.at(attention_set);
    const json& metadata = captured.at("metadata");
    if (metadata.value("layer_index", -1) != 3 ||
        metadata.value("block_size", 0ULL) != kCacheBlockTokens ||
        metadata.value("query_heads", 0ULL) != kQueryHeads ||
        metadata.value("kv_heads", 0ULL) != kKvHeads ||
        metadata.value("head_size", 0ULL) != kHeadSize ||
        metadata.value("max_seqlen_q", 0ULL) != 1 ||
        metadata.value("causal", false) != true ||
        metadata.value("softmax_scale", 0.0) != 0.0625 ||
        metadata.value("softcap", -1.0) != 0.0 ||
        metadata.value("sequence_threshold_3d", 0ULL) !=
            kSequenceThreshold3d ||
        metadata.value("softmax_segments", 0ULL) != kSoftmaxSegments ||
        metadata.value("attention_path", "") !=
            "segmented_3d_plus_reduce" ||
        metadata.value("window_size", json::array()) !=
            json::array({-1, -1})) {
      throw std::runtime_error(
          "unified-attention captured metadata changed");
    }
    const std::size_t sequence_length =
        metadata.at("sequence_length").get<std::size_t>();
    const std::size_t logical_blocks =
        metadata.at("logical_blocks").get<std::size_t>();
    if (sequence_length == 0 ||
        logical_blocks !=
            (sequence_length + kCacheBlockTokens - 1) / kCacheBlockTokens) {
      throw std::runtime_error(
          "unified-attention captured block count changed");
    }

    const json& components = captured.at("components");
    constexpr std::size_t query_bytes =
        kQueryHeads * kHeadSize * sizeof(std::uint16_t);
    const std::size_t cache_elements =
        logical_blocks * kCacheBlockTokens * kKvHeads * kHeadSize;
    const std::size_t cache_bytes = cache_elements * sizeof(std::uint16_t);
    const auto component = [&](const char* name, const json& shape,
                               const char* dtype, std::size_t bytes) {
      return read_component(capture_root, components.at(name), shape, dtype,
                            bytes);
    };
    const std::vector<unsigned char> query = component(
        "query", json::array({1, kQueryHeads, kHeadSize}),
        "torch.bfloat16", query_bytes);
    const std::vector<unsigned char> key_cache = component(
        "key_cache",
        json::array(
            {logical_blocks, kCacheBlockTokens, kKvHeads, kHeadSize}),
        "torch.bfloat16", cache_bytes);
    const std::vector<unsigned char> value_cache = component(
        "value_cache",
        json::array(
            {logical_blocks, kCacheBlockTokens, kKvHeads, kHeadSize}),
        "torch.bfloat16", cache_bytes);
    const std::vector<unsigned char> block_table = component(
        "block_table", json::array({1, logical_blocks}), "torch.int32",
        logical_blocks * sizeof(std::int32_t));
    const std::vector<unsigned char> sequence_lengths = component(
        "sequence_lengths", json::array({1}), "torch.int32",
        sizeof(std::int32_t));
    const std::vector<unsigned char> query_starts = component(
        "query_starts", json::array({2}), "torch.int32",
        2 * sizeof(std::int32_t));
    const std::vector<unsigned char> k_descale = component(
        "k_descale", json::array({1, kKvHeads}), "torch.float32",
        kKvHeads * sizeof(float));
    const std::vector<unsigned char> v_descale = component(
        "v_descale", json::array({1, kKvHeads}), "torch.float32",
        kKvHeads * sizeof(float));
    const std::vector<unsigned char> expected_output = component(
        "output", json::array({1, kQueryHeads, kHeadSize}),
        "torch.bfloat16", query_bytes);
    for (const std::vector<unsigned char>* values : {&k_descale, &v_descale}) {
      for (std::size_t index = 0; index < kKvHeads; ++index) {
        float value = 0.0F;
        std::memcpy(&value, values->data() + index * sizeof(float),
                    sizeof(float));
        if (value != 1.0F) {
          throw std::runtime_error(
              "unified-attention captured descale changed");
        }
      }
    }

    hipDeviceProp_t properties{};
    check_hip(hipGetDeviceProperties(&properties, 0),
              "hipGetDeviceProperties unified-attention probe");
    if (std::string(properties.gcnArchName).rfind("gfx1151", 0) != 0) {
      throw std::runtime_error(
          "unified-attention probe requires gfx1151");
    }

    DeviceBuffer query_device(query.size());
    DeviceBuffer key_device(key_cache.size());
    DeviceBuffer value_device(value_cache.size());
    DeviceBuffer block_table_device(block_table.size());
    DeviceBuffer sequence_lengths_device(sequence_lengths.size());
    DeviceBuffer query_starts_device(query_starts.size());
    DeviceBuffer k_descale_device(k_descale.size());
    DeviceBuffer v_descale_device(v_descale.size());
    DeviceBuffer segment_output_device(
        kSequenceThreshold3d * kQueryHeads * kSoftmaxSegments *
        kHeadSize * sizeof(float));
    DeviceBuffer segment_max_device(
        kSequenceThreshold3d * kQueryHeads * kSoftmaxSegments *
        sizeof(float));
    DeviceBuffer segment_expsum_device(
        kSequenceThreshold3d * kQueryHeads * kSoftmaxSegments *
        sizeof(float));
    DeviceBuffer output_device(expected_output.size());
    DeviceBuffer repeat_device(expected_output.size());
    upload(query, query_device);
    upload(key_cache, key_device);
    upload(value_cache, value_device);
    upload(block_table, block_table_device);
    upload(sequence_lengths, sequence_lengths_device);
    upload(query_starts, query_starts_device);
    upload(k_descale, k_descale_device);
    upload(v_descale, v_descale_device);

    aima::AotKernel attention_kernel = aima::AotKernel::from_file(
        attention_image_path, "kernel_unified_attention_3d");
    aima::AotKernel reduce_kernel =
        aima::AotKernel::from_file(reduce_image_path, "reduce_segments");
    const auto launch = [&](DeviceBuffer& output) {
      void* output_pointer = output.get();
      void* segment_output_pointer = segment_output_device.get();
      void* segment_max_pointer = segment_max_device.get();
      void* segment_expsum_pointer = segment_expsum_device.get();
      void* query_pointer = query_device.get();
      void* key_pointer = key_device.get();
      void* value_pointer = value_device.get();
      void* block_table_pointer = block_table_device.get();
      void* sequence_lengths_pointer = sequence_lengths_device.get();
      void* query_starts_pointer = query_starts_device.get();
      void* k_descale_pointer = k_descale_device.get();
      void* v_descale_pointer = v_descale_device.get();
      float scale = 0.0625F;
      float softcap = 0.0F;
      float out_scale_inverse = 1.0F;
      std::int64_t block_table_stride =
          static_cast<std::int64_t>(logical_blocks);
      std::int64_t query_stride_0 = kQueryHeads * kHeadSize;
      std::int64_t query_stride_1 = kHeadSize;
      std::int64_t output_stride_0 = kQueryHeads * kHeadSize;
      std::int64_t output_stride_1 = kHeadSize;
      std::int64_t qq_bias_stride_0 = 0;
      std::int64_t cache_stride_0 =
          kCacheBlockTokens * kKvHeads * kHeadSize;
      std::int64_t cache_stride_1 = kKvHeads * kHeadSize;
      std::int64_t cache_stride_2 = kHeadSize;
      std::int32_t num_seqs = 1;
      std::int32_t zero_stride = 0;
      std::vector<void*> attention_parameters{
          &segment_output_pointer,
          &segment_max_pointer,
          &segment_expsum_pointer,
          &query_pointer,
          &key_pointer,
          &value_pointer,
          &block_table_pointer,
          &sequence_lengths_pointer,
          &scale,
          &k_descale_pointer,
          &v_descale_pointer,
          &softcap,
          &block_table_stride,
          &query_stride_0,
          &query_stride_1,
          &qq_bias_stride_0,
          &cache_stride_0,
          &cache_stride_1,
          &cache_stride_2,
          &cache_stride_0,
          &cache_stride_1,
          &cache_stride_2,
          &query_starts_pointer,
          &num_seqs,
          &zero_stride,
          &zero_stride,
          &zero_stride,
          &zero_stride,
          &zero_stride,
          &zero_stride,
      };
      if (attention_parameters.size() != 30) {
        throw std::runtime_error(
            "segmented attention regular ABI argument count changed");
      }
      attention_kernel.launch(
          aima::AotLaunchConfig{1, 2, 16, 4, 32, 16384},
          attention_parameters);
      check_hip(hipDeviceSynchronize(),
                "hipDeviceSynchronize segmented attention probe");
      std::vector<void*> reduce_parameters{
          &output_pointer,
          &segment_output_pointer,
          &segment_max_pointer,
          &segment_expsum_pointer,
          &sequence_lengths_pointer,
          &out_scale_inverse,
          &output_stride_0,
          &output_stride_1,
          &block_table_stride,
          &query_starts_pointer,
      };
      if (reduce_parameters.size() != 10) {
        throw std::runtime_error(
            "attention reduce regular ABI argument count changed");
      }
      reduce_kernel.launch(
          aima::AotLaunchConfig{1, 16, 1, 4, 32, 2048},
          reduce_parameters);
      check_hip(hipDeviceSynchronize(),
                "hipDeviceSynchronize attention reduce probe");
    };
    launch(output_device);
    launch(repeat_device);
    check_hip(hipDeviceSynchronize(),
              "hipDeviceSynchronize unified-attention probe");

    const json output_comparison =
        compare_device(output_device, expected_output);
    const json repeat_comparison =
        compare_device(repeat_device, expected_output);
    const bool complete = output_comparison.at("bit_exact").get<bool>() &&
                          repeat_comparison.at("bit_exact").get<bool>();
    const json result = {
        {"schema",
         "aima-amd395-qwen36/unified-attention-decode-aot-model-probe/v1"},
        {"complete", complete},
        {"source_commit", AIMA_SOURCE_COMMIT},
        {"capture_manifest_sha256", aima::sha256_file(manifest_path)},
        {"capture_source_commit", manifest.at("source").at("commit")},
        {"case_id", case_id},
        {"attention_set", attention_set},
        {"sequence_length", sequence_length},
        {"logical_blocks", logical_blocks},
        {"gpu_arch", properties.gcnArchName},
        {"attention_image_sha256",
         aima::sha256_file(attention_image_path)},
        {"reduce_image_sha256", aima::sha256_file(reduce_image_path)},
        {"launches",
         {{{"symbol", "kernel_unified_attention_3d"},
           {"grid", {1, 2, 16}},
           {"num_warps", 4},
           {"warp_size", 32},
           {"shared_memory_bytes", 16384}},
          {{"symbol", "reduce_segments"},
           {"grid", {1, 16, 1}},
           {"num_warps", 4},
           {"warp_size", 32},
           {"shared_memory_bytes", 2048}}}},
        {"comparisons", {{"output", output_comparison},
                         {"repeat_output", repeat_comparison}}},
        {"decision",
         {{"model_tensor_numerical_closure", complete},
          {"qualified_for_native_decode_replacement", false},
          {"promotion_result", false},
          {"g1_passed", false},
          {"g2_passed", false},
          {"g3_passed", false},
          {"g4_passed", false},
          {"g5_passed", false}}},
    };
    std::cout << result.dump(2) << '\n';
    return complete ? 0 : 3;
  } catch (const std::exception& error) {
    std::cerr << "unified-attention decode AOT probe: " << error.what()
              << '\n';
    return 1;
  }
}
