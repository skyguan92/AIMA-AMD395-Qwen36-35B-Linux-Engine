// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/aot_kernel.h"
#include "aima/bf16_wvsplitk.h"
#include "aima/native_weight_store.h"
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
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#ifndef AIMA_SOURCE_COMMIT
#define AIMA_SOURCE_COMMIT "unknown"
#endif

namespace {

using json = nlohmann::json;

constexpr std::size_t kHidden = 2048;
constexpr std::size_t kExperts = 256;
constexpr std::size_t kTopK = 8;
constexpr std::size_t kIntermediate = 512;
constexpr std::size_t kGateUp = 2 * kIntermediate;
constexpr unsigned kThreads = 256;

void check_hip(hipError_t status, const char* operation) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(status));
  }
}

class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t bytes) : bytes_(bytes) {
    check_hip(hipMalloc(&pointer_, bytes_), "hipMalloc routed MoE probe");
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
    throw std::runtime_error("routed MoE component path is not relative");
  }
  for (const auto& component : relative) {
    if (component == "..") {
      throw std::runtime_error("routed MoE component path escapes its root");
    }
  }
  return root / relative;
}

std::vector<unsigned char> read_component(
    const std::filesystem::path& root, const json& record,
    std::size_t expected_bytes, std::string_view expected_dtype) {
  if (record.value("bytes", 0ULL) != expected_bytes ||
      record.value("dtype", "") != expected_dtype) {
    throw std::runtime_error("routed MoE component geometry changed");
  }
  const std::filesystem::path path = checked_component_path(
      root, record.at("path").get<std::string>());
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream || stream.tellg() !=
                     static_cast<std::streamoff>(expected_bytes)) {
    throw std::runtime_error("routed MoE component size mismatch: " +
                             path.string());
  }
  std::vector<unsigned char> bytes(expected_bytes);
  stream.seekg(0);
  if (expected_bytes != 0 &&
      !stream.read(reinterpret_cast<char*>(bytes.data()),
                   static_cast<std::streamsize>(bytes.size()))) {
    throw std::runtime_error("routed MoE component read failed: " +
                             path.string());
  }
  if (aima::sha256_bytes(bytes.data(), bytes.size()) !=
      record.at("sha256").get<std::string>()) {
    throw std::runtime_error("routed MoE component SHA-256 mismatch: " +
                             path.string());
  }
  return bytes;
}

void upload(const std::vector<unsigned char>& host, DeviceBuffer& device) {
  if (host.size() != device.bytes()) {
    throw std::invalid_argument("routed MoE upload byte count changed");
  }
  check_hip(hipMemcpy(device.get(), host.data(), host.size(),
                      hipMemcpyHostToDevice),
            "hipMemcpy routed MoE input");
}

json compare_device(const DeviceBuffer& actual,
                    const std::vector<unsigned char>& expected,
                    std::size_t element_bytes) {
  if (actual.bytes() != expected.size() || expected.empty() ||
      expected.size() % element_bytes != 0) {
    throw std::invalid_argument("routed MoE comparison geometry changed");
  }
  std::vector<unsigned char> host(actual.bytes());
  check_hip(hipMemcpy(host.data(), actual.get(), host.size(),
                      hipMemcpyDeviceToHost),
            "hipMemcpy routed MoE output");
  const std::size_t elements = host.size() / element_bytes;
  std::size_t exact = 0;
  std::size_t first = elements;
  for (std::size_t index = 0; index < elements; ++index) {
    if (std::memcmp(host.data() + index * element_bytes,
                    expected.data() + index * element_bytes,
                    element_bytes) == 0) {
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

const aima::NativeTensorView& require_weight(
    const aima::NativeWeightStore& weights, const std::string& name,
    std::uint64_t bytes, std::initializer_list<std::uint32_t> shape) {
  const aima::NativeTensorView* view = weights.find(name);
  if (view == nullptr || view->device_pointer == nullptr ||
      view->payload_bytes != bytes || view->rank != shape.size()) {
    throw std::runtime_error("routed MoE weight is unavailable: " + name);
  }
  std::size_t index = 0;
  for (const std::uint32_t dimension : shape) {
    if (view->shape[index++] != dimension) {
      throw std::runtime_error("routed MoE weight shape changed: " + name);
    }
  }
  return *view;
}

__global__ void router_topk8_softmax_256_kernel(
    const __hip_bfloat16* logits, float* weights, std::int32_t* indices) {
  constexpr int kRouterWave = 32;
  constexpr int kValuesPerLane = kExperts / kRouterWave;
  const int lane = threadIdx.x;
  float row_chunk[kValuesPerLane];
#pragma unroll
  for (int value = 0; value < kValuesPerLane; ++value) {
    row_chunk[value] =
        __bfloat162float(logits[lane * kValuesPerLane + value]);
  }

  float thread_max = row_chunk[0];
#pragma unroll
  for (int value = 1; value < kValuesPerLane; ++value) {
    thread_max = fmaxf(thread_max, row_chunk[value]);
  }
#pragma unroll
  for (int mask = kRouterWave / 2; mask > 0; mask /= 2) {
    thread_max = fmaxf(thread_max,
                       __shfl_xor(thread_max, mask, kRouterWave));
  }

  float row_sum = 0.0f;
#pragma unroll
  for (int value = 0; value < kValuesPerLane; ++value) {
    row_chunk[value] = expf(row_chunk[value] - thread_max);
    row_sum += row_chunk[value];
  }
#pragma unroll
  for (int mask = kRouterWave / 2; mask > 0; mask /= 2) {
    row_sum += __shfl_xor(row_sum, mask, kRouterWave);
  }
  const float reciprocal_row_sum = 1.0f / row_sum;
#pragma unroll
  for (int value = 0; value < kValuesPerLane; ++value) {
    row_chunk[value] *= reciprocal_row_sum;
  }

  float selected_probabilities[kTopK];
  int selected_indices[kTopK];
  float selected_sum = 0.0f;
  for (int rank = 0; rank < static_cast<int>(kTopK); ++rank) {
    float max_probability = row_chunk[0];
    int expert = lane * kValuesPerLane;
#pragma unroll
    for (int value = 0; value < kValuesPerLane; ++value) {
      const float candidate = row_chunk[value];
      if (candidate > max_probability) {
        max_probability = candidate;
        expert = lane * kValuesPerLane + value;
      }
    }
#pragma unroll
    for (int mask = kRouterWave / 2; mask > 0; mask /= 2) {
      const float other_probability =
          __shfl_xor(max_probability, mask, kRouterWave);
      const int other_expert = __shfl_xor(expert, mask, kRouterWave);
      if (other_probability > max_probability ||
          (other_probability == max_probability && other_expert < expert)) {
        max_probability = other_probability;
        expert = other_expert;
      }
    }
    if (lane == 0) {
      selected_probabilities[rank] = max_probability;
      selected_indices[rank] = expert;
      selected_sum += max_probability;
    }
    if (expert / kValuesPerLane == lane) {
      row_chunk[expert % kValuesPerLane] = -10000.0f;
    }
  }

  if (lane == 0) {
    const float denominator = selected_sum > 0.0f ? selected_sum : 1.0f;
#pragma unroll
    for (std::size_t rank = 0; rank < kTopK; ++rank) {
      indices[rank] = selected_indices[rank];
      weights[rank] = selected_probabilities[rank] / denominator;
    }
  }
}

__global__ void routed_silu_multiply_kernel(
    const __hip_bfloat16* gate_up, __hip_bfloat16* activated) {
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= kTopK * kIntermediate) return;
  const std::size_t row = index / kIntermediate;
  const std::size_t column = index - row * kIntermediate;
  const std::size_t base = row * kGateUp;
  const float gate = __bfloat162float(gate_up[base + column]);
  const float up =
      __bfloat162float(gate_up[base + kIntermediate + column]);
  const float silu = gate / (1.0f + expf(-gate));
  const __hip_bfloat16 silu_bf16 = __float2bfloat16(silu);
  activated[index] =
      __float2bfloat16(__bfloat162float(silu_bf16) * up);
}

__global__ void routed_sum8_kernel(const __hip_bfloat16* input,
                                   __hip_bfloat16* output) {
  const std::size_t hidden = blockIdx.x * blockDim.x + threadIdx.x;
  if (hidden >= kHidden) return;
  const float sum04 = __bfloat162float(input[hidden]) +
                      __bfloat162float(input[4 * kHidden + hidden]);
  const float sum15 = __bfloat162float(input[kHidden + hidden]) +
                      __bfloat162float(input[5 * kHidden + hidden]);
  const float sum26 = __bfloat162float(input[2 * kHidden + hidden]) +
                      __bfloat162float(input[6 * kHidden + hidden]);
  const float sum37 = __bfloat162float(input[3 * kHidden + hidden]) +
                      __bfloat162float(input[7 * kHidden + hidden]);
  float sum = sum04 + sum15;
  sum += sum26;
  sum += sum37;
  output[hidden] = __float2bfloat16(sum);
}

void launch_router(const void* logits, void* weights, void* indices) {
  hipLaunchKernelGGL(router_topk8_softmax_256_kernel, dim3(1), dim3(32), 0,
                     nullptr,
                     static_cast<const __hip_bfloat16*>(logits),
                     static_cast<float*>(weights),
                     static_cast<std::int32_t*>(indices));
  check_hip(hipGetLastError(), "router_topk8_softmax_256_kernel");
}

void launch_activation(const void* gate_up, void* activated) {
  constexpr std::size_t elements = kTopK * kIntermediate;
  hipLaunchKernelGGL(
      routed_silu_multiply_kernel,
      dim3(static_cast<unsigned>((elements + kThreads - 1) / kThreads)),
      dim3(kThreads), 0, nullptr,
      static_cast<const __hip_bfloat16*>(gate_up),
      static_cast<__hip_bfloat16*>(activated));
  check_hip(hipGetLastError(), "routed_silu_multiply_kernel");
}

void launch_sum(const void* input, void* output) {
  hipLaunchKernelGGL(
      routed_sum8_kernel,
      dim3(static_cast<unsigned>((kHidden + kThreads - 1) / kThreads)),
      dim3(kThreads), 0, nullptr,
      static_cast<const __hip_bfloat16*>(input),
      static_cast<__hip_bfloat16*>(output));
  check_hip(hipGetLastError(), "routed_sum8_kernel");
}

void launch_fused_moe(aima::AotKernel& kernel,
                      const aima::AotLaunchConfig& config, void* activation,
                      void* weight, void* output, void* topk_weights,
                      void* expert_ids, void* num_tokens_post_padded,
                      std::int32_t n, std::int32_t k,
                      std::int32_t stride_be, std::int32_t stride_bn,
                      std::int32_t stride_am, std::int32_t stride_cm) {
  std::int32_t em = 128;
  std::int32_t num_valid_tokens = 8;
  std::int32_t zero = 0;
  std::vector<void*> parameters = {
      &activation,
      &weight,
      &output,
      &topk_weights,
      &expert_ids,
      &num_tokens_post_padded,
      &n,
      &k,
      &em,
      &num_valid_tokens,
      &stride_am,
      &stride_be,
      &stride_bn,
      &stride_cm,
      &zero,
      &zero,
      &zero,
      &zero,
      &zero,
      &zero,
      &zero,
  };
  kernel.launch(config, parameters);
}

const json& find_case(const json& manifest, const std::string& case_id) {
  if (manifest.value("schema", "") !=
          "aima-amd395-qwen36/vl-generation-layer-diagnostic/v1" ||
      !manifest.value("complete", false) ||
      !manifest.contains("cases") || !manifest.at("cases").is_array()) {
    throw std::runtime_error("routed MoE diagnostic manifest is incomplete");
  }
  for (const json& value : manifest.at("cases")) {
    if (value.value("case_id", "") == case_id) return value;
  }
  throw std::runtime_error("routed MoE diagnostic case was not found");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 9) {
    std::cerr << "usage: native-routed-moe-decode-aot-probe MODEL_DIR "
                 "LOAD_REPORT GATE_UP_IMAGE DOWN_IMAGE CAPTURE_MANIFEST "
                 "CAPTURE_ROOT CASE_ID TAIL_SET\n";
    return 2;
  }
  try {
    const std::filesystem::path model_dir =
        std::filesystem::absolute(argv[1]);
    const std::filesystem::path load_report =
        std::filesystem::absolute(argv[2]);
    const std::filesystem::path gate_up_image =
        std::filesystem::absolute(argv[3]);
    const std::filesystem::path down_image =
        std::filesystem::absolute(argv[4]);
    const std::filesystem::path manifest_path =
        std::filesystem::absolute(argv[5]);
    const std::filesystem::path capture_root =
        std::filesystem::absolute(argv[6]);
    const std::string case_id = argv[7];
    const std::string tail_set = argv[8];
    if (tail_set != "layer0_tail" &&
        tail_set != "first_decode_layer0_tail") {
      throw std::runtime_error("routed MoE tail set is unsupported");
    }

    const json manifest = read_json(manifest_path);
    const json& case_record = find_case(manifest, case_id);
    const json& components = case_record.at(tail_set).at("components");
    const auto component = [&](const char* name, std::size_t bytes,
                               const char* dtype) {
      return read_component(capture_root, components.at(name), bytes, dtype);
    };
    const std::vector<unsigned char> post_attention_norm = component(
        "post_attention_norm", kHidden * sizeof(std::uint16_t),
        "torch.bfloat16");
    const std::vector<unsigned char> expected_router_logits = component(
        "router_logits", kExperts * sizeof(std::uint16_t),
        "torch.bfloat16");
    const std::vector<unsigned char> expected_router_weights = component(
        "router_weights", kTopK * sizeof(float), "torch.float32");
    const std::vector<unsigned char> expected_router_indices = component(
        "router_indices", kTopK * sizeof(std::int32_t), "torch.int32");
    const std::vector<unsigned char> expected_gate_up = component(
        "routed_gate_up_projection",
        kTopK * kGateUp * sizeof(std::uint16_t), "torch.bfloat16");
    const std::vector<unsigned char> expected_activation = component(
        "routed_activation", kTopK * kIntermediate * sizeof(std::uint16_t),
        "torch.bfloat16");
    const std::vector<unsigned char> expected_weighted = component(
        "routed_weighted_expert_outputs",
        kTopK * kHidden * sizeof(std::uint16_t), "torch.bfloat16");
    const std::vector<unsigned char> expected_sum = component(
        "routed_moe_output", kHidden * sizeof(std::uint16_t),
        "torch.bfloat16");

    aima::NativeWeightLoadOptions load_options;
    load_options.model_dir = model_dir;
    load_options.native_report = load_report;
    load_options.worker_count = 4;
    aima::NativeWeightStore weights;
    const aima::NativeWeightLoadMetrics load = weights.load(load_options);
    const std::string prefix = "model.language_model.layers.0.mlp.";
    const auto& router_weight = require_weight(
        weights, prefix + "gate.weight", 1048576ULL, {256, 2048});
    const auto& gate_up_weight = require_weight(
        weights, prefix + "experts.gate_up_proj", 1073741824ULL,
        {256, 1024, 2048});
    const auto& down_weight = require_weight(
        weights, prefix + "experts.down_proj", 536870912ULL,
        {256, 2048, 512});

    hipDeviceProp_t properties{};
    check_hip(hipGetDeviceProperties(&properties, 0),
              "hipGetDeviceProperties routed MoE probe");
    if (std::string(properties.gcnArchName).rfind("gfx1151", 0) != 0) {
      throw std::runtime_error("routed MoE probe requires gfx1151");
    }

    DeviceBuffer hidden(post_attention_norm.size());
    DeviceBuffer router_logits(expected_router_logits.size());
    DeviceBuffer router_weights(expected_router_weights.size());
    DeviceBuffer router_indices(expected_router_indices.size());
    DeviceBuffer seeded_router_weights(expected_router_weights.size());
    DeviceBuffer seeded_router_indices(expected_router_indices.size());
    DeviceBuffer num_tokens_post_padded(sizeof(std::int32_t));
    DeviceBuffer gate_up(expected_gate_up.size());
    DeviceBuffer activation(expected_activation.size());
    DeviceBuffer weighted(expected_weighted.size());
    DeviceBuffer sum(expected_sum.size());
    DeviceBuffer end_to_end_gate_up(expected_gate_up.size());
    DeviceBuffer end_to_end_activation(expected_activation.size());
    DeviceBuffer end_to_end_weighted(expected_weighted.size());
    DeviceBuffer end_to_end_sum(expected_sum.size());
    upload(post_attention_norm, hidden);
    upload(expected_router_weights, seeded_router_weights);
    upload(expected_router_indices, seeded_router_indices);
    const std::int32_t padded = 128;
    check_hip(hipMemcpy(num_tokens_post_padded.get(), &padded, sizeof(padded),
                        hipMemcpyHostToDevice),
              "hipMemcpy routed MoE padded count");

    aima::launch_bf16_wvsplitk(
        router_weight.device_pointer, hidden.get(), nullptr,
        router_logits.get(), kExperts, kHidden,
        properties.multiProcessorCount);
    launch_router(router_logits.get(), router_weights.get(),
                  router_indices.get());

    aima::AotKernel gate_up_kernel = aima::AotKernel::from_file(
        gate_up_image, "fused_moe_kernel");
    aima::AotKernel down_kernel = aima::AotKernel::from_file(
        down_image, "fused_moe_kernel");
    launch_fused_moe(
        gate_up_kernel, aima::AotLaunchConfig{512, 1, 1, 4, 32, 16384},
        hidden.get(), gate_up_weight.device_pointer, gate_up.get(),
        seeded_router_weights.get(), seeded_router_indices.get(),
        num_tokens_post_padded.get(), 1024, 2048, 2097152, 2048, 2048,
        1024);
    launch_activation(gate_up.get(), activation.get());
    launch_fused_moe(
        down_kernel, aima::AotLaunchConfig{1024, 1, 1, 4, 32, 16384},
        activation.get(), down_weight.device_pointer, weighted.get(),
        seeded_router_weights.get(), seeded_router_indices.get(),
        num_tokens_post_padded.get(), 2048, 512, 1048576, 512, 512, 2048);
    launch_sum(weighted.get(), sum.get());
    launch_fused_moe(
        gate_up_kernel, aima::AotLaunchConfig{512, 1, 1, 4, 32, 16384},
        hidden.get(), gate_up_weight.device_pointer, end_to_end_gate_up.get(),
        router_weights.get(), router_indices.get(),
        num_tokens_post_padded.get(), 1024, 2048, 2097152, 2048, 2048,
        1024);
    launch_activation(end_to_end_gate_up.get(), end_to_end_activation.get());
    launch_fused_moe(
        down_kernel, aima::AotLaunchConfig{1024, 1, 1, 4, 32, 16384},
        end_to_end_activation.get(), down_weight.device_pointer,
        end_to_end_weighted.get(), router_weights.get(),
        router_indices.get(), num_tokens_post_padded.get(), 2048, 512,
        1048576, 512, 512, 2048);
    launch_sum(end_to_end_weighted.get(), end_to_end_sum.get());
    check_hip(hipDeviceSynchronize(),
              "hipDeviceSynchronize routed MoE probe");

    json comparisons = {
        {"router_logits",
         compare_device(router_logits, expected_router_logits,
                        sizeof(std::uint16_t))},
        {"router_weights",
         compare_device(router_weights, expected_router_weights,
                        sizeof(float))},
        {"router_indices",
         compare_device(router_indices, expected_router_indices,
                        sizeof(std::int32_t))},
        {"routed_gate_up_projection",
         compare_device(gate_up, expected_gate_up, sizeof(std::uint16_t))},
        {"routed_activation",
         compare_device(activation, expected_activation,
                        sizeof(std::uint16_t))},
        {"routed_weighted_expert_outputs",
         compare_device(weighted, expected_weighted,
                        sizeof(std::uint16_t))},
        {"routed_moe_output",
         compare_device(sum, expected_sum, sizeof(std::uint16_t))},
        {"end_to_end_routed_gate_up_projection",
         compare_device(end_to_end_gate_up, expected_gate_up,
                        sizeof(std::uint16_t))},
        {"end_to_end_routed_activation",
         compare_device(end_to_end_activation, expected_activation,
                        sizeof(std::uint16_t))},
        {"end_to_end_routed_weighted_expert_outputs",
         compare_device(end_to_end_weighted, expected_weighted,
                        sizeof(std::uint16_t))},
        {"end_to_end_routed_moe_output",
         compare_device(end_to_end_sum, expected_sum,
                        sizeof(std::uint16_t))},
    };
    bool complete = true;
    for (const auto& [name, comparison] : comparisons.items()) {
      static_cast<void>(name);
      complete = complete && comparison.at("bit_exact").get<bool>();
    }
    const json result = {
        {"schema",
         "aima-amd395-qwen36/routed-moe-decode-aot-model-probe/v1"},
        {"complete", complete},
        {"source_commit", AIMA_SOURCE_COMMIT},
        {"capture_manifest_sha256", aima::sha256_file(manifest_path)},
        {"capture_source_commit", manifest.at("source").at("commit")},
        {"case_id", case_id},
        {"tail_set", tail_set},
        {"gpu_arch", properties.gcnArchName},
        {"cu_count", properties.multiProcessorCount},
        {"language_weight_payload_bytes", load.payload_bytes},
        {"gate_up_image_sha256", aima::sha256_file(gate_up_image)},
        {"down_image_sha256", aima::sha256_file(down_image)},
        {"expert_aot_seeded_from_reference_router", true},
        {"router_evaluated_from_reference_hidden_state", true},
        {"end_to_end_router_outputs_consumed", true},
        {"comparisons", std::move(comparisons)},
        {"decision",
         {{"model_weight_numerical_closure", complete},
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
    std::cerr << "routed MoE decode AOT probe: " << error.what() << '\n';
    return 1;
  }
}
