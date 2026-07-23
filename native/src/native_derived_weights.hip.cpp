// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_derived_weights.h"

#include "aima/bf16_gemm.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace aima {
namespace {

constexpr std::uint32_t kHidden = 2048;
constexpr std::uint32_t kLinearFused = 8192 + 4096 + 32 + 32;
constexpr std::uint32_t kFullFused = 8192 + 512 + 512;
constexpr std::uint32_t kSharedFused = 1 + 512 + 512;
constexpr std::uint32_t kTile = 16;

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

__global__ void transpose_into_fused_kernel(
    const hip_bfloat16* source, hip_bfloat16* destination,
    std::uint32_t source_rows, std::uint32_t source_columns,
    std::uint32_t destination_columns, std::uint32_t destination_column_offset) {
  __shared__ hip_bfloat16 tile[kTile][kTile + 1];
  const std::uint32_t source_column = blockIdx.x * kTile + threadIdx.x;
  const std::uint32_t source_row = blockIdx.y * kTile + threadIdx.y;
  if (source_row < source_rows && source_column < source_columns) {
    tile[threadIdx.y][threadIdx.x] =
        source[static_cast<std::size_t>(source_row) * source_columns + source_column];
  }
  __syncthreads();
  const std::uint32_t output_column = blockIdx.y * kTile + threadIdx.x;
  const std::uint32_t output_row = blockIdx.x * kTile + threadIdx.y;
  if (output_row < source_columns && output_column < source_rows) {
    destination[static_cast<std::size_t>(output_row) * destination_columns +
                destination_column_offset + output_column] =
        tile[threadIdx.x][threadIdx.y];
  }
}

std::string prefix(int layer) {
  return "model.language_model.layers." + std::to_string(layer);
}

struct SourcePart {
  const NativeTensorView* tensor = nullptr;
  std::uint32_t rows = 0;
};

struct Plan {
  std::string name;
  std::uint32_t columns = 0;
  std::uint64_t byte_offset = 0;
  std::vector<SourcePart> parts;
};

struct ChecksumTask16 {
  const std::uint16_t* data;
  std::uint64_t elements;
};

struct PayloadChecksum16 {
  std::uint64_t xor_value = 0;
  std::uint64_t sum_value = 0;
};

__global__ void checksum_u16_tasks_kernel(const ChecksumTask16* tasks,
                                          std::uint64_t* block_xor,
                                          std::uint64_t* block_sum) {
  __shared__ std::uint64_t xor_values[256];
  __shared__ std::uint64_t sum_values[256];
  const ChecksumTask16 task = tasks[blockIdx.x];
  std::uint64_t local_xor = 0;
  std::uint64_t local_sum = 0;
  for (std::uint64_t index = threadIdx.x; index < task.elements;
       index += blockDim.x) {
    const std::uint64_t value = task.data[index];
    local_xor ^= value;
    local_sum += value;
  }
  xor_values[threadIdx.x] = local_xor;
  sum_values[threadIdx.x] = local_sum;
  __syncthreads();
  for (unsigned offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      xor_values[threadIdx.x] ^= xor_values[threadIdx.x + offset];
      sum_values[threadIdx.x] += sum_values[threadIdx.x + offset];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    block_xor[blockIdx.x] = xor_values[0];
    block_sum[blockIdx.x] = sum_values[0];
  }
}

PayloadChecksum16 checksum_tasks(const std::vector<ChecksumTask16>& tasks) {
  if (tasks.empty()) throw std::runtime_error("empty derived checksum task list");
  ChecksumTask16* device_tasks = nullptr;
  std::uint64_t* device_xor = nullptr;
  std::uint64_t* device_sum = nullptr;
  try {
    check_hip(hipMalloc(&device_tasks, tasks.size() * sizeof(tasks[0])),
              "hipMalloc derived checksum tasks");
    check_hip(hipMalloc(&device_xor, tasks.size() * sizeof(std::uint64_t)),
              "hipMalloc derived checksum xor");
    check_hip(hipMalloc(&device_sum, tasks.size() * sizeof(std::uint64_t)),
              "hipMalloc derived checksum sum");
    check_hip(hipMemcpy(device_tasks, tasks.data(), tasks.size() * sizeof(tasks[0]),
                        hipMemcpyHostToDevice),
              "hipMemcpy derived checksum tasks");
    hipLaunchKernelGGL(checksum_u16_tasks_kernel,
                       dim3(static_cast<unsigned>(tasks.size())), dim3(256),
                       0, nullptr, device_tasks, device_xor, device_sum);
    check_hip(hipGetLastError(), "checksum_u16_tasks_kernel");
    std::vector<std::uint64_t> host_xor(tasks.size());
    std::vector<std::uint64_t> host_sum(tasks.size());
    check_hip(hipMemcpy(host_xor.data(), device_xor,
                        host_xor.size() * sizeof(host_xor[0]),
                        hipMemcpyDeviceToHost),
              "hipMemcpy derived checksum xor");
    check_hip(hipMemcpy(host_sum.data(), device_sum,
                        host_sum.size() * sizeof(host_sum[0]),
                        hipMemcpyDeviceToHost),
              "hipMemcpy derived checksum sum");
    PayloadChecksum16 result;
    for (std::size_t index = 0; index < tasks.size(); ++index) {
      result.xor_value ^= host_xor[index];
      result.sum_value += host_sum[index];
    }
    (void)hipFree(device_sum);
    (void)hipFree(device_xor);
    (void)hipFree(device_tasks);
    return result;
  } catch (...) {
    if (device_sum) (void)hipFree(device_sum);
    if (device_xor) (void)hipFree(device_xor);
    if (device_tasks) (void)hipFree(device_tasks);
    throw;
  }
}

std::pair<PayloadChecksum16, PayloadChecksum16> full_payload_checksums(
    const std::vector<Plan>& plans, void* allocation, std::uint64_t bytes) {
  std::vector<ChecksumTask16> source_tasks;
  source_tasks.reserve(270);
  for (const Plan& plan : plans) {
    for (const SourcePart& part : plan.parts) {
      source_tasks.push_back({
          static_cast<const std::uint16_t*>(part.tensor->device_pointer),
          static_cast<std::uint64_t>(part.rows) * kHidden,
      });
    }
  }
  constexpr std::uint64_t elements_per_task = 8ULL * 1024 * 1024;
  const std::uint64_t elements = bytes / sizeof(std::uint16_t);
  std::vector<ChecksumTask16> derived_tasks;
  const auto* derived = static_cast<const std::uint16_t*>(allocation);
  for (std::uint64_t offset = 0; offset < elements;
       offset += elements_per_task) {
    derived_tasks.push_back({derived + offset,
                             std::min(elements_per_task, elements - offset)});
  }
  return {checksum_tasks(source_tasks), checksum_tasks(derived_tasks)};
}

const NativeTensorView& require_tensor(const NativeWeightStore& weights,
                                       const std::string& name,
                                       std::uint32_t rows) {
  const NativeTensorView* tensor = weights.find(name);
  if (tensor == nullptr || tensor->rank != 2 || tensor->shape[0] != rows ||
      tensor->shape[1] != kHidden || tensor->payload_bytes !=
          static_cast<std::uint64_t>(rows) * kHidden * sizeof(hip_bfloat16)) {
    throw std::runtime_error("native derived-weight source shape mismatch: " + name);
  }
  return *tensor;
}

std::vector<Plan> build_plans(const NativeWeightStore& weights,
                              std::uint64_t* total_bytes) {
  std::vector<Plan> plans;
  plans.reserve(120);
  std::uint64_t offset = 0;
  for (int layer = 0; layer < 40; ++layer) {
    const std::string base = prefix(layer);
    if (layer % 4 == 3) {
      Plan plan;
      plan.name = "layer" + std::to_string(layer) + ".full_qkv_fused_t";
      plan.columns = kFullFused;
      plan.byte_offset = offset;
      plan.parts = {
          {&require_tensor(weights, base + ".self_attn.q_proj.weight", 8192), 8192},
          {&require_tensor(weights, base + ".self_attn.k_proj.weight", 512), 512},
          {&require_tensor(weights, base + ".self_attn.v_proj.weight", 512), 512},
      };
      offset += static_cast<std::uint64_t>(kHidden) * plan.columns *
                sizeof(hip_bfloat16);
      plans.push_back(std::move(plan));
    } else {
      Plan plan;
      plan.name = "layer" + std::to_string(layer) + ".linear_input_fused_t";
      plan.columns = kLinearFused;
      plan.byte_offset = offset;
      plan.parts = {
          {&require_tensor(weights, base + ".linear_attn.in_proj_qkv.weight", 8192), 8192},
          {&require_tensor(weights, base + ".linear_attn.in_proj_z.weight", 4096), 4096},
          {&require_tensor(weights, base + ".linear_attn.in_proj_a.weight", 32), 32},
          {&require_tensor(weights, base + ".linear_attn.in_proj_b.weight", 32), 32},
      };
      offset += static_cast<std::uint64_t>(kHidden) * plan.columns *
                sizeof(hip_bfloat16);
      plans.push_back(std::move(plan));
    }

    Plan shared;
    shared.name = "layer" + std::to_string(layer) + ".shared_input_fused_t";
    shared.columns = kSharedFused;
    shared.byte_offset = offset;
    shared.parts = {
        {&require_tensor(weights, base + ".mlp.shared_expert_gate.weight", 1), 1},
        {&require_tensor(weights, base + ".mlp.shared_expert.gate_proj.weight", 512), 512},
        {&require_tensor(weights, base + ".mlp.shared_expert.up_proj.weight", 512), 512},
    };
    offset += static_cast<std::uint64_t>(kHidden) * shared.columns *
              sizeof(hip_bfloat16);
    plans.push_back(std::move(shared));

    Plan router;
    router.name = "layer" + std::to_string(layer) + ".router_t";
    router.columns = 256;
    router.byte_offset = offset;
    router.parts = {
        {&require_tensor(weights, base + ".mlp.gate.weight", 256), 256},
    };
    offset += static_cast<std::uint64_t>(kHidden) * router.columns *
              sizeof(hip_bfloat16);
    plans.push_back(std::move(router));
  }
  *total_bytes = offset;
  return plans;
}

std::size_t validate_samples(const std::vector<Plan>& plans, void* allocation) {
  std::size_t exact = 0;
  auto* base = static_cast<unsigned char*>(allocation);
  for (const Plan& plan : plans) {
    std::uint32_t column_offset = 0;
    for (const SourcePart& part : plan.parts) {
      const std::uint32_t sample_row = part.rows - 1;
      const std::uint32_t sample_column = kHidden - 1;
      std::uint16_t source = 0;
      std::uint16_t destination = 0;
      const auto* source_pointer = static_cast<const unsigned char*>(
          part.tensor->device_pointer) +
          (static_cast<std::uint64_t>(sample_row) * kHidden + sample_column) *
              sizeof(std::uint16_t);
      const auto* destination_pointer = base + plan.byte_offset +
          (static_cast<std::uint64_t>(sample_column) * plan.columns +
           column_offset + sample_row) * sizeof(std::uint16_t);
      check_hip(hipMemcpy(&source, source_pointer, sizeof(source),
                          hipMemcpyDeviceToHost),
                "hipMemcpy derived source sample");
      check_hip(hipMemcpy(&destination, destination_pointer,
                          sizeof(destination), hipMemcpyDeviceToHost),
                "hipMemcpy derived destination sample");
      exact += source == destination ? 1 : 0;
      column_offset += part.rows;
    }
  }
  return exact;
}

float bf16_to_float(std::uint16_t value) {
  const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16;
  float result = 0.0f;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

class ProbeAllocation {
 public:
  explicit ProbeAllocation(std::size_t bytes) {
    check_hip(hipMalloc(&pointer_, bytes), "hipMalloc derived projection probe");
  }
  ~ProbeAllocation() {
    if (pointer_) (void)hipFree(pointer_);
  }
  ProbeAllocation(const ProbeAllocation&) = delete;
  ProbeAllocation& operator=(const ProbeAllocation&) = delete;
  void* get() const { return pointer_; }

 private:
  void* pointer_ = nullptr;
};

}  // namespace

NativeDerivedWeightStore::~NativeDerivedWeightStore() { reset(); }

NativeDerivedWeightMetrics NativeDerivedWeightStore::build(
    const NativeWeightStore& weights, int device) {
  if (built() || !weights.loaded()) {
    throw std::runtime_error(
        "native derived weights require one loaded, unbuilt weight store");
  }
  const auto started = std::chrono::steady_clock::now();
  device_ = device;
  check_hip(hipSetDevice(device_), "hipSetDevice");
  NativeDerivedWeightMetrics metrics;
  std::size_t total = 0;
  check_hip(hipMemGetInfo(&metrics.free_bytes_before, &total), "hipMemGetInfo before");
  std::uint64_t bytes = 0;
  const std::vector<Plan> plans = build_plans(weights, &bytes);
  constexpr std::uint64_t expected_bytes = 2105180160ULL;
  if (plans.size() != 120 || bytes != expected_bytes) {
    throw std::runtime_error("native derived-weight plan geometry drift");
  }
  if (metrics.free_bytes_before < bytes + 64ULL * 1024 * 1024) {
    throw std::runtime_error("insufficient device memory for native derived weights");
  }

  try {
    const auto allocation_started = std::chrono::steady_clock::now();
    check_hip(hipMalloc(&allocation_, bytes), "hipMalloc derived weights");
    allocation_bytes_ = bytes;
    check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize derived allocation");
    metrics.allocation_ms = elapsed_ms(allocation_started);

    const auto pack_started = std::chrono::steady_clock::now();
    auto* base = static_cast<unsigned char*>(allocation_);
    views_.reserve(plans.size());
    name_to_index_.reserve(plans.size());
    for (const Plan& plan : plans) {
      auto* destination = reinterpret_cast<hip_bfloat16*>(base + plan.byte_offset);
      std::uint32_t column_offset = 0;
      for (const SourcePart& part : plan.parts) {
        const dim3 block(kTile, kTile);
        const dim3 grid((kHidden + kTile - 1) / kTile,
                        (part.rows + kTile - 1) / kTile);
        hipLaunchKernelGGL(
            transpose_into_fused_kernel, grid, block, 0, nullptr,
            static_cast<const hip_bfloat16*>(part.tensor->device_pointer),
            destination, part.rows, kHidden, plan.columns, column_offset);
        check_hip(hipGetLastError(), "transpose_into_fused_kernel");
        column_offset += part.rows;
      }
      if (column_offset != plan.columns) {
        throw std::runtime_error("native derived-weight fused column mismatch");
      }
      const std::size_t index = views_.size();
      views_.push_back(NativeDerivedWeightView{
          plan.name,
          destination,
          kHidden,
          plan.columns,
          static_cast<std::uint64_t>(kHidden) * plan.columns *
              sizeof(hip_bfloat16),
      });
      name_to_index_.emplace(plan.name, index);
    }
    check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize derived pack");
    metrics.pack_ms = elapsed_ms(pack_started);
    const auto checksum_started = std::chrono::steady_clock::now();
    const auto checksums = full_payload_checksums(plans, allocation_, bytes);
    metrics.checksum_ms = elapsed_ms(checksum_started);
    metrics.source_u16_xor = checksums.first.xor_value;
    metrics.source_u16_sum = checksums.first.sum_value;
    metrics.derived_u16_xor = checksums.second.xor_value;
    metrics.derived_u16_sum = checksums.second.sum_value;
    metrics.full_payload_checksum_equal =
        metrics.source_u16_xor == metrics.derived_u16_xor &&
        metrics.source_u16_sum == metrics.derived_u16_sum;
    metrics.expected_sample_elements = 310;
    metrics.exact_sample_elements = validate_samples(plans, allocation_);
    if (!metrics.full_payload_checksum_equal ||
        metrics.exact_sample_elements != metrics.expected_sample_elements ||
        name_to_index_.size() != plans.size()) {
      throw std::runtime_error("native derived-weight exact sample validation failed");
    }
    check_hip(hipMemGetInfo(&metrics.free_bytes_after, &total), "hipMemGetInfo after");
    metrics.view_count = views_.size();
    metrics.payload_bytes = allocation_bytes_;
    metrics.build_wall_ms = elapsed_ms(started);
    return metrics;
  } catch (...) {
    reset();
    throw;
  }
}

const NativeDerivedWeightView* NativeDerivedWeightStore::find(
    const std::string& name) const {
  const auto found = name_to_index_.find(name);
  return found == name_to_index_.end() ? nullptr : &views_[found->second];
}

void NativeDerivedWeightStore::reset() noexcept {
  (void)hipSetDevice(device_);
  if (allocation_) (void)hipFree(allocation_);
  allocation_ = nullptr;
  allocation_bytes_ = 0;
  views_.clear();
  name_to_index_.clear();
}

NativeDerivedProjectionResult probe_layer0_derived_projection(
    const NativeWeightStore& weights,
    const NativeDerivedWeightStore& derived) {
  constexpr std::size_t rows = 8;
  constexpr std::array<std::uint32_t, 4> part_rows = {8192, 4096, 32, 32};
  const std::array<std::string, 4> part_names = {
      "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
      "model.language_model.layers.0.linear_attn.in_proj_z.weight",
      "model.language_model.layers.0.linear_attn.in_proj_a.weight",
      "model.language_model.layers.0.linear_attn.in_proj_b.weight",
  };
  const NativeDerivedWeightView* fused =
      derived.find("layer0.linear_input_fused_t");
  if (fused == nullptr || fused->rows != kHidden ||
      fused->columns != kLinearFused) {
    throw std::runtime_error("layer0 native derived projection is missing");
  }

  std::vector<std::uint16_t> host_input(rows * kHidden, 0x3f80);
  ProbeAllocation input(host_input.size() * sizeof(host_input[0]));
  ProbeAllocation fused_output(rows * kLinearFused * sizeof(std::uint16_t));
  check_hip(hipMemcpy(input.get(), host_input.data(),
                      host_input.size() * sizeof(host_input[0]),
                      hipMemcpyHostToDevice),
            "hipMemcpy derived projection input");
  Bf16GemmPlan fused_plan(rows, kLinearFused, kHidden);
  fused_plan.launch(input.get(), fused->device_pointer, fused_output.get());

  std::array<std::unique_ptr<ProbeAllocation>, 4> raw_outputs;
  for (std::size_t index = 0; index < part_rows.size(); ++index) {
    const NativeTensorView* source = weights.find(part_names[index]);
    if (source == nullptr) {
      throw std::runtime_error("layer0 raw projection source is missing");
    }
    raw_outputs[index] = std::make_unique<ProbeAllocation>(
        rows * part_rows[index] * sizeof(std::uint16_t));
    Bf16GemmPlan raw_plan(rows, part_rows[index], kHidden,
                         128ULL * 1024 * 1024, true);
    raw_plan.launch(input.get(), source->device_pointer,
                    raw_outputs[index]->get());
  }
  check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize derived projection");

  std::vector<std::uint16_t> host_fused(rows * kLinearFused);
  check_hip(hipMemcpy(host_fused.data(), fused_output.get(),
                      host_fused.size() * sizeof(host_fused[0]),
                      hipMemcpyDeviceToHost),
            "hipMemcpy fused projection output");
  std::array<std::vector<std::uint16_t>, 4> host_raw;
  for (std::size_t index = 0; index < part_rows.size(); ++index) {
    host_raw[index].resize(rows * part_rows[index]);
    check_hip(hipMemcpy(host_raw[index].data(), raw_outputs[index]->get(),
                        host_raw[index].size() * sizeof(std::uint16_t),
                        hipMemcpyDeviceToHost),
              "hipMemcpy raw projection output");
  }

  NativeDerivedProjectionResult result;
  result.elements = host_fused.size();
  long double error_squared = 0.0;
  long double reference_squared = 0.0;
  for (std::size_t row = 0; row < rows; ++row) {
    std::size_t column_offset = 0;
    for (std::size_t part = 0; part < part_rows.size(); ++part) {
      for (std::size_t column = 0; column < part_rows[part]; ++column) {
        const std::uint16_t actual =
            host_fused[row * kLinearFused + column_offset + column];
        const std::uint16_t expected =
            host_raw[part][row * part_rows[part] + column];
        result.exact_elements += actual == expected ? 1 : 0;
        const double actual_float = bf16_to_float(actual);
        const double expected_float = bf16_to_float(expected);
        const double error = actual_float - expected_float;
        result.maximum_absolute_error =
            std::max(result.maximum_absolute_error, std::abs(error));
        error_squared += static_cast<long double>(error) * error;
        reference_squared +=
            static_cast<long double>(expected_float) * expected_float;
      }
      column_offset += part_rows[part];
    }
  }
  result.relative_l2_error = std::sqrt(
      static_cast<double>(error_squared / std::max(reference_squared, 1.0L)));
  if (result.relative_l2_error > 0.005) {
    throw std::runtime_error("layer0 native derived projection exceeds relL2 gate");
  }
  return result;
}

NativeDerivedProjectionResult validate_layer0_router_transpose(
    const NativeWeightStore& weights,
    const NativeDerivedWeightStore& derived) {
  constexpr std::size_t experts = 256;
  const NativeTensorView* source =
      weights.find("model.language_model.layers.0.mlp.gate.weight");
  const NativeDerivedWeightView* transposed = derived.find("layer0.router_t");
  if (source == nullptr || source->rank != 2 || source->shape[0] != experts ||
      source->shape[1] != kHidden || transposed == nullptr ||
      transposed->rows != kHidden || transposed->columns != experts) {
    throw std::runtime_error("layer0 native router transpose is missing");
  }
  const std::size_t elements = experts * kHidden;
  std::vector<std::uint16_t> host_source(elements);
  std::vector<std::uint16_t> host_transposed(elements);
  check_hip(hipMemcpy(host_source.data(), source->device_pointer,
                      elements * sizeof(std::uint16_t), hipMemcpyDeviceToHost),
            "hipMemcpy router source validation");
  check_hip(hipMemcpy(host_transposed.data(), transposed->device_pointer,
                      elements * sizeof(std::uint16_t), hipMemcpyDeviceToHost),
            "hipMemcpy router transpose validation");

  NativeDerivedProjectionResult result;
  result.elements = elements;
  long double error_squared = 0.0;
  long double reference_squared = 0.0;
  for (std::size_t input = 0; input < kHidden; ++input) {
    for (std::size_t expert = 0; expert < experts; ++expert) {
      const std::uint16_t actual =
          host_transposed[input * experts + expert];
      const std::uint16_t expected = host_source[expert * kHidden + input];
      result.exact_elements += actual == expected ? 1 : 0;
      const double actual_float = bf16_to_float(actual);
      const double expected_float = bf16_to_float(expected);
      const double error = actual_float - expected_float;
      result.maximum_absolute_error =
          std::max(result.maximum_absolute_error, std::abs(error));
      error_squared += static_cast<long double>(error) * error;
      reference_squared +=
          static_cast<long double>(expected_float) * expected_float;
    }
  }
  result.relative_l2_error = std::sqrt(
      static_cast<double>(error_squared / std::max(reference_squared, 1.0L)));
  if (result.exact_elements != result.elements) {
    throw std::runtime_error("layer0 native router transpose is not bit exact");
  }
  return result;
}

}  // namespace aima
