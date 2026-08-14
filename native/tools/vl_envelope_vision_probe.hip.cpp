// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_vision_pipeline.h"
#include "aima/native_weight_store.h"
#include "aima/sha256.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kMediaItems = 16;
constexpr std::size_t kPatchFeatures = 1536;
constexpr std::size_t kLanguageHidden = 2048;

void check_hip(hipError_t status, const char* operation) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(status));
  }
}

class DeviceAllocation {
 public:
  DeviceAllocation(std::size_t bytes, const char* label) : bytes_(bytes) {
    check_hip(hipMalloc(&pointer_, bytes), label);
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
  std::size_t bytes() const { return bytes_; }

 private:
  void* pointer_ = nullptr;
  std::size_t bytes_ = 0;
};

class Event {
 public:
  Event() { check_hip(hipEventCreate(&value_), "hipEventCreate"); }
  ~Event() {
    if (value_ != nullptr) {
      const hipError_t ignored = hipEventDestroy(value_);
      static_cast<void>(ignored);
    }
  }
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;
  operator hipEvent_t() const { return value_; }

 private:
  hipEvent_t value_ = nullptr;
};

std::size_t finite_bf16(const std::vector<unsigned char>& bytes) {
  if (bytes.size() % sizeof(std::uint16_t) != 0) {
    throw std::runtime_error("vision envelope output is not BF16-aligned");
  }
  std::size_t finite = 0;
  for (std::size_t offset = 0; offset < bytes.size(); offset += 2) {
    const std::uint16_t bits = static_cast<std::uint16_t>(bytes[offset]) |
                               (static_cast<std::uint16_t>(bytes[offset + 1])
                                << 8U);
    if ((bits & 0x7f80U) != 0x7f80U) ++finite;
  }
  return finite;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "usage: native-vl-envelope-vision-probe MODEL_DIR "
                 "ATTENTION_IMAGE LOAD_REPORT\n";
    return 2;
  }
  try {
    const aima::NativeVlGrid maximum_image_grid{1, 256, 256};
    const std::vector<aima::NativeVlGrid> all_grids(kMediaItems,
                                                    maximum_image_grid);
    const std::vector<aima::NativeVlVisionBatch> batches =
        aima::native_qwen36_vision_batches(all_grids);
    if (batches.size() != kMediaItems) {
      throw std::runtime_error("full image envelope did not form 16 batches");
    }

    aima::NativeWeightLoadOptions options;
    options.model_dir = std::filesystem::absolute(argv[1]);
    options.native_report = std::filesystem::absolute(argv[3]);
    aima::NativeWeightStore weights;
    const aima::NativeWeightLoadMetrics load = weights.load_visual(options);
    aima::NativeVisionPipelinePlan pipeline(
        weights, std::filesystem::absolute(argv[2]), {maximum_image_grid});
    if (pipeline.patch_count() != 65536 ||
        pipeline.merged_token_count() != 16384) {
      throw std::runtime_error("maximum image vision plan shape drifted");
    }

    const std::size_t pixel_bytes =
        pipeline.patch_count() * kPatchFeatures * sizeof(std::uint16_t);
    const std::size_t output_bytes = pipeline.merged_token_count() *
                                     kLanguageHidden * sizeof(std::uint16_t);
    DeviceAllocation pixels(pixel_bytes, "hipMalloc envelope vision pixels");
    DeviceAllocation output(output_bytes, "hipMalloc envelope vision output");
    DeviceAllocation temporary(pipeline.temporary_bytes(),
                               "hipMalloc envelope vision temporary");
    check_hip(hipMemset(pixels.get(), 0, pixels.bytes()),
              "hipMemset envelope vision pixels");

    std::vector<unsigned char> host(output_bytes);
    std::vector<std::string> output_hashes;
    std::vector<double> measured_ms;
    output_hashes.reserve(kMediaItems);
    measured_ms.reserve(kMediaItems);
    std::size_t finite_elements = 0;
    Event start;
    Event stop;
    for (std::size_t batch = 0; batch < batches.size(); ++batch) {
      check_hip(hipMemset(output.get(), 0xff, output.bytes()),
                "hipMemset envelope vision output");
      check_hip(hipEventRecord(start), "hipEventRecord envelope vision start");
      pipeline.launch(pixels.get(), output.get(), temporary.get(),
                      temporary.bytes());
      check_hip(hipEventRecord(stop), "hipEventRecord envelope vision stop");
      check_hip(hipEventSynchronize(stop),
                "hipEventSynchronize envelope vision stop");
      float milliseconds = 0.0F;
      check_hip(hipEventElapsedTime(&milliseconds, start, stop),
                "hipEventElapsedTime envelope vision");
      measured_ms.push_back(milliseconds);
      check_hip(hipMemcpy(host.data(), output.get(), host.size(),
                          hipMemcpyDeviceToHost),
                "hipMemcpy envelope vision output");
      finite_elements += finite_bf16(host);
      output_hashes.push_back(aima::sha256_bytes(host.data(), host.size()));
    }

    const std::size_t output_elements = output_bytes / sizeof(std::uint16_t);
    const std::size_t expected_finite = output_elements * kMediaItems;
    const bool deterministic = std::all_of(
        output_hashes.begin(), output_hashes.end(),
        [&](const std::string& value) { return value == output_hashes.front(); });
    const bool complete = finite_elements == expected_finite && deterministic &&
                          load.payload_bytes == 893142496ULL;
    double total_ms = 0.0;
    for (double value : measured_ms) total_ms += value;

    std::cout << std::setprecision(17)
              << "{\"schema\":\"aima-amd395-qwen36/"
                 "native-vl-envelope-vision-probe/v1\","
              << "\"complete\":" << (complete ? "true" : "false")
              << ",\"cell_id\":\"image_full_encoder_budget\""
              << ",\"media_items\":" << kMediaItems
              << ",\"visual_tokens\":"
              << pipeline.merged_token_count() * kMediaItems
              << ",\"vision_patches\":"
              << pipeline.patch_count() * kMediaItems
              << ",\"vision_batch_count\":" << batches.size()
              << ",\"vision_max_batch_tokens\":"
              << pipeline.merged_token_count()
              << ",\"vision_max_batch_patches\":"
              << pipeline.patch_count()
              << ",\"executed_batches\":" << measured_ms.size()
              << ",\"output_elements_per_batch\":" << output_elements
              << ",\"finite_output_elements\":" << finite_elements
              << ",\"expected_finite_output_elements\":" << expected_finite
              << ",\"repeat_output_sha256\":\""
              << output_hashes.front() << "\""
              << ",\"repeat_deterministic\":"
              << (deterministic ? "true" : "false")
              << ",\"weight_payload_bytes\":" << load.payload_bytes
              << ",\"pixel_device_bytes\":" << pixels.bytes()
              << ",\"output_device_bytes\":" << output.bytes()
              << ",\"temporary_device_bytes\":" << temporary.bytes()
              << ",\"metadata_resident_bytes\":"
              << pipeline.metadata_resident_bytes()
              << ",\"library_workspace_bytes\":"
              << pipeline.library_workspace_bytes()
              << ",\"attention_image_sha256\":\""
              << pipeline.attention_image_sha256() << "\""
              << ",\"batch_wall_ms\":[";
    for (std::size_t index = 0; index < measured_ms.size(); ++index) {
      if (index != 0) std::cout << ',';
      std::cout << measured_ms[index];
    }
    std::cout << "],\"total_vision_wall_ms\":" << total_ms << "}\n";
    return complete ? 0 : 3;
  } catch (const std::exception& error) {
    std::cerr << "native VL envelope vision probe: " << error.what() << '\n';
    return 1;
  }
}
