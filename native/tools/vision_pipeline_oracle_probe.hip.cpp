// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_vision_pipeline.h"
#include "aima/native_weight_store.h"
#include "aima/sha256.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

void check_hip(hipError_t status, const char* operation) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(status));
  }
}

class DeviceAllocation {
 public:
  explicit DeviceAllocation(std::size_t bytes) {
    check_hip(hipMalloc(&pointer_, bytes), "hipMalloc vision pipeline tensor");
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

std::vector<std::string> split(std::string_view input, char delimiter) {
  std::vector<std::string> parts;
  std::size_t begin = 0;
  while (begin <= input.size()) {
    const std::size_t end = input.find(delimiter, begin);
    const std::size_t length =
        end == std::string_view::npos ? input.size() - begin : end - begin;
    if (length == 0) {
      throw std::invalid_argument("vision pipeline list contains an empty item");
    }
    parts.emplace_back(input.substr(begin, length));
    if (end == std::string_view::npos) break;
    begin = end + 1;
  }
  return parts;
}

std::size_t parse_size(std::string_view text, const char* description) {
  if (text.empty() ||
      !std::all_of(text.begin(), text.end(),
                   [](char value) { return value >= '0' && value <= '9'; })) {
    throw std::invalid_argument(std::string(description) + " is invalid");
  }
  const unsigned long long value = std::stoull(std::string(text));
  const std::size_t converted = static_cast<std::size_t>(value);
  if (value == 0 || static_cast<unsigned long long>(converted) != value) {
    throw std::invalid_argument(std::string(description) + " is invalid");
  }
  return converted;
}

std::vector<aima::NativeVlGrid> parse_grids(std::string_view specification) {
  std::vector<aima::NativeVlGrid> grids;
  for (const std::string& item : split(specification, ';')) {
    const std::vector<std::string> dimensions = split(item, 'x');
    if (dimensions.size() != 3) {
      throw std::invalid_argument("vision grid must be T x H x W");
    }
    grids.push_back(aima::NativeVlGrid{
        parse_size(dimensions[0], "vision grid temporal dimension"),
        parse_size(dimensions[1], "vision grid height"),
        parse_size(dimensions[2], "vision grid width")});
  }
  return grids;
}

std::vector<unsigned char> read_file(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream || stream.tellg() < 0) {
    throw std::runtime_error("oracle file is unavailable: " + path.string());
  }
  const std::size_t bytes = static_cast<std::size_t>(stream.tellg());
  std::vector<unsigned char> value(bytes);
  stream.seekg(0);
  if (bytes != 0 &&
      !stream.read(reinterpret_cast<char*>(value.data()),
                   static_cast<std::streamsize>(bytes))) {
    throw std::runtime_error("oracle file read failed: " + path.string());
  }
  return value;
}

std::vector<unsigned char> read_concatenated_files(std::string_view paths) {
  std::vector<unsigned char> result;
  for (const std::string& path : split(paths, ',')) {
    std::vector<unsigned char> part =
        read_file(std::filesystem::absolute(path));
    if (part.size() > std::numeric_limits<std::size_t>::max() - result.size()) {
      throw std::invalid_argument("concatenated pixel inputs overflow");
    }
    result.insert(result.end(), part.begin(), part.end());
  }
  return result;
}

void write_file(const std::filesystem::path& path,
                const std::vector<unsigned char>& bytes) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream ||
      !stream.write(reinterpret_cast<const char*>(bytes.data()),
                    static_cast<std::streamsize>(bytes.size()))) {
    throw std::runtime_error("pipeline output write failed: " + path.string());
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

  bool passed() const {
    return finite_elements == elements && relative_l2_error <= 0.002 &&
           cosine_similarity >= 0.999;
  }
};

Comparison compare_bf16(const std::vector<unsigned char>& actual,
                        const std::vector<unsigned char>& expected) {
  if (actual.size() != expected.size() || actual.size() % 2 != 0) {
    throw std::invalid_argument("pipeline comparison shape is invalid");
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

void print_comparison(const Comparison& comparison) {
  std::cout << "{\"passed\":"
            << (comparison.passed() ? "true" : "false")
            << ",\"elements\":" << comparison.elements
            << ",\"exact_elements\":" << comparison.exact_elements
            << ",\"finite_elements\":" << comparison.finite_elements
            << ",\"first_mismatch_index\":"
            << (comparison.first_mismatch_index ==
                        std::numeric_limits<std::size_t>::max()
                    ? -1LL
                    : static_cast<long long>(comparison.first_mismatch_index))
            << ",\"first_expected_bits\":"
            << comparison.first_expected_bits
            << ",\"first_actual_bits\":" << comparison.first_actual_bits
            << ",\"maximum_absolute_error\":"
            << comparison.maximum_absolute_error
            << ",\"relative_l2_error\":" << comparison.relative_l2_error
            << ",\"cosine_similarity\":" << comparison.cosine_similarity
            << ",\"expected_sha256\":\"" << comparison.expected_sha256
            << "\",\"actual_sha256\":\"" << comparison.actual_sha256
            << "\",\"bit_exact\":"
            << (comparison.exact_elements == comparison.elements ? "true"
                                                                  : "false")
            << '}';
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 12) {
    std::cerr
        << "usage: native-vision-pipeline-probe MODEL_DIR ATTENTION_IMAGE "
           "GRID_SPEC PIXEL_INPUTS EXPECTED_BLOCK0 EXPECTED_BLOCK26 "
           "EXPECTED_MERGER LOAD_REPORT ACTUAL_BLOCK0 ACTUAL_BLOCK26 "
           "ACTUAL_MERGER\n";
    return 2;
  }
  try {
    constexpr std::size_t kPatchFeatures = 1536;
    constexpr std::size_t kVisionHidden = 1152;
    constexpr std::size_t kLanguageHidden = 2048;
    const std::vector<std::string> grid_groups = split(argv[3], '|');
    const std::vector<std::string> pixel_groups = split(argv[4], '|');
    if (grid_groups.size() != pixel_groups.size()) {
      throw std::invalid_argument(
          "vision grid and pixel batch counts do not match");
    }

    aima::NativeWeightLoadOptions options;
    options.model_dir = std::filesystem::absolute(argv[1]);
    options.native_report = std::filesystem::absolute(argv[8]);
    aima::NativeWeightStore weights;
    const aima::NativeWeightLoadMetrics load = weights.load_visual(options);
    std::vector<std::unique_ptr<aima::NativeVisionPipelinePlan>> plans;
    std::vector<std::vector<unsigned char>> grouped_pixels;
    std::size_t total_patches = 0;
    std::size_t total_merged_tokens = 0;
    std::size_t temporary_bytes = 0;
    std::size_t metadata_resident_bytes = 0;
    std::size_t library_workspace_bytes = 0;
    plans.reserve(grid_groups.size());
    grouped_pixels.reserve(pixel_groups.size());
    for (std::size_t group = 0; group < grid_groups.size(); ++group) {
      plans.push_back(std::make_unique<aima::NativeVisionPipelinePlan>(
          weights, std::filesystem::absolute(argv[2]),
          parse_grids(grid_groups[group])));
      grouped_pixels.push_back(read_concatenated_files(pixel_groups[group]));
      const auto& plan = *plans.back();
      const std::size_t expected_pixel_bytes =
          plan.patch_count() * kPatchFeatures * sizeof(std::uint16_t);
      if (grouped_pixels.back().size() != expected_pixel_bytes) {
        throw std::runtime_error("pipeline pixel batch byte size mismatch");
      }
      total_patches += plan.patch_count();
      total_merged_tokens += plan.merged_token_count();
      temporary_bytes = std::max(temporary_bytes, plan.temporary_bytes());
      metadata_resident_bytes += plan.metadata_resident_bytes();
      library_workspace_bytes += plan.library_workspace_bytes();
    }

    const std::size_t pixel_bytes =
        total_patches * kPatchFeatures * sizeof(std::uint16_t);
    const std::size_t hidden_bytes =
        total_patches * kVisionHidden * sizeof(std::uint16_t);
    const std::size_t output_bytes =
        total_merged_tokens * kLanguageHidden * sizeof(std::uint16_t);
    std::vector<unsigned char> pixels;
    pixels.reserve(pixel_bytes);
    for (const auto& group : grouped_pixels) {
      pixels.insert(pixels.end(), group.begin(), group.end());
    }
    const std::vector<unsigned char> expected_block0 =
        read_file(std::filesystem::absolute(argv[5]));
    const std::vector<unsigned char> expected_block26 =
        read_file(std::filesystem::absolute(argv[6]));
    const std::vector<unsigned char> expected_merger =
        read_file(std::filesystem::absolute(argv[7]));
    if (pixels.size() != pixel_bytes ||
        expected_block0.size() != hidden_bytes ||
        expected_block26.size() != hidden_bytes ||
        expected_merger.size() != output_bytes) {
      throw std::runtime_error("pipeline oracle byte size mismatch");
    }

    DeviceAllocation pixel_device(pixel_bytes);
    DeviceAllocation hidden_device(hidden_bytes);
    DeviceAllocation output_device(output_bytes);
    DeviceAllocation temporary(temporary_bytes);
    check_hip(hipMemcpy(pixel_device.get(), pixels.data(), pixels.size(),
                        hipMemcpyHostToDevice),
              "hipMemcpy vision pipeline pixels");

    const auto launch_all = [&]() {
      std::size_t patch_offset = 0;
      std::size_t token_offset = 0;
      for (const auto& plan : plans) {
        auto* pixel_input =
            static_cast<unsigned char*>(pixel_device.get()) +
            patch_offset * kPatchFeatures * sizeof(std::uint16_t);
        auto* output = static_cast<unsigned char*>(output_device.get()) +
                       token_offset * kLanguageHidden *
                           sizeof(std::uint16_t);
        plan->launch(pixel_input, output, temporary.get(), temporary_bytes);
        patch_offset += plan->patch_count();
        token_offset += plan->merged_token_count();
      }
    };

    const auto launch_encoder_all = [&](std::size_t last_block_index) {
      std::size_t patch_offset = 0;
      for (const auto& plan : plans) {
        auto* pixel_input =
            static_cast<unsigned char*>(pixel_device.get()) +
            patch_offset * kPatchFeatures * sizeof(std::uint16_t);
        auto* hidden_output =
            static_cast<unsigned char*>(hidden_device.get()) +
            patch_offset * kVisionHidden * sizeof(std::uint16_t);
        plan->launch_encoder_through(
            last_block_index, pixel_input, hidden_output, temporary.get(),
            temporary_bytes);
        patch_offset += plan->patch_count();
      }
    };

    launch_encoder_all(0);
    check_hip(hipDeviceSynchronize(),
              "hipDeviceSynchronize pipeline block 0");
    std::vector<unsigned char> actual_block0(hidden_bytes);
    check_hip(hipMemcpy(actual_block0.data(), hidden_device.get(), hidden_bytes,
                        hipMemcpyDeviceToHost),
              "hipMemcpy pipeline block 0");
    write_file(std::filesystem::absolute(argv[9]), actual_block0);
    const Comparison block0_comparison =
        compare_bf16(actual_block0, expected_block0);

    launch_encoder_all(26);
    check_hip(hipDeviceSynchronize(),
              "hipDeviceSynchronize pipeline block 26");
    std::vector<unsigned char> actual_block26(hidden_bytes);
    check_hip(hipMemcpy(actual_block26.data(), hidden_device.get(), hidden_bytes,
                        hipMemcpyDeviceToHost),
              "hipMemcpy pipeline block 26");
    write_file(std::filesystem::absolute(argv[10]), actual_block26);
    const Comparison block26_comparison =
        compare_bf16(actual_block26, expected_block26);

    launch_all();
    check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize pipeline warmup");
    std::vector<unsigned char> first(output_bytes);
    check_hip(hipMemcpy(first.data(), output_device.get(), output_bytes,
                        hipMemcpyDeviceToHost),
              "hipMemcpy first pipeline output");
    const std::string first_sha256 =
        aima::sha256_bytes(first.data(), first.size());

    Event start;
    Event stop;
    std::vector<double> measured_ms;
    measured_ms.reserve(5);
    for (std::size_t repetition = 0; repetition < 5; ++repetition) {
      check_hip(hipEventRecord(start), "hipEventRecord pipeline start");
      launch_all();
      check_hip(hipEventRecord(stop), "hipEventRecord pipeline stop");
      check_hip(hipEventSynchronize(stop),
                "hipEventSynchronize pipeline stop");
      float milliseconds = 0.0f;
      check_hip(hipEventElapsedTime(&milliseconds, start, stop),
                "hipEventElapsedTime pipeline");
      measured_ms.push_back(milliseconds);
    }
    std::vector<unsigned char> actual(output_bytes);
    check_hip(hipMemcpy(actual.data(), output_device.get(), output_bytes,
                        hipMemcpyDeviceToHost),
              "hipMemcpy pipeline output");
    write_file(std::filesystem::absolute(argv[11]), actual);
    const Comparison merger_comparison = compare_bf16(actual, expected_merger);
    std::vector<double> sorted_ms = measured_ms;
    std::sort(sorted_ms.begin(), sorted_ms.end());
    const double median_ms = sorted_ms[sorted_ms.size() / 2];
    const bool deterministic =
        first_sha256 == merger_comparison.actual_sha256;
    const bool passed = block0_comparison.passed() &&
                        block26_comparison.passed() &&
                        merger_comparison.passed() && deterministic;

    std::cout << std::setprecision(17)
              << "{\"schema\":\"aima-amd395-qwen36/"
                 "native-vision-pipeline-oracle/v1\","
              << "\"complete\":" << (passed ? "true" : "false")
              << ",\"patches\":" << total_patches
              << ",\"merged_tokens\":" << total_merged_tokens
              << ",\"group_count\":" << plans.size()
              << ",\"groups\":[";
    for (std::size_t group = 0; group < plans.size(); ++group) {
      if (group != 0) std::cout << ',';
      const auto& plan = *plans[group];
      std::cout << "{\"patches\":" << plan.patch_count()
                << ",\"merged_tokens\":" << plan.merged_token_count()
                << ",\"rotary_cos_sha256\":\""
                << plan.rotary_cos_sha256()
                << "\",\"rotary_sin_sha256\":\""
                << plan.rotary_sin_sha256() << "\",\"cu_seqlens\":[";
      for (std::size_t index = 0; index < plan.cu_seqlens().size(); ++index) {
        if (index != 0) std::cout << ',';
        std::cout << plan.cu_seqlens()[index];
      }
      std::cout << "]}";
    }
    std::cout << "],\"weight_payload_bytes\":" << load.payload_bytes
              << ",\"temporary_bytes\":" << temporary_bytes
              << ",\"metadata_resident_bytes\":"
              << metadata_resident_bytes
              << ",\"library_workspace_bytes\":"
              << library_workspace_bytes
              << ",\"attention_image_sha256\":\""
              << plans.front()->attention_image_sha256()
              << "\",\"measured_ms\":[";
    for (std::size_t index = 0; index < measured_ms.size(); ++index) {
      if (index != 0) std::cout << ',';
      std::cout << measured_ms[index];
    }
    std::cout << "],\"median_ms\":" << median_ms
              << ",\"comparisons\":{\"vision_block_0\":";
    print_comparison(block0_comparison);
    std::cout << ",\"vision_block_26\":";
    print_comparison(block26_comparison);
    std::cout << ",\"vision_merger\":";
    print_comparison(merger_comparison);
    std::cout << "},\"repeat_actual_sha256\":\"" << first_sha256
              << "\",\"repeat_deterministic\":"
              << (deterministic ? "true" : "false")
              << "}\n";
    return passed ? 0 : 3;
  } catch (const std::exception& error) {
    std::cerr << "native vision pipeline probe: " << error.what() << '\n';
    return 1;
  }
}
