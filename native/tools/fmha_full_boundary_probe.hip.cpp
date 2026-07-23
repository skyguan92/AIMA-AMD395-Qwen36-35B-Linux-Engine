// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include <dlfcn.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kFeatures = 4096;
constexpr std::size_t kKvFeatures = 512;
constexpr std::size_t kHeads = 16;
constexpr std::size_t kHeadDim = 256;

std::vector<unsigned char> read_file(const char* path,
                                     std::size_t expected_bytes) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) throw std::runtime_error(std::string("open failed: ") + path);
  const std::size_t bytes = static_cast<std::size_t>(stream.tellg());
  if (bytes != expected_bytes) {
    throw std::runtime_error(std::string("file size mismatch: ") + path);
  }
  stream.seekg(0);
  std::vector<unsigned char> value(bytes);
  stream.read(reinterpret_cast<char*>(value.data()), bytes);
  if (!stream) throw std::runtime_error("file read failed");
  return value;
}

void write_file(const char* path, const void* data, std::size_t bytes) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) throw std::runtime_error(std::string("open failed: ") + path);
  stream.write(static_cast<const char*>(data), bytes);
  if (!stream) throw std::runtime_error("file write failed");
}

void check(hipError_t status, const char* operation) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(status));
  }
}

std::uint16_t float_to_bf16(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  if ((bits & 0x7f800000u) == 0x7f800000u) {
    std::uint16_t upper = static_cast<std::uint16_t>(bits >> 16);
    if ((bits & 0x007fffffu) != 0) upper |= 0x0040u;
    return upper;
  }
  const std::uint32_t lsb = (bits >> 16) & 1u;
  return static_cast<std::uint16_t>((bits + 0x7fffu + lsb) >> 16);
}

float bf16_to_float(std::uint16_t value) {
  const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16;
  float output = 0.0f;
  std::memcpy(&output, &bits, sizeof(output));
  return output;
}

struct Metrics {
  std::size_t elements = 0;
  std::size_t exact = 0;
  double squared_error = 0.0;
  double squared_reference = 0.0;
  double squared_actual = 0.0;
  double dot = 0.0;
  double maximum_absolute = 0.0;
};

void update_metrics(Metrics* metrics, std::uint16_t actual_bits,
                    std::uint16_t expected_bits) {
  ++metrics->elements;
  if (actual_bits == expected_bits) ++metrics->exact;
  const double actual = bf16_to_float(actual_bits);
  const double reference = bf16_to_float(expected_bits);
  const double difference = actual - reference;
  metrics->squared_error += difference * difference;
  metrics->squared_reference += reference * reference;
  metrics->squared_actual += actual * actual;
  metrics->dot += actual * reference;
  metrics->maximum_absolute =
      std::max(metrics->maximum_absolute, std::abs(difference));
}

void print_metrics(const Metrics& metrics) {
  const double relative_l2 = metrics.squared_reference == 0.0
                                 ? 0.0
                                 : std::sqrt(metrics.squared_error /
                                             metrics.squared_reference);
  const double cosine = metrics.squared_reference == 0.0 ||
                                metrics.squared_actual == 0.0
                            ? 0.0
                            : metrics.dot /
                                  std::sqrt(metrics.squared_reference *
                                            metrics.squared_actual);
  std::cout << "{\"elements\":" << metrics.elements
            << ",\"exact_elements\":" << metrics.exact
            << ",\"mismatched_elements\":"
            << metrics.elements - metrics.exact
            << ",\"maximum_absolute_error\":"
            << metrics.maximum_absolute
            << ",\"relative_l2_error\":" << relative_l2
            << ",\"cosine_similarity\":" << cosine << '}';
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 7 && argc != 8) {
    std::cerr << "usage: probe PROVIDER TOKENS Q K V EXPECTED_FULL "
                 "[ACTUAL_BF16]\n";
    return 2;
  }
  try {
    const std::size_t tokens = std::stoull(argv[2]);
    if (tokens == 0 || tokens % 2 != 0) {
      throw std::runtime_error("TOKENS must be positive and even");
    }
    const std::size_t elements = tokens * kFeatures;
    const std::size_t q_bytes = elements * sizeof(std::uint16_t);
    const std::size_t kv_bytes =
        tokens * kKvFeatures * sizeof(std::uint16_t);
    const std::size_t output_bytes = elements * sizeof(float);

    void* handle = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) throw std::runtime_error(dlerror());
    using Prepare = int (*)(unsigned int);
    using Launch = int (*)(const void*, const void*, const void*, void*,
                           unsigned int, void*);
    using Release = int (*)();
    auto prepare = reinterpret_cast<Prepare>(
        dlsym(handle, "qrt_ck_fmha_prepare"));
    auto launch = reinterpret_cast<Launch>(
        dlsym(handle, "qrt_ck_fmha_bf16_launch"));
    auto release = reinterpret_cast<Release>(
        dlsym(handle, "qrt_ck_fmha_release"));
    if (prepare == nullptr || launch == nullptr || release == nullptr ||
        prepare(static_cast<unsigned int>(tokens)) !=
            static_cast<int>(hipSuccess)) {
      throw std::runtime_error("provider prepare failed");
    }

    void* q = nullptr;
    void* k = nullptr;
    void* v = nullptr;
    void* output = nullptr;
    check(hipMalloc(&q, q_bytes), "hipMalloc q");
    check(hipMalloc(&k, kv_bytes), "hipMalloc k");
    check(hipMalloc(&v, kv_bytes), "hipMalloc v");
    check(hipMalloc(&output, output_bytes), "hipMalloc output");
    auto upload = [](const char* path, void* destination, std::size_t bytes) {
      const auto host = read_file(path, bytes);
      check(hipMemcpy(destination, host.data(), bytes, hipMemcpyHostToDevice),
            "hipMemcpy input");
    };
    upload(argv[3], q, q_bytes);
    upload(argv[4], k, kv_bytes);
    upload(argv[5], v, kv_bytes);

    std::vector<double> samples;
    for (int iteration = 0; iteration < 6; ++iteration) {
      hipEvent_t start = nullptr;
      hipEvent_t end = nullptr;
      check(hipEventCreate(&start), "hipEventCreate start");
      check(hipEventCreate(&end), "hipEventCreate end");
      check(hipEventRecord(start), "hipEventRecord start");
      const int status = launch(q, k, v, output,
                                static_cast<unsigned int>(tokens), nullptr);
      if (status != static_cast<int>(hipSuccess)) {
        throw std::runtime_error("provider launch failed: " +
                                 std::to_string(status));
      }
      check(hipEventRecord(end), "hipEventRecord end");
      check(hipEventSynchronize(end), "hipEventSynchronize");
      float elapsed = 0.0f;
      check(hipEventElapsedTime(&elapsed, start, end),
            "hipEventElapsedTime");
      if (iteration > 0) samples.push_back(elapsed);
      check(hipEventDestroy(end), "hipEventDestroy end");
      check(hipEventDestroy(start), "hipEventDestroy start");
    }

    std::vector<float> actual_f32(elements);
    check(hipMemcpy(actual_f32.data(), output, output_bytes,
                    hipMemcpyDeviceToHost),
          "hipMemcpy output");
    const auto expected_bytes =
        read_file(argv[6], elements * sizeof(std::uint16_t));
    const auto* expected =
        reinterpret_cast<const std::uint16_t*>(expected_bytes.data());
    std::vector<std::uint16_t> actual(elements);
    Metrics overall;
    Metrics windows[2];
    Metrics window_heads[2][kHeads];
    struct Mismatch {
      std::size_t token;
      std::size_t head;
      std::size_t dimension;
      std::uint16_t actual_bits;
      std::uint16_t expected_bits;
    };
    std::vector<Mismatch> first_mismatches;
    for (std::size_t index = 0; index < elements; ++index) {
      const std::uint16_t actual_bits = float_to_bf16(actual_f32[index]);
      actual[index] = actual_bits;
      const std::size_t token = index / kFeatures;
      const std::size_t feature = index % kFeatures;
      const std::size_t head = feature / kHeadDim;
      const std::size_t dimension = feature % kHeadDim;
      const std::size_t window = token < tokens / 2 ? 0 : 1;
      update_metrics(&overall, actual_bits, expected[index]);
      update_metrics(&windows[window], actual_bits, expected[index]);
      update_metrics(&window_heads[window][head], actual_bits,
                     expected[index]);
      if (actual_bits != expected[index] && first_mismatches.size() < 32) {
        first_mismatches.push_back(
            {token, head, dimension, actual_bits, expected[index]});
      }
    }
    if (argc == 8) {
      write_file(argv[7], actual.data(),
                 actual.size() * sizeof(std::uint16_t));
    }

    std::sort(samples.begin(), samples.end());
    std::cout << std::setprecision(10)
              << "{\"median_ms\":" << samples[2]
              << ",\"samples_ms\":[" << samples[0] << ',' << samples[1]
              << ',' << samples[2] << ',' << samples[3] << ',' << samples[4]
              << "],\"overall\":";
    print_metrics(overall);
    std::cout << ",\"windows\":[";
    for (std::size_t window = 0; window < 2; ++window) {
      if (window != 0) std::cout << ',';
      std::cout << "{\"window\":" << window << ",\"metrics\":";
      print_metrics(windows[window]);
      std::cout << ",\"heads\":[";
      for (std::size_t head = 0; head < kHeads; ++head) {
        if (head != 0) std::cout << ',';
        std::cout << "{\"head\":" << head << ",\"metrics\":";
        print_metrics(window_heads[window][head]);
        std::cout << '}';
      }
      std::cout << "]}";
    }
    std::cout << "],\"first_mismatches\":[";
    for (std::size_t index = 0; index < first_mismatches.size(); ++index) {
      if (index != 0) std::cout << ',';
      const auto& mismatch = first_mismatches[index];
      std::cout << "{\"token\":" << mismatch.token
                << ",\"head\":" << mismatch.head
                << ",\"dimension\":" << mismatch.dimension
                << ",\"actual_bits\":" << mismatch.actual_bits
                << ",\"expected_bits\":" << mismatch.expected_bits
                << ",\"actual\":" << bf16_to_float(mismatch.actual_bits)
                << ",\"expected\":"
                << bf16_to_float(mismatch.expected_bits) << '}';
    }
    std::cout << "]}\n";

    check(hipFree(output), "hipFree output");
    check(hipFree(v), "hipFree v");
    check(hipFree(k), "hipFree k");
    check(hipFree(q), "hipFree q");
    (void)release();
    (void)dlclose(handle);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
