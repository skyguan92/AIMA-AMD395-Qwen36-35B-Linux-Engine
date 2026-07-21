// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <unordered_set>
#include <vector>

namespace {

constexpr std::size_t kAlignment = 4096;
constexpr int kLanes = 2;
constexpr int kBuffersPerLane = 2;
constexpr int kThreads = 256;
constexpr std::uint64_t kChecksumWordsPerBlock = 1ULL << 20;

struct ChecksumTask {
  const std::uint64_t* data;
  std::uint64_t words;
};

void hip_check(hipError_t status, const char* expression) {
  if (status != hipSuccess) {
    std::ostringstream message;
    message << expression << ": " << hipGetErrorName(status) << " ("
            << hipGetErrorString(status) << ")";
    throw std::runtime_error(message.str());
  }
}

#define HIP_CHECK(expression) hip_check((expression), #expression)

void ignore_hip(hipError_t status) { (void)status; }

std::string json_escape(const std::string& value) {
  std::ostringstream result;
  for (const unsigned char ch : value) {
    switch (ch) {
      case '\\': result << "\\\\"; break;
      case '"': result << "\\\""; break;
      case '\n': result << "\\n"; break;
      case '\r': result << "\\r"; break;
      case '\t': result << "\\t"; break;
      default:
        if (ch < 0x20) {
          result << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                 << static_cast<unsigned>(ch) << std::dec;
        } else {
          result << ch;
        }
    }
  }
  return result.str();
}

std::string hex64(std::uint64_t value) {
  std::ostringstream result;
  result << "0x" << std::hex << std::setw(16) << std::setfill('0') << value;
  return result.str();
}

void write_failure(const char* output_path, const std::string& message,
                   bool cleanup_complete) {
  if (output_path == nullptr) return;
  std::ofstream output(output_path);
  output << "{\n"
         << "  \"schema\": \"amd395-qwen36-r4/row3170-independent-tensor-striped-scatter/v1\",\n"
         << "  \"complete\": false,\n"
         << "  \"error\": \"" << json_escape(message) << "\",\n"
         << "  \"destination_freed_by_native\": false,\n"
         << "  \"cleanup_complete\": "
         << (cleanup_complete ? "true" : "false") << "\n}\n";
}

__global__ void checksum_tasks_kernel(const ChecksumTask* tasks,
                                      std::uint64_t* block_xor,
                                      std::uint64_t* block_sum) {
  __shared__ std::uint64_t xor_values[kThreads];
  __shared__ std::uint64_t sum_values[kThreads];
  const ChecksumTask task = tasks[blockIdx.x];
  std::uint64_t local_xor = 0;
  std::uint64_t local_sum = 0;
  for (std::uint64_t index = threadIdx.x; index < task.words;
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

}  // namespace

extern "C" int torch_owned_striped_tensor_scatter_ingest(
    const char* lane0_path, const char* lane1_path,
    const std::uint64_t* aggregate_offsets,
    const std::uint64_t* payload_bytes,
    const std::uint64_t* destination_ptrs,
    std::size_t tensor_count, std::size_t image_bytes,
    std::size_t lane0_bytes, std::size_t lane1_bytes,
    std::size_t chunk_bytes, std::uint64_t expected_xor,
    std::uint64_t expected_sum, const char* output_path) {
  void* host_buffers[kLanes][kBuffersPerLane] = {};
  hipStream_t streams[kLanes][kBuffersPerLane] = {};
  int descriptors[kLanes] = {-1, -1};
  ChecksumTask* device_tasks = nullptr;
  std::uint64_t* device_block_xor = nullptr;
  std::uint64_t* device_block_sum = nullptr;
  bool cleanup_complete = false;
  try {
    if (lane0_path == nullptr || lane1_path == nullptr ||
        aggregate_offsets == nullptr || payload_bytes == nullptr ||
        destination_ptrs == nullptr || output_path == nullptr) {
      throw std::runtime_error("null argument");
    }
    if (tensor_count == 0 || lane0_bytes == 0 || lane1_bytes == 0 ||
        lane0_bytes % kAlignment != 0 || lane1_bytes % kAlignment != 0 ||
        image_bytes != lane0_bytes + lane1_bytes ||
        chunk_bytes == 0 || chunk_bytes % kAlignment != 0) {
      throw std::runtime_error("invalid aligned byte arguments");
    }

    const std::size_t lane_bytes[kLanes] = {lane0_bytes, lane1_bytes};
    const std::size_t lane_base[kLanes] = {0, lane0_bytes};
    const std::string paths[kLanes] = {lane0_path, lane1_path};
    std::uint64_t total_payload_bytes = 0;
    std::unordered_set<std::uint64_t> unique_destinations;
    bool all_pointer_types_device = true;
    bool all_device_pointers_match = true;
    for (std::size_t index = 0; index < tensor_count; ++index) {
      const std::uint64_t offset = aggregate_offsets[index];
      const std::uint64_t bytes = payload_bytes[index];
      const std::uint64_t next_offset =
          index + 1 < tensor_count ? aggregate_offsets[index + 1] : image_bytes;
      if (destination_ptrs[index] == 0 || offset % kAlignment != 0 ||
          bytes == 0 || bytes % sizeof(std::uint64_t) != 0 ||
          next_offset <= offset || next_offset > image_bytes ||
          bytes > next_offset - offset) {
        throw std::runtime_error("invalid tensor scatter geometry at index " +
                                 std::to_string(index));
      }
      if ((offset < lane0_bytes && offset + bytes > lane0_bytes) ||
          (offset >= lane0_bytes && offset + bytes > image_bytes)) {
        throw std::runtime_error("tensor payload crosses a lane boundary");
      }
      if (!unique_destinations.insert(destination_ptrs[index]).second) {
        throw std::runtime_error("duplicate destination pointer");
      }
      hipPointerAttribute_t attributes{};
      void* pointer = reinterpret_cast<void*>(
          static_cast<std::uintptr_t>(destination_ptrs[index]));
      HIP_CHECK(hipPointerGetAttributes(&attributes, pointer));
      all_pointer_types_device =
          all_pointer_types_device && attributes.type == hipMemoryTypeDevice;
      all_device_pointers_match =
          all_device_pointers_match && attributes.devicePointer == pointer;
      total_payload_bytes += bytes;
    }
    if (aggregate_offsets[0] != 0 ||
        total_payload_bytes > image_bytes) {
      throw std::runtime_error("invalid aggregate scatter geometry");
    }

    hipDeviceProp_t properties{};
    HIP_CHECK(hipGetDeviceProperties(&properties, 0));
    std::vector<std::size_t> host_alignment;
    std::size_t read_chunks[kLanes] = {0, 0};
    for (int lane = 0; lane < kLanes; ++lane) {
      for (int buffer = 0; buffer < kBuffersPerLane; ++buffer) {
        HIP_CHECK(hipHostMalloc(&host_buffers[lane][buffer], chunk_bytes,
                                hipHostMallocDefault));
        HIP_CHECK(hipStreamCreateWithFlags(&streams[lane][buffer],
                                           hipStreamNonBlocking));
        host_alignment.push_back(
            reinterpret_cast<std::uintptr_t>(host_buffers[lane][buffer]) %
            kAlignment);
      }
      descriptors[lane] = open(paths[lane].c_str(),
                               O_RDONLY | O_DIRECT | O_CLOEXEC);
      if (descriptors[lane] < 0) {
        throw std::runtime_error("O_DIRECT open failed: " +
                                 std::string(std::strerror(errno)));
      }
      struct stat status {};
      if (fstat(descriptors[lane], &status) != 0 ||
          static_cast<std::uint64_t>(status.st_size) != lane_bytes[lane]) {
        throw std::runtime_error("lane file size mismatch");
      }
    }
    HIP_CHECK(hipDeviceSynchronize());

    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    std::mutex error_mutex;
    std::string lane_errors[kLanes];
    double lane_ms[kLanes] = {0.0, 0.0};
    std::uint64_t scheduled_payload_bytes[kLanes] = {0, 0};
    auto lane_worker = [&](int lane) {
      ready.fetch_add(1, std::memory_order_release);
      try {
        HIP_CHECK(hipSetDevice(0));
        while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
        const auto start = std::chrono::steady_clock::now();
        std::size_t offset = 0;
        std::size_t chunk = 0;
        std::size_t first_entry = static_cast<std::size_t>(
            std::lower_bound(aggregate_offsets,
                             aggregate_offsets + tensor_count,
                             static_cast<std::uint64_t>(lane_base[lane])) -
            aggregate_offsets);
        while (offset < lane_bytes[lane]) {
          const int buffer = static_cast<int>(chunk % kBuffersPerLane);
          const std::size_t wanted =
              std::min(chunk_bytes, lane_bytes[lane] - offset);
          HIP_CHECK(hipStreamSynchronize(streams[lane][buffer]));
          ssize_t amount;
          do {
            amount = pread(descriptors[lane], host_buffers[lane][buffer],
                           wanted, static_cast<off_t>(offset));
          } while (amount < 0 && errno == EINTR);
          if (amount < 0) {
            throw std::runtime_error("O_DIRECT pread failed: " +
                                     std::string(std::strerror(errno)));
          }
          if (static_cast<std::size_t>(amount) != wanted) {
            throw std::runtime_error("O_DIRECT pread ended early");
          }

          const std::uint64_t global_start = lane_base[lane] + offset;
          const std::uint64_t global_end = global_start + wanted;
          while (first_entry < tensor_count &&
                 aggregate_offsets[first_entry] +
                         payload_bytes[first_entry] <=
                     global_start) {
            ++first_entry;
          }
          for (std::size_t index = first_entry;
               index < tensor_count &&
               aggregate_offsets[index] < global_end;
               ++index) {
            const std::uint64_t entry_start = aggregate_offsets[index];
            const std::uint64_t entry_end = entry_start + payload_bytes[index];
            const std::uint64_t copy_start = std::max(entry_start, global_start);
            const std::uint64_t copy_end = std::min(entry_end, global_end);
            if (copy_start >= copy_end) continue;
            auto* source = static_cast<unsigned char*>(
                               host_buffers[lane][buffer]) +
                           (copy_start - global_start);
            auto* destination = reinterpret_cast<unsigned char*>(
                                    static_cast<std::uintptr_t>(
                                        destination_ptrs[index])) +
                                (copy_start - entry_start);
            const std::size_t copy_bytes =
                static_cast<std::size_t>(copy_end - copy_start);
            HIP_CHECK(hipMemcpyAsync(destination, source, copy_bytes,
                                     hipMemcpyHostToDevice,
                                     streams[lane][buffer]));
            scheduled_payload_bytes[lane] += copy_bytes;
          }
          offset += wanted;
          ++chunk;
        }
        for (int buffer = 0; buffer < kBuffersPerLane; ++buffer) {
          HIP_CHECK(hipStreamSynchronize(streams[lane][buffer]));
        }
        read_chunks[lane] = chunk;
        lane_ms[lane] = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - start)
                            .count();
      } catch (const std::exception& error) {
        std::lock_guard<std::mutex> guard(error_mutex);
        lane_errors[lane] = error.what();
      }
    };

    std::thread workers[kLanes] = {std::thread(lane_worker, 0),
                                   std::thread(lane_worker, 1)};
    while (ready.load(std::memory_order_acquire) != kLanes)
      std::this_thread::yield();
    const auto aggregate_start = std::chrono::steady_clock::now();
    go.store(true, std::memory_order_release);
    for (auto& worker : workers) worker.join();
    const auto aggregate_end = std::chrono::steady_clock::now();
    for (int lane = 0; lane < kLanes; ++lane) {
      if (!lane_errors[lane].empty()) {
        throw std::runtime_error("lane " + std::to_string(lane) + ": " +
                                 lane_errors[lane]);
      }
      close(descriptors[lane]);
      descriptors[lane] = -1;
    }
    const std::uint64_t scheduled_total =
        scheduled_payload_bytes[0] + scheduled_payload_bytes[1];
    if (scheduled_total != total_payload_bytes) {
      throw std::runtime_error("scheduled payload byte mismatch");
    }
    const double aggregate_ms =
        std::chrono::duration<double, std::milli>(aggregate_end -
                                                  aggregate_start)
            .count();
    const double aggregate_gib_s =
        (static_cast<double>(image_bytes) /
         static_cast<double>(1ULL << 30)) /
        (aggregate_ms / 1000.0);

    std::vector<ChecksumTask> checksum_tasks;
    checksum_tasks.reserve(static_cast<std::size_t>(
        (total_payload_bytes +
         kChecksumWordsPerBlock * sizeof(std::uint64_t) - 1) /
        (kChecksumWordsPerBlock * sizeof(std::uint64_t))));
    for (std::size_t index = 0; index < tensor_count; ++index) {
      const auto* pointer = reinterpret_cast<const std::uint64_t*>(
          static_cast<std::uintptr_t>(destination_ptrs[index]));
      std::uint64_t remaining =
          payload_bytes[index] / sizeof(std::uint64_t);
      std::uint64_t consumed = 0;
      while (remaining != 0) {
        const std::uint64_t words =
            std::min(remaining, kChecksumWordsPerBlock);
        checksum_tasks.push_back({pointer + consumed, words});
        consumed += words;
        remaining -= words;
      }
    }
    if (checksum_tasks.empty()) {
      throw std::runtime_error("empty checksum task list");
    }
    const std::size_t task_bytes = checksum_tasks.size() * sizeof(ChecksumTask);
    const std::size_t output_bytes =
        checksum_tasks.size() * sizeof(std::uint64_t);
    HIP_CHECK(hipMalloc(&device_tasks, task_bytes));
    HIP_CHECK(hipMalloc(&device_block_xor, output_bytes));
    HIP_CHECK(hipMalloc(&device_block_sum, output_bytes));
    HIP_CHECK(hipMemcpy(device_tasks, checksum_tasks.data(), task_bytes,
                        hipMemcpyHostToDevice));
    const auto checksum_start = std::chrono::steady_clock::now();
    hipLaunchKernelGGL(checksum_tasks_kernel,
                       dim3(static_cast<unsigned>(checksum_tasks.size())),
                       dim3(kThreads), 0, 0, device_tasks,
                       device_block_xor, device_block_sum);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<std::uint64_t> block_xor(checksum_tasks.size());
    std::vector<std::uint64_t> block_sum(checksum_tasks.size());
    HIP_CHECK(hipMemcpy(block_xor.data(), device_block_xor, output_bytes,
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(block_sum.data(), device_block_sum, output_bytes,
                        hipMemcpyDeviceToHost));
    const double checksum_ms = std::chrono::duration<double, std::milli>(
                                   std::chrono::steady_clock::now() -
                                   checksum_start)
                                   .count();
    std::uint64_t gpu_xor = 0;
    std::uint64_t gpu_sum = 0;
    for (std::size_t index = 0; index < checksum_tasks.size(); ++index) {
      gpu_xor ^= block_xor[index];
      gpu_sum += block_sum[index];
    }
    HIP_CHECK(hipFree(device_tasks));
    device_tasks = nullptr;
    HIP_CHECK(hipFree(device_block_xor));
    device_block_xor = nullptr;
    HIP_CHECK(hipFree(device_block_sum));
    device_block_sum = nullptr;

    for (int lane = 0; lane < kLanes; ++lane) {
      for (int buffer = 0; buffer < kBuffersPerLane; ++buffer) {
        HIP_CHECK(hipStreamDestroy(streams[lane][buffer]));
        streams[lane][buffer] = nullptr;
        HIP_CHECK(hipHostFree(host_buffers[lane][buffer]));
        host_buffers[lane][buffer] = nullptr;
      }
    }
    cleanup_complete = true;

    const bool checksum_equal =
        gpu_xor == expected_xor && gpu_sum == expected_sum;
    std::ofstream output(output_path);
    output << std::setprecision(17)
           << "{\n"
           << "  \"schema\": \"amd395-qwen36-r4/row3170-independent-tensor-striped-scatter/v1\",\n"
           << "  \"complete\": true,\n"
           << "  \"device\": \"" << json_escape(properties.name) << "\",\n"
           << "  \"lane_bytes\": [" << lane0_bytes << ", "
           << lane1_bytes << "],\n"
           << "  \"image_bytes\": " << image_bytes << ",\n"
           << "  \"payload_bytes\": " << total_payload_bytes << ",\n"
           << "  \"padding_bytes\": "
           << (image_bytes - total_payload_bytes) << ",\n"
           << "  \"tensor_count\": " << tensor_count << ",\n"
           << "  \"unique_destination_pointers\": "
           << unique_destinations.size() << ",\n"
           << "  \"chunk_bytes\": " << chunk_bytes << ",\n"
           << "  \"buffers_per_lane\": " << kBuffersPerLane << ",\n"
           << "  \"read_chunks\": [" << read_chunks[0] << ", "
           << read_chunks[1] << "],\n"
           << "  \"host_alignment_mod_4096\": [";
    for (std::size_t index = 0; index < host_alignment.size(); ++index) {
      if (index) output << ", ";
      output << host_alignment[index];
    }
    output << "],\n"
           << "  \"lane_ms\": [" << lane_ms[0] << ", " << lane_ms[1]
           << "],\n"
           << "  \"lane_gib_s\": ["
           << ((static_cast<double>(lane0_bytes) /
                static_cast<double>(1ULL << 30)) /
               (lane_ms[0] / 1000.0))
           << ", "
           << ((static_cast<double>(lane1_bytes) /
                static_cast<double>(1ULL << 30)) /
               (lane_ms[1] / 1000.0))
           << "],\n"
           << "  \"aggregate_ms\": " << aggregate_ms << ",\n"
           << "  \"aggregate_gib_s\": " << aggregate_gib_s << ",\n"
           << "  \"scheduled_payload_bytes\": ["
           << scheduled_payload_bytes[0] << ", "
           << scheduled_payload_bytes[1] << "],\n"
           << "  \"all_pointer_types_device\": "
           << (all_pointer_types_device ? "true" : "false") << ",\n"
           << "  \"all_device_pointers_match\": "
           << (all_device_pointers_match ? "true" : "false") << ",\n"
           << "  \"checksum_tasks\": " << checksum_tasks.size() << ",\n"
           << "  \"checksum_ms\": " << checksum_ms << ",\n"
           << "  \"checksum_padding_rule\": \"image padding is zero and contributes zero to xor/sum\",\n"
           << "  \"expected_xor\": \"" << hex64(expected_xor)
           << "\",\n"
           << "  \"gpu_payload_xor\": \"" << hex64(gpu_xor)
           << "\",\n"
           << "  \"expected_sum\": \"" << hex64(expected_sum)
           << "\",\n"
           << "  \"gpu_payload_sum\": \"" << hex64(gpu_sum)
           << "\",\n"
           << "  \"gpu_payload_checksum_equal\": "
           << (checksum_equal ? "true" : "false") << ",\n"
           << "  \"destination_freed_by_native\": false,\n"
           << "  \"cleanup_complete\": true\n}\n";
    return checksum_equal && all_pointer_types_device &&
                   all_device_pointers_match
               ? 0
               : 2;
  } catch (const std::exception& error) {
    for (int lane = 0; lane < kLanes; ++lane) {
      if (descriptors[lane] >= 0) close(descriptors[lane]);
    }
    if (device_tasks != nullptr) ignore_hip(hipFree(device_tasks));
    if (device_block_xor != nullptr) ignore_hip(hipFree(device_block_xor));
    if (device_block_sum != nullptr) ignore_hip(hipFree(device_block_sum));
    for (int lane = 0; lane < kLanes; ++lane) {
      for (int buffer = 0; buffer < kBuffersPerLane; ++buffer) {
        if (streams[lane][buffer] != nullptr)
          ignore_hip(hipStreamDestroy(streams[lane][buffer]));
        if (host_buffers[lane][buffer] != nullptr)
          ignore_hip(hipHostFree(host_buffers[lane][buffer]));
      }
    }
    cleanup_complete = true;
    write_failure(output_path, error.what(), cleanup_complete);
    return 2;
  }
}
