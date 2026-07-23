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
constexpr int kBuffersPerWorker = 2;
constexpr int kThreads = 256;
constexpr std::uint64_t kChecksumWordsPerBlock = 1ULL << 20;

struct ChecksumTask {
  const std::uint64_t* data;
  std::uint64_t words;
};

struct WorkerState {
  void* buffers[kBuffersPerWorker] = {};
  hipStream_t streams[kBuffersPerWorker] = {};
  std::uint64_t bytes_read = 0;
  std::uint64_t payload_scheduled = 0;
  std::size_t chunks = 0;
  std::size_t shards = 0;
  std::size_t direct_io_shards = 0;
  std::size_t buffered_shards = 0;
  double elapsed_ms = 0.0;
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

std::uint64_t align_down(std::uint64_t value) {
  return value - value % kAlignment;
}

std::uint64_t align_up(std::uint64_t value) {
  const std::uint64_t remainder = value % kAlignment;
  return remainder == 0 ? value : value + (kAlignment - remainder);
}

ssize_t pread_retry(int descriptor, void* buffer, std::size_t bytes,
                    std::uint64_t offset) {
  ssize_t amount;
  do {
    amount = pread(descriptor, buffer, bytes, static_cast<off_t>(offset));
  } while (amount < 0 && errno == EINTR);
  return amount;
}

void write_failure(const char* output_path, const std::string& message,
                   bool cleanup_complete) {
  if (output_path == nullptr) return;
  std::ofstream output(output_path);
  output << "{\n"
         << "  \"schema\": \"aima-amd395-qwen36/direct-safetensors-scatter/v1\",\n"
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

extern "C" int torch_owned_safetensors_tensor_scatter_ingest(
    const char* const* shard_paths, const std::uint64_t* shard_bytes,
    std::size_t shard_count, const std::uint32_t* tensor_shard_indices,
    const std::uint64_t* source_offsets,
    const std::uint64_t* payload_bytes,
    const std::uint64_t* destination_ptrs, std::size_t tensor_count,
    std::size_t chunk_bytes, std::size_t requested_worker_count,
    std::uint64_t expected_xor, std::uint64_t expected_sum,
    const char* output_path) {
  std::vector<WorkerState> workers;
  ChecksumTask* device_tasks = nullptr;
  std::uint64_t* device_block_xor = nullptr;
  std::uint64_t* device_block_sum = nullptr;
  bool cleanup_complete = false;
  try {
    if (shard_paths == nullptr || shard_bytes == nullptr ||
        tensor_shard_indices == nullptr || source_offsets == nullptr ||
        payload_bytes == nullptr || destination_ptrs == nullptr ||
        output_path == nullptr) {
      throw std::runtime_error("null argument");
    }
    if (shard_count == 0 || tensor_count == 0 || chunk_bytes == 0 ||
        chunk_bytes % kAlignment != 0 || requested_worker_count == 0 ||
        requested_worker_count > 16) {
      throw std::runtime_error("invalid count or aligned byte argument");
    }
    const std::size_t worker_count =
        std::min(requested_worker_count, shard_count);

    std::vector<std::string> paths;
    paths.reserve(shard_count);
    std::vector<std::vector<std::size_t>> entries_by_shard(shard_count);
    for (std::size_t shard = 0; shard < shard_count; ++shard) {
      if (shard_paths[shard] == nullptr || shard_paths[shard][0] == '\0' ||
          shard_bytes[shard] == 0) {
        throw std::runtime_error("invalid shard path or size");
      }
      paths.emplace_back(shard_paths[shard]);
      struct stat status {};
      if (stat(paths.back().c_str(), &status) != 0 ||
          static_cast<std::uint64_t>(status.st_size) != shard_bytes[shard]) {
        throw std::runtime_error("checkpoint shard size mismatch: " +
                                 paths.back());
      }
    }

    std::uint64_t total_payload_bytes = 0;
    std::unordered_set<std::uint64_t> unique_destinations;
    bool all_pointer_types_device = true;
    bool all_device_pointers_match = true;
    for (std::size_t index = 0; index < tensor_count; ++index) {
      const std::uint32_t shard = tensor_shard_indices[index];
      const std::uint64_t offset = source_offsets[index];
      const std::uint64_t bytes = payload_bytes[index];
      if (shard >= shard_count || destination_ptrs[index] == 0 || bytes == 0 ||
          bytes % sizeof(std::uint64_t) != 0 ||
          offset > shard_bytes[shard] ||
          bytes > shard_bytes[shard] - offset) {
        throw std::runtime_error("invalid tensor source geometry at index " +
                                 std::to_string(index));
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
      entries_by_shard[shard].push_back(index);
      total_payload_bytes += bytes;
    }
    for (std::size_t shard = 0; shard < shard_count; ++shard) {
      auto& entries = entries_by_shard[shard];
      std::sort(entries.begin(), entries.end(), [&](std::size_t left,
                                                    std::size_t right) {
        return source_offsets[left] < source_offsets[right];
      });
      std::uint64_t previous_end = 0;
      for (const std::size_t index : entries) {
        if (source_offsets[index] < previous_end) {
          throw std::runtime_error("overlapping tensor payloads in shard " +
                                   std::to_string(shard));
        }
        previous_end = source_offsets[index] + payload_bytes[index];
      }
    }

    hipDeviceProp_t properties{};
    HIP_CHECK(hipGetDeviceProperties(&properties, 0));
    workers.resize(worker_count);
    std::vector<std::size_t> host_alignment;
    for (std::size_t worker = 0; worker < worker_count; ++worker) {
      for (int buffer = 0; buffer < kBuffersPerWorker; ++buffer) {
        HIP_CHECK(hipHostMalloc(&workers[worker].buffers[buffer], chunk_bytes,
                                hipHostMallocDefault));
        HIP_CHECK(hipStreamCreateWithFlags(&workers[worker].streams[buffer],
                                           hipStreamNonBlocking));
        host_alignment.push_back(
            reinterpret_cast<std::uintptr_t>(
                workers[worker].buffers[buffer]) %
            kAlignment);
      }
    }
    HIP_CHECK(hipDeviceSynchronize());

    std::atomic<std::size_t> next_shard{0};
    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    std::mutex error_mutex;
    std::vector<std::string> worker_errors(worker_count);
    auto worker_body = [&](std::size_t worker_index) {
      WorkerState& state = workers[worker_index];
      ready.fetch_add(1, std::memory_order_release);
      try {
        HIP_CHECK(hipSetDevice(0));
        while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
        const auto worker_start = std::chrono::steady_clock::now();
        for (;;) {
          const std::size_t shard =
              next_shard.fetch_add(1, std::memory_order_relaxed);
          if (shard >= shard_count) break;
          const auto& entries = entries_by_shard[shard];
          if (entries.empty()) continue;

          int descriptor = -1;
          bool direct_io = true;
          try {
            descriptor = open(paths[shard].c_str(),
                              O_RDONLY | O_DIRECT | O_CLOEXEC);
            if (descriptor < 0) {
              direct_io = false;
              descriptor = open(paths[shard].c_str(), O_RDONLY | O_CLOEXEC);
            }
            if (descriptor < 0) {
              throw std::runtime_error("checkpoint shard open failed: " +
                                       std::string(std::strerror(errno)));
            }
            const std::uint64_t range_start =
                align_down(source_offsets[entries.front()]);
            std::uint64_t range_end = 0;
            for (const std::size_t index : entries) {
              range_end = std::max(
                  range_end, source_offsets[index] + payload_bytes[index]);
            }
            std::uint64_t offset = range_start;
            std::size_t first_entry = 0;
            while (offset < range_end) {
              const int buffer =
                  static_cast<int>(state.chunks % kBuffersPerWorker);
              HIP_CHECK(hipStreamSynchronize(state.streams[buffer]));
              std::uint64_t remaining = range_end - offset;
              std::size_t wanted = static_cast<std::size_t>(
                  std::min<std::uint64_t>(chunk_bytes, remaining));
              if (direct_io) wanted = static_cast<std::size_t>(align_up(wanted));
              ssize_t amount = pread_retry(descriptor, state.buffers[buffer],
                                           wanted, offset);
              if (amount < 0 && direct_io && errno == EINVAL) {
                close(descriptor);
                descriptor = open(paths[shard].c_str(),
                                  O_RDONLY | O_CLOEXEC);
                if (descriptor < 0) {
                  throw std::runtime_error("buffered checkpoint shard open failed: " +
                                           std::string(std::strerror(errno)));
                }
                direct_io = false;
                wanted = static_cast<std::size_t>(
                    std::min<std::uint64_t>(chunk_bytes, remaining));
                amount = pread_retry(descriptor, state.buffers[buffer],
                                     wanted, offset);
              }
              if (amount <= 0) {
                throw std::runtime_error("checkpoint shard pread failed: " +
                                         std::string(std::strerror(errno)));
              }
              const std::uint64_t valid_bytes =
                  std::min<std::uint64_t>(static_cast<std::uint64_t>(amount),
                                          remaining);
              const std::uint64_t window_end = offset + valid_bytes;
              while (first_entry < entries.size()) {
                const std::size_t index = entries[first_entry];
                if (source_offsets[index] + payload_bytes[index] > offset) break;
                ++first_entry;
              }
              for (std::size_t position = first_entry;
                   position < entries.size(); ++position) {
                const std::size_t index = entries[position];
                const std::uint64_t entry_start = source_offsets[index];
                if (entry_start >= window_end) break;
                const std::uint64_t entry_end =
                    entry_start + payload_bytes[index];
                const std::uint64_t copy_start =
                    std::max(entry_start, offset);
                const std::uint64_t copy_end =
                    std::min(entry_end, window_end);
                if (copy_start >= copy_end) continue;
                auto* source = static_cast<unsigned char*>(
                                   state.buffers[buffer]) +
                               (copy_start - offset);
                auto* destination = reinterpret_cast<unsigned char*>(
                                        static_cast<std::uintptr_t>(
                                            destination_ptrs[index])) +
                                    (copy_start - entry_start);
                const std::size_t copy_bytes = static_cast<std::size_t>(
                    copy_end - copy_start);
                HIP_CHECK(hipMemcpyAsync(destination, source, copy_bytes,
                                         hipMemcpyHostToDevice,
                                         state.streams[buffer]));
                state.payload_scheduled += copy_bytes;
              }
              if (!direct_io) {
                (void)posix_fadvise(descriptor, static_cast<off_t>(offset),
                                    static_cast<off_t>(valid_bytes),
                                    POSIX_FADV_DONTNEED);
              }
              state.bytes_read += static_cast<std::uint64_t>(amount);
              ++state.chunks;
              offset += valid_bytes;
              if (valid_bytes < remaining &&
                  static_cast<std::size_t>(amount) < wanted && direct_io &&
                  offset % kAlignment != 0) {
                close(descriptor);
                descriptor = open(paths[shard].c_str(),
                                  O_RDONLY | O_CLOEXEC);
                if (descriptor < 0) {
                  throw std::runtime_error("short-read fallback open failed: " +
                                           std::string(std::strerror(errno)));
                }
                direct_io = false;
              }
            }
            close(descriptor);
            descriptor = -1;
            ++state.shards;
            if (direct_io) {
              ++state.direct_io_shards;
            } else {
              ++state.buffered_shards;
            }
          } catch (...) {
            if (descriptor >= 0) close(descriptor);
            throw;
          }
        }
        for (int buffer = 0; buffer < kBuffersPerWorker; ++buffer) {
          HIP_CHECK(hipStreamSynchronize(state.streams[buffer]));
        }
        state.elapsed_ms = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - worker_start)
                               .count();
      } catch (const std::exception& error) {
        std::lock_guard<std::mutex> guard(error_mutex);
        worker_errors[worker_index] = error.what();
      }
    };

    std::vector<std::thread> threads;
    threads.reserve(worker_count);
    for (std::size_t worker = 0; worker < worker_count; ++worker) {
      threads.emplace_back(worker_body, worker);
    }
    while (ready.load(std::memory_order_acquire) !=
           static_cast<int>(worker_count)) {
      std::this_thread::yield();
    }
    const auto aggregate_start = std::chrono::steady_clock::now();
    go.store(true, std::memory_order_release);
    for (auto& thread : threads) thread.join();
    const double aggregate_ms = std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() -
                                    aggregate_start)
                                    .count();
    for (std::size_t worker = 0; worker < worker_count; ++worker) {
      if (!worker_errors[worker].empty()) {
        throw std::runtime_error("worker " + std::to_string(worker) + ": " +
                                 worker_errors[worker]);
      }
    }
    std::uint64_t scheduled_total = 0;
    std::uint64_t bytes_read_total = 0;
    std::size_t read_chunks_total = 0;
    std::size_t direct_io_shards = 0;
    std::size_t buffered_shards = 0;
    for (const auto& worker : workers) {
      scheduled_total += worker.payload_scheduled;
      bytes_read_total += worker.bytes_read;
      read_chunks_total += worker.chunks;
      direct_io_shards += worker.direct_io_shards;
      buffered_shards += worker.buffered_shards;
    }
    if (scheduled_total != total_payload_bytes) {
      throw std::runtime_error("scheduled payload byte mismatch");
    }

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

    for (auto& worker : workers) {
      for (int buffer = 0; buffer < kBuffersPerWorker; ++buffer) {
        HIP_CHECK(hipStreamDestroy(worker.streams[buffer]));
        worker.streams[buffer] = nullptr;
        HIP_CHECK(hipHostFree(worker.buffers[buffer]));
        worker.buffers[buffer] = nullptr;
      }
    }
    cleanup_complete = true;

    const bool checksum_equal =
        gpu_xor == expected_xor && gpu_sum == expected_sum;
    const bool passed = checksum_equal && all_pointer_types_device &&
                        all_device_pointers_match;
    const double payload_gib_s =
        (static_cast<double>(total_payload_bytes) /
         static_cast<double>(1ULL << 30)) /
        (aggregate_ms / 1000.0);
    std::ofstream output(output_path);
    output << std::setprecision(17)
           << "{\n"
           << "  \"schema\": \"aima-amd395-qwen36/direct-safetensors-scatter/v1\",\n"
           << "  \"complete\": " << (passed ? "true" : "false") << ",\n"
           << "  \"device\": \"" << json_escape(properties.name) << "\",\n"
           << "  \"shard_count\": " << shard_count << ",\n"
           << "  \"tensor_count\": " << tensor_count << ",\n"
           << "  \"unique_destination_pointers\": "
           << unique_destinations.size() << ",\n"
           << "  \"payload_bytes\": " << total_payload_bytes << ",\n"
           << "  \"source_bytes_read\": " << bytes_read_total << ",\n"
           << "  \"scheduled_payload_bytes\": " << scheduled_total << ",\n"
           << "  \"chunk_bytes\": " << chunk_bytes << ",\n"
           << "  \"worker_count\": " << worker_count << ",\n"
           << "  \"buffers_per_worker\": " << kBuffersPerWorker << ",\n"
           << "  \"read_chunks\": " << read_chunks_total << ",\n"
           << "  \"direct_io_shards\": " << direct_io_shards << ",\n"
           << "  \"buffered_shards\": " << buffered_shards << ",\n"
           << "  \"host_alignment_mod_4096\": [";
    for (std::size_t index = 0; index < host_alignment.size(); ++index) {
      if (index) output << ", ";
      output << host_alignment[index];
    }
    output << "],\n  \"workers\": [\n";
    for (std::size_t index = 0; index < workers.size(); ++index) {
      const auto& worker = workers[index];
      output << "    {\"worker\": " << index
             << ", \"elapsed_ms\": " << worker.elapsed_ms
             << ", \"shards\": " << worker.shards
             << ", \"chunks\": " << worker.chunks
             << ", \"bytes_read\": " << worker.bytes_read
             << ", \"payload_scheduled\": "
             << worker.payload_scheduled << "}"
             << (index + 1 == workers.size() ? "" : ",") << "\n";
    }
    output << "  ],\n"
           << "  \"aggregate_ms\": " << aggregate_ms << ",\n"
           << "  \"payload_gib_s\": " << payload_gib_s << ",\n"
           << "  \"all_pointer_types_device\": "
           << (all_pointer_types_device ? "true" : "false") << ",\n"
           << "  \"all_device_pointers_match\": "
           << (all_device_pointers_match ? "true" : "false") << ",\n"
           << "  \"checksum_tasks\": " << checksum_tasks.size() << ",\n"
           << "  \"checksum_ms\": " << checksum_ms << ",\n"
           << "  \"expected_xor\": \"" << hex64(expected_xor) << "\",\n"
           << "  \"gpu_payload_xor\": \"" << hex64(gpu_xor) << "\",\n"
           << "  \"expected_sum\": \"" << hex64(expected_sum) << "\",\n"
           << "  \"gpu_payload_sum\": \"" << hex64(gpu_sum) << "\",\n"
           << "  \"gpu_payload_checksum_equal\": "
           << (checksum_equal ? "true" : "false") << ",\n"
           << "  \"destination_freed_by_native\": false,\n"
           << "  \"cleanup_complete\": true\n}\n";
    return passed ? 0 : 2;
  } catch (const std::exception& error) {
    if (device_tasks != nullptr) ignore_hip(hipFree(device_tasks));
    if (device_block_xor != nullptr) ignore_hip(hipFree(device_block_xor));
    if (device_block_sum != nullptr) ignore_hip(hipFree(device_block_sum));
    for (auto& worker : workers) {
      for (int buffer = 0; buffer < kBuffersPerWorker; ++buffer) {
        if (worker.streams[buffer] != nullptr)
          ignore_hip(hipStreamDestroy(worker.streams[buffer]));
        if (worker.buffers[buffer] != nullptr)
          ignore_hip(hipHostFree(worker.buffers[buffer]));
      }
    }
    cleanup_complete = true;
    write_failure(output_path, error.what(), cleanup_complete);
    return 2;
  }
}
