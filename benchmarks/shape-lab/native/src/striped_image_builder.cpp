// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

struct Entry {
  std::uint64_t target_offset{};
  std::uint64_t aligned_bytes{};
  std::uint64_t payload_bytes{};
  std::uint64_t source_offset{};
  std::string source_path;
};

struct Plan {
  std::uint64_t total_bytes{};
  std::size_t expected_entries{};
  std::vector<Entry> entries;
};

struct LaneResult {
  bool complete{false};
  std::uint64_t payload_bytes{};
  std::uint64_t image_bytes{};
  std::uint64_t source_xor{};
  std::uint64_t source_sum{};
  std::uint64_t target_xor{};
  std::uint64_t target_sum{};
  double build_ms{};
  double verify_ms{};
  std::string error;
};

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

std::vector<std::string> split_tabs(const std::string& line) {
  std::vector<std::string> fields;
  std::size_t start = 0;
  while (true) {
    const std::size_t end = line.find('\t', start);
    fields.push_back(line.substr(start, end == std::string::npos ? end : end - start));
    if (end == std::string::npos) break;
    start = end + 1;
  }
  return fields;
}

std::uint64_t parse_u64(const std::string& value, const char* field) {
  std::size_t consumed = 0;
  const auto parsed = std::stoull(value, &consumed, 10);
  if (consumed != value.size()) {
    throw std::runtime_error(std::string("invalid ") + field + ": " + value);
  }
  return parsed;
}

Plan read_plan(const std::string& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open plan: " + path);
  std::string line;
  if (!std::getline(input, line)) throw std::runtime_error("empty plan: " + path);
  auto header = split_tabs(line);
  if (header.size() != 3 || header[0] != "v1") {
    throw std::runtime_error("invalid plan header: " + path);
  }
  Plan plan;
  plan.total_bytes = parse_u64(header[1], "total bytes");
  plan.expected_entries = static_cast<std::size_t>(parse_u64(header[2], "entry count"));
  while (std::getline(input, line)) {
    if (line.empty()) continue;
    auto fields = split_tabs(line);
    if (fields.size() != 5) throw std::runtime_error("invalid plan row: " + path);
    plan.entries.push_back(
        {parse_u64(fields[0], "target offset"),
         parse_u64(fields[1], "aligned bytes"),
         parse_u64(fields[2], "payload bytes"),
         parse_u64(fields[3], "source offset"), fields[4]});
  }
  if (plan.entries.size() != plan.expected_entries) {
    throw std::runtime_error("plan entry count mismatch: " + path);
  }
  std::uint64_t expected_offset = 0;
  for (const auto& entry : plan.entries) {
    if (entry.target_offset != expected_offset || entry.payload_bytes == 0 ||
        entry.payload_bytes > entry.aligned_bytes || entry.aligned_bytes % 4096 != 0 ||
        entry.payload_bytes % sizeof(std::uint64_t) != 0) {
      throw std::runtime_error("invalid plan geometry: " + path);
    }
    expected_offset += entry.aligned_bytes;
  }
  if (expected_offset != plan.total_bytes) {
    throw std::runtime_error("plan total mismatch: " + path);
  }
  return plan;
}

void update_checksum(const unsigned char* data, std::size_t bytes,
                     std::uint64_t& xor_value, std::uint64_t& sum_value) {
  if (bytes % sizeof(std::uint64_t) != 0) {
    throw std::runtime_error("checksum input is not uint64 aligned");
  }
  const auto* words = reinterpret_cast<const std::uint64_t*>(data);
  for (std::size_t index = 0; index < bytes / sizeof(std::uint64_t); ++index) {
    xor_value ^= words[index];
    sum_value += words[index];
  }
}

void read_exact(int descriptor, unsigned char* data, std::size_t bytes,
                std::uint64_t offset) {
  std::size_t consumed = 0;
  while (consumed < bytes) {
    const ssize_t amount = pread(descriptor, data + consumed, bytes - consumed,
                                 static_cast<off_t>(offset + consumed));
    if (amount < 0 && errno == EINTR) continue;
    if (amount <= 0) {
      throw std::runtime_error("pread failed: " + std::string(std::strerror(errno)));
    }
    consumed += static_cast<std::size_t>(amount);
  }
}

void write_exact(int descriptor, const unsigned char* data, std::size_t bytes,
                 std::uint64_t offset) {
  std::size_t consumed = 0;
  while (consumed < bytes) {
    const ssize_t amount = pwrite(descriptor, data + consumed, bytes - consumed,
                                  static_cast<off_t>(offset + consumed));
    if (amount < 0 && errno == EINTR) continue;
    if (amount <= 0) {
      throw std::runtime_error("pwrite failed: " + std::string(std::strerror(errno)));
    }
    consumed += static_cast<std::size_t>(amount);
  }
}

LaneResult build_lane(const Plan& plan, const std::string& output_path,
                      std::size_t chunk_bytes) {
  LaneResult result;
  int output = -1;
  std::map<std::string, int> sources;
  try {
    if (chunk_bytes == 0 || chunk_bytes % 4096 != 0 ||
        chunk_bytes % sizeof(std::uint64_t) != 0) {
      throw std::runtime_error("invalid chunk size");
    }
    std::vector<unsigned char> buffer(chunk_bytes);
    std::vector<unsigned char> zeros(4096, 0);
    output = open(output_path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    if (output < 0) {
      throw std::runtime_error("output open failed: " + std::string(std::strerror(errno)));
    }
    const int fallocate_status = posix_fallocate(output, 0, static_cast<off_t>(plan.total_bytes));
    if (fallocate_status != 0) {
      throw std::runtime_error("posix_fallocate failed: " + std::string(std::strerror(fallocate_status)));
    }
    const auto build_start = std::chrono::steady_clock::now();
    for (const auto& entry : plan.entries) {
      int source = -1;
      auto found = sources.find(entry.source_path);
      if (found == sources.end()) {
        source = open(entry.source_path.c_str(), O_RDONLY | O_CLOEXEC);
        if (source < 0) {
          throw std::runtime_error("source open failed: " + std::string(std::strerror(errno)));
        }
        sources.emplace(entry.source_path, source);
      } else {
        source = found->second;
      }
      std::uint64_t consumed = 0;
      while (consumed < entry.payload_bytes) {
        const std::size_t wanted = static_cast<std::size_t>(
            std::min<std::uint64_t>(chunk_bytes, entry.payload_bytes - consumed));
        read_exact(source, buffer.data(), wanted, entry.source_offset + consumed);
        update_checksum(buffer.data(), wanted, result.source_xor, result.source_sum);
        write_exact(output, buffer.data(), wanted, entry.target_offset + consumed);
        consumed += wanted;
      }
      const std::uint64_t padding = entry.aligned_bytes - entry.payload_bytes;
      if (padding > zeros.size()) throw std::runtime_error("padding exceeds 4096 bytes");
      if (padding != 0) {
        write_exact(output, zeros.data(), static_cast<std::size_t>(padding),
                    entry.target_offset + entry.payload_bytes);
      }
      result.payload_bytes += entry.payload_bytes;
    }
    if (fdatasync(output) != 0) {
      throw std::runtime_error("fdatasync failed: " + std::string(std::strerror(errno)));
    }
    close(output);
    output = -1;
    result.build_ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - build_start)
                          .count();
    for (const auto& item : sources) close(item.second);
    sources.clear();

    const auto verify_start = std::chrono::steady_clock::now();
    const int verify = open(output_path.c_str(), O_RDONLY | O_CLOEXEC);
    if (verify < 0) {
      throw std::runtime_error("verify open failed: " + std::string(std::strerror(errno)));
    }
    std::uint64_t offset = 0;
    while (offset < plan.total_bytes) {
      const std::size_t wanted = static_cast<std::size_t>(
          std::min<std::uint64_t>(chunk_bytes, plan.total_bytes - offset));
      read_exact(verify, buffer.data(), wanted, offset);
      update_checksum(buffer.data(), wanted, result.target_xor, result.target_sum);
      offset += wanted;
    }
    close(verify);
    result.verify_ms = std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - verify_start)
                           .count();
    result.image_bytes = plan.total_bytes;
    result.complete = result.source_xor == result.target_xor &&
                      result.source_sum == result.target_sum;
    if (!result.complete) result.error = "source/target checksum mismatch";
  } catch (const std::exception& error) {
    result.error = error.what();
    if (output >= 0) close(output);
    for (const auto& item : sources) close(item.second);
  }
  return result;
}

void write_result(const std::string& path, const LaneResult lanes[2],
                  double aggregate_ms, const std::string outputs[2],
                  std::size_t entries0, std::size_t entries1) {
  const bool complete = lanes[0].complete && lanes[1].complete;
  const std::uint64_t aggregate_xor = lanes[0].target_xor ^ lanes[1].target_xor;
  const std::uint64_t aggregate_sum = lanes[0].target_sum + lanes[1].target_sum;
  std::ofstream output(path);
  output << std::setprecision(17)
         << "{\n"
         << "  \"schema\": \"amd395-qwen36-r4/row3074-striped-image-builder/v1\",\n"
         << "  \"complete\": " << (complete ? "true" : "false") << ",\n"
         << "  \"aggregate_wall_ms\": " << aggregate_ms << ",\n"
         << "  \"aggregate_xor\": \"" << hex64(aggregate_xor) << "\",\n"
         << "  \"aggregate_sum\": \"" << hex64(aggregate_sum) << "\",\n"
         << "  \"lanes\": [\n";
  for (int lane = 0; lane < 2; ++lane) {
    output << "    {\"lane\": " << lane << ", \"path\": \""
           << json_escape(outputs[lane]) << "\", \"entries\": "
           << (lane == 0 ? entries0 : entries1)
           << ", \"payload_bytes\": " << lanes[lane].payload_bytes
           << ", \"image_bytes\": " << lanes[lane].image_bytes
           << ", \"source_xor\": \"" << hex64(lanes[lane].source_xor)
           << "\", \"source_sum\": \"" << hex64(lanes[lane].source_sum)
           << "\", \"target_xor\": \"" << hex64(lanes[lane].target_xor)
           << "\", \"target_sum\": \"" << hex64(lanes[lane].target_sum)
           << "\", \"build_ms\": " << lanes[lane].build_ms
           << ", \"verify_ms\": " << lanes[lane].verify_ms
           << ", \"complete\": " << (lanes[lane].complete ? "true" : "false")
           << ", \"error\": \"" << json_escape(lanes[lane].error) << "\"}"
           << (lane == 0 ? "," : "") << "\n";
  }
  output << "  ]\n}\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 7) {
    return 64;
  }
  LaneResult lanes[2];
  try {
    const Plan plans[2] = {read_plan(argv[1]), read_plan(argv[2])};
    const std::string outputs[2] = {argv[3], argv[4]};
    const std::size_t chunk_bytes = static_cast<std::size_t>(parse_u64(argv[6], "chunk bytes"));
    const auto started = std::chrono::steady_clock::now();
    std::thread workers[2] = {
        std::thread([&] { lanes[0] = build_lane(plans[0], outputs[0], chunk_bytes); }),
        std::thread([&] { lanes[1] = build_lane(plans[1], outputs[1], chunk_bytes); })};
    for (auto& worker : workers) worker.join();
    const double aggregate_ms = std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() - started)
                                    .count();
    write_result(argv[5], lanes, aggregate_ms, outputs, plans[0].entries.size(),
                 plans[1].entries.size());
  } catch (const std::exception& error) {
    std::ofstream output(argv[5]);
    output << "{\n  \"schema\": \"amd395-qwen36-r4/row3074-striped-image-builder/v1\",\n"
           << "  \"complete\": false,\n  \"error\": \""
           << json_escape(error.what()) << "\"\n}\n";
    return 2;
  }
  return lanes[0].complete && lanes[1].complete ? 0 : 2;
}
