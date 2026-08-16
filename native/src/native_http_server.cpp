// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_http_server.h"

#include "aima/native_chat_protocol.h"
#include "aima/native_resident_engine.h"
#include "aima/native_tokenizer.h"
#include "aima/native_vl_request.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <nlohmann/json.hpp>
#include <signal.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/un.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

namespace aima {
namespace {

using Json = NativeOrderedJson;
constexpr const char* kModelId = "aima-amd395-qwen36-35b";
constexpr std::size_t kMaximumRequestBytes = 1024 * 1024;
std::atomic<bool> g_shutdown{false};

double elapsed_ms(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - start)
      .count();
}

void signal_handler(int) { g_shutdown.store(true); }

class FileDescriptor {
 public:
  explicit FileDescriptor(int value = -1) : value_(value) {}
  ~FileDescriptor() {
    if (value_ >= 0) (void)::close(value_);
  }
  FileDescriptor(const FileDescriptor&) = delete;
  FileDescriptor& operator=(const FileDescriptor&) = delete;
  int get() const { return value_; }
  int release() {
    const int value = value_;
    value_ = -1;
    return value;
  }

 private:
  int value_ = -1;
};

void systemd_notify(std::string_view state) noexcept {
  const char* value = std::getenv("NOTIFY_SOCKET");
  if (value == nullptr || *value == '\0' || state.empty()) return;
  const std::string socket_name(value);
  const bool abstract = socket_name.front() == '@';
  const std::size_t name_size = socket_name.size() - (abstract ? 1 : 0);
  sockaddr_un address{};
  if (name_size == 0 || name_size >= sizeof(address.sun_path)) return;
  address.sun_family = AF_UNIX;
  const char* name = socket_name.data() + (abstract ? 1 : 0);
  const std::size_t prefix = offsetof(sockaddr_un, sun_path);
  socklen_t address_size = 0;
  if (abstract) {
    address.sun_path[0] = '\0';
    std::memcpy(address.sun_path + 1, name, name_size);
    address_size = static_cast<socklen_t>(prefix + 1 + name_size);
  } else {
    std::memcpy(address.sun_path, name, name_size);
    address.sun_path[name_size] = '\0';
    address_size = static_cast<socklen_t>(prefix + name_size + 1);
  }
  FileDescriptor socket(
      ::socket(AF_UNIX, SOCK_DGRAM | SOCK_CLOEXEC, 0));
  if (socket.get() < 0) return;
  (void)::sendto(socket.get(), state.data(), state.size(), MSG_NOSIGNAL,
                 reinterpret_cast<const sockaddr*>(&address), address_size);
}

struct HttpRequest {
  std::string method;
  std::string path;
  std::unordered_map<std::string, std::string> headers;
  std::string body;
};

std::string lower_ascii(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char ch) {
                   return static_cast<char>(std::tolower(ch));
                 });
  return value;
}

std::string trim_ascii(std::string value) {
  const auto space = [](unsigned char ch) { return std::isspace(ch) != 0; };
  while (!value.empty() && space(value.front())) value.erase(value.begin());
  while (!value.empty() && space(value.back())) value.pop_back();
  return value;
}

bool try_send_all(int fd, std::string_view value) {
  std::size_t sent = 0;
  while (sent < value.size()) {
    const ssize_t result =
        ::send(fd, value.data() + sent, value.size() - sent, MSG_NOSIGNAL);
    if (result < 0 && errno == EINTR) continue;
    if (result <= 0) return false;
    sent += static_cast<std::size_t>(result);
  }
  return true;
}

bool send_http_chunk(int fd, std::string_view value) {
  std::ostringstream size;
  size << std::hex << value.size() << "\r\n";
  return try_send_all(fd, size.str()) && try_send_all(fd, value) &&
         try_send_all(fd, "\r\n");
}

bool send_sse_event(int fd, std::string_view data) {
  std::string event = "data: ";
  event.append(data);
  event += "\n\n";
  return send_http_chunk(fd, event);
}

bool begin_sse(int fd) {
  return try_send_all(
      fd,
      "HTTP/1.1 200 OK\r\n"
      "Content-Type: text/event-stream; charset=utf-8\r\n"
      "Cache-Control: no-cache\r\n"
      "X-Accel-Buffering: no\r\n"
      "Transfer-Encoding: chunked\r\n"
      "Connection: close\r\n"
      "X-Content-Type-Options: nosniff\r\n\r\n");
}

std::string status_text(int status) {
  switch (status) {
    case 200: return "OK";
    case 400: return "Bad Request";
    case 401: return "Unauthorized";
    case 408: return "Request Timeout";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 413: return "Payload Too Large";
    case 500: return "Internal Server Error";
    default: return "Error";
  }
}

bool send_json(int fd, int status, const Json& payload,
               std::string_view extra_headers = {}) {
  const std::string body = payload.dump();
  std::string header =
      "HTTP/1.1 " + std::to_string(status) + " " + status_text(status) +
      "\r\nContent-Type: application/json\r\nContent-Length: " +
      std::to_string(body.size()) +
      "\r\nConnection: close\r\nX-Content-Type-Options: nosniff\r\n";
  header.append(extra_headers);
  header += "\r\n";
  return try_send_all(fd, header) && try_send_all(fd, body);
}

Json error_payload(const std::string& message, const std::string& code,
                   const std::string& type = "invalid_request_error") {
  return {{"error", {{"message", message},
                      {"type", type},
                      {"param", nullptr},
                      {"code", code}}}};
}

ssize_t receive_before(int fd, void* buffer, std::size_t size,
                       std::chrono::steady_clock::time_point deadline,
                       const char* timeout_message) {
  while (true) {
    const auto now = std::chrono::steady_clock::now();
    if (now >= deadline) {
      throw std::system_error(std::make_error_code(std::errc::timed_out),
                              timeout_message);
    }
    const auto remaining =
        std::chrono::duration_cast<std::chrono::microseconds>(deadline - now);
    timeval timeout{};
    timeout.tv_sec = static_cast<time_t>(remaining.count() / 1000000);
    timeout.tv_usec =
        static_cast<suseconds_t>(std::max<std::int64_t>(1, remaining.count() % 1000000));
    if (::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout,
                     sizeof(timeout)) != 0) {
      throw std::runtime_error("failed to configure request read deadline");
    }
    const ssize_t count = ::recv(fd, buffer, size, 0);
    if (count < 0 && errno == EINTR) continue;
    if (count < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
      throw std::system_error(std::make_error_code(std::errc::timed_out),
                              timeout_message);
    }
    return count;
  }
}

HttpRequest read_request(int fd, std::size_t timeout_ms) {
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(timeout_ms);
  std::string wire;
  wire.reserve(8192);
  std::size_t header_end = std::string::npos;
  while (header_end == std::string::npos) {
    char buffer[8192];
    const ssize_t count = receive_before(
        fd, buffer, sizeof(buffer), deadline, "HTTP request header timed out");
    if (count <= 0) throw std::runtime_error("incomplete HTTP request");
    wire.append(buffer, static_cast<std::size_t>(count));
    if (wire.size() > 65536) {
      throw std::runtime_error("HTTP header is too large");
    }
    header_end = wire.find("\r\n\r\n");
  }

  const std::size_t first_line_end = wire.find("\r\n");
  if (first_line_end == std::string::npos) {
    throw std::runtime_error("invalid HTTP request line");
  }
  const std::string first_line = wire.substr(0, first_line_end);
  const std::size_t method_end = first_line.find(' ');
  const std::size_t path_end =
      method_end == std::string::npos
          ? std::string::npos
          : first_line.find(' ', method_end + 1);
  if (method_end == std::string::npos || path_end == std::string::npos ||
      first_line.substr(path_end + 1).rfind("HTTP/1.", 0) != 0) {
    throw std::runtime_error("invalid HTTP request line");
  }

  HttpRequest request;
  request.method = first_line.substr(0, method_end);
  request.path = first_line.substr(method_end + 1,
                                   path_end - method_end - 1);
  std::size_t cursor = first_line_end + 2;
  while (cursor < header_end) {
    const std::size_t end = wire.find("\r\n", cursor);
    if (end == std::string::npos || end > header_end) {
      throw std::runtime_error("invalid HTTP header");
    }
    const std::string line = wire.substr(cursor, end - cursor);
    const std::size_t colon = line.find(':');
    if (colon == std::string::npos) {
      throw std::runtime_error("invalid HTTP header");
    }
    request.headers.emplace(lower_ascii(trim_ascii(line.substr(0, colon))),
                            trim_ascii(line.substr(colon + 1)));
    cursor = end + 2;
  }

  std::size_t content_length = 0;
  const auto length = request.headers.find("content-length");
  if (length != request.headers.end()) {
    std::size_t consumed = 0;
    try {
      content_length = std::stoull(length->second, &consumed);
    } catch (...) {
      throw std::runtime_error("invalid Content-Length");
    }
    if (consumed != length->second.size()) {
      throw std::runtime_error("invalid Content-Length");
    }
  }
  if (content_length > kMaximumRequestBytes) {
    throw std::length_error("HTTP request body is too large");
  }
  const std::size_t body_begin = header_end + 4;
  while (wire.size() - body_begin < content_length) {
    char buffer[8192];
    const std::size_t remaining =
        content_length - (wire.size() - body_begin);
    const ssize_t count = receive_before(
        fd, buffer, std::min(sizeof(buffer), remaining), deadline,
        "HTTP request body timed out");
    if (count <= 0) throw std::runtime_error("incomplete HTTP request body");
    wire.append(buffer, static_cast<std::size_t>(count));
  }
  request.body = wire.substr(body_begin, content_length);
  return request;
}

std::size_t parse_size(const std::string& value, const char* name) {
  std::size_t consumed = 0;
  unsigned long long parsed = 0;
  try {
    parsed = std::stoull(value, &consumed);
  } catch (...) {
    throw std::runtime_error(std::string(name) + " must be an integer");
  }
  if (consumed != value.size() || parsed == 0 ||
      parsed > std::numeric_limits<std::size_t>::max()) {
    throw std::runtime_error(std::string(name) + " is invalid");
  }
  return static_cast<std::size_t>(parsed);
}

std::vector<std::size_t> parse_size_list(const std::string& value,
                                         const char* name) {
  std::vector<std::size_t> result;
  std::size_t begin = 0;
  while (begin < value.size()) {
    const std::size_t end = value.find(',', begin);
    const std::string item = value.substr(
        begin, end == std::string::npos ? std::string::npos : end - begin);
    if (item.empty()) {
      throw std::runtime_error(std::string(name) +
                               " contains an empty layer index");
    }
    result.push_back(parse_size(item, name));
    if (end == std::string::npos) break;
    begin = end + 1;
  }
  return result;
}

int parse_port(const std::string& value) {
  const std::size_t parsed = parse_size(value, "--port");
  if (parsed > 65535) throw std::runtime_error("--port is invalid");
  return static_cast<int>(parsed);
}

std::string read_api_key(const std::filesystem::path& path) {
  FileDescriptor descriptor(
      ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
  if (descriptor.get() < 0) {
    throw std::runtime_error(
        "--api-key-file must name a readable, non-symlink regular file");
  }
  struct stat metadata {};
  if (::fstat(descriptor.get(), &metadata) != 0 ||
      !S_ISREG(metadata.st_mode)) {
    throw std::runtime_error("--api-key-file must name a regular file");
  }
  if ((metadata.st_mode & 0027) != 0) {
    throw std::runtime_error(
        "--api-key-file must use mode 0640 or stricter");
  }
  if (metadata.st_size <= 0 || metadata.st_size > 4097) {
    throw std::runtime_error("--api-key-file must contain 16 to 4096 bytes");
  }
  std::string content;
  content.reserve(static_cast<std::size_t>(metadata.st_size));
  char buffer[4097];
  while (true) {
    const ssize_t count = ::read(descriptor.get(), buffer, sizeof(buffer));
    if (count < 0 && errno == EINTR) continue;
    if (count < 0) {
      throw std::runtime_error("failed to read --api-key-file");
    }
    if (count == 0) break;
    content.append(buffer, static_cast<std::size_t>(count));
    if (content.size() > sizeof(buffer)) {
      throw std::runtime_error(
          "--api-key-file must contain 16 to 4096 bytes");
    }
  }
  std::string key = trim_ascii(std::move(content));
  if (key.size() < 16 || key.size() > 4096 ||
      key.find_first_of("\r\n") != std::string::npos) {
    throw std::runtime_error(
        "--api-key-file must contain one 16 to 4096 byte line");
  }
  return key;
}

bool is_loopback_address(const std::string& host) {
  in_addr address{};
  if (::inet_pton(AF_INET, host.c_str(), &address) != 1) return false;
  return (ntohl(address.s_addr) & 0xff000000U) == 0x7f000000U;
}

bool constant_time_equal(std::string_view left, std::string_view right) {
  std::size_t difference = left.size() ^ right.size();
  const std::size_t count = std::max(left.size(), right.size());
  for (std::size_t index = 0; index < count; ++index) {
    const unsigned char lhs =
        index < left.size() ? static_cast<unsigned char>(left[index]) : 0;
    const unsigned char rhs =
        index < right.size() ? static_cast<unsigned char>(right[index]) : 0;
    difference |= static_cast<std::size_t>(lhs ^ rhs);
  }
  return difference == 0;
}

bool authorized(const HttpRequest& request, const std::string& api_key) {
  if (api_key.empty()) return true;
  const auto header = request.headers.find("authorization");
  if (header == request.headers.end()) return false;
  constexpr std::string_view prefix = "Bearer ";
  if (header->second.size() < prefix.size() ||
      lower_ascii(header->second.substr(0, prefix.size())) != "bearer ") {
    return false;
  }
  return constant_time_equal(
      std::string_view(header->second).substr(prefix.size()), api_key);
}

void configure_client_timeout(int fd, std::size_t milliseconds) {
  timeval timeout{};
  timeout.tv_sec = static_cast<time_t>(milliseconds / 1000);
  timeout.tv_usec = static_cast<suseconds_t>((milliseconds % 1000) * 1000);
  if (::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout,
                   sizeof(timeout)) != 0 ||
      ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout,
                   sizeof(timeout)) != 0) {
    throw std::runtime_error("failed to configure client socket timeout");
  }
}

struct ServerOptions {
  NativeResidentEngineOptions engine;
  NativeMediaPolicy media_policy;
  std::uint64_t media_cache_capacity_bytes =
      kNativeVlDefaultMediaCacheBytes;
  std::string host = "127.0.0.1";
  int port = 8000;
  std::size_t maximum_requests = 0;
  std::size_t request_timeout_ms = 15000;
  std::string api_key;
  bool http_shutdown = true;
  bool allow_insecure_remote = false;
};

bool requires_authentication(const HttpRequest& request,
                             const ServerOptions& options) {
  return !options.api_key.empty() &&
         (request.path == "/v1/models" ||
          request.path == "/v1/chat/completions" ||
          (options.http_shutdown && request.path == "/shutdown"));
}

ServerOptions parse_options(int argc, char** argv) {
  ServerOptions options;
  options.engine.weights.native_report =
      std::filesystem::absolute("native-http-weight-load.json");
  bool have_model = false;
  bool cache_capacity_explicit = false;
  std::filesystem::path api_key_file;
  for (int index = 2; index < argc; ++index) {
    const std::string argument = argv[index];
    auto next = [&](const char* name) {
      if (++index >= argc) {
        throw std::runtime_error(std::string(name) + " requires a value");
      }
      return std::string(argv[index]);
    };
    if (argument == "--model-dir") {
      options.engine.weights.model_dir =
          std::filesystem::absolute(next("--model-dir"));
      have_model = true;
    } else if (argument == "--fmha-provider" ||
               argument == "--ck-provider") {
      options.engine.ck_provider =
          std::filesystem::absolute(next("--fmha-provider"));
    } else if (argument == "--secondary-fmha-provider") {
      options.engine.secondary_fmha_provider = std::filesystem::absolute(
          next("--secondary-fmha-provider"));
    } else if (argument == "--secondary-fmha-layers") {
      options.engine.secondary_fmha_layers = parse_size_list(
          next("--secondary-fmha-layers"),
          "--secondary-fmha-layers");
    } else if (argument == "--vision-attention-image") {
      options.engine.vision_attention_image = std::filesystem::absolute(
          next("--vision-attention-image"));
    } else if (argument == "--host") {
      options.host = next("--host");
    } else if (argument == "--port") {
      options.port = parse_port(next("--port"));
    } else if (argument == "--max-requests") {
      options.maximum_requests =
          parse_size(next("--max-requests"), "--max-requests");
    } else if (argument == "--request-timeout-ms") {
      options.request_timeout_ms =
          parse_size(next("--request-timeout-ms"), "--request-timeout-ms");
      if (options.request_timeout_ms > 600000) {
        throw std::runtime_error(
            "--request-timeout-ms must not exceed 600000");
      }
    } else if (argument == "--api-key-file") {
      api_key_file = std::filesystem::absolute(next("--api-key-file"));
    } else if (argument == "--disable-http-shutdown") {
      options.http_shutdown = false;
    } else if (argument == "--allow-insecure-remote") {
      options.allow_insecure_remote = true;
    } else if (argument == "--allowed-local-media-path") {
      options.media_policy.allowed_local_roots.push_back(
          std::filesystem::absolute(next("--allowed-local-media-path")));
    } else if (argument == "--allowed-media-domain" ||
               argument == "--allowed-media-domains") {
      options.media_policy.allowed_media_domains.push_back(
          next("--allowed-media-domain"));
    } else if (argument == "--allowed-private-media-domain") {
      options.media_policy.allowed_private_media_domains.push_back(
          next("--allowed-private-media-domain"));
    } else if (argument == "--remote-tls-ca-bundle") {
      options.media_policy.remote_tls_ca_bundle =
          std::filesystem::absolute(next("--remote-tls-ca-bundle"));
    } else if (argument == "--disable-media-cache") {
      options.media_cache_capacity_bytes = 0;
    } else if (argument == "--media-cache-capacity-bytes") {
      options.media_cache_capacity_bytes =
          parse_size(next("--media-cache-capacity-bytes"),
                     "--media-cache-capacity-bytes");
      if (options.media_cache_capacity_bytes >
          kNativeVlDefaultMediaCacheBytes) {
        throw std::runtime_error(
            "--media-cache-capacity-bytes must not exceed 4294967296");
      }
    } else if (argument == "--cache-capacity") {
      options.engine.cache_capacity =
          parse_size(next("--cache-capacity"), "--cache-capacity");
      cache_capacity_explicit = true;
    } else if (argument == "--context-tokens") {
      options.engine.prompt_tokens =
          parse_size(next("--context-tokens"), "--context-tokens");
    } else if (argument == "--report") {
      options.engine.weights.native_report =
          std::filesystem::absolute(next("--report"));
    } else if (argument == "--device") {
      options.engine.weights.device =
          static_cast<int>(parse_size(next("--device"), "--device"));
    } else if (argument == "--workers") {
      options.engine.weights.worker_count =
          parse_size(next("--workers"), "--workers");
    } else if (argument == "--chunk-bytes") {
      options.engine.weights.chunk_bytes =
          parse_size(next("--chunk-bytes"), "--chunk-bytes");
    } else {
      throw std::runtime_error("unknown native serve argument: " + argument);
    }
  }
  if (!have_model) {
    throw std::runtime_error("native serve requires --model-dir");
  }
  if (!cache_capacity_explicit) {
    options.engine.cache_capacity = options.engine.prompt_tokens + 1024;
  }
  if (!api_key_file.empty()) options.api_key = read_api_key(api_key_file);
  if (!is_loopback_address(options.host) && options.api_key.empty() &&
      !options.allow_insecure_remote) {
    throw std::runtime_error(
        "non-loopback --host requires --api-key-file; "
        "--allow-insecure-remote is an explicit unsafe override");
  }
  return options;
}

void require_number(const Json& request, const char* field, double expected) {
  if (!request.contains(field)) return;
  if (!request[field].is_number() ||
      request[field].get<double>() != expected) {
    throw std::invalid_argument(std::string(field) +
                                " is outside the deterministic contract");
  }
}

Json request_metrics_json(const NativeResidentRequestMetrics& metrics) {
  Json result = {{"runtime", "native-resident-q" +
                          std::to_string(metrics.prompt_tokens)},
          {"prompt_tokens", metrics.prompt_tokens},
          {"oracle_tensor_reads", metrics.oracle_tensor_reads},
          {"request_index", metrics.request_index},
          {"model_loads", metrics.model_loads},
          {"client_cancelled", metrics.client_cancelled},
          {"output_token_ids_sha256", metrics.output_token_ids_sha256},
          {"prefill_tokens_per_second", metrics.prefill_tokens_per_second},
          {"decode_tokens_per_second", metrics.decode_tokens_per_second},
          {"ttft_ms", metrics.prefill_wall_ms},
          {"request_wall_ms", metrics.request_wall_ms},
          {"prompt_execution", metrics.prompt_execution},
          {"aot_prefill_tokens", metrics.aot_prefill_tokens},
          {"aot_prefill_bucket_tokens",
           metrics.aot_prefill_bucket_tokens},
          {"aot_prefill_segments", metrics.aot_prefill_segments},
          {"padded_prefill_tokens", metrics.padded_prefill_tokens},
          {"structured_decoding",
           {{"enabled", metrics.constrained_decoding},
            {"token_selections", metrics.constrained_token_selections},
            {"token_mask_upload_bytes",
             metrics.constrained_token_mask_upload_bytes}}},
          {"mrope", {{"enabled", metrics.mrope_enabled},
                     {"position_delta", metrics.mrope_position_delta},
                     {"position_upload_bytes",
                      metrics.mrope_position_upload_bytes},
                     {"full_attention_launches",
                      metrics.mrope_full_attention_launches},
                     {"fmha_launches",
                      metrics.prefill_ck_fmha_launches},
                     {"unified_attention_launches",
                      metrics.prefill_vl_unified_attention_launches},
                     {"decode_steps", metrics.mrope_decode_steps}}},
          {"vl", {{"enabled", metrics.vl_enabled},
                  {"media_count", metrics.vl_media_count},
                  {"image_count", metrics.vl_image_count},
                  {"video_count", metrics.vl_video_count},
                  {"source_bytes", metrics.vl_source_bytes},
                  {"vision_patches", metrics.vl_vision_patches},
                  {"visual_tokens", metrics.vl_visual_tokens},
                  {"media_cache_hits", metrics.vl_media_cache_hits},
                  {"media_cache_misses", metrics.vl_media_cache_misses},
                  {"media_cache_entries", metrics.vl_media_cache_entries},
                  {"media_cache_resident_bytes",
                   metrics.vl_media_cache_resident_bytes},
                  {"vision_batch_count",
                   metrics.vl_vision_batch_count},
                  {"vision_max_batch_patches",
                   metrics.vl_vision_max_batch_patches},
                  {"vision_max_batch_tokens",
                   metrics.vl_vision_max_batch_tokens},
                  {"vision_plan_cache_hit",
                   metrics.vl_vision_plan_cache_hit},
                  {"vision_plan_cache_entries",
                   metrics.vl_vision_plan_cache_entries},
                  {"host_to_device_bytes",
                   metrics.vl_host_to_device_bytes},
                  {"media_load_decode_wall_ms",
                   metrics.vl_media_load_decode_wall_ms},
                  {"media_load_wall_ms",
                   metrics.vl_media_load_wall_ms},
                  {"media_decode_wall_ms",
                   metrics.vl_media_decode_wall_ms},
                  {"processor_wall_ms", metrics.vl_processor_wall_ms},
                  {"vision_plan_build_wall_ms",
                   metrics.vl_vision_plan_build_wall_ms},
                  {"vision_encode_wall_ms",
                   metrics.vl_vision_encode_wall_ms},
                  {"embedding_injection_wall_ms",
                   metrics.vl_embedding_injection_wall_ms}}},
          {"cold_prompt_decode_tokens", metrics.cold_prompt_decode_tokens},
          {"cold_prompt_decode_wall_ms",
           metrics.cold_prompt_decode_wall_ms},
          {"prefix_cache", {{"implemented", true},
                            {"scope", "capacity-bounded-variable-exact-or-append-lru"},
                            {"lookup", metrics.prefix_cache_lookup},
                            {"matched_tokens",
                             metrics.prefix_cache_matched_tokens},
                            {"suffix_tokens",
                             metrics.prefix_cache_suffix_tokens},
                            {"hits", metrics.prefix_cache_hits},
                            {"misses", metrics.prefix_cache_misses},
                            {"transfer_bytes",
                             metrics.prefix_cache_transfer_bytes},
                            {"suffix_decode_tokens",
                             metrics.prefix_cache_suffix_decode_tokens},
                            {"suffix_aot_launches",
                             metrics.prefix_cache_suffix_aot_launches},
                            {"suffix_native_launches",
                             metrics.prefix_cache_suffix_native_launches},
                            {"suffix_wall_ms",
                             metrics.prefix_cache_suffix_wall_ms}}}};
  if (!metrics.prompt_token_ids_sha256.empty()) {
    result["prompt_token_ids_sha256"] = metrics.prompt_token_ids_sha256;
  }
  if (!metrics.output_token_ids_canonical_sha256.empty()) {
    result["output_token_ids_canonical_sha256"] =
        metrics.output_token_ids_canonical_sha256;
  }
  return result;
}

Json oracle_comparison_json(const NativeOracleComparison& comparison) {
  return {{"label", comparison.label},
          {"dtype", comparison.dtype},
          {"elements", comparison.elements},
          {"exact_elements", comparison.exact_elements},
          {"finite_elements", comparison.finite_elements},
          {"first_mismatch_provided", comparison.first_mismatch_provided},
          {"first_mismatch_index", comparison.first_mismatch_index},
          {"first_mismatch_expected", comparison.first_mismatch_expected},
          {"first_mismatch_actual", comparison.first_mismatch_actual},
          {"maximum_absolute_error", comparison.maximum_absolute_error},
          {"relative_l2_error", comparison.relative_l2_error},
          {"cosine_similarity", comparison.cosine_similarity},
          {"expected_sha256", comparison.expected_sha256},
          {"actual_sha256", comparison.actual_sha256}};
}

std::string decode_boundary_label(std::size_t boundary_index) {
  if (boundary_index == 40) return "language_final_norm";
  if (boundary_index >= 40) {
    throw std::invalid_argument("native decode boundary index is invalid");
  }
  std::ostringstream label;
  label << "layer_" << std::setfill('0') << std::setw(3) << boundary_index
        << "_output";
  return label.str();
}

std::string prefill_state_label(std::size_t layer_index,
                                std::string_view state_kind) {
  if (layer_index >= 40 || layer_index % 4 == 3 ||
      (state_kind != "conv_state" && state_kind != "recurrent_state")) {
    throw std::invalid_argument("native prefill state label is invalid");
  }
  std::ostringstream label;
  label << "layer_" << std::setfill('0') << std::setw(3) << layer_index
        << '_' << state_kind;
  return label.str();
}

struct DecodeLinearBoundaryContract {
  const char* label;
  const char* dtype;
  std::uint64_t bytes;
  DecodeTensorDtype tensor_dtype;
};

constexpr std::array<DecodeLinearBoundaryContract, 13>
    kDecodeLinearBoundaryContracts{{
        {"input_norm", "bfloat16", 2048ULL * 2ULL,
         DecodeTensorDtype::kBfloat16},
        {"conv_state_before", "bfloat16", 8192ULL * 3ULL * 2ULL,
         DecodeTensorDtype::kBfloat16},
        {"qkv_projection", "bfloat16", 8192ULL * 2ULL,
         DecodeTensorDtype::kBfloat16},
        {"z_projection", "bfloat16", 4096ULL * 2ULL,
         DecodeTensorDtype::kBfloat16},
        {"a_projection", "bfloat16", 32ULL * 2ULL,
         DecodeTensorDtype::kBfloat16},
        {"b_projection", "bfloat16", 32ULL * 2ULL,
         DecodeTensorDtype::kBfloat16},
        {"post_conv_mixed_qkv", "bfloat16", 8192ULL * 2ULL,
         DecodeTensorDtype::kBfloat16},
        {"conv_state_after", "bfloat16", 8192ULL * 3ULL * 2ULL,
         DecodeTensorDtype::kBfloat16},
        {"recurrent_state_before", "float32",
         32ULL * 128ULL * 128ULL * 4ULL, DecodeTensorDtype::kFloat32},
        {"recurrent_output", "bfloat16", 4096ULL * 2ULL,
         DecodeTensorDtype::kBfloat16},
        {"recurrent_state_after", "float32",
         32ULL * 128ULL * 128ULL * 4ULL, DecodeTensorDtype::kFloat32},
        {"gated_norm", "bfloat16", 4096ULL * 2ULL,
         DecodeTensorDtype::kBfloat16},
        {"attention_output", "bfloat16", 2048ULL * 2ULL,
         DecodeTensorDtype::kBfloat16},
    }};

constexpr std::array<DecodeLinearBoundaryContract, 15>
    kDecodeLayer0TailBoundaryContracts{{
        {"attention_residual", "bfloat16", 2048ULL * 2ULL,
         DecodeTensorDtype::kBfloat16},
        {"post_attention_norm", "bfloat16", 2048ULL * 2ULL,
         DecodeTensorDtype::kBfloat16},
        {"shared_gate_logits", "bfloat16", 1ULL * 2ULL,
         DecodeTensorDtype::kBfloat16},
        {"shared_gate_up_projection", "bfloat16", 1024ULL * 2ULL,
         DecodeTensorDtype::kBfloat16},
        {"shared_activation", "bfloat16", 512ULL * 2ULL,
         DecodeTensorDtype::kBfloat16},
        {"shared_down_projection", "bfloat16", 2048ULL * 2ULL,
         DecodeTensorDtype::kBfloat16},
        {"shared_moe_output", "bfloat16", 2048ULL * 2ULL,
         DecodeTensorDtype::kBfloat16},
        {"router_logits", "bfloat16", 256ULL * 2ULL,
         DecodeTensorDtype::kBfloat16},
        {"router_weights", "float32", 8ULL * 4ULL,
         DecodeTensorDtype::kFloat32},
        {"router_indices", "int32", 8ULL * 4ULL,
         DecodeTensorDtype::kInt32},
        {"routed_gate_up_projection", "bfloat16",
         8ULL * 1024ULL * 2ULL, DecodeTensorDtype::kBfloat16},
        {"routed_activation", "bfloat16", 8ULL * 512ULL * 2ULL,
         DecodeTensorDtype::kBfloat16},
        {"routed_weighted_expert_outputs", "bfloat16",
         8ULL * 2048ULL * 2ULL, DecodeTensorDtype::kBfloat16},
        {"routed_moe_output", "bfloat16", 2048ULL * 2ULL,
         DecodeTensorDtype::kBfloat16},
        {"combined_moe_output", "bfloat16", 2048ULL * 2ULL,
         DecodeTensorDtype::kBfloat16},
    }};

struct ParsedCompletionRequest {
  NativePreparedChat chat;
  std::vector<std::uint32_t> prompt;
  std::string multimodal_cache_namespace;
  std::optional<NativeMropePlan> mrope_plan;
  std::optional<NativeResidentVlInput> vl_input;
  std::shared_ptr<NativeNamedToolJsonConstraint>
      named_tool_json_constraint;
  std::size_t max_tokens = 16;
  bool stream = false;
  bool include_usage = false;
  bool raw_prompt_tokens = false;
};

std::size_t positive_integer_field(const Json& request, const char* field) {
  if (!request[field].is_number_unsigned()) {
    throw std::invalid_argument(std::string(field) +
                                " must be a positive integer");
  }
  const std::uint64_t value = request[field].get<std::uint64_t>();
  if (value == 0 || value > std::numeric_limits<std::size_t>::max()) {
    throw std::invalid_argument(std::string(field) +
                                " must be a positive integer");
  }
  return static_cast<std::size_t>(value);
}

ParsedCompletionRequest parse_completion_request(
    NativeTokenizer& tokenizer, const Json& request,
    const NativeMediaPolicy& media_policy,
    NativeVlMediaCache& media_cache, std::size_t context_capacity) {
  if (!request.is_object()) {
    throw std::invalid_argument("request body must be a JSON object");
  }
  if (request.contains("model") &&
      (!request["model"].is_string() ||
       request["model"].get<std::string>() != kModelId)) {
    throw std::invalid_argument("model is not served by this engine");
  }
  require_number(request, "temperature", 0.0);
  require_number(request, "top_p", 1.0);
  require_number(request, "n", 1.0);
  ParsedCompletionRequest parsed;
  if (request.contains("stream")) {
    if (!request["stream"].is_boolean()) {
      throw std::invalid_argument("stream must be boolean");
    }
    parsed.stream = request["stream"].get<bool>();
  }
  if (request.contains("functions") || request.contains("response_format") ||
      request.contains("stop")) {
    throw std::invalid_argument(
        "deprecated functions, response_format, and custom stop are not "
        "supported");
  }
  if (request.contains("stream_options")) {
    if (!parsed.stream || !request["stream_options"].is_object()) {
      throw std::invalid_argument(
          "stream_options requires stream=true and an object value");
    }
    const Json& options = request["stream_options"];
    if (options.contains("include_usage")) {
      if (!options["include_usage"].is_boolean()) {
        throw std::invalid_argument(
            "stream_options.include_usage must be boolean");
      }
      parsed.include_usage = options["include_usage"].get<bool>();
    }
  }
  if (request.contains("max_completion_tokens")) {
    parsed.max_tokens =
        positive_integer_field(request, "max_completion_tokens");
  } else if (request.contains("max_tokens")) {
    parsed.max_tokens = positive_integer_field(request, "max_tokens");
  }
  if (context_capacity == 0 || parsed.max_tokens >= context_capacity) {
    throw std::invalid_argument(
        "requested output leaves no room for a prompt");
  }
  parsed.chat = prepare_native_chat(request);
  if (!parsed.chat.media.empty()) {
    if (parsed.chat.tool_choice == NativeToolChoiceMode::kSpecific) {
      const auto selected = std::find_if(
          parsed.chat.function_tools.begin(),
          parsed.chat.function_tools.end(),
          [&](const NativeFunctionTool& tool) {
            return tool.name == parsed.chat.required_function_name;
          });
      if (selected == parsed.chat.function_tools.end()) {
        throw std::runtime_error(
            "named VL tool choice lost its selected function");
      }
      parsed.named_tool_json_constraint =
          std::make_shared<NativeNamedToolJsonConstraint>(tokenizer,
                                                          *selected);
    }
    if (request.contains("prompt_token_ids")) {
      throw std::invalid_argument(
          "prompt_token_ids cannot be combined with image or video content");
    }
    NativeMediaPolicy effective_media_policy = media_policy;
    if (parsed.chat.image_io_override.has_value()) {
      effective_media_policy.image_io = *parsed.chat.image_io_override;
    }
    if (parsed.chat.video_io_override.has_value()) {
      effective_media_policy.video_io = *parsed.chat.video_io_override;
    }
    NativeVlPreparedRequest vl = prepare_native_vl_request(
        tokenizer, parsed.chat, effective_media_policy, &media_cache);
    parsed.prompt = std::move(vl.prompt_token_ids);
    parsed.multimodal_cache_namespace =
        std::move(vl.multimodal_cache_namespace);
    parsed.mrope_plan = std::move(vl.mrope_plan);
    NativeResidentVlInput resident;
    resident.grids = std::move(vl.grids);
    resident.pixel_values_bf16 = std::move(vl.pixel_values_bf16);
    resident.embedding_spans = std::move(vl.embedding_spans);
    resident.media_count = vl.metrics.media_count;
    resident.image_count = vl.metrics.image_count;
    resident.video_count = vl.metrics.video_count;
    resident.source_bytes = vl.metrics.source_bytes;
    resident.media_cache_hits = vl.metrics.media_cache_hits;
    resident.media_cache_misses = vl.metrics.media_cache_misses;
    resident.media_cache_entries = vl.metrics.media_cache_entries;
    resident.media_cache_resident_bytes =
        vl.metrics.media_cache_resident_bytes;
    resident.media_load_wall_ms = vl.metrics.media_load_wall_ms;
    resident.media_decode_wall_ms = vl.metrics.media_decode_wall_ms;
    resident.media_load_decode_wall_ms =
        vl.metrics.media_load_decode_wall_ms;
    resident.processor_wall_ms = vl.metrics.processor_wall_ms;
    parsed.vl_input = std::move(resident);
  }
  if (request.contains("prompt_token_ids")) {
    if (!request["prompt_token_ids"].is_array() ||
        request["prompt_token_ids"].empty()) {
      throw std::invalid_argument(
          "prompt_token_ids must be a non-empty integer array");
    }
    if (!parsed.chat.function_tools.empty()) {
      throw std::invalid_argument(
          "prompt_token_ids cannot be combined with tools");
    }
    parsed.prompt.reserve(request["prompt_token_ids"].size());
    for (const Json& value : request["prompt_token_ids"]) {
      if (!value.is_number_unsigned()) {
        throw std::invalid_argument(
            "prompt_token_ids must contain unsigned integers");
      }
      const std::uint64_t token = value.get<std::uint64_t>();
      if (token >= tokenizer.size()) {
        throw std::invalid_argument(
            "prompt_token_ids contains a token outside the vocabulary");
      }
      parsed.prompt.push_back(static_cast<std::uint32_t>(token));
    }
    parsed.raw_prompt_tokens = true;
  } else if (parsed.chat.media.empty()) {
    parsed.prompt = tokenizer.encode_chat(
        parsed.chat.messages, parsed.chat.prompt_tools, true);
  }
  return parsed;
}

Json usage_json(const NativeResidentRequestMetrics& metrics) {
  return {{"prompt_tokens", metrics.prompt_tokens},
          {"completion_tokens", metrics.completion_tokens},
          {"total_tokens",
           metrics.prompt_tokens + metrics.completion_tokens}};
}

Json tool_calls_json(const std::vector<NativeParsedToolCall>& calls) {
  Json result = Json::array();
  for (const NativeParsedToolCall& call : calls) {
    result.push_back(
        {{"id", call.id},
         {"type", "function"},
         {"function", {{"name", call.name},
                       {"arguments", call.serialized_arguments}}}});
  }
  return result;
}

std::int64_t unix_time_seconds() {
  return std::chrono::duration_cast<std::chrono::seconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

NativeAssistantOutput visible_assistant_output(
    const NativeResidentRequestMetrics& metrics,
    const NativeTokenizer& tokenizer, const NativePreparedChat& chat,
    const NativeNamedToolJsonConstraint* named_tool_json_constraint,
    std::string_view call_id_prefix) {
  std::vector<std::uint32_t> visible = metrics.output_token_ids;
  if (metrics.stopped && !visible.empty() &&
      visible.back() == tokenizer.eos_token_id()) {
    visible.pop_back();
  }
  const std::string text = tokenizer.decode(visible);
  if (named_tool_json_constraint != nullptr) {
    return named_tool_json_constraint->parse_output(text, call_id_prefix);
  }
  if (chat.tool_choice == NativeToolChoiceMode::kNone ||
      chat.prompt_tools.empty()) {
    return {text, {}};
  }
  NativeAssistantOutput output =
      parse_qwen_tool_output(text, chat.function_tools, call_id_prefix);
  if (!chat.parallel_tool_calls && output.tool_calls.size() > 1) {
    output.tool_calls.resize(1);
  }
  if (chat.tool_choice == NativeToolChoiceMode::kSpecific) {
    output.tool_calls.erase(
        std::remove_if(
            output.tool_calls.begin(), output.tool_calls.end(),
            [&](const NativeParsedToolCall& call) {
              return call.name != chat.required_function_name;
            }),
        output.tool_calls.end());
  }
  return output;
}

bool required_tool_choice_satisfied(
    const NativePreparedChat& chat,
    const NativeAssistantOutput& output) {
  return (chat.tool_choice != NativeToolChoiceMode::kRequired &&
          chat.tool_choice != NativeToolChoiceMode::kSpecific) ||
         !output.tool_calls.empty();
}

Json chat_completion(NativeResidentEngine& engine, NativeTokenizer& tokenizer,
                     ParsedCompletionRequest parsed) {
  const bool raw_prompt_tokens = parsed.raw_prompt_tokens;
  const bool named_vl_tool_choice =
      parsed.vl_input.has_value() &&
      parsed.chat.tool_choice == NativeToolChoiceMode::kSpecific;
  NativeResidentRequestOptions native_request;
  native_request.input_token_ids = std::move(parsed.prompt);
  native_request.multimodal_cache_namespace =
      std::move(parsed.multimodal_cache_namespace);
  native_request.mrope_plan = std::move(parsed.mrope_plan);
  native_request.vl_input = std::move(parsed.vl_input);
  native_request.max_new_tokens = parsed.max_tokens;
  native_request.stop_token_ids = {tokenizer.eos_token_id()};
  if (parsed.named_tool_json_constraint != nullptr) {
    const auto constraint = parsed.named_tool_json_constraint;
    native_request.next_token_mask =
        [constraint](const std::vector<std::uint32_t>& generated,
                     std::vector<std::uint8_t>* mask) {
          constraint->allowed_token_mask(generated, mask);
        };
  }
  const NativeResidentRequestMetrics metrics = engine.run(native_request);
  const std::string id =
      "chatcmpl-native-" + std::to_string(metrics.request_index);
  NativeAssistantOutput output = visible_assistant_output(
      metrics, tokenizer, parsed.chat,
      parsed.named_tool_json_constraint.get(), id + "-call-");
  if (!required_tool_choice_satisfied(parsed.chat, output)) {
    throw std::runtime_error(
        "model output did not satisfy the required tool_choice");
  }
  Json message = {{"role", "assistant"}};
  if (named_vl_tool_choice) {
    // vLLM exposes named forced-tool generations as an empty content string
    // and keeps the model stop/length reason even though a tool call is
    // present.  Automatic tool selection retains the normal tool_calls reason.
    message["content"] = "";
  } else if (output.tool_calls.empty() || !output.content.empty()) {
    message["content"] = output.content;
  } else {
    message["content"] = nullptr;
  }
  if (!output.tool_calls.empty()) {
    message["tool_calls"] = tool_calls_json(output.tool_calls);
  }
  Json response = {{"id", id},
                   {"object", "chat.completion"},
                   {"created", unix_time_seconds()},
                   {"model", kModelId},
                   {"choices", Json::array({{{"index", 0},
                                              {"message", std::move(message)},
                                              {"finish_reason",
                                               named_vl_tool_choice
                                                   ? (metrics.stopped
                                                          ? "stop"
                                                          : "length")
                                                   : !output.tool_calls.empty()
                                                   ? "tool_calls"
                                                   : (metrics.stopped
                                                          ? "stop"
                                                          : "length")}}})},
                   {"usage", usage_json(metrics)},
                   {"aima_amd395", request_metrics_json(metrics)}};
  response["aima_amd395"]["prompt_source"] =
      raw_prompt_tokens ? "token_ids" : "chat_template";
  return response;
}

Json stream_chunk_base(std::string_view id, std::int64_t created) {
  return {{"id", id},
          {"object", "chat.completion.chunk"},
          {"created", created},
          {"model", kModelId}};
}

bool stream_chat_completion(
    int fd, NativeResidentEngine& engine, NativeTokenizer& tokenizer,
    ParsedCompletionRequest parsed,
    std::chrono::steady_clock::time_point http_started) {
  const bool raw_prompt_tokens = parsed.raw_prompt_tokens;
  const bool named_vl_tool_choice =
      parsed.vl_input.has_value() &&
      parsed.chat.tool_choice == NativeToolChoiceMode::kSpecific;
  const std::string id = "chatcmpl-native-" +
                         std::to_string(engine.request_count() + 1);
  const std::int64_t created = unix_time_seconds();
  bool connected = begin_sse(fd);
  auto send_chunk = [&](Json chunk) {
    if (!connected) return false;
    connected = send_sse_event(fd, chunk.dump());
    return connected;
  };
  Json role = stream_chunk_base(id, created);
  role["choices"] = Json::array(
      {{{"index", 0},
        {"delta", {{"role", "assistant"}}},
        {"finish_reason", nullptr}}});
  if (!send_chunk(std::move(role))) return false;

  NativeIncrementalUtf8Decoder utf8;
  NativeToolStreamGate gate;
  const bool hold_content_for_required_tool =
      parsed.chat.tool_choice == NativeToolChoiceMode::kRequired ||
      parsed.chat.tool_choice == NativeToolChoiceMode::kSpecific;
  NativeResidentRequestOptions native_request;
  native_request.input_token_ids = std::move(parsed.prompt);
  native_request.multimodal_cache_namespace =
      std::move(parsed.multimodal_cache_namespace);
  native_request.mrope_plan = std::move(parsed.mrope_plan);
  native_request.vl_input = std::move(parsed.vl_input);
  native_request.max_new_tokens = parsed.max_tokens;
  native_request.stop_token_ids = {tokenizer.eos_token_id()};
  if (parsed.named_tool_json_constraint != nullptr) {
    const auto constraint = parsed.named_tool_json_constraint;
    native_request.next_token_mask =
        [constraint](const std::vector<std::uint32_t>& generated,
                     std::vector<std::uint8_t>* mask) {
          constraint->allowed_token_mask(generated, mask);
        };
  }
  native_request.token_callback =
      [&](std::uint32_t token_id, std::size_t) -> bool {
    if (!connected) return false;
    if (token_id == tokenizer.eos_token_id()) return true;
    const std::string decoded =
        utf8.push(tokenizer.decode_token_bytes(token_id));
    const std::string content = gate.push(decoded);
    if (content.empty() || hold_content_for_required_tool) return true;
    Json chunk = stream_chunk_base(id, created);
    chunk["choices"] = Json::array(
        {{{"index", 0},
          {"delta", {{"content", content}}},
          {"finish_reason", nullptr}}});
    return send_chunk(std::move(chunk));
  };

  NativeResidentRequestMetrics metrics;
  try {
    metrics = engine.run(native_request);
  } catch (const std::exception& error) {
    if (connected) {
      (void)send_sse_event(
          fd, error_payload(error.what(), "native_engine_error",
                            "server_error")
                  .dump());
      (void)send_sse_event(fd, "[DONE]");
      (void)try_send_all(fd, "0\r\n\r\n");
    }
    return false;
  }
  if (!connected || metrics.client_cancelled) return false;

  const std::string terminal_utf8 = utf8.finish();
  std::string terminal_streamable;
  if (!terminal_utf8.empty()) {
    terminal_streamable = gate.push(terminal_utf8);
  }
  NativeAssistantOutput output;
  if (parsed.named_tool_json_constraint != nullptr) {
    output = parsed.named_tool_json_constraint->parse_output(
        gate.complete_text(), id + "-call-");
  } else if (parsed.chat.tool_choice != NativeToolChoiceMode::kNone &&
      !parsed.chat.prompt_tools.empty()) {
    output = parse_qwen_tool_output(
        gate.complete_text(), parsed.chat.function_tools, id + "-call-");
    if (!parsed.chat.parallel_tool_calls && output.tool_calls.size() > 1) {
      output.tool_calls.resize(1);
    }
    if (parsed.chat.tool_choice == NativeToolChoiceMode::kSpecific) {
      output.tool_calls.erase(
          std::remove_if(
              output.tool_calls.begin(), output.tool_calls.end(),
              [&](const NativeParsedToolCall& call) {
                return call.name != parsed.chat.required_function_name;
              }),
          output.tool_calls.end());
    }
  } else {
    output.content = gate.complete_text();
  }

  if (!required_tool_choice_satisfied(parsed.chat, output)) {
    (void)send_sse_event(
        fd,
        error_payload(
            "model output did not satisfy the required tool_choice",
            "tool_choice_not_satisfied", "server_error")
            .dump());
    (void)send_sse_event(fd, "[DONE]");
    (void)try_send_all(fd, "0\r\n\r\n");
    return false;
  }
  if (hold_content_for_required_tool && !output.content.empty()) {
    Json chunk = stream_chunk_base(id, created);
    chunk["choices"] = Json::array(
        {{{"index", 0},
          {"delta", {{"content", output.content}}},
          {"finish_reason", nullptr}}});
    if (!send_chunk(std::move(chunk))) return false;
  }
  std::string remaining;
  if (!hold_content_for_required_tool) {
    remaining = std::move(terminal_streamable);
  }
  remaining += gate.finish(!output.tool_calls.empty());
  if (!remaining.empty()) {
    Json chunk = stream_chunk_base(id, created);
    chunk["choices"] = Json::array(
        {{{"index", 0},
          {"delta", {{"content", remaining}}},
          {"finish_reason", nullptr}}});
    if (!send_chunk(std::move(chunk))) return false;
  }
  if (!output.tool_calls.empty()) {
    Json deltas = Json::array();
    for (std::size_t index = 0; index < output.tool_calls.size(); ++index) {
      const NativeParsedToolCall& call = output.tool_calls[index];
      deltas.push_back(
          {{"index", index},
           {"id", call.id},
           {"type", "function"},
           {"function", {{"name", call.name},
                         {"arguments", call.serialized_arguments}}}});
    }
    Json chunk = stream_chunk_base(id, created);
    chunk["choices"] = Json::array(
        {{{"index", 0},
          {"delta", {{"tool_calls", std::move(deltas)}}},
          {"finish_reason", nullptr}}});
    if (!send_chunk(std::move(chunk))) return false;
  }

  Json terminal = stream_chunk_base(id, created);
  terminal["choices"] = Json::array(
      {{{"index", 0},
        {"delta", Json::object()},
        {"finish_reason",
         named_vl_tool_choice
             ? (metrics.stopped ? "stop" : "length")
             : !output.tool_calls.empty()
             ? "tool_calls"
             : (metrics.stopped ? "stop" : "length")}}});
  terminal["aima_amd395"] = request_metrics_json(metrics);
  terminal["aima_amd395"]["prompt_source"] =
      raw_prompt_tokens ? "token_ids" : "chat_template";
  terminal["aima_amd395"]["http_request_wall_ms"] =
      elapsed_ms(http_started);
  if (!send_chunk(std::move(terminal))) return false;
  if (parsed.include_usage) {
    Json usage = stream_chunk_base(id, created);
    usage["choices"] = Json::array();
    usage["usage"] = usage_json(metrics);
    if (!send_chunk(std::move(usage))) return false;
  }
  if (!send_sse_event(fd, "[DONE]")) return false;
  return try_send_all(fd, "0\r\n\r\n");
}

}  // namespace

int run_native_http_server(int argc, char** argv) {
  const ServerOptions options = parse_options(argc, argv);
  g_shutdown.store(false);
  const auto process_started = std::chrono::steady_clock::now();

  FileDescriptor server(::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0));
  if (server.get() < 0) throw std::runtime_error("socket creation failed");
  int reuse = 1;
  if (::setsockopt(server.get(), SOL_SOCKET, SO_REUSEADDR, &reuse,
                   sizeof(reuse)) != 0) {
    throw std::runtime_error("setsockopt SO_REUSEADDR failed");
  }
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_port = htons(static_cast<std::uint16_t>(options.port));
  if (::inet_pton(AF_INET, options.host.c_str(), &address.sin_addr) != 1) {
    throw std::runtime_error("--host must be an IPv4 address");
  }
  if (::bind(server.get(), reinterpret_cast<sockaddr*>(&address),
             sizeof(address)) != 0) {
    throw std::runtime_error("bind failed: " +
                             std::string(std::strerror(errno)));
  }

  systemd_notify("STATUS=Loading tokenizer and model");
  NativeTokenizer tokenizer;
  tokenizer.load(options.engine.weights.model_dir);
  NativeResidentEngine engine;
  const NativeResidentLoadMetrics load = engine.load(options.engine);
  NativeVlMediaCache media_cache(options.media_cache_capacity_bytes);

  if (::listen(server.get(), 16) != 0) {
    throw std::runtime_error("listen failed");
  }
  struct sigaction action {};
  action.sa_handler = signal_handler;
  sigemptyset(&action.sa_mask);
  action.sa_flags = 0;
  if (::sigaction(SIGINT, &action, nullptr) != 0 ||
      ::sigaction(SIGTERM, &action, nullptr) != 0) {
    throw std::runtime_error("failed to install shutdown signal handlers");
  }
  systemd_notify("READY=1\nSTATUS=Ready");
  const double command_to_ready_ms = elapsed_ms(process_started);
  std::cout << Json({{"event", "ready"},
                     {"host", options.host},
                     {"port", options.port},
                     {"model", kModelId},
                     {"command_to_ready_wall_ms", command_to_ready_ms},
                     {"engine_load_wall_ms", load.command_to_ready_wall_ms},
                     {"model_payload_bytes", load.model_payload_bytes},
                     {"model_tensor_count", load.model_tensor_count},
                     {"model_shard_count", load.model_shard_count},
                     {"language_model_payload_bytes",
                      load.language_model_payload_bytes},
                     {"language_model_tensor_count",
                      load.language_model_tensor_count},
                     {"language_layout_manifest_sha256",
                      load.language_layout_manifest_sha256},
                     {"visual_model_payload_bytes",
                      load.visual_model_payload_bytes},
                     {"visual_model_tensor_count",
                      load.visual_model_tensor_count},
                     {"visual_layout_manifest_sha256",
                      load.visual_layout_manifest_sha256},
                     {"fmha_provider_backend", load.fmha_provider_backend},
                     {"fmha_provider_path", load.fmha_provider_path},
                     {"secondary_fmha_provider_backend",
                      load.secondary_fmha_provider_backend},
                     {"secondary_fmha_provider_path",
                      load.secondary_fmha_provider_path},
                     {"secondary_fmha_layers",
                      load.secondary_fmha_layers},
                     {"authentication",
                      options.api_key.empty() ? "none" : "bearer"},
                     {"http_shutdown_enabled", options.http_shutdown},
                     {"request_timeout_ms", options.request_timeout_ms},
                     {"context_capacity", load.cache_capacity},
                     {"static_prefill_tokens", load.prompt_tokens},
                     {"mrope_position_state_bytes",
                      load.mrope_position_state_bytes},
                     {"structured_token_mask_bytes",
                      load.structured_token_mask_bytes},
                     {"resident_prefill_buckets",
                      load.resident_prefill_buckets},
                     {"prefix_cache_entries", load.prefix_cache_entries},
                     {"native_vl", true},
                     {"vision_plan_cache_capacity",
                      load.vision_plan_cache_capacity},
                     {"allowed_local_media_roots",
                      options.media_policy.allowed_local_roots.size()},
                     {"allowed_media_domains",
                      options.media_policy.allowed_media_domains.size()},
                     {"media_cache_capacity_bytes",
                      media_cache.capacity_bytes()},
                     {"media_cache_capacity_entries",
                      media_cache.capacity_entries()},
                     {"prompt_token_ids_extension", true},
                     {"runtime_python", false},
                     {"runtime_torch", false},
                     {"runtime_vllm", false},
                     {"runtime_triton", false}})
                   .dump()
            << std::endl;

  std::size_t served = 0;
  while (!g_shutdown.load()) {
    sockaddr_in peer{};
    socklen_t peer_size = sizeof(peer);
    const int accepted = ::accept4(server.get(),
                                   reinterpret_cast<sockaddr*>(&peer),
                                   &peer_size, SOCK_CLOEXEC);
    if (accepted < 0) {
      if (errno == EINTR) continue;
      throw std::runtime_error("accept failed");
    }
    FileDescriptor client(accepted);
    try {
      configure_client_timeout(client.get(), options.request_timeout_ms);
      const HttpRequest request =
          read_request(client.get(), options.request_timeout_ms);
      if (requires_authentication(request, options) &&
          !authorized(request, options.api_key)) {
        (void)send_json(
            client.get(), 401,
            error_payload("a valid bearer token is required", "unauthorized"),
            "WWW-Authenticate: Bearer\r\n");
        continue;
      }
      if (request.method == "GET" && request.path == "/health") {
        (void)send_json(client.get(), 200,
                  {{"status", "ok"},
                   {"model", kModelId},
                   {"model_loaded", engine.loaded()},
                   {"resident", true},
                   {"served", served},
                   {"uptime_ms", elapsed_ms(process_started)},
                   {"command_to_ready_wall_ms", command_to_ready_ms},
                   {"runtime", "native"},
                   {"fmha_provider_backend", load.fmha_provider_backend},
                   {"secondary_fmha_provider_backend",
                    load.secondary_fmha_provider_backend},
                   {"secondary_fmha_layers", load.secondary_fmha_layers},
                   {"authentication",
                    options.api_key.empty() ? "none" : "bearer"},
                   {"http_shutdown_enabled", options.http_shutdown},
                   {"request_timeout_ms", options.request_timeout_ms},
                   {"admitted_prompt_tokens", load.cache_capacity},
                   {"context_capacity", load.cache_capacity},
                   {"static_prefill_tokens", load.prompt_tokens},
                   {"structured_token_mask_bytes",
                    load.structured_token_mask_bytes},
                   {"resident_prefill_buckets",
                    load.resident_prefill_buckets},
                   {"prefix_cache_entries", load.prefix_cache_entries},
                   {"native_vl", true},
                   {"vision_plan_cache_capacity",
                    load.vision_plan_cache_capacity},
                   {"media_cache_capacity_bytes",
                    media_cache.capacity_bytes()},
                   {"media_cache_capacity_entries",
                    media_cache.capacity_entries()},
                   {"media_cache_entries", media_cache.entries()},
                   {"media_cache_resident_bytes",
                    media_cache.resident_bytes()},
                   {"prompt_token_ids_extension", true}});
      } else if (request.method == "GET" && request.path == "/v1/models") {
        (void)send_json(client.get(), 200,
                  {{"object", "list"},
                   {"data", Json::array({{{"id", kModelId},
                                           {"object", "model"},
                                           {"owned_by", "approaching-ai"}}})}});
      } else if (request.method == "POST" && request.path == "/shutdown" &&
                 options.http_shutdown) {
        (void)send_json(client.get(), 200, {{"status", "shutting_down"}});
        g_shutdown.store(true);
      } else if (request.method == "POST" &&
                 request.path == "/v1/chat/completions") {
        Json body;
        try {
          body = Json::parse(request.body);
          const auto started = std::chrono::steady_clock::now();
          ParsedCompletionRequest parsed =
              parse_completion_request(tokenizer, body,
                                       options.media_policy, media_cache,
                                       load.cache_capacity);
          if (parsed.stream) {
            if (stream_chat_completion(client.get(), engine, tokenizer,
                                       std::move(parsed), started)) {
              ++served;
            }
          } else {
            Json response =
                chat_completion(engine, tokenizer, std::move(parsed));
            response["aima_amd395"]["http_request_wall_ms"] =
                elapsed_ms(started);
            if (send_json(client.get(), 200, response)) ++served;
          }
        } catch (const std::invalid_argument& error) {
          (void)send_json(client.get(), 400,
                    error_payload(error.what(), "bad_request"));
        } catch (const Json::exception& error) {
          (void)send_json(client.get(), 400,
                    error_payload(error.what(), "invalid_json"));
        } catch (const std::exception& error) {
          (void)send_json(client.get(), 500,
                    error_payload(error.what(), "native_engine_error",
                                  "server_error"));
        }
      } else {
        const int status = request.path == "/health" ||
                                   request.path == "/v1/models" ||
                                   (options.http_shutdown &&
                                    request.path == "/shutdown") ||
                                   request.path == "/v1/chat/completions"
                               ? 405
                               : 404;
        (void)send_json(client.get(), status,
                  error_payload("unsupported method or path",
                                status == 404 ? "not_found"
                                              : "method_not_allowed"));
      }
    } catch (const std::length_error& error) {
      (void)send_json(client.get(), 413,
                error_payload(error.what(), "request_too_large"));
    } catch (const std::system_error& error) {
      const bool timed_out =
          error.code() == std::make_error_code(std::errc::timed_out);
      (void)send_json(client.get(), timed_out ? 408 : 400,
                      error_payload(error.what(),
                                    timed_out ? "request_timeout"
                                              : "bad_request"));
    } catch (const std::exception& error) {
      (void)send_json(client.get(), 400,
                error_payload(error.what(), "bad_request"));
    }
    if (options.maximum_requests != 0 && served >= options.maximum_requests) {
      g_shutdown.store(true);
    }
  }
  systemd_notify("STOPPING=1\nSTATUS=Stopping");
  std::cout << Json({{"event", "stopped"},
                     {"served", served},
                     {"model_loads", 1}})
                   .dump()
            << std::endl;
  return 0;
}

int run_native_vl_generation_logits_probe(int argc, char** argv) {
  std::filesystem::path cases_path;
  std::vector<std::string> forwarded;
  forwarded.reserve(static_cast<std::size_t>(argc));
  forwarded.emplace_back(argv[0]);
  forwarded.emplace_back(argv[1]);
  for (int index = 2; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--cases-json") {
      if (++index >= argc) {
        throw std::runtime_error("--cases-json requires a value");
      }
      cases_path = std::filesystem::absolute(argv[index]);
      continue;
    }
    forwarded.push_back(argument);
  }
  if (cases_path.empty()) {
    throw std::runtime_error(
        "VL generation logits probe requires --cases-json");
  }
  std::vector<char*> forwarded_argv;
  forwarded_argv.reserve(forwarded.size());
  for (std::string& argument : forwarded) {
    forwarded_argv.push_back(argument.data());
  }
  const ServerOptions options = parse_options(
      static_cast<int>(forwarded_argv.size()), forwarded_argv.data());

  std::ifstream stream(cases_path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error("cannot open VL generation cases file");
  }
  stream.seekg(0, std::ios::end);
  const std::streamoff bytes = stream.tellg();
  if (bytes <= 0 || bytes > static_cast<std::streamoff>(kMaximumRequestBytes)) {
    throw std::runtime_error("VL generation cases file size is invalid");
  }
  stream.seekg(0, std::ios::beg);
  std::string serialized(static_cast<std::size_t>(bytes), '\0');
  stream.read(serialized.data(), bytes);
  if (!stream) {
    throw std::runtime_error("cannot read VL generation cases file");
  }
  const Json specification = Json::parse(serialized);
  if (!specification.is_object() || !specification.contains("cases") ||
      !specification["cases"].is_array() ||
      specification["cases"].empty()) {
    throw std::runtime_error(
        "VL generation cases file must contain a non-empty cases array");
  }

  NativeTokenizer tokenizer;
  tokenizer.load(options.engine.weights.model_dir);
  NativeResidentEngine engine;
  const NativeResidentLoadMetrics load = engine.load(options.engine);
  NativeVlMediaCache media_cache(options.media_cache_capacity_bytes);
  Json results = Json::array();
  bool all_prefixes_exact = true;
  bool all_reference_rows_bound = true;
  bool all_native_top1_exact = true;
  bool all_decode_boundaries_compared = true;
  bool all_decode_linear_boundaries_compared = true;
  bool all_decode_layer0_tail_boundaries_compared = true;
  bool all_decode_full_attention_compared = true;
  bool all_prefill_states_compared = true;

  for (const Json& item : specification["cases"]) {
    if (!item.is_object() || !item.contains("case_id") ||
        !item["case_id"].is_string() ||
        item["case_id"].get<std::string>().empty() ||
        !item.contains("request") || !item["request"].is_object() ||
        !item.contains("expected_prefix_token_ids") ||
        !item["expected_prefix_token_ids"].is_array() ||
        item["expected_prefix_token_ids"].empty() ||
        !item.contains("expected_reference_token_id") ||
        !item["expected_reference_token_id"].is_number_unsigned() ||
        !item.contains("reference_logits") ||
        !item["reference_logits"].is_string() ||
        (item.contains("diagnostic_allow_prefix_divergence") &&
         !item["diagnostic_allow_prefix_divergence"].is_boolean())) {
      throw std::runtime_error("VL generation case is malformed");
    }
    const std::string case_id = item["case_id"].get<std::string>();
    const bool diagnostic_allow_prefix_divergence =
        item.value("diagnostic_allow_prefix_divergence", false);
    std::vector<std::uint32_t> expected_prefix;
    expected_prefix.reserve(item["expected_prefix_token_ids"].size());
    for (const Json& token : item["expected_prefix_token_ids"]) {
      if (!token.is_number_unsigned()) {
        throw std::runtime_error(
            "VL generation expected prefix contains a non-integer token");
      }
      const std::uint64_t value = token.get<std::uint64_t>();
      if (value >= tokenizer.size()) {
        throw std::runtime_error(
            "VL generation expected prefix token exceeds the vocabulary");
      }
      expected_prefix.push_back(static_cast<std::uint32_t>(value));
    }
    const std::uint64_t reference_token_value =
        item["expected_reference_token_id"].get<std::uint64_t>();
    if (reference_token_value >= tokenizer.size()) {
      throw std::runtime_error(
          "VL generation reference token exceeds the vocabulary");
    }
    const std::uint32_t expected_reference_token =
        static_cast<std::uint32_t>(reference_token_value);
    const std::filesystem::path reference_logits =
        std::filesystem::absolute(
            item["reference_logits"].get<std::string>());
    std::vector<std::filesystem::path> reference_decode_boundaries;
    if (item.contains("reference_decode_boundary_dir")) {
      if (!item["reference_decode_boundary_dir"].is_string() ||
          item["reference_decode_boundary_dir"].get<std::string>().empty()) {
        throw std::runtime_error(
            "VL generation decode boundary directory is malformed");
      }
      const std::filesystem::path boundary_dir = std::filesystem::absolute(
          item["reference_decode_boundary_dir"].get<std::string>());
      reference_decode_boundaries.reserve(41);
      for (std::size_t boundary_index = 0; boundary_index <= 40;
           ++boundary_index) {
        reference_decode_boundaries.push_back(
            find_native_oracle_tensor_file(
                boundary_dir, decode_boundary_label(boundary_index)));
      }
    } else {
      all_decode_boundaries_compared = false;
    }
    std::vector<std::filesystem::path> reference_decode_linear_boundaries;
    if (item.contains("reference_decode_linear_boundary_dir")) {
      if (!item["reference_decode_linear_boundary_dir"].is_string() ||
          item["reference_decode_linear_boundary_dir"]
              .get<std::string>()
              .empty()) {
        throw std::runtime_error(
            "VL generation decode linear boundary directory is malformed");
      }
      const std::filesystem::path boundary_dir = std::filesystem::absolute(
          item["reference_decode_linear_boundary_dir"].get<std::string>());
      reference_decode_linear_boundaries.reserve(
          kDecodeLinearBoundaryContracts.size());
      for (const DecodeLinearBoundaryContract& contract :
           kDecodeLinearBoundaryContracts) {
        reference_decode_linear_boundaries.push_back(
            find_native_oracle_tensor_file(boundary_dir, contract.label));
      }
    } else {
      all_decode_linear_boundaries_compared = false;
    }
    std::vector<std::filesystem::path>
        reference_decode_layer0_tail_boundaries;
    if (item.contains("reference_decode_layer0_tail_boundary_dir")) {
      if (!item["reference_decode_layer0_tail_boundary_dir"].is_string() ||
          item["reference_decode_layer0_tail_boundary_dir"]
              .get<std::string>()
              .empty()) {
        throw std::runtime_error(
            "VL generation decode layer-0 tail boundary directory is malformed");
      }
      const std::filesystem::path boundary_dir = std::filesystem::absolute(
          item["reference_decode_layer0_tail_boundary_dir"]
              .get<std::string>());
      reference_decode_layer0_tail_boundaries.reserve(
          kDecodeLayer0TailBoundaryContracts.size());
      for (const DecodeLinearBoundaryContract& contract :
           kDecodeLayer0TailBoundaryContracts) {
        reference_decode_layer0_tail_boundaries.push_back(
            find_native_oracle_tensor_file(boundary_dir, contract.label));
      }
    } else {
      all_decode_layer0_tail_boundaries_compared = false;
    }
    struct ReferenceDecodeFullAttention {
      std::size_t layer_index = 3;
      std::filesystem::path qkv_projection;
      std::filesystem::path query;
      std::filesystem::path key_cache;
      std::filesystem::path value_cache;
      std::filesystem::path output;
      std::filesystem::path gated_attention;
      std::filesystem::path projected_attention;
      std::filesystem::path attention_residual;
      std::filesystem::path post_attention_norm;
      std::filesystem::path shared_gate_logits;
      std::filesystem::path shared_gate_up_projection;
      std::filesystem::path shared_activation;
      std::filesystem::path shared_down_projection;
      std::filesystem::path shared_moe_output;
      std::filesystem::path router_logits;
      std::filesystem::path router_weights;
      std::filesystem::path router_indices;
      std::filesystem::path routed_gate_up_projection;
      std::filesystem::path routed_activation;
      std::filesystem::path routed_weighted_expert_outputs;
      std::filesystem::path routed_moe_output;
      std::filesystem::path combined_moe_output;

      std::size_t comparison_count() const {
        return 6 + static_cast<std::size_t>(!qkv_projection.empty()) +
               static_cast<std::size_t>(!gated_attention.empty()) +
               static_cast<std::size_t>(!projected_attention.empty()) +
               static_cast<std::size_t>(!attention_residual.empty()) +
               static_cast<std::size_t>(!post_attention_norm.empty()) +
               static_cast<std::size_t>(!shared_gate_logits.empty()) +
               static_cast<std::size_t>(
                   !shared_gate_up_projection.empty()) +
               static_cast<std::size_t>(!shared_activation.empty()) +
               static_cast<std::size_t>(!shared_down_projection.empty()) +
               static_cast<std::size_t>(!shared_moe_output.empty()) +
               static_cast<std::size_t>(!router_logits.empty()) +
               static_cast<std::size_t>(!router_weights.empty()) +
               static_cast<std::size_t>(!router_indices.empty()) +
               static_cast<std::size_t>(
                   !routed_gate_up_projection.empty()) +
               static_cast<std::size_t>(!routed_activation.empty()) +
               static_cast<std::size_t>(
                   !routed_weighted_expert_outputs.empty()) +
               static_cast<std::size_t>(!routed_moe_output.empty()) +
               static_cast<std::size_t>(!combined_moe_output.empty());
      }
    };
    std::optional<ReferenceDecodeFullAttention>
        reference_decode_full_attention;
    if (item.contains("reference_decode_full_attention_dir")) {
      if (!item["reference_decode_full_attention_dir"].is_string() ||
          item["reference_decode_full_attention_dir"]
              .get<std::string>()
              .empty()) {
        throw std::runtime_error(
            "VL generation decode full-attention directory is malformed");
      }
      const std::filesystem::path attention_dir =
          std::filesystem::absolute(
              item["reference_decode_full_attention_dir"]
                  .get<std::string>());
      std::size_t attention_layer_index = 3;
      if (item.contains("reference_decode_full_attention_layer_index")) {
        if (!item["reference_decode_full_attention_layer_index"]
                 .is_number_unsigned()) {
          throw std::runtime_error(
              "VL generation decode full-attention layer is malformed");
        }
        attention_layer_index =
            item["reference_decode_full_attention_layer_index"]
                .get<std::size_t>();
      }
      if (attention_layer_index >= 40 || attention_layer_index % 4 != 3) {
        throw std::runtime_error(
            "VL generation decode full-attention layer is unsupported");
      }
      reference_decode_full_attention = ReferenceDecodeFullAttention{
          attention_layer_index,
          find_native_oracle_tensor_file_if_present(
              attention_dir, "qkv_projection"),
          find_native_oracle_tensor_file(attention_dir, "query"),
          find_native_oracle_tensor_file(attention_dir, "key_cache"),
          find_native_oracle_tensor_file(attention_dir, "value_cache"),
          find_native_oracle_tensor_file(attention_dir, "output"),
          find_native_oracle_tensor_file_if_present(
              attention_dir, "gated_attention"),
          find_native_oracle_tensor_file_if_present(
              attention_dir, "projected_attention"),
          find_native_oracle_tensor_file_if_present(
              attention_dir, "attention_residual"),
          find_native_oracle_tensor_file_if_present(
              attention_dir, "post_attention_norm"),
          find_native_oracle_tensor_file_if_present(
              attention_dir, "shared_gate_logits"),
          find_native_oracle_tensor_file_if_present(
              attention_dir, "shared_gate_up_projection"),
          find_native_oracle_tensor_file_if_present(
              attention_dir, "shared_activation"),
          find_native_oracle_tensor_file_if_present(
              attention_dir, "shared_down_projection"),
          find_native_oracle_tensor_file_if_present(
              attention_dir, "shared_moe_output"),
          find_native_oracle_tensor_file_if_present(
              attention_dir, "router_logits"),
          find_native_oracle_tensor_file_if_present(
              attention_dir, "router_weights"),
          find_native_oracle_tensor_file_if_present(
              attention_dir, "router_indices"),
          find_native_oracle_tensor_file_if_present(
              attention_dir, "routed_gate_up_projection"),
          find_native_oracle_tensor_file_if_present(
              attention_dir, "routed_activation"),
          find_native_oracle_tensor_file_if_present(
              attention_dir, "routed_weighted_expert_outputs"),
          find_native_oracle_tensor_file_if_present(
              attention_dir, "routed_moe_output"),
          find_native_oracle_tensor_file_if_present(
              attention_dir, "combined_moe_output"),
      };
    } else {
      all_decode_full_attention_compared = false;
    }
    struct ReferencePrefillState {
      std::size_t layer_index = 0;
      std::filesystem::path conv;
      std::filesystem::path recurrent;
    };
    std::vector<ReferencePrefillState> reference_prefill_states;
    if (item.contains("reference_prefill_state_dir")) {
      if (!item["reference_prefill_state_dir"].is_string() ||
          item["reference_prefill_state_dir"].get<std::string>().empty()) {
        throw std::runtime_error(
            "VL generation prefill state directory is malformed");
      }
      const std::filesystem::path state_dir = std::filesystem::absolute(
          item["reference_prefill_state_dir"].get<std::string>());
      reference_prefill_states.reserve(30);
      for (std::size_t layer_index = 0; layer_index < 40; ++layer_index) {
        if (layer_index % 4 == 3) continue;
        reference_prefill_states.push_back(
            {layer_index,
             find_native_oracle_tensor_file(
                 state_dir,
                 prefill_state_label(layer_index, "conv_state")),
             find_native_oracle_tensor_file(
                 state_dir,
                 prefill_state_label(layer_index, "recurrent_state"))});
      }
    } else {
      all_prefill_states_compared = false;
    }

    ParsedCompletionRequest parsed = parse_completion_request(
        tokenizer, item["request"], options.media_policy, media_cache,
        load.cache_capacity);
    if (!parsed.vl_input.has_value() || parsed.stream ||
        parsed.raw_prompt_tokens) {
      throw std::runtime_error(
          "VL generation diagnosis requires a non-streaming media chat request");
    }
    NativeResidentRequestOptions request;
    request.input_token_ids = std::move(parsed.prompt);
    request.multimodal_cache_namespace =
        std::move(parsed.multimodal_cache_namespace);
    request.mrope_plan = std::move(parsed.mrope_plan);
    request.vl_input = std::move(parsed.vl_input);
    request.max_new_tokens = expected_prefix.size() + 1;
    request.stop_token_ids = {tokenizer.eos_token_id()};
    request.disable_prefix_cache = true;
    std::vector<NativeOracleComparison> prefill_state_comparisons;
    if (!reference_prefill_states.empty()) {
      prefill_state_comparisons.reserve(reference_prefill_states.size() * 2);
      request.prefill_linear_state_observer =
          [&](std::size_t layer_index, const void* conv_state,
              std::uint64_t conv_state_bytes, const void* recurrent_state,
              std::uint64_t recurrent_state_bytes) {
            const std::size_t state_index =
                prefill_state_comparisons.size() / 2;
            if (state_index >= reference_prefill_states.size() ||
                reference_prefill_states[state_index].layer_index !=
                    layer_index ||
                conv_state_bytes != 8192ULL * 3ULL * sizeof(std::uint16_t) ||
                recurrent_state_bytes !=
                    32ULL * 128ULL * 128ULL * sizeof(float)) {
              throw std::runtime_error(
                  "native VL prefill state observer order changed");
            }
            const ReferencePrefillState& expected =
                reference_prefill_states[state_index];
            prefill_state_comparisons.push_back(
                compare_native_oracle_tensor(
                    prefill_state_label(layer_index, "conv_state"),
                    "bfloat16", conv_state, conv_state_bytes,
                    expected.conv));
            prefill_state_comparisons.push_back(
                compare_native_oracle_tensor(
                    prefill_state_label(layer_index, "recurrent_state"),
                    "float32", recurrent_state, recurrent_state_bytes,
                    expected.recurrent));
          };
    }
    std::vector<NativeOracleComparison> decode_boundary_comparisons;
    if (!reference_decode_boundaries.empty()) {
      decode_boundary_comparisons.reserve(41);
      request.decode_layer_observer_output_index = expected_prefix.size();
      request.decode_layer_observer =
          [&](std::size_t boundary_index, const void* device_row) {
            if (boundary_index != decode_boundary_comparisons.size() ||
                boundary_index >= reference_decode_boundaries.size()) {
              throw std::runtime_error(
                  "native VL decode boundary observer order changed");
            }
            const std::string label =
                decode_boundary_label(boundary_index);
            decode_boundary_comparisons.push_back(
                compare_native_oracle_tensor(
                    label, "bfloat16", device_row,
                    2048 * sizeof(std::uint16_t),
                    reference_decode_boundaries[boundary_index]));
          };
    }
    std::vector<NativeOracleComparison> decode_linear_boundary_comparisons;
    if (!reference_decode_linear_boundaries.empty()) {
      decode_linear_boundary_comparisons.reserve(
          kDecodeLinearBoundaryContracts.size());
      request.decode_layer_observer_output_index = expected_prefix.size();
      request.decode_linear_layer0_observer =
          [&](const char* boundary_name, const void* device_tensor,
              std::uint64_t tensor_bytes, DecodeTensorDtype dtype) {
            const std::size_t boundary_index =
                decode_linear_boundary_comparisons.size();
            if (boundary_index >= kDecodeLinearBoundaryContracts.size()) {
              throw std::runtime_error(
                  "native VL decode linear observer emitted extra boundaries");
            }
            const DecodeLinearBoundaryContract& expected =
                kDecodeLinearBoundaryContracts[boundary_index];
            if (std::string_view(boundary_name) != expected.label ||
                tensor_bytes != expected.bytes ||
                dtype != expected.tensor_dtype) {
              throw std::runtime_error(
                  "native VL decode linear observer order changed");
            }
            decode_linear_boundary_comparisons.push_back(
                compare_native_oracle_tensor(
                    expected.label, expected.dtype, device_tensor,
                    tensor_bytes,
                    reference_decode_linear_boundaries[boundary_index]));
          };
    }
    std::vector<NativeOracleComparison>
        decode_layer0_tail_boundary_comparisons;
    if (!reference_decode_layer0_tail_boundaries.empty()) {
      decode_layer0_tail_boundary_comparisons.reserve(
          kDecodeLayer0TailBoundaryContracts.size());
      request.decode_layer_observer_output_index = expected_prefix.size();
      request.decode_layer0_tail_observer =
          [&](const char* boundary_name, const void* device_tensor,
              std::uint64_t tensor_bytes, DecodeTensorDtype dtype) {
            const std::size_t boundary_index =
                decode_layer0_tail_boundary_comparisons.size();
            if (boundary_index >=
                kDecodeLayer0TailBoundaryContracts.size()) {
              throw std::runtime_error(
                  "native VL decode layer-0 tail observer emitted extra boundaries");
            }
            const DecodeLinearBoundaryContract& expected =
                kDecodeLayer0TailBoundaryContracts[boundary_index];
            if (std::string_view(boundary_name) != expected.label ||
                tensor_bytes != expected.bytes ||
                dtype != expected.tensor_dtype) {
              throw std::runtime_error(
                  "native VL decode layer-0 tail observer order changed");
            }
            decode_layer0_tail_boundary_comparisons.push_back(
                compare_native_oracle_tensor(
                    expected.label, expected.dtype, device_tensor,
                    tensor_bytes,
                    reference_decode_layer0_tail_boundaries[boundary_index]));
          };
    }
    std::vector<NativeOracleComparison>
        decode_full_attention_comparisons;
    if (reference_decode_full_attention.has_value()) {
      const std::size_t expected_full_attention_comparisons =
          reference_decode_full_attention->comparison_count();
      decode_full_attention_comparisons.reserve(
          expected_full_attention_comparisons);
      request.decode_layer_observer_output_index = expected_prefix.size();
      request.decode_full_attention_observer =
          [&](const NativeDecodeFullAttentionObservation& observed) {
            const ReferenceDecodeFullAttention& expected =
                *reference_decode_full_attention;
            if (observed.layer_index != expected.layer_index) return;
            if (!decode_full_attention_comparisons.empty() ||
                observed.cache_end == 0) {
              throw std::runtime_error(
                  "native VL decode full-attention observer order changed");
            }
            constexpr std::size_t kQueryBytes =
                16ULL * 256ULL * sizeof(std::uint16_t);
            constexpr std::size_t kKvRowBytes =
                2ULL * 256ULL * sizeof(std::uint16_t);
            const std::size_t cache_bytes =
                observed.cache_end * kKvRowBytes;
            const std::size_t current_offset =
                (observed.cache_end - 1) * kKvRowBytes;
            if (!expected.qkv_projection.empty()) {
              decode_full_attention_comparisons.push_back(
                  compare_native_oracle_tensor(
                      "qkv_projection", "bfloat16",
                      observed.qkv_projection,
                      9216ULL * sizeof(std::uint16_t),
                      expected.qkv_projection));
            }
            decode_full_attention_comparisons.push_back(
                compare_native_oracle_tensor(
                    "query", "bfloat16", observed.query, kQueryBytes,
                    expected.query));
            decode_full_attention_comparisons.push_back(
                compare_native_oracle_tensor_slice(
                    "current_key", "bfloat16", observed.current_key,
                    kKvRowBytes, expected.key_cache, current_offset));
            decode_full_attention_comparisons.push_back(
                compare_native_oracle_tensor_slice(
                    "current_value", "bfloat16", observed.current_value,
                    kKvRowBytes, expected.value_cache, current_offset));
            decode_full_attention_comparisons.push_back(
                compare_native_oracle_tensor_prefix(
                    "key_cache", "bfloat16", observed.key_cache,
                    cache_bytes,
                    expected.key_cache));
            decode_full_attention_comparisons.push_back(
                compare_native_oracle_tensor_prefix(
                    "value_cache", "bfloat16", observed.value_cache,
                    cache_bytes, expected.value_cache));
            decode_full_attention_comparisons.push_back(
                compare_native_oracle_tensor(
                    "output", "bfloat16", observed.attention_output,
                    kQueryBytes, expected.output));
            if (!expected.gated_attention.empty()) {
              decode_full_attention_comparisons.push_back(
                  compare_native_oracle_tensor(
                      "gated_attention", "bfloat16",
                      observed.gated_attention, kQueryBytes,
                      expected.gated_attention));
            }
            constexpr std::size_t kHiddenBytes =
                2048ULL * sizeof(std::uint16_t);
            if (!expected.projected_attention.empty()) {
              decode_full_attention_comparisons.push_back(
                  compare_native_oracle_tensor(
                      "projected_attention", "bfloat16",
                      observed.projected_attention, kHiddenBytes,
                      expected.projected_attention));
            }
            if (!expected.attention_residual.empty()) {
              decode_full_attention_comparisons.push_back(
                  compare_native_oracle_tensor(
                      "attention_residual", "bfloat16",
                      observed.attention_residual, kHiddenBytes,
                      expected.attention_residual));
            }
            if (!expected.post_attention_norm.empty()) {
              decode_full_attention_comparisons.push_back(
                  compare_native_oracle_tensor(
                      "post_attention_norm", "bfloat16",
                      observed.post_attention_norm, kHiddenBytes,
                      expected.post_attention_norm));
            }
            const auto compare_optional =
                [&](const char* label, const char* dtype,
                    const void* device_tensor, std::size_t tensor_bytes,
                    const std::filesystem::path& reference) {
                  if (reference.empty()) return;
                  decode_full_attention_comparisons.push_back(
                      compare_native_oracle_tensor(
                          label, dtype, device_tensor, tensor_bytes,
                          reference));
                };
            compare_optional(
                "shared_gate_logits", "bfloat16",
                observed.shared_gate_logits, sizeof(std::uint16_t),
                expected.shared_gate_logits);
            compare_optional(
                "shared_gate_up_projection", "bfloat16",
                observed.shared_gate_up_projection,
                1024ULL * sizeof(std::uint16_t),
                expected.shared_gate_up_projection);
            compare_optional(
                "shared_activation", "bfloat16",
                observed.shared_activation,
                512ULL * sizeof(std::uint16_t), expected.shared_activation);
            compare_optional(
                "shared_down_projection", "bfloat16",
                observed.shared_down_projection, kHiddenBytes,
                expected.shared_down_projection);
            compare_optional(
                "shared_moe_output", "bfloat16",
                observed.shared_moe_output, kHiddenBytes,
                expected.shared_moe_output);
            compare_optional(
                "router_logits", "bfloat16", observed.router_logits,
                256ULL * sizeof(std::uint16_t), expected.router_logits);
            compare_optional(
                "router_weights", "float32", observed.router_weights,
                8ULL * sizeof(float), expected.router_weights);
            compare_optional(
                "router_indices", "int32", observed.router_indices,
                8ULL * sizeof(std::int32_t), expected.router_indices);
            compare_optional(
                "routed_gate_up_projection", "bfloat16",
                observed.routed_gate_up_projection,
                8ULL * 1024ULL * sizeof(std::uint16_t),
                expected.routed_gate_up_projection);
            compare_optional(
                "routed_activation", "bfloat16",
                observed.routed_activation,
                8ULL * 512ULL * sizeof(std::uint16_t),
                expected.routed_activation);
            compare_optional(
                "routed_weighted_expert_outputs", "bfloat16",
                observed.routed_weighted_expert_outputs,
                8ULL * kHiddenBytes,
                expected.routed_weighted_expert_outputs);
            compare_optional(
                "routed_moe_output", "bfloat16",
                observed.routed_moe_output, kHiddenBytes,
                expected.routed_moe_output);
            compare_optional(
                "combined_moe_output", "bfloat16",
                observed.combined_moe_output, kHiddenBytes,
                expected.combined_moe_output);
          };
    }
    if (parsed.named_tool_json_constraint != nullptr) {
      const auto constraint = parsed.named_tool_json_constraint;
      request.next_token_mask =
          [constraint](const std::vector<std::uint32_t>& generated,
                       std::vector<std::uint8_t>* mask) {
            constraint->allowed_token_mask(generated, mask);
          };
    }

    const NativeResidentRequestMetrics metrics = engine.run(request);
    if (!reference_prefill_states.empty() &&
        prefill_state_comparisons.size() !=
            reference_prefill_states.size() * 2) {
      throw std::runtime_error(
          "native VL generation prefill state capture is incomplete");
    }
    if (!reference_decode_boundaries.empty() &&
        decode_boundary_comparisons.size() != 41) {
      throw std::runtime_error(
          "native VL generation decode boundary capture is incomplete");
    }
    if (!reference_decode_linear_boundaries.empty() &&
        decode_linear_boundary_comparisons.size() !=
            kDecodeLinearBoundaryContracts.size()) {
      throw std::runtime_error(
          "native VL generation decode linear capture is incomplete");
    }
    if (!reference_decode_layer0_tail_boundaries.empty() &&
        decode_layer0_tail_boundary_comparisons.size() !=
            kDecodeLayer0TailBoundaryContracts.size()) {
      throw std::runtime_error(
          "native VL generation decode layer-0 tail capture is incomplete");
    }
    if (reference_decode_full_attention.has_value() &&
        decode_full_attention_comparisons.size() !=
            reference_decode_full_attention->comparison_count()) {
      throw std::runtime_error(
          "native VL generation decode full-attention capture is incomplete");
    }
    if (metrics.output_token_ids.size() != expected_prefix.size() + 1) {
      throw std::runtime_error(
          "native VL generation ended before the diagnostic token");
    }
    const bool prefix_exact = std::equal(
        expected_prefix.begin(), expected_prefix.end(),
        metrics.output_token_ids.begin());
    if (!prefix_exact && !diagnostic_allow_prefix_divergence) {
      const auto mismatch = std::mismatch(
          expected_prefix.begin(), expected_prefix.end(),
          metrics.output_token_ids.begin());
      const std::size_t mismatch_index = static_cast<std::size_t>(
          std::distance(expected_prefix.begin(), mismatch.first));
      throw std::runtime_error(
          "native VL generation diverged before the diagnostic token: " +
          case_id + " output_index=" + std::to_string(mismatch_index) +
          " expected=" + std::to_string(expected_prefix[mismatch_index]) +
          " actual=" +
          std::to_string(metrics.output_token_ids[mismatch_index]));
    }
    const NativeLogitsComparison comparison =
        engine.compare_current_logits(reference_logits);
    const bool reference_row_bound =
        comparison.reference_top1_token_id == expected_reference_token;
    if (!reference_row_bound) {
      throw std::runtime_error(
          "VL generation reference row top1 differs from its contract");
    }
    const std::uint32_t selected_token = metrics.output_token_ids.back();
    const bool selected_reference =
        selected_token == expected_reference_token;
    const bool native_top1_exact = comparison.top1_match &&
                                   selected_reference &&
                                   comparison.actual_top1_token_id ==
                                       selected_token;
    all_prefixes_exact = all_prefixes_exact && prefix_exact;
    all_reference_rows_bound =
        all_reference_rows_bound && reference_row_bound;
    all_native_top1_exact = all_native_top1_exact && native_top1_exact;

    Json decode_boundaries = Json::array();
    bool decode_boundaries_finite = !decode_boundary_comparisons.empty();
    for (const NativeOracleComparison& boundary :
         decode_boundary_comparisons) {
      decode_boundaries.push_back(oracle_comparison_json(boundary));
      decode_boundaries_finite =
          decode_boundaries_finite &&
          boundary.finite_elements == boundary.elements;
    }
    Json prefill_states = Json::array();
    bool prefill_states_finite = !prefill_state_comparisons.empty();
    for (const NativeOracleComparison& state : prefill_state_comparisons) {
      prefill_states.push_back(oracle_comparison_json(state));
      prefill_states_finite =
          prefill_states_finite && state.finite_elements == state.elements;
    }
    Json decode_linear_boundaries = Json::array();
    bool decode_linear_boundaries_finite =
        !decode_linear_boundary_comparisons.empty();
    for (const NativeOracleComparison& boundary :
         decode_linear_boundary_comparisons) {
      decode_linear_boundaries.push_back(oracle_comparison_json(boundary));
      decode_linear_boundaries_finite =
          decode_linear_boundaries_finite &&
          boundary.finite_elements == boundary.elements;
    }
    Json decode_layer0_tail_boundaries = Json::array();
    bool decode_layer0_tail_boundaries_finite =
        !decode_layer0_tail_boundary_comparisons.empty();
    for (const NativeOracleComparison& boundary :
         decode_layer0_tail_boundary_comparisons) {
      decode_layer0_tail_boundaries.push_back(
          oracle_comparison_json(boundary));
      decode_layer0_tail_boundaries_finite =
          decode_layer0_tail_boundaries_finite &&
          boundary.finite_elements == boundary.elements;
    }
    Json decode_full_attention = Json::array();
    bool decode_full_attention_finite =
        !decode_full_attention_comparisons.empty();
    for (const NativeOracleComparison& boundary :
         decode_full_attention_comparisons) {
      decode_full_attention.push_back(oracle_comparison_json(boundary));
      decode_full_attention_finite =
          decode_full_attention_finite &&
          boundary.finite_elements == boundary.elements;
    }

    results.push_back(
        {{"case_id", case_id},
         {"complete", true},
         {"divergence_output_index", expected_prefix.size()},
         {"expected_prefix_token_ids", expected_prefix},
         {"expected_reference_token_id", expected_reference_token},
         {"selected_native_token_id", selected_token},
         {"output_token_ids", metrics.output_token_ids},
         {"prefix_exact", prefix_exact},
         {"diagnostic_prefix_divergence_allowed",
          diagnostic_allow_prefix_divergence},
         {"selected_reference_token", selected_reference},
         {"request_metrics", request_metrics_json(metrics)},
         {"prefill_states_complete",
          prefill_state_comparisons.size() == 60},
         {"prefill_states_finite", prefill_states_finite},
         {"prefill_states", std::move(prefill_states)},
         {"decode_boundaries_complete",
          decode_boundary_comparisons.size() == 41},
         {"decode_boundaries_finite", decode_boundaries_finite},
         {"decode_boundaries", std::move(decode_boundaries)},
         {"decode_linear_boundaries_complete",
          decode_linear_boundary_comparisons.size() ==
              kDecodeLinearBoundaryContracts.size()},
         {"decode_linear_boundaries_finite",
          decode_linear_boundaries_finite},
         {"decode_linear_boundaries",
          std::move(decode_linear_boundaries)},
         {"decode_layer0_tail_boundaries_complete",
          decode_layer0_tail_boundary_comparisons.size() ==
              kDecodeLayer0TailBoundaryContracts.size()},
         {"decode_layer0_tail_boundaries_finite",
          decode_layer0_tail_boundaries_finite},
         {"decode_layer0_tail_boundaries",
          std::move(decode_layer0_tail_boundaries)},
         {"decode_full_attention_complete",
          reference_decode_full_attention.has_value() &&
              decode_full_attention_comparisons.size() ==
                  reference_decode_full_attention->comparison_count()},
         {"decode_full_attention_finite",
          decode_full_attention_finite},
         {"decode_full_attention",
          std::move(decode_full_attention)},
         {"reference_logits",
          {{"elements", comparison.elements},
           {"exact_elements", comparison.exact_elements},
           {"finite_elements", comparison.finite_elements},
           {"reference_top1_token_id",
            comparison.reference_top1_token_id},
           {"actual_top1_token_id", comparison.actual_top1_token_id},
           {"top1_match", comparison.top1_match},
           {"maximum_absolute_error", comparison.maximum_absolute_error},
           {"relative_l2_error", comparison.relative_l2_error},
           {"kl_divergence", comparison.kl_divergence},
           {"expected_sha256", comparison.expected_sha256},
           {"actual_sha256", comparison.actual_sha256}}},
         {"native_top1_exact", native_top1_exact}});
  }

  std::cout
      << Json(
             {{"schema",
               "aima-amd395-qwen36/native-vl-generation-logits-probe/v1"},
              {"complete", true},
              {"qualified_for_attribution", all_prefixes_exact &&
                                                   all_reference_rows_bound},
              {"model_loads", engine.load_metrics().model_tensor_count > 0
                                    ? 1
                                    : 0},
              {"cases", std::move(results)},
              {"decision",
               {{"all_shared_prefixes_exact", all_prefixes_exact},
                {"all_reference_rows_bound", all_reference_rows_bound},
                {"all_decode_boundaries_compared",
                 all_decode_boundaries_compared},
                {"all_decode_linear_boundaries_compared",
                 all_decode_linear_boundaries_compared},
                {"all_decode_layer0_tail_boundaries_compared",
                 all_decode_layer0_tail_boundaries_compared},
                {"all_decode_full_attention_compared",
                 all_decode_full_attention_compared},
                {"all_prefill_states_compared",
                 all_prefill_states_compared},
                {"all_native_generation_top1_exact",
                 all_native_top1_exact},
                {"product_http_oracle_dependency", false}}}})
             .dump()
      << std::endl;
  return 0;
}

}  // namespace aima
