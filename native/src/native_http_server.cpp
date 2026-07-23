// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_http_server.h"

#include "aima/native_resident_engine.h"
#include "aima/native_tokenizer.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <nlohmann/json.hpp>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace aima {
namespace {

using Json = nlohmann::json;
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

void send_all(int fd, std::string_view value) {
  std::size_t sent = 0;
  while (sent < value.size()) {
    const ssize_t result =
        ::send(fd, value.data() + sent, value.size() - sent, MSG_NOSIGNAL);
    if (result < 0 && errno == EINTR) continue;
    if (result <= 0) throw std::runtime_error("HTTP response send failed");
    sent += static_cast<std::size_t>(result);
  }
}

std::string status_text(int status) {
  switch (status) {
    case 200: return "OK";
    case 400: return "Bad Request";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 413: return "Payload Too Large";
    case 500: return "Internal Server Error";
    default: return "Error";
  }
}

void send_json(int fd, int status, const Json& payload) {
  const std::string body = payload.dump();
  std::string header =
      "HTTP/1.1 " + std::to_string(status) + " " + status_text(status) +
      "\r\nContent-Type: application/json\r\nContent-Length: " +
      std::to_string(body.size()) +
      "\r\nConnection: close\r\nX-Content-Type-Options: nosniff\r\n\r\n";
  send_all(fd, header);
  send_all(fd, body);
}

Json error_payload(const std::string& message, const std::string& code,
                   const std::string& type = "invalid_request_error") {
  return {{"error", {{"message", message},
                      {"type", type},
                      {"param", nullptr},
                      {"code", code}}}};
}

HttpRequest read_request(int fd) {
  std::string wire;
  wire.reserve(8192);
  std::size_t header_end = std::string::npos;
  while (header_end == std::string::npos) {
    char buffer[8192];
    const ssize_t count = ::recv(fd, buffer, sizeof(buffer), 0);
    if (count < 0 && errno == EINTR) continue;
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
    const ssize_t count =
        ::recv(fd, buffer, std::min(sizeof(buffer), remaining), 0);
    if (count < 0 && errno == EINTR) continue;
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

struct ServerOptions {
  NativeResidentEngineOptions engine;
  std::string host = "127.0.0.1";
  int port = 8000;
  std::size_t maximum_requests = 0;
};

ServerOptions parse_options(int argc, char** argv) {
  ServerOptions options;
  options.engine.weights.native_report =
      std::filesystem::absolute("native-http-weight-load.json");
  bool have_model = false;
  bool cache_capacity_explicit = false;
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
    } else if (argument == "--disable-decode-moe-overlap") {
      options.engine.decode_moe_overlap = false;
    } else if (argument == "--host") {
      options.host = next("--host");
    } else if (argument == "--port") {
      options.port = parse_port(next("--port"));
    } else if (argument == "--max-requests") {
      options.maximum_requests =
          parse_size(next("--max-requests"), "--max-requests");
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

std::pair<std::string, std::string> parse_messages(const Json& request) {
  if (!request.contains("messages") || !request["messages"].is_array() ||
      request["messages"].empty()) {
    throw std::invalid_argument("messages must be a non-empty array");
  }
  std::string system;
  std::string user;
  bool saw_user = false;
  for (const Json& message : request["messages"]) {
    if (!message.is_object() || !message.contains("role") ||
        !message.contains("content") || !message["role"].is_string() ||
        !message["content"].is_string()) {
      throw std::invalid_argument(
          "each message requires string role and content");
    }
    const std::string role = message["role"].get<std::string>();
    const std::string content = message["content"].get<std::string>();
    if (role == "system" && !saw_user) {
      if (!system.empty()) system += '\n';
      system += content;
    } else if (role == "user") {
      saw_user = true;
      if (!user.empty()) user += '\n';
      user += content;
    } else {
      throw std::invalid_argument(
          "only leading system messages followed by user messages are supported");
    }
  }
  if (user.empty()) throw std::invalid_argument("a user message is required");
  return {system, user};
}

Json request_metrics_json(const NativeResidentRequestMetrics& metrics) {
  return {{"runtime", "native-resident-q" +
                          std::to_string(metrics.prompt_tokens)},
          {"oracle_tensor_reads", metrics.oracle_tensor_reads},
          {"request_index", metrics.request_index},
          {"model_loads", metrics.model_loads},
          {"output_token_ids_sha256", metrics.output_token_ids_sha256},
          {"prefill_tokens_per_second", metrics.prefill_tokens_per_second},
          {"decode_tokens_per_second", metrics.decode_tokens_per_second},
          {"decode_layer_submission_ms",
           metrics.decode_layer_submission_ms},
          {"ttft_ms", metrics.prefill_wall_ms},
          {"request_wall_ms", metrics.request_wall_ms},
          {"prefix_cache", {{"implemented", true},
                            {"scope", "one-entry-exact-or-append"},
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
}

Json chat_completion(NativeResidentEngine& engine, NativeTokenizer& tokenizer,
                     const Json& request, std::size_t admitted_prompt_tokens) {
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
  if (request.value("stream", false)) {
    throw std::invalid_argument("streaming is not supported");
  }
  if (request.contains("tools") || request.contains("functions") ||
      request.contains("response_format") || request.contains("stop")) {
    throw std::invalid_argument(
        "tools, functions, response_format, and custom stop are not supported");
  }
  const auto [system, user] = parse_messages(request);
  std::vector<std::uint32_t> prompt =
      tokenizer.encode_chat(system, user, true);
  if (prompt.size() < admitted_prompt_tokens) {
    throw std::invalid_argument(
        "the current native product route requires a cold prompt of " +
        std::to_string(admitted_prompt_tokens) +
        " tokens or a longer prompt extending that cached prefix; encoded "
        "prompt has " +
        std::to_string(prompt.size()));
  }
  std::size_t max_tokens = 16;
  if (request.contains("max_completion_tokens")) {
    if (!request["max_completion_tokens"].is_number_unsigned()) {
      throw std::invalid_argument("max_completion_tokens must be a positive integer");
    }
    max_tokens = request["max_completion_tokens"].get<std::size_t>();
  } else if (request.contains("max_tokens")) {
    if (!request["max_tokens"].is_number_unsigned()) {
      throw std::invalid_argument("max_tokens must be a positive integer");
    }
    max_tokens = request["max_tokens"].get<std::size_t>();
  }
  if (max_tokens == 0) {
    throw std::invalid_argument("max_tokens must be positive");
  }

  NativeResidentRequestOptions native_request;
  native_request.input_token_ids = std::move(prompt);
  native_request.max_new_tokens = max_tokens;
  native_request.stop_token_ids = {tokenizer.eos_token_id()};
  const NativeResidentRequestMetrics metrics = engine.run(native_request);
  std::vector<std::uint32_t> visible = metrics.output_token_ids;
  if (metrics.stopped && !visible.empty() &&
      visible.back() == tokenizer.eos_token_id()) {
    visible.pop_back();
  }
  const std::string content = tokenizer.decode(visible);
  const std::int64_t created = std::chrono::duration_cast<std::chrono::seconds>(
                                   std::chrono::system_clock::now()
                                       .time_since_epoch())
                                   .count();
  const std::string id =
      "chatcmpl-native-" + std::to_string(metrics.request_index);
  return {{"id", id},
          {"object", "chat.completion"},
          {"created", created},
          {"model", kModelId},
          {"choices", Json::array({{{"index", 0},
                                     {"message", {{"role", "assistant"},
                                                  {"content", content}}},
                                     {"finish_reason",
                                      metrics.stopped ? "stop" : "length"}}})},
          {"usage", {{"prompt_tokens", metrics.prompt_tokens},
                     {"completion_tokens", metrics.completion_tokens},
                     {"total_tokens", metrics.prompt_tokens +
                                          metrics.completion_tokens}}},
          {"aima_amd395", request_metrics_json(metrics)}};
}

}  // namespace

int run_native_http_server(int argc, char** argv) {
  const ServerOptions options = parse_options(argc, argv);
  g_shutdown.store(false);
  const auto process_started = std::chrono::steady_clock::now();

  NativeTokenizer tokenizer;
  tokenizer.load(options.engine.weights.model_dir);
  NativeResidentEngine engine;
  const NativeResidentLoadMetrics load = engine.load(options.engine);

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
  if (::listen(server.get(), 16) != 0) {
    throw std::runtime_error("listen failed");
  }
  (void)::signal(SIGINT, signal_handler);
  (void)::signal(SIGTERM, signal_handler);
  const double command_to_ready_ms = elapsed_ms(process_started);
  std::cout << Json({{"event", "ready"},
                     {"host", options.host},
                     {"port", options.port},
                     {"model", kModelId},
                     {"command_to_ready_wall_ms", command_to_ready_ms},
                     {"engine_load_wall_ms", load.command_to_ready_wall_ms},
                     {"decode_moe_overlap", load.decode_moe_overlap},
                     {"fmha_provider_backend", load.fmha_provider_backend},
                     {"fmha_provider_path", load.fmha_provider_path},
                     {"secondary_fmha_provider_backend",
                      load.secondary_fmha_provider_backend},
                     {"secondary_fmha_provider_path",
                      load.secondary_fmha_provider_path},
                     {"secondary_fmha_layers",
                      load.secondary_fmha_layers},
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
      const HttpRequest request = read_request(client.get());
      if (request.method == "GET" && request.path == "/health") {
        send_json(client.get(), 200,
                  {{"status", "ok"},
                   {"model", kModelId},
                   {"model_loaded", engine.loaded()},
                   {"resident", true},
                   {"served", served},
                   {"uptime_ms", elapsed_ms(process_started)},
                   {"command_to_ready_wall_ms", command_to_ready_ms},
                   {"runtime", "native"},
                   {"decode_moe_overlap", load.decode_moe_overlap},
                   {"fmha_provider_backend", load.fmha_provider_backend},
                   {"secondary_fmha_provider_backend",
                    load.secondary_fmha_provider_backend},
                   {"secondary_fmha_layers", load.secondary_fmha_layers},
                   {"admitted_prompt_tokens", load.prompt_tokens}});
      } else if (request.method == "GET" && request.path == "/v1/models") {
        send_json(client.get(), 200,
                  {{"object", "list"},
                   {"data", Json::array({{{"id", kModelId},
                                           {"object", "model"},
                                           {"owned_by", "approaching-ai"}}})}});
      } else if (request.method == "POST" && request.path == "/shutdown") {
        send_json(client.get(), 200, {{"status", "shutting_down"}});
        g_shutdown.store(true);
      } else if (request.method == "POST" &&
                 request.path == "/v1/chat/completions") {
        Json body;
        try {
          body = Json::parse(request.body);
          const auto started = std::chrono::steady_clock::now();
          Json response = chat_completion(
              engine, tokenizer, body, load.prompt_tokens);
          response["aima_amd395"]["http_request_wall_ms"] =
              elapsed_ms(started);
          send_json(client.get(), 200, response);
          ++served;
        } catch (const std::invalid_argument& error) {
          send_json(client.get(), 400,
                    error_payload(error.what(), "bad_request"));
        } catch (const Json::exception& error) {
          send_json(client.get(), 400,
                    error_payload(error.what(), "invalid_json"));
        } catch (const std::exception& error) {
          send_json(client.get(), 500,
                    error_payload(error.what(), "native_engine_error",
                                  "server_error"));
        }
      } else {
        const int status = request.path == "/health" ||
                                   request.path == "/v1/models" ||
                                   request.path == "/shutdown" ||
                                   request.path == "/v1/chat/completions"
                               ? 405
                               : 404;
        send_json(client.get(), status,
                  error_payload("unsupported method or path",
                                status == 404 ? "not_found"
                                              : "method_not_allowed"));
      }
    } catch (const std::length_error& error) {
      send_json(client.get(), 413,
                error_payload(error.what(), "request_too_large"));
    } catch (const std::exception& error) {
      send_json(client.get(), 400,
                error_payload(error.what(), "bad_request"));
    }
    if (options.maximum_requests != 0 && served >= options.maximum_requests) {
      g_shutdown.store(true);
    }
  }
  std::cout << Json({{"event", "stopped"},
                     {"served", served},
                     {"model_loads", 1}})
                   .dump()
            << std::endl;
  return 0;
}

}  // namespace aima
