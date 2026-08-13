// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_remote_media.h"

#include <algorithm>
#include <arpa/inet.h>
#include <chrono>
#include <cstdint>
#include <curl/curl.h>
#include <cstring>
#include <filesystem>
#include <limits>
#include <mutex>
#include <netinet/in.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/socket.h>
#include <utility>

namespace aima {
namespace {

struct ParsedRemoteUrl {
  NativeMediaTransport transport = NativeMediaTransport::kHttpUrl;
  std::string host;
};

std::string lower_ascii(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char byte) {
                   if (byte >= 'A' && byte <= 'Z') {
                     return static_cast<char>(byte - 'A' + 'a');
                   }
                   return static_cast<char>(byte);
                 });
  return value;
}

bool exact_domain_match(std::string_view host,
                        const std::vector<std::string>& candidates) {
  return std::any_of(candidates.begin(), candidates.end(),
                     [&](std::string candidate) {
                       return lower_ascii(std::move(candidate)) == host;
                     });
}

ParsedRemoteUrl parse_remote_url(std::string_view source,
                                 const NativeMediaPolicy& policy) {
  std::string_view scheme;
  ParsedRemoteUrl parsed;
  if (source.rfind("http://", 0) == 0) {
    scheme = "http://";
    parsed.transport = NativeMediaTransport::kHttpUrl;
  } else if (source.rfind("https://", 0) == 0) {
    scheme = "https://";
    parsed.transport = NativeMediaTransport::kHttpsUrl;
  } else {
    throw std::invalid_argument("unsupported remote media URL scheme");
  }
  if (std::any_of(source.begin(), source.end(), [](unsigned char byte) {
        return byte <= 0x20 || byte == 0x7f || byte == '\\';
      })) {
    throw std::invalid_argument("remote media URL contains an unsafe character");
  }
  const std::string_view remainder = source.substr(scheme.size());
  const std::size_t authority_end = remainder.find_first_of("/?#");
  const std::string authority(remainder.substr(0, authority_end));
  if (authority.empty() || authority.find('@') != std::string::npos) {
    throw std::invalid_argument("remote media URL has an invalid authority");
  }
  std::string_view port;
  bool bracketed_ipv6 = false;
  if (authority.front() == '[') {
    bracketed_ipv6 = true;
    const std::size_t closing = authority.find(']');
    if (closing == std::string::npos) {
      throw std::invalid_argument("remote media URL has malformed IPv6");
    }
    parsed.host = authority.substr(1, closing - 1);
    if (closing + 1 < authority.size()) {
      if (authority[closing + 1] != ':') {
        throw std::invalid_argument("remote media URL has malformed authority");
      }
      port = std::string_view(authority).substr(closing + 2);
    }
  } else {
    const std::size_t colon = authority.rfind(':');
    if (colon != std::string::npos && authority.find(':') != colon) {
      throw std::invalid_argument(
          "remote media IPv6 addresses must use brackets");
    }
    parsed.host = colon == std::string::npos
                      ? authority
                      : authority.substr(0, colon);
    if (colon != std::string::npos) {
      port = std::string_view(authority).substr(colon + 1);
    }
  }
  parsed.host = lower_ascii(std::move(parsed.host));
  if (parsed.host.empty()) {
    throw std::invalid_argument("remote media URL has no host");
  }
  if (parsed.host.find('%') != std::string::npos) {
    throw std::invalid_argument(
        "remote media URL cannot percent-encode its authority");
  }
  if (bracketed_ipv6) {
    in6_addr address {};
    if (::inet_pton(AF_INET6, parsed.host.c_str(), &address) != 1) {
      throw std::invalid_argument("remote media URL has malformed IPv6");
    }
  } else {
    const bool valid = std::all_of(
        parsed.host.begin(), parsed.host.end(), [](unsigned char byte) {
          return (byte >= 'a' && byte <= 'z') ||
                 (byte >= '0' && byte <= '9') || byte == '.' || byte == '-';
        });
    if (!valid || parsed.host.front() == '.' || parsed.host.back() == '.' ||
        parsed.host.find("..") != std::string::npos) {
      throw std::invalid_argument("remote media URL has malformed host");
    }
  }
  if (!port.empty()) {
    if (!std::all_of(port.begin(), port.end(), [](unsigned char byte) {
          return byte >= '0' && byte <= '9';
        })) {
      throw std::invalid_argument("remote media URL has malformed port");
    }
    std::uint32_t parsed_port = 0;
    for (unsigned char byte : port) {
      const std::uint32_t digit = static_cast<std::uint32_t>(byte - '0');
      if (parsed_port > 6553 || (parsed_port == 6553 && digit > 5)) {
        throw std::invalid_argument("remote media URL has invalid port");
      }
      parsed_port = parsed_port * 10 + digit;
    }
    if (parsed_port == 0) {
      throw std::invalid_argument("remote media URL has invalid port");
    }
  } else if (authority.back() == ':') {
    throw std::invalid_argument("remote media URL has an empty port");
  }
  if (!exact_domain_match(parsed.host, policy.allowed_media_domains)) {
    throw std::invalid_argument("remote media domain is not allowlisted");
  }
  return parsed;
}

std::uint64_t maximum_bytes(NativeMediaKind kind,
                            const NativeMediaPolicy& policy) {
  const std::uint64_t value = kind == NativeMediaKind::kImage
                                  ? policy.maximum_image_bytes
                                  : policy.maximum_video_bytes;
  if (value == 0 || value > std::numeric_limits<std::size_t>::max() ||
      value > static_cast<std::uint64_t>(
                  std::numeric_limits<curl_off_t>::max())) {
    throw std::invalid_argument("native media byte limit is invalid");
  }
  return value;
}

std::uint32_t fetch_timeout_milliseconds(
    NativeMediaKind kind, const NativeMediaPolicy& policy) {
  const std::uint32_t value =
      kind == NativeMediaKind::kImage
          ? policy.maximum_image_fetch_milliseconds
          : policy.maximum_video_fetch_milliseconds;
  if (value == 0 || policy.maximum_remote_connect_milliseconds == 0 ||
      policy.remote_low_speed_bytes_per_second == 0 ||
      policy.remote_low_speed_seconds == 0 ||
      policy.maximum_remote_redirects > 20) {
    throw std::invalid_argument("remote media fetch policy is invalid");
  }
  return value;
}

std::string tls_ca_bundle(const NativeMediaPolicy& policy) {
  std::error_code error;
  std::filesystem::path candidate = policy.remote_tls_ca_bundle;
  const bool explicitly_configured = !candidate.empty();
#if defined(__linux__)
  if (candidate.empty()) {
    const std::filesystem::path executable =
        std::filesystem::read_symlink("/proc/self/exe", error);
    if (!error && !executable.empty()) {
      candidate = executable.parent_path().parent_path() / "share" / "certs" /
                  "ca-certificates.crt";
    }
  }
#endif
  if (candidate.empty()) return {};
  if (!std::filesystem::is_regular_file(candidate, error) || error) {
    if (!explicitly_configured) return {};
    throw std::runtime_error("remote TLS CA bundle is unavailable");
  }
  return candidate.string();
}

bool ipv4_is_loopback(const in_addr& address) {
  const std::uint32_t value = ntohl(address.s_addr);
  return (value & 0xff000000U) == 0x7f000000U;
}

bool ipv4_is_hard_blocked(const in_addr& address) {
  const std::uint32_t value = ntohl(address.s_addr);
  return (value & 0xff000000U) == 0x00000000U ||
         (value & 0xffc00000U) == 0x64400000U ||
         (value & 0xffff0000U) == 0xa9fe0000U ||
         (value & 0xfffe0000U) == 0xc6120000U || value >= 0xe0000000U;
}

bool ipv4_is_public(const in_addr& address) {
  const std::uint32_t value = ntohl(address.s_addr);
  if (ipv4_is_hard_blocked(address) ||
      (value & 0xff000000U) == 0x0a000000U ||
      (value & 0xff000000U) == 0x7f000000U ||
      (value & 0xfff00000U) == 0xac100000U ||
      (value & 0xffff0000U) == 0xc0a80000U) {
    return false;
  }
  return true;
}

bool ipv6_is_ipv4_mapped(const in6_addr& address, in_addr* mapped) {
  const unsigned char prefix[12] = {0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0xff, 0xff};
  if (std::memcmp(address.s6_addr, prefix, sizeof(prefix)) != 0) return false;
  std::memcpy(&mapped->s_addr, address.s6_addr + 12, sizeof(mapped->s_addr));
  return true;
}

bool ipv6_is_public(const in6_addr& address) {
  in_addr mapped {};
  if (ipv6_is_ipv4_mapped(address, &mapped)) return ipv4_is_public(mapped);
  if (IN6_IS_ADDR_UNSPECIFIED(&address) || IN6_IS_ADDR_LOOPBACK(&address) ||
      IN6_IS_ADDR_MULTICAST(&address) || IN6_IS_ADDR_LINKLOCAL(&address) ||
      (address.s6_addr[0] & 0xfeU) == 0xfcU) {
    return false;
  }
  return (address.s6_addr[0] & 0xe0U) == 0x20U;
}

enum class AddressRuleKind {
  kExactIpv4,
  kExactIpv6,
  kLoopbackOnly,
  kPublicOnly,
  kPrivateExplicitlyAllowed,
};

struct AddressRule {
  AddressRuleKind kind = AddressRuleKind::kPublicOnly;
  in_addr ipv4 {};
  in6_addr ipv6 {};
};

AddressRule address_rule(const ParsedRemoteUrl& parsed,
                         const NativeMediaPolicy& policy) {
  AddressRule rule;
  if (::inet_pton(AF_INET, parsed.host.c_str(), &rule.ipv4) == 1) {
    if (ipv4_is_hard_blocked(rule.ipv4)) {
      throw std::invalid_argument("remote media address is unsafe");
    }
    rule.kind = AddressRuleKind::kExactIpv4;
    return rule;
  }
  if (::inet_pton(AF_INET6, parsed.host.c_str(), &rule.ipv6) == 1) {
    if (IN6_IS_ADDR_UNSPECIFIED(&rule.ipv6) ||
        IN6_IS_ADDR_MULTICAST(&rule.ipv6) ||
        IN6_IS_ADDR_LINKLOCAL(&rule.ipv6)) {
      throw std::invalid_argument("remote media address is unsafe");
    }
    rule.kind = AddressRuleKind::kExactIpv6;
    return rule;
  }
  if (parsed.host == "localhost") {
    rule.kind = AddressRuleKind::kLoopbackOnly;
  } else if (exact_domain_match(parsed.host,
                                policy.allowed_private_media_domains)) {
    rule.kind = AddressRuleKind::kPrivateExplicitlyAllowed;
  }
  return rule;
}

bool socket_address_allowed(const sockaddr* address, const AddressRule& rule) {
  if (address->sa_family == AF_INET) {
    const auto* value = reinterpret_cast<const sockaddr_in*>(address);
    if (rule.kind == AddressRuleKind::kExactIpv4) {
      return value->sin_addr.s_addr == rule.ipv4.s_addr;
    }
    if (rule.kind == AddressRuleKind::kExactIpv6) return false;
    if (rule.kind == AddressRuleKind::kLoopbackOnly) {
      return ipv4_is_loopback(value->sin_addr);
    }
    if (rule.kind == AddressRuleKind::kPrivateExplicitlyAllowed) {
      return !ipv4_is_hard_blocked(value->sin_addr);
    }
    return ipv4_is_public(value->sin_addr);
  }
  if (address->sa_family == AF_INET6) {
    const auto* value = reinterpret_cast<const sockaddr_in6*>(address);
    if (rule.kind == AddressRuleKind::kExactIpv6) {
      return std::memcmp(&value->sin6_addr, &rule.ipv6,
                         sizeof(rule.ipv6)) == 0;
    }
    if (rule.kind == AddressRuleKind::kExactIpv4) {
      in_addr mapped {};
      return ipv6_is_ipv4_mapped(value->sin6_addr, &mapped) &&
             mapped.s_addr == rule.ipv4.s_addr;
    }
    if (rule.kind == AddressRuleKind::kLoopbackOnly) {
      return IN6_IS_ADDR_LOOPBACK(&value->sin6_addr) != 0;
    }
    if (rule.kind == AddressRuleKind::kPrivateExplicitlyAllowed) {
      return IN6_IS_ADDR_UNSPECIFIED(&value->sin6_addr) == 0 &&
             IN6_IS_ADDR_MULTICAST(&value->sin6_addr) == 0 &&
             IN6_IS_ADDR_LINKLOCAL(&value->sin6_addr) == 0;
    }
    return ipv6_is_public(value->sin6_addr);
  }
  return false;
}

curl_socket_t open_checked_socket(void* client_data,
                                  curlsocktype purpose,
                                  struct curl_sockaddr* address) {
  const auto* rule = static_cast<const AddressRule*>(client_data);
  if (purpose != CURLSOCKTYPE_IPCXN || address == nullptr ||
      !socket_address_allowed(&address->addr, *rule)) {
    return CURL_SOCKET_BAD;
  }
  int socket_type = address->socktype;
#if defined(SOCK_CLOEXEC)
  socket_type |= SOCK_CLOEXEC;
#endif
  return ::socket(address->family, socket_type, address->protocol);
}

struct WriteContext {
  std::vector<unsigned char> bytes;
  std::uint64_t* downloaded = nullptr;
  std::uint64_t maximum = 0;
  bool exceeded = false;
};

std::size_t write_remote_bytes(char* data, std::size_t size,
                               std::size_t count, void* opaque) {
  auto* context = static_cast<WriteContext*>(opaque);
  if (size != 0 && count > std::numeric_limits<std::size_t>::max() / size) {
    context->exceeded = true;
    return 0;
  }
  const std::size_t bytes = size * count;
  if (*context->downloaded > context->maximum ||
      bytes > context->maximum - *context->downloaded) {
    context->exceeded = true;
    return 0;
  }
  context->bytes.insert(context->bytes.end(),
                        reinterpret_cast<unsigned char*>(data),
                        reinterpret_cast<unsigned char*>(data) + bytes);
  *context->downloaded += bytes;
  return bytes;
}

class CurlHandle {
 public:
  CurlHandle() : value_(curl_easy_init()) {
    if (value_ == nullptr) {
      throw std::runtime_error("remote media client initialization failed");
    }
  }
  CurlHandle(const CurlHandle&) = delete;
  CurlHandle& operator=(const CurlHandle&) = delete;
  ~CurlHandle() { curl_easy_cleanup(value_); }
  CURL* get() const { return value_; }

 private:
  CURL* value_ = nullptr;
};

void require_curl(CURLcode result) {
  if (result != CURLE_OK) {
    throw std::runtime_error("remote media client configuration failed");
  }
}

void require_curl_or_compiled_out(CURLcode result) {
  if (result != CURLE_OK && result != CURLE_UNKNOWN_OPTION) {
    throw std::runtime_error("remote media client configuration failed");
  }
}

void initialize_curl() {
  static std::once_flag once;
  static CURLcode result = CURLE_FAILED_INIT;
  std::call_once(once, []() { result = curl_global_init(CURL_GLOBAL_DEFAULT); });
  if (result != CURLE_OK) {
    throw std::runtime_error("remote media client initialization failed");
  }
}

std::string redirect_url(CURL* curl) {
  char* value = nullptr;
  require_curl(curl_easy_getinfo(curl, CURLINFO_REDIRECT_URL, &value));
  if (value == nullptr || *value == '\0') {
    throw std::invalid_argument("remote media redirect is malformed");
  }
  return value;
}

std::chrono::milliseconds remaining_timeout(
    std::chrono::steady_clock::time_point deadline) {
  const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
      deadline - std::chrono::steady_clock::now());
  if (remaining.count() <= 0) {
    throw std::invalid_argument("remote media fetch exceeded the time limit");
  }
  return remaining;
}

bool is_redirect_status(long status) {
  return status == 301 || status == 302 || status == 303 || status == 307 ||
         status == 308;
}

}  // namespace

NativeMediaTransport validate_native_remote_media_source(
    const NativeMediaPart& media, const NativeMediaPolicy& policy) {
  return parse_remote_url(media.source, policy).transport;
}

std::vector<unsigned char> fetch_native_remote_media(
    const NativeMediaPart& media, const NativeMediaPolicy& policy) {
  const std::uint64_t byte_limit = maximum_bytes(media.kind, policy);
  const std::uint32_t timeout_ms =
      fetch_timeout_milliseconds(media.kind, policy);
  const std::string ca_bundle = tls_ca_bundle(policy);
  initialize_curl();
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(timeout_ms);
  std::string url = media.source;
  std::uint64_t downloaded = 0;
  NativeMediaTransport previous_transport =
      validate_native_remote_media_source(media, policy);
  for (std::uint32_t redirects = 0;; ++redirects) {
    const ParsedRemoteUrl parsed = parse_remote_url(url, policy);
    if (redirects != 0 &&
        previous_transport == NativeMediaTransport::kHttpsUrl &&
        parsed.transport == NativeMediaTransport::kHttpUrl) {
      throw std::invalid_argument("remote media redirect cannot downgrade TLS");
    }
    previous_transport = parsed.transport;
    const AddressRule rule = address_rule(parsed, policy);
    WriteContext output;
    output.downloaded = &downloaded;
    output.maximum = byte_limit;
    CurlHandle handle;
    CURL* curl = handle.get();
    const auto remaining = remaining_timeout(deadline);
    const long total_milliseconds = static_cast<long>(std::min<std::int64_t>(
        remaining.count(), std::numeric_limits<long>::max()));
    const long connect_milliseconds = static_cast<long>(std::min<std::uint64_t>(
        policy.maximum_remote_connect_milliseconds,
        static_cast<std::uint64_t>(total_milliseconds)));
    require_curl(curl_easy_setopt(curl, CURLOPT_URL, url.c_str()));
    require_curl(curl_easy_setopt(curl, CURLOPT_PROTOCOLS_STR, "http,https"));
    require_curl(
        curl_easy_setopt(curl, CURLOPT_REDIR_PROTOCOLS_STR, "http,https"));
    require_curl(curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 0L));
    require_curl(curl_easy_setopt(curl, CURLOPT_MAXREDIRS, 0L));
    require_curl(curl_easy_setopt(curl, CURLOPT_HTTPGET, 1L));
    require_curl(curl_easy_setopt(curl, CURLOPT_HTTP_VERSION,
                                  CURL_HTTP_VERSION_1_1));
    require_curl_or_compiled_out(curl_easy_setopt(curl, CURLOPT_PROXY, ""));
    require_curl_or_compiled_out(curl_easy_setopt(curl, CURLOPT_NOPROXY, "*"));
    require_curl_or_compiled_out(
        curl_easy_setopt(curl, CURLOPT_NETRC, CURL_NETRC_IGNORED));
    require_curl(curl_easy_setopt(curl, CURLOPT_DISALLOW_USERNAME_IN_URL, 1L));
    require_curl(curl_easy_setopt(curl, CURLOPT_UNRESTRICTED_AUTH, 0L));
    require_curl(curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L));
    require_curl(curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 2L));
    if (!ca_bundle.empty()) {
      require_curl(curl_easy_setopt(curl, CURLOPT_CAINFO, ca_bundle.c_str()));
    }
    require_curl(curl_easy_setopt(curl, CURLOPT_SSLVERSION,
                                  CURL_SSLVERSION_TLSv1_2));
    require_curl(curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L));
    require_curl(curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT_MS,
                                  connect_milliseconds));
    require_curl(
        curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, total_milliseconds));
    require_curl(curl_easy_setopt(
        curl, CURLOPT_LOW_SPEED_LIMIT,
        static_cast<long>(policy.remote_low_speed_bytes_per_second)));
    require_curl(curl_easy_setopt(
        curl, CURLOPT_LOW_SPEED_TIME,
        static_cast<long>(policy.remote_low_speed_seconds)));
    require_curl(curl_easy_setopt(curl, CURLOPT_MAXFILESIZE_LARGE,
                                  static_cast<curl_off_t>(byte_limit)));
    require_curl(curl_easy_setopt(curl, CURLOPT_OPENSOCKETFUNCTION,
                                  open_checked_socket));
    require_curl(curl_easy_setopt(curl, CURLOPT_OPENSOCKETDATA, &rule));
    require_curl(
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_remote_bytes));
    require_curl(curl_easy_setopt(curl, CURLOPT_WRITEDATA, &output));
    require_curl(curl_easy_setopt(curl, CURLOPT_USERAGENT,
                                  "aima-engine-native/1"));
    const CURLcode result = curl_easy_perform(curl);
    if (output.exceeded || result == CURLE_FILESIZE_EXCEEDED) {
      throw std::invalid_argument("remote media payload exceeds the byte limit");
    }
    if (result == CURLE_OPERATION_TIMEDOUT) {
      throw std::invalid_argument("remote media fetch exceeded the time limit");
    }
    if (result != CURLE_OK) {
      throw std::invalid_argument("remote media fetch failed");
    }
    long status = 0;
    require_curl(curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status));
    if (is_redirect_status(status)) {
      if (redirects >= policy.maximum_remote_redirects) {
        throw std::invalid_argument("remote media redirect limit exceeded");
      }
      url = redirect_url(curl);
      continue;
    }
    if (status != 200) {
      throw std::invalid_argument("remote media server returned an invalid status");
    }
    if (output.bytes.empty()) {
      throw std::invalid_argument("remote media payload is empty");
    }
    return std::move(output.bytes);
  }
}

}  // namespace aima
