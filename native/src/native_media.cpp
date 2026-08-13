// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_media.h"

#include "aima/sha256.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

namespace aima {
namespace {

std::string lower_ascii(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char byte) {
                   return static_cast<char>(std::tolower(byte));
                 });
  return value;
}

std::uint64_t maximum_bytes(NativeMediaKind kind,
                            const NativeMediaPolicy& policy) {
  const std::uint64_t value = kind == NativeMediaKind::kImage
                                  ? policy.maximum_image_bytes
                                  : policy.maximum_video_bytes;
  if (value == 0) {
    throw std::invalid_argument("native media byte limit must be positive");
  }
  return value;
}

int hexadecimal_value(char value) {
  if (value >= '0' && value <= '9') return value - '0';
  if (value >= 'a' && value <= 'f') return value - 'a' + 10;
  if (value >= 'A' && value <= 'F') return value - 'A' + 10;
  return -1;
}

std::string percent_decode_path(std::string_view value) {
  std::string decoded;
  decoded.reserve(value.size());
  for (std::size_t index = 0; index < value.size(); ++index) {
    if (value[index] != '%') {
      decoded.push_back(value[index]);
      continue;
    }
    if (index + 2 >= value.size()) {
      throw std::invalid_argument("local media URI has malformed percent encoding");
    }
    const int high = hexadecimal_value(value[index + 1]);
    const int low = hexadecimal_value(value[index + 2]);
    if (high < 0 || low < 0) {
      throw std::invalid_argument("local media URI has malformed percent encoding");
    }
    const char byte = static_cast<char>((high << 4) | low);
    if (byte == '\0') {
      throw std::invalid_argument("local media URI contains a NUL byte");
    }
    decoded.push_back(byte);
    index += 2;
  }
  return decoded;
}

bool path_is_within(const std::filesystem::path& candidate,
                    const std::filesystem::path& root) {
  auto candidate_part = candidate.begin();
  for (auto root_part = root.begin(); root_part != root.end(); ++root_part) {
    if (candidate_part == candidate.end() || *candidate_part != *root_part) {
      return false;
    }
    ++candidate_part;
  }
  return true;
}

std::filesystem::path admitted_local_path(
    const NativeMediaPart& media, const NativeMediaPolicy& policy) {
  constexpr std::string_view prefix = "file://";
  if (media.source.rfind(prefix, 0) != 0) {
    throw std::invalid_argument("local media source must use file://");
  }
  std::string_view encoded(media.source);
  encoded.remove_prefix(prefix.size());
  if (encoded.empty() || encoded.front() != '/') {
    throw std::invalid_argument(
        "local media URI authority must be empty and path absolute");
  }
  if (encoded.find_first_of("?#") != std::string_view::npos) {
    throw std::invalid_argument("local media URI cannot contain query or fragment");
  }
  if (policy.allowed_local_roots.empty()) {
    throw std::invalid_argument("local media access has no allowed root");
  }
  std::error_code error;
  const std::filesystem::path candidate = std::filesystem::canonical(
      std::filesystem::path(percent_decode_path(encoded)), error);
  if (error || candidate.empty()) {
    throw std::invalid_argument("local media path is unavailable");
  }
  bool admitted = false;
  for (const std::filesystem::path& configured_root :
       policy.allowed_local_roots) {
    const std::filesystem::path root =
        std::filesystem::canonical(configured_root, error);
    if (error || root.empty() || !std::filesystem::is_directory(root)) {
      throw std::invalid_argument("configured local media root is unavailable");
    }
    if (path_is_within(candidate, root)) {
      admitted = true;
      break;
    }
  }
  if (!admitted) {
    throw std::invalid_argument("local media path is outside the allowed roots");
  }
  if (!std::filesystem::is_regular_file(candidate, error) || error) {
    throw std::invalid_argument("local media path is not a regular file");
  }
  return candidate;
}

int base64_value(unsigned char value) {
  if (value >= 'A' && value <= 'Z') return value - 'A';
  if (value >= 'a' && value <= 'z') return value - 'a' + 26;
  if (value >= '0' && value <= '9') return value - '0' + 52;
  if (value == '+') return 62;
  if (value == '/') return 63;
  return -1;
}

std::vector<unsigned char> decode_base64(std::string_view encoded,
                                         std::uint64_t limit) {
  if (encoded.empty() || encoded.size() % 4 != 0) {
    throw std::invalid_argument("media data URI has malformed base64 length");
  }
  const std::uint64_t quartets =
      limit / 3 + static_cast<std::uint64_t>(limit % 3 != 0);
  const std::uint64_t maximum_encoded =
      quartets > std::numeric_limits<std::uint64_t>::max() / 4
          ? std::numeric_limits<std::uint64_t>::max()
          : quartets * 4;
  if (encoded.size() > maximum_encoded) {
    throw std::invalid_argument("media data URI exceeds the byte limit");
  }
  std::vector<unsigned char> decoded;
  decoded.reserve(encoded.size() / 4 * 3);
  for (std::size_t index = 0; index < encoded.size(); index += 4) {
    std::array<int, 4> values{};
    std::size_t padding = 0;
    for (std::size_t offset = 0; offset < 4; ++offset) {
      const unsigned char byte =
          static_cast<unsigned char>(encoded[index + offset]);
      if (byte == '=') {
        values[offset] = 0;
        ++padding;
      } else {
        if (padding != 0 || (values[offset] = base64_value(byte)) < 0) {
          throw std::invalid_argument("media data URI has invalid base64");
        }
      }
    }
    if (padding > 2 || (padding != 0 && index + 4 != encoded.size())) {
      throw std::invalid_argument("media data URI has invalid base64 padding");
    }
    if ((padding == 2 &&
         (encoded[index + 2] != '=' || encoded[index + 3] != '=' ||
          (values[1] & 0x0f) != 0)) ||
        (padding == 1 &&
         (encoded[index + 3] != '=' || (values[2] & 0x03) != 0))) {
      throw std::invalid_argument(
          "media data URI has non-canonical base64 padding");
    }
    const std::uint32_t word =
        (static_cast<std::uint32_t>(values[0]) << 18U) |
        (static_cast<std::uint32_t>(values[1]) << 12U) |
        (static_cast<std::uint32_t>(values[2]) << 6U) |
        static_cast<std::uint32_t>(values[3]);
    decoded.push_back(static_cast<unsigned char>((word >> 16U) & 0xffU));
    if (padding < 2) {
      decoded.push_back(static_cast<unsigned char>((word >> 8U) & 0xffU));
    }
    if (padding == 0) decoded.push_back(static_cast<unsigned char>(word & 0xffU));
  }
  if (decoded.empty() || decoded.size() > limit) {
    throw std::invalid_argument("decoded media payload is empty or too large");
  }
  return decoded;
}

std::string sniff_mime(const std::vector<unsigned char>& bytes) {
  const auto starts = [&](std::initializer_list<unsigned char> prefix) {
    return bytes.size() >= prefix.size() &&
           std::equal(prefix.begin(), prefix.end(), bytes.begin());
  };
  if (starts({0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a})) {
    return "image/png";
  }
  if (starts({0xff, 0xd8, 0xff})) return "image/jpeg";
  if (bytes.size() >= 12 && starts({'R', 'I', 'F', 'F'}) &&
      std::equal(bytes.begin() + 8, bytes.begin() + 12, "WEBP")) {
    return "image/webp";
  }
  if (bytes.size() >= 12 && starts({'R', 'I', 'F', 'F'}) &&
      std::equal(bytes.begin() + 8, bytes.begin() + 12, "AVI ")) {
    return "video/x-msvideo";
  }
  if (bytes.size() >= 12 &&
      std::equal(bytes.begin() + 4, bytes.begin() + 8, "ftyp")) {
    return "video/mp4";
  }
  throw std::invalid_argument("media payload has an unsupported or corrupt format");
}

void require_kind_matches(NativeMediaKind kind, std::string_view mime) {
  const bool image = mime.rfind("image/", 0) == 0;
  if ((kind == NativeMediaKind::kImage) != image) {
    throw std::invalid_argument("media content type does not match the content part");
  }
}

std::pair<std::string, std::vector<unsigned char>> parse_data_uri(
    const NativeMediaPart& media, const NativeMediaPolicy& policy) {
  if (!policy.allow_data_uri) {
    throw std::invalid_argument("media data URI transport is disabled");
  }
  const std::size_t comma = media.source.find(',');
  if (comma == std::string::npos) {
    throw std::invalid_argument("media data URI has no payload separator");
  }
  const std::string metadata = lower_ascii(media.source.substr(5, comma - 5));
  constexpr std::string_view suffix = ";base64";
  if (metadata.size() <= suffix.size() ||
      metadata.compare(metadata.size() - suffix.size(), suffix.size(), suffix) != 0) {
    throw std::invalid_argument("media data URI must use base64 encoding");
  }
  const std::string declared_mime =
      metadata.substr(0, metadata.size() - suffix.size());
  std::vector<unsigned char> bytes = decode_base64(
      std::string_view(media.source).substr(comma + 1),
      maximum_bytes(media.kind, policy));
  const std::string actual_mime = sniff_mime(bytes);
  require_kind_matches(media.kind, actual_mime);
  const bool equivalent_jpeg = declared_mime == "image/jpg" &&
                               actual_mime == "image/jpeg";
  const bool equivalent_avi = declared_mime == "video/avi" &&
                              actual_mime == "video/x-msvideo";
  if (declared_mime != actual_mime && !equivalent_jpeg && !equivalent_avi) {
    throw std::invalid_argument("media data URI MIME does not match its payload");
  }
  return {actual_mime, std::move(bytes)};
}

std::string remote_host(std::string_view source, std::string_view scheme) {
  if (source.find_first_of("\\\r\n\t ") != std::string_view::npos) {
    throw std::invalid_argument("remote media URL contains an unsafe character");
  }
  std::string_view remainder = source.substr(scheme.size());
  const std::size_t authority_end = remainder.find_first_of("/?#");
  std::string authority(remainder.substr(0, authority_end));
  if (authority.empty() || authority.find('@') != std::string::npos) {
    throw std::invalid_argument("remote media URL has an invalid authority");
  }
  std::string host;
  std::string_view port;
  bool bracketed_ipv6 = false;
  if (authority.front() == '[') {
    bracketed_ipv6 = true;
    const std::size_t closing = authority.find(']');
    if (closing == std::string::npos) {
      throw std::invalid_argument("remote media URL has malformed IPv6");
    }
    host = authority.substr(1, closing - 1);
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
    host = colon == std::string::npos ? authority : authority.substr(0, colon);
    if (colon != std::string::npos) {
      port = std::string_view(authority).substr(colon + 1);
    }
  }
  host = lower_ascii(std::move(host));
  if (host.empty()) throw std::invalid_argument("remote media URL has no host");
  if (host.find('%') != std::string::npos) {
    throw std::invalid_argument(
        "remote media URL cannot percent-encode its authority");
  }
  if (bracketed_ipv6) {
    const bool valid = std::all_of(
        host.begin(), host.end(), [](unsigned char byte) {
          return std::isxdigit(byte) != 0 || byte == ':' || byte == '.';
        });
    if (!valid || host.find(':') == std::string::npos) {
      throw std::invalid_argument("remote media URL has malformed IPv6");
    }
  } else {
    const bool valid = std::all_of(
        host.begin(), host.end(), [](unsigned char byte) {
          return std::isalnum(byte) != 0 || byte == '.' || byte == '-';
        });
    if (!valid || host.front() == '.' || host.back() == '.' ||
        host.find("..") != std::string::npos) {
      throw std::invalid_argument("remote media URL has malformed host");
    }
  }
  if (!port.empty()) {
    if (!std::all_of(port.begin(), port.end(), [](unsigned char byte) {
          return std::isdigit(byte) != 0;
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
  return host;
}

void require_allowed_domain(std::string host,
                            const NativeMediaPolicy& policy) {
  const bool allowed = std::any_of(
      policy.allowed_media_domains.begin(), policy.allowed_media_domains.end(),
      [&](std::string candidate) {
        return lower_ascii(std::move(candidate)) == host;
      });
  if (!allowed) {
    throw std::invalid_argument("remote media domain is not allowlisted");
  }
}

NativeMediaPayload finalize_payload(
    const NativeMediaPart& media, NativeMediaTransport transport,
    std::string mime, std::vector<unsigned char> bytes) {
  NativeMediaPayload payload;
  payload.kind = media.kind;
  payload.transport = transport;
  payload.mime_type = std::move(mime);
  payload.content_sha256 = sha256_bytes(bytes.data(), bytes.size());
  payload.bytes = std::move(bytes);
  return payload;
}

}  // namespace

NativeMediaTransport validate_native_media_source(
    const NativeMediaPart& media, const NativeMediaPolicy& policy) {
  if (media.source.rfind("data:", 0) == 0) {
    if (!policy.allow_data_uri) {
      throw std::invalid_argument("media data URI transport is disabled");
    }
    return NativeMediaTransport::kDataUri;
  }
  if (media.source.rfind("file://", 0) == 0) {
    (void)admitted_local_path(media, policy);
    return NativeMediaTransport::kLocalFile;
  }
  if (media.source.rfind("http://", 0) == 0) {
    require_allowed_domain(remote_host(media.source, "http://"), policy);
    return NativeMediaTransport::kHttpUrl;
  }
  if (media.source.rfind("https://", 0) == 0) {
    require_allowed_domain(remote_host(media.source, "https://"), policy);
    return NativeMediaTransport::kHttpsUrl;
  }
  throw std::invalid_argument("unsupported media source scheme");
}

NativeMediaPayload load_native_media_payload(
    const NativeMediaPart& media, const NativeMediaPolicy& policy) {
  const NativeMediaTransport transport =
      validate_native_media_source(media, policy);
  if (transport == NativeMediaTransport::kDataUri) {
    auto parsed = parse_data_uri(media, policy);
    return finalize_payload(media, transport, std::move(parsed.first),
                            std::move(parsed.second));
  }
  if (transport == NativeMediaTransport::kLocalFile) {
    const std::filesystem::path path = admitted_local_path(media, policy);
    std::error_code error;
    const std::uintmax_t size = std::filesystem::file_size(path, error);
    if (error || size == 0 || size > maximum_bytes(media.kind, policy) ||
        size > std::numeric_limits<std::size_t>::max()) {
      throw std::invalid_argument("local media file is empty or too large");
    }
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::invalid_argument("local media file cannot be opened");
    std::vector<unsigned char> bytes(static_cast<std::size_t>(size));
    input.read(reinterpret_cast<char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
    if (!input || input.peek() != std::ifstream::traits_type::eof()) {
      throw std::invalid_argument("local media file changed while being read");
    }
    std::string mime = sniff_mime(bytes);
    require_kind_matches(media.kind, mime);
    return finalize_payload(media, transport, std::move(mime), std::move(bytes));
  }
  throw std::invalid_argument(
      "remote media transport is validated but not linked in this build");
}

std::string_view native_media_kind_name(NativeMediaKind kind) {
  return kind == NativeMediaKind::kImage ? "image" : "video";
}

std::string_view native_media_transport_name(NativeMediaTransport transport) {
  switch (transport) {
    case NativeMediaTransport::kDataUri:
      return "data-uri";
    case NativeMediaTransport::kLocalFile:
      return "local-file";
    case NativeMediaTransport::kHttpUrl:
      return "http";
    case NativeMediaTransport::kHttpsUrl:
      return "https";
  }
  return "unknown";
}

}  // namespace aima
