// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace aima {

enum class NativeMediaKind {
  kImage,
  kVideo,
};

enum class NativeMediaTransport {
  kDataUri,
  kLocalFile,
  kHttpUrl,
  kHttpsUrl,
};

// An ordered OpenAI content part. The source is retained only for the native
// media loader; prompts receive the model's canonical special-token marker.
struct NativeMediaPart {
  NativeMediaKind kind = NativeMediaKind::kImage;
  std::string source;
  std::size_t message_index = 0;
  std::size_t content_part_index = 0;
  std::size_t media_index = 0;
};

struct NativeMediaPolicy {
  std::vector<std::filesystem::path> allowed_local_roots;
  // Exact host matches only. Subdomains are never implied.
  std::vector<std::string> allowed_media_domains;
  // Exact hostname subset permitted to resolve to RFC1918/ULA addresses.
  // Literal private IPs and localhost still require allowed_media_domains.
  std::vector<std::string> allowed_private_media_domains;
  // Empty uses the portable bundle CA file when running from a bundle, then
  // the source-build libcurl defaults when no bundle is present.
  std::filesystem::path remote_tls_ca_bundle;
  std::uint64_t maximum_image_bytes = 64ULL * 1024ULL * 1024ULL;
  std::uint64_t maximum_video_bytes = 512ULL * 1024ULL * 1024ULL;
  std::uint32_t maximum_remote_redirects = 5;
  std::uint32_t maximum_remote_connect_milliseconds = 5000;
  std::uint32_t maximum_image_fetch_milliseconds = 10000;
  std::uint32_t maximum_video_fetch_milliseconds = 30000;
  std::uint32_t remote_low_speed_bytes_per_second = 1024;
  std::uint32_t remote_low_speed_seconds = 5;
  std::uint64_t maximum_decoded_image_pixels = 8192ULL * 8192ULL;
  std::uint32_t maximum_decoded_image_dimension = 8192;
  std::uint64_t maximum_decoded_video_pixels = 25165824ULL;
  std::uint32_t maximum_decoded_video_dimension = 8192;
  std::uint32_t maximum_video_source_frames = 18432;
  std::uint32_t maximum_video_sampled_frames = 768;
  double maximum_video_duration_seconds = 768.0;
  std::uint32_t maximum_video_decode_milliseconds = 30000;
  bool allow_data_uri = true;
};

struct NativeMediaPayload {
  NativeMediaKind kind = NativeMediaKind::kImage;
  NativeMediaTransport transport = NativeMediaTransport::kDataUri;
  std::string mime_type;
  std::string content_sha256;
  std::vector<unsigned char> bytes;
};

// Validates source syntax and allowlists without doing network I/O.
NativeMediaTransport validate_native_media_source(
    const NativeMediaPart& media, const NativeMediaPolicy& policy);

// Loads data:, file:, HTTP and HTTPS media under the configured byte, path,
// domain, redirect, address and timeout policy.
NativeMediaPayload load_native_media_payload(
    const NativeMediaPart& media, const NativeMediaPolicy& policy);

std::string_view native_media_kind_name(NativeMediaKind kind);
std::string_view native_media_transport_name(NativeMediaTransport transport);

}  // namespace aima
