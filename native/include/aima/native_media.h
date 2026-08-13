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
  std::vector<std::string> allowed_media_domains;
  std::uint64_t maximum_image_bytes = 64ULL * 1024ULL * 1024ULL;
  std::uint64_t maximum_video_bytes = 512ULL * 1024ULL * 1024ULL;
  std::uint64_t maximum_decoded_image_pixels = 8192ULL * 8192ULL;
  std::uint32_t maximum_decoded_image_dimension = 8192;
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

// Loads data: and file: media. Remote URLs are validated first and remain
// fail-closed until the bounded redirect-aware HTTP transport is attached.
NativeMediaPayload load_native_media_payload(
    const NativeMediaPart& media, const NativeMediaPolicy& policy);

std::string_view native_media_kind_name(NativeMediaKind kind);
std::string_view native_media_transport_name(NativeMediaTransport transport);

}  // namespace aima
