// SPDX-License-Identifier: Apache-2.0

#include "aima/native_media.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace {

void require(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "native_media_test: " << message << '\n';
    std::exit(1);
  }
}

template <typename Function>
void require_invalid(Function&& function, const char* message) {
  try {
    function();
  } catch (const std::invalid_argument&) {
    return;
  }
  require(false, message);
}

std::filesystem::path temporary_directory() {
  std::string pattern = "/tmp/aima-native-media-test.XXXXXX";
  char* value = ::mkdtemp(pattern.data());
  if (value == nullptr) throw std::runtime_error("mkdtemp failed");
  return std::filesystem::path(value);
}

void write_bytes(const std::filesystem::path& path,
                 std::initializer_list<unsigned char> bytes) {
  std::ofstream output(path, std::ios::binary);
  for (unsigned char byte : bytes) output.put(static_cast<char>(byte));
  if (!output) throw std::runtime_error("fixture write failed");
}

}  // namespace

int main() {
  const std::filesystem::path root = temporary_directory();
  const std::filesystem::path outside = temporary_directory();
  const auto cleanup = [&]() {
    std::error_code error;
    std::filesystem::remove_all(root, error);
    std::filesystem::remove_all(outside, error);
  };
  try {
    const std::filesystem::path png = root / "space image.png";
    write_bytes(png, {0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a,
                      0x00, 0x00, 0x00, 0x00});
    const std::filesystem::path outside_png = outside / "outside.png";
    write_bytes(outside_png, {0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a});
    std::filesystem::create_symlink(outside_png, root / "escape.png");

    aima::NativeMediaPolicy policy;
    policy.allowed_local_roots = {root};
    policy.allowed_media_domains = {"EXAMPLE.COM", "127.0.0.1"};

    aima::NativeMediaPart local;
    local.kind = aima::NativeMediaKind::kImage;
    local.source = "file://" + root.string() + "/space%20image.png";
    const auto local_payload = aima::load_native_media_payload(local, policy);
    require(local_payload.transport == aima::NativeMediaTransport::kLocalFile &&
                local_payload.mime_type == "image/png" &&
                local_payload.bytes.size() == 12 &&
                local_payload.content_sha256.size() == 64,
            "local PNG load contract changed");

    aima::NativeMediaPart data = local;
    data.source = "data:image/png;base64,iVBORw0KGgoAAAAA";
    const auto data_payload = aima::load_native_media_payload(data, policy);
    require(data_payload.transport == aima::NativeMediaTransport::kDataUri &&
                data_payload.bytes == local_payload.bytes &&
                data_payload.content_sha256 == local_payload.content_sha256,
            "data/local content identity changed");

    aima::NativeMediaPart escaped = local;
    escaped.source = "file://" + (root / "escape.png").string();
    require_invalid(
        [&]() { (void)aima::load_native_media_payload(escaped, policy); },
        "symlink escape left the local allowlist");

    aima::NativeMediaPart mismatch = data;
    mismatch.kind = aima::NativeMediaKind::kVideo;
    require_invalid(
        [&]() { (void)aima::load_native_media_payload(mismatch, policy); },
        "image bytes were admitted as video");

    aima::NativeMediaPolicy tiny = policy;
    tiny.maximum_image_bytes = 8;
    require_invalid(
        [&]() { (void)aima::load_native_media_payload(data, tiny); },
        "data URI exceeded its byte limit");

    aima::NativeMediaPart noncanonical = data;
    noncanonical.source = "data:image/png;base64,iVBORw0KGgp=";
    require_invalid(
        [&]() { (void)aima::load_native_media_payload(noncanonical, policy); },
        "non-canonical base64 padding bits were admitted");

    aima::NativeMediaPart remote = local;
    remote.source = "https://example.com/path/image.png";
    require(aima::validate_native_media_source(remote, policy) ==
                aima::NativeMediaTransport::kHttpsUrl,
            "allowlisted HTTPS source was rejected");
    remote.source = "http://127.0.0.1:8080/image.png";
    require(aima::validate_native_media_source(remote, policy) ==
                aima::NativeMediaTransport::kHttpUrl,
            "allowlisted HTTP port was rejected");
    remote.source = "https://user@example.com/image.png";
    require_invalid(
        [&]() { (void)aima::validate_native_media_source(remote, policy); },
        "remote URL credentials were admitted");
    remote.source = "https://not-example.com/image.png";
    require_invalid(
        [&]() { (void)aima::validate_native_media_source(remote, policy); },
        "unlisted remote domain was admitted");
    remote.source = "https://example.com:bogus/image.png";
    require_invalid(
        [&]() { (void)aima::validate_native_media_source(remote, policy); },
        "nonnumeric remote URL port was admitted");
    remote.source = "https://example.com:/image.png";
    require_invalid(
        [&]() { (void)aima::validate_native_media_source(remote, policy); },
        "empty remote URL port was admitted");
    remote.source = "https://example.com\\@not-example.com/image.png";
    require_invalid(
        [&]() { (void)aima::validate_native_media_source(remote, policy); },
        "backslash authority ambiguity was admitted");

    cleanup();
  } catch (...) {
    cleanup();
    throw;
  }
  std::cout << "native_media_test: PASS\n";
  return 0;
}
