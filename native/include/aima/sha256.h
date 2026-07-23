// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <filesystem>
#include <string>

namespace aima {

std::string sha256_file(const std::filesystem::path& path);
std::string sha256_bytes(const void* data, std::size_t size);

}  // namespace aima
