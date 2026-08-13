// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_mrope.h"
#include "aima/sha256.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using json = nlohmann::json;

json read_json(const std::filesystem::path& path) {
  std::ifstream stream(path);
  if (!stream) {
    throw std::runtime_error("JSON file is unavailable: " + path.string());
  }
  json value;
  stream >> value;
  if (!value.is_object()) {
    throw std::runtime_error("JSON root is not an object: " + path.string());
  }
  return value;
}

std::filesystem::path checked_oracle_path(
    const std::filesystem::path& root, std::string_view relative_text) {
  const std::filesystem::path relative(relative_text);
  if (relative.empty() || relative.is_absolute()) {
    throw std::runtime_error("oracle tensor path is not relative");
  }
  for (const auto& component : relative) {
    if (component == "..") {
      throw std::runtime_error("oracle tensor path escapes its root");
    }
  }
  return root / relative;
}

std::vector<unsigned char> read_tensor_record(
    const std::filesystem::path& root, const json& record) {
  const std::size_t expected_bytes = record.at("bytes").get<std::size_t>();
  const std::filesystem::path path = checked_oracle_path(
      root, record.at("path").get<std::string>());
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream || stream.tellg() < 0 ||
      static_cast<std::size_t>(stream.tellg()) != expected_bytes) {
    throw std::runtime_error("oracle tensor size mismatch: " + path.string());
  }
  std::vector<unsigned char> bytes(expected_bytes);
  stream.seekg(0);
  if (expected_bytes != 0 &&
      !stream.read(reinterpret_cast<char*>(bytes.data()),
                   static_cast<std::streamsize>(bytes.size()))) {
    throw std::runtime_error("oracle tensor read failed: " + path.string());
  }
  if (aima::sha256_bytes(bytes.data(), bytes.size()) !=
      record.at("sha256").get<std::string>()) {
    throw std::runtime_error("oracle tensor SHA-256 mismatch: " +
                             path.string());
  }
  return bytes;
}

std::int64_t decode_i64_le(const unsigned char* bytes) {
  std::uint64_t value = 0;
  for (std::size_t index = 0; index < 8; ++index) {
    value |= static_cast<std::uint64_t>(bytes[index]) << (8 * index);
  }
  std::int64_t result = 0;
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

std::vector<std::int64_t> decode_i64_tensor(
    const std::vector<unsigned char>& bytes) {
  if (bytes.size() % sizeof(std::int64_t) != 0) {
    throw std::runtime_error("int64 oracle tensor byte count is invalid");
  }
  std::vector<std::int64_t> result;
  result.reserve(bytes.size() / sizeof(std::int64_t));
  for (std::size_t offset = 0; offset < bytes.size(); offset += 8) {
    result.push_back(decode_i64_le(bytes.data() + offset));
  }
  return result;
}

std::vector<unsigned char> encode_i64_tensor(
    const std::vector<std::int64_t>& values) {
  std::vector<unsigned char> result(values.size() * sizeof(std::int64_t));
  for (std::size_t index = 0; index < values.size(); ++index) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &values[index], sizeof(bits));
    for (std::size_t byte = 0; byte < 8; ++byte) {
      result[index * 8 + byte] =
          static_cast<unsigned char>((bits >> (8 * byte)) & 0xffU);
    }
  }
  return result;
}

std::vector<std::uint32_t> prompt_token_ids(const json& case_record) {
  const json& values = case_record.at("processor").at("prompt_token_ids");
  if (!values.is_array() || values.empty()) {
    throw std::runtime_error("oracle prompt token ids are empty");
  }
  std::vector<std::uint32_t> result;
  result.reserve(values.size());
  for (const json& value : values) {
    const std::uint64_t token = value.get<std::uint64_t>();
    if (token > std::numeric_limits<std::uint32_t>::max()) {
      throw std::runtime_error("oracle prompt token id exceeds uint32");
    }
    result.push_back(static_cast<std::uint32_t>(token));
  }
  return result;
}

std::vector<aima::NativeVlGrid> read_grids(
    const std::filesystem::path& oracle_root, const json& case_record,
    const char* tensor_name) {
  const json& record =
      case_record.at("processor").at("tensors").at(tensor_name);
  const json& shape = record.at("shape");
  if (record.value("dtype", "") != "torch.int64" ||
      !shape.is_array() || shape.size() != 2 ||
      shape.at(1).get<std::size_t>() != 3) {
    throw std::runtime_error("oracle grid tensor shape/dtype is invalid");
  }
  const std::size_t count = shape.at(0).get<std::size_t>();
  const std::vector<std::int64_t> values =
      decode_i64_tensor(read_tensor_record(oracle_root, record));
  if (values.size() != count * 3) {
    throw std::runtime_error("oracle grid tensor element count is invalid");
  }
  std::vector<aima::NativeVlGrid> result;
  result.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    const std::int64_t temporal = values[index * 3];
    const std::int64_t height = values[index * 3 + 1];
    const std::int64_t width = values[index * 3 + 2];
    if (temporal <= 0 || height <= 0 || width <= 0) {
      throw std::runtime_error("oracle grid tensor has a non-positive value");
    }
    result.push_back(aima::NativeVlGrid{
        static_cast<std::size_t>(temporal),
        static_cast<std::size_t>(height),
        static_cast<std::size_t>(width)});
  }
  return result;
}

std::vector<aima::NativeMropeMedia> media_records(
    const std::filesystem::path& oracle_root, const json& case_record) {
  const json& processor = case_record.at("processor");
  const json& placeholders = processor.at("placeholders");
  const json& tensors = processor.at("tensors");
  std::vector<aima::NativeMropeMedia> result;
  for (const auto& descriptor :
       {std::pair<const char*, aima::NativeMediaKind>{
            "image", aima::NativeMediaKind::kImage},
        std::pair<const char*, aima::NativeMediaKind>{
            "video", aima::NativeMediaKind::kVideo}}) {
    if (!placeholders.contains(descriptor.first)) continue;
    const std::string tensor_name =
        std::string(descriptor.first) + "_grid_thw";
    if (!tensors.contains(tensor_name)) {
      throw std::runtime_error("oracle media grid tensor is missing");
    }
    const std::vector<aima::NativeVlGrid> grids =
        read_grids(oracle_root, case_record, tensor_name.c_str());
    const json& spans = placeholders.at(descriptor.first);
    if (!spans.is_array() || spans.size() != grids.size()) {
      throw std::runtime_error("oracle media span/grid count differs");
    }
    for (std::size_t index = 0; index < grids.size(); ++index) {
      const json& span = spans.at(index);
      result.push_back(aima::NativeMropeMedia{
          descriptor.second, span.at("offset").get<std::size_t>(),
          span.at("length").get<std::size_t>(), grids[index]});
    }
  }
  return result;
}

json qualify_case(const std::filesystem::path& oracle_root,
                  const json& case_record) {
  const std::vector<std::uint32_t> tokens = prompt_token_ids(case_record);
  const std::vector<aima::NativeMropeMedia> media =
      media_records(oracle_root, case_record);
  const aima::NativeMropePlan plan =
      aima::build_native_mrope_plan(tokens, media);
  const json& expected_record =
      case_record.at("boundaries").at("mrope_positions");
  if (expected_record.value("dtype", "") != "torch.int64" ||
      expected_record.at("shape") !=
          json::array({3, plan.prompt_token_count()})) {
    throw std::runtime_error("M-RoPE oracle shape/dtype is invalid");
  }
  const std::vector<unsigned char> expected_bytes =
      read_tensor_record(oracle_root, expected_record);
  const std::vector<std::int64_t> expected =
      decode_i64_tensor(expected_bytes);
  const std::vector<std::int64_t>& actual = plan.positions();
  if (actual.size() != expected.size()) {
    throw std::runtime_error("M-RoPE actual/oracle element count differs");
  }
  std::size_t exact_elements = 0;
  std::size_t first_mismatch = std::numeric_limits<std::size_t>::max();
  for (std::size_t index = 0; index < expected.size(); ++index) {
    if (actual[index] == expected[index]) {
      ++exact_elements;
    } else if (first_mismatch == std::numeric_limits<std::size_t>::max()) {
      first_mismatch = index;
    }
  }
  const std::vector<unsigned char> actual_bytes = encode_i64_tensor(actual);
  const std::int64_t expected_delta =
      expected_record.at("position_delta").get<std::int64_t>();
  const bool exact = exact_elements == expected.size() &&
                     plan.position_delta() == expected_delta;
  return json{
      {"schema", "aima-amd395-qwen36/native-vl-mrope-oracle/v1"},
      {"complete", exact},
      {"case_id", case_record.at("case_id")},
      {"prompt_tokens", plan.prompt_token_count()},
      {"media_count", media.size()},
      {"elements", expected.size()},
      {"exact_elements", exact_elements},
      {"first_mismatch_index",
       first_mismatch == std::numeric_limits<std::size_t>::max()
           ? -1LL
           : static_cast<long long>(first_mismatch)},
      {"expected_position_delta", expected_delta},
      {"actual_position_delta", plan.position_delta()},
      {"maximum_position", plan.maximum_position()},
      {"expected_sha256", expected_record.at("sha256")},
      {"actual_sha256",
       aima::sha256_bytes(actual_bytes.data(), actual_bytes.size())},
  };
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr <<
        "usage: native-vl-mrope-probe ORACLE_MANIFEST ORACLE_ROOT\n";
    return 2;
  }
  try {
    const std::filesystem::path manifest_path =
        std::filesystem::absolute(argv[1]);
    const std::filesystem::path oracle_root =
        std::filesystem::absolute(argv[2]);
    const json manifest = read_json(manifest_path);
    if (manifest.value("schema", "") !=
            "aima-amd395-qwen36/vl-oracle-manifest/v1" ||
        !manifest.value("complete", false) ||
        !manifest.contains("cases") || !manifest.at("cases").is_array()) {
      throw std::runtime_error("VL oracle manifest is incomplete");
    }
    json cases = json::array();
    bool complete = true;
    std::size_t total_elements = 0;
    std::size_t total_exact_elements = 0;
    for (const json& case_record : manifest.at("cases")) {
      json result = qualify_case(oracle_root, case_record);
      complete = complete && result.at("complete").get<bool>();
      total_elements += result.at("elements").get<std::size_t>();
      total_exact_elements +=
          result.at("exact_elements").get<std::size_t>();
      cases.push_back(std::move(result));
    }
    const json result = {
        {"schema",
         "aima-amd395-qwen36/native-vl-mrope-qualification-run/v1"},
        {"complete", complete},
        {"case_count", cases.size()},
        {"oracle_manifest_sha256", aima::sha256_file(manifest_path)},
        {"total_elements", total_elements},
        {"total_exact_elements", total_exact_elements},
        {"all_integer_exact", total_elements == total_exact_elements},
        {"cases", std::move(cases)},
    };
    std::cout << result.dump() << '\n';
    return complete ? 0 : 3;
  } catch (const std::exception& error) {
    std::cerr << "native VL M-RoPE probe: " << error.what() << '\n';
    return 1;
  }
}
