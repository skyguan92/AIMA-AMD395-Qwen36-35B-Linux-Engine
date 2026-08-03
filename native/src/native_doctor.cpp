// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/native_doctor.h"

#include "aima/native_chat_protocol.h"
#include "aima/sha256.h"
#include "model_layout.h"

#include <hip/hip_runtime.h>
#include <sys/utsname.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace aima {
namespace {

using Json = NativeOrderedJson;

constexpr std::uint64_t kExpectedVramBytes = 536870912ULL;
constexpr std::uint64_t kMinimumGttBytes = 103079215104ULL;

struct DoctorCheck {
  std::string id;
  bool required = true;
  bool passed = false;
  Json actual;
  Json expected;
  std::string remediation;
};

std::string read_text(const std::filesystem::path& path) {
  std::ifstream stream(path);
  if (!stream) return {};
  std::ostringstream output;
  output << stream.rdbuf();
  std::string value = output.str();
  while (!value.empty() &&
         (value.back() == '\n' || value.back() == '\r')) {
    value.pop_back();
  }
  return value;
}

std::optional<std::uint64_t> parse_uint64(const std::string& value) {
  if (value.empty()) return std::nullopt;
  try {
    std::size_t consumed = 0;
    const std::uint64_t parsed = std::stoull(value, &consumed);
    if (consumed != value.size()) return std::nullopt;
    return parsed;
  } catch (...) {
    return std::nullopt;
  }
}

bool cmdline_has(const std::string& cmdline, const std::string& value) {
  std::istringstream words(cmdline);
  std::string word;
  while (words >> word) {
    if (word == value) return true;
  }
  return false;
}

std::vector<std::filesystem::path> render_nodes() {
  std::vector<std::filesystem::path> result;
  std::error_code error;
  const std::filesystem::path directory = "/dev/dri";
  if (!std::filesystem::is_directory(directory, error)) return result;
  for (const auto& entry : std::filesystem::directory_iterator(directory, error)) {
    if (error) break;
    if (entry.path().filename().string().rfind("renderD", 0) == 0) {
      result.push_back(entry.path());
    }
  }
  std::sort(result.begin(), result.end());
  return result;
}

std::pair<std::optional<std::uint64_t>, std::optional<std::uint64_t>>
memory_totals() {
  std::optional<std::uint64_t> vram;
  std::optional<std::uint64_t> gtt;
  std::error_code error;
  const std::filesystem::path drm = "/sys/class/drm";
  if (!std::filesystem::is_directory(drm, error)) return {vram, gtt};
  for (const auto& entry : std::filesystem::directory_iterator(drm, error)) {
    if (error) break;
    const std::string name = entry.path().filename().string();
    if (name.rfind("card", 0) != 0 || name.find('-') != std::string::npos) {
      continue;
    }
    const auto candidate_vram =
        parse_uint64(read_text(entry.path() / "device/mem_info_vram_total"));
    const auto candidate_gtt =
        parse_uint64(read_text(entry.path() / "device/mem_info_gtt_total"));
    if (candidate_vram && (!vram || *candidate_vram > *vram)) {
      vram = candidate_vram;
    }
    if (candidate_gtt && (!gtt || *candidate_gtt > *gtt)) {
      gtt = candidate_gtt;
    }
  }
  return {vram, gtt};
}

Json check_json(const DoctorCheck& check) {
  return {{"id", check.id},
          {"required", check.required},
          {"passed", check.passed},
          {"actual", check.actual},
          {"expected", check.expected},
          {"remediation", check.remediation}};
}

void add_model_checks(const std::filesystem::path& model_dir,
                      std::vector<DoctorCheck>* checks) {
  const auto hash_check = [&](const char* id, const char* name,
                              const char* expected) {
    const std::filesystem::path path = model_dir / name;
    std::string actual = "missing";
    bool passed = false;
    if (std::filesystem::is_regular_file(path)) {
      try {
        actual = sha256_file(path);
        passed = actual == expected;
      } catch (const std::exception& error) {
        actual = std::string("error: ") + error.what();
      }
    }
    checks->push_back({id, true, passed, actual, expected,
                       std::string("install the qualified ") + name});
  };
  hash_check("model.config", "config.json", generated::kModelConfigSha256);
  hash_check("model.index", "model.safetensors.index.json",
             generated::kCheckpointIndexSha256);
  hash_check("model.tokenizer", "tokenizer.json", generated::kTokenizerSha256);
  hash_check("model.tokenizer_config", "tokenizer_config.json",
             generated::kTokenizerConfigSha256);

  std::size_t present = 0;
  std::uintmax_t total_bytes = 0;
  for (const char* name : generated::kShardNames) {
    std::error_code error;
    const std::filesystem::path path = model_dir / name;
    if (std::filesystem::is_regular_file(path, error) &&
        ::access(path.c_str(), R_OK) == 0) {
      const std::uintmax_t bytes = std::filesystem::file_size(path, error);
      if (!error) {
        ++present;
        total_bytes += bytes;
      }
    }
  }
  checks->push_back(
      {"model.shards", true,
       present == generated::kShardNames.size() &&
           total_bytes >= generated::kPayloadBytes,
       {{"readable", present}, {"bytes", total_bytes}},
       {{"readable", generated::kShardNames.size()},
        {"minimum_payload_bytes", generated::kPayloadBytes}},
       "install all 26 qualified Safetensors shards and grant read access"});
}

}  // namespace

int run_native_doctor(int argc, char** argv, const char* version,
                      const char* source_commit) {
  std::filesystem::path model_dir;
  bool check_model = false;
  int device = 0;
  for (int index = 2; index < argc; ++index) {
    const std::string argument = argv[index];
    auto next = [&](const char* name) {
      if (++index >= argc) {
        throw std::runtime_error(std::string(name) + " requires a value");
      }
      return std::string(argv[index]);
    };
    if (argument == "--model-dir") {
      model_dir = std::filesystem::absolute(next("--model-dir"));
      check_model = true;
    } else if (argument == "--device") {
      const std::string value = next("--device");
      std::size_t consumed = 0;
      device = std::stoi(value, &consumed);
      if (consumed != value.size() || device < 0) {
        throw std::runtime_error("--device must be a non-negative integer");
      }
    } else if (argument == "--json") {
      // JSON is the only stable output format; accept this for CLI symmetry.
    } else if (argument == "--help" || argument == "-h") {
      std::cout << "Usage: aima-engine doctor [--model-dir PATH] [--device INDEX] [--json]\n";
      return 0;
    } else {
      throw std::runtime_error("unknown doctor argument: " + argument);
    }
  }

  std::vector<DoctorCheck> checks;
  utsname host{};
  const bool uname_ok = ::uname(&host) == 0;
  checks.push_back(
      {"host.platform", true,
       uname_ok && std::string(host.sysname) == "Linux" &&
           std::string(host.machine) == "x86_64",
       uname_ok ? Json({{"sysname", host.sysname}, {"machine", host.machine}})
                : Json("unavailable"),
       {{"sysname", "Linux"}, {"machine", "x86_64"}},
       "use a Linux x86-64 host"});

  const bool kfd_exists = std::filesystem::exists("/dev/kfd");
  const bool kfd_access = ::access("/dev/kfd", R_OK | W_OK) == 0;
  checks.push_back(
      {"device.kfd", true, kfd_exists && kfd_access,
       {{"exists", kfd_exists}, {"read_write", kfd_access}},
       {{"exists", true}, {"read_write", true}},
       "load amdgpu/KFD and add the service user to the render group"});

  const auto renders = render_nodes();
  const auto readable_render = std::find_if(
      renders.begin(), renders.end(), [](const std::filesystem::path& path) {
        return ::access(path.c_str(), R_OK | W_OK) == 0;
      });
  checks.push_back(
      {"device.render", true, readable_render != renders.end(),
       {{"count", renders.size()},
        {"read_write", readable_render != renders.end()}},
       {{"minimum_count", 1}, {"read_write", true}},
       "load amdgpu DRM and add the service user to render/video groups"});

  int device_count = 0;
  hipError_t hip_status = hipGetDeviceCount(&device_count);
  checks.push_back(
      {"hip.devices", true, hip_status == hipSuccess && device_count > device,
       {{"count", device_count},
        {"status", hip_status == hipSuccess ? "success"
                                            : hipGetErrorString(hip_status)}},
       {{"minimum_count", device + 1}},
       "verify the bundled HIP userspace and amdgpu device permissions"});
  std::string architecture = "unavailable";
  std::string device_name = "unavailable";
  bool gfx1151 = false;
  if (hip_status == hipSuccess && device_count > device) {
    hipDeviceProp_t properties{};
    hip_status = hipGetDeviceProperties(&properties, device);
    if (hip_status == hipSuccess) {
      architecture = properties.gcnArchName;
      device_name = properties.name;
      gfx1151 = architecture.rfind("gfx1151", 0) == 0;
    }
  }
  checks.push_back(
      {"gpu.architecture", true, gfx1151,
       {{"device", device_name}, {"architecture", architecture}},
       {{"architecture_prefix", "gfx1151"}},
       "run the portable engine on Radeon 8060S / gfx1151"});

  const std::string cmdline = read_text("/proc/cmdline");
  const bool pages_ok = cmdline_has(cmdline, "ttm.pages_limit=25165824");
  const bool gtt_arg_ok = cmdline_has(cmdline, "amdgpu.gttsize=98304");
  checks.push_back(
      {"memory.kernel_parameters", true, pages_ok && gtt_arg_ok,
       {{"ttm_pages_limit", pages_ok}, {"amdgpu_gttsize", gtt_arg_ok}},
       {{"ttm.pages_limit", 25165824}, {"amdgpu.gttsize", 98304}},
       "apply the documented GRUB parameters and reboot"});

  const auto [vram, gtt] = memory_totals();
  checks.push_back(
      {"memory.vram", true, vram && *vram == kExpectedVramBytes,
       vram ? Json(*vram) : Json(nullptr), kExpectedVramBytes,
       "set the BIOS UMA frame buffer to 512 MiB"});
  checks.push_back(
      {"memory.gtt", true, gtt && *gtt >= kMinimumGttBytes,
       gtt ? Json(*gtt) : Json(nullptr),
       {{"minimum_bytes", kMinimumGttBytes}},
       "configure the documented 96 GiB AMDGPU GTT pool and reboot"});

  std::error_code error;
  const std::filesystem::path executable =
      std::filesystem::read_symlink("/proc/self/exe", error);
  const std::filesystem::path bundle_root =
      error ? std::filesystem::path() : executable.parent_path().parent_path();
  const bool bundle_detected = !bundle_root.empty() &&
                               std::filesystem::is_regular_file(
                                   bundle_root / "manifest.json");
  bool bundle_complete = false;
  if (bundle_detected) {
    bundle_complete = true;
    for (const auto& relative : {
             "bin/aima-engine", "libexec/aima-engine.real",
             "lib/libamdhip64.so.7", "share/aima/product-contract.json",
             "share/aima/qualification.json"}) {
      bundle_complete = bundle_complete &&
                        std::filesystem::exists(bundle_root / relative);
    }
  }
  checks.push_back(
      {"runtime.bundle", bundle_detected, !bundle_detected || bundle_complete,
       {{"detected", bundle_detected}, {"complete", bundle_complete}},
       bundle_detected ? Json({{"complete", true}}) : Json("source-build"),
       "re-extract the checksummed portable release archive"});

  if (check_model) add_model_checks(model_dir, &checks);

  const bool qualified = std::all_of(
      checks.begin(), checks.end(),
      [](const DoctorCheck& check) { return !check.required || check.passed; });
  Json serialized_checks = Json::array();
  for (const DoctorCheck& check : checks) {
    serialized_checks.push_back(check_json(check));
  }
  std::cout
      << Json({{"schema", "aima-amd395-qwen36/native-doctor/v1"},
               {"complete", true},
               {"qualified", qualified},
               {"version", version},
               {"source_commit", source_commit},
               {"model_checked", check_model},
               {"checks", std::move(serialized_checks)}})
             .dump(2)
      << '\n';
  return qualified ? 0 : 3;
}

}  // namespace aima
