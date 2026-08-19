// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#include "aima/aot_registry.h"
#include "aima/bf16_gemm.h"
#include "aima/bf16_wvsplitk.h"
#include "aima/decode_schedule.h"
#include "aima/native_derived_weights.h"
#include "aima/native_doctor.h"
#include "aima/native_chat_protocol.h"
#include "aima/native_decode_bindings.h"
#include "aima/native_decode_executor.h"
#include "aima/native_decode_invocation.h"
#include "aima/native_decode_runner.h"
#include "aima/native_decode_workspace.h"
#include "aima/native_full_prefill.h"
#include "aima/native_http_server.h"
#include "aima/native_layer_oracle.h"
#include "aima/native_linear_prefill.h"
#include "aima/native_lm_head.h"
#include "aima/native_moe_prefill.h"
#include "aima/native_pointwise.h"
#include "aima/native_prefill_invocation.h"
#include "aima/native_prefill_gemm_plans.h"
#include "aima/native_prefill_workspace.h"
#include "aima/native_resident_engine.h"
#include "aima/native_weight_store.h"
#include "aima/native_tokenizer.h"
#include "aima/prefill_schedule.h"
#include "aima/sha256.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr const char* kVersion = "1.5.1-native";
#ifndef AIMA_SOURCE_COMMIT
#define AIMA_SOURCE_COMMIT "unknown"
#endif
constexpr const char* kSourceCommit = AIMA_SOURCE_COMMIT;

void configure_bundled_rocm() {
  std::error_code error;
  const std::filesystem::path executable =
      std::filesystem::read_symlink("/proc/self/exe", error);
  if (error || executable.empty()) return;
  const std::filesystem::path root = executable.parent_path().parent_path();
  const std::filesystem::path hip = root / "lib/libamdhip64.so.7";
  const std::filesystem::path device_libs = root / "amdgcn/bitcode";
  const std::filesystem::path hipblaslt_library = root / "lib/hipblaslt/library";
  if (!std::filesystem::is_regular_file(hip) ||
      !std::filesystem::is_directory(device_libs)) {
    return;
  }
  const std::string root_value = root.string();
  const std::string device_lib_value = device_libs.string();
  if (setenv("ROCM_PATH", root_value.c_str(), 1) != 0 ||
      setenv("HIP_PATH", root_value.c_str(), 1) != 0 ||
      setenv("HIP_DEVICE_LIB_PATH", device_lib_value.c_str(), 1) != 0) {
    throw std::runtime_error("failed to configure bundled ROCm paths");
  }
  if (std::filesystem::is_directory(hipblaslt_library)) {
    const std::string hipblaslt_value = hipblaslt_library.string();
    if (setenv("HIPBLASLT_TENSILE_LIBPATH", hipblaslt_value.c_str(), 1) != 0) {
      throw std::runtime_error("failed to configure bundled hipBLASLt path");
    }
  }
}

std::string json_escape(const std::string& value) {
  std::ostringstream output;
  for (const unsigned char ch : value) {
    switch (ch) {
      case '\\': output << "\\\\"; break;
      case '"': output << "\\\""; break;
      case '\n': output << "\\n"; break;
      case '\r': output << "\\r"; break;
      case '\t': output << "\\t"; break;
      default:
        if (ch < 0x20) {
          output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                 << static_cast<unsigned>(ch) << std::dec;
        } else {
          output << ch;
        }
    }
  }
  return output.str();
}

std::string json_number(double value) {
  if (!std::isfinite(value)) return "null";
  std::ostringstream output;
  output << std::setprecision(17) << value;
  return output.str();
}

std::size_t parse_size(const std::string& value, const char* label) {
  std::size_t consumed = 0;
  const unsigned long long parsed = std::stoull(value, &consumed, 0);
  if (consumed != value.size() || parsed == 0) {
    throw std::runtime_error(std::string(label) + " must be a positive integer");
  }
  return static_cast<std::size_t>(parsed);
}

int parse_int(const std::string& value, const char* label) {
  std::size_t consumed = 0;
  const long parsed = std::stol(value, &consumed, 0);
  if (consumed != value.size() || parsed < 0) {
    throw std::runtime_error(std::string(label) + " must be a non-negative integer");
  }
  return static_cast<int>(parsed);
}

void usage(std::ostream& output) {
  output
      << "Usage:\n"
      << "  aima-engine-native --version\n"
      << "  aima-engine-native --build-info\n"
      << "  aima-engine-native doctor [--model-dir PATH] [--device INDEX] [--json]\n"
      << "  aima-engine-native aot-closure-probe\n"
      << "  aima-engine-native decode-schedule-probe\n"
      << "  aima-engine-native prefill-schedule-probe\n"
      << "  aima-engine-native lm-head-probe --model-dir PATH [options]\n"
      << "  aima-engine-native decode-bindings-probe --model-dir PATH [options]\n"
      << "  aima-engine-native linear-prefill-oracle-probe --model-dir PATH --oracle-dir PATH [options]\n"
      << "  aima-engine-native moe-prefill-oracle-probe --model-dir PATH --oracle-dir PATH [--boundary-oracle-dir PATH] [options]\n"
      << "  aima-engine-native prefill-layer0-oracle-probe --model-dir PATH (--oracle-dir PATH | --attention-oracle-dir PATH --moe-oracle-dir PATH) [--boundary-oracle-dir PATH] [options]\n"
      << "  aima-engine-native prefill-linear-layer-oracle-probe --model-dir PATH (--oracle-dir PATH | --attention-oracle-dir PATH --moe-oracle-dir PATH) --layer INDEX [--boundary-oracle-dir PATH] [options]\n"
      << "  aima-engine-native prefill-linear-prefix-oracle-probe --model-dir PATH --layer0-attention-oracle-dir PATH --layer0-moe-oracle-dir PATH --later-oracle-dir PATH [--full-layer-oracle-dir PATH --ck-provider PATH --through-layer 3] [--chain-output-oracle-dir PATH] [options]\n"
      << "  aima-engine-native prefill-all-layers-oracle-probe --model-dir PATH --ck-provider PATH (--chain-output-oracle-dir PATH | --execution-only) [--uniform-input-token-id ID | --input-token-id-cycle ID,ID,...] [--entry-input-oracle-dir PATH] [--prefill-state-oracle-dir PATH] [--decode-oracle-dir PATH] [--decode-logits-oracle-dir PATH] [--decode-logits-oracle-label-prefix LABEL] [options]\n"
      << "  aima-engine-native resident-session-probe --model-dir PATH (--uniform-input-token-id ID | --input-token-id-cycle ID,ID,...) [--context-tokens 1024|2048|4096|8192|16384|32768] [--prompt-tokens N] [--fmha-provider PATH] [--secondary-fmha-provider PATH --secondary-fmha-layers 3,7,...] [--max-new-tokens N | --max-new-tokens-sequence N,N,...] [--requests N] [--cached-suffix-token-ids ID,ID,...] [--expected-token-ids ID,ID,...] [--reference-logits PATH] [--layer-tail-oracle-dir PATH [--layer-tail-oracle-index 0..39]] [options]\n"
      << "  aima-engine-native vl-generation-logits-probe --model-dir PATH --cases-json PATH [--fmha-provider PATH] [--vision-attention-image PATH] [options]\n"
      << "  aima-engine-native serve --model-dir PATH [--context-tokens 1024|2048|4096|8192|16384|32768] [--fmha-provider PATH] [--secondary-fmha-provider PATH --secondary-fmha-layers 3,7,...] [--host 127.0.0.1] [--port 8000] [--cache-capacity N] [options]\n"
      << "  aima-engine-native prefill-full-layer-oracle-probe --model-dir PATH --oracle-dir PATH --ck-provider PATH [--layer 3] [options]\n"
      << "  aima-engine-native full-attention-core-oracle-probe --oracle-dir PATH [--layer INDEX] [--cache-end N]\n"
      << "  aima-engine-native full-layer-oracle-probe --model-dir PATH --oracle-dir PATH [--layer INDEX|--all-full-layers] [options]\n"
      << "  aima-engine-native decode-oracle-probe --model-dir PATH --oracle-dir PATH [options]\n"
      << "  aima-engine-native linear-layer-oracle-probe --model-dir PATH --oracle-dir PATH [--all-linear-layers] [options]\n"
      << "  aima-engine-native bf16-gemm-probe\n"
      << "  aima-engine-native bf16-wvsplitk-probe\n"
      << "  aima-engine-native derived-weights-probe --model-dir PATH [options]\n"
      << "  aima-engine-native tokenizer-probe --model-dir PATH --text TEXT\n"
      << "  aima-engine-native chat-template-probe --model-dir PATH (--user TEXT | --request-json JSON) [options]\n"
      << "  aima-engine-native weights-probe --model-dir PATH [options]\n\n"
      << "Options:\n"
      << "  --report PATH       Native loader report (default: native-weight-load.json)\n"
      << "  --device INDEX      HIP device index (default: 0)\n"
      << "  --workers COUNT     O_DIRECT reader workers (default: 1)\n"
      << "  --chunk-bytes N     Per-buffer bytes (default: 134217728)\n"
      << "  --compact           Omit long per-token arrays from oracle JSON\n";
}

double elapsed_ms(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - start)
      .count();
}

void write_token_ids(const std::vector<std::uint32_t>& token_ids) {
  std::cout << '[';
  for (std::size_t index = 0; index < token_ids.size(); ++index) {
    if (index != 0) std::cout << ',';
    std::cout << token_ids[index];
  }
  std::cout << ']';
}

int run_tokenizer_probe(int argc, char** argv, bool chat) {
  std::filesystem::path model_dir;
  std::string text;
  std::string system;
  std::string user;
  std::string request_json;
  bool have_model_dir = false;
  bool have_text = false;
  bool have_user = false;
  bool have_request_json = false;
  bool disable_thinking = false;
  for (int index = 2; index < argc; ++index) {
    const std::string argument = argv[index];
    auto next = [&](const char* name) -> std::string {
      if (++index >= argc) throw std::runtime_error(std::string(name) + " requires a value");
      return argv[index];
    };
    if (argument == "--model-dir") {
      model_dir = std::filesystem::absolute(next("--model-dir"));
      have_model_dir = true;
    } else if (argument == "--text" && !chat) {
      text = next("--text");
      have_text = true;
    } else if (argument == "--system" && chat) {
      system = next("--system");
    } else if (argument == "--user" && chat) {
      user = next("--user");
      have_user = true;
    } else if (argument == "--request-json" && chat) {
      request_json = next("--request-json");
      have_request_json = true;
    } else if (argument == "--disable-thinking" && chat) {
      disable_thinking = true;
    } else if (argument == "--help" || argument == "-h") {
      usage(std::cout);
      return 0;
    } else {
      throw std::runtime_error("unknown tokenizer argument: " + argument);
    }
  }
  if (!have_model_dir) throw std::runtime_error("--model-dir is required");
  if (!chat && !have_text) throw std::runtime_error("--text is required");
  if (chat && have_user == have_request_json) {
    throw std::runtime_error(
        "exactly one of --user or --request-json is required");
  }

  aima::NativeTokenizer tokenizer;
  const auto load_started = std::chrono::steady_clock::now();
  tokenizer.load(model_dir);
  const double load_ms = elapsed_ms(load_started);
  std::string rendered;
  const auto encode_started = std::chrono::steady_clock::now();
  std::vector<std::uint32_t> token_ids;
  if (chat) {
    if (have_request_json) {
      const aima::NativeOrderedJson request =
          aima::NativeOrderedJson::parse(request_json);
      aima::NativePreparedChat prepared =
          aima::prepare_native_chat(request);
      rendered = tokenizer.render_chat_prompt(
          prepared.messages, prepared.prompt_tools, disable_thinking);
    } else {
      rendered = tokenizer.render_chat_prompt(system, user, disable_thinking);
    }
    token_ids = tokenizer.encode(rendered);
  } else {
    rendered = text;
    token_ids = tokenizer.encode(text);
  }
  const double encode_ms = elapsed_ms(encode_started);
  const std::string decoded = tokenizer.decode(token_ids);
  std::cout << std::setprecision(17)
            << "{\n"
            << "  \"schema\": \"aima-amd395-qwen36/native-tokenizer-probe/v1\",\n"
            << "  \"complete\": true,\n"
            << "  \"mode\": \"" << (chat ? "chat_template" : "text") << "\",\n"
            << "  \"runtime_python\": false,\n"
            << "  \"tokenizer_size\": " << tokenizer.size() << ",\n"
            << "  \"eos_token_id\": " << tokenizer.eos_token_id() << ",\n"
            << "  \"pad_token_id\": " << tokenizer.pad_token_id() << ",\n"
            << "  \"load_ms\": " << load_ms << ",\n"
            << "  \"encode_ms\": " << encode_ms << ",\n"
            << "  \"prompt_text\": \"" << json_escape(rendered) << "\",\n"
            << "  \"decoded\": \"" << json_escape(decoded) << "\",\n"
            << "  \"token_ids\": ";
  write_token_ids(token_ids);
  std::cout << "\n}\n";
  return 0;
}

int run_weights_probe(int argc, char** argv) {
  aima::NativeWeightLoadOptions options;
  options.native_report = std::filesystem::absolute("native-weight-load.json");
  bool have_model_dir = false;
  for (int index = 2; index < argc; ++index) {
    const std::string argument = argv[index];
    auto next = [&](const char* name) -> std::string {
      if (++index >= argc) throw std::runtime_error(std::string(name) + " requires a value");
      return argv[index];
    };
    if (argument == "--model-dir") {
      options.model_dir = std::filesystem::absolute(next("--model-dir"));
      have_model_dir = true;
    } else if (argument == "--report") {
      options.native_report = std::filesystem::absolute(next("--report"));
    } else if (argument == "--device") {
      options.device = parse_int(next("--device"), "--device");
    } else if (argument == "--workers") {
      options.worker_count = parse_size(next("--workers"), "--workers");
    } else if (argument == "--chunk-bytes") {
      options.chunk_bytes = parse_size(next("--chunk-bytes"), "--chunk-bytes");
    } else if (argument == "--help" || argument == "-h") {
      usage(std::cout);
      return 0;
    } else {
      throw std::runtime_error("unknown argument: " + argument);
    }
  }
  if (!have_model_dir) throw std::runtime_error("--model-dir is required");

  aima::NativeWeightStore weights;
  const aima::NativeWeightLoadMetrics metrics = weights.load(options);
  const auto* lm_head = weights.find("lm_head.weight");
  const auto* layer0 = weights.find("model.language_model.layers.0.input_layernorm.weight");
  if (lm_head == nullptr || layer0 == nullptr) {
    throw std::runtime_error("native tensor registry is missing required model boundaries");
  }
  std::cout << std::setprecision(17)
            << "{\n"
            << "  \"schema\": \"aima-amd395-qwen36/native-weight-probe/v1\",\n"
            << "  \"complete\": true,\n"
            << "  \"version\": \"" << kVersion << "\",\n"
            << "  \"runtime_python\": false,\n"
            << "  \"runtime_torch\": false,\n"
            << "  \"runtime_vllm\": false,\n"
            << "  \"runtime_triton\": false,\n"
            << "  \"device\": \"" << json_escape(metrics.device_name) << "\",\n"
            << "  \"gpu_arch\": \"" << json_escape(metrics.gpu_arch) << "\",\n"
            << "  \"model_dir\": \"" << json_escape(options.model_dir.string()) << "\",\n"
            << "  \"native_report\": \"" << json_escape(options.native_report.string()) << "\",\n"
            << "  \"checkpoint_index_sha256\": \"" << metrics.checkpoint_index_sha256 << "\",\n"
            << "  \"model_config_sha256\": \"" << metrics.model_config_sha256 << "\",\n"
            << "  \"tensor_count\": " << metrics.tensor_count << ",\n"
            << "  \"shard_count\": " << metrics.shard_count << ",\n"
            << "  \"payload_bytes\": " << metrics.payload_bytes << ",\n"
            << "  \"free_bytes_before\": " << metrics.free_bytes_before << ",\n"
            << "  \"free_bytes_after\": " << metrics.free_bytes_after << ",\n"
            << "  \"total_device_bytes\": " << metrics.total_device_bytes << ",\n"
            << "  \"allocation_ms\": " << metrics.allocation_ms << ",\n"
            << "  \"ingest_ms\": " << metrics.ingest_ms << ",\n"
            << "  \"load_wall_ms\": " << metrics.load_wall_ms << ",\n"
            << "  \"tensor_registry_checks\": {\"lm_head\": true, \"layer0_input_norm\": true}\n"
            << "}\n";
  return 0;
}

int run_aot_closure_probe() {
  const aima::AotClosureProbeResult result = aima::probe_embedded_aot_closure();
  std::cout << std::setprecision(17)
            << "{\n"
            << "  \"schema\": \"aima-amd395-qwen36/native-aot-closure-probe/v1\",\n"
            << "  \"complete\": true,\n"
            << "  \"runtime_python\": false,\n"
            << "  \"runtime_torch\": false,\n"
            << "  \"runtime_vllm\": false,\n"
            << "  \"runtime_triton\": false,\n"
            << "  \"gpu_arch\": \"" << json_escape(result.gpu_arch) << "\",\n"
            << "  \"image_count\": " << result.image_count << ",\n"
            << "  \"loaded_count\": " << result.loaded_count << ",\n"
            << "  \"image_bytes\": " << result.image_bytes << ",\n"
            << "  \"exact_bf16_elements\": " << result.exact_bf16_elements << ",\n"
            << "  \"expected_bf16_elements\": "
            << result.expected_bf16_elements << ",\n"
            << "  \"exact_probe_ms\": " << result.exact_probe_ms << "\n"
            << "}\n";
  return 0;
}

int run_schedule_probe(bool prefill) {
  const aima::DecodeScheduleProbeResult result =
      prefill ? aima::probe_native_prefill_schedule()
              : aima::probe_native_decode_schedule();
  const char* phase = prefill ? "prefill" : "decode";
  std::cout << "{\n"
            << "  \"schema\": \"aima-amd395-qwen36/native-" << phase
            << "-schedule-probe/v1\",\n"
            << "  \"complete\": true,\n"
            << "  \"runtime_python\": false,\n"
            << "  \"runtime_torch\": false,\n"
            << "  \"runtime_vllm\": false,\n"
            << "  \"runtime_triton\": false,\n"
            << "  \"schedule_sha256\": \""
            << result.schedule_sha256 << "\",\n"
            << "  \"launch_count\": " << result.launch_count << ",\n"
            << "  \"layer_launch_count\": " << result.layer_launch_count << ",\n"
            << "  \"final_logit_launch_count\": "
            << result.final_logit_launch_count << ",\n"
            << "  \"tensor_argument_count\": "
            << result.tensor_argument_count << ",\n"
            << "  \"scalar_argument_count\": "
            << result.scalar_argument_count << ",\n"
            << "  \"model_binding_arguments\": "
            << result.model_binding_arguments << ",\n"
            << "  \"resident_binding_arguments\": "
            << result.resident_binding_arguments << ",\n"
            << "  \"transient_binding_arguments\": "
            << result.transient_binding_arguments << ",\n"
            << "  \"unique_kernel_count\": "
            << result.unique_kernel_count << ",\n"
            << "  \"embedded_kernel_matches\": "
            << result.embedded_kernel_matches << "\n"
            << "}\n";
  return 0;
}

int run_bf16_gemm_probe() {
  const aima::Bf16GemmProbeResult result = aima::probe_bf16_gemm();
  std::cout << std::setprecision(17)
            << "{\n"
            << "  \"schema\": \"aima-amd395-qwen36/native-bf16-gemm-probe/v1\",\n"
            << "  \"complete\": true,\n"
            << "  \"runtime_python\": false,\n"
            << "  \"runtime_torch\": false,\n"
            << "  \"runtime_vllm\": false,\n"
            << "  \"runtime_triton\": false,\n"
            << "  \"provider\": \"hipBLASLt\",\n"
            << "  \"gpu_arch\": \"" << json_escape(result.gpu_arch) << "\",\n"
            << "  \"m\": " << result.m << ",\n"
            << "  \"n\": " << result.n << ",\n"
            << "  \"k\": " << result.k << ",\n"
            << "  \"library_version\": " << result.library_version << ",\n"
            << "  \"heuristic_count\": " << result.heuristic_count << ",\n"
            << "  \"workspace_bytes\": " << result.workspace_bytes << ",\n"
            << "  \"exact_bf16_elements\": " << result.exact_bf16_elements << ",\n"
            << "  \"expected_bf16_elements\": "
            << result.expected_bf16_elements << ",\n"
            << "  \"measured_ms\": [";
  for (std::size_t index = 0; index < result.measured_ms.size(); ++index) {
    if (index != 0) std::cout << ',';
    std::cout << result.measured_ms[index];
  }
  std::cout << "],\n"
            << "  \"median_ms\": " << result.median_ms << ",\n"
            << "  \"tflops\": " << result.tflops << "\n"
            << "}\n";
  return 0;
}

int run_bf16_wvsplitk_probe() {
  const aima::Bf16WvSplitKProbeResult result = aima::probe_bf16_wvsplitk();
  std::cout << std::setprecision(17)
            << "{\n"
            << "  \"schema\": \"aima-amd395-qwen36/native-bf16-wvsplitk-probe/v2\",\n"
            << "  \"complete\": true,\n"
            << "  \"runtime_python\": false,\n"
            << "  \"runtime_torch\": false,\n"
            << "  \"runtime_vllm\": false,\n"
            << "  \"runtime_triton\": false,\n"
            << "  \"provider\": \"native_vllm_wvSplitK_bf16_n1\",\n"
            << "  \"upstream_vllm_commit\": "
               "\"29e5d102050669d03992a2eb863ad364ea50fab2\",\n"
            << "  \"gpu_arch\": \"" << json_escape(result.gpu_arch) << "\",\n"
            << "  \"cases\": [\n";
  for (std::size_t case_index = 0; case_index < result.cases.size(); ++case_index) {
    const auto& value = result.cases[case_index];
    std::cout << "    {\"m\":" << value.m
              << ",\"n\":1,\"k\":" << value.k
              << ",\"cu_count\":" << value.cu_count
              << ",\"active_waves_per_group\":"
              << value.active_waves_per_group
              << ",\"launches_per_sample\":" << value.launches_per_sample
              << ",\"measured_ms\":[";
    for (std::size_t index = 0; index < value.measured_ms.size(); ++index) {
      if (index != 0) std::cout << ',';
      std::cout << value.measured_ms[index];
    }
    std::cout << "],\"median_ms\":" << value.median_ms
              << ",\"effective_weight_bandwidth_gbs\":"
              << value.effective_weight_bandwidth_gbs
              << ",\"maximum_absolute_error\":"
              << value.maximum_absolute_error
              << ",\"relative_l2_error\":" << value.relative_l2_error
              << ",\"finite_elements\":" << value.finite_elements
              << ",\"expected_elements\":" << value.expected_elements
              << ",\"output_bf16_sha256\":\""
              << value.output_bf16_sha256 << "\"}"
              << (case_index + 1 == result.cases.size() ? "\n" : ",\n");
  }
  std::cout << "  ]\n}\n";
  return 0;
}

int run_derived_weights_probe(int argc, char** argv) {
  aima::NativeWeightLoadOptions options;
  options.native_report = std::filesystem::absolute("native-derived-weight-load.json");
  bool have_model_dir = false;
  for (int index = 2; index < argc; ++index) {
    const std::string argument = argv[index];
    auto next = [&](const char* name) -> std::string {
      if (++index >= argc) throw std::runtime_error(std::string(name) + " requires a value");
      return argv[index];
    };
    if (argument == "--model-dir") {
      options.model_dir = std::filesystem::absolute(next("--model-dir"));
      have_model_dir = true;
    } else if (argument == "--report") {
      options.native_report = std::filesystem::absolute(next("--report"));
    } else if (argument == "--device") {
      options.device = parse_int(next("--device"), "--device");
    } else if (argument == "--workers") {
      options.worker_count = parse_size(next("--workers"), "--workers");
    } else if (argument == "--chunk-bytes") {
      options.chunk_bytes = parse_size(next("--chunk-bytes"), "--chunk-bytes");
    } else {
      throw std::runtime_error("unknown derived-weight argument: " + argument);
    }
  }
  if (!have_model_dir) throw std::runtime_error("--model-dir is required");

  aima::NativeWeightStore weights;
  const aima::NativeWeightLoadMetrics load = weights.load(options);
  aima::NativeDerivedWeightStore derived;
  const aima::NativeDerivedWeightMetrics metrics = derived.build(weights, options.device);
  const aima::NativeDerivedProjectionResult projection =
      aima::probe_layer0_derived_projection(weights, derived);
  const aima::NativeDerivedProjectionResult router =
      aima::validate_layer0_router_transpose(weights, derived);
  std::cout << std::setprecision(17)
            << "{\n"
            << "  \"schema\": \"aima-amd395-qwen36/native-derived-weights-probe/v1\",\n"
            << "  \"complete\": true,\n"
            << "  \"runtime_python\": false,\n"
            << "  \"runtime_torch\": false,\n"
            << "  \"runtime_vllm\": false,\n"
            << "  \"runtime_triton\": false,\n"
            << "  \"raw_tensor_count\": " << load.tensor_count << ",\n"
            << "  \"raw_payload_bytes\": " << load.payload_bytes << ",\n"
            << "  \"raw_load_wall_ms\": " << load.load_wall_ms << ",\n"
            << "  \"derived_view_count\": " << metrics.view_count << ",\n"
            << "  \"derived_payload_bytes\": " << metrics.payload_bytes << ",\n"
            << "  \"allocation_ms\": " << metrics.allocation_ms << ",\n"
            << "  \"pack_ms\": " << metrics.pack_ms << ",\n"
            << "  \"checksum_ms\": " << metrics.checksum_ms << ",\n"
            << "  \"build_wall_ms\": " << metrics.build_wall_ms << ",\n"
            << "  \"free_bytes_before\": " << metrics.free_bytes_before << ",\n"
            << "  \"free_bytes_after\": " << metrics.free_bytes_after << ",\n"
            << "  \"source_u16_xor\": " << metrics.source_u16_xor << ",\n"
            << "  \"source_u16_sum\": " << metrics.source_u16_sum << ",\n"
            << "  \"derived_u16_xor\": " << metrics.derived_u16_xor << ",\n"
            << "  \"derived_u16_sum\": " << metrics.derived_u16_sum << ",\n"
            << "  \"full_payload_checksum_equal\": "
            << (metrics.full_payload_checksum_equal ? "true" : "false") << ",\n"
            << "  \"exact_sample_elements\": " << metrics.exact_sample_elements << ",\n"
            << "  \"expected_sample_elements\": "
            << metrics.expected_sample_elements << ",\n"
            << "  \"projection_elements\": " << projection.elements << ",\n"
            << "  \"projection_exact_elements\": "
            << projection.exact_elements << ",\n"
            << "  \"projection_maximum_absolute_error\": "
            << projection.maximum_absolute_error << ",\n"
            << "  \"projection_relative_l2_error\": "
            << projection.relative_l2_error << ",\n"
            << "  \"router_transpose_elements\": " << router.elements << ",\n"
            << "  \"router_transpose_exact_elements\": "
            << router.exact_elements << ",\n"
            << "  \"router_transpose_maximum_absolute_error\": "
            << router.maximum_absolute_error << ",\n"
            << "  \"router_transpose_relative_l2_error\": "
            << router.relative_l2_error << "\n"
            << "}\n";
  return 0;
}

int run_lm_head_probe(int argc, char** argv) {
  aima::NativeWeightLoadOptions options;
  options.native_report = std::filesystem::absolute("native-lm-head-weight-load.json");
  std::filesystem::path dump_scales;
  std::filesystem::path dump_residual_l2;
  bool have_model_dir = false;
  for (int index = 2; index < argc; ++index) {
    const std::string argument = argv[index];
    auto next = [&](const char* name) -> std::string {
      if (++index >= argc) throw std::runtime_error(std::string(name) + " requires a value");
      return argv[index];
    };
    if (argument == "--model-dir") {
      options.model_dir = std::filesystem::absolute(next("--model-dir"));
      have_model_dir = true;
    } else if (argument == "--report") {
      options.native_report = std::filesystem::absolute(next("--report"));
    } else if (argument == "--device") {
      options.device = parse_int(next("--device"), "--device");
    } else if (argument == "--workers") {
      options.worker_count = parse_size(next("--workers"), "--workers");
    } else if (argument == "--chunk-bytes") {
      options.chunk_bytes = parse_size(next("--chunk-bytes"), "--chunk-bytes");
    } else if (argument == "--dump-scales") {
      dump_scales = std::filesystem::absolute(next("--dump-scales"));
    } else if (argument == "--dump-residual-l2") {
      dump_residual_l2 =
          std::filesystem::absolute(next("--dump-residual-l2"));
    } else {
      throw std::runtime_error("unknown LM-head argument: " + argument);
    }
  }
  if (!have_model_dir) throw std::runtime_error("--model-dir is required");

  aima::NativeWeightStore weights;
  const aima::NativeWeightLoadMetrics load = weights.load(options);
  aima::NativeLmHeadStore lm_head;
  const aima::NativeLmHeadMetrics metrics = lm_head.build(weights, options.device);
  if (!dump_scales.empty()) lm_head.write_scales_for_validation(dump_scales);
  if (!dump_residual_l2.empty()) {
    lm_head.write_residual_l2_for_validation(dump_residual_l2);
  }
  const bool complete =
      metrics.q_weight_reference_exact && metrics.scales_reference_exact;
  std::cout << std::setprecision(17)
            << "{\n"
            << "  \"schema\": \"aima-amd395-qwen36/native-lm-head-probe/v1\",\n"
            << "  \"complete\": " << (complete ? "true" : "false") << ",\n"
            << "  \"runtime_python\": false,\n"
            << "  \"runtime_torch\": false,\n"
            << "  \"runtime_vllm\": false,\n"
            << "  \"runtime_triton\": false,\n"
            << "  \"raw_load_wall_ms\": " << load.load_wall_ms << ",\n"
            << "  \"payload_bytes\": " << metrics.payload_bytes << ",\n"
            << "  \"allocation_ms\": " << metrics.allocation_ms << ",\n"
            << "  \"quantize_ms\": " << metrics.quantize_ms << ",\n"
            << "  \"hash_ms\": " << metrics.hash_ms << ",\n"
            << "  \"build_wall_ms\": " << metrics.build_wall_ms << ",\n"
            << "  \"free_bytes_before\": " << metrics.free_bytes_before << ",\n"
            << "  \"free_bytes_after\": " << metrics.free_bytes_after << ",\n"
            << "  \"q_weight_sha256\": \"" << metrics.q_weight_sha256 << "\",\n"
            << "  \"scales_sha256\": \"" << metrics.scales_sha256 << "\",\n"
            << "  \"residual_l2_sha256\": \""
            << metrics.residual_l2_sha256 << "\",\n"
            << "  \"q_weight_reference_exact\": "
            << (metrics.q_weight_reference_exact ? "true" : "false") << ",\n"
            << "  \"scales_reference_exact\": "
            << (metrics.scales_reference_exact ? "true" : "false") << ",\n"
            << "  \"residual_l2_reference_exact\": "
            << (metrics.residual_l2_reference_exact ? "true" : "false") << ",\n"
            << "  \"q_weight_samples\": [";
  for (std::size_t index = 0; index < metrics.q_weight_samples.size(); ++index) {
    if (index != 0) std::cout << ',';
    std::cout << static_cast<int>(metrics.q_weight_samples[index]);
  }
  std::cout << "],\n  \"scale_samples\": [";
  for (std::size_t index = 0; index < metrics.scale_samples.size(); ++index) {
    if (index != 0) std::cout << ',';
    std::cout << metrics.scale_samples[index];
  }
  std::cout << "],\n  \"residual_l2_samples\": [";
  for (std::size_t index = 0; index < metrics.residual_l2_samples.size(); ++index) {
    if (index != 0) std::cout << ',';
    std::cout << metrics.residual_l2_samples[index];
  }
  std::cout << "]\n"
            << "}\n";
  return complete ? 0 : 3;
}

int run_decode_bindings_probe(int argc, char** argv) {
  aima::NativeWeightLoadOptions options;
  options.native_report =
      std::filesystem::absolute("native-decode-bindings-weight-load.json");
  bool have_model_dir = false;
  for (int index = 2; index < argc; ++index) {
    const std::string argument = argv[index];
    auto next = [&](const char* name) -> std::string {
      if (++index >= argc) throw std::runtime_error(std::string(name) + " requires a value");
      return argv[index];
    };
    if (argument == "--model-dir") {
      options.model_dir = std::filesystem::absolute(next("--model-dir"));
      have_model_dir = true;
    } else if (argument == "--report") {
      options.native_report = std::filesystem::absolute(next("--report"));
    } else if (argument == "--device") {
      options.device = parse_int(next("--device"), "--device");
    } else if (argument == "--workers") {
      options.worker_count = parse_size(next("--workers"), "--workers");
    } else if (argument == "--chunk-bytes") {
      options.chunk_bytes = parse_size(next("--chunk-bytes"), "--chunk-bytes");
    } else {
      throw std::runtime_error("unknown decode-binding argument: " + argument);
    }
  }
  if (!have_model_dir) throw std::runtime_error("--model-dir is required");

  const auto started = std::chrono::steady_clock::now();
  aima::NativeWeightStore weights;
  const aima::NativeWeightLoadMetrics load = weights.load(options);
  aima::NativeDerivedWeightStore derived;
  const aima::NativeDerivedWeightMetrics derived_metrics =
      derived.build(weights, options.device);
  aima::NativeLmHeadStore lm_head;
  const aima::NativeLmHeadMetrics lm_head_metrics =
      lm_head.build(weights, options.device);
  aima::NativeDecodeBindings bindings;
  const aima::NativeDecodeBindingMetrics binding_metrics =
      bindings.build(weights, derived, lm_head);
  aima::NativeDecodeWorkspace workspace;
  const aima::NativeDecodeWorkspaceMetrics workspace_metrics =
      workspace.build(options.device);
  aima::NativeDecodeInvocations invocations;
  const aima::NativeDecodeInvocationMetrics invocation_metrics =
      invocations.build(bindings, workspace);
  const double total_wall_ms = elapsed_ms(started);
  const bool complete =
      lm_head_metrics.q_weight_reference_exact &&
      lm_head_metrics.scales_reference_exact &&
      binding_metrics.unique_bindings == 423 &&
      invocation_metrics.abi_argument_count == 1927;
  std::cout << std::setprecision(17)
            << "{\n"
            << "  \"schema\": \"aima-amd395-qwen36/native-decode-bindings-probe/v1\",\n"
            << "  \"complete\": " << (complete ? "true" : "false") << ",\n"
            << "  \"runtime_python\": false,\n"
            << "  \"runtime_torch\": false,\n"
            << "  \"runtime_vllm\": false,\n"
            << "  \"runtime_triton\": false,\n"
            << "  \"raw_load_wall_ms\": " << load.load_wall_ms << ",\n"
            << "  \"layer_derived_build_wall_ms\": "
            << derived_metrics.build_wall_ms << ",\n"
            << "  \"lm_head_build_wall_ms\": "
            << lm_head_metrics.build_wall_ms << ",\n"
            << "  \"total_wall_ms\": " << total_wall_ms << ",\n"
            << "  \"layer_derived_payload_bytes\": "
            << derived_metrics.payload_bytes << ",\n"
            << "  \"lm_head_derived_payload_bytes\": "
            << lm_head_metrics.payload_bytes << ",\n"
            << "  \"lm_head_q_weight_reference_exact\": "
            << (lm_head_metrics.q_weight_reference_exact ? "true" : "false") << ",\n"
            << "  \"lm_head_scales_reference_exact\": "
            << (lm_head_metrics.scales_reference_exact ? "true" : "false") << ",\n"
            << "  \"lm_head_residual_l2_sha256\": \""
            << lm_head_metrics.residual_l2_sha256 << "\",\n"
            << "  \"schedule_weight_arguments\": "
            << binding_metrics.schedule_weight_arguments << ",\n"
            << "  \"unique_bindings\": "
            << binding_metrics.unique_bindings << ",\n"
            << "  \"raw_weight_bindings\": "
            << binding_metrics.raw_weight_bindings << ",\n"
            << "  \"layer_derived_bindings\": "
            << binding_metrics.layer_derived_bindings << ",\n"
            << "  \"lm_head_derived_bindings\": "
            << binding_metrics.lm_head_derived_bindings << ",\n"
            << "  \"device_pointer_checks\": "
            << binding_metrics.device_pointer_checks << ",\n"
            << "  \"exact_payload_byte_checks\": "
            << binding_metrics.exact_payload_byte_checks << ",\n"
            << "  \"workspace_unique_bindings\": "
            << workspace_metrics.unique_bindings << ",\n"
            << "  \"workspace_resident_bindings\": "
            << workspace_metrics.resident_bindings << ",\n"
            << "  \"workspace_transient_bindings\": "
            << workspace_metrics.transient_bindings << ",\n"
            << "  \"workspace_logical_payload_bytes\": "
            << workspace_metrics.logical_payload_bytes << ",\n"
            << "  \"workspace_allocation_bytes\": "
            << workspace_metrics.allocation_bytes << ",\n"
            << "  \"workspace_allocation_and_zero_ms\": "
            << workspace_metrics.allocation_and_zero_ms << ",\n"
            << "  \"prepared_launches\": "
            << invocation_metrics.launch_count << ",\n"
            << "  \"prepared_abi_arguments\": "
            << invocation_metrics.abi_argument_count << ",\n"
            << "  \"prepared_tensor_arguments\": "
            << invocation_metrics.tensor_argument_count << ",\n"
            << "  \"prepared_scalar_arguments\": "
            << invocation_metrics.scalar_argument_count << ",\n"
            << "  \"prepared_pointer_offset_checks\": "
            << invocation_metrics.pointer_offset_checks << "\n"
            << "}\n";
  return complete ? 0 : 3;
}

int run_linear_prefill_oracle_probe(int argc, char** argv) {
  aima::NativeWeightLoadOptions options;
  options.native_report =
      std::filesystem::absolute("native-linear-prefill-oracle-weight-load.json");
  std::filesystem::path oracle_dir;
  bool have_model_dir = false;
  bool have_oracle_dir = false;
  for (int index = 2; index < argc; ++index) {
    const std::string argument = argv[index];
    auto next = [&](const char* name) -> std::string {
      if (++index >= argc) {
        throw std::runtime_error(std::string(name) + " requires a value");
      }
      return argv[index];
    };
    if (argument == "--model-dir") {
      options.model_dir = std::filesystem::absolute(next("--model-dir"));
      have_model_dir = true;
    } else if (argument == "--oracle-dir") {
      oracle_dir = std::filesystem::absolute(next("--oracle-dir"));
      have_oracle_dir = true;
    } else if (argument == "--report") {
      options.native_report = std::filesystem::absolute(next("--report"));
    } else if (argument == "--device") {
      options.device = parse_int(next("--device"), "--device");
    } else if (argument == "--workers") {
      options.worker_count = parse_size(next("--workers"), "--workers");
    } else if (argument == "--chunk-bytes") {
      options.chunk_bytes = parse_size(next("--chunk-bytes"), "--chunk-bytes");
    } else {
      throw std::runtime_error(
          "unknown linear-prefill oracle argument: " + argument);
    }
  }
  if (!have_model_dir || !have_oracle_dir) {
    throw std::runtime_error("--model-dir and --oracle-dir are required");
  }

  const auto started = std::chrono::steady_clock::now();
  aima::NativeWeightStore weights;
  const aima::NativeWeightLoadMetrics load = weights.load(options);
  aima::NativeDerivedWeightStore derived;
  const aima::NativeDerivedWeightMetrics derived_metrics =
      derived.build(weights, options.device);
  aima::NativeLmHeadStore lm_head;
  const aima::NativeLmHeadMetrics lm_head_metrics =
      lm_head.build(weights, options.device);
  aima::NativeDecodeBindings bindings;
  const aima::NativeDecodeBindingMetrics binding_metrics =
      bindings.build(weights, derived, lm_head);
  aima::NativePrefillWorkspace workspace;
  const aima::NativePrefillWorkspaceMetrics workspace_metrics =
      workspace.build(options.device);
  aima::NativePrefillInvocations invocations;
  const aima::NativePrefillInvocationMetrics invocation_metrics =
      invocations.build(bindings, workspace);
  aima::NativeDecodeExecutor executor;
  const aima::NativeDecodeExecutorMetrics executor_load = executor.load();
  const aima::NativeLinearPrefillOracleResult oracle =
      aima::probe_native_q8192_linear_prefill_layer0_oracle(
          oracle_dir, weights, workspace, invocations, executor);

  bool every_boundary_gate = true;
  std::size_t exact_comparisons = 0;
  for (const auto& comparison : oracle.comparisons) {
    every_boundary_gate =
        every_boundary_gate &&
        comparison.finite_elements == comparison.elements &&
        comparison.relative_l2_error <= 0.002 &&
        comparison.cosine_similarity >= 0.999;
    exact_comparisons +=
        comparison.exact_elements == comparison.elements ? 1 : 0;
  }
  const bool qualified =
      oracle.all_finite && oracle.final_state_gate_passed &&
      oracle.attention_output_gate_passed &&
      oracle.post_attention_gate_passed && every_boundary_gate &&
      oracle.layer.dense_gemm_launches == 5 &&
      oracle.layer.native_pointwise_launches == 1 &&
      oracle.layer.semantic_alias_rebindings == 2 &&
      oracle.layer.aot_launches == 11;
  std::cout << std::setprecision(17)
            << "{\n"
            << "  \"schema\": \"aima-amd395-qwen36/native-linear-prefill-oracle-probe/v1\",\n"
            << "  \"complete\": true,\n"
            << "  \"qualified\": " << (qualified ? "true" : "false")
            << ",\n"
            << "  \"runtime_python\": false,\n"
            << "  \"runtime_torch\": false,\n"
            << "  \"runtime_vllm\": false,\n"
            << "  \"runtime_triton\": false,\n"
            << "  \"tokens\": " << oracle.layer.tokens << ",\n"
            << "  \"layer_index\": " << oracle.layer.layer_index << ",\n"
            << "  \"raw_load_wall_ms\": " << load.load_wall_ms << ",\n"
            << "  \"layer_derived_build_wall_ms\": "
            << derived_metrics.build_wall_ms << ",\n"
            << "  \"lm_head_build_wall_ms\": "
            << lm_head_metrics.build_wall_ms << ",\n"
            << "  \"decode_weight_bindings\": "
            << binding_metrics.unique_bindings << ",\n"
            << "  \"prefill_workspace_bytes\": "
            << workspace_metrics.allocation_bytes << ",\n"
            << "  \"prefill_workspace_runtime_scratch_bindings\": "
            << workspace_metrics.runtime_scratch_bindings << ",\n"
            << "  \"prefill_workspace_runtime_scratch_payload_bytes\": "
            << workspace_metrics.runtime_scratch_payload_bytes << ",\n"
            << "  \"prefill_workspace_mixed_dtype_bindings\": "
            << workspace_metrics.mixed_dtype_bindings << ",\n"
            << "  \"prefill_prepared_launches\": "
            << invocation_metrics.launch_count << ",\n"
            << "  \"prefill_prepared_abi_arguments\": "
            << invocation_metrics.abi_argument_count << ",\n"
            << "  \"aot_loaded_modules\": "
            << executor_load.loaded_modules << ",\n"
            << "  \"seed_tensors\": " << oracle.seed_tensors << ",\n"
            << "  \"seed_bytes\": " << oracle.seed_bytes << ",\n"
            << "  \"dense_gemm_launches\": "
            << oracle.layer.dense_gemm_launches << ",\n"
            << "  \"native_pointwise_launches\": "
            << oracle.layer.native_pointwise_launches << ",\n"
            << "  \"semantic_alias_rebindings\": "
            << oracle.layer.semantic_alias_rebindings << ",\n"
            << "  \"diagnostic_gemm_launches\": "
            << oracle.layer.diagnostic_gemm_launches << ",\n"
            << "  \"aot_launches\": " << oracle.layer.aot_launches << ",\n"
            << "  \"gemm_workspace_bytes\": "
            << oracle.layer.gemm_workspace_bytes << ",\n"
            << "  \"layer_wall_ms\": " << oracle.layer.wall_ms << ",\n"
            << "  \"all_finite\": "
            << (oracle.all_finite ? "true" : "false") << ",\n"
            << "  \"exact_comparisons\": " << exact_comparisons << ",\n"
            << "  \"comparison_count\": " << oracle.comparisons.size()
            << ",\n"
            << "  \"comparisons\": [\n";
  for (std::size_t index = 0; index < oracle.comparisons.size(); ++index) {
    const auto& value = oracle.comparisons[index];
    std::cout << "    {\"label\":\"" << json_escape(value.label)
              << "\",\"dtype\":\"" << json_escape(value.dtype)
              << "\",\"elements\":" << value.elements
              << ",\"exact_elements\":" << value.exact_elements
              << ",\"finite_elements\":" << value.finite_elements
              << ",\"maximum_absolute_error\":"
              << value.maximum_absolute_error
              << ",\"relative_l2_error\":" << value.relative_l2_error
              << ",\"cosine_similarity\":" << value.cosine_similarity
              << ",\"expected_sha256\":\"" << value.expected_sha256
              << "\",\"actual_sha256\":\"" << value.actual_sha256 << "\"}"
              << (index + 1 == oracle.comparisons.size() ? "\n" : ",\n");
  }
  std::cout << "  ],\n"
            << "  \"total_wall_ms\": " << elapsed_ms(started) << "\n"
            << "}\n";
  return qualified ? 0 : 3;
}

int run_moe_prefill_oracle_probe(int argc, char** argv) {
  aima::NativeWeightLoadOptions options;
  options.native_report =
      std::filesystem::absolute("native-moe-prefill-oracle-weight-load.json");
  std::filesystem::path oracle_dir;
  std::filesystem::path boundary_oracle_dir;
  bool have_model_dir = false;
  bool have_oracle_dir = false;
  for (int index = 2; index < argc; ++index) {
    const std::string argument = argv[index];
    auto next = [&](const char* name) -> std::string {
      if (++index >= argc) {
        throw std::runtime_error(std::string(name) + " requires a value");
      }
      return argv[index];
    };
    if (argument == "--model-dir") {
      options.model_dir = std::filesystem::absolute(next("--model-dir"));
      have_model_dir = true;
    } else if (argument == "--oracle-dir") {
      oracle_dir = std::filesystem::absolute(next("--oracle-dir"));
      have_oracle_dir = true;
    } else if (argument == "--boundary-oracle-dir") {
      boundary_oracle_dir =
          std::filesystem::absolute(next("--boundary-oracle-dir"));
    } else if (argument == "--report") {
      options.native_report = std::filesystem::absolute(next("--report"));
    } else if (argument == "--device") {
      options.device = parse_int(next("--device"), "--device");
    } else if (argument == "--workers") {
      options.worker_count = parse_size(next("--workers"), "--workers");
    } else if (argument == "--chunk-bytes") {
      options.chunk_bytes = parse_size(next("--chunk-bytes"), "--chunk-bytes");
    } else {
      throw std::runtime_error(
          "unknown moe-prefill oracle argument: " + argument);
    }
  }
  if (!have_model_dir || !have_oracle_dir) {
    throw std::runtime_error("--model-dir and --oracle-dir are required");
  }

  const auto started = std::chrono::steady_clock::now();
  aima::NativeWeightStore weights;
  const aima::NativeWeightLoadMetrics load = weights.load(options);
  aima::NativeDerivedWeightStore derived;
  const aima::NativeDerivedWeightMetrics derived_metrics =
      derived.build(weights, options.device);
  aima::NativeLmHeadStore lm_head;
  const aima::NativeLmHeadMetrics lm_head_metrics =
      lm_head.build(weights, options.device);
  aima::NativeDecodeBindings bindings;
  const aima::NativeDecodeBindingMetrics binding_metrics =
      bindings.build(weights, derived, lm_head);
  aima::NativePrefillWorkspace workspace;
  const aima::NativePrefillWorkspaceMetrics workspace_metrics =
      workspace.build(options.device);
  aima::NativePrefillInvocations invocations;
  const aima::NativePrefillInvocationMetrics invocation_metrics =
      invocations.build(bindings, workspace);
  aima::NativeDecodeExecutor executor;
  const aima::NativeDecodeExecutorMetrics executor_load = executor.load();
  aima::NativeMoePrefillOracleOptions oracle_options;
  oracle_options.boundary_oracle_dir = boundary_oracle_dir;
  const aima::NativeMoePrefillOracleResult oracle =
      aima::probe_native_q8192_moe_prefill_layer0_oracle(
          oracle_dir, weights, workspace, invocations, executor,
          oracle_options);

  std::size_t exact_comparisons = 0;
  for (const auto& comparison : oracle.comparisons) {
    exact_comparisons +=
        comparison.exact_elements == comparison.elements ? 1 : 0;
  }
  const bool qualified =
      oracle.all_finite && oracle.router_ids_exact &&
      oracle.router_weights_gate_passed && oracle.dispatch_count_exact &&
      oracle.shared_expert_gate_passed && oracle.combined_moe_gate_passed &&
      oracle.expert_boundaries_gate_passed &&
      oracle.oracle_seeded_combined_moe_gate_passed &&
      oracle.final_hidden_gate_passed &&
      oracle.layer.dense_gemm_launches == 5 &&
      oracle.layer.native_router_launches == 1 &&
      oracle.layer.native_dispatch_launches == 1 &&
      oracle.layer.native_pointwise_launches == 5 &&
      oracle.layer.aot_launches == 2;
  std::cout << std::setprecision(17)
            << "{\n"
            << "  \"schema\": \"aima-amd395-qwen36/native-moe-prefill-oracle-probe/v1\",\n"
            << "  \"complete\": true,\n"
            << "  \"qualified\": " << (qualified ? "true" : "false")
            << ",\n"
            << "  \"runtime_python\": false,\n"
            << "  \"runtime_torch\": false,\n"
            << "  \"runtime_vllm\": false,\n"
            << "  \"runtime_triton\": false,\n"
            << "  \"tokens\": " << oracle.layer.tokens << ",\n"
            << "  \"layer_index\": " << oracle.layer.layer_index << ",\n"
            << "  \"raw_load_wall_ms\": " << load.load_wall_ms << ",\n"
            << "  \"layer_derived_build_wall_ms\": "
            << derived_metrics.build_wall_ms << ",\n"
            << "  \"lm_head_build_wall_ms\": "
            << lm_head_metrics.build_wall_ms << ",\n"
            << "  \"decode_weight_bindings\": "
            << binding_metrics.unique_bindings << ",\n"
            << "  \"prefill_workspace_bytes\": "
            << workspace_metrics.allocation_bytes << ",\n"
            << "  \"prefill_workspace_runtime_scratch_bindings\": "
            << workspace_metrics.runtime_scratch_bindings << ",\n"
            << "  \"prefill_workspace_runtime_scratch_payload_bytes\": "
            << workspace_metrics.runtime_scratch_payload_bytes << ",\n"
            << "  \"prefill_prepared_launches\": "
            << invocation_metrics.launch_count << ",\n"
            << "  \"prefill_prepared_abi_arguments\": "
            << invocation_metrics.abi_argument_count << ",\n"
            << "  \"aot_loaded_modules\": "
            << executor_load.loaded_modules << ",\n"
            << "  \"seed_tensors\": " << oracle.seed_tensors << ",\n"
            << "  \"seed_bytes\": " << oracle.seed_bytes << ",\n"
            << "  \"post_attention_seeded\": "
            << (oracle.post_attention_seeded ? "true" : "false") << ",\n"
            << "  \"dense_gemm_launches\": "
            << oracle.layer.dense_gemm_launches << ",\n"
            << "  \"native_router_launches\": "
            << oracle.layer.native_router_launches << ",\n"
            << "  \"native_dispatch_launches\": "
            << oracle.layer.native_dispatch_launches << ",\n"
            << "  \"native_pointwise_launches\": "
            << oracle.layer.native_pointwise_launches << ",\n"
            << "  \"aot_launches\": " << oracle.layer.aot_launches << ",\n"
            << "  \"diagnostic_aot_launches\": "
            << oracle.layer.diagnostic_aot_launches << ",\n"
            << "  \"diagnostic_pointwise_launches\": "
            << oracle.layer.diagnostic_pointwise_launches << ",\n"
            << "  \"padded_routed_rows\": "
            << oracle.layer.padded_routed_rows << ",\n"
            << "  \"gemm_workspace_bytes\": "
            << oracle.layer.gemm_workspace_bytes << ",\n"
            << "  \"layer_wall_ms\": " << oracle.layer.wall_ms << ",\n"
            << "  \"router_ids_exact\": "
            << (oracle.router_ids_exact ? "true" : "false") << ",\n"
            << "  \"router_expert_set_rows\": "
            << oracle.router_expert_set_rows << ",\n"
            << "  \"router_expert_set_rows_exact\": "
            << oracle.router_expert_set_rows_exact << ",\n"
            << "  \"router_expert_sets_exact\": "
            << (oracle.router_expert_sets_exact ? "true" : "false")
            << ",\n"
            << "  \"router_weights_gate_passed\": "
            << (oracle.router_weights_gate_passed ? "true" : "false")
            << ",\n"
            << "  \"dispatch_count_exact\": "
            << (oracle.dispatch_count_exact ? "true" : "false") << ",\n"
            << "  \"shared_expert_gate_passed\": "
            << (oracle.shared_expert_gate_passed ? "true" : "false")
            << ",\n"
            << "  \"combined_moe_gate_passed\": "
            << (oracle.combined_moe_gate_passed ? "true" : "false")
            << ",\n"
            << "  \"expert_boundaries_provided\": "
            << (oracle.expert_boundaries_provided ? "true" : "false")
            << ",\n"
            << "  \"expert_boundaries_gate_passed\": "
            << (oracle.expert_boundaries_gate_passed ? "true" : "false")
            << ",\n"
            << "  \"oracle_seeded_combined_moe_gate_passed\": "
            << (oracle.oracle_seeded_combined_moe_gate_passed ? "true"
                                                              : "false")
            << ",\n"
            << "  \"final_hidden_gate_passed\": "
            << (oracle.final_hidden_gate_passed ? "true" : "false")
            << ",\n"
            << "  \"all_finite\": "
            << (oracle.all_finite ? "true" : "false") << ",\n"
            << "  \"exact_comparisons\": " << exact_comparisons << ",\n"
            << "  \"comparison_count\": " << oracle.comparisons.size()
            << ",\n"
            << "  \"comparisons\": [\n";
  for (std::size_t index = 0; index < oracle.comparisons.size(); ++index) {
    const auto& value = oracle.comparisons[index];
    std::cout << "    {\"label\":\"" << json_escape(value.label)
              << "\",\"dtype\":\"" << json_escape(value.dtype)
              << "\",\"elements\":" << value.elements
              << ",\"exact_elements\":" << value.exact_elements
              << ",\"finite_elements\":" << value.finite_elements
              << ",\"maximum_absolute_error\":"
              << value.maximum_absolute_error
              << ",\"relative_l2_error\":" << value.relative_l2_error
              << ",\"cosine_similarity\":" << value.cosine_similarity
              << ",\"expected_sha256\":\"" << value.expected_sha256
              << "\",\"actual_sha256\":\"" << value.actual_sha256 << "\"}"
              << (index + 1 == oracle.comparisons.size() ? "\n" : ",\n");
  }
  std::cout << "  ],\n"
            << "  \"total_wall_ms\": " << elapsed_ms(started) << "\n"
            << "}\n";
  return qualified ? 0 : 3;
}

int run_prefill_layer0_oracle_probe(int argc, char** argv) {
  aima::NativeWeightLoadOptions options;
  options.native_report = std::filesystem::absolute(
      "native-prefill-layer0-oracle-weight-load.json");
  std::filesystem::path oracle_dir;
  std::filesystem::path attention_oracle_dir;
  std::filesystem::path moe_oracle_dir;
  std::filesystem::path boundary_oracle_dir;
  std::filesystem::path linear_boundary_oracle_dir;
  std::string oracle_label_prefix;
  std::string boundary_oracle_label_prefix;
  std::size_t layer_index = 0;
  bool have_model_dir = false;
  for (int index = 2; index < argc; ++index) {
    const std::string argument = argv[index];
    auto next = [&](const char* name) -> std::string {
      if (++index >= argc) {
        throw std::runtime_error(std::string(name) + " requires a value");
      }
      return argv[index];
    };
    if (argument == "--model-dir") {
      options.model_dir = std::filesystem::absolute(next("--model-dir"));
      have_model_dir = true;
    } else if (argument == "--oracle-dir") {
      oracle_dir = std::filesystem::absolute(next("--oracle-dir"));
    } else if (argument == "--attention-oracle-dir") {
      attention_oracle_dir =
          std::filesystem::absolute(next("--attention-oracle-dir"));
    } else if (argument == "--moe-oracle-dir") {
      moe_oracle_dir =
          std::filesystem::absolute(next("--moe-oracle-dir"));
    } else if (argument == "--boundary-oracle-dir") {
      boundary_oracle_dir =
          std::filesystem::absolute(next("--boundary-oracle-dir"));
    } else if (argument == "--linear-boundary-oracle-dir") {
      linear_boundary_oracle_dir = std::filesystem::absolute(
          next("--linear-boundary-oracle-dir"));
    } else if (argument == "--oracle-label-prefix") {
      oracle_label_prefix = next("--oracle-label-prefix");
    } else if (argument == "--boundary-oracle-label-prefix") {
      boundary_oracle_label_prefix = next("--boundary-oracle-label-prefix");
    } else if (argument == "--layer") {
      layer_index = static_cast<std::size_t>(
          parse_int(next("--layer"), "--layer"));
    } else if (argument == "--report") {
      options.native_report = std::filesystem::absolute(next("--report"));
    } else if (argument == "--device") {
      options.device = parse_int(next("--device"), "--device");
    } else if (argument == "--workers") {
      options.worker_count = parse_size(next("--workers"), "--workers");
    } else if (argument == "--chunk-bytes") {
      options.chunk_bytes = parse_size(next("--chunk-bytes"), "--chunk-bytes");
    } else {
      throw std::runtime_error(
          "unknown prefill-layer0 oracle argument: " + argument);
    }
  }
  if (!oracle_dir.empty()) {
    if (attention_oracle_dir.empty()) attention_oracle_dir = oracle_dir;
    if (moe_oracle_dir.empty()) moe_oracle_dir = oracle_dir;
  }
  if (!have_model_dir || attention_oracle_dir.empty() ||
      moe_oracle_dir.empty()) {
    throw std::runtime_error(
        "--model-dir and either --oracle-dir or both split oracle dirs are required");
  }
  if (layer_index >= 40 || layer_index % 4 == 3) {
    throw std::runtime_error(
        "--layer must identify one of the 30 linear-attention layers");
  }
  if (oracle_label_prefix.empty() && layer_index != 0) {
    std::ostringstream prefix;
    prefix << "layer-" << std::setw(3) << std::setfill('0') << layer_index
           << '-';
    oracle_label_prefix = prefix.str();
  }

  const auto started = std::chrono::steady_clock::now();
  aima::NativeWeightStore weights;
  const aima::NativeWeightLoadMetrics load = weights.load(options);
  aima::NativeDerivedWeightStore derived;
  const aima::NativeDerivedWeightMetrics derived_metrics =
      derived.build(weights, options.device);
  aima::NativeLmHeadStore lm_head;
  const aima::NativeLmHeadMetrics lm_head_metrics =
      lm_head.build(weights, options.device);
  aima::NativeDecodeBindings bindings;
  const aima::NativeDecodeBindingMetrics binding_metrics =
      bindings.build(weights, derived, lm_head);
  aima::NativePrefillWorkspace workspace;
  const aima::NativePrefillWorkspaceMetrics workspace_metrics =
      workspace.build(options.device);
  aima::NativePrefillInvocations invocations;
  const aima::NativePrefillInvocationMetrics invocation_metrics =
      invocations.build(bindings, workspace);
  aima::NativeDecodeExecutor executor;
  const aima::NativeDecodeExecutorMetrics executor_load = executor.load();

  // The only production-chain seed is the layer input.  The linear probe's
  // second seed and the MoE probe's four seeds are post-measurement diagnostic
  // reruns; none feed the attention -> MoE boundary.
  aima::NativeLinearPrefillOracleOptions attention_options;
  attention_options.layer_index = layer_index;
  attention_options.oracle_label_prefix = oracle_label_prefix;
  attention_options.boundary_oracle_dir = linear_boundary_oracle_dir;
  attention_options.boundary_oracle_label_prefix = oracle_label_prefix;
  const aima::NativeLinearPrefillOracleResult attention =
      aima::probe_native_q8192_linear_prefill_layer0_oracle(
          attention_oracle_dir, weights, workspace, invocations, executor,
          attention_options);
  aima::NativeMoePrefillOracleOptions moe_options;
  moe_options.layer_index = layer_index;
  moe_options.seed_post_attention = false;
  moe_options.oracle_label_prefix = oracle_label_prefix;
  moe_options.boundary_oracle_dir = boundary_oracle_dir;
  moe_options.boundary_oracle_label_prefix = boundary_oracle_label_prefix;
  const aima::NativeMoePrefillOracleResult moe =
      aima::probe_native_q8192_moe_prefill_layer0_oracle(
          moe_oracle_dir, weights, workspace, invocations, executor,
          moe_options);

  bool every_attention_gate = true;
  std::size_t exact_comparisons = 0;
  for (const auto& comparison : attention.comparisons) {
    every_attention_gate =
        every_attention_gate &&
        comparison.finite_elements == comparison.elements &&
        comparison.relative_l2_error <= 0.002 &&
        comparison.cosine_similarity >= 0.999;
    exact_comparisons +=
        comparison.exact_elements == comparison.elements ? 1 : 0;
  }
  for (const auto& comparison : moe.comparisons) {
    exact_comparisons +=
        comparison.exact_elements == comparison.elements ? 1 : 0;
  }
  const bool attention_qualified =
      attention.all_finite && attention.final_state_gate_passed &&
      attention.attention_output_gate_passed &&
      attention.post_attention_gate_passed && every_attention_gate &&
      attention.layer.dense_gemm_launches == 5 &&
      attention.layer.native_pointwise_launches == 1 &&
      attention.layer.semantic_alias_rebindings == 2 &&
      attention.layer.aot_launches == 11;
  const bool moe_qualified =
      moe.all_finite && moe.router_expert_sets_exact &&
      moe.router_weights_gate_passed && moe.dispatch_count_exact &&
      moe.shared_expert_gate_passed && moe.combined_moe_gate_passed &&
      moe.expert_boundaries_gate_passed &&
      moe.oracle_seeded_combined_moe_gate_passed &&
      moe.final_hidden_gate_passed &&
      moe.layer.dense_gemm_launches == 5 &&
      moe.layer.native_router_launches == 1 &&
      moe.layer.native_dispatch_launches == 1 &&
      moe.layer.native_pointwise_launches == 5 &&
      moe.layer.aot_launches == 2;
  const std::size_t mid_layer_oracle_seed_tensors =
      moe.post_attention_seeded ? 2 : 0;
  const bool qualified =
      attention_qualified && moe_qualified &&
      mid_layer_oracle_seed_tensors == 0 && attention.seed_tensors == 2 &&
      moe.seed_tensors == 4;

  std::cout << std::setprecision(17)
            << "{\n"
            << "  \"schema\": \"aima-amd395-qwen36/native-prefill-layer0-oracle-probe/v1\",\n"
            << "  \"complete\": true,\n"
            << "  \"qualified\": " << (qualified ? "true" : "false")
            << ",\n"
            << "  \"performance_claim\": false,\n"
            << "  \"timing_includes_boundary_checks\": true,\n"
            << "  \"runtime_python\": false,\n"
            << "  \"runtime_torch\": false,\n"
            << "  \"runtime_vllm\": false,\n"
            << "  \"runtime_triton\": false,\n"
            << "  \"tokens\": " << attention.layer.tokens << ",\n"
            << "  \"layer_index\": " << layer_index << ",\n"
            << "  \"raw_load_wall_ms\": " << load.load_wall_ms << ",\n"
            << "  \"layer_derived_build_wall_ms\": "
            << derived_metrics.build_wall_ms << ",\n"
            << "  \"lm_head_build_wall_ms\": "
            << lm_head_metrics.build_wall_ms << ",\n"
            << "  \"decode_weight_bindings\": "
            << binding_metrics.unique_bindings << ",\n"
            << "  \"prefill_workspace_bytes\": "
            << workspace_metrics.allocation_bytes << ",\n"
            << "  \"prefill_prepared_launches\": "
            << invocation_metrics.launch_count << ",\n"
            << "  \"prefill_prepared_abi_arguments\": "
            << invocation_metrics.abi_argument_count << ",\n"
            << "  \"aot_loaded_modules\": "
            << executor_load.loaded_modules << ",\n"
            << "  \"entry_oracle_seed_tensors\": 1,\n"
            << "  \"mid_layer_oracle_seed_tensors\": "
            << mid_layer_oracle_seed_tensors << ",\n"
            << "  \"diagnostic_oracle_seed_tensors\": "
            << attention.seed_tensors + moe.seed_tensors - 1 << ",\n"
            << "  \"attention_qualified\": "
            << (attention_qualified ? "true" : "false") << ",\n"
            << "  \"moe_qualified\": "
            << (moe_qualified ? "true" : "false") << ",\n"
            << "  \"attention_production_aot_launches\": "
            << attention.layer.aot_launches << ",\n"
            << "  \"moe_production_aot_launches\": "
            << moe.layer.aot_launches << ",\n"
            << "  \"production_dense_gemm_launches\": "
            << attention.layer.dense_gemm_launches +
                   moe.layer.dense_gemm_launches
            << ",\n"
            << "  \"production_native_pointwise_launches\": "
            << attention.layer.native_pointwise_launches +
                   moe.layer.native_pointwise_launches
            << ",\n"
            << "  \"qualification_layer_wall_ms\": "
            << attention.layer.wall_ms + moe.layer.wall_ms << ",\n"
            << "  \"router_ids_exact\": "
            << (moe.router_ids_exact ? "true" : "false") << ",\n"
            << "  \"router_expert_set_rows\": "
            << moe.router_expert_set_rows << ",\n"
            << "  \"router_expert_set_rows_exact\": "
            << moe.router_expert_set_rows_exact << ",\n"
            << "  \"router_expert_sets_exact\": "
            << (moe.router_expert_sets_exact ? "true" : "false")
            << ",\n"
            << "  \"dispatch_count_exact\": "
            << (moe.dispatch_count_exact ? "true" : "false") << ",\n"
            << "  \"final_hidden_gate_passed\": "
            << (moe.final_hidden_gate_passed ? "true" : "false") << ",\n"
            << "  \"expert_boundaries_provided\": "
            << (moe.expert_boundaries_provided ? "true" : "false")
            << ",\n"
            << "  \"expert_boundaries_gate_passed\": "
            << (moe.expert_boundaries_gate_passed ? "true" : "false")
            << ",\n"
            << "  \"exact_comparisons\": " << exact_comparisons << ",\n"
            << "  \"comparison_count\": "
            << attention.comparisons.size() + moe.comparisons.size()
            << ",\n"
            << "  \"attention_comparisons\": [\n";
  for (std::size_t index = 0; index < attention.comparisons.size(); ++index) {
    const auto& value = attention.comparisons[index];
    std::cout << "    {\"label\":\"" << json_escape(value.label)
              << "\",\"dtype\":\"" << json_escape(value.dtype)
              << "\",\"elements\":" << value.elements
              << ",\"exact_elements\":" << value.exact_elements
              << ",\"finite_elements\":" << value.finite_elements
              << ",\"maximum_absolute_error\":"
              << value.maximum_absolute_error
              << ",\"relative_l2_error\":" << value.relative_l2_error
              << ",\"cosine_similarity\":" << value.cosine_similarity
              << ",\"expected_sha256\":\"" << value.expected_sha256
              << "\",\"actual_sha256\":\"" << value.actual_sha256 << "\"}"
              << (index + 1 == attention.comparisons.size() ? "\n" : ",\n");
  }
  std::cout << "  ],\n  \"attention_boundary_comparisons\": [\n";
  for (std::size_t index = 0;
       index < attention.boundary_comparisons.size(); ++index) {
    const auto& value = attention.boundary_comparisons[index];
    std::cout << "    {\"label\":\"" << json_escape(value.label)
              << "\",\"dtype\":\"" << json_escape(value.dtype)
              << "\",\"elements\":" << value.elements
              << ",\"exact_elements\":" << value.exact_elements
              << ",\"finite_elements\":" << value.finite_elements
              << ",\"maximum_absolute_error\":"
              << json_number(value.maximum_absolute_error)
              << ",\"relative_l2_error\":"
              << json_number(value.relative_l2_error)
              << ",\"cosine_similarity\":"
              << json_number(value.cosine_similarity) << "}"
              << (index + 1 == attention.boundary_comparisons.size()
                      ? "\n"
                      : ",\n");
  }
  std::cout << "  ],\n  \"moe_comparisons\": [\n";
  for (std::size_t index = 0; index < moe.comparisons.size(); ++index) {
    const auto& value = moe.comparisons[index];
    std::cout << "    {\"label\":\"" << json_escape(value.label)
              << "\",\"dtype\":\"" << json_escape(value.dtype)
              << "\",\"elements\":" << value.elements
              << ",\"exact_elements\":" << value.exact_elements
              << ",\"finite_elements\":" << value.finite_elements
              << ",\"maximum_absolute_error\":"
              << value.maximum_absolute_error
              << ",\"relative_l2_error\":" << value.relative_l2_error
              << ",\"cosine_similarity\":" << value.cosine_similarity
              << ",\"expected_sha256\":\"" << value.expected_sha256
              << "\",\"actual_sha256\":\"" << value.actual_sha256 << "\"}"
              << (index + 1 == moe.comparisons.size() ? "\n" : ",\n");
  }
  std::cout << "  ],\n"
            << "  \"total_wall_ms\": " << elapsed_ms(started) << "\n"
            << "}\n";
  return qualified ? 0 : 3;
}

int run_prefill_full_layer_oracle_probe(int argc, char** argv) {
  aima::NativeWeightLoadOptions options;
  options.native_report = std::filesystem::absolute(
      "native-prefill-full-layer-oracle-weight-load.json");
  std::filesystem::path oracle_dir;
  std::filesystem::path ck_provider_path;
  std::size_t layer_index = 3;
  bool have_model_dir = false;
  for (int index = 2; index < argc; ++index) {
    const std::string argument = argv[index];
    auto next = [&](const char* name) -> std::string {
      if (++index >= argc) {
        throw std::runtime_error(std::string(name) + " requires a value");
      }
      return argv[index];
    };
    if (argument == "--model-dir") {
      options.model_dir = std::filesystem::absolute(next("--model-dir"));
      have_model_dir = true;
    } else if (argument == "--oracle-dir") {
      oracle_dir = std::filesystem::absolute(next("--oracle-dir"));
    } else if (argument == "--ck-provider") {
      ck_provider_path = std::filesystem::absolute(next("--ck-provider"));
    } else if (argument == "--layer") {
      layer_index = static_cast<std::size_t>(
          parse_int(next("--layer"), "--layer"));
    } else if (argument == "--report") {
      options.native_report = std::filesystem::absolute(next("--report"));
    } else if (argument == "--device") {
      options.device = parse_int(next("--device"), "--device");
    } else if (argument == "--workers") {
      options.worker_count = parse_size(next("--workers"), "--workers");
    } else if (argument == "--chunk-bytes") {
      options.chunk_bytes = parse_size(next("--chunk-bytes"), "--chunk-bytes");
    } else {
      throw std::runtime_error(
          "unknown prefill full-layer oracle argument: " + argument);
    }
  }
  if (!have_model_dir || oracle_dir.empty() || ck_provider_path.empty()) {
    throw std::runtime_error(
        "--model-dir, --oracle-dir, and --ck-provider are required");
  }
  if (layer_index >= 40 || layer_index % 4 != 3) {
    throw std::runtime_error(
        "--layer must identify one of the ten full-attention layers");
  }

  std::ostringstream prefix_stream;
  prefix_stream << "layer-" << std::setw(3) << std::setfill('0')
                << layer_index << '-';
  const std::string oracle_label_prefix = prefix_stream.str();
  const auto started = std::chrono::steady_clock::now();
  aima::NativeWeightStore weights;
  const aima::NativeWeightLoadMetrics load = weights.load(options);
  aima::NativeDerivedWeightStore derived;
  const aima::NativeDerivedWeightMetrics derived_metrics =
      derived.build(weights, options.device);
  aima::NativeLmHeadStore lm_head;
  const aima::NativeLmHeadMetrics lm_head_metrics =
      lm_head.build(weights, options.device);
  aima::NativeDecodeBindings bindings;
  const aima::NativeDecodeBindingMetrics binding_metrics =
      bindings.build(weights, derived, lm_head);
  aima::NativePrefillWorkspace workspace;
  const aima::NativePrefillWorkspaceMetrics workspace_metrics =
      workspace.build(options.device);
  aima::NativePrefillInvocations invocations;
  const aima::NativePrefillInvocationMetrics invocation_metrics =
      invocations.build(bindings, workspace);
  aima::NativeDecodeExecutor executor;
  const aima::NativeDecodeExecutorMetrics executor_load = executor.load();
  aima::NativeQ8192CkProvider provider;
  const aima::NativeQ8192CkProviderMetrics provider_load =
      provider.load(ck_provider_path);

  aima::NativeFullPrefillOracleOptions attention_options;
  attention_options.layer_index = layer_index;
  attention_options.oracle_label_prefix = oracle_label_prefix;
  const aima::NativeFullPrefillOracleResult attention =
      aima::probe_native_q8192_full_prefill_oracle(
          oracle_dir, weights, workspace, invocations, executor, provider,
          attention_options);
  aima::NativeMoePrefillOracleOptions moe_options;
  moe_options.layer_index = layer_index;
  moe_options.seed_post_attention = false;
  moe_options.run_routing_diagnostic = true;
  moe_options.oracle_label_prefix = oracle_label_prefix;
  const aima::NativeMoePrefillOracleResult moe =
      aima::probe_native_q8192_moe_prefill_layer0_oracle(
          oracle_dir, weights, workspace, invocations, executor, moe_options);

  const bool attention_qualified =
      attention.all_finite && attention.q_gate_passed &&
      attention.q_passed && attention.k_passed && attention.v_passed &&
      attention.attention_output_passed &&
      attention.projected_output_passed && attention.post_attention_passed &&
      attention.layer.dense_gemm_launches == 4 &&
      attention.layer.native_pointwise_launches == 3 &&
      attention.layer.native_ck_fmha_launches == 1 &&
      attention.layer.aot_launches == 2;
  const bool moe_qualified =
      moe.all_finite && moe.router_expert_sets_exact &&
      moe.router_weights_gate_passed && moe.dispatch_count_exact &&
      moe.shared_expert_gate_passed && moe.combined_moe_gate_passed &&
      moe.oracle_seeded_combined_moe_gate_passed &&
      moe.final_hidden_gate_passed && moe.layer.dense_gemm_launches == 5 &&
      moe.layer.native_router_launches == 1 &&
      moe.layer.native_dispatch_launches == 1 &&
      moe.layer.native_pointwise_launches == 5 &&
      moe.layer.aot_launches == 2;
  const bool qualified =
      attention_qualified && moe_qualified &&
      attention.seed_tensors == 1 && !moe.post_attention_seeded;

  std::cout << std::setprecision(17)
            << "{\n"
            << "  \"schema\": \"aima-amd395-qwen36/native-prefill-full-layer-oracle-probe/v1\",\n"
            << "  \"complete\": true,\n"
            << "  \"qualified\": " << (qualified ? "true" : "false")
            << ",\n"
            << "  \"correctness_claim\": false,\n"
            << "  \"performance_claim\": false,\n"
            << "  \"runtime_python\": false,\n"
            << "  \"runtime_torch\": false,\n"
            << "  \"runtime_vllm\": false,\n"
            << "  \"runtime_triton\": false,\n"
            << "  \"tokens\": 8192,\n"
            << "  \"layer_index\": " << layer_index << ",\n"
            << "  \"raw_load_wall_ms\": " << load.load_wall_ms << ",\n"
            << "  \"layer_derived_build_wall_ms\": "
            << derived_metrics.build_wall_ms << ",\n"
            << "  \"lm_head_build_wall_ms\": "
            << lm_head_metrics.build_wall_ms << ",\n"
            << "  \"decode_weight_bindings\": "
            << binding_metrics.unique_bindings << ",\n"
            << "  \"prefill_workspace_bytes\": "
            << workspace_metrics.allocation_bytes << ",\n"
            << "  \"prefill_prepared_launches\": "
            << invocation_metrics.launch_count << ",\n"
            << "  \"aot_loaded_modules\": "
            << executor_load.loaded_modules << ",\n"
            << "  \"ck_provider_loaded\": "
            << (provider_load.loaded ? "true" : "false") << ",\n"
            << "  \"ck_provider_path\": \""
            << json_escape(provider_load.library_path.string()) << "\",\n"
            << "  \"entry_oracle_seed_tensors\": "
            << attention.seed_tensors << ",\n"
            << "  \"mid_layer_oracle_seed_tensors\": 0,\n"
            << "  \"diagnostic_oracle_seed_tensors\": "
            << moe.seed_tensors << ",\n"
            << "  \"attention_qualified\": "
            << (attention_qualified ? "true" : "false") << ",\n"
            << "  \"moe_qualified\": "
            << (moe_qualified ? "true" : "false") << ",\n"
            << "  \"attention_dense_gemm_launches\": "
            << attention.layer.dense_gemm_launches << ",\n"
            << "  \"attention_native_pointwise_launches\": "
            << attention.layer.native_pointwise_launches << ",\n"
            << "  \"attention_ck_fmha_launches\": "
            << attention.layer.native_ck_fmha_launches << ",\n"
            << "  \"attention_aot_launches\": "
            << attention.layer.aot_launches << ",\n"
            << "  \"attention_wall_ms\": "
            << attention.layer.wall_ms << ",\n"
            << "  \"router_expert_set_rows_exact\": "
            << moe.router_expert_set_rows_exact << ",\n"
            << "  \"attention_comparisons\": [\n";
  for (std::size_t index = 0; index < attention.comparisons.size(); ++index) {
    const auto& value = attention.comparisons[index];
    std::cout << "    {\"label\":\"" << json_escape(value.label)
              << "\",\"elements\":" << value.elements
              << ",\"exact_elements\":" << value.exact_elements
              << ",\"finite_elements\":" << value.finite_elements
              << ",\"maximum_absolute_error\":"
              << json_number(value.maximum_absolute_error)
              << ",\"relative_l2_error\":"
              << json_number(value.relative_l2_error)
              << ",\"cosine_similarity\":"
              << json_number(value.cosine_similarity) << "}"
              << (index + 1 == attention.comparisons.size() ? "\n" : ",\n");
  }
  std::cout << "  ],\n  \"moe_comparisons\": [\n";
  for (std::size_t index = 0; index < moe.comparisons.size(); ++index) {
    const auto& value = moe.comparisons[index];
    std::cout << "    {\"label\":\"" << json_escape(value.label)
              << "\",\"elements\":" << value.elements
              << ",\"exact_elements\":" << value.exact_elements
              << ",\"finite_elements\":" << value.finite_elements
              << ",\"maximum_absolute_error\":"
              << json_number(value.maximum_absolute_error)
              << ",\"relative_l2_error\":"
              << json_number(value.relative_l2_error)
              << ",\"cosine_similarity\":"
              << json_number(value.cosine_similarity) << "}"
              << (index + 1 == moe.comparisons.size() ? "\n" : ",\n");
  }
  std::cout << "  ],\n"
            << "  \"total_wall_ms\": " << elapsed_ms(started) << "\n"
            << "}\n";
  return qualified ? 0 : 3;
}

int run_prefill_linear_prefix_oracle_probe(int argc, char** argv) {
  aima::NativeWeightLoadOptions options;
  options.native_report = std::filesystem::absolute(
      "native-prefill-linear-prefix-oracle-weight-load.json");
  std::filesystem::path layer0_attention_oracle_dir;
  std::filesystem::path layer0_moe_oracle_dir;
  std::filesystem::path later_oracle_dir;
  std::filesystem::path linear_boundary_oracle_dir;
  std::filesystem::path full_layer_oracle_dir;
  std::filesystem::path ck_provider_path;
  std::filesystem::path chain_output_oracle_dir;
  std::size_t through_layer = 2;
  bool diagnostic_seed_layer0_post_attention = false;
  bool have_model_dir = false;
  for (int index = 2; index < argc; ++index) {
    const std::string argument = argv[index];
    auto next = [&](const char* name) -> std::string {
      if (++index >= argc) {
        throw std::runtime_error(std::string(name) + " requires a value");
      }
      return argv[index];
    };
    if (argument == "--model-dir") {
      options.model_dir = std::filesystem::absolute(next("--model-dir"));
      have_model_dir = true;
    } else if (argument == "--layer0-attention-oracle-dir") {
      layer0_attention_oracle_dir = std::filesystem::absolute(
          next("--layer0-attention-oracle-dir"));
    } else if (argument == "--layer0-moe-oracle-dir") {
      layer0_moe_oracle_dir = std::filesystem::absolute(
          next("--layer0-moe-oracle-dir"));
    } else if (argument == "--later-oracle-dir" ||
               argument == "--oracle-dir") {
      later_oracle_dir = std::filesystem::absolute(next(argument.c_str()));
    } else if (argument == "--linear-boundary-oracle-dir") {
      linear_boundary_oracle_dir = std::filesystem::absolute(
          next("--linear-boundary-oracle-dir"));
    } else if (argument == "--full-layer-oracle-dir") {
      full_layer_oracle_dir = std::filesystem::absolute(
          next("--full-layer-oracle-dir"));
    } else if (argument == "--ck-provider") {
      ck_provider_path = std::filesystem::absolute(next("--ck-provider"));
    } else if (argument == "--chain-output-oracle-dir") {
      chain_output_oracle_dir = std::filesystem::absolute(
          next("--chain-output-oracle-dir"));
    } else if (argument == "--through-layer") {
      through_layer = static_cast<std::size_t>(
          parse_int(next("--through-layer"), "--through-layer"));
    } else if (argument == "--diagnostic-seed-layer0-post-attention") {
      diagnostic_seed_layer0_post_attention = true;
    } else if (argument == "--report") {
      options.native_report = std::filesystem::absolute(next("--report"));
    } else if (argument == "--device") {
      options.device = parse_int(next("--device"), "--device");
    } else if (argument == "--workers") {
      options.worker_count = parse_size(next("--workers"), "--workers");
    } else if (argument == "--chunk-bytes") {
      options.chunk_bytes = parse_size(next("--chunk-bytes"), "--chunk-bytes");
    } else {
      throw std::runtime_error(
          "unknown prefill linear-prefix oracle argument: " + argument);
    }
  }
  if (!have_model_dir || layer0_attention_oracle_dir.empty() ||
      layer0_moe_oracle_dir.empty() ||
      (through_layer > 0 && later_oracle_dir.empty())) {
    throw std::runtime_error(
        "model and layer-0 oracle dirs are required; later layers also require --later-oracle-dir");
  }
  if (through_layer > 3) {
    throw std::runtime_error(
        "--through-layer must be 0, 1, 2, or 3 for this bounded prefix probe");
  }
  if (through_layer == 3 &&
      (full_layer_oracle_dir.empty() || ck_provider_path.empty())) {
    throw std::runtime_error(
        "--through-layer 3 requires --full-layer-oracle-dir and --ck-provider");
  }

  const auto started = std::chrono::steady_clock::now();
  aima::NativeWeightStore weights;
  const aima::NativeWeightLoadMetrics load = weights.load(options);
  aima::NativeDerivedWeightStore derived;
  const aima::NativeDerivedWeightMetrics derived_metrics =
      derived.build(weights, options.device);
  aima::NativeLmHeadStore lm_head;
  const aima::NativeLmHeadMetrics lm_head_metrics =
      lm_head.build(weights, options.device);
  aima::NativeDecodeBindings bindings;
  const aima::NativeDecodeBindingMetrics binding_metrics =
      bindings.build(weights, derived, lm_head);
  aima::NativePrefillWorkspace workspace;
  const aima::NativePrefillWorkspaceMetrics workspace_metrics =
      workspace.build(options.device);
  aima::NativePrefillInvocations invocations;
  const aima::NativePrefillInvocationMetrics invocation_metrics =
      invocations.build(bindings, workspace);
  aima::NativeDecodeExecutor executor;
  const aima::NativeDecodeExecutorMetrics executor_load = executor.load();
  aima::NativeQ8192CkProvider provider;
  aima::NativeQ8192CkProviderMetrics provider_load;
  if (through_layer == 3) {
    provider_load = provider.load(ck_provider_path);
  }

  struct LayerResult {
    bool full_attention = false;
    aima::NativeLinearPrefillOracleResult linear;
    aima::NativeFullPrefillOracleResult full;
    aima::NativeMoePrefillOracleResult moe;
  };
  std::vector<LayerResult> layers;
  layers.reserve(through_layer + 1);
  std::size_t seed_tensors = 0;
  std::size_t mid_layer_seed_tensors = 0;
  std::size_t production_aot_launches = 0;
  std::size_t production_dense_gemm_launches = 0;
  std::size_t production_native_pointwise_launches = 0;
  std::size_t production_ck_fmha_launches = 0;
  bool qualified = true;

  const auto require_comparison = [](
      const std::vector<aima::NativeOracleComparison>& comparisons,
      const char* label) -> const aima::NativeOracleComparison& {
    const auto found = std::find_if(
        comparisons.begin(), comparisons.end(),
        [label](const aima::NativeOracleComparison& value) {
          return value.label == label;
        });
    if (found == comparisons.end()) {
      throw std::runtime_error(
          std::string("missing native prefix comparison: ") + label);
    }
    return *found;
  };

  const std::size_t linear_through_layer = std::min<std::size_t>(
      through_layer, 2);
  for (std::size_t layer_index = 0; layer_index <= linear_through_layer;
       ++layer_index) {
    std::string label_prefix;
    if (layer_index != 0) {
      std::ostringstream prefix;
      prefix << "layer-" << std::setw(3) << std::setfill('0')
             << layer_index << '-';
      label_prefix = prefix.str();
    }
    const std::filesystem::path& attention_oracle =
        layer_index == 0 ? layer0_attention_oracle_dir : later_oracle_dir;
    const std::filesystem::path& moe_oracle =
        layer_index == 0 ? layer0_moe_oracle_dir : later_oracle_dir;

    aima::NativeLinearPrefillOracleOptions attention_options;
    attention_options.layer_index = layer_index;
    attention_options.seed_layer_input = layer_index == 0;
    attention_options.run_output_projection_diagnostic = false;
    attention_options.oracle_label_prefix = label_prefix;
    if (layer_index == 1 && !linear_boundary_oracle_dir.empty()) {
      attention_options.boundary_oracle_dir = linear_boundary_oracle_dir;
      attention_options.boundary_oracle_label_prefix = label_prefix;
    }
    aima::NativeLinearPrefillOracleResult attention =
        aima::probe_native_q8192_linear_prefill_layer0_oracle(
            attention_oracle, weights, workspace, invocations, executor,
            attention_options);

    aima::NativeMoePrefillOracleOptions moe_options;
    moe_options.layer_index = layer_index;
    moe_options.seed_post_attention =
        diagnostic_seed_layer0_post_attention && layer_index == 0;
    moe_options.run_routing_diagnostic = false;
    moe_options.oracle_label_prefix = label_prefix;
    if (!chain_output_oracle_dir.empty()) {
      std::ostringstream chain_label;
      chain_label << "layer-" << std::setw(3) << std::setfill('0')
                  << layer_index << "-return-layer_body-output";
      moe_options.chain_output_oracle_dir = chain_output_oracle_dir;
      moe_options.chain_output_oracle_label = chain_label.str();
    }
    aima::NativeMoePrefillOracleResult moe =
        aima::probe_native_q8192_moe_prefill_layer0_oracle(
            moe_oracle, weights, workspace, invocations, executor,
            moe_options);

    const auto& layer_input =
        require_comparison(attention.comparisons, "layer_input");
    const auto& layer_output =
        require_comparison(moe.comparisons, "layer_output");
    const auto& qualified_layer_output =
        moe.chain_output_comparison_provided
            ? moe.chain_output_comparison
            : layer_output;
    const bool layer_qualified =
        attention.all_finite && moe.all_finite &&
        layer_input.relative_l2_error <= 0.02 &&
        layer_input.cosine_similarity >= 0.999 &&
        qualified_layer_output.relative_l2_error <= 0.02 &&
        qualified_layer_output.cosine_similarity >= 0.999 &&
        moe.router_expert_set_rows_exact * 100 >=
            moe.router_expert_set_rows * 98 &&
        attention.layer.aot_launches == 11 &&
        attention.layer.dense_gemm_launches == 5 &&
        attention.layer.native_pointwise_launches == 1 &&
        attention.layer.diagnostic_gemm_launches == 0 &&
        moe.layer.aot_launches == 2 &&
        moe.layer.dense_gemm_launches == 5 &&
        moe.layer.native_pointwise_launches == 5 &&
        moe.layer.diagnostic_aot_launches == 0 &&
        moe.layer.diagnostic_pointwise_launches == 0 &&
        !moe.post_attention_seeded && !moe.routing_diagnostic_ran;
    qualified = qualified && layer_qualified;
    seed_tensors += attention.seed_tensors + moe.seed_tensors;
    if (moe.post_attention_seeded) mid_layer_seed_tensors += 2;
    production_aot_launches +=
        attention.layer.aot_launches + moe.layer.aot_launches;
    production_dense_gemm_launches +=
        attention.layer.dense_gemm_launches + moe.layer.dense_gemm_launches;
    production_native_pointwise_launches +=
        attention.layer.native_pointwise_launches +
        moe.layer.native_pointwise_launches;
    LayerResult layer;
    layer.linear = std::move(attention);
    layer.moe = std::move(moe);
    layers.push_back(std::move(layer));
  }

  if (through_layer == 3) {
    constexpr std::size_t layer_index = 3;
    const std::string label_prefix = "layer-003-";
    aima::NativeFullPrefillOracleOptions attention_options;
    attention_options.layer_index = layer_index;
    attention_options.seed_layer_input = false;
    attention_options.prepare_rotary_table = true;
    attention_options.oracle_label_prefix = label_prefix;
    aima::NativeFullPrefillOracleResult attention =
        aima::probe_native_q8192_full_prefill_oracle(
            full_layer_oracle_dir, weights, workspace, invocations, executor,
            provider, attention_options);

    aima::NativeMoePrefillOracleOptions moe_options;
    moe_options.layer_index = layer_index;
    moe_options.seed_post_attention = false;
    moe_options.run_routing_diagnostic = false;
    moe_options.oracle_label_prefix = label_prefix;
    if (!chain_output_oracle_dir.empty()) {
      moe_options.chain_output_oracle_dir = chain_output_oracle_dir;
      moe_options.chain_output_oracle_label =
          "layer-003-return-layer_body-output";
    }
    aima::NativeMoePrefillOracleResult moe =
        aima::probe_native_q8192_moe_prefill_layer0_oracle(
            full_layer_oracle_dir, weights, workspace, invocations, executor,
            moe_options);

    const auto& layer_input =
        require_comparison(attention.comparisons, "layer_input");
    const auto& layer_output =
        require_comparison(moe.comparisons, "layer_output");
    const auto& qualified_layer_output =
        moe.chain_output_comparison_provided
            ? moe.chain_output_comparison
            : layer_output;
    const bool layer_qualified =
        attention.all_finite && moe.all_finite &&
        layer_input.relative_l2_error <= 0.02 &&
        layer_input.cosine_similarity >= 0.999 &&
        qualified_layer_output.relative_l2_error <= 0.02 &&
        qualified_layer_output.cosine_similarity >= 0.999 &&
        moe.router_expert_set_rows_exact * 100 >=
            moe.router_expert_set_rows * 98 &&
        attention.layer.aot_launches == 2 &&
        attention.layer.dense_gemm_launches == 4 &&
        attention.layer.native_pointwise_launches == 3 &&
        attention.layer.native_ck_fmha_launches == 1 &&
        moe.layer.aot_launches == 2 &&
        moe.layer.dense_gemm_launches == 5 &&
        moe.layer.native_pointwise_launches == 5 &&
        moe.layer.diagnostic_aot_launches == 0 &&
        moe.layer.diagnostic_pointwise_launches == 0 &&
        !moe.post_attention_seeded && !moe.routing_diagnostic_ran;
    qualified = qualified && layer_qualified;
    seed_tensors += attention.seed_tensors + moe.seed_tensors;
    production_aot_launches +=
        attention.layer.aot_launches + moe.layer.aot_launches;
    production_dense_gemm_launches +=
        attention.layer.dense_gemm_launches + moe.layer.dense_gemm_launches;
    production_native_pointwise_launches +=
        attention.layer.native_pointwise_launches +
        moe.layer.native_pointwise_launches;
    production_ck_fmha_launches +=
        attention.layer.native_ck_fmha_launches;
    LayerResult layer;
    layer.full_attention = true;
    layer.full = std::move(attention);
    layer.moe = std::move(moe);
    layers.push_back(std::move(layer));
  }
  const std::size_t linear_layer_count = linear_through_layer + 1;
  const std::size_t full_layer_count = through_layer == 3 ? 1 : 0;
  qualified = qualified && seed_tensors == 1 &&
              mid_layer_seed_tensors == 0 &&
              production_aot_launches ==
                  linear_layer_count * 13 + full_layer_count * 4 &&
              production_dense_gemm_launches ==
                  linear_layer_count * 10 + full_layer_count * 9 &&
              production_native_pointwise_launches ==
                  linear_layer_count * 6 + full_layer_count * 8 &&
              production_ck_fmha_launches == full_layer_count;

  std::cout << std::setprecision(17)
            << "{\n"
            << "  \"schema\": \"aima-amd395-qwen36/native-prefill-prefix-oracle-probe/v2\",\n"
            << "  \"complete\": true,\n"
            << "  \"qualified\": " << (qualified ? "true" : "false")
            << ",\n"
            << "  \"correctness_claim\": false,\n"
            << "  \"performance_claim\": false,\n"
            << "  \"requires_end_to_end_logits_gate\": true,\n"
            << "  \"runtime_python\": false,\n"
            << "  \"runtime_torch\": false,\n"
            << "  \"runtime_vllm\": false,\n"
            << "  \"runtime_triton\": false,\n"
            << "  \"tokens\": 8192,\n"
            << "  \"through_layer\": " << through_layer << ",\n"
            << "  \"raw_load_wall_ms\": " << load.load_wall_ms << ",\n"
            << "  \"layer_derived_build_wall_ms\": "
            << derived_metrics.build_wall_ms << ",\n"
            << "  \"lm_head_build_wall_ms\": "
            << lm_head_metrics.build_wall_ms << ",\n"
            << "  \"decode_weight_bindings\": "
            << binding_metrics.unique_bindings << ",\n"
            << "  \"prefill_workspace_bytes\": "
            << workspace_metrics.allocation_bytes << ",\n"
            << "  \"prefill_prepared_launches\": "
            << invocation_metrics.launch_count << ",\n"
            << "  \"aot_loaded_modules\": "
            << executor_load.loaded_modules << ",\n"
            << "  \"ck_provider_loaded\": "
            << (provider_load.loaded ? "true" : "false") << ",\n"
            << "  \"oracle_seed_tensors\": " << seed_tensors << ",\n"
            << "  \"mid_layer_oracle_seed_tensors\": "
            << mid_layer_seed_tensors << ",\n"
            << "  \"diagnostic_oracle_seed_tensors\": 0,\n"
            << "  \"production_aot_launches\": "
            << production_aot_launches << ",\n"
            << "  \"production_dense_gemm_launches\": "
            << production_dense_gemm_launches << ",\n"
            << "  \"production_native_pointwise_launches\": "
            << production_native_pointwise_launches << ",\n"
            << "  \"production_ck_fmha_launches\": "
            << production_ck_fmha_launches << ",\n"
            << "  \"total_wall_ms\": " << elapsed_ms(started) << ",\n"
            << "  \"layers\": [\n";
  for (std::size_t index = 0; index < layers.size(); ++index) {
    const auto& layer = layers[index];
    const auto& attention_comparisons =
        layer.full_attention ? layer.full.comparisons
                             : layer.linear.comparisons;
    const auto& moe = layer.moe;
    const std::size_t layer_index =
        layer.full_attention ? layer.full.layer.layer_index
                             : layer.linear.layer.layer_index;
    const bool attention_all_finite =
        layer.full_attention ? layer.full.all_finite
                             : layer.linear.all_finite;
    const auto& layer_input =
        require_comparison(attention_comparisons, "layer_input");
    const auto& post_norm =
        require_comparison(attention_comparisons, "post_attention_norm");
    const auto& combined =
        require_comparison(moe.comparisons, "combined_moe_output");
    const auto& layer_output =
        require_comparison(moe.comparisons, "layer_output");
    std::cout
        << "    {\"layer_index\":" << layer_index
        << ",\"attention_type\":\""
        << (layer.full_attention ? "full" : "linear") << "\""
        << ",\"attention_all_finite\":"
        << (attention_all_finite ? "true" : "false")
        << ",\"moe_all_finite\":"
        << (moe.all_finite ? "true" : "false")
        << ",\"layer_input_relative_l2_error\":"
        << json_number(layer_input.relative_l2_error)
        << ",\"post_attention_relative_l2_error\":"
        << json_number(post_norm.relative_l2_error)
        << ",\"router_expert_set_rows\":"
        << moe.router_expert_set_rows
        << ",\"router_expert_set_rows_exact\":"
        << moe.router_expert_set_rows_exact
        << ",\"combined_moe_relative_l2_error\":"
        << json_number(combined.relative_l2_error)
        << ",\"layer_output_relative_l2_error\":"
        << json_number(layer_output.relative_l2_error)
        << ",\"layer_output_cosine_similarity\":"
        << json_number(layer_output.cosine_similarity)
        << ",\"same_request_layer_output_relative_l2_error\":"
        << (moe.chain_output_comparison_provided
                ? json_number(
                      moe.chain_output_comparison.relative_l2_error)
                : "null")
        << ",\"same_request_layer_output_cosine_similarity\":"
        << (moe.chain_output_comparison_provided
                ? json_number(
                      moe.chain_output_comparison.cosine_similarity)
                : "null")
        << ",\"same_request_layer_output_exact_elements\":"
        << (moe.chain_output_comparison_provided
                ? std::to_string(
                      moe.chain_output_comparison.exact_elements)
                : "null")
        << ",\"attention_comparisons\":[";
    for (std::size_t comparison_index = 0;
         comparison_index < attention_comparisons.size();
         ++comparison_index) {
      const auto& value = attention_comparisons[comparison_index];
      std::cout << "{\"label\":\"" << json_escape(value.label)
                << "\",\"finite_elements\":" << value.finite_elements
                << ",\"elements\":" << value.elements
                << ",\"relative_l2_error\":"
                << json_number(value.relative_l2_error)
                << "}"
                << (comparison_index + 1 == attention_comparisons.size()
                        ? ""
                        : ",");
    }
    std::cout << "],\"moe_comparisons\":[";
    for (std::size_t comparison_index = 0;
         comparison_index < moe.comparisons.size(); ++comparison_index) {
      const auto& value = moe.comparisons[comparison_index];
      std::cout << "{\"label\":\"" << json_escape(value.label)
                << "\",\"finite_elements\":" << value.finite_elements
                << ",\"elements\":" << value.elements
                << ",\"relative_l2_error\":"
                << json_number(value.relative_l2_error)
                << "}"
                << (comparison_index + 1 == moe.comparisons.size() ? ""
                                                                   : ",");
    }
    std::cout << "],\"fla_boundary_comparisons\":[";
    if (!layer.full_attention) {
      for (std::size_t comparison_index = 0;
           comparison_index < layer.linear.boundary_comparisons.size();
           ++comparison_index) {
        const auto& value =
            layer.linear.boundary_comparisons[comparison_index];
        std::cout << "{\"label\":\"" << json_escape(value.label)
                  << "\",\"exact_elements\":" << value.exact_elements
                  << ",\"finite_elements\":" << value.finite_elements
                  << ",\"elements\":" << value.elements
                  << ",\"relative_l2_error\":"
                  << json_number(value.relative_l2_error) << "}"
                  << (comparison_index + 1 ==
                              layer.linear.boundary_comparisons.size()
                          ? ""
                          : ",");
      }
    }
    std::cout << "]}"
        << (index + 1 == layers.size() ? "\n" : ",\n");
  }
  std::cout << "  ]\n}\n";
  return qualified ? 0 : 3;
}

int run_prefill_all_layers_oracle_probe(int argc, char** argv) {
  aima::NativeWeightLoadOptions options;
  options.native_report = std::filesystem::absolute(
      "native-prefill-all-layers-oracle-weight-load.json");
  std::filesystem::path chain_output_oracle_dir;
  std::filesystem::path entry_input_oracle_dir;
  std::filesystem::path prefill_state_oracle_dir;
  std::filesystem::path decode_oracle_dir;
  std::filesystem::path decode_logits_oracle_dir;
  std::string decode_logits_oracle_label_prefix = "return-final_logits";
  std::filesystem::path ck_provider_path;
  bool have_model_dir = false;
  bool diagnostic_progress = false;
  std::size_t start_layer = 0;
  std::size_t through_layer = 39;
  std::uint32_t expected_top1_token_id = 0;
  bool have_expected_top1_token_id = false;
  std::uint32_t expected_next_token_id = 0;
  bool have_expected_next_token_id = false;
  std::uint32_t uniform_input_token_id = 0;
  bool have_uniform_input_token_id = false;
  std::vector<std::uint32_t> input_token_id_cycle;
  bool execution_only = false;
  for (int index = 2; index < argc; ++index) {
    const std::string argument = argv[index];
    auto next = [&](const char* name) -> std::string {
      if (++index >= argc) {
        throw std::runtime_error(std::string(name) + " requires a value");
      }
      return argv[index];
    };
    if (argument == "--model-dir") {
      options.model_dir = std::filesystem::absolute(next("--model-dir"));
      have_model_dir = true;
    } else if (argument == "--chain-output-oracle-dir" ||
               argument == "--oracle-dir") {
      chain_output_oracle_dir = std::filesystem::absolute(
          next(argument.c_str()));
    } else if (argument == "--ck-provider") {
      ck_provider_path = std::filesystem::absolute(next("--ck-provider"));
    } else if (argument == "--entry-input-oracle-dir") {
      entry_input_oracle_dir = std::filesystem::absolute(
          next("--entry-input-oracle-dir"));
    } else if (argument == "--prefill-state-oracle-dir") {
      prefill_state_oracle_dir = std::filesystem::absolute(
          next("--prefill-state-oracle-dir"));
    } else if (argument == "--decode-oracle-dir") {
      decode_oracle_dir =
          std::filesystem::absolute(next("--decode-oracle-dir"));
    } else if (argument == "--decode-logits-oracle-dir") {
      decode_logits_oracle_dir = std::filesystem::absolute(
          next("--decode-logits-oracle-dir"));
    } else if (argument == "--decode-logits-oracle-label-prefix") {
      decode_logits_oracle_label_prefix =
          next("--decode-logits-oracle-label-prefix");
      if (decode_logits_oracle_label_prefix.empty()) {
        throw std::runtime_error(
            "--decode-logits-oracle-label-prefix must be non-empty");
      }
    } else if (argument == "--uniform-input-token-id") {
      uniform_input_token_id = static_cast<std::uint32_t>(
          parse_int(next("--uniform-input-token-id"),
                    "--uniform-input-token-id"));
      have_uniform_input_token_id = true;
    } else if (argument == "--input-token-id-cycle") {
      const std::string value = next("--input-token-id-cycle");
      std::size_t begin = 0;
      while (begin < value.size()) {
        const std::size_t end = value.find(',', begin);
        const std::string item = value.substr(
            begin, end == std::string::npos ? std::string::npos
                                            : end - begin);
        if (item.empty()) {
          throw std::runtime_error(
              "--input-token-id-cycle contains an empty token id");
        }
        input_token_id_cycle.push_back(static_cast<std::uint32_t>(
            parse_int(item, "--input-token-id-cycle")));
        if (end == std::string::npos) break;
        begin = end + 1;
      }
    } else if (argument == "--diagnostic-progress") {
      diagnostic_progress = true;
    } else if (argument == "--execution-only") {
      execution_only = true;
    } else if (argument == "--start-layer") {
      start_layer = static_cast<std::size_t>(
          parse_int(next("--start-layer"), "--start-layer"));
    } else if (argument == "--through-layer") {
      through_layer = static_cast<std::size_t>(
          parse_int(next("--through-layer"), "--through-layer"));
    } else if (argument == "--expected-top1-token-id") {
      expected_top1_token_id = static_cast<std::uint32_t>(
          parse_int(next("--expected-top1-token-id"),
                    "--expected-top1-token-id"));
      have_expected_top1_token_id = true;
    } else if (argument == "--expected-next-token-id") {
      expected_next_token_id = static_cast<std::uint32_t>(
          parse_int(next("--expected-next-token-id"),
                    "--expected-next-token-id"));
      have_expected_next_token_id = true;
    } else if (argument == "--report") {
      options.native_report = std::filesystem::absolute(next("--report"));
    } else if (argument == "--device") {
      options.device = parse_int(next("--device"), "--device");
    } else if (argument == "--workers") {
      options.worker_count = parse_size(next("--workers"), "--workers");
    } else if (argument == "--chunk-bytes") {
      options.chunk_bytes = parse_size(next("--chunk-bytes"), "--chunk-bytes");
    } else {
      throw std::runtime_error(
          "unknown prefill all-layers oracle argument: " + argument);
    }
  }
  if (!have_model_dir || ck_provider_path.empty() ||
      (!execution_only && chain_output_oracle_dir.empty())) {
    throw std::runtime_error(
        "--model-dir and --ck-provider are required; qualification also requires --chain-output-oracle-dir");
  }
  if (start_layer > through_layer || through_layer >= 40) {
    throw std::runtime_error(
        "--start-layer and --through-layer must describe a non-empty subset of layers 0..39");
  }
  if (have_uniform_input_token_id && !input_token_id_cycle.empty()) {
    throw std::runtime_error(
        "--uniform-input-token-id and --input-token-id-cycle are mutually exclusive");
  }
  const bool have_native_input_token_ids =
      have_uniform_input_token_id || !input_token_id_cycle.empty();
  const bool invalid_cycle_token = std::any_of(
      input_token_id_cycle.begin(), input_token_id_cycle.end(),
      [](std::uint32_t token_id) { return token_id >= 248320; });
  if (have_native_input_token_ids &&
      (start_layer != 0 || uniform_input_token_id >= 248320 ||
       invalid_cycle_token)) {
    throw std::runtime_error(
        "native input token ids require start layer 0 and valid model token ids");
  }
  if (execution_only &&
      (!have_native_input_token_ids || start_layer != 0 ||
       through_layer != 39 || !chain_output_oracle_dir.empty() ||
       !entry_input_oracle_dir.empty() ||
       !prefill_state_oracle_dir.empty() || !decode_oracle_dir.empty() ||
       !decode_logits_oracle_dir.empty())) {
    throw std::runtime_error(
        "--execution-only requires native token ids, the complete 40-layer chain, and no oracle directories");
  }

  constexpr std::size_t kTokens = 8192;
  constexpr std::size_t kHidden = 2048;
  constexpr std::size_t kHiddenBytes =
      kTokens * kHidden * sizeof(std::uint16_t);
  const auto started = std::chrono::steady_clock::now();
  aima::NativeWeightStore weights;
  const aima::NativeWeightLoadMetrics load = weights.load(options);
  aima::NativeDerivedWeightStore derived;
  const aima::NativeDerivedWeightMetrics derived_metrics =
      derived.build(weights, options.device);
  aima::NativeLmHeadStore lm_head;
  const aima::NativeLmHeadMetrics lm_head_metrics =
      lm_head.build(weights, options.device);
  aima::NativeDecodeBindings bindings;
  const aima::NativeDecodeBindingMetrics binding_metrics =
      bindings.build(weights, derived, lm_head);
  aima::NativePrefillWorkspace workspace;
  const aima::NativePrefillWorkspaceMetrics workspace_metrics =
      workspace.build(options.device);
  aima::NativePrefillInvocations invocations;
  const aima::NativePrefillInvocationMetrics invocation_metrics =
      invocations.build(bindings, workspace);
  aima::NativeQ8192PrefillGemmPlans prefill_gemm_plans;
  aima::NativeDecodeWorkspace decode_workspace;
  const aima::NativeDecodeWorkspaceMetrics decode_workspace_metrics =
      decode_workspace.build(options.device);
  aima::NativeDecodeInvocations decode_invocations;
  const aima::NativeDecodeInvocationMetrics decode_invocation_metrics =
      decode_invocations.build(bindings, decode_workspace);
  aima::NativeDecodeExecutor executor;
  const aima::NativeDecodeExecutorMetrics executor_load = executor.load();
  aima::NativeQ8192CkProvider provider;
  const aima::NativeQ8192CkProviderMetrics provider_load =
      provider.load(ck_provider_path);
  aima::NativeFullAttentionState attention_state;
  const aima::NativeFullAttentionStateMetrics attention_state_metrics =
      attention_state.build(kTokens + 1, options.device);
  hipDeviceProp_t properties{};
  if (hipGetDeviceProperties(&properties, options.device) != hipSuccess) {
    throw std::runtime_error(
        "hipGetDeviceProperties failed for native prefill continuation");
  }
  double prefill_gemm_plan_build_wall_ms = 0.0;
  if (execution_only) {
    const auto plan_started = std::chrono::steady_clock::now();
    prefill_gemm_plans.prepare_all();
    prefill_gemm_plan_build_wall_ms = elapsed_ms(plan_started);
  }

  std::size_t start_sequence = invocation_metrics.launch_count;
  for (std::size_t sequence = 0;
       sequence < invocations.launches().size(); ++sequence) {
    const auto* launch = invocations.launches()[sequence].launch;
    if (launch != nullptr &&
        static_cast<std::size_t>(launch->layer_index) == start_layer &&
        std::string(launch->symbol) == "triton_rmsnorm_kernel") {
      start_sequence = sequence;
      break;
    }
  }
  if (start_sequence == invocation_metrics.launch_count) {
    throw std::runtime_error("native prefill start-layer binding is missing");
  }
  std::ostringstream entry_seed_label;
  if (start_layer == 0) {
    entry_seed_label << "layer-000-return-layer_body-inp";
  } else {
    entry_seed_label << "layer-" << std::setw(3) << std::setfill('0')
                     << start_layer - 1
                     << "-return-layer_body-output";
  }
  std::size_t entry_seed_bytes = 0;
  std::size_t entry_native_embedding_launches = 0;
  std::size_t entry_native_input_tokens = 0;
  aima::NativeOracleComparison entry_input_comparison;
  bool entry_input_comparison_provided = false;
  const auto request_started = std::chrono::steady_clock::now();
  if (have_native_input_token_ids) {
    const aima::NativeTensorView* embedding =
        weights.find("model.language_model.embed_tokens.weight");
    const aima::NativePrefillWorkspaceView* token_ids =
        workspace.find("native.prompt_token_ids");
    if (embedding == nullptr || embedding->device_pointer == nullptr ||
        embedding->payload_bytes !=
            248320ULL * kHidden * sizeof(std::uint16_t) ||
        token_ids == nullptr || token_ids->device_pointer == nullptr ||
        token_ids->payload_bytes < kTokens * sizeof(std::uint32_t)) {
      throw std::runtime_error(
          "native prompt embedding ownership is incomplete");
    }
    std::vector<std::uint32_t> input_token_ids(kTokens);
    if (have_uniform_input_token_id) {
      std::fill(input_token_ids.begin(), input_token_ids.end(),
                uniform_input_token_id);
    } else {
      for (std::size_t index = 0; index < input_token_ids.size(); ++index) {
        input_token_ids[index] =
            input_token_id_cycle[index % input_token_id_cycle.size()];
      }
    }
    aima::launch_prompt_embeddings(
        embedding->device_pointer, input_token_ids.data(),
        token_ids->device_pointer,
        invocations.tensor_pointer(start_sequence, "x"), kTokens);
    entry_native_embedding_launches = 1;
    entry_native_input_tokens = input_token_ids.size();
    const std::filesystem::path& entry_comparison_dir =
        entry_input_oracle_dir.empty() ? chain_output_oracle_dir
                                       : entry_input_oracle_dir;
    if (!entry_comparison_dir.empty()) {
      entry_input_comparison = aima::compare_native_oracle_tensor(
          "native_prompt_embeddings", "bfloat16",
          invocations.tensor_pointer(start_sequence, "x"), kHiddenBytes,
          aima::find_native_oracle_tensor_file(
              entry_comparison_dir, entry_seed_label.str()));
      entry_input_comparison_provided = true;
    }
  } else {
    entry_seed_bytes = aima::seed_native_oracle_tensor(
        aima::find_native_oracle_tensor_file(
            chain_output_oracle_dir, entry_seed_label.str()),
        invocations.tensor_pointer(start_sequence, "x"), kHiddenBytes);
  }

  struct LayerOutput {
    std::size_t layer_index = 0;
    bool full_attention = false;
    bool comparison_provided = false;
    aima::NativeOracleComparison output;
  };
  std::vector<LayerOutput> layer_outputs;
  layer_outputs.reserve(through_layer - start_layer + 1);
  std::size_t linear_layers = 0;
  std::size_t full_layers = 0;
  std::size_t production_aot_launches = 0;
  std::size_t production_dense_gemm_launches = 0;
  std::size_t production_native_pointwise_launches = 0;
  std::size_t production_ck_fmha_launches = 0;
  std::size_t production_resident_state_direct_bindings = 0;
  std::size_t production_resident_state_payload_bytes = 0;
  bool all_layers_finite = true;
  double maximum_layer_output_relative_l2_error = 0.0;
  double minimum_layer_output_cosine_similarity = 1.0;
  std::size_t layer_output_comparison_count = 0;

  for (std::size_t layer_index = start_layer;
       layer_index <= through_layer; ++layer_index) {
    const bool full_attention = layer_index % 4 == 3;
    if (full_attention) {
      aima::NativeFullPrefillOracleOptions attention_options;
      attention_options.layer_index = layer_index;
      attention_options.seed_layer_input = false;
      attention_options.prepare_rotary_table = true;
      attention_options.collect_oracle_comparisons = false;
      attention_options.decode_attention_state = &attention_state;
      attention_options.gemm_plans = &prefill_gemm_plans;
      attention_options.cache_position_start = 0;
      attention_options.synchronize_substages =
          diagnostic_progress && layer_index == 39;
      const aima::NativeFullPrefillOracleResult attention =
          aima::probe_native_q8192_full_prefill_oracle(
              {}, weights, workspace, invocations, executor, provider,
              attention_options);
      ++full_layers;
      production_aot_launches += attention.layer.aot_launches;
      production_dense_gemm_launches +=
          attention.layer.dense_gemm_launches;
      production_native_pointwise_launches +=
          attention.layer.native_pointwise_launches;
      production_ck_fmha_launches +=
          attention.layer.native_ck_fmha_launches;
      production_resident_state_direct_bindings +=
          attention.layer.resident_kv_direct_bindings;
      production_resident_state_payload_bytes +=
          attention.layer.resident_kv_payload_bytes;
    } else {
      aima::NativeLinearPrefillOracleOptions attention_options;
      attention_options.layer_index = layer_index;
      attention_options.seed_layer_input = false;
      attention_options.run_output_projection_diagnostic = false;
      attention_options.collect_oracle_comparisons = false;
      attention_options.decode_state_workspace = &decode_workspace;
      attention_options.gemm_plans = &prefill_gemm_plans;
      const aima::NativeLinearPrefillOracleResult attention =
          aima::probe_native_q8192_linear_prefill_layer0_oracle(
              {}, weights, workspace, invocations, executor,
              attention_options);
      ++linear_layers;
      production_aot_launches += attention.layer.aot_launches;
      production_dense_gemm_launches +=
          attention.layer.dense_gemm_launches;
      production_native_pointwise_launches +=
          attention.layer.native_pointwise_launches;
      production_resident_state_direct_bindings +=
          attention.layer.resident_state_direct_bindings;
      production_resident_state_payload_bytes +=
          attention.layer.resident_state_payload_bytes;
    }

    std::ostringstream output_label;
    output_label << "layer-" << std::setw(3) << std::setfill('0')
                 << layer_index << "-return-layer_body-output";
    aima::NativeMoePrefillOracleOptions moe_options;
    moe_options.layer_index = layer_index;
    moe_options.seed_post_attention = false;
    moe_options.run_routing_diagnostic = false;
    moe_options.collect_oracle_comparisons = false;
    moe_options.synchronize_substages =
        diagnostic_progress && layer_index == 39;
    moe_options.gemm_plans = &prefill_gemm_plans;
    moe_options.chain_output_oracle_dir = chain_output_oracle_dir;
    moe_options.chain_output_oracle_label = output_label.str();
    const aima::NativeMoePrefillOracleResult moe =
        aima::probe_native_q8192_moe_prefill_layer0_oracle(
            {}, weights, workspace, invocations, executor, moe_options);
    if (!chain_output_oracle_dir.empty() &&
        !moe.chain_output_comparison_provided) {
      throw std::runtime_error(
          "native all-layer prefill output comparison is missing");
    }
    production_aot_launches += moe.layer.aot_launches;
    production_dense_gemm_launches += moe.layer.dense_gemm_launches;
    production_native_pointwise_launches +=
        moe.layer.native_pointwise_launches;
    all_layers_finite = all_layers_finite && moe.all_finite;
    if (moe.chain_output_comparison_provided) {
      const auto& comparison = moe.chain_output_comparison;
      maximum_layer_output_relative_l2_error = std::max(
          maximum_layer_output_relative_l2_error,
          comparison.relative_l2_error);
      minimum_layer_output_cosine_similarity = std::min(
          minimum_layer_output_cosine_similarity,
          comparison.cosine_similarity);
      ++layer_output_comparison_count;
    }
    layer_outputs.push_back({layer_index, full_attention,
                             moe.chain_output_comparison_provided,
                             moe.chain_output_comparison});
    if (diagnostic_progress) {
      std::cerr << "{\"event\":\"native_prefill_layer_complete\","
                << "\"layer_index\":" << layer_index << ","
                << "\"attention_type\":\""
                << (full_attention ? "full" : "linear") << "\","
                << "\"relative_l2_error\":"
                << (moe.chain_output_comparison_provided
                        ? json_number(
                              moe.chain_output_comparison.relative_l2_error)
                        : "null")
                << "}\n";
    }
  }

  aima::NativeLmHeadTop1Metrics first_token;
  bool first_token_produced = false;
  double execution_prefill_wall_ms = 0.0;
  if (through_layer == 39) {
    const aima::NativePrefillWorkspaceView* terminal =
        workspace.find("transient.31");
    if (terminal == nullptr || terminal->device_pointer == nullptr ||
        terminal->payload_bytes < kHiddenBytes) {
      throw std::runtime_error(
          "native prefill terminal hidden state is unavailable");
    }
    const auto* terminal_bytes =
        static_cast<const unsigned char*>(terminal->device_pointer);
    const void* last_hidden_row =
        terminal_bytes + (kTokens - 1) * kHidden * sizeof(std::uint16_t);
    first_token = aima::run_native_lm_head_top1(
        last_hidden_row, weights, lm_head, decode_workspace,
        decode_invocations, executor, properties.multiProcessorCount);
    first_token_produced = true;
    if (execution_only) {
      execution_prefill_wall_ms = elapsed_ms(request_started);
    }
  }

  std::vector<aima::NativeOracleComparison> prefill_state_comparisons;
  bool prefill_state_all_finite = true;
  std::size_t prefill_state_exact_tensors = 0;
  double prefill_state_worst_recurrent_relative_l2_error = 0.0;
  std::size_t prefill_state_worst_recurrent_layer = 0;
  double prefill_state_worst_conv_relative_l2_error = 0.0;
  std::size_t prefill_state_worst_conv_layer = 0;
  double prefill_state_worst_kv_relative_l2_error = 0.0;
  std::size_t prefill_state_worst_kv_layer = 0;
  std::string prefill_state_worst_kv_kind;
  const bool coherent_prefill_state_oracle =
      !prefill_state_oracle_dir.empty();
  const std::filesystem::path& state_oracle_dir =
      coherent_prefill_state_oracle ? prefill_state_oracle_dir
                                    : decode_oracle_dir;
  if (!state_oracle_dir.empty()) {
    if (start_layer != 0 || through_layer != 39) {
      throw std::runtime_error(
          "prefill state comparison requires the complete layer-0-to-39 chain");
    }
    const auto record = [&](aima::NativeOracleComparison value) {
      prefill_state_all_finite =
          prefill_state_all_finite &&
          value.finite_elements == value.elements;
      prefill_state_exact_tensors +=
          value.exact_elements == value.elements ? 1 : 0;
      prefill_state_comparisons.push_back(std::move(value));
      return prefill_state_comparisons.back();
    };
    constexpr std::size_t kPrefillKvCacheBytes =
        kTokens * 2 * 256 * sizeof(std::uint16_t);
    for (std::size_t layer_index = 0; layer_index < 40; ++layer_index) {
      std::ostringstream prefix;
      prefix << "layer-" << std::setw(3) << std::setfill('0')
             << layer_index << '-';
      if (layer_index % 4 == 3) {
        for (const auto& cache :
             std::array<std::pair<const char*, const void*>, 2>{
                 std::pair<const char*, const void*>{
                     "k", attention_state.k_cache(layer_index)},
                 std::pair<const char*, const void*>{
                     "v", attention_state.v_cache(layer_index)}}) {
          const auto value = record(
              aima::compare_native_oracle_tensor_prefix(
                  "prefill_" + std::string(cache.first) +
                      "_cache_layer_" + std::to_string(layer_index),
                  "bfloat16", cache.second, kPrefillKvCacheBytes,
                  aima::find_native_oracle_tensor_file(
                      state_oracle_dir,
                      prefix.str() + "return-full_attention-" +
                          cache.first + "_cache")));
          if (value.relative_l2_error >
              prefill_state_worst_kv_relative_l2_error) {
            prefill_state_worst_kv_relative_l2_error =
                value.relative_l2_error;
            prefill_state_worst_kv_layer = layer_index;
            prefill_state_worst_kv_kind = cache.first;
          }
        }
        continue;
      }
      const std::string layer = std::to_string(layer_index);
      const aima::NativeDecodeWorkspaceView* conv = decode_workspace.find(
          "linear_attention_initial_conv_states." + layer);
      const aima::NativeDecodeWorkspaceView* recurrent =
          decode_workspace.find(
              "linear_attention_initial_ssm_states_vllm." + layer);
      if (conv == nullptr || recurrent == nullptr) {
        throw std::runtime_error(
            "prefill state comparison binding is missing");
      }
      const auto conv_value = record(aima::compare_native_oracle_tensor(
          "prefill_conv_state_layer_" + layer, "bfloat16",
          conv->device_pointer, conv->payload_bytes,
          aima::find_native_oracle_tensor_file(
              state_oracle_dir,
              prefix.str() +
                  (coherent_prefill_state_oracle
                       ? "launch-001-initial_states_ptr"
                       : "launch-001-state_in"))));
      if (conv_value.relative_l2_error >
          prefill_state_worst_conv_relative_l2_error) {
        prefill_state_worst_conv_relative_l2_error =
            conv_value.relative_l2_error;
        prefill_state_worst_conv_layer = layer_index;
      }
      const auto recurrent_value = record(
          aima::compare_native_oracle_tensor(
              "prefill_recurrent_state_layer_" + layer, "float32",
              recurrent->device_pointer, recurrent->payload_bytes,
              aima::find_native_oracle_tensor_file(
                  state_oracle_dir,
                  prefix.str() +
                      (coherent_prefill_state_oracle ? "launch-007-ht"
                                                     : "launch-002-h0"))));
      if (recurrent_value.relative_l2_error >
          prefill_state_worst_recurrent_relative_l2_error) {
        prefill_state_worst_recurrent_relative_l2_error =
            recurrent_value.relative_l2_error;
        prefill_state_worst_recurrent_layer = layer_index;
      }
    }
  }

  aima::NativeOracleComparison first_token_hidden_comparison;
  aima::NativeLogitsComparison first_token_logits_comparison;
  bool first_token_distribution_comparison_provided = false;
  if (coherent_prefill_state_oracle && first_token_produced) {
    const aima::NativeDecodeWorkspaceView* final_hidden =
        decode_workspace.find("rmsnorm_final_output");
    const aima::NativeDecodeWorkspaceView* full_logits =
        decode_workspace.find("certified_lm_head_logits_output");
    constexpr std::size_t kVocabulary = 248320;
    if (final_hidden == nullptr || final_hidden->device_pointer == nullptr ||
        final_hidden->payload_bytes < kHidden * sizeof(std::uint16_t) ||
        full_logits == nullptr || full_logits->device_pointer == nullptr ||
        full_logits->payload_bytes < kVocabulary * sizeof(float)) {
      throw std::runtime_error(
          "native first-token distribution bindings are missing");
    }
    first_token_hidden_comparison = aima::compare_native_oracle_tensor(
        "first_token_final_hidden", "bfloat16",
        final_hidden->device_pointer, kHidden * sizeof(std::uint16_t),
        aima::find_native_oracle_tensor_file(
            state_oracle_dir, "return-final_logits-final_hidden"));
    first_token_logits_comparison = aima::compare_native_logits_fp32(
        full_logits->device_pointer, kVocabulary,
        aima::find_native_oracle_tensor_file(
            state_oracle_dir, "return-final_logits-output"));
    first_token_distribution_comparison_provided = true;
  }
  constexpr double kFirstTokenKldThreshold = 0.005;
  const bool first_token_distribution_gate_passed =
      !first_token_distribution_comparison_provided ||
      (first_token_logits_comparison.finite_elements ==
           first_token_logits_comparison.elements &&
       first_token_logits_comparison.top1_match &&
       first_token_logits_comparison.actual_top1_token_id ==
           first_token.top1_token_id &&
       first_token_logits_comparison.kl_divergence <
           kFirstTokenKldThreshold);

  aima::NativeDecodePrepareMetrics continuation_prepare;
  aima::NativeDecodeRunMetrics next_token;
  bool next_token_produced = false;
  double execution_request_wall_ms = 0.0;
  if (start_layer == 0 && through_layer == 39 && first_token_produced) {
    continuation_prepare = aima::prepare_native_decode_step(
        kTokens, first_token.top1_token_id, weights, decode_invocations);
    next_token = aima::run_native_decode_token(
        kTokens, kTokens + 1, weights, lm_head, decode_workspace,
        decode_invocations, executor, attention_state,
        properties.multiProcessorCount);
    next_token_produced = true;
    if (execution_only) {
      execution_request_wall_ms = elapsed_ms(request_started);
    }
  }

  aima::NativeOracleComparison next_token_hidden_comparison;
  aima::NativeLogitsComparison next_token_logits_comparison;
  bool next_token_distribution_comparison_provided = false;
  if (!decode_logits_oracle_dir.empty()) {
    if (!next_token_produced) {
      throw std::runtime_error(
          "decode logits comparison requires a complete token2 continuation");
    }
    const aima::NativeDecodeWorkspaceView* final_hidden =
        decode_workspace.find("rmsnorm_final_output");
    const aima::NativeDecodeWorkspaceView* full_logits =
        decode_workspace.find("certified_lm_head_logits_output");
    constexpr std::size_t kVocabulary = 248320;
    if (final_hidden == nullptr || final_hidden->device_pointer == nullptr ||
        final_hidden->payload_bytes < kHidden * sizeof(std::uint16_t) ||
        full_logits == nullptr || full_logits->device_pointer == nullptr ||
        full_logits->payload_bytes < kVocabulary * sizeof(float)) {
      throw std::runtime_error(
          "native token2 distribution bindings are missing");
    }
    next_token_hidden_comparison = aima::compare_native_oracle_tensor(
        "next_token_final_hidden", "bfloat16",
        final_hidden->device_pointer, kHidden * sizeof(std::uint16_t),
        aima::find_native_oracle_tensor_file(
            decode_logits_oracle_dir,
            decode_logits_oracle_label_prefix + "-final_hidden"));
    next_token_logits_comparison = aima::compare_native_logits_fp32(
        full_logits->device_pointer, kVocabulary,
        aima::find_native_oracle_tensor_file(
            decode_logits_oracle_dir,
            decode_logits_oracle_label_prefix + "-output"));
    next_token_distribution_comparison_provided = true;
  }
  const bool next_token_distribution_gate_passed =
      !next_token_distribution_comparison_provided ||
      (next_token_logits_comparison.finite_elements ==
           next_token_logits_comparison.elements &&
       next_token_logits_comparison.top1_match &&
       next_token_logits_comparison.actual_top1_token_id ==
           next_token.top1_token_id &&
       next_token_logits_comparison.kl_divergence <
           kFirstTokenKldThreshold);

  std::vector<aima::NativeOracleComparison> continuation_comparisons;
  bool continuation_all_finite = true;
  std::size_t continuation_router_layers_exact = 0;
  double continuation_final_hidden_relative_l2_error = 0.0;
  double continuation_final_hidden_cosine_similarity = 1.0;
  double continuation_worst_recurrent_relative_l2_error = 0.0;
  double continuation_minimum_recurrent_cosine_similarity = 1.0;
  double continuation_worst_conv_relative_l2_error = 0.0;
  double continuation_minimum_conv_cosine_similarity = 1.0;
  double continuation_worst_kv_relative_l2_error = 0.0;
  double continuation_minimum_kv_cosine_similarity = 1.0;
  if (!decode_oracle_dir.empty()) {
    if (!next_token_produced) {
      throw std::runtime_error(
          "decode boundary comparison requires a complete layer-0-to-39 continuation");
    }
    const auto compare = [&](const std::string& label,
                             const std::string& dtype,
                             const void* pointer, std::size_t bytes,
                             const std::string& oracle_label) {
      continuation_comparisons.push_back(
          aima::compare_native_oracle_tensor(
              label, dtype, pointer, bytes,
              aima::find_native_oracle_tensor_file(
                  decode_oracle_dir, oracle_label)));
      return continuation_comparisons.back();
    };
    const aima::NativeOracleComparison final_hidden = compare(
        "continuation_final_hidden", "bfloat16",
        decode_invocations.tensor_pointer(400, "x"),
        kHidden * sizeof(std::uint16_t),
        "layer-039-return-layer_body-output");
    continuation_final_hidden_relative_l2_error =
        final_hidden.relative_l2_error;
    continuation_final_hidden_cosine_similarity =
        final_hidden.cosine_similarity;
    constexpr std::size_t kKvCacheTokenBytes =
        2 * 256 * sizeof(std::uint16_t);
    const std::size_t kv_cache_bytes =
        (kTokens + 1) * kKvCacheTokenBytes;
    for (std::size_t layer_index = 0; layer_index < 40; ++layer_index) {
      const std::size_t base = layer_index * 10;
      std::ostringstream prefix;
      prefix << "layer-" << std::setw(3) << std::setfill('0')
             << layer_index << '-';
      const aima::NativeOracleComparison router = compare(
          "continuation_router_ids_layer_" + std::to_string(layer_index),
          "int32", decode_invocations.tensor_pointer(base + 7, "out_ids"),
          8 * sizeof(std::int32_t), prefix.str() + "launch-007-out_ids");
      continuation_router_layers_exact +=
          router.exact_elements == router.elements ? 1 : 0;
      const bool full_attention = layer_index % 4 == 3;
      if (full_attention) {
        for (const auto& cache :
             std::array<std::pair<const char*, const void*>, 2>{
                 std::pair<const char*, const void*>{
                     "k", attention_state.k_cache(layer_index)},
                 std::pair<const char*, const void*>{
                     "v", attention_state.v_cache(layer_index)}}) {
          const aima::NativeOracleComparison value = compare(
              "continuation_" + std::string(cache.first) +
                  "_cache_layer_" + std::to_string(layer_index),
              "bfloat16", cache.second, kv_cache_bytes,
              prefix.str() + "return-full_attention-" + cache.first +
                  "_cache");
          continuation_worst_kv_relative_l2_error = std::max(
              continuation_worst_kv_relative_l2_error,
              value.relative_l2_error);
          continuation_minimum_kv_cosine_similarity = std::min(
              continuation_minimum_kv_cosine_similarity,
              value.cosine_similarity);
        }
        continue;
      }
      const aima::NativeOracleComparison recurrent = compare(
          "continuation_recurrent_state_layer_" +
              std::to_string(layer_index),
          "float32", decode_invocations.tensor_pointer(base + 2, "h0"),
          32 * 128 * 128 * sizeof(float),
          prefix.str() + "launch-002-ht");
      continuation_worst_recurrent_relative_l2_error = std::max(
          continuation_worst_recurrent_relative_l2_error,
          recurrent.relative_l2_error);
      continuation_minimum_recurrent_cosine_similarity = std::min(
          continuation_minimum_recurrent_cosine_similarity,
          recurrent.cosine_similarity);
      const aima::NativeOracleComparison conv = compare(
          "continuation_conv_state_layer_" + std::to_string(layer_index),
          "bfloat16", decode_invocations.tensor_pointer(base + 1, "state_in"),
          8192 * 3 * sizeof(std::uint16_t),
          prefix.str() + "launch-001-state_out");
      continuation_worst_conv_relative_l2_error = std::max(
          continuation_worst_conv_relative_l2_error,
          conv.relative_l2_error);
      continuation_minimum_conv_cosine_similarity = std::min(
          continuation_minimum_conv_cosine_similarity,
          conv.cosine_similarity);
    }
    for (const auto& comparison : continuation_comparisons) {
      continuation_all_finite =
          continuation_all_finite &&
          comparison.finite_elements == comparison.elements;
    }
  }

  const bool entry_input_qualified =
      have_native_input_token_ids
          ? (entry_native_input_tokens == kTokens &&
             entry_native_embedding_launches == 1 &&
             (!entry_input_comparison_provided ||
              (entry_input_comparison.exact_elements ==
                   entry_input_comparison.elements &&
               entry_input_comparison.finite_elements ==
                   entry_input_comparison.elements)))
          : entry_seed_bytes == kHiddenBytes;
  const bool operation_closure_qualified =
      all_layers_finite && entry_input_qualified &&
      production_aot_launches == linear_layers * 13 + full_layers * 4 &&
      production_dense_gemm_launches ==
          linear_layers * 10 + full_layers * 9 &&
      production_native_pointwise_launches ==
          linear_layers * 6 + full_layers * 8 &&
      production_ck_fmha_launches == full_layers &&
      production_resident_state_direct_bindings ==
          2 * (linear_layers + full_layers);
  const bool first_token_gate_passed =
      !have_expected_top1_token_id ||
      (first_token_produced && first_token.certified &&
       first_token.top1_token_id == expected_top1_token_id);
  const bool next_token_gate_passed =
      !have_expected_next_token_id ||
      (next_token_produced && next_token.lm_head_certified &&
       next_token.top1_token_id == expected_next_token_id);
  const bool continuation_boundary_gate_passed =
      decode_oracle_dir.empty() ||
      (continuation_all_finite && continuation_router_layers_exact == 40 &&
       continuation_final_hidden_relative_l2_error < 0.02 &&
       continuation_final_hidden_cosine_similarity > 0.999);
  const bool continuation_acceptance_gate_passed =
      next_token_distribution_comparison_provided
          ? next_token_distribution_gate_passed
          : continuation_boundary_gate_passed;
  const bool execution_complete =
      operation_closure_qualified && first_token_produced &&
      first_token.certified && next_token_produced &&
      next_token.lm_head_certified;
  const bool qualified = !execution_only &&
      operation_closure_qualified && first_token_gate_passed &&
      first_token_distribution_gate_passed && next_token_gate_passed &&
      continuation_acceptance_gate_passed;
  const bool success = execution_only ? execution_complete : qualified;
  const std::size_t oracle_tensor_reads =
      (have_native_input_token_ids ? 0 : 1) +
      static_cast<std::size_t>(entry_input_comparison_provided) +
      layer_output_comparison_count + prefill_state_comparisons.size() +
      (first_token_distribution_comparison_provided ? 2 : 0) +
      (next_token_distribution_comparison_provided ? 2 : 0) +
      continuation_comparisons.size();

  std::cout << std::setprecision(17)
            << "{\n"
            << "  \"schema\": \"aima-amd395-qwen36/native-prefill-all-layers-oracle-probe/v4\",\n"
            << "  \"complete\": true,\n"
            << "  \"execution_only\": "
            << (execution_only ? "true" : "false") << ",\n"
            << "  \"execution_complete\": "
            << (execution_complete ? "true" : "false") << ",\n"
            << "  \"qualified\": "
            << (qualified ? "true" : "false") << ",\n"
            << "  \"qualification_scope\": \""
            << (execution_only
                    ? "oracle-free-product-execution-no-correctness-claim"
                    : "operation-closure-resident-state-handoff-first-token-full-distribution-and-token2-boundaries")
            << "\",\n"
            << "  \"correctness_claim\": false,\n"
            << "  \"performance_claim\": false,\n"
            << "  \"oracle_tensor_reads\": " << oracle_tensor_reads
            << ",\n"
            << "  \"requires_end_to_end_logits_gate\": "
            << (first_token_distribution_comparison_provided &&
                        next_token_distribution_comparison_provided
                    ? "false"
                    : "true")
            << ",\n"
            << "  \"requires_multi_prompt_correctness_gate\": true,\n"
            << "  \"runtime_python\": false,\n"
            << "  \"runtime_torch\": false,\n"
            << "  \"runtime_vllm\": false,\n"
            << "  \"runtime_triton\": false,\n"
            << "  \"tokens\": 8192,\n"
            << "  \"start_layer\": " << start_layer << ",\n"
            << "  \"through_layer\": " << through_layer << ",\n"
            << "  \"layers\": " << layer_outputs.size() << ",\n"
            << "  \"linear_layers\": " << linear_layers << ",\n"
            << "  \"full_attention_layers\": " << full_layers << ",\n"
            << "  \"entry_input_source\": \""
            << (have_native_input_token_ids
                    ? "native-resident-token-embedding"
                    : "qualification-oracle-seed")
            << "\",\n"
            << "  \"entry_input_qualified\": "
            << (entry_input_qualified ? "true" : "false") << ",\n"
            << "  \"entry_input_comparison_provided\": "
            << (entry_input_comparison_provided ? "true" : "false")
            << ",\n"
            << "  \"entry_uniform_input_token_id\": "
            << (have_uniform_input_token_id
                    ? std::to_string(uniform_input_token_id)
                    : "null")
            << ",\n"
            << "  \"entry_input_token_id_cycle_length\": "
            << input_token_id_cycle.size() << ",\n"
            << "  \"entry_native_input_tokens\": "
            << entry_native_input_tokens << ",\n"
            << "  \"entry_native_embedding_launches\": "
            << entry_native_embedding_launches << ",\n"
            << "  \"entry_native_embedding_exact_elements\": "
            << (entry_input_comparison_provided
                    ? std::to_string(entry_input_comparison.exact_elements)
                    : "null")
            << ",\n"
            << "  \"entry_native_embedding_elements\": "
            << (entry_input_comparison_provided
                    ? std::to_string(entry_input_comparison.elements)
                    : "null")
            << ",\n"
            << "  \"entry_oracle_seed_tensors\": "
            << (have_native_input_token_ids ? 0 : 1) << ",\n"
            << "  \"entry_oracle_seed_bytes\": " << entry_seed_bytes
            << ",\n"
            << "  \"mid_layer_oracle_seed_tensors\": 0,\n"
            << "  \"all_layers_finite\": "
            << (all_layers_finite ? "true" : "false") << ",\n"
            << "  \"production_aot_launches\": "
            << production_aot_launches << ",\n"
            << "  \"production_dense_gemm_launches\": "
            << production_dense_gemm_launches << ",\n"
            << "  \"production_native_pointwise_launches\": "
            << production_native_pointwise_launches << ",\n"
            << "  \"production_ck_fmha_launches\": "
            << production_ck_fmha_launches << ",\n"
            << "  \"production_resident_state_direct_bindings\": "
            << production_resident_state_direct_bindings << ",\n"
            << "  \"production_resident_state_payload_bytes\": "
            << production_resident_state_payload_bytes << ",\n"
            << "  \"first_token_produced\": "
            << (first_token_produced ? "true" : "false") << ",\n"
            << "  \"lm_head_certified\": "
            << (first_token.certified ? "true" : "false") << ",\n"
            << "  \"lm_head_candidate_count\": "
            << first_token.candidate_count << ",\n"
            << "  \"top1_token_id\": "
            << first_token.top1_token_id << ",\n"
            << "  \"top1_logit\": "
            << json_number(first_token.top1_logit) << ",\n"
            << "  \"expected_top1_token_id\": "
            << (have_expected_top1_token_id
                    ? std::to_string(expected_top1_token_id)
                    : "null")
            << ",\n"
            << "  \"first_token_gate_passed\": "
            << (first_token_gate_passed ? "true" : "false") << ",\n"
            << "  \"first_token_full_distribution_representation\": \"resident-int8-global-with-exact-certificate-candidate-scatter\",\n"
            << "  \"first_token_distribution_comparison_provided\": "
            << (first_token_distribution_comparison_provided ? "true"
                                                             : "false")
            << ",\n"
            << "  \"first_token_reference_top1_token_id\": "
            << (first_token_distribution_comparison_provided
                    ? std::to_string(
                          first_token_logits_comparison
                              .reference_top1_token_id)
                    : "null")
            << ",\n"
            << "  \"first_token_full_logits_top1_token_id\": "
            << (first_token_distribution_comparison_provided
                    ? std::to_string(
                          first_token_logits_comparison.actual_top1_token_id)
                    : "null")
            << ",\n"
            << "  \"first_token_full_logits_top1_match\": "
            << (first_token_distribution_comparison_provided
                    ? (first_token_logits_comparison.top1_match ? "true"
                                                                : "false")
                    : "null")
            << ",\n"
            << "  \"first_token_full_logits_elements\": "
            << (first_token_distribution_comparison_provided
                    ? std::to_string(first_token_logits_comparison.elements)
                    : "null")
            << ",\n"
            << "  \"first_token_full_logits_finite_elements\": "
            << (first_token_distribution_comparison_provided
                    ? std::to_string(
                          first_token_logits_comparison.finite_elements)
                    : "null")
            << ",\n"
            << "  \"first_token_full_logits_exact_elements\": "
            << (first_token_distribution_comparison_provided
                    ? std::to_string(
                          first_token_logits_comparison.exact_elements)
                    : "null")
            << ",\n"
            << "  \"first_token_final_hidden_relative_l2_error\": "
            << (first_token_distribution_comparison_provided
                    ? json_number(
                          first_token_hidden_comparison.relative_l2_error)
                    : "null")
            << ",\n"
            << "  \"first_token_final_hidden_cosine_similarity\": "
            << (first_token_distribution_comparison_provided
                    ? json_number(
                          first_token_hidden_comparison.cosine_similarity)
                    : "null")
            << ",\n"
            << "  \"first_token_full_logits_relative_l2_error\": "
            << (first_token_distribution_comparison_provided
                    ? json_number(
                          first_token_logits_comparison.relative_l2_error)
                    : "null")
            << ",\n"
            << "  \"first_token_full_logits_kl_divergence\": "
            << (first_token_distribution_comparison_provided
                    ? json_number(
                          first_token_logits_comparison.kl_divergence)
                    : "null")
            << ",\n"
            << "  \"first_token_full_logits_kl_divergence_threshold\": "
            << json_number(kFirstTokenKldThreshold) << ",\n"
            << "  \"first_token_distribution_gate_passed\": "
            << (first_token_distribution_gate_passed ? "true" : "false")
            << ",\n"
            << "  \"next_token_produced\": "
            << (next_token_produced ? "true" : "false") << ",\n"
            << "  \"continuation_prepare_launches\": "
            << continuation_prepare.native_kernel_launches << ",\n"
            << "  \"next_token_lm_head_certified\": "
            << (next_token.lm_head_certified ? "true" : "false") << ",\n"
            << "  \"next_token_id\": " << next_token.top1_token_id << ",\n"
            << "  \"next_token_logit\": "
            << json_number(next_token.top1_logit) << ",\n"
            << "  \"expected_next_token_id\": "
            << (have_expected_next_token_id
                    ? std::to_string(expected_next_token_id)
                    : "null")
            << ",\n"
            << "  \"next_token_gate_passed\": "
            << (next_token_gate_passed ? "true" : "false") << ",\n"
            << "  \"next_token_decode_wall_ms\": "
            << json_number(next_token.synchronized_wall_ms) << ",\n"
            << "  \"next_token_distribution_comparison_provided\": "
            << (next_token_distribution_comparison_provided ? "true"
                                                            : "false")
            << ",\n"
            << "  \"next_token_reference_top1_token_id\": "
            << (next_token_distribution_comparison_provided
                    ? std::to_string(
                          next_token_logits_comparison
                              .reference_top1_token_id)
                    : "null")
            << ",\n"
            << "  \"next_token_full_logits_top1_token_id\": "
            << (next_token_distribution_comparison_provided
                    ? std::to_string(
                          next_token_logits_comparison.actual_top1_token_id)
                    : "null")
            << ",\n"
            << "  \"next_token_full_logits_top1_match\": "
            << (next_token_distribution_comparison_provided
                    ? (next_token_logits_comparison.top1_match ? "true"
                                                               : "false")
                    : "null")
            << ",\n"
            << "  \"next_token_full_logits_finite_elements\": "
            << (next_token_distribution_comparison_provided
                    ? std::to_string(
                          next_token_logits_comparison.finite_elements)
                    : "null")
            << ",\n"
            << "  \"next_token_final_hidden_relative_l2_error\": "
            << (next_token_distribution_comparison_provided
                    ? json_number(
                          next_token_hidden_comparison.relative_l2_error)
                    : "null")
            << ",\n"
            << "  \"next_token_final_hidden_cosine_similarity\": "
            << (next_token_distribution_comparison_provided
                    ? json_number(
                          next_token_hidden_comparison.cosine_similarity)
                    : "null")
            << ",\n"
            << "  \"next_token_full_logits_relative_l2_error\": "
            << (next_token_distribution_comparison_provided
                    ? json_number(
                          next_token_logits_comparison.relative_l2_error)
                    : "null")
            << ",\n"
            << "  \"next_token_full_logits_kl_divergence\": "
            << (next_token_distribution_comparison_provided
                    ? json_number(
                          next_token_logits_comparison.kl_divergence)
                    : "null")
            << ",\n"
            << "  \"next_token_full_logits_kl_divergence_threshold\": "
            << json_number(kFirstTokenKldThreshold) << ",\n"
            << "  \"next_token_distribution_gate_passed\": "
            << (next_token_distribution_gate_passed ? "true" : "false")
            << ",\n"
            << "  \"prefill_state_oracle_scheme\": \""
            << (coherent_prefill_state_oracle
                    ? "same-request-official-prefill"
                    : (state_oracle_dir.empty() ? "none"
                                                : "legacy-decode-fixture"))
            << "\",\n"
            << "  \"prefill_state_comparison_count\": "
            << prefill_state_comparisons.size() << ",\n"
            << "  \"prefill_state_exact_tensors\": "
            << prefill_state_exact_tensors << ",\n"
            << "  \"prefill_state_all_finite\": "
            << (prefill_state_all_finite ? "true" : "false") << ",\n"
            << "  \"prefill_state_worst_recurrent_relative_l2_error\": "
            << json_number(
                   prefill_state_worst_recurrent_relative_l2_error)
            << ",\n"
            << "  \"prefill_state_worst_recurrent_layer\": "
            << prefill_state_worst_recurrent_layer << ",\n"
            << "  \"prefill_state_worst_conv_relative_l2_error\": "
            << json_number(prefill_state_worst_conv_relative_l2_error)
            << ",\n"
            << "  \"prefill_state_worst_conv_layer\": "
            << prefill_state_worst_conv_layer << ",\n"
            << "  \"prefill_state_worst_kv_relative_l2_error\": "
            << json_number(prefill_state_worst_kv_relative_l2_error)
            << ",\n"
            << "  \"prefill_state_worst_kv_layer\": "
            << prefill_state_worst_kv_layer << ",\n"
            << "  \"prefill_state_worst_kv_kind\": \""
            << json_escape(prefill_state_worst_kv_kind) << "\",\n"
            << "  \"continuation_boundary_comparison_count\": "
            << continuation_comparisons.size() << ",\n"
            << "  \"continuation_all_finite\": "
            << (continuation_all_finite ? "true" : "false") << ",\n"
            << "  \"continuation_router_layers_exact\": "
            << continuation_router_layers_exact << ",\n"
            << "  \"continuation_final_hidden_relative_l2_error\": "
            << json_number(continuation_final_hidden_relative_l2_error)
            << ",\n"
            << "  \"continuation_final_hidden_cosine_similarity\": "
            << json_number(continuation_final_hidden_cosine_similarity)
            << ",\n"
            << "  \"continuation_worst_recurrent_relative_l2_error\": "
            << json_number(continuation_worst_recurrent_relative_l2_error)
            << ",\n"
            << "  \"continuation_minimum_recurrent_cosine_similarity\": "
            << json_number(continuation_minimum_recurrent_cosine_similarity)
            << ",\n"
            << "  \"continuation_worst_conv_relative_l2_error\": "
            << json_number(continuation_worst_conv_relative_l2_error)
            << ",\n"
            << "  \"continuation_minimum_conv_cosine_similarity\": "
            << json_number(continuation_minimum_conv_cosine_similarity)
            << ",\n"
            << "  \"continuation_worst_kv_relative_l2_error\": "
            << json_number(continuation_worst_kv_relative_l2_error) << ",\n"
            << "  \"continuation_minimum_kv_cosine_similarity\": "
            << json_number(continuation_minimum_kv_cosine_similarity)
            << ",\n"
            << "  \"continuation_boundary_gate_passed\": "
            << (continuation_boundary_gate_passed ? "true" : "false")
            << ",\n"
            << "  \"continuation_acceptance_basis\": \""
            << (execution_only
                    ? "none-execution-only"
                    : (next_token_distribution_comparison_provided
                           ? "full-vocabulary-kld-and-top1"
                           : "internal-boundary-diagnostic"))
            << "\",\n"
            << "  \"continuation_acceptance_gate_passed\": "
            << (continuation_acceptance_gate_passed ? "true" : "false")
            << ",\n"
            << "  \"maximum_layer_output_relative_l2_error\": "
            << (layer_output_comparison_count == 0
                    ? "null"
                    : json_number(maximum_layer_output_relative_l2_error))
            << ",\n"
            << "  \"minimum_layer_output_cosine_similarity\": "
            << (layer_output_comparison_count == 0
                    ? "null"
                    : json_number(minimum_layer_output_cosine_similarity))
            << ",\n"
            << "  \"layer_output_comparison_count\": "
            << layer_output_comparison_count << ",\n"
            << "  \"final_hidden_relative_l2_error\": "
            << (layer_outputs.back().comparison_provided
                    ? json_number(
                          layer_outputs.back().output.relative_l2_error)
                    : "null")
            << ",\n"
            << "  \"final_hidden_cosine_similarity\": "
            << (layer_outputs.back().comparison_provided
                    ? json_number(
                          layer_outputs.back().output.cosine_similarity)
                    : "null")
            << ",\n"
            << "  \"raw_load_wall_ms\": " << load.load_wall_ms << ",\n"
            << "  \"layer_derived_build_wall_ms\": "
            << derived_metrics.build_wall_ms << ",\n"
            << "  \"lm_head_build_wall_ms\": "
            << lm_head_metrics.build_wall_ms << ",\n"
            << "  \"decode_weight_bindings\": "
            << binding_metrics.unique_bindings << ",\n"
            << "  \"prefill_workspace_bytes\": "
            << workspace_metrics.allocation_bytes << ",\n"
            << "  \"prefill_workspace_runtime_scratch_bindings\": "
            << workspace_metrics.runtime_scratch_bindings << ",\n"
            << "  \"prefill_workspace_runtime_scratch_payload_bytes\": "
            << workspace_metrics.runtime_scratch_payload_bytes << ",\n"
            << "  \"prefill_prepared_launches\": "
            << invocation_metrics.launch_count << ",\n"
            << "  \"prefill_resident_gemm_plans\": "
            << prefill_gemm_plans.built_plan_count() << ",\n"
            << "  \"prefill_resident_gemm_workspace_bytes\": "
            << prefill_gemm_plans.workspace_bytes() << ",\n"
            << "  \"prefill_gemm_plan_build_wall_ms\": "
            << prefill_gemm_plan_build_wall_ms << ",\n"
            << "  \"resident_prefill_wall_ms\": "
            << (execution_only ? json_number(execution_prefill_wall_ms)
                               : "null")
            << ",\n"
            << "  \"resident_prefill_tokens_per_second\": "
            << (execution_only && execution_prefill_wall_ms > 0.0
                    ? json_number(8192.0 * 1000.0 /
                                  execution_prefill_wall_ms)
                    : "null")
            << ",\n"
            << "  \"resident_request_wall_ms\": "
            << (execution_only ? json_number(execution_request_wall_ms)
                               : "null")
            << ",\n"
            << "  \"decode_workspace_bytes\": "
            << decode_workspace_metrics.allocation_bytes << ",\n"
            << "  \"decode_prepared_launches\": "
            << decode_invocation_metrics.launch_count << ",\n"
            << "  \"attention_state_allocation_bytes\": "
            << attention_state_metrics.allocation_bytes << ",\n"
            << "  \"aot_loaded_modules\": "
            << executor_load.loaded_modules << ",\n"
            << "  \"ck_provider_loaded\": "
            << (provider_load.loaded ? "true" : "false") << ",\n"
            << "  \"total_wall_ms\": " << elapsed_ms(started) << ",\n"
            << "  \"layer_outputs\": [\n";
  for (std::size_t index = 0; index < layer_outputs.size(); ++index) {
    const auto& layer = layer_outputs[index];
    std::cout << "    {\"layer_index\":" << layer.layer_index
              << ",\"attention_type\":\""
              << (layer.full_attention ? "full" : "linear") << "\""
              << ",\"comparison_provided\":"
              << (layer.comparison_provided ? "true" : "false")
              << ",\"finite_elements\":"
              << (layer.comparison_provided
                      ? std::to_string(layer.output.finite_elements)
                      : "null")
              << ",\"elements\":"
              << (layer.comparison_provided
                      ? std::to_string(layer.output.elements)
                      : "null")
              << ",\"exact_elements\":"
              << (layer.comparison_provided
                      ? std::to_string(layer.output.exact_elements)
                      : "null")
              << ",\"relative_l2_error\":"
              << (layer.comparison_provided
                      ? json_number(layer.output.relative_l2_error)
                      : "null")
              << ",\"cosine_similarity\":"
              << (layer.comparison_provided
                      ? json_number(layer.output.cosine_similarity)
                      : "null")
              << "}"
              << (index + 1 == layer_outputs.size() ? "\n" : ",\n");
  }
  std::cout << "  ]\n}\n";
  return success ? 0 : 3;
}

int run_linear_layer_oracle_probe(int argc, char** argv) {
  aima::NativeWeightLoadOptions options;
  options.native_report =
      std::filesystem::absolute("native-linear-layer-oracle-weight-load.json");
  std::filesystem::path oracle_dir;
  bool have_model_dir = false;
  bool have_oracle_dir = false;
  bool all_linear_layers = false;
  for (int index = 2; index < argc; ++index) {
    const std::string argument = argv[index];
    auto next = [&](const char* name) -> std::string {
      if (++index >= argc) {
        throw std::runtime_error(std::string(name) + " requires a value");
      }
      return argv[index];
    };
    if (argument == "--model-dir") {
      options.model_dir = std::filesystem::absolute(next("--model-dir"));
      have_model_dir = true;
    } else if (argument == "--oracle-dir") {
      oracle_dir = std::filesystem::absolute(next("--oracle-dir"));
      have_oracle_dir = true;
    } else if (argument == "--all-linear-layers") {
      all_linear_layers = true;
    } else if (argument == "--report") {
      options.native_report = std::filesystem::absolute(next("--report"));
    } else if (argument == "--device") {
      options.device = parse_int(next("--device"), "--device");
    } else if (argument == "--workers") {
      options.worker_count = parse_size(next("--workers"), "--workers");
    } else if (argument == "--chunk-bytes") {
      options.chunk_bytes = parse_size(next("--chunk-bytes"), "--chunk-bytes");
    } else {
      throw std::runtime_error("unknown linear-layer oracle argument: " +
                               argument);
    }
  }
  if (!have_model_dir || !have_oracle_dir) {
    throw std::runtime_error("--model-dir and --oracle-dir are required");
  }

  const auto started = std::chrono::steady_clock::now();
  aima::NativeWeightStore weights;
  const aima::NativeWeightLoadMetrics load = weights.load(options);
  aima::NativeDerivedWeightStore derived;
  const aima::NativeDerivedWeightMetrics derived_metrics =
      derived.build(weights, options.device);
  aima::NativeLmHeadStore lm_head;
  const aima::NativeLmHeadMetrics lm_head_metrics =
      lm_head.build(weights, options.device);
  aima::NativeDecodeBindings bindings;
  const aima::NativeDecodeBindingMetrics binding_metrics =
      bindings.build(weights, derived, lm_head);
  aima::NativeDecodeWorkspace workspace;
  const aima::NativeDecodeWorkspaceMetrics workspace_metrics =
      workspace.build(options.device);
  aima::NativeDecodeInvocations invocations;
  const aima::NativeDecodeInvocationMetrics invocation_metrics =
      invocations.build(bindings, workspace);
  aima::NativeDecodeExecutor executor;
  const aima::NativeDecodeExecutorMetrics executor_load = executor.load();
  hipDeviceProp_t properties{};
  if (hipGetDeviceProperties(&properties, options.device) != hipSuccess) {
    throw std::runtime_error("hipGetDeviceProperties failed for linear-layer oracle");
  }
  if (all_linear_layers) {
    std::size_t schedule_count = 0;
    const aima::DecodeLaunch* schedule =
        aima::native_decode_schedule(&schedule_count);
    std::vector<aima::NativeLinearLayerOracleResult> results;
    bool all_qualified = true;
    std::size_t total_comparisons = 0;
    std::size_t exact_comparisons = 0;
    std::size_t total_aot_launches = 0;
    double layer_wall_ms = 0.0;
    for (std::size_t layer_index = 0; layer_index < 40; ++layer_index) {
      const std::size_t base = layer_index * 10;
      if (base + 10 >= schedule_count ||
          std::string(schedule[base + 1].symbol) !=
              "triton_fused_input_proj_conv_kernel") {
        continue;
      }
      results.push_back(aima::probe_native_linear_layer_oracle(
          oracle_dir, layer_index, weights, workspace, invocations, executor,
          properties.multiProcessorCount));
      const auto& value = results.back();
      const bool layer_qualified =
          value.all_finite && value.router_ids_exact &&
          value.aot_boundaries_exact && value.final_relative_l2_error < 0.02 &&
          value.final_cosine_similarity > 0.999;
      all_qualified = all_qualified && layer_qualified;
      total_aot_launches += value.layer.aot_launches;
      layer_wall_ms += value.layer.wall_ms;
      total_comparisons += value.comparisons.size();
      for (const auto& comparison : value.comparisons) {
        exact_comparisons +=
            comparison.exact_elements == comparison.elements ? 1 : 0;
      }
    }
    const bool complete = results.size() == 30 && total_aot_launches == 240;
    const bool qualified = complete && all_qualified &&
                           exact_comparisons == total_comparisons;
    std::cout << std::setprecision(17)
              << "{\n"
              << "  \"schema\": \"aima-amd395-qwen36/native-linear-layers-oracle-probe/v1\",\n"
              << "  \"complete\": " << (complete ? "true" : "false") << ",\n"
              << "  \"qualified\": " << (qualified ? "true" : "false") << ",\n"
              << "  \"runtime_python\": false,\n"
              << "  \"runtime_torch\": false,\n"
              << "  \"runtime_vllm\": false,\n"
              << "  \"runtime_triton\": false,\n"
              << "  \"raw_load_wall_ms\": " << load.load_wall_ms << ",\n"
              << "  \"layer_derived_build_wall_ms\": "
              << derived_metrics.build_wall_ms << ",\n"
              << "  \"lm_head_build_wall_ms\": "
              << lm_head_metrics.build_wall_ms << ",\n"
              << "  \"aot_loaded_modules\": " << executor_load.loaded_modules
              << ",\n"
              << "  \"linear_layer_count\": " << results.size() << ",\n"
              << "  \"total_aot_launches\": " << total_aot_launches << ",\n"
              << "  \"total_comparisons\": " << total_comparisons << ",\n"
              << "  \"exact_comparisons\": " << exact_comparisons << ",\n"
              << "  \"linear_layers_wall_ms\": " << layer_wall_ms << ",\n"
              << "  \"total_wall_ms\": " << elapsed_ms(started) << ",\n"
              << "  \"layers\": [\n";
    for (std::size_t index = 0; index < results.size(); ++index) {
      const auto& value = results[index];
      std::size_t layer_exact = 0;
      for (const auto& comparison : value.comparisons) {
        layer_exact += comparison.exact_elements == comparison.elements ? 1 : 0;
      }
      std::cout << "    {\"layer_index\":" << value.layer.layer_index
                << ",\"wall_ms\":" << value.layer.wall_ms
                << ",\"aot_boundaries_exact\":"
                << (value.aot_boundaries_exact ? "true" : "false")
                << ",\"router_ids_exact\":"
                << (value.router_ids_exact ? "true" : "false")
                << ",\"comparisons\":" << value.comparisons.size()
                << ",\"exact_comparisons\":" << layer_exact
                << ",\"final_relative_l2_error\":"
                << value.final_relative_l2_error
                << ",\"final_cosine_similarity\":"
                << value.final_cosine_similarity << "}"
                << (index + 1 == results.size() ? "\n" : ",\n");
    }
    std::cout << "  ]\n}\n";
    return qualified ? 0 : 3;
  }
  const aima::NativeLinearLayerOracleResult oracle =
      aima::probe_native_linear_layer_oracle(
          oracle_dir, 0, weights, workspace, invocations, executor,
          properties.multiProcessorCount);
  const double total_wall_ms = elapsed_ms(started);
  const bool qualified = oracle.all_finite && oracle.router_ids_exact &&
                         oracle.aot_boundaries_exact &&
                         oracle.final_relative_l2_error < 0.02 &&
                         oracle.final_cosine_similarity > 0.999;
  std::cout << std::setprecision(17)
            << "{\n"
            << "  \"schema\": \"aima-amd395-qwen36/native-linear-layer-oracle-probe/v1\",\n"
            << "  \"complete\": true,\n"
            << "  \"qualified\": " << (qualified ? "true" : "false") << ",\n"
            << "  \"runtime_python\": false,\n"
            << "  \"runtime_torch\": false,\n"
            << "  \"runtime_vllm\": false,\n"
            << "  \"runtime_triton\": false,\n"
            << "  \"raw_load_wall_ms\": " << load.load_wall_ms << ",\n"
            << "  \"layer_derived_build_wall_ms\": "
            << derived_metrics.build_wall_ms << ",\n"
            << "  \"lm_head_build_wall_ms\": "
            << lm_head_metrics.build_wall_ms << ",\n"
            << "  \"workspace_allocation_and_zero_ms\": "
            << workspace_metrics.allocation_and_zero_ms << ",\n"
            << "  \"prepared_launches\": " << invocation_metrics.launch_count
            << ",\n"
            << "  \"weight_bindings\": " << binding_metrics.unique_bindings
            << ",\n"
            << "  \"aot_loaded_modules\": " << executor_load.loaded_modules
            << ",\n"
            << "  \"aot_module_load_wall_ms\": "
            << executor_load.module_load_wall_ms << ",\n"
            << "  \"seed_tensors\": " << oracle.seed_tensors << ",\n"
            << "  \"seed_bytes\": " << oracle.seed_bytes << ",\n"
            << "  \"layer_wall_ms\": " << oracle.layer.wall_ms << ",\n"
            << "  \"aot_launches\": " << oracle.layer.aot_launches << ",\n"
            << "  \"native_projection_launches\": "
            << oracle.layer.native_projection_launches << ",\n"
            << "  \"native_pointwise_launches\": "
            << oracle.layer.native_pointwise_launches << ",\n"
            << "  \"all_finite\": "
            << (oracle.all_finite ? "true" : "false") << ",\n"
            << "  \"router_ids_exact\": "
            << (oracle.router_ids_exact ? "true" : "false") << ",\n"
            << "  \"aot_boundaries_exact\": "
            << (oracle.aot_boundaries_exact ? "true" : "false") << ",\n"
            << "  \"final_relative_l2_error\": "
            << oracle.final_relative_l2_error << ",\n"
            << "  \"final_cosine_similarity\": "
            << oracle.final_cosine_similarity << ",\n"
            << "  \"total_wall_ms\": " << total_wall_ms << ",\n"
            << "  \"comparisons\": [\n";
  for (std::size_t index = 0; index < oracle.comparisons.size(); ++index) {
    const auto& value = oracle.comparisons[index];
    std::cout << "    {\"label\":\"" << json_escape(value.label)
              << "\",\"dtype\":\"" << value.dtype
              << "\",\"elements\":" << value.elements
              << ",\"exact_elements\":" << value.exact_elements
              << ",\"finite_elements\":" << value.finite_elements
              << ",\"maximum_absolute_error\":"
              << value.maximum_absolute_error
              << ",\"relative_l2_error\":" << value.relative_l2_error
              << ",\"cosine_similarity\":" << value.cosine_similarity
              << ",\"expected_sha256\":\"" << value.expected_sha256
              << "\",\"actual_sha256\":\"" << value.actual_sha256 << "\"}"
              << (index + 1 == oracle.comparisons.size() ? "\n" : ",\n");
  }
  std::cout << "  ]\n}\n";
  return qualified ? 0 : 3;
}

int run_full_attention_core_oracle_probe(int argc, char** argv) {
  std::filesystem::path oracle_dir;
  bool have_oracle_dir = false;
  std::size_t layer_index = 3;
  std::size_t cache_end = 8193;
  int device = 0;
  for (int index = 2; index < argc; ++index) {
    const std::string argument = argv[index];
    auto next = [&](const char* name) -> std::string {
      if (++index >= argc) {
        throw std::runtime_error(std::string(name) + " requires a value");
      }
      return argv[index];
    };
    if (argument == "--oracle-dir") {
      oracle_dir = std::filesystem::absolute(next("--oracle-dir"));
      have_oracle_dir = true;
    } else if (argument == "--layer") {
      layer_index = parse_size(next("--layer"), "--layer");
    } else if (argument == "--cache-end") {
      cache_end = parse_size(next("--cache-end"), "--cache-end");
    } else if (argument == "--device") {
      device = parse_int(next("--device"), "--device");
    } else {
      throw std::runtime_error(
          "unknown full-attention core oracle argument: " + argument);
    }
  }
  if (!have_oracle_dir) throw std::runtime_error("--oracle-dir is required");
  const auto started = std::chrono::steady_clock::now();
  aima::NativeFullAttentionState state;
  const aima::NativeFullAttentionStateMetrics state_metrics =
      state.build(cache_end, device);
  const aima::NativeFullAttentionCoreOracleResult oracle =
      aima::probe_native_full_attention_core_oracle(
          oracle_dir, layer_index, cache_end, state);
  const bool qualified =
      oracle.all_finite && oracle.kv_cache_exact &&
      oracle.attention_relative_l2_error < 0.02 &&
      oracle.attention_cosine_similarity > 0.999;
  std::cout << std::setprecision(17)
            << "{\n"
            << "  \"schema\": \"aima-amd395-qwen36/native-full-attention-core-oracle-probe/v1\",\n"
            << "  \"complete\": true,\n"
            << "  \"qualified\": " << (qualified ? "true" : "false") << ",\n"
            << "  \"runtime_python\": false,\n"
            << "  \"runtime_torch\": false,\n"
            << "  \"runtime_vllm\": false,\n"
            << "  \"runtime_triton\": false,\n"
            << "  \"layer_index\": " << layer_index << ",\n"
            << "  \"cache_end\": " << cache_end << ",\n"
            << "  \"state_allocation_bytes\": "
            << state_metrics.allocation_bytes << ",\n"
            << "  \"state_allocation_and_zero_ms\": "
            << state_metrics.allocation_and_zero_ms << ",\n"
            << "  \"seed_tensors\": " << oracle.seed_tensors << ",\n"
            << "  \"seed_bytes\": " << oracle.seed_bytes << ",\n"
            << "  \"native_kernel_launches\": "
            << oracle.core.native_kernel_launches << ",\n"
            << "  \"pv_splits\": " << oracle.core.pv_splits << ",\n"
            << "  \"all_finite\": "
            << (oracle.all_finite ? "true" : "false") << ",\n"
            << "  \"kv_cache_exact\": "
            << (oracle.kv_cache_exact ? "true" : "false") << ",\n"
            << "  \"scores_relative_l2_error\": "
            << oracle.scores_relative_l2_error << ",\n"
            << "  \"probabilities_relative_l2_error\": "
            << oracle.probabilities_relative_l2_error << ",\n"
            << "  \"attention_relative_l2_error\": "
            << oracle.attention_relative_l2_error << ",\n"
            << "  \"attention_cosine_similarity\": "
            << oracle.attention_cosine_similarity << ",\n"
            << "  \"total_wall_ms\": " << elapsed_ms(started) << ",\n"
            << "  \"comparisons\": [\n";
  for (std::size_t index = 0; index < oracle.comparisons.size(); ++index) {
    const auto& value = oracle.comparisons[index];
    std::cout << "    {\"label\":\"" << json_escape(value.label)
              << "\",\"dtype\":\"" << value.dtype
              << "\",\"elements\":" << value.elements
              << ",\"exact_elements\":" << value.exact_elements
              << ",\"finite_elements\":" << value.finite_elements
              << ",\"maximum_absolute_error\":"
              << value.maximum_absolute_error
              << ",\"relative_l2_error\":" << value.relative_l2_error
              << ",\"cosine_similarity\":" << value.cosine_similarity
              << ",\"expected_sha256\":\"" << value.expected_sha256
              << "\",\"actual_sha256\":\"" << value.actual_sha256 << "\"}"
              << (index + 1 == oracle.comparisons.size() ? "\n" : ",\n");
  }
  std::cout << "  ]\n}\n";
  return qualified ? 0 : 3;
}

int run_full_layer_oracle_probe(int argc, char** argv) {
  aima::NativeWeightLoadOptions options;
  options.native_report =
      std::filesystem::absolute("native-full-layer-oracle-weight-load.json");
  std::filesystem::path oracle_dir;
  bool have_model_dir = false;
  bool have_oracle_dir = false;
  bool all_full_layers = false;
  std::size_t layer_index = 3;
  std::size_t cache_end = 8193;
  for (int index = 2; index < argc; ++index) {
    const std::string argument = argv[index];
    auto next = [&](const char* name) -> std::string {
      if (++index >= argc) {
        throw std::runtime_error(std::string(name) + " requires a value");
      }
      return argv[index];
    };
    if (argument == "--model-dir") {
      options.model_dir = std::filesystem::absolute(next("--model-dir"));
      have_model_dir = true;
    } else if (argument == "--oracle-dir") {
      oracle_dir = std::filesystem::absolute(next("--oracle-dir"));
      have_oracle_dir = true;
    } else if (argument == "--layer") {
      layer_index = parse_size(next("--layer"), "--layer");
    } else if (argument == "--all-full-layers") {
      all_full_layers = true;
    } else if (argument == "--cache-end") {
      cache_end = parse_size(next("--cache-end"), "--cache-end");
    } else if (argument == "--report") {
      options.native_report = std::filesystem::absolute(next("--report"));
    } else if (argument == "--device") {
      options.device = parse_int(next("--device"), "--device");
    } else if (argument == "--workers") {
      options.worker_count = parse_size(next("--workers"), "--workers");
    } else if (argument == "--chunk-bytes") {
      options.chunk_bytes = parse_size(next("--chunk-bytes"), "--chunk-bytes");
    } else {
      throw std::runtime_error("unknown full-layer oracle argument: " +
                               argument);
    }
  }
  if (!have_model_dir || !have_oracle_dir) {
    throw std::runtime_error("--model-dir and --oracle-dir are required");
  }

  const auto started = std::chrono::steady_clock::now();
  aima::NativeWeightStore weights;
  const aima::NativeWeightLoadMetrics load = weights.load(options);
  aima::NativeDerivedWeightStore derived;
  const aima::NativeDerivedWeightMetrics derived_metrics =
      derived.build(weights, options.device);
  aima::NativeLmHeadStore lm_head;
  const aima::NativeLmHeadMetrics lm_head_metrics =
      lm_head.build(weights, options.device);
  aima::NativeDecodeBindings bindings;
  const aima::NativeDecodeBindingMetrics binding_metrics =
      bindings.build(weights, derived, lm_head);
  aima::NativeDecodeWorkspace workspace;
  const aima::NativeDecodeWorkspaceMetrics workspace_metrics =
      workspace.build(options.device);
  aima::NativeDecodeInvocations invocations;
  const aima::NativeDecodeInvocationMetrics invocation_metrics =
      invocations.build(bindings, workspace);
  aima::NativeDecodeExecutor executor;
  const aima::NativeDecodeExecutorMetrics executor_load = executor.load();
  aima::NativeFullAttentionState attention_state;
  const aima::NativeFullAttentionStateMetrics state_metrics =
      attention_state.build(cache_end, options.device);
  hipDeviceProp_t properties{};
  if (hipGetDeviceProperties(&properties, options.device) != hipSuccess) {
    throw std::runtime_error("hipGetDeviceProperties failed for full-layer oracle");
  }
  const auto layer_qualified = [](const aima::NativeFullLayerOracleResult& value) {
    return value.all_finite && value.kv_cache_exact &&
           value.router_ids_exact && value.pre_attention_aot_exact &&
           value.final_relative_l2_error < 0.02 &&
           value.final_cosine_similarity > 0.999;
  };
  if (all_full_layers) {
    std::vector<aima::NativeFullLayerOracleResult> results;
    bool qualified = true;
    std::size_t total_comparisons = 0;
    std::size_t exact_comparisons = 0;
    std::size_t total_aot_launches = 0;
    std::size_t total_attention_launches = 0;
    double layer_wall_ms = 0.0;
    for (std::size_t current = 3; current < 40; current += 4) {
      results.push_back(aima::probe_native_full_layer_oracle(
          oracle_dir, current, cache_end, weights, workspace, invocations,
          executor, attention_state, properties.multiProcessorCount));
      const auto& value = results.back();
      qualified = qualified && layer_qualified(value);
      total_aot_launches += value.layer.aot_launches;
      total_attention_launches += value.layer.native_attention_launches;
      layer_wall_ms += value.layer.wall_ms;
      total_comparisons += value.comparisons.size();
      for (const auto& comparison : value.comparisons) {
        exact_comparisons +=
            comparison.exact_elements == comparison.elements ? 1 : 0;
      }
    }
    const bool complete = results.size() == 10 && total_aot_launches == 80 &&
                          total_attention_launches == 40;
    qualified = qualified && complete;
    std::cout << std::setprecision(17)
              << "{\n"
              << "  \"schema\": \"aima-amd395-qwen36/native-full-layers-oracle-probe/v1\",\n"
              << "  \"complete\": " << (complete ? "true" : "false") << ",\n"
              << "  \"qualified\": " << (qualified ? "true" : "false") << ",\n"
              << "  \"runtime_python\": false,\n"
              << "  \"runtime_torch\": false,\n"
              << "  \"runtime_vllm\": false,\n"
              << "  \"runtime_triton\": false,\n"
              << "  \"raw_load_wall_ms\": " << load.load_wall_ms << ",\n"
              << "  \"layer_derived_build_wall_ms\": "
              << derived_metrics.build_wall_ms << ",\n"
              << "  \"lm_head_build_wall_ms\": "
              << lm_head_metrics.build_wall_ms << ",\n"
              << "  \"state_allocation_bytes\": "
              << state_metrics.allocation_bytes << ",\n"
              << "  \"aot_loaded_modules\": " << executor_load.loaded_modules
              << ",\n"
              << "  \"full_layer_count\": " << results.size() << ",\n"
              << "  \"total_aot_launches\": " << total_aot_launches << ",\n"
              << "  \"total_attention_launches\": "
              << total_attention_launches << ",\n"
              << "  \"total_comparisons\": " << total_comparisons << ",\n"
              << "  \"exact_comparisons\": " << exact_comparisons << ",\n"
              << "  \"full_layers_wall_ms\": " << layer_wall_ms << ",\n"
              << "  \"total_wall_ms\": " << elapsed_ms(started) << ",\n"
              << "  \"layers\": [\n";
    for (std::size_t index = 0; index < results.size(); ++index) {
      const auto& value = results[index];
      std::size_t exact = 0;
      for (const auto& comparison : value.comparisons) {
        exact += comparison.exact_elements == comparison.elements ? 1 : 0;
      }
      std::cout << "    {\"layer_index\":" << value.layer.layer_index
                << ",\"qualified\":"
                << (layer_qualified(value) ? "true" : "false")
                << ",\"wall_ms\":" << value.layer.wall_ms
                << ",\"pv_splits\":" << value.layer.pv_splits
                << ",\"kv_cache_exact\":"
                << (value.kv_cache_exact ? "true" : "false")
                << ",\"router_ids_exact\":"
                << (value.router_ids_exact ? "true" : "false")
                << ",\"comparisons\":" << value.comparisons.size()
                << ",\"exact_comparisons\":" << exact
                << ",\"attention_relative_l2_error\":"
                << value.attention_relative_l2_error
                << ",\"projected_attention_relative_l2_error\":"
                << value.projected_attention_relative_l2_error
                << ",\"final_relative_l2_error\":"
                << value.final_relative_l2_error
                << ",\"final_cosine_similarity\":"
                << value.final_cosine_similarity << "}"
                << (index + 1 == results.size() ? "\n" : ",\n");
    }
    std::cout << "  ]\n}\n";
    return qualified ? 0 : 3;
  }

  const aima::NativeFullLayerOracleResult oracle =
      aima::probe_native_full_layer_oracle(
          oracle_dir, layer_index, cache_end, weights, workspace, invocations,
          executor, attention_state, properties.multiProcessorCount);
  const bool qualified = layer_qualified(oracle);
  std::cout << std::setprecision(17)
            << "{\n"
            << "  \"schema\": \"aima-amd395-qwen36/native-full-layer-oracle-probe/v1\",\n"
            << "  \"complete\": true,\n"
            << "  \"qualified\": " << (qualified ? "true" : "false") << ",\n"
            << "  \"runtime_python\": false,\n"
            << "  \"runtime_torch\": false,\n"
            << "  \"runtime_vllm\": false,\n"
            << "  \"runtime_triton\": false,\n"
            << "  \"raw_load_wall_ms\": " << load.load_wall_ms << ",\n"
            << "  \"layer_derived_build_wall_ms\": "
            << derived_metrics.build_wall_ms << ",\n"
            << "  \"lm_head_build_wall_ms\": "
            << lm_head_metrics.build_wall_ms << ",\n"
            << "  \"workspace_allocation_and_zero_ms\": "
            << workspace_metrics.allocation_and_zero_ms << ",\n"
            << "  \"state_allocation_bytes\": "
            << state_metrics.allocation_bytes << ",\n"
            << "  \"prepared_launches\": " << invocation_metrics.launch_count
            << ",\n"
            << "  \"weight_bindings\": " << binding_metrics.unique_bindings
            << ",\n"
            << "  \"aot_loaded_modules\": " << executor_load.loaded_modules
            << ",\n"
            << "  \"layer_index\": " << oracle.layer.layer_index << ",\n"
            << "  \"layer_wall_ms\": " << oracle.layer.wall_ms << ",\n"
            << "  \"aot_launches\": " << oracle.layer.aot_launches << ",\n"
            << "  \"native_attention_launches\": "
            << oracle.layer.native_attention_launches << ",\n"
            << "  \"native_projection_launches\": "
            << oracle.layer.native_projection_launches << ",\n"
            << "  \"native_pointwise_launches\": "
            << oracle.layer.native_pointwise_launches << ",\n"
            << "  \"pv_splits\": " << oracle.layer.pv_splits << ",\n"
            << "  \"seed_tensors\": " << oracle.seed_tensors << ",\n"
            << "  \"seed_bytes\": " << oracle.seed_bytes << ",\n"
            << "  \"all_finite\": "
            << (oracle.all_finite ? "true" : "false") << ",\n"
            << "  \"kv_cache_exact\": "
            << (oracle.kv_cache_exact ? "true" : "false") << ",\n"
            << "  \"router_ids_exact\": "
            << (oracle.router_ids_exact ? "true" : "false") << ",\n"
            << "  \"pre_attention_aot_exact\": "
            << (oracle.pre_attention_aot_exact ? "true" : "false") << ",\n"
            << "  \"attention_relative_l2_error\": "
            << oracle.attention_relative_l2_error << ",\n"
            << "  \"projected_attention_relative_l2_error\": "
            << oracle.projected_attention_relative_l2_error << ",\n"
            << "  \"final_relative_l2_error\": "
            << oracle.final_relative_l2_error << ",\n"
            << "  \"final_cosine_similarity\": "
            << oracle.final_cosine_similarity << ",\n"
            << "  \"total_wall_ms\": " << elapsed_ms(started) << ",\n"
            << "  \"comparisons\": [\n";
  for (std::size_t index = 0; index < oracle.comparisons.size(); ++index) {
    const auto& value = oracle.comparisons[index];
    std::cout << "    {\"label\":\"" << json_escape(value.label)
              << "\",\"dtype\":\"" << value.dtype
              << "\",\"elements\":" << value.elements
              << ",\"exact_elements\":" << value.exact_elements
              << ",\"finite_elements\":" << value.finite_elements
              << ",\"maximum_absolute_error\":"
              << value.maximum_absolute_error
              << ",\"relative_l2_error\":" << value.relative_l2_error
              << ",\"cosine_similarity\":" << value.cosine_similarity
              << ",\"expected_sha256\":\"" << value.expected_sha256
              << "\",\"actual_sha256\":\"" << value.actual_sha256 << "\"}"
              << (index + 1 == oracle.comparisons.size() ? "\n" : ",\n");
  }
  std::cout << "  ]\n}\n";
  return qualified ? 0 : 3;
}

int run_decode_oracle_probe(int argc, char** argv) {
  aima::NativeWeightLoadOptions options;
  options.native_report =
      std::filesystem::absolute("native-decode-oracle-weight-load.json");
  std::filesystem::path oracle_dir;
  bool have_model_dir = false;
  bool have_oracle_dir = false;
  std::size_t cache_end = 8193;
  std::size_t warmup_runs = 0;
  std::size_t benchmark_runs = 1;
  std::size_t sequence_steps = 1;
  bool compact = false;
  for (int index = 2; index < argc; ++index) {
    const std::string argument = argv[index];
    auto next = [&](const char* name) -> std::string {
      if (++index >= argc) {
        throw std::runtime_error(std::string(name) + " requires a value");
      }
      return argv[index];
    };
    if (argument == "--model-dir") {
      options.model_dir = std::filesystem::absolute(next("--model-dir"));
      have_model_dir = true;
    } else if (argument == "--oracle-dir") {
      oracle_dir = std::filesystem::absolute(next("--oracle-dir"));
      have_oracle_dir = true;
    } else if (argument == "--cache-end") {
      cache_end = parse_size(next("--cache-end"), "--cache-end");
    } else if (argument == "--warmup-runs") {
      warmup_runs = parse_size(next("--warmup-runs"), "--warmup-runs");
    } else if (argument == "--benchmark-runs") {
      benchmark_runs =
          parse_size(next("--benchmark-runs"), "--benchmark-runs");
    } else if (argument == "--sequence-steps") {
      sequence_steps =
          parse_size(next("--sequence-steps"), "--sequence-steps");
    } else if (argument == "--compact") {
      compact = true;
    } else if (argument == "--report") {
      options.native_report = std::filesystem::absolute(next("--report"));
    } else if (argument == "--device") {
      options.device = parse_int(next("--device"), "--device");
    } else if (argument == "--workers") {
      options.worker_count = parse_size(next("--workers"), "--workers");
    } else if (argument == "--chunk-bytes") {
      options.chunk_bytes = parse_size(next("--chunk-bytes"), "--chunk-bytes");
    } else {
      throw std::runtime_error("unknown decode oracle argument: " + argument);
    }
  }
  if (!have_model_dir || !have_oracle_dir) {
    throw std::runtime_error("--model-dir and --oracle-dir are required");
  }
  if (sequence_steps > 1 && benchmark_runs != 1) {
    throw std::runtime_error(
        "--sequence-steps requires exactly one measured run");
  }
  const auto started = std::chrono::steady_clock::now();
  aima::NativeWeightStore weights;
  const aima::NativeWeightLoadMetrics load = weights.load(options);
  aima::NativeDerivedWeightStore derived;
  const aima::NativeDerivedWeightMetrics derived_metrics =
      derived.build(weights, options.device);
  aima::NativeLmHeadStore lm_head;
  const aima::NativeLmHeadMetrics lm_head_metrics =
      lm_head.build(weights, options.device);
  aima::NativeDecodeBindings bindings;
  const aima::NativeDecodeBindingMetrics binding_metrics =
      bindings.build(weights, derived, lm_head);
  aima::NativeDecodeWorkspace workspace;
  const aima::NativeDecodeWorkspaceMetrics workspace_metrics =
      workspace.build(options.device);
  aima::NativeDecodeInvocations invocations;
  const aima::NativeDecodeInvocationMetrics invocation_metrics =
      invocations.build(bindings, workspace);
  aima::NativeDecodeExecutor executor;
  const aima::NativeDecodeExecutorMetrics executor_load = executor.load();
  aima::NativeFullAttentionState attention_state;
  const aima::NativeFullAttentionStateMetrics state_metrics =
      attention_state.build(cache_end + sequence_steps - 1, options.device);
  hipDeviceProp_t properties{};
  if (hipGetDeviceProperties(&properties, options.device) != hipSuccess) {
    throw std::runtime_error("hipGetDeviceProperties failed for decode oracle");
  }
  const aima::NativeDecodeOracleResult oracle =
      aima::probe_native_decode_oracle(
          oracle_dir, cache_end, weights, lm_head, workspace, invocations,
          executor, attention_state, properties.multiProcessorCount,
          {warmup_runs, benchmark_runs});
  std::vector<aima::NativeDecodeRunMetrics> sequence_decodes;
  sequence_decodes.reserve(sequence_steps);
  sequence_decodes.push_back(oracle.decode);
  std::size_t sequence_prepare_launches = 0;
  for (std::size_t step = 1; step < sequence_steps; ++step) {
    const std::size_t position = cache_end + step - 1;
    const aima::NativeDecodePrepareMetrics prepared =
        aima::prepare_native_decode_step(
            position, sequence_decodes.back().top1_token_id, weights,
            invocations);
    sequence_prepare_launches += prepared.native_kernel_launches;
    sequence_decodes.push_back(aima::run_native_decode_token(
        position, position + 1, weights, lm_head, workspace, invocations,
        executor, attention_state, properties.multiProcessorCount));
  }
  std::ostringstream sequence_token_payload;
  for (std::size_t index = 0; index < sequence_decodes.size(); ++index) {
    if (index != 0) sequence_token_payload << ',';
    sequence_token_payload << sequence_decodes[index].top1_token_id;
  }
  const std::string sequence_token_payload_value = sequence_token_payload.str();
  const std::string sequence_token_ids_sha256 = aima::sha256_bytes(
      sequence_token_payload_value.data(), sequence_token_payload_value.size());
  std::vector<double> sequence_wall_ms;
  sequence_wall_ms.reserve(sequence_decodes.size());
  double sequence_total_wall_ms = 0.0;
  for (const auto& run : sequence_decodes) {
    sequence_wall_ms.push_back(run.synchronized_wall_ms);
    sequence_total_wall_ms += run.synchronized_wall_ms;
  }
  std::vector<double> sorted_sequence_wall_ms = sequence_wall_ms;
  std::sort(sorted_sequence_wall_ms.begin(), sorted_sequence_wall_ms.end());
  const double sequence_median_wall_ms =
      sorted_sequence_wall_ms.size() % 2 == 0
          ? (sorted_sequence_wall_ms[sorted_sequence_wall_ms.size() / 2 - 1] +
             sorted_sequence_wall_ms[sorted_sequence_wall_ms.size() / 2]) /
                2.0
          : sorted_sequence_wall_ms[sorted_sequence_wall_ms.size() / 2];
  const double sequence_min_wall_ms = sorted_sequence_wall_ms.front();
  const double sequence_max_wall_ms = sorted_sequence_wall_ms.back();
  std::vector<double> benchmark_wall_ms;
  benchmark_wall_ms.reserve(oracle.measured_decodes.size());
  for (const auto& run : oracle.measured_decodes) {
    benchmark_wall_ms.push_back(run.synchronized_wall_ms);
  }
  std::vector<double> sorted_wall_ms = benchmark_wall_ms;
  std::sort(sorted_wall_ms.begin(), sorted_wall_ms.end());
  const double benchmark_median_wall_ms =
      sorted_wall_ms.size() % 2 == 0
          ? (sorted_wall_ms[sorted_wall_ms.size() / 2 - 1] +
             sorted_wall_ms[sorted_wall_ms.size() / 2]) /
                2.0
          : sorted_wall_ms[sorted_wall_ms.size() / 2];
  const double benchmark_min_wall_ms = sorted_wall_ms.front();
  const double benchmark_max_wall_ms = sorted_wall_ms.back();
  double worst_recurrent_relative_l2 = 0.0;
  double minimum_recurrent_cosine = 1.0;
  std::vector<std::size_t> router_mismatches;
  for (const auto& comparison : oracle.comparisons) {
    if (comparison.label.rfind("recurrent_state_layer_", 0) == 0) {
      worst_recurrent_relative_l2 =
          std::max(worst_recurrent_relative_l2,
                   comparison.relative_l2_error);
      minimum_recurrent_cosine =
          std::min(minimum_recurrent_cosine, comparison.cosine_similarity);
    } else if (comparison.label.rfind("router_ids_layer_", 0) == 0 &&
               comparison.exact_elements != comparison.elements) {
      router_mismatches.push_back(parse_size(
          comparison.label.substr(std::string("router_ids_layer_").size()),
          "router mismatch layer"));
    }
  }
  constexpr std::uint32_t kFixtureExpectedToken = 1000;
  const bool qualified =
      oracle.all_finite && oracle.router_layers_exact == 40 &&
      oracle.final_hidden_relative_l2_error < 0.02 &&
      oracle.final_hidden_cosine_similarity > 0.999 &&
      std::all_of(oracle.measured_decodes.begin(),
                  oracle.measured_decodes.end(), [](const auto& run) {
                    return run.lm_head_certified &&
                           run.top1_token_id == kFixtureExpectedToken;
                  }) &&
      std::all_of(sequence_decodes.begin(), sequence_decodes.end(),
                  [](const auto& run) {
                    return run.lm_head_certified &&
                           run.top1_token_id == kFixtureExpectedToken;
                  });
  std::cout << std::setprecision(17)
            << "{\n"
            << "  \"schema\": \"aima-amd395-qwen36/native-decode-oracle-probe/v1\",\n"
            << "  \"complete\": true,\n"
            << "  \"qualified\": " << (qualified ? "true" : "false") << ",\n"
            << "  \"runtime_python\": false,\n"
            << "  \"runtime_torch\": false,\n"
            << "  \"runtime_vllm\": false,\n"
            << "  \"runtime_triton\": false,\n"
            << "  \"cache_end\": " << cache_end << ",\n"
            << "  \"raw_load_wall_ms\": " << load.load_wall_ms << ",\n"
            << "  \"layer_derived_build_wall_ms\": "
            << derived_metrics.build_wall_ms << ",\n"
            << "  \"lm_head_build_wall_ms\": "
            << lm_head_metrics.build_wall_ms << ",\n"
            << "  \"workspace_allocation_bytes\": "
            << workspace_metrics.allocation_bytes << ",\n"
            << "  \"attention_state_allocation_bytes\": "
            << state_metrics.allocation_bytes << ",\n"
            << "  \"prepared_launches\": " << invocation_metrics.launch_count
            << ",\n"
            << "  \"weight_bindings\": " << binding_metrics.unique_bindings
            << ",\n"
            << "  \"aot_loaded_modules\": " << executor_load.loaded_modules
            << ",\n"
            << "  \"seed_tensors\": " << oracle.seed_tensors << ",\n"
            << "  \"seed_bytes\": " << oracle.seed_bytes << ",\n"
            << "  \"layer_count\": " << oracle.decode.layer_count << ",\n"
            << "  \"linear_layer_count\": "
            << oracle.decode.linear_layer_count << ",\n"
            << "  \"full_layer_count\": " << oracle.decode.full_layer_count
            << ",\n"
            << "  \"aot_launches\": " << oracle.decode.aot_launches << ",\n"
            << "  \"native_attention_launches\": "
            << oracle.decode.native_attention_launches << ",\n"
            << "  \"native_projection_launches\": "
            << oracle.decode.native_projection_launches << ",\n"
            << "  \"native_pointwise_launches\": "
            << oracle.decode.native_pointwise_launches << ",\n"
            << "  \"resident_state_pointer_swaps\": "
            << oracle.decode.resident_state_pointer_swaps << ",\n"
            << "  \"native_lm_head_certificate_launches\": "
            << oracle.decode.native_lm_head_certificate_launches << ",\n"
            << "  \"lm_head_candidate_count\": "
            << oracle.decode.lm_head_candidate_count << ",\n"
            << "  \"lm_head_certified\": "
            << (oracle.decode.lm_head_certified ? "true" : "false") << ",\n"
            << "  \"sequence_steps\": " << sequence_decodes.size() << ",\n"
            << "  \"sequence_prepare_launches\": "
            << sequence_prepare_launches << ",\n"
            << "  \"sequence_token_ids\": ";
  if (compact) {
    std::cout << "null";
  } else {
    std::cout << '[';
    for (std::size_t index = 0; index < sequence_decodes.size(); ++index) {
      if (index != 0) std::cout << ',';
      std::cout << sequence_decodes[index].top1_token_id;
    }
    std::cout << ']';
  }
  std::cout << ",\n"
            << "  \"sequence_token_ids_sha256\": \""
            << sequence_token_ids_sha256 << "\",\n"
            << "  \"sequence_total_decode_wall_ms\": "
            << sequence_total_wall_ms << ",\n"
            << "  \"sequence_decode_tokens_per_s\": "
            << 1000.0 * static_cast<double>(sequence_decodes.size()) /
                   sequence_total_wall_ms
            << ",\n"
            << "  \"sequence_median_decode_wall_ms\": "
            << sequence_median_wall_ms << ",\n"
            << "  \"sequence_median_decode_tokens_per_s\": "
            << 1000.0 / sequence_median_wall_ms << ",\n"
            << "  \"sequence_min_decode_wall_ms\": "
            << sequence_min_wall_ms << ",\n"
            << "  \"sequence_max_decode_wall_ms\": "
            << sequence_max_wall_ms << ",\n"
            << "  \"sequence_spread_percent_of_median\": "
            << (sequence_max_wall_ms - sequence_min_wall_ms) /
                   sequence_median_wall_ms * 100.0
            << ",\n"
            << "  \"sequence_decode_wall_ms\": ";
  if (compact) {
    std::cout << "null";
  } else {
    std::cout << '[';
    for (std::size_t index = 0; index < sequence_decodes.size(); ++index) {
      if (index != 0) std::cout << ',';
      std::cout << sequence_decodes[index].synchronized_wall_ms;
    }
    std::cout << ']';
  }
  std::cout << ",\n"
            << "  \"warmup_runs\": " << oracle.warmup_decodes.size()
            << ",\n"
            << "  \"benchmark_runs\": " << oracle.measured_decodes.size()
            << ",\n"
            << "  \"warmup_decode_wall_ms\": [";
  for (std::size_t index = 0; index < oracle.warmup_decodes.size(); ++index) {
    if (index != 0) std::cout << ',';
    std::cout << oracle.warmup_decodes[index].synchronized_wall_ms;
  }
  std::cout << "],\n"
            << "  \"benchmark_decode_wall_ms\": [";
  for (std::size_t index = 0; index < benchmark_wall_ms.size(); ++index) {
    if (index != 0) std::cout << ',';
    std::cout << benchmark_wall_ms[index];
  }
  std::cout << "],\n"
            << "  \"benchmark_median_decode_wall_ms\": "
            << benchmark_median_wall_ms << ",\n"
            << "  \"benchmark_median_decode_tokens_per_s\": "
            << 1000.0 / benchmark_median_wall_ms << ",\n"
            << "  \"benchmark_min_decode_wall_ms\": "
            << benchmark_min_wall_ms << ",\n"
            << "  \"benchmark_max_decode_wall_ms\": "
            << benchmark_max_wall_ms << ",\n"
            << "  \"benchmark_spread_percent_of_median\": "
            << (benchmark_max_wall_ms - benchmark_min_wall_ms) /
                   benchmark_median_wall_ms * 100.0
            << ",\n"
            << "  \"layer_submission_ms\": "
            << oracle.decode.layer_submission_ms << ",\n"
            << "  \"decode_synchronized_wall_ms\": "
            << oracle.decode.synchronized_wall_ms << ",\n"
            << "  \"decode_tokens_per_s\": "
            << 1000.0 / oracle.decode.synchronized_wall_ms << ",\n"
            << "  \"top1_token_id\": "
            << oracle.decode.top1_token_id << ",\n"
            << "  \"fixture_expected_token_id\": "
            << kFixtureExpectedToken << ",\n"
            << "  \"top1_logit\": "
            << oracle.decode.top1_logit << ",\n"
            << "  \"all_finite\": "
            << (oracle.all_finite ? "true" : "false") << ",\n"
            << "  \"router_layers_exact\": " << oracle.router_layers_exact
            << ",\n"
            << "  \"recurrent_states_exact\": "
            << oracle.recurrent_states_exact << ",\n"
            << "  \"worst_recurrent_relative_l2_error\": "
            << worst_recurrent_relative_l2 << ",\n"
            << "  \"minimum_recurrent_cosine_similarity\": "
            << minimum_recurrent_cosine << ",\n"
            << "  \"final_hidden_relative_l2_error\": "
            << oracle.final_hidden_relative_l2_error << ",\n"
            << "  \"final_hidden_cosine_similarity\": "
            << oracle.final_hidden_cosine_similarity << ",\n"
            << "  \"comparison_count\": " << oracle.comparisons.size()
            << ",\n"
            << "  \"router_mismatch_layers\": [";
  for (std::size_t index = 0; index < router_mismatches.size(); ++index) {
    if (index != 0) std::cout << ',';
    std::cout << router_mismatches[index];
  }
  std::cout << "],\n"
            << "  \"total_wall_ms\": " << elapsed_ms(started) << "\n"
            << "}\n";
  return qualified ? 0 : 3;
}

int run_resident_session_probe(int argc, char** argv) {
  aima::NativeResidentEngineOptions options;
  options.weights.native_report =
      std::filesystem::absolute("native-resident-weight-load.json");
  bool have_model_dir = false;
  bool have_uniform_token = false;
  std::uint32_t uniform_token = 0;
  std::vector<std::uint32_t> input_cycle;
  std::vector<std::uint32_t> cached_suffix_token_ids;
  std::vector<std::uint32_t> expected_token_ids;
  std::vector<std::size_t> max_new_tokens_sequence;
  std::vector<std::vector<std::size_t>> secondary_fmha_layer_sets;
  std::filesystem::path reference_logits;
  std::filesystem::path layer_tail_oracle_dir;
  std::filesystem::path layer_sequence_oracle_dir;
  std::size_t layer_tail_oracle_index = 40;
  std::size_t request_count = 2;
  std::size_t max_new_tokens = 2;
  std::size_t request_prompt_tokens = 0;
  bool cache_capacity_explicit = false;
  bool request_count_explicit = false;
  bool max_new_tokens_explicit = false;
  bool disable_prefix_cache = false;

  const auto parse_ids = [](const std::string& value,
                            const char* name) {
    std::vector<std::uint32_t> result;
    std::size_t begin = 0;
    while (begin < value.size()) {
      const std::size_t end = value.find(',', begin);
      const std::string item = value.substr(
          begin, end == std::string::npos ? std::string::npos : end - begin);
      if (item.empty()) {
        throw std::runtime_error(std::string(name) +
                                 " contains an empty token id");
      }
      const int parsed = parse_int(item, name);
      if (parsed < 0 || parsed >= 248320) {
        throw std::runtime_error(std::string(name) +
                                 " contains an invalid token id");
      }
      result.push_back(static_cast<std::uint32_t>(parsed));
      if (end == std::string::npos) break;
      begin = end + 1;
    }
    return result;
  };
  const auto parse_sizes = [](const std::string& value,
                              const char* name) {
    std::vector<std::size_t> result;
    std::size_t begin = 0;
    while (begin < value.size()) {
      const std::size_t end = value.find(',', begin);
      const std::string item = value.substr(
          begin, end == std::string::npos ? std::string::npos : end - begin);
      if (item.empty()) {
        throw std::runtime_error(std::string(name) +
                                 " contains an empty length");
      }
      result.push_back(parse_size(item, name));
      if (end == std::string::npos) break;
      begin = end + 1;
    }
    return result;
  };
  const auto parse_layer_sets = [&](const std::string& value) {
    std::vector<std::vector<std::size_t>> result;
    std::size_t begin = 0;
    while (begin < value.size()) {
      const std::size_t end = value.find(';', begin);
      const std::string item = value.substr(
          begin, end == std::string::npos ? std::string::npos : end - begin);
      if (item.empty()) {
        throw std::runtime_error(
            "--secondary-fmha-layer-sets contains an empty set");
      }
      result.push_back(parse_sizes(item, "--secondary-fmha-layer-sets"));
      if (end == std::string::npos) break;
      begin = end + 1;
    }
    return result;
  };

  for (int index = 2; index < argc; ++index) {
    const std::string argument = argv[index];
    auto next = [&](const char* name) -> std::string {
      if (++index >= argc) {
        throw std::runtime_error(std::string(name) + " requires a value");
      }
      return argv[index];
    };
    if (argument == "--model-dir") {
      options.weights.model_dir =
          std::filesystem::absolute(next("--model-dir"));
      have_model_dir = true;
    } else if (argument == "--fmha-provider" ||
               argument == "--ck-provider") {
      options.ck_provider =
          std::filesystem::absolute(next("--fmha-provider"));
    } else if (argument == "--secondary-fmha-provider") {
      options.secondary_fmha_provider = std::filesystem::absolute(
          next("--secondary-fmha-provider"));
    } else if (argument == "--secondary-fmha-layers") {
      options.secondary_fmha_layers = parse_sizes(
          next("--secondary-fmha-layers"),
          "--secondary-fmha-layers");
    } else if (argument == "--secondary-fmha-layer-sets") {
      secondary_fmha_layer_sets = parse_layer_sets(
          next("--secondary-fmha-layer-sets"));
    } else if (argument == "--disable-prefix-cache") {
      disable_prefix_cache = true;
    } else if (argument == "--uniform-input-token-id") {
      const int parsed = parse_int(next("--uniform-input-token-id"),
                                   "--uniform-input-token-id");
      if (parsed < 0 || parsed >= 248320) {
        throw std::runtime_error("--uniform-input-token-id is invalid");
      }
      uniform_token = static_cast<std::uint32_t>(parsed);
      have_uniform_token = true;
    } else if (argument == "--input-token-id-cycle") {
      input_cycle = parse_ids(next("--input-token-id-cycle"),
                              "--input-token-id-cycle");
    } else if (argument == "--cached-suffix-token-ids") {
      cached_suffix_token_ids = parse_ids(
          next("--cached-suffix-token-ids"),
          "--cached-suffix-token-ids");
    } else if (argument == "--expected-token-ids") {
      expected_token_ids = parse_ids(next("--expected-token-ids"),
                                     "--expected-token-ids");
    } else if (argument == "--reference-logits") {
      reference_logits =
          std::filesystem::absolute(next("--reference-logits"));
    } else if (argument == "--layer-tail-oracle-dir") {
      layer_tail_oracle_dir =
          std::filesystem::absolute(next("--layer-tail-oracle-dir"));
    } else if (argument == "--layer-tail-oracle-index") {
      layer_tail_oracle_index = static_cast<std::size_t>(parse_int(
          next("--layer-tail-oracle-index"),
          "--layer-tail-oracle-index"));
    } else if (argument == "--layer-sequence-oracle-dir") {
      layer_sequence_oracle_dir = std::filesystem::absolute(
          next("--layer-sequence-oracle-dir"));
    } else if (argument == "--max-new-tokens") {
      max_new_tokens =
          parse_size(next("--max-new-tokens"), "--max-new-tokens");
      max_new_tokens_explicit = true;
    } else if (argument == "--max-new-tokens-sequence") {
      max_new_tokens_sequence = parse_sizes(
          next("--max-new-tokens-sequence"),
          "--max-new-tokens-sequence");
    } else if (argument == "--context-tokens") {
      options.prompt_tokens =
          parse_size(next("--context-tokens"), "--context-tokens");
    } else if (argument == "--prompt-tokens") {
      request_prompt_tokens =
          parse_size(next("--prompt-tokens"), "--prompt-tokens");
    } else if (argument == "--requests") {
      request_count = parse_size(next("--requests"), "--requests");
      request_count_explicit = true;
    } else if (argument == "--cache-capacity") {
      options.cache_capacity =
          parse_size(next("--cache-capacity"), "--cache-capacity");
      cache_capacity_explicit = true;
    } else if (argument == "--report") {
      options.weights.native_report =
          std::filesystem::absolute(next("--report"));
    } else if (argument == "--device") {
      options.weights.device = parse_int(next("--device"), "--device");
    } else if (argument == "--workers") {
      options.weights.worker_count =
          parse_size(next("--workers"), "--workers");
    } else if (argument == "--chunk-bytes") {
      options.weights.chunk_bytes =
          parse_size(next("--chunk-bytes"), "--chunk-bytes");
    } else {
      throw std::runtime_error(
          "unknown resident session argument: " + argument);
    }
  }
  if (!max_new_tokens_sequence.empty()) {
    if (request_count_explicit || max_new_tokens_explicit) {
      throw std::runtime_error(
          "--max-new-tokens-sequence cannot be combined with --requests or --max-new-tokens");
    }
    request_count = max_new_tokens_sequence.size();
    max_new_tokens = *std::max_element(max_new_tokens_sequence.begin(),
                                       max_new_tokens_sequence.end());
    if (!expected_token_ids.empty() || !reference_logits.empty() ||
        !cached_suffix_token_ids.empty()) {
      throw std::runtime_error(
          "variable output lengths are a performance probe and cannot be combined with correctness or prefix-extension options");
    }
  }
  if (!secondary_fmha_layer_sets.empty()) {
    if (request_count_explicit || !max_new_tokens_sequence.empty() ||
        !cached_suffix_token_ids.empty() ||
        !options.secondary_fmha_layers.empty()) {
      throw std::runtime_error(
          "--secondary-fmha-layer-sets cannot be combined with request count, variable outputs, prefix extension, or a fixed secondary layer set");
    }
    request_count = secondary_fmha_layer_sets.size();
    std::array<bool, 40> union_layers{};
    for (const auto& layer_set : secondary_fmha_layer_sets) {
      for (std::size_t layer : layer_set) {
        if (layer >= 40 || layer % 4 != 3) {
          throw std::runtime_error(
              "--secondary-fmha-layer-sets accepts only full-attention layers");
        }
        union_layers[layer] = true;
      }
    }
    for (std::size_t layer = 0; layer < union_layers.size(); ++layer) {
      if (union_layers[layer]) options.secondary_fmha_layers.push_back(layer);
    }
    disable_prefix_cache = true;
  }
  if (!have_model_dir || request_count == 0 ||
      max_new_tokens == 0 || have_uniform_token == !input_cycle.empty()) {
    throw std::runtime_error(
        "resident session requires a model, one input token source, and non-zero request/output counts");
  }
  if (!expected_token_ids.empty() &&
      expected_token_ids.size() != max_new_tokens) {
    throw std::runtime_error(
        "--expected-token-ids must match --max-new-tokens");
  }
  if (!reference_logits.empty() && max_new_tokens != 1) {
    throw std::runtime_error(
        "--reference-logits requires --max-new-tokens 1 so the compared buffer is the prefill distribution");
  }
  if (!cached_suffix_token_ids.empty() && request_count < 2) {
    throw std::runtime_error(
        "--cached-suffix-token-ids requires at least two requests");
  }
  if (layer_tail_oracle_index > 39 && layer_tail_oracle_index != 40) {
    throw std::runtime_error(
        "--layer-tail-oracle-index must be in 0..39");
  }
  if (!cache_capacity_explicit) {
    options.cache_capacity =
        options.prompt_tokens + cached_suffix_token_ids.size() +
        max_new_tokens;
  }
  if (request_prompt_tokens == 0) {
    request_prompt_tokens = options.prompt_tokens;
  }
  if (request_prompt_tokens > options.cache_capacity ||
      max_new_tokens > options.cache_capacity - request_prompt_tokens) {
    throw std::runtime_error(
        "--prompt-tokens plus output exceeds --cache-capacity");
  }
  options.prefix_cache_enabled = !disable_prefix_cache;

  std::vector<std::uint32_t> prompt(request_prompt_tokens);
  if (have_uniform_token) {
    std::fill(prompt.begin(), prompt.end(), uniform_token);
  } else {
    for (std::size_t index = 0; index < prompt.size(); ++index) {
      prompt[index] = input_cycle[index % input_cycle.size()];
    }
  }

  aima::NativeResidentEngine engine;
  const aima::NativeResidentLoadMetrics load = engine.load(options);
  std::vector<aima::NativeResidentRequestMetrics> requests;
  std::vector<aima::NativeLogitsComparison> request_logits_comparisons;
  requests.reserve(request_count);
  request_logits_comparisons.reserve(request_count);
  for (std::size_t index = 0; index < request_count; ++index) {
    aima::NativeResidentRequestOptions request;
    request.input_token_ids = prompt;
    if (index != 0) {
      request.input_token_ids.insert(request.input_token_ids.end(),
                                     cached_suffix_token_ids.begin(),
                                     cached_suffix_token_ids.end());
    }
    request.max_new_tokens = max_new_tokens_sequence.empty()
                                 ? max_new_tokens
                                 : max_new_tokens_sequence[index];
    request.layer_tail_oracle_dir = layer_tail_oracle_dir;
    request.layer_tail_oracle_index = layer_tail_oracle_index;
    request.layer_sequence_oracle_dir = layer_sequence_oracle_dir;
    request.disable_prefix_cache = disable_prefix_cache;
    if (!secondary_fmha_layer_sets.empty()) {
      request.secondary_fmha_layers_override_provided = true;
      request.secondary_fmha_layers_override =
          secondary_fmha_layer_sets[index];
    }
    requests.push_back(engine.run(request));
    if (!reference_logits.empty()) {
      request_logits_comparisons.push_back(
          engine.compare_current_logits(reference_logits));
    }
  }

  bool repeat_tokens_identical = true;
  bool expected_tokens_match = !expected_token_ids.empty();
  bool all_requests_complete = true;
  for (std::size_t index = 0; index < requests.size(); ++index) {
    const auto& request = requests[index];
    const std::size_t expected_completion_tokens =
        max_new_tokens_sequence.empty() ? max_new_tokens
                                        : max_new_tokens_sequence[index];
    repeat_tokens_identical =
        repeat_tokens_identical &&
        request.output_token_ids == requests.front().output_token_ids;
    if (!expected_token_ids.empty() &&
        (cached_suffix_token_ids.empty() ||
         &request == &requests.back())) {
      expected_tokens_match =
          expected_tokens_match &&
          request.output_token_ids == expected_token_ids;
    }
    all_requests_complete =
        all_requests_complete &&
        request.output_token_ids.size() == expected_completion_tokens &&
        request.first_token_certified && request.all_decode_tokens_certified &&
        (layer_tail_oracle_dir.empty() && layer_sequence_oracle_dir.empty()
             ? request.oracle_tensor_reads == 0
             : request.oracle_tensor_reads > 0) &&
        request.model_loads == 1;
  }
  aima::NativeLogitsComparison logits_comparison;
  const bool reference_logits_provided = !reference_logits.empty();
  if (reference_logits_provided) {
    logits_comparison = request_logits_comparisons.back();
  }
  const auto request_logits_qualified = [&](std::size_t index) {
    const auto& comparison = request_logits_comparisons[index];
    return comparison.elements == 248320 &&
           comparison.finite_elements == comparison.elements &&
           comparison.top1_match && comparison.kl_divergence < 0.005 &&
           !requests[index].output_token_ids.empty() &&
           comparison.actual_top1_token_id ==
               requests[index].output_token_ids.front();
  };
  bool logits_qualified = reference_logits_provided &&
                          request_logits_comparisons.size() == requests.size();
  if (logits_qualified) {
    for (std::size_t index = 0; index < requests.size(); ++index) {
      logits_qualified = logits_qualified && request_logits_qualified(index);
    }
  }
  const bool success =
      all_requests_complete &&
      (!max_new_tokens_sequence.empty() ||
       !cached_suffix_token_ids.empty() || repeat_tokens_identical) &&
      (expected_token_ids.empty() || expected_tokens_match) &&
      (!reference_logits_provided || logits_qualified);

  std::cout << std::setprecision(17)
            << "{\n"
            << "  \"schema\": \"aima-amd395-qwen36/native-resident-session-probe/v1\",\n"
            << "  \"complete\": " << (success ? "true" : "false")
            << ",\n"
            << "  \"qualified\": "
            << (reference_logits_provided && success ? "true" : "false")
            << ",\n"
            << "  \"correctness_claim\": "
            << (reference_logits_provided && success ? "true" : "false")
            << ",\n"
            << "  \"performance_claim\": false,\n"
            << "  \"qualification_scope\": \""
            << (reference_logits_provided
                    ? "single-prompt-full-vocabulary-kld-and-top1"
                    : "oracle-free-residency-and-repeat-execution")
            << "\",\n"
            << "  \"runtime_python\": false,\n"
            << "  \"runtime_torch\": false,\n"
            << "  \"runtime_vllm\": false,\n"
            << "  \"runtime_triton\": false,\n"
            << "  \"model_loads\": 1,\n"
            << "  \"request_count\": " << requests.size() << ",\n"
            << "  \"variable_output_lengths\": "
            << (!max_new_tokens_sequence.empty() ? "true" : "false")
            << ",\n"
            << "  \"prefix_extension_mode\": "
            << (!cached_suffix_token_ids.empty() ? "true" : "false")
            << ",\n"
            << "  \"repeat_tokens_identical\": "
            << (repeat_tokens_identical ? "true" : "false") << ",\n"
            << "  \"expected_tokens_provided\": "
            << (!expected_token_ids.empty() ? "true" : "false") << ",\n"
            << "  \"expected_tokens_match\": "
            << (!expected_token_ids.empty()
                    ? (expected_tokens_match ? "true" : "false")
                    : "null")
            << ",\n"
            << "  \"reference_logits\": {\n"
            << "    \"provided\": "
            << (reference_logits_provided ? "true" : "false") << ",\n"
            << "    \"qualification_reads\": "
            << (reference_logits_provided ? 1 : 0) << ",\n"
            << "    \"elements\": "
            << (reference_logits_provided
                    ? std::to_string(logits_comparison.elements)
                    : "null")
            << ",\n"
            << "    \"finite_elements\": "
            << (reference_logits_provided
                    ? std::to_string(logits_comparison.finite_elements)
                    : "null")
            << ",\n"
            << "    \"reference_top1_token_id\": "
            << (reference_logits_provided
                    ? std::to_string(
                          logits_comparison.reference_top1_token_id)
                    : "null")
            << ",\n"
            << "    \"actual_top1_token_id\": "
            << (reference_logits_provided
                    ? std::to_string(logits_comparison.actual_top1_token_id)
                    : "null")
            << ",\n"
            << "    \"top1_match\": "
            << (reference_logits_provided
                    ? (logits_comparison.top1_match ? "true" : "false")
                    : "null")
            << ",\n"
            << "    \"exact_elements\": "
            << (reference_logits_provided
                    ? std::to_string(logits_comparison.exact_elements)
                    : "null")
            << ",\n"
            << "    \"maximum_absolute_error\": "
            << (reference_logits_provided
                    ? json_number(logits_comparison.maximum_absolute_error)
                    : "null")
            << ",\n"
            << "    \"relative_l2_error\": "
            << (reference_logits_provided
                    ? json_number(logits_comparison.relative_l2_error)
                    : "null")
            << ",\n"
            << "    \"kl_divergence\": "
            << (reference_logits_provided
                    ? json_number(logits_comparison.kl_divergence)
                    : "null")
            << ",\n"
            << "    \"kl_divergence_threshold\": 0.005,\n"
            << "    \"qualified\": "
            << (reference_logits_provided
                    ? (logits_qualified ? "true" : "false")
                    : "null")
            << "\n"
            << "  },\n"
            << "  \"load\": {\n"
            << "    \"device_name\": \"" << json_escape(load.device_name)
            << "\",\n"
            << "    \"gpu_arch\": \"" << json_escape(load.gpu_arch)
            << "\",\n"
            << "    \"model_payload_bytes\": "
            << load.model_payload_bytes << ",\n"
            << "    \"model_tensor_count\": " << load.model_tensor_count
            << ",\n"
            << "    \"model_shard_count\": " << load.model_shard_count
            << ",\n"
            << "    \"language_model_payload_bytes\": "
            << load.language_model_payload_bytes << ",\n"
            << "    \"language_model_tensor_count\": "
            << load.language_model_tensor_count << ",\n"
            << "    \"language_model_shard_count\": "
            << load.language_model_shard_count << ",\n"
            << "    \"language_layout_manifest_sha256\": \""
            << load.language_layout_manifest_sha256 << "\",\n"
            << "    \"visual_model_payload_bytes\": "
            << load.visual_model_payload_bytes << ",\n"
            << "    \"visual_model_tensor_count\": "
            << load.visual_model_tensor_count << ",\n"
            << "    \"visual_model_shard_count\": "
            << load.visual_model_shard_count << ",\n"
            << "    \"visual_layout_manifest_sha256\": \""
            << load.visual_layout_manifest_sha256 << "\",\n"
            << "    \"decode_weight_bindings\": "
            << load.decode_weight_bindings << ",\n"
            << "    \"prefill_prepared_launches\": "
            << load.prefill_prepared_launches << ",\n"
            << "    \"decode_prepared_launches\": "
            << load.decode_prepared_launches << ",\n"
            << "    \"aot_loaded_modules\": " << load.aot_loaded_modules
            << ",\n"
            << "    \"prefill_gemm_plans\": " << load.prefill_gemm_plans
            << ",\n"
            << "    \"prefill_workspace_bytes\": "
            << load.prefill_workspace_bytes << ",\n"
            << "    \"mrope_position_state_bytes\": "
            << load.mrope_position_state_bytes << ",\n"
            << "    \"vl_unified_attention_metadata_bytes\": "
            << load.vl_unified_attention_metadata_bytes << ",\n"
            << "    \"vl_unified_attention_decode_scratch_bytes\": "
            << load.vl_unified_attention_decode_scratch_bytes << ",\n"
            << "    \"vl_unified_attention_image_bytes\": "
            << load.vl_unified_attention_image_bytes << ",\n"
            << "    \"vl_unified_attention_loaded\": "
            << (load.vl_unified_attention_loaded ? "true" : "false")
            << ",\n"
            << "    \"vl_logical_projection_weight_bytes\": "
            << load.vl_logical_projection_weight_bytes << ",\n"
            << "    \"vl_logical_projection_output_scratch_bytes\": "
            << load.vl_logical_projection_output_scratch_bytes << ",\n"
            << "    \"vl_logical_projection_weights_loaded\": "
            << (load.vl_logical_projection_weights_loaded ? "true" : "false")
            << ",\n"
            << "    \"decode_workspace_bytes\": "
            << load.decode_workspace_bytes << ",\n"
            << "    \"attention_state_bytes\": "
            << load.attention_state_bytes << ",\n"
            << "    \"exact_prefix_cache_bytes\": "
            << load.exact_prefix_cache_bytes << ",\n"
            << "    \"prefix_cache_entries\": "
            << load.prefix_cache_entries << ",\n"
            << "    \"vision_warmup_patches\": "
            << load.vision_warmup_patches << ",\n"
            << "    \"vision_warmup_visual_tokens\": "
            << load.vision_warmup_visual_tokens << ",\n"
            << "    \"vision_plan_cache_entries_at_ready\": "
            << load.vision_plan_cache_entries_at_ready << ",\n"
            << "    \"vision_warmup_plan_build_wall_ms\": "
            << load.vision_warmup_plan_build_wall_ms << ",\n"
            << "    \"vision_warmup_encode_wall_ms\": "
            << load.vision_warmup_encode_wall_ms << ",\n"
            << "    \"vision_warmup_completed\": "
            << (load.vision_warmup_completed ? "true" : "false")
            << ",\n"
            << "    \"cache_capacity\": " << load.cache_capacity << ",\n"
            << "    \"prompt_tokens\": " << load.prompt_tokens << ",\n"
            << "    \"resident_prefill_buckets\": [";
  for (std::size_t index = 0;
       index < load.resident_prefill_buckets.size(); ++index) {
    if (index != 0) std::cout << ',';
    std::cout << load.resident_prefill_buckets[index];
  }
  std::cout << "],\n"
            << "    \"fmha_provider_backend\": \""
            << json_escape(load.fmha_provider_backend) << "\",\n"
            << "    \"fmha_provider_path\": \""
            << json_escape(load.fmha_provider_path) << "\",\n"
            << "    \"fmha_provider_loaded\": "
            << (load.fmha_provider_loaded ? "true" : "false") << ",\n"
            << "    \"secondary_fmha_provider_backend\": \""
            << json_escape(load.secondary_fmha_provider_backend) << "\",\n"
            << "    \"secondary_fmha_provider_path\": \""
            << json_escape(load.secondary_fmha_provider_path) << "\",\n"
            << "    \"secondary_fmha_provider_loaded\": "
            << (load.secondary_fmha_provider_loaded ? "true" : "false")
            << ",\n"
            << "    \"secondary_fmha_layers\": [";
  for (std::size_t index = 0;
       index < load.secondary_fmha_layers.size(); ++index) {
    if (index != 0) std::cout << ',';
    std::cout << load.secondary_fmha_layers[index];
  }
  std::cout << "],\n"
            << "    \"ck_provider_loaded\": "
            << (load.ck_provider_loaded ? "true" : "false") << ",\n"
            << "    \"raw_weight_load_wall_ms\": "
            << load.raw_weight_load_wall_ms << ",\n"
            << "    \"derived_weight_build_wall_ms\": "
            << load.derived_weight_build_wall_ms << ",\n"
            << "    \"lm_head_build_wall_ms\": "
            << load.lm_head_build_wall_ms << ",\n"
            << "    \"vl_logical_projection_weight_build_wall_ms\": "
            << load.vl_logical_projection_weight_build_wall_ms << ",\n"
            << "    \"prefill_gemm_plan_build_wall_ms\": "
            << load.prefill_gemm_plan_build_wall_ms << ",\n"
            << "    \"command_to_ready_wall_ms\": "
            << load.command_to_ready_wall_ms << "\n"
            << "  },\n"
            << "  \"requests\": [\n";
  for (std::size_t index = 0; index < requests.size(); ++index) {
    const auto& request = requests[index];
    std::cout << "    {\"request_index\":" << request.request_index
              << ",\"prompt_tokens\":" << request.prompt_tokens
              << ",\"completion_tokens\":" << request.completion_tokens
              << ",\"output_token_ids\":[";
    for (std::size_t token = 0; token < request.output_token_ids.size();
         ++token) {
      if (token != 0) std::cout << ',';
      std::cout << request.output_token_ids[token];
    }
    std::cout << "]"
              << ",\"output_token_ids_sha256\":\""
              << request.output_token_ids_sha256 << "\""
              << ",\"oracle_tensor_reads\":"
              << request.oracle_tensor_reads
              << ",\"layer_tail_comparison_count\":"
              << request.layer_tail_comparisons.size()
              << ",\"model_loads\":" << request.model_loads
              << ",\"first_token_certified\":"
              << (request.first_token_certified ? "true" : "false")
              << ",\"all_decode_tokens_certified\":"
              << (request.all_decode_tokens_certified ? "true" : "false")
              << ",\"state_orientation_resets\":"
              << request.state_orientation_resets
              << ",\"prompt_execution\":\""
              << json_escape(request.prompt_execution) << "\""
              << ",\"aot_prefill_tokens\":"
              << request.aot_prefill_tokens
              << ",\"aot_prefill_bucket_tokens\":"
              << request.aot_prefill_bucket_tokens
              << ",\"aot_prefill_segments\":"
              << request.aot_prefill_segments
              << ",\"padded_prefill_tokens\":"
              << request.padded_prefill_tokens
              << ",\"mrope_enabled\":"
              << (request.mrope_enabled ? "true" : "false")
              << ",\"mrope_position_delta\":"
              << request.mrope_position_delta
              << ",\"mrope_position_upload_bytes\":"
              << request.mrope_position_upload_bytes
              << ",\"mrope_full_attention_launches\":"
              << request.mrope_full_attention_launches
              << ",\"mrope_decode_steps\":"
              << request.mrope_decode_steps
              << ",\"cold_prompt_decode_tokens\":"
              << request.cold_prompt_decode_tokens
              << ",\"cold_prompt_decode_wall_ms\":"
              << request.cold_prompt_decode_wall_ms
              << ",\"prefix_cache_lookup\":\""
              << json_escape(request.prefix_cache_lookup) << "\""
              << ",\"prefix_cache_matched_tokens\":"
              << request.prefix_cache_matched_tokens
              << ",\"prefix_cache_suffix_tokens\":"
              << request.prefix_cache_suffix_tokens
              << ",\"prefix_cache_hits\":"
              << request.prefix_cache_hits
              << ",\"prefix_cache_misses\":"
              << request.prefix_cache_misses
              << ",\"prefix_cache_transfer_bytes\":"
              << request.prefix_cache_transfer_bytes
              << ",\"prefix_cache_active_kv_reused\":"
              << (request.prefix_cache_active_kv_reused ? "true" : "false")
              << ",\"prefix_cache_restore_wall_ms\":"
              << request.prefix_cache_restore_wall_ms
              << ",\"prefix_cache_suffix_decode_tokens\":"
              << request.prefix_cache_suffix_decode_tokens
              << ",\"prefix_cache_suffix_aot_launches\":"
              << request.prefix_cache_suffix_aot_launches
              << ",\"prefix_cache_suffix_native_launches\":"
              << request.prefix_cache_suffix_native_launches
              << ",\"prefix_cache_suffix_wall_ms\":"
              << request.prefix_cache_suffix_wall_ms
              << ",\"prefill_aot_launches\":"
              << request.prefill_aot_launches
              << ",\"prefill_dense_gemm_launches\":"
              << request.prefill_dense_gemm_launches
              << ",\"prefill_native_pointwise_launches\":"
              << request.prefill_native_pointwise_launches
              << ",\"prefill_ck_fmha_launches\":"
              << request.prefill_ck_fmha_launches
              << ",\"prefill_vl_unified_attention_launches\":"
              << request.prefill_vl_unified_attention_launches
              << ",\"vl_logical_projections_enabled\":"
              << (request.vl_logical_projections_enabled ? "true" : "false")
              << ",\"vl_logical_projection_tokens\":"
              << request.vl_logical_projection_tokens
              << ",\"vl_logical_projection_plan_count\":"
              << request.vl_logical_projection_plan_count
              << ",\"vl_logical_projection_workspace_bytes\":"
              << request.vl_logical_projection_workspace_bytes
              << ",\"vl_logical_projection_plan_build_wall_ms\":"
              << request.vl_logical_projection_plan_build_wall_ms
              << ",\"vl_logical_projection_plan_reused\":"
              << (request.vl_logical_projection_plan_reused ? "true"
                                                            : "false")
              << ",\"decode_tokens_executed\":"
              << request.decode_tokens_executed
              << ",\"decode_aot_launches\":"
              << request.decode_aot_launches
              << ",\"decode_native_launches\":"
              << request.decode_native_launches
              << ",\"prefill_wall_ms\":" << request.prefill_wall_ms
              << ",\"prefill_tokens_per_second\":"
              << request.prefill_tokens_per_second
              << ",\"decode_wall_ms\":" << request.decode_wall_ms
              << ",\"decode_tokens_per_second\":"
              << request.decode_tokens_per_second
              << ",\"request_wall_ms\":" << request.request_wall_ms
              << ",\"reference_logits_kl_divergence\":";
    if (reference_logits_provided) {
      std::cout << request_logits_comparisons[index].kl_divergence;
    } else {
      std::cout << "null";
    }
    std::cout << ",\"reference_logits_top1_match\":";
    if (reference_logits_provided) {
      std::cout << (request_logits_comparisons[index].top1_match
                        ? "true" : "false");
    } else {
      std::cout << "null";
    }
    std::cout << ",\"reference_logits_qualified\":";
    if (reference_logits_provided) {
      std::cout << (request_logits_qualified(index) ? "true" : "false");
    } else {
      std::cout << "null";
    }
    std::cout << ",\"layer_tail_comparisons\":[";
    for (std::size_t layer = 0;
         layer < request.layer_tail_comparisons.size(); ++layer) {
      const auto& comparison = request.layer_tail_comparisons[layer];
      if (layer != 0) std::cout << ',';
      std::cout << "{\"comparison_index\":" << layer
                << ",\"label\":\""
                << json_escape(comparison.label) << "\""
                << ",\"finite_elements\":"
                << comparison.finite_elements
                << ",\"elements\":" << comparison.elements
                << ",\"exact_elements\":" << comparison.exact_elements
                << ",\"first_mismatch_provided\":"
                << (comparison.first_mismatch_provided ? "true" : "false")
                << ",\"first_mismatch_index\":"
                << comparison.first_mismatch_index
                << ",\"first_mismatch_expected\":"
                << comparison.first_mismatch_expected
                << ",\"first_mismatch_actual\":"
                << comparison.first_mismatch_actual
                << ",\"maximum_absolute_error\":"
                << comparison.maximum_absolute_error
                << ",\"relative_l2_error\":"
                << comparison.relative_l2_error
                << ",\"cosine_similarity\":"
                << comparison.cosine_similarity << "}";
    }
    std::cout << "]}"
              << (index + 1 == requests.size() ? "\n" : ",\n");
  }
  std::cout << "  ]\n}\n";
  return success ? 0 : 3;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    configure_bundled_rocm();
    if (argc == 2 && std::string(argv[1]) == "--version") {
      std::cout << "aima-engine-native " << kVersion << "\n";
      return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--build-info") {
      std::cout << "{\"version\":\"" << kVersion
                << "\",\"source_commit\":\""
                << json_escape(kSourceCommit) << "\"}\n";
      return 0;
    }
    if (argc < 2 || std::string(argv[1]) == "--help" ||
        std::string(argv[1]) == "-h") {
      usage(std::cout);
      return argc < 2 ? 2 : 0;
    }
    if (std::string(argv[1]) == "weights-probe") {
      return run_weights_probe(argc, argv);
    }
    if (std::string(argv[1]) == "doctor") {
      return aima::run_native_doctor(argc, argv, kVersion, kSourceCommit);
    }
    if (std::string(argv[1]) == "aot-closure-probe" && argc == 2) {
      return run_aot_closure_probe();
    }
    if (std::string(argv[1]) == "decode-schedule-probe" && argc == 2) {
      return run_schedule_probe(false);
    }
    if (std::string(argv[1]) == "prefill-schedule-probe" && argc == 2) {
      return run_schedule_probe(true);
    }
    if (std::string(argv[1]) == "bf16-gemm-probe" && argc == 2) {
      return run_bf16_gemm_probe();
    }
    if (std::string(argv[1]) == "bf16-wvsplitk-probe" && argc == 2) {
      return run_bf16_wvsplitk_probe();
    }
    if (std::string(argv[1]) == "derived-weights-probe") {
      return run_derived_weights_probe(argc, argv);
    }
    if (std::string(argv[1]) == "lm-head-probe") {
      return run_lm_head_probe(argc, argv);
    }
    if (std::string(argv[1]) == "decode-bindings-probe") {
      return run_decode_bindings_probe(argc, argv);
    }
    if (std::string(argv[1]) == "linear-prefill-oracle-probe") {
      return run_linear_prefill_oracle_probe(argc, argv);
    }
    if (std::string(argv[1]) == "moe-prefill-oracle-probe") {
      return run_moe_prefill_oracle_probe(argc, argv);
    }
    if (std::string(argv[1]) == "prefill-layer0-oracle-probe") {
      return run_prefill_layer0_oracle_probe(argc, argv);
    }
    if (std::string(argv[1]) == "prefill-linear-layer-oracle-probe") {
      return run_prefill_layer0_oracle_probe(argc, argv);
    }
    if (std::string(argv[1]) == "prefill-linear-prefix-oracle-probe") {
      return run_prefill_linear_prefix_oracle_probe(argc, argv);
    }
    if (std::string(argv[1]) == "prefill-all-layers-oracle-probe") {
      return run_prefill_all_layers_oracle_probe(argc, argv);
    }
    if (std::string(argv[1]) == "resident-session-probe") {
      return run_resident_session_probe(argc, argv);
    }
    if (std::string(argv[1]) == "vl-generation-logits-probe") {
      return aima::run_native_vl_generation_logits_probe(argc, argv);
    }
    if (std::string(argv[1]) == "serve") {
      return aima::run_native_http_server(argc, argv);
    }
    if (std::string(argv[1]) == "prefill-full-layer-oracle-probe") {
      return run_prefill_full_layer_oracle_probe(argc, argv);
    }
    if (std::string(argv[1]) == "full-attention-core-oracle-probe") {
      return run_full_attention_core_oracle_probe(argc, argv);
    }
    if (std::string(argv[1]) == "full-layer-oracle-probe") {
      return run_full_layer_oracle_probe(argc, argv);
    }
    if (std::string(argv[1]) == "decode-oracle-probe") {
      return run_decode_oracle_probe(argc, argv);
    }
    if (std::string(argv[1]) == "linear-layer-oracle-probe") {
      return run_linear_layer_oracle_probe(argc, argv);
    }
    if (std::string(argv[1]) == "tokenizer-probe") {
      return run_tokenizer_probe(argc, argv, false);
    }
    if (std::string(argv[1]) == "chat-template-probe") {
      return run_tokenizer_probe(argc, argv, true);
    }
    throw std::runtime_error("unknown command: " + std::string(argv[1]));
  } catch (const std::exception& error) {
    std::cerr << "{\"complete\":false,\"error\":\""
              << json_escape(error.what()) << "\"}\n";
    return 2;
  }
}
