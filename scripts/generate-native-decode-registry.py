#!/usr/bin/env python3
"""Validate and compile the qualified decode schedule into static C++ data."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


FROZEN_TEXT_Q1024_MANIFEST_SHA256 = (
    "93853b9f9837deba0a9e051bf5be4c516d74d1c5ea1a33e8e7e47ee81e914125"
)
FROZEN_TEXT_Q1024_SCHEDULE_SHA256 = (
    "10565e59b0805ca407ef453caf72f3dfd254752d150903131e188527b910fb97"
)


EXPECTED_LAYER_SEQUENCE = {
    "linear_attention": [
        "triton_rmsnorm_kernel",
        "triton_fused_input_proj_conv_kernel",
        "fused_sigmoid_gating_delta_rule_update_kernel",
        "triton_linear_gated_norm_kernel",
        "triton_rmsnorm_kernel",
        "triton_matvec_kernel",
        "triton_router_topk_stage1_kernel",
        "triton_router_topk_stage2_softmax_kernel",
        "raw_row_gate_up_activation_kernel",
        "raw_row_down_sum_kernel",
    ],
    "full_attention": [
        "triton_rmsnorm_kernel",
        "triton_matvec_kernel",
        "triton_head_norm_rope_kernel",
        "triton_head_norm_rope_kernel",
        "triton_rmsnorm_kernel",
        "triton_matvec_kernel",
        "triton_router_topk_stage1_kernel",
        "triton_router_topk_stage2_softmax_kernel",
        "raw_row_gate_up_activation_kernel",
        "raw_row_down_sum_kernel",
    ],
    "final_logits": [
        "triton_rmsnorm_kernel",
        "triton_lm_head_rowwise_int8_gemv_kernel",
    ],
}
EXPECTED_PREFILL_LAYER_SEQUENCE = {
    "linear_attention": [
        "triton_rmsnorm_kernel",
        "_causal_conv1d_fwd_kernel",
        "_fused_post_conv_kernel",
        "chunk_local_cumsum_scalar_kernel",
        "chunk_scaled_dot_kkt_fwd_kernel",
        "merge_16x16_to_32x32_inverse_kernel",
        "recompute_w_u_fwd_kernel",
        "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
        "chunk_fwd_kernel_o",
        "triton_linear_gated_norm_from_invstd_kernel",
        "triton_prefill_fused_add_rmsnorm_kernel",
        "fused_moe_kernel",
        "fused_moe_kernel",
    ],
    "full_attention": [
        "triton_rmsnorm_kernel",
        "triton_prefill_fused_add_rmsnorm_kernel",
        "fused_moe_kernel",
        "fused_moe_kernel",
    ],
    "final_logits": ["triton_lm_head_rowwise_int8_gemv_kernel"],
}
EXPECTED_PREFILL_Q32768_LAYER_SEQUENCE = {
    "linear_attention": [
        "triton_rmsnorm_kernel",
        "triton_prefill_direct_conv_kernel",
        "_fused_post_conv_kernel",
        "chunk_local_cumsum_scalar_kernel",
        "chunk_scaled_dot_kkt_fwd_kernel",
        "merge_16x16_to_32x32_inverse_kernel",
        "recompute_w_u_fwd_kernel",
        "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
        "chunk_fwd_kernel_o",
        "triton_prefill_fused_add_rmsnorm_kernel",
        "fused_moe_kernel",
        "fused_moe_kernel",
    ],
    "full_attention": [
        "triton_rmsnorm_kernel",
        "triton_prefill_fused_add_rmsnorm_kernel",
        "fused_moe_kernel",
        "fused_moe_kernel",
    ],
    "final_logits": ["triton_lm_head_rowwise_int8_gemv_kernel"],
}
EXPECTED_PREFILL_Q1024_LAYER_SEQUENCE = {
    "linear_attention": [
        "triton_rmsnorm_kernel",
        "triton_prefill_direct_conv_kernel",
        "_fused_post_conv_kernel",
        "chunk_local_cumsum_scalar_kernel",
        "chunk_scaled_dot_kkt_fwd_kernel",
        "merge_16x16_to_64x64_inverse_kernel",
        "recompute_w_u_fwd_kernel",
        "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
        "chunk_fwd_kernel_o",
        "triton_prefill_fused_add_rmsnorm_kernel",
        "fused_moe_kernel",
        "fused_moe_kernel",
    ],
    "full_attention": [
        "triton_rmsnorm_kernel",
        "triton_prefill_fused_add_rmsnorm_kernel",
        "fused_moe_kernel",
        "fused_moe_kernel",
    ],
    "final_logits": ["triton_lm_head_rowwise_int8_gemv_kernel"],
}
EXPECTED_PREFILL_SPLIT_TAIL_LAYER_SEQUENCE = {
    "linear_attention": [
        "triton_rmsnorm_kernel",
        "_causal_conv1d_fwd_kernel",
        "_fused_post_conv_kernel",
        "chunk_local_cumsum_scalar_kernel",
        "chunk_scaled_dot_kkt_fwd_kernel",
        "merge_16x16_to_32x32_inverse_kernel",
        "recompute_w_u_fwd_kernel",
        "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
        "chunk_fwd_kernel_o",
        "triton_prefill_fused_add_rmsnorm_kernel",
        "fused_moe_kernel",
        "fused_moe_kernel",
    ],
    "full_attention": [
        "triton_rmsnorm_kernel",
        "triton_prefill_fused_add_rmsnorm_kernel",
        "fused_moe_kernel",
        "fused_moe_kernel",
    ],
    "final_logits": ["triton_lm_head_rowwise_int8_gemv_kernel"],
}

DTYPE_ENUM = {
    "torch.bfloat16": "DecodeTensorDtype::kBfloat16",
    "torch.float32": "DecodeTensorDtype::kFloat32",
    "torch.int32": "DecodeTensorDtype::kInt32",
    "torch.int8": "DecodeTensorDtype::kInt8",
    "torch.bool": "DecodeTensorDtype::kBool",
}
BINDING_ENUM = {
    "model_or_derived_weight": "DecodeBindingKind::kModelOrDerivedWeight",
    "resident_state_or_workspace": "DecodeBindingKind::kResidentStateOrWorkspace",
    "transient_workspace": "DecodeBindingKind::kTransientWorkspace",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def c_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def validate(
    schedule: dict[str, Any],
    manifest: dict[str, Any],
    raw: str,
    phase: str,
    prefill_registry: str = "default",
) -> None:
    if schedule.get("schema") != f"aima-amd395-qwen36/native-{phase}-aot-schedule/v1":
        raise RuntimeError(f"unsupported native {phase} schedule schema")
    if schedule.get("status") != "qualified_schedule_contract_native_executor_wiring_pending":
        raise RuntimeError(f"native {phase} schedule is not at the qualified wiring boundary")
    launches = schedule.get("schedule")
    context_tokens = int(schedule.get("request", {}).get("context_tokens", 0))
    if phase == "decode":
        templates = EXPECTED_LAYER_SEQUENCE
    elif context_tokens == 1024:
        templates = (
            EXPECTED_PREFILL_Q32768_LAYER_SEQUENCE
            if prefill_registry == "frozen-text"
            else EXPECTED_PREFILL_Q1024_LAYER_SEQUENCE
        )
    elif context_tokens == 8192:
        templates = EXPECTED_PREFILL_LAYER_SEQUENCE
    elif context_tokens in {7168, 7680, 8191}:
        templates = EXPECTED_PREFILL_SPLIT_TAIL_LAYER_SEQUENCE
    elif context_tokens > 0:
        templates = EXPECTED_PREFILL_Q32768_LAYER_SEQUENCE
    else:
        raise RuntimeError(
            f"no qualified native prefill template for context {context_tokens}"
        )
    expected_launches = (
        30 * len(templates["linear_attention"])
        + 10 * len(templates["full_attention"])
        + len(templates["final_logits"])
    )
    if not isinstance(launches, list) or len(launches) != expected_launches:
        raise RuntimeError(
            f"native {phase} schedule must contain {expected_launches} launches"
        )
    if any(marker in raw for marker in ('"data_ptr"', '"storage_data_ptr"', "/home/", "/data/")):
        raise RuntimeError(f"native {phase} schedule retains a pointer or private path")
    manifest_hashes = {str(item["kernel_hash"]) for item in manifest["kernels"]}
    if any(str(item.get("kernel_hash")) not in manifest_hashes for item in launches):
        raise RuntimeError(f"native {phase} schedule uses a non-embedded kernel")
    if phase == "prefill" and context_tokens == 1024:
        manifest_by_hash = {
            str(item["kernel_hash"]): item for item in manifest["kernels"]
        }
        official_fla_symbols = {
            "chunk_local_cumsum_scalar_kernel",
            "chunk_scaled_dot_kkt_fwd_kernel",
            "merge_16x16_to_32x32_inverse_kernel",
            "merge_16x16_to_64x64_inverse_kernel",
            "recompute_w_u_fwd_kernel",
            "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
            "chunk_fwd_kernel_o",
        }
        expected_chunk_size = 32 if prefill_registry == "frozen-text" else 64
        for launch in launches:
            if (
                launch.get("layer_type") == "linear_attention"
                and launch.get("symbol") in official_fla_symbols
            ):
                kernel = manifest_by_hash[str(launch["kernel_hash"])]
                if (
                    kernel.get("compile_constants", {}).get("BT")
                    != expected_chunk_size
                ):
                    raise RuntimeError(
                        "native q1024 prefill has the wrong FLA chunk size for "
                        f"the {prefill_registry} registry"
                    )
    sequence_key = f"{phase}_sequence"
    for sequence, launch in enumerate(launches):
        if launch.get(sequence_key) != sequence:
            raise RuntimeError(f"native {phase} sequence is not contiguous")
        arguments = launch.get("arguments")
        if not isinstance(arguments, list) or not arguments:
            raise RuntimeError(f"native decode launch {sequence} has no arguments")
        for argument in arguments:
            kind = argument.get("kind")
            abi_type = argument.get("abi_type")
            if kind == "tensor":
                if argument.get("dtype") not in DTYPE_ENUM:
                    raise RuntimeError(f"unsupported native tensor dtype: {argument.get('dtype')}")
                if argument.get("binding_kind") not in BINDING_ENUM:
                    raise RuntimeError("unsupported native tensor binding kind")
                if not str(abi_type).startswith("*"):
                    raise RuntimeError("native tensor argument does not have pointer ABI")
            elif kind == "scalar":
                if abi_type not in {"fp32", "i32", "i64"}:
                    raise RuntimeError(f"unsupported native scalar ABI: {abi_type}")
            else:
                raise RuntimeError(f"unsupported native argument kind: {kind}")
    for layer_index in range(40):
        actual = [
            str(item["symbol"])
            for item in launches
            if item.get("layer_index") == layer_index
        ]
        expected = templates[
            "full_attention" if layer_index % 4 == 3 else "linear_attention"
        ]
        if actual != expected:
            raise RuntimeError(f"native {phase} template drift at layer {layer_index}")
    final_template = templates["final_logits"]
    if [str(item["symbol"]) for item in launches[-len(final_template):]] != final_template:
        raise RuntimeError(f"native final-logit {phase} template drift")


def argument_cpp(argument: dict[str, Any]) -> str:
    name = c_string(str(argument["name"]))
    abi_type = c_string(str(argument["abi_type"]))
    if argument["kind"] == "tensor":
        return (
            "  {%s, %s, DecodeArgumentKind::kTensor, %s, %s, %s, %dULL, %dULL, 0.0f, 0, 0LL},"
            % (
                name,
                abi_type,
                DTYPE_ENUM[str(argument["dtype"])],
                BINDING_ENUM[str(argument["binding_kind"])],
                c_string(str(argument["binding"])),
                int(argument["storage_nbytes"]),
                int(argument["byte_offset"]),
            )
        )
    if argument["abi_type"] == "fp32":
        value = repr(float(argument["value"]))
        if "." not in value and "e" not in value.lower():
            value += ".0"
        return (
            "  {%s, %s, DecodeArgumentKind::kFloat32, DecodeTensorDtype::kNone, "
            "DecodeBindingKind::kNone, \"\", 0ULL, 0ULL, %sf, 0, 0LL},"
            % (name, abi_type, value)
        )
    if argument["abi_type"] == "i32":
        return (
            "  {%s, %s, DecodeArgumentKind::kInt32, DecodeTensorDtype::kNone, "
            "DecodeBindingKind::kNone, \"\", 0ULL, 0ULL, 0.0f, %d, 0LL},"
            % (name, abi_type, int(argument["value"]))
        )
    return (
        "  {%s, %s, DecodeArgumentKind::kInt64, DecodeTensorDtype::kNone, "
        "DecodeBindingKind::kNone, \"\", 0ULL, 0ULL, 0.0f, 0, %dLL},"
        % (name, abi_type, int(argument["value"]))
    )


def generate_arrays(
    schedule: dict[str, Any], phase: str, suffix: str = ""
) -> tuple[str, str]:
    arguments: list[str] = []
    launches: list[str] = []
    offset = 0
    for launch in schedule["schedule"]:
        launch_arguments = launch["arguments"]
        arguments.extend(argument_cpp(argument) for argument in launch_arguments)
        layer_index = -1 if launch["layer_index"] is None else int(launch["layer_index"])
        grid = launch["grid"]
        launches.append(
            "  {%d, %d, %s, %s, {%d, %d, %d, %d, %d, %d}, kArguments + %d, %d},"
            % (
                int(launch[f"{phase}_sequence"]),
                layer_index,
                c_string(str(launch["kernel_hash"])),
                c_string(str(launch["symbol"])),
                int(grid[0]),
                int(grid[1]),
                int(grid[2]),
                int(launch["num_warps"]),
                int(launch["warp_size"]),
                int(launch["shared_memory_bytes"]),
                offset,
                len(launch_arguments),
            )
        )
        offset += len(launch_arguments)
    arguments_name = f"kArguments{suffix}"
    launches_name = f"kLaunches{suffix}"
    launches_text = "\n".join(launches).replace(
        "kArguments +", f"{arguments_name} +"
    )
    return (
        "const DecodeArgument %s[] = {\n%s\n};" % (arguments_name, "\n".join(arguments)),
        "const DecodeLaunch %s[] = {\n%s\n};" % (launches_name, launches_text),
    )


def generate_cpp(schedule: dict[str, Any], schedule_sha256: str, phase: str) -> str:
    arguments, launches = generate_arrays(schedule, phase)
    function_name = f"native_{phase}_schedule"
    hash_function_name = f"native_{phase}_schedule_sha256"
    header = "aima/decode_schedule.h" if phase == "decode" else "aima/prefill_schedule.h"
    return """// Generated by scripts/generate-native-decode-registry.py. Do not edit.
// SPDX-License-Identifier: Apache-2.0

#include "%s"

namespace aima {
namespace {
%s
%s
}  // namespace

const DecodeLaunch* %s(std::size_t* count) {
  if (count != nullptr) *count = sizeof(kLaunches) / sizeof(kLaunches[0]);
  return kLaunches;
}

const char* %s() {
  return "%s";
}

}  // namespace aima
""" % (
        header,
        arguments,
        launches,
        function_name,
        hash_function_name,
        schedule_sha256,
    )


def generate_prefill_cpp(
    entries: list[tuple[dict[str, Any], str]],
    registry_variant: str = "default",
) -> str:
    if registry_variant not in ("default", "frozen-text"):
        raise ValueError(f"unsupported prefill registry variant: {registry_variant}")
    function_name = (
        "native_prefill_schedule"
        if registry_variant == "default"
        else "native_frozen_text_prefill_schedule"
    )
    hash_function_name = function_name + "_sha256"
    arrays: list[str] = []
    selectors: list[str] = []
    hash_selectors: list[str] = []
    contexts: list[int] = []
    for schedule, digest in entries:
        context = int(schedule["request"]["context_tokens"])
        if context in contexts:
            raise RuntimeError(f"duplicate native prefill context: {context}")
        contexts.append(context)
        suffix = f"Q{context}"
        arguments, launches = generate_arrays(schedule, "prefill", suffix)
        arrays.extend((arguments, launches))
        selectors.append(
            "    case %d: if (count != nullptr) *count = sizeof(kLaunches%s) / "
            "sizeof(kLaunches%s[0]); return kLaunches%s;"
            % (context, suffix, suffix, suffix)
        )
        hash_selectors.append(
            '    case %d: return "%s";' % (context, digest)
        )
    # The no-context overload is the original q8192 compatibility surface.
    # Product execution selects a context explicitly, but probes and older
    # callers must not silently change shape when multi-context arguments are
    # reordered or a shorter resident bucket is added first.
    default_context = 8192 if 8192 in contexts else contexts[0]
    return """// Generated by scripts/generate-native-decode-registry.py. Do not edit.
// SPDX-License-Identifier: Apache-2.0

#include "aima/prefill_schedule.h"

namespace aima {
namespace {
%s
}  // namespace

const DecodeLaunch* %s(std::size_t context_tokens,
                       std::size_t* count) {
  switch (context_tokens) {
%s
    default: if (count != nullptr) *count = 0; return nullptr;
  }
}

const char* %s(std::size_t context_tokens) {
  switch (context_tokens) {
%s
    default: return "";
  }
}

const DecodeLaunch* %s(std::size_t* count) {
  return %s(%d, count);
}

const char* %s() {
  return %s(%d);
}

}  // namespace aima
""" % (
        "\n\n".join(arrays),
        function_name,
        "\n".join(selectors),
        hash_function_name,
        "\n".join(hash_selectors),
        function_name,
        function_name,
        default_context,
        hash_function_name,
        hash_function_name,
        default_context,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", type=Path, action="append", required=True)
    parser.add_argument("--aot-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output-cpp", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--phase", choices=("decode", "prefill"), default="decode")
    parser.add_argument(
        "--prefill-registry",
        choices=("default", "frozen-text"),
        default="default",
    )
    args = parser.parse_args()
    if args.output_cpp is None and not args.check:
        parser.error("--output-cpp is required unless --check is used")
    if len(args.schedule) != len(args.aot_manifest):
        raise RuntimeError("each native schedule requires one matching AOT manifest")
    if args.phase == "decode" and len(args.schedule) != 1:
        raise RuntimeError("native decode accepts exactly one schedule")
    if args.phase != "prefill" and args.prefill_registry != "default":
        raise RuntimeError("prefill registry variants require --phase prefill")
    entries: list[tuple[dict[str, Any], str]] = []
    summaries: list[dict[str, Any]] = []
    for schedule_arg, manifest_arg in zip(args.schedule, args.aot_manifest):
        schedule_path = schedule_arg.resolve()
        manifest_path = manifest_arg.resolve()
        raw = schedule_path.read_text(encoding="utf-8")
        schedule = json.loads(raw)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(schedule, dict) or not isinstance(manifest, dict):
            raise RuntimeError("native schedule and manifest roots must be objects")
        if args.prefill_registry == "frozen-text" and (
            sha256_file(schedule_path) != FROZEN_TEXT_Q1024_SCHEDULE_SHA256
            or sha256_file(manifest_path) != FROZEN_TEXT_Q1024_MANIFEST_SHA256
        ):
            raise RuntimeError("frozen text q1024 closure identity changed")
        validate(
            schedule,
            manifest,
            raw,
            args.phase,
            args.prefill_registry,
        )
        digest = sha256_file(schedule_path)
        entries.append((schedule, digest))
        summaries.append(
            {
                "argument_count": sum(len(item["arguments"]) for item in schedule["schedule"]),
                "context_tokens": int(schedule.get("request", {}).get("context_tokens", 0)),
                "launch_count": len(schedule["schedule"]),
                "schedule_sha256": digest,
            }
        )
    if args.output_cpp is not None:
        output = args.output_cpp.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        generated = (
            generate_prefill_cpp(entries, args.prefill_registry)
            if args.phase == "prefill"
            else generate_cpp(entries[0][0], entries[0][1], args.phase)
        )
        output.write_text(generated, encoding="utf-8")
    print(
        json.dumps(
            {
                "output_cpp": None if args.output_cpp is None else str(args.output_cpp.resolve()),
                "schedules": summaries,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
