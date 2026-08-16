#!/usr/bin/env python3
"""Compare native VL tool generation logits with frozen vLLM rows."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aima_engine.vl_generation_oracle import (  # noqa: E402
    CASE_CONTRACTS,
    CASE_ORDER,
    MODEL_VOCABULARY_SIZE,
    validate_generation_oracle_manifest,
)
from aima_engine.vl_generation_layer_oracle import (  # noqa: E402
    BOUNDARY_NAMES,
    LAYER0_TAIL_BOUNDARY_SPECS,
    NATIVE_LINEAR_ATTENTION_BOUNDARY_NAMES,
    validate_generation_layer_oracle_manifest,
)
from aima_engine.vl_prefill_state_oracle import (  # noqa: E402
    STATE_COMPONENT_NAMES,
    validate_vl_prefill_state_oracle_manifest,
)
from aima_engine.vl_reference import (  # noqa: E402
    atomic_json,
    file_component,
    git_identity,
    load_json_object,
    seal_manifest,
    sha256_file,
)


SCHEMA = "aima-amd395-qwen36/native-vl-generation-qualification/v1"
PROBE_SCHEMA = "aima-amd395-qwen36/native-vl-generation-logits-probe/v1"
MODEL_ID = "aima-amd395-qwen36-35b"
VISION_ATTENTION_SHA256 = (
    "b709a058a77d61e14db73c1ff7d7f4c20859d997bec811cad7339d3e59223d00"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def materialize_request(value: Any, fixture_root: Path) -> Any:
    """Replace public fixture identities with verified local file URIs."""

    if isinstance(value, list):
        return [materialize_request(item, fixture_root) for item in value]
    if not isinstance(value, dict):
        return value
    if {
        "fixture",
        "transport",
        "bytes",
        "sha256",
    }.issubset(value):
        if value["transport"] != "local":
            raise RuntimeError("generation qualification requires local fixtures")
        fixture = fixture_root / value["fixture"]
        if (
            not fixture.is_file()
            or fixture.stat().st_size != value["bytes"]
            or sha256_file(fixture) != value["sha256"]
        ):
            raise RuntimeError(f"frozen media fixture changed: {fixture.name}")
        return fixture.resolve().as_uri()
    return {
        key: materialize_request(item, fixture_root)
        for key, item in value.items()
    }


def build_probe_cases(
    oracle: dict[str, Any],
    oracle_root: Path,
    fixture_root: Path,
    layer_oracle: dict[str, Any] | None = None,
    layer_oracle_root: Path | None = None,
    prefill_state_oracle: dict[str, Any] | None = None,
    prefill_state_oracle_root: Path | None = None,
) -> dict[str, Any]:
    layer_cases = (
        {case["case_id"]: case for case in layer_oracle["cases"]}
        if layer_oracle is not None
        else {}
    )
    if (layer_oracle is None) != (layer_oracle_root is None):
        raise RuntimeError("generation layer oracle inputs must be paired")
    state_cases = (
        {case["case_id"]: case for case in prefill_state_oracle["cases"]}
        if prefill_state_oracle is not None
        else {}
    )
    if (prefill_state_oracle is None) != (prefill_state_oracle_root is None):
        raise RuntimeError("VL prefill state oracle inputs must be paired")
    cases = []
    for case in oracle["cases"]:
        case_id = case["case_id"]
        target = case["divergence_output_index"]
        output_ids = case["generation"]["output_token_ids"]
        request = materialize_request(case["request"], fixture_root)
        request["model"] = MODEL_ID
        component = case["reference_logits"]["component"]
        component_root = oracle_root
        probe_case = {
            "case_id": case_id,
            "request": request,
            "expected_prefix_token_ids": output_ids[:target],
            "expected_reference_token_id": case["reference_logits"][
                "raw_top_tokens"
            ][0]["token_id"],
            "expected_selected_token_id": output_ids[target],
            "reference_logits": str(
                (component_root / component["path"]).resolve()
            ),
            "reference_logits_output_index": target,
        }
        if layer_oracle_root is not None:
            layer_case = layer_cases.get(case_id)
            if not isinstance(layer_case, dict):
                raise RuntimeError(
                    f"generation layer oracle case is missing: {case_id}"
                )
            if layer_case.get("target_output_index") not in (None, target):
                raise RuntimeError(
                    f"generation layer oracle output index differs: {case_id}"
                )
            layer_logits = layer_case.get("target_logits_component")
            if isinstance(layer_logits, dict):
                component = layer_logits
                component_root = layer_oracle_root
                probe_case["reference_logits"] = str(
                    (component_root / component["path"]).resolve()
                )
            probe_case["reference_decode_output_index"] = target
            linear_attention = layer_case.get("linear_attention")
            linear_metadata = (
                linear_attention.get("metadata")
                if isinstance(linear_attention, dict)
                else None
            )
            linear_layer_index = (
                linear_metadata.get("layer_index")
                if isinstance(linear_metadata, dict)
                else 0
            )
            if (
                not isinstance(linear_layer_index, int)
                or isinstance(linear_layer_index, bool)
                or linear_layer_index < 0
                or linear_layer_index >= 40
                or linear_layer_index % 4 == 3
            ):
                raise RuntimeError(
                    f"generation linear observer layer is invalid: {case_id}"
                )
            probe_case["reference_decode_linear_layer_index"] = (
                linear_layer_index
            )
            probe_case["reference_decode_boundary_dir"] = str(
                (layer_oracle_root / case_id).resolve()
            )
            probe_case["reference_decode_linear_boundary_dir"] = str(
                (layer_oracle_root / case_id / "linear").resolve()
            )
            probe_case["reference_decode_layer0_tail_boundary_dir"] = str(
                (layer_oracle_root / case_id / "layer0-tail").resolve()
            )
            probe_case["reference_decode_full_attention_dir"] = str(
                (layer_oracle_root / case_id / "full-attention").resolve()
            )
            full_attention = layer_case.get("full_attention")
            metadata = (
                full_attention.get("metadata")
                if isinstance(full_attention, dict)
                else None
            )
            if isinstance(metadata, dict) and isinstance(
                metadata.get("layer_index"), int
            ):
                probe_case[
                    "reference_decode_full_attention_layer_index"
                ] = metadata["layer_index"]
        if prefill_state_oracle_root is not None:
            state_case = state_cases.get(case_id)
            if not isinstance(state_case, dict):
                raise RuntimeError(
                    f"VL prefill state oracle case is missing: {case_id}"
                )
            probe_case["reference_prefill_state_dir"] = str(
                (prefill_state_oracle_root / case_id).resolve()
            )
        cases.append(probe_case)
    if tuple(case["case_id"] for case in cases) != CASE_ORDER:
        raise RuntimeError("generation qualification case order changed")
    return {"cases": cases}


def qualification_checks(
    probe: dict[str, Any],
    oracle: dict[str, Any],
    layer_oracle: dict[str, Any] | None = None,
    prefill_state_oracle: dict[str, Any] | None = None,
) -> dict[str, bool]:
    probe_cases = {
        case.get("case_id"): case
        for case in probe.get("cases", [])
        if isinstance(case, dict)
    }
    oracle_cases = {case["case_id"]: case for case in oracle["cases"]}
    layer_cases = (
        {case["case_id"]: case for case in layer_oracle["cases"]}
        if layer_oracle is not None
        else {}
    )
    state_cases = (
        {case["case_id"]: case for case in prefill_state_oracle["cases"]}
        if prefill_state_oracle is not None
        else {}
    )
    checks = {
        "probe_schema_exact": probe.get("schema") == PROBE_SCHEMA,
        "probe_complete": probe.get("complete") is True,
        "probe_attribution_qualified": probe.get("qualified_for_attribution")
        is True,
        "single_resident_model_load": probe.get("model_loads") == 1,
        "case_order_exact": tuple(
            case.get("case_id")
            for case in probe.get("cases", [])
            if isinstance(case, dict)
        )
        == CASE_ORDER,
    }
    for case_id in CASE_ORDER:
        observed = probe_cases.get(case_id, {})
        reference = oracle_cases[case_id]
        contract = CASE_CONTRACTS[case_id]
        logits = (
            observed.get("reference_logits")
            if isinstance(observed.get("reference_logits"), dict)
            else {}
        )
        metrics = (
            observed.get("request_metrics")
            if isinstance(observed.get("request_metrics"), dict)
            else {}
        )
        checks[f"{case_id}_prefix_exact"] = observed.get("prefix_exact") is True
        checks[f"{case_id}_prompt_exact"] = (
            metrics.get("prompt_token_ids_sha256")
            == reference["prompt_token_ids_sha256"]
        )
        checks[f"{case_id}_reference_row_bound"] = (
            logits.get("expected_sha256")
            == reference["reference_logits"]["component"]["sha256"]
            and logits.get("reference_top1_token_id")
            == contract["reference_token_id"]
        )
        checks[f"{case_id}_full_vocabulary_finite"] = (
            logits.get("elements") == MODEL_VOCABULARY_SIZE
            and logits.get("finite_elements") == MODEL_VOCABULARY_SIZE
        )
        checks[f"{case_id}_native_top1_exact"] = (
            observed.get("native_top1_exact") is True
            and logits.get("top1_match") is True
        )
        checks[f"{case_id}_selected_token_exact"] = (
            observed.get("selected_native_token_id")
            == contract["reference_token_id"]
        )
        kld = logits.get("kl_divergence")
        checks[f"{case_id}_kld_under_0_005"] = (
            isinstance(kld, (int, float))
            and not isinstance(kld, bool)
            and 0.0 <= kld < 0.005
        )
        checks[f"{case_id}_vl_and_mrope_executed"] = (
            isinstance(metrics.get("vl"), dict)
            and metrics["vl"].get("enabled") is True
            and isinstance(metrics.get("mrope"), dict)
            and metrics["mrope"].get("enabled") is True
        )
        if layer_oracle is not None:
            layer_case = layer_cases[case_id]
            boundaries = observed.get("decode_boundaries")
            boundaries_well_formed = (
                isinstance(boundaries, list)
                and len(boundaries) == len(BOUNDARY_NAMES)
                and all(isinstance(boundary, dict) for boundary in boundaries)
            )
            checks[f"{case_id}_decode_boundaries_complete"] = (
                observed.get("decode_boundaries_complete") is True
                and boundaries_well_formed
            )
            checks[f"{case_id}_decode_boundaries_finite"] = (
                observed.get("decode_boundaries_finite") is True
                and boundaries_well_formed
                and all(
                    boundary.get("elements") == 2_048
                    and boundary.get("finite_elements") == 2_048
                    for boundary in boundaries
                )
            )
            checks[f"{case_id}_decode_boundary_rows_bound"] = (
                boundaries_well_formed
                and [boundary.get("label") for boundary in boundaries]
                == list(BOUNDARY_NAMES)
                and all(
                    boundary.get("expected_sha256")
                    == layer_case["components"][boundary["label"]]["sha256"]
                    for boundary in boundaries
                )
            )
            linear_boundaries = observed.get("decode_linear_boundaries")
            linear_well_formed = (
                isinstance(linear_boundaries, list)
                and len(linear_boundaries)
                == len(NATIVE_LINEAR_ATTENTION_BOUNDARY_NAMES)
                and all(
                    isinstance(boundary, dict)
                    for boundary in linear_boundaries
                )
            )
            checks[f"{case_id}_decode_linear_boundaries_complete"] = (
                observed.get("decode_linear_boundaries_complete") is True
                and linear_well_formed
            )
            checks[f"{case_id}_decode_linear_boundaries_finite"] = (
                observed.get("decode_linear_boundaries_finite") is True
                and linear_well_formed
                and all(
                    boundary.get("finite_elements")
                    == boundary.get("elements")
                    for boundary in linear_boundaries
                )
            )
            linear_components = layer_case["linear_attention"]["components"]
            checks[f"{case_id}_decode_linear_boundary_rows_bound"] = (
                linear_well_formed
                and [boundary.get("label") for boundary in linear_boundaries]
                == list(NATIVE_LINEAR_ATTENTION_BOUNDARY_NAMES)
                and all(
                    boundary.get("expected_sha256")
                    == linear_components[boundary["label"]]["sha256"]
                    for boundary in linear_boundaries
                )
            )
            tail_boundaries = observed.get("decode_layer0_tail_boundaries")
            tail_well_formed = (
                isinstance(tail_boundaries, list)
                and len(tail_boundaries) == len(LAYER0_TAIL_BOUNDARY_SPECS)
                and all(isinstance(boundary, dict) for boundary in tail_boundaries)
            )
            checks[f"{case_id}_decode_layer0_tail_boundaries_complete"] = (
                observed.get("decode_layer0_tail_boundaries_complete") is True
                and tail_well_formed
            )
            checks[f"{case_id}_decode_layer0_tail_boundaries_finite"] = (
                observed.get("decode_layer0_tail_boundaries_finite") is True
                and tail_well_formed
                and all(
                    boundary.get("finite_elements")
                    == boundary.get("elements")
                    for boundary in tail_boundaries
                )
            )
            tail_components = layer_case["layer0_tail"]["components"]
            checks[f"{case_id}_decode_layer0_tail_boundary_rows_bound"] = (
                tail_well_formed
                and [boundary.get("label") for boundary in tail_boundaries]
                == list(LAYER0_TAIL_BOUNDARY_SPECS)
                and all(
                    boundary.get("expected_sha256")
                    == tail_components[boundary["label"]]["sha256"]
                    for boundary in tail_boundaries
                )
            )
        if prefill_state_oracle is not None:
            state_case = state_cases[case_id]
            states = observed.get("prefill_states")
            states_well_formed = (
                isinstance(states, list)
                and len(states) == len(STATE_COMPONENT_NAMES)
                and all(isinstance(state, dict) for state in states)
            )
            checks[f"{case_id}_prefill_states_complete"] = (
                observed.get("prefill_states_complete") is True
                and states_well_formed
            )
            checks[f"{case_id}_prefill_states_finite"] = (
                observed.get("prefill_states_finite") is True
                and states_well_formed
                and all(
                    state.get("elements") in (8_192 * 3, 32 * 128 * 128)
                    and state.get("finite_elements") == state.get("elements")
                    for state in states
                )
            )
            checks[f"{case_id}_prefill_state_rows_bound"] = (
                states_well_formed
                and [state.get("label") for state in states]
                == list(STATE_COMPONENT_NAMES)
                and all(
                    state.get("expected_sha256")
                    == state_case["components"][state["label"]]["sha256"]
                    for state in states
                )
            )
    return checks


def qualify(args: argparse.Namespace) -> dict[str, Any]:
    binary = args.binary.resolve()
    model_dir = args.model_dir.resolve()
    fmha_provider = args.fmha_provider.resolve()
    vision_attention = args.vision_attention_image.resolve()
    fixture_root = args.fixture_root.resolve()
    oracle_path = args.oracle_manifest.resolve()
    oracle_root = args.oracle_root.resolve()
    layer_oracle_path = (
        args.layer_oracle_manifest.resolve()
        if args.layer_oracle_manifest is not None
        else None
    )
    layer_oracle_root = (
        args.layer_oracle_root.resolve()
        if args.layer_oracle_root is not None
        else None
    )
    if (layer_oracle_path is None) != (layer_oracle_root is None):
        raise SystemExit(
            "--layer-oracle-manifest and --layer-oracle-root must be paired"
        )
    prefill_state_oracle_path = (
        args.prefill_state_oracle_manifest.resolve()
        if args.prefill_state_oracle_manifest is not None
        else None
    )
    prefill_state_oracle_root = (
        args.prefill_state_oracle_root.resolve()
        if args.prefill_state_oracle_root is not None
        else None
    )
    if (prefill_state_oracle_path is None) != (
        prefill_state_oracle_root is None
    ):
        raise SystemExit(
            "--prefill-state-oracle-manifest and --prefill-state-oracle-root "
            "must be paired"
        )
    output = args.output.resolve()
    raw_root = output.parent / f"{output.stem}-raw"
    for path in (
        binary,
        fmha_provider,
        vision_attention,
        fixture_root / "fixtures-manifest.json",
        oracle_path,
    ):
        if not path.exists():
            raise SystemExit(f"generation qualification input is missing: {path}")
    if not model_dir.is_dir() or not oracle_root.is_dir():
        raise SystemExit("generation model or oracle directory is missing")
    if layer_oracle_path is not None and (
        not layer_oracle_path.is_file() or not layer_oracle_root.is_dir()
    ):
        raise SystemExit("generation layer oracle input is missing")
    if prefill_state_oracle_path is not None and (
        not prefill_state_oracle_path.is_file()
        or not prefill_state_oracle_root.is_dir()
    ):
        raise SystemExit("VL prefill state oracle input is missing")
    if output.exists() or raw_root.exists():
        raise SystemExit("generation qualification output already exists")

    source = git_identity(ROOT)
    if source["dirty"]:
        raise SystemExit("generation qualification requires clean source")
    build_info = json.loads(
        subprocess.run(
            [str(binary), "--build-info"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    if build_info.get("source_commit") != source["commit"]:
        raise SystemExit("native binary source commit differs from checkout")
    if sha256_file(vision_attention) != VISION_ATTENTION_SHA256:
        raise SystemExit("vision-attention image differs from frozen artifact")

    oracle = load_json_object(oracle_path)
    oracle_errors = validate_generation_oracle_manifest(
        oracle, oracle_root=oracle_root
    )
    if oracle_errors:
        raise SystemExit(
            "invalid generation oracle:\n- " + "\n- ".join(oracle_errors)
        )
    layer_oracle = None
    if layer_oracle_path is not None:
        layer_oracle = load_json_object(layer_oracle_path)
        layer_errors = validate_generation_layer_oracle_manifest(
            layer_oracle, oracle_root=layer_oracle_root
        )
        if layer_errors:
            raise SystemExit(
                "invalid generation layer oracle:\n- "
                + "\n- ".join(layer_errors)
            )
        generation_binding = layer_oracle.get("generation_oracle", {})
        if generation_binding.get("sha256") != sha256_file(oracle_path):
            raise SystemExit(
                "generation layer oracle is bound to a different logits oracle"
            )
    prefill_state_oracle = None
    if prefill_state_oracle_path is not None:
        prefill_state_oracle = load_json_object(prefill_state_oracle_path)
        state_errors = validate_vl_prefill_state_oracle_manifest(
            prefill_state_oracle, oracle_root=prefill_state_oracle_root
        )
        if state_errors:
            raise SystemExit(
                "invalid VL prefill state oracle:\n- "
                + "\n- ".join(state_errors)
            )
        generation_binding = prefill_state_oracle.get(
            "generation_oracle", {}
        )
        if generation_binding.get("sha256") != sha256_file(oracle_path):
            raise SystemExit(
                "VL prefill state oracle is bound to a different logits oracle"
            )

    raw_root.mkdir(parents=True)
    isolated_home = raw_root / "home"
    isolated_home.mkdir()
    load_report = raw_root / "native-weight-load.json"
    stdout_path = raw_root / "probe.stdout.json"
    stderr_path = raw_root / "probe.stderr.log"
    with tempfile.TemporaryDirectory(prefix="aima-vl-generation-cases-") as tmp:
        cases_path = Path(tmp) / "cases.json"
        atomic_json(
            cases_path,
            build_probe_cases(
                oracle,
                oracle_root,
                fixture_root,
                layer_oracle,
                layer_oracle_root,
                prefill_state_oracle,
                prefill_state_oracle_root,
            ),
        )
        command = [
            str(binary),
            "vl-generation-logits-probe",
            "--model-dir",
            str(model_dir),
            "--context-tokens",
            "1024",
            "--cache-capacity",
            "2048",
            "--fmha-provider",
            str(fmha_provider),
            "--vision-attention-image",
            str(vision_attention),
            "--allowed-local-media-path",
            str(fixture_root),
            "--cases-json",
            str(cases_path),
            "--report",
            str(load_report),
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            env={"HOME": str(isolated_home), "LANG": "C", "PATH": "/usr/bin:/bin"},
            timeout=args.timeout_seconds,
            check=False,
        )
    stdout_path.write_bytes(completed.stdout)
    stderr_path.write_bytes(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            f"native generation probe failed with code {completed.returncode}: "
            + completed.stderr.decode("utf-8", errors="replace")
        )
    try:
        probe = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("native generation probe emitted invalid JSON") from error
    checks = qualification_checks(
        probe, oracle, layer_oracle, prefill_state_oracle
    )
    checks["stderr_empty"] = len(completed.stderr) == 0
    exact_checks = [
        value
        for name, value in checks.items()
        if name.endswith("_native_top1_exact")
        or name.endswith("_selected_token_exact")
    ]
    kld_checks = [
        value for name, value in checks.items() if name.endswith("_kld_under_0_005")
    ]
    result_check_names = {
        name
        for name in checks
        if name.endswith("_native_top1_exact")
        or name.endswith("_selected_token_exact")
        or name.endswith("_kld_under_0_005")
    }
    setup_checks = {
        name: value
        for name, value in checks.items()
        if name not in result_check_names
    }
    qualified = all(checks.values())
    payload = seal_manifest(
        {
            "schema": SCHEMA,
            "captured_at": utc_now(),
            "complete": all(setup_checks.values()),
            "qualified": qualified,
            "scope": (
                "current-head-two-vl-tool-generation-logits-layers-and-prefill-state"
                if layer_oracle is not None and prefill_state_oracle is not None
                else "current-head-two-vl-tool-generation-divergence-logits-and-layers"
                if layer_oracle is not None
                else "current-head-two-vl-tool-generation-divergence-logits"
            ),
            "source": {
                **source,
                "files": [
                    file_component(path, path.relative_to(ROOT).as_posix())
                    for path in (
                        ROOT / "native/include/aima/native_http_server.h",
                        ROOT / "native/include/aima/bf16_gemm.h",
                        ROOT / "native/include/aima/native_decode_runner.h",
                        ROOT / "native/include/aima/native_full_layer.h",
                        ROOT / "native/include/aima/native_linear_layer.h",
                        ROOT / "native/include/aima/native_pointwise.h",
                        ROOT / "native/src/bf16_gemm.hip.cpp",
                        ROOT / "native/src/native_decode_runner.hip.cpp",
                        ROOT / "native/src/native_resident_engine.hip.cpp",
                        ROOT / "native/src/native_http_server.cpp",
                        ROOT / "native/src/native_full_layer.hip.cpp",
                        ROOT / "native/src/native_linear_layer.hip.cpp",
                        ROOT / "native/src/native_pointwise.hip.cpp",
                        ROOT / "native/src/main.cpp",
                        Path(__file__).resolve(),
                        ROOT / "aima_engine/vl_generation_oracle.py",
                        ROOT / "aima_engine/vl_generation_layer_oracle.py",
                        ROOT / "aima_engine/vl_prefill_state_oracle.py",
                        ROOT
                        / "native/aot/gfx1151/causal-conv-decode-v0.1.0/manifest.json",
                    )
                ],
            },
            "build_info": build_info,
            "binary": file_component(binary, "build/native/aima-engine-native"),
            "dependencies": {
                "fmha_provider": file_component(
                    fmha_provider, "${AIMA_FMHA_PROVIDER}"
                ),
                "vision_attention": file_component(
                    vision_attention, "${AIMA_VISION_ATTENTION_IMAGE}"
                ),
                "generation_oracle": file_component(
                    oracle_path,
                    "benchmarks/results/vl-generation-oracle-v0.1.0.json",
                ),
                "fixture_manifest": file_component(
                    fixture_root / "fixtures-manifest.json",
                    "benchmarks/fixtures/vl-capability-v0.1.0/fixtures-manifest.json",
                ),
                **(
                    {
                        "generation_layer_oracle": file_component(
                            layer_oracle_path,
                            "benchmarks/results/"
                            "vl-generation-layer-oracle-v0.1.0.json",
                        )
                    }
                    if layer_oracle_path is not None
                    else {}
                ),
                **(
                    {
                        "vl_prefill_state_oracle": file_component(
                            prefill_state_oracle_path,
                            "benchmarks/results/"
                            "vl-prefill-state-oracle-v0.1.0.json",
                        )
                    }
                    if prefill_state_oracle_path is not None
                    else {}
                ),
            },
            "launch": {
                "kind": "single-resident-native-qualification-probe",
                "command": [
                    "${AIMA_NATIVE_BINARY}",
                    "vl-generation-logits-probe",
                    "--model-dir",
                    "${AIMA_MODEL_DIR}",
                    "--context-tokens",
                    "1024",
                    "--cache-capacity",
                    "2048",
                    "--fmha-provider",
                    "${AIMA_FMHA_PROVIDER}",
                    "--vision-attention-image",
                    "${AIMA_VISION_ATTENTION_IMAGE}",
                    "--allowed-local-media-path",
                    "${AIMA_VL_FIXTURE_ROOT}",
                    "--cases-json",
                    "${AIMA_VL_GENERATION_CASES}",
                    "--report",
                    "${AIMA_NATIVE_WEIGHT_REPORT}",
                ],
            },
            "probe": probe,
            "checks": checks,
            "raw": {
                "stdout": file_component(
                    stdout_path, f"{raw_root.name}/probe.stdout.json"
                ),
                "stderr": file_component(
                    stderr_path, f"{raw_root.name}/probe.stderr.log"
                ),
                "weight_load": file_component(
                    load_report, f"{raw_root.name}/native-weight-load.json"
                ),
            },
            "decision": {
                "two_shared_prefixes_exact": all(
                    checks[f"{case_id}_prefix_exact"] for case_id in CASE_ORDER
                ),
                "two_reference_rows_bound": all(
                    checks[f"{case_id}_reference_row_bound"]
                    for case_id in CASE_ORDER
                ),
                "two_native_full_vocab_finite": all(
                    checks[f"{case_id}_full_vocabulary_finite"]
                    for case_id in CASE_ORDER
                ),
                "two_decode_boundary_sets_bound": (
                    layer_oracle is not None
                    and all(
                        checks[f"{case_id}_decode_boundary_rows_bound"]
                        for case_id in CASE_ORDER
                    )
                ),
                "two_decode_linear_boundary_sets_bound": (
                    layer_oracle is not None
                    and all(
                        checks[
                            f"{case_id}_decode_linear_boundary_rows_bound"
                        ]
                        for case_id in CASE_ORDER
                    )
                ),
                "two_decode_layer0_tail_boundary_sets_bound": (
                    layer_oracle is not None
                    and all(
                        checks[
                            f"{case_id}_decode_layer0_tail_boundary_rows_bound"
                        ]
                        for case_id in CASE_ORDER
                    )
                ),
                "two_prefill_state_sets_bound": (
                    prefill_state_oracle is not None
                    and all(
                        checks[f"{case_id}_prefill_state_rows_bound"]
                        for case_id in CASE_ORDER
                    )
                ),
                "two_native_generation_top1_exact": all(exact_checks),
                "two_generation_logits_kld_under_0_005": all(kld_checks),
                "g1_generation_closed": qualified,
                "g1_passed": False,
                "g2_passed": False,
                "g3_passed": False,
                "g4_passed": False,
                "g5_passed": False,
            },
        }
    )
    atomic_json(output, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--fmha-provider", type=Path, required=True)
    parser.add_argument("--vision-attention-image", type=Path, required=True)
    parser.add_argument("--fixture-root", type=Path, required=True)
    parser.add_argument("--oracle-manifest", type=Path, required=True)
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument("--layer-oracle-manifest", type=Path)
    parser.add_argument("--layer-oracle-root", type=Path)
    parser.add_argument("--prefill-state-oracle-manifest", type=Path)
    parser.add_argument("--prefill-state-oracle-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    return parser.parse_args()


def main() -> int:
    result = qualify(parse_args())
    print(
        json.dumps(
            {
                "output": result["schema"],
                "complete": result["complete"],
                "qualified": result["qualified"],
                "g1_generation_closed": result["decision"][
                    "g1_generation_closed"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
