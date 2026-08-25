#!/usr/bin/env python3
"""Run and seal the exact-source G5 repository, security and evidence gates."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aima_engine.public_hygiene import scan_bytes  # noqa: E402
from aima_engine.vl_reference import (  # noqa: E402
    atomic_json,
    file_component,
    seal_manifest,
    verify_manifest_integrity,
)


SCHEMA = "aima-amd395-qwen36/native-vl-release-gates/v1"
PRODUCT_SCHEMA = "aima-amd395-qwen36/native-vl-product-qualification/v1"


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(ROOT), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(arguments)} failed: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def run_gate(
    *,
    gate_id: str,
    command: list[str],
    output_dir: Path,
    required_text: str | None,
) -> dict[str, Any]:
    log_path = output_dir / "raw" / f"{gate_id}.log"
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    elapsed = time.monotonic() - started
    text = (
        log_path.read_text(encoding="utf-8", errors="replace")
        .replace(str(ROOT), "${AIMA_REPO_ROOT}")
        .replace(str(Path.home()), "${AIMA_HOME}")
    )
    log_path.write_text(text, encoding="utf-8")
    findings = scan_bytes(f"raw/{log_path.name}", text.encode("utf-8"))
    checks = {
        "exit_zero": completed.returncode == 0,
        "required_text_present": (
            required_text is None or required_text in text
        ),
        "public_hygiene": not findings,
    }
    return {
        "gate_id": gate_id,
        "command": command,
        "exit_code": completed.returncode,
        "elapsed_seconds": elapsed,
        "checks": checks,
        "qualified": all(checks.values()),
        "log": file_component(log_path, f"raw/{log_path.name}"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    product_result_path = args.product_result.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and (
        not output_dir.is_dir() or any(output_dir.iterdir())
    ):
        raise SystemExit(f"output directory must be empty: {output_dir}")
    if not product_result_path.is_file():
        raise SystemExit(f"product result is missing: {product_result_path}")
    product_result = load_object(product_result_path)
    if (
        product_result.get("schema") != PRODUCT_SCHEMA
        or product_result.get("complete") is not True
        or product_result.get("qualified") is not True
        or verify_manifest_integrity(product_result)
    ):
        raise SystemExit("native VL product result is incomplete or unsealed")
    source = product_result.get("components", {}).get("source", {})
    if not isinstance(source, dict):
        raise SystemExit("product result source identity is missing")
    release_commit = source.get("release_commit")
    release_tag = source.get("release_tag")
    if (
        not isinstance(release_commit, str)
        or not isinstance(release_tag, str)
        or git("rev-parse", "HEAD") != release_commit
        or git("rev-parse", "--verify", f"refs/tags/{release_tag}^{{commit}}")
        != release_commit
        or git("status", "--porcelain", "--untracked-files=normal")
    ):
        raise SystemExit("release checkout or immutable tag identity differs")

    expected_makefile = product_result.get("inputs", {}).get("makefile", {})
    actual_makefile = file_component(ROOT / "Makefile", "Makefile")
    if expected_makefile != actual_makefile:
        raise SystemExit("release Makefile differs from product qualification")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "raw").mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    gates = [
        run_gate(
            gate_id="make-check",
            command=["make", "check"],
            output_dir=output_dir,
            required_text="OK",
        ),
        run_gate(
            gate_id="make-security-scan",
            command=["make", "security-scan"],
            output_dir=output_dir,
            required_text=None,
        ),
        run_gate(
            gate_id="make-verify-evidence",
            command=["make", "verify-evidence"],
            output_dir=output_dir,
            required_text="release evidence",
        ),
    ]
    qualified = all(gate["qualified"] for gate in gates)
    payload = seal_manifest(
        {
            "schema": SCHEMA,
            "release": product_result["release"],
            "recorded_at": datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ),
            "started_at": started_at,
            "complete": True,
            "qualified": qualified,
            "source": source,
            "product_result": file_component(
                product_result_path,
                "share/aima/qualification.json",
            ),
            "protocol": {
                "clean_tagged_checkout": True,
                "commands": [
                    ["make", "check"],
                    ["make", "security-scan"],
                    ["make", "verify-evidence"],
                ],
                "logs_are_combined_stdout_stderr": True,
                "nonzero_is_blocking": True,
            },
            "gates": gates,
            "checks": {
                "all_commands_exit_zero": all(
                    gate["checks"]["exit_zero"] for gate in gates
                ),
                "all_gate_logs_public": all(
                    gate["checks"]["public_hygiene"] for gate in gates
                ),
                "all_required_markers_present": all(
                    gate["checks"]["required_text_present"] for gate in gates
                ),
                "release_tag_exact": True,
                "release_checkout_clean": True,
                "product_makefile_exact": True,
            },
            "decision": {
                "make_check_passed": gates[0]["qualified"],
                "make_security_scan_passed": gates[1]["qualified"],
                "make_verify_evidence_passed": gates[2]["qualified"],
                "g5_static_release_gates_passed": qualified,
            },
        }
    )
    output = output_dir / "release-gates.json"
    digest = atomic_json(output, payload)
    print(
        json.dumps(
            {
                "complete": True,
                "qualified": qualified,
                "output": str(output),
                "sha256": digest,
            },
            sort_keys=True,
        )
    )
    return 0 if qualified else 1


if __name__ == "__main__":
    raise SystemExit(main())
