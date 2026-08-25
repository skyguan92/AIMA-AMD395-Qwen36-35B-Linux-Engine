"""Verification helpers for published native qualification evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from aima_engine.vl_reference import verify_manifest_integrity


DEFAULT_RELEASE = "1.5.0"
NATIVE_VL_RELEASE = "1.5.1-native-vl.3"
NATIVE_VL_RAW_IMMUTABLE_KEYS = {
    "g1",
    "g2",
    "g3",
    "g4",
    "envelope",
    "temperature_sampling",
}
RELEASE_RECORDS: dict[str, dict[str, Path]] = {
    NATIVE_VL_RELEASE: {
        "product_result": Path(
            "benchmarks/results/native-vl-g5-release-v1.5.1-native-vl.3.json"
        ),
        "bundle_result": Path(
            "benchmarks/results/"
            "native-portable-bundle-v1.5.1-native-vl.3.json"
        ),
        "provenance": Path(
            "benchmarks/results/"
            "native-release-provenance-v1.5.1-native-vl.3.json"
        ),
        "product_contract": Path(
            "native/product-contract-v1.5.1-native-vl.3.json"
        ),
        "package_input": Path(
            "benchmarks/results/"
            "native-portable-product-v1.5.1-native-vl.3.json"
        ),
        "g1": Path(
            "benchmarks/results/native-vl-g1-coverage-audit-v0.1.0.json"
        ),
        "g2": Path("benchmarks/results/vl-correctness-v0.1.0.json"),
        "g3": Path(
            "benchmarks/results/text-v151-nonregression-v0.1.0.json"
        ),
        "g4": Path("benchmarks/results/vl-performance-v0.1.0.json"),
        "envelope": Path(
            "benchmarks/results/native-vl-envelope-v0.1.0.json"
        ),
        "temperature_sampling": Path(
            "benchmarks/results/native-temperature-sampling-v0.1.0.json"
        ),
    },
    "1.5.0": {
        "product_result": Path(
            "benchmarks/results/native-portable-product-v1.5.0.json"
        ),
        "bundle_result": Path(
            "benchmarks/results/native-portable-bundle-v1.5.0.json"
        ),
        "provenance": Path(
            "benchmarks/results/native-release-provenance-v1.5.0.json"
        ),
        "product_contract": Path("native/product-contract-v1.5.0.json"),
    },
    "1.4.1": {
        "product_result": Path(
            "benchmarks/results/native-portable-product-v1.4.1.json"
        ),
        "bundle_result": Path(
            "benchmarks/results/native-portable-bundle-v1.4.1.json"
        ),
        "provenance": Path(
            "benchmarks/results/native-release-provenance-v1.4.1.json"
        ),
        "product_contract": Path("native/product-contract-v1.4.1.json"),
    },
    "1.4.0": {
        "product_result": Path(
            "benchmarks/results/native-portable-product-v1.4.0.json"
        ),
        "bundle_result": Path(
            "benchmarks/results/native-portable-bundle-v1.4.0.json"
        ),
        "provenance": Path(
            "benchmarks/results/native-release-provenance-v1.4.0.json"
        ),
        "product_contract": Path("native/product-contract-v1.4.0.json"),
    },
    "1.3.0": {
        "product_result": Path(
            "benchmarks/results/native-portable-product-v1.3.0.json"
        ),
        "bundle_result": Path(
            "benchmarks/results/native-portable-bundle-v1.3.0.json"
        ),
        "provenance": Path(
            "benchmarks/results/native-release-provenance-v1.3.0.json"
        ),
        "product_contract": Path("native/product-contract-v1.3.0.json"),
    },
}

CORE_PRODUCT_EVIDENCE_KEYS = {
    "matrix",
    "correctness",
    "surfaces",
    "openai_features",
}
PRODUCT_EVIDENCE_KEYS = {
    "1.5.1-native-vl.3": {
        "primary_bundle",
        "second_bundle",
        "resident_soak",
        "rollback",
        "release_gates",
    },
    "1.5.0": CORE_PRODUCT_EVIDENCE_KEYS | {"capability_eval"},
}
STANDALONE_EVIDENCE_KEYS = {
    "1.5.1-native-vl.3": {
        "g1_g2",
        "g1_generation_raw",
        "g3_correctness",
        "g3_doctor",
        "g3_mmlu",
        "g3_openai_features",
        "g3_product_surfaces",
        "g3_text_matrix",
        "g4_reference_availability_raw",
        "g4_vl_performance",
    },
    "1.5.0": {"portable_bundle", "second_host_compat"},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evidence_tree(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    total_bytes = 0
    files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    for candidate in files:
        total_bytes += candidate.stat().st_size
        line = f"{sha256(candidate)}  {candidate.relative_to(path).as_posix()}\n"
        digest.update(line.encode("utf-8"))
    return {
        "path": path.as_posix(),
        "file_count": len(files),
        "bytes": total_bytes,
        "tree_sha256": digest.hexdigest(),
    }


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"evidence JSON root is not an object: {path}")
    return value


def _public_mirror_path(value: str) -> Path | None:
    marker = "benchmarks/runs/"
    if marker in value:
        return Path(marker + value.split(marker, 1)[1])
    for prefix in ("${AIMA_REPO_ROOT}/output/", "output/"):
        if value.startswith(prefix):
            return Path("benchmarks/runs") / value.removeprefix(prefix)
    return None


def _resolve_recorded_path(root: Path, owner: Path, value: str) -> Path | None:
    mirrored = _public_mirror_path(value)
    if mirrored is not None:
        exact = root / mirrored
        if exact.is_file():
            return exact
        runs_root = root / "benchmarks/runs"
        try:
            owner_relative = owner.relative_to(runs_root)
            recorded_relative = mirrored.relative_to("benchmarks/runs")
        except ValueError:
            return exact
        if len(owner_relative.parts) > 1 and len(recorded_relative.parts) > 1:
            renamed_run = runs_root / owner_relative.parts[0]
            rebased = renamed_run.joinpath(*recorded_relative.parts[1:])
            if rebased.is_file():
                return rebased
        return exact
    if value.startswith("raw/"):
        return owner.parent / value
    if value.startswith("${AIMA_OUTPUT_DIR}/"):
        return owner.parent / value.removeprefix("${AIMA_OUTPUT_DIR}/")
    if value.startswith(("${", "/")):
        return None
    return owner.parent / value


def _is_public_artifact_path(value: str) -> bool:
    return bool(
        value.startswith(("raw/", "output/", "${AIMA_OUTPUT_DIR}/"))
        or "-raw/" in value
        or "benchmarks/runs/" in value
    )


def _recorded_artifact_paths(root: Path, owner: Path, value: Any) -> list[Path]:
    paths: list[Path] = []
    if isinstance(value, list):
        for item in value:
            paths.extend(_recorded_artifact_paths(root, owner, item))
        return paths
    if not isinstance(value, dict):
        return paths
    recorded_path = value.get("path")
    digest = value.get("sha256")
    if (
        isinstance(recorded_path, str)
        and isinstance(digest, str)
        and _is_public_artifact_path(recorded_path)
    ):
        resolved = _resolve_recorded_path(root, owner, recorded_path)
        if resolved is not None:
            paths.append(resolved)
    for nested in value.values():
        paths.extend(_recorded_artifact_paths(root, owner, nested))
    return paths


def _verify_recorded_artifacts(root: Path, owner: Path, value: Any) -> list[str]:
    errors: list[str] = []
    if isinstance(value, list):
        for item in value:
            errors.extend(_verify_recorded_artifacts(root, owner, item))
        return errors
    if not isinstance(value, dict):
        return errors

    recorded_path = value.get("path")
    component_digest = value.get("sha256")
    if (
        isinstance(recorded_path, str)
        and isinstance(component_digest, str)
        and _is_public_artifact_path(recorded_path)
    ):
        path = _resolve_recorded_path(root, owner, recorded_path)
        if path is not None:
            if not path.is_file():
                errors.append(f"missing evidence artifact: {path.relative_to(root)}")
            elif sha256(path) != component_digest:
                errors.append(f"evidence hash mismatch: {path.relative_to(root)}")

    pairs = (
        ("report", "report_sha256"),
        ("reports", "report_sha256"),
        ("server_reports", "server_report_sha256"),
        ("load_report", "load_report_sha256"),
        ("response", "response_sha256"),
    )
    for path_key, digest_key in pairs:
        paths = value.get(path_key)
        digests = value.get(digest_key)
        if paths is None or digests is None:
            continue
        if isinstance(paths, str):
            paths = [paths]
        if isinstance(digests, str):
            digests = [digests]
        if (
            not isinstance(paths, list)
            or not isinstance(digests, list)
            or len(paths) != len(digests)
        ):
            errors.append(f"{owner}: malformed {path_key}/{digest_key} pair")
            continue
        for recorded_path, expected in zip(paths, digests, strict=True):
            if not isinstance(recorded_path, str) or not isinstance(expected, str):
                errors.append(f"{owner}: non-string {path_key}/{digest_key} value")
                continue
            path = _resolve_recorded_path(root, owner, recorded_path)
            if path is None:
                continue
            if not path.is_file():
                errors.append(f"missing evidence artifact: {path.relative_to(root)}")
            elif sha256(path) != expected:
                errors.append(f"evidence hash mismatch: {path.relative_to(root)}")
    for nested in value.values():
        errors.extend(_verify_recorded_artifacts(root, owner, nested))
    return errors


def _release_record(release: str) -> dict[str, Path] | None:
    return RELEASE_RECORDS.get(release)


def verify_release_evidence(
    root: Path, release: str = DEFAULT_RELEASE
) -> list[str]:
    root = root.resolve()
    release_record = _release_record(release)
    if release_record is None:
        return [f"unsupported release evidence: {release}"]

    public_result = _load(root / release_record["product_result"])
    bundle_result = _load(root / release_record["bundle_result"])
    provenance = _load(root / release_record["provenance"])
    errors: list[str] = []

    if release == NATIVE_VL_RELEASE:
        for label, value in (
            ("product result", public_result),
            ("portable bundle result", bundle_result),
            ("release provenance", provenance),
        ):
            for error in verify_manifest_integrity(value):
                errors.append(f"{label} integrity failed: {error}")
        sealed_paths = [
            release_record["product_result"],
            release_record["bundle_result"],
            release_record["provenance"],
            release_record["package_input"],
            release_record["g1"],
            release_record["g2"],
            release_record["g3"],
            release_record["g4"],
            release_record["envelope"],
            release_record["temperature_sampling"],
        ]
        for relative in sealed_paths:
            path = root / relative
            sidecar = path.with_name(path.name + ".sha256")
            expected = f"{sha256(path)}  {path.name}\n" if path.is_file() else ""
            if not sidecar.is_file() or sidecar.read_text(
                encoding="utf-8"
            ) != expected:
                errors.append(f"sealed record sidecar differs: {relative}")
            if path.is_file():
                for error in verify_manifest_integrity(_load(path)):
                    errors.append(f"sealed record integrity failed: {relative}: {error}")

    for label, value in (
        ("product result", public_result),
        ("portable bundle result", bundle_result),
        ("release provenance", provenance),
    ):
        if value.get("release") != release:
            errors.append(f"{label} release mismatch")
        if value.get("complete") is not True:
            errors.append(f"{label} is incomplete")
    if public_result.get("qualified") is not True:
        errors.append("product result is not qualified")
    if bundle_result.get("qualified") is not True:
        errors.append("portable bundle result is not qualified")

    expected_immutable = {
        "product_result": release_record["product_result"],
        "portable_bundle_result": release_record["bundle_result"],
        "product_contract": release_record["product_contract"],
    }
    if "package_input" in release_record:
        expected_immutable["package_input_qualification"] = release_record[
            "package_input"
        ]
    for gate in ("g1", "g2", "g3", "g4", "envelope"):
        if gate in release_record:
            expected_immutable[gate] = release_record[gate]
    if "temperature_sampling" in release_record:
        expected_immutable["temperature_sampling"] = release_record[
            "temperature_sampling"
        ]
    immutable_records = provenance.get("immutable_records")
    if not isinstance(immutable_records, dict):
        errors.append("immutable release records are missing")
    else:
        if set(immutable_records) != set(expected_immutable):
            errors.append("immutable release records are incomplete or unexpected")
        for key, expected_path in expected_immutable.items():
            record = immutable_records.get(key)
            if not isinstance(record, dict):
                errors.append(f"immutable release record is missing: {key}")
                continue
            if record.get("path") != expected_path.as_posix():
                errors.append(f"immutable release record path mismatch: {key}")
                continue
            path = root / expected_path
            if not path.is_file():
                errors.append(f"missing immutable release record: {expected_path}")
            elif sha256(path) != record.get("sha256"):
                errors.append(f"immutable release record changed: {expected_path}")
            elif (
                release == NATIVE_VL_RELEASE
                and key in NATIVE_VL_RAW_IMMUTABLE_KEYS
            ):
                errors.extend(
                    _verify_recorded_artifacts(root, path, _load(path))
                )

    public_evidence = provenance.get("public_evidence")
    product_evidence_keys = PRODUCT_EVIDENCE_KEYS.get(
        release, CORE_PRODUCT_EVIDENCE_KEYS
    )
    standalone_evidence_keys = STANDALONE_EVIDENCE_KEYS.get(
        release, {"portable_bundle"}
    )
    expected_keys = product_evidence_keys | standalone_evidence_keys
    if not isinstance(public_evidence, dict) or set(public_evidence) != expected_keys:
        errors.append("public evidence records are missing or incomplete")
        public_evidence = {}

    for key in product_evidence_keys:
        provenance_record = public_evidence.get(key)
        result_record = public_result.get("evidence", {}).get(key)
        if not isinstance(provenance_record, dict) or not isinstance(result_record, dict):
            errors.append(f"public evidence record is missing: {key}")
            continue
        recorded_path = result_record.get("path")
        mirrored = (
            _public_mirror_path(recorded_path)
            if isinstance(recorded_path, str)
            else None
        )
        if mirrored is None or mirrored.as_posix() != provenance_record.get("path"):
            errors.append(f"public result path mismatch: {key}")
        if result_record.get("sha256") != provenance_record.get("sha256"):
            errors.append(f"provenance evidence mismatch: {key}")
        path = root / str(provenance_record.get("path", ""))
        if not path.is_file():
            errors.append(f"missing evidence summary: {provenance_record.get('path')}")
        elif sha256(path) != provenance_record.get("sha256"):
            errors.append(f"evidence summary hash mismatch: {key}")

    for key in standalone_evidence_keys:
        record = public_evidence.get(key)
        if not isinstance(record, dict):
            errors.append(f"standalone evidence record is missing: {key}")
            continue
        summary = root / str(record.get("path", ""))
        if not summary.is_file():
            errors.append(f"missing evidence summary: {record.get('path')}")
        elif sha256(summary) != record.get("sha256"):
            errors.append(f"standalone evidence hash mismatch: {key}")

    for record in public_evidence.values():
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            continue
        path = root / record["path"]
        if path.is_file():
            errors.extend(_verify_recorded_artifacts(root, path, _load(path)))
        tree_path = record.get("tree_path")
        directory = (
            root / tree_path
            if isinstance(tree_path, str)
            else path.parent
        )
        if directory.is_dir():
            for raw_json in sorted(directory.rglob("*.json")):
                errors.extend(_verify_recorded_artifacts(root, raw_json, _load(raw_json)))

    tree_records = provenance.get("public_evidence_trees")
    if not isinstance(tree_records, dict) or set(tree_records) != expected_keys:
        errors.append("public evidence tree records are missing or incomplete")
    else:
        for key, record in public_evidence.items():
            if not isinstance(record, dict) or not isinstance(record.get("path"), str):
                continue
            tree_path = record.get("tree_path")
            directory = (
                root / tree_path
                if isinstance(tree_path, str)
                else (root / record["path"]).parent
            )
            actual = evidence_tree(directory)
            actual["path"] = directory.relative_to(root).as_posix()
            if tree_records.get(key) != actual:
                errors.append(f"public evidence tree mismatch: {key}")
    return sorted(set(errors))


def evidence_paths(root: Path, release: str = DEFAULT_RELEASE) -> list[Path]:
    root = root.resolve()
    release_record = _release_record(release)
    if release_record is None:
        raise ValueError(f"unsupported release evidence: {release}")
    provenance_path = root / release_record["provenance"]
    provenance = _load(provenance_path)
    paths = [provenance_path]
    if release == NATIVE_VL_RELEASE:
        provenance_sidecar = provenance_path.with_name(
            provenance_path.name + ".sha256"
        )
        if provenance_sidecar.is_file():
            paths.append(provenance_sidecar)
    for key, record in provenance["immutable_records"].items():
        path = root / record["path"]
        paths.append(path)
        if release == NATIVE_VL_RELEASE:
            sidecar = path.with_name(path.name + ".sha256")
            if sidecar.is_file():
                paths.append(sidecar)
            if key in NATIVE_VL_RAW_IMMUTABLE_KEYS and path.is_file():
                paths.extend(_recorded_artifact_paths(root, path, _load(path)))
    for record in provenance["public_evidence"].values():
        summary = root / record["path"]
        tree_path = record.get("tree_path")
        tree = (
            root / tree_path
            if isinstance(tree_path, str)
            else summary.parent
        )
        paths.append(tree)
        try:
            summary.relative_to(tree)
        except ValueError:
            paths.append(summary)
            sidecar = summary.with_name(summary.name + ".sha256")
            if sidecar.is_file():
                paths.append(sidecar)
    return list(dict.fromkeys(paths))
