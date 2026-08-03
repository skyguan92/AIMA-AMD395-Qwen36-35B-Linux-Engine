"""Verification helpers for the public v1.3 qualification evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PUBLIC_RESULT = Path("benchmarks/results/native-portable-product-v1.3.0.json")
BUNDLE_RESULT = Path("benchmarks/results/native-portable-bundle-v1.3.0.json")
PROVENANCE_RESULT = Path(
    "benchmarks/results/native-release-provenance-v1.3.0.json"
)
EVIDENCE_SUMMARIES = {
    "matrix": Path("benchmarks/runs/native-full-matrix-20260723-v130/matrix.json"),
    "correctness": Path(
        "benchmarks/runs/native-correctness-20260723-v130/correctness.json"
    ),
    "surfaces": Path(
        "benchmarks/runs/native-product-surfaces-20260723-v130/surfaces.json"
    ),
    "openai_features": Path(
        "benchmarks/runs/native-openai-features-20260723-v130/features.json"
    ),
    "portable_bundle": Path(
        "benchmarks/runs/native-portable-bundle-20260723-v130/bundle.json"
    ),
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
    files = sorted(
        candidate for candidate in path.rglob("*") if candidate.is_file()
    )
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


def _resolve_recorded_path(root: Path, owner: Path, value: str) -> Path | None:
    marker = "benchmarks/runs/"
    if marker in value:
        return root / marker / value.split(marker, 1)[1]
    if value.startswith("raw/"):
        return owner.parent / value
    if value.startswith("${AIMA_OUTPUT_DIR}/"):
        return owner.parent / value.removeprefix("${AIMA_OUTPUT_DIR}/")
    if value.startswith(("${", "/")):
        return None
    return owner.parent / value


def _verify_recorded_artifacts(root: Path, owner: Path, value: Any) -> list[str]:
    errors: list[str] = []
    if isinstance(value, list):
        for item in value:
            errors.extend(_verify_recorded_artifacts(root, owner, item))
        return errors
    if not isinstance(value, dict):
        return errors

    pairs = (
        ("report", "report_sha256"),
        ("reports", "report_sha256"),
        ("server_reports", "server_report_sha256"),
        ("load_report", "load_report_sha256"),
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
        if not isinstance(paths, list) or not isinstance(digests, list) or len(paths) != len(digests):
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


def verify_release_evidence(root: Path) -> list[str]:
    root = root.resolve()
    public_result = _load(root / PUBLIC_RESULT)
    provenance = _load(root / PROVENANCE_RESULT)
    errors: list[str] = []

    for record in provenance["immutable_records"].values():
        path = root / record["path"]
        if not path.is_file():
            errors.append(f"missing immutable release record: {record['path']}")
        elif sha256(path) != record["sha256"]:
            errors.append(f"immutable release record changed: {record['path']}")

    for key in ("matrix", "correctness", "surfaces", "openai_features"):
        path = root / EVIDENCE_SUMMARIES[key]
        record = public_result["evidence"][key]
        provenance_record = provenance["public_evidence"][key]
        if record["path"] != EVIDENCE_SUMMARIES[key].as_posix():
            errors.append(f"public result path mismatch: {key}")
        if provenance_record != record:
            errors.append(f"provenance evidence mismatch: {key}")
        if not path.is_file():
            errors.append(f"missing evidence summary: {EVIDENCE_SUMMARIES[key]}")
        elif sha256(path) != record["sha256"]:
            errors.append(f"evidence summary hash mismatch: {key}")

    bundle_record = provenance["public_evidence"]["portable_bundle"]
    bundle_summary = root / EVIDENCE_SUMMARIES["portable_bundle"]
    if bundle_record.get("path") != EVIDENCE_SUMMARIES["portable_bundle"].as_posix():
        errors.append("portable bundle evidence path mismatch")
    if not bundle_summary.is_file():
        errors.append(f"missing evidence summary: {EVIDENCE_SUMMARIES['portable_bundle']}")
    elif sha256(bundle_summary) != bundle_record.get("sha256"):
        errors.append("portable bundle evidence hash mismatch")

    for relative in EVIDENCE_SUMMARIES.values():
        path = root / relative
        if path.is_file():
            errors.extend(_verify_recorded_artifacts(root, path, _load(path)))
        directory = path.parent
        if directory.is_dir():
            for raw_json in sorted(directory.rglob("*.json")):
                errors.extend(_verify_recorded_artifacts(root, raw_json, _load(raw_json)))
    tree_records = provenance.get("public_evidence_trees")
    if not isinstance(tree_records, dict):
        errors.append("public evidence tree records are missing")
    else:
        if set(tree_records) != set(EVIDENCE_SUMMARIES):
            errors.append("public evidence tree record keys mismatch")
        for key, summary in EVIDENCE_SUMMARIES.items():
            directory = root / summary.parent
            actual = evidence_tree(directory)
            actual["path"] = summary.parent.as_posix()
            if tree_records.get(key) != actual:
                errors.append(f"public evidence tree mismatch: {key}")
    return sorted(set(errors))


def evidence_paths(root: Path) -> list[Path]:
    root = root.resolve()
    paths = [
        root / PUBLIC_RESULT,
        root / BUNDLE_RESULT,
        root / PROVENANCE_RESULT,
        root / "native/product-contract.json",
    ]
    paths.extend(root / relative.parent for relative in EVIDENCE_SUMMARIES.values())
    return paths
