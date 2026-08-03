"""Fail-closed binding between a release qualification and package inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping


REQUIRED_COMPONENTS = (
    "native_engine",
    "static_launcher",
    "aotriton_fmha_provider",
    "ck_fmha_provider",
    "q16384_hybrid_fmha_provider",
    "aotriton_runtime",
    "aotriton_gfx1151_image",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_package_qualification(
    qualification_path: Path,
    *,
    release: str,
    release_tag: str,
    source_commit: str,
    components: Mapping[str, Path],
) -> list[str]:
    errors: list[str] = []
    try:
        qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [f"cannot read qualification record: {error}"]
    if not isinstance(qualification, dict):
        return ["qualification record root must be an object"]
    if qualification.get("complete") is not True:
        errors.append("qualification record is not complete")
    if qualification.get("qualified") is not True:
        errors.append("qualification record is not qualified")
    if qualification.get("release") != release:
        errors.append("qualification release does not match package release")

    records = qualification.get("components")
    if not isinstance(records, dict):
        return errors + ["qualification components must be an object"]
    supplied = set(components)
    required = set(REQUIRED_COMPONENTS)
    if supplied != required:
        errors.append(
            "package component map must contain exactly: "
            + ", ".join(REQUIRED_COMPONENTS)
        )
    for name in REQUIRED_COMPONENTS:
        path = components.get(name)
        record = records.get(name)
        if path is None or not isinstance(record, dict):
            errors.append(f"qualification component is missing: {name}")
            continue
        if not path.is_file():
            errors.append(f"package component is missing: {name}")
            continue
        actual_size = path.stat().st_size
        if record.get("bytes") != actual_size:
            errors.append(f"qualification size mismatch: {name}")
        if record.get("sha256") != _sha256(path):
            errors.append(f"qualification SHA-256 mismatch: {name}")

    source = records.get("source")
    if isinstance(source, dict):
        if source.get("release_tag") != release_tag:
            errors.append("qualification release tag does not match package tag")
        if source.get("release_commit") != source_commit:
            errors.append("qualification release commit does not match checkout")
        if source.get("native_source_commit") != source_commit:
            errors.append("qualification native source commit does not match checkout")
    elif release != "1.3.0":
        errors.append("qualification source provenance is missing")
    return errors
