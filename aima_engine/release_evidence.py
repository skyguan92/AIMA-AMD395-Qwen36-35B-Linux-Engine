"""Verification helpers for published native qualification evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from aima_engine.vl_reference import verify_manifest_integrity
from aima_engine.qualification_runtime import expected_binding


DEFAULT_RELEASE = "1.5.1-native-vl.7"
NATIVE_VL_RELEASE = "1.5.1-native-vl.4"
PATCH_VL_RELEASE = "1.5.1-native-vl.5"
PATCH_VL_RELEASE_COMMIT = "eb7d8ac30cea4401a068fd25f1f1379c72eaf448"
PATCH_VL_NATIVE_COMMIT = "06a35e36269a9fe443c56e99c5fedf7ca25304cc"
PATCH_VL_ENGINE_SHA256 = (
    "1138a62b9515118a1237849bfe02ea8daeccec94d88a92e49c885775619bf829"
)
PATCH_VL_ARCHIVE_SHA256 = (
    "59f30c4232b8459f3efcd7b8506cc71b957614c0aac1fa96a2eb4e15f52940a3"
)
PATCH_VL_IDENTITIES = {
    "1.5.1-native-vl.7": {
        "release_commit": "9bd8a0fabcf2fc6ef1b882c10b04390c0e31fb00",
        "native_source_commit": "edb584ee16f0ee1fc5f902459598b9446d96387e",
        "engine_sha256": "1de6f8f1b1300ef9ecd300a783e87b289b520a78e00429c2da04e404a3d7d850",
        "archive_sha256": "fb88242d52b6d3a1c152c8c7119c641275090fb7210e672d6c3fae358d215303",
        "archive_name": "aima-engine-native-portable-5a699d1600cc.tar.zst",
    },
    PATCH_VL_RELEASE: {
        "release_commit": PATCH_VL_RELEASE_COMMIT,
        "native_source_commit": PATCH_VL_NATIVE_COMMIT,
        "engine_sha256": PATCH_VL_ENGINE_SHA256,
        "archive_sha256": PATCH_VL_ARCHIVE_SHA256,
        "archive_name": "aima-engine-native-portable-194f2a673904.tar.zst",
    },
    "1.5.1-native-vl.6": {
        "release_commit": "1aac61df50cc686b9982cff4c9023b68237652d0",
        "native_source_commit": "66c6cf5ad8b3337dca9b85b0f46b23a0d222cdaa",
        "engine_sha256": "f224f79b86c4dac14402e893da0e6a07ca383e1afd905c1e070b7fcd104f46f4",
        "archive_sha256": "9acc3f91c5d7f7ebff617dcb642489a39c249d1d7ddece67bc3a0de1fcbd0f4c",
        "archive_name": "aima-engine-native-portable-290d1d084f9b.tar.zst",
    },
}
SEALED_RELEASES = {NATIVE_VL_RELEASE, *PATCH_VL_IDENTITIES}
NATIVE_VL_RAW_IMMUTABLE_KEYS = {
    "g1",
    "g2",
    "g3",
    "g4",
    "envelope",
    "temperature_sampling",
}
# These raw components were sealed into the completed native-VL release before
# the repository-wide ``*.log`` ignore rule was noticed.  They live in the
# separately published evidence archive, but a Git source checkout intentionally
# does not contain them (one log also contains builder-local paths).  Keep the
# exact archive identities here so source verification can project the sealed
# tree without accepting any other missing path or digest.
NATIVE_VL_ARCHIVE_ONLY_COMPONENTS: dict[Path, tuple[str, int]] = {
    Path(
        "benchmarks/results/native-vl-envelope-v0.1.0-raw/"
        "processor-probe.stderr.log"
    ): (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        0,
    ),
    Path(
        "benchmarks/results/native-vl-envelope-v0.1.0-raw/"
        "processor-probe.stdout.log"
    ): (
        "058478823fa1d09b40fdd1587f469ae8355530b05fc31c3d56aa33011075bedd",
        31,
    ),
    Path(
        "benchmarks/results/native-vl-envelope-v0.1.0-raw/server.stderr.log"
    ): (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        0,
    ),
    Path(
        "benchmarks/results/native-vl-envelope-v0.1.0-raw/server.stdout.log"
    ): (
        "e39b4b98321c8ab4194dd7253fb2a54ad6d169d4440fb75ecba27f93063bfbea",
        2153,
    ),
    Path(
        "benchmarks/results/native-vl-envelope-v0.1.0-raw/"
        "vision-probe.stderr.log"
    ): (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        0,
    ),
    Path(
        "benchmarks/results/native-vl-generation-current-head-v0.1.0-raw/"
        "probe.stderr.log"
    ): (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        0,
    ),
}
RELEASE_RECORDS: dict[str, dict[str, Path]] = {
    PATCH_VL_RELEASE: {
        "product_result": Path(
            "benchmarks/results/native-vl-g5-release-v1.5.1-native-vl.5.json"
        ),
        "bundle_result": Path(
            "benchmarks/results/"
            "native-portable-bundle-v1.5.1-native-vl.5.json"
        ),
        "provenance": Path(
            "benchmarks/results/"
            "native-release-provenance-v1.5.1-native-vl.5.json"
        ),
        "product_contract": Path(
            "native/product-contract-v1.5.1-native-vl.5.json"
        ),
        "package_input": Path(
            "benchmarks/results/"
            "native-portable-product-v1.5.1-native-vl.5.json"
        ),
        "chat_protocol": Path(
            "benchmarks/results/"
            "native-chat-protocol-v1.5.1-native-vl.5.json"
        ),
        "http_control_plane": Path(
            "benchmarks/results/"
            "native-http-control-plane-v1.5.1-native-vl.5.json"
        ),
        "archive_manifest": Path(
            "benchmarks/results/"
            "native-portable-manifest-v1.5.1-native-vl.5.json"
        ),
        "archive_checksum": Path(
            "benchmarks/results/"
            "aima-engine-native-portable-194f2a673904.tar.zst.sha256"
        ),
        "baseline_g5": Path(
            "benchmarks/results/native-vl-g5-release-v1.5.1-native-vl.4.json"
        ),
        "baseline_package_input": Path(
            "benchmarks/results/"
            "native-portable-product-v1.5.1-native-vl.4.json"
        ),
    },
    NATIVE_VL_RELEASE: {
        "product_result": Path(
            "benchmarks/results/native-vl-g5-release-v1.5.1-native-vl.4.json"
        ),
        "bundle_result": Path(
            "benchmarks/results/"
            "native-portable-bundle-v1.5.1-native-vl.4.json"
        ),
        "provenance": Path(
            "benchmarks/results/"
            "native-release-provenance-v1.5.1-native-vl.4.json"
        ),
        "product_contract": Path(
            "native/product-contract-v1.5.1-native-vl.4.json"
        ),
        "package_input": Path(
            "benchmarks/results/"
            "native-portable-product-v1.5.1-native-vl.4.json"
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
    PATCH_VL_RELEASE: {
        "bundle",
        "resident_soak",
        "rollback",
        "release_gates",
    },
    "1.5.1-native-vl.4": {
        "primary_bundle",
        "second_bundle",
        "resident_soak",
        "rollback",
        "release_gates",
    },
    "1.5.0": CORE_PRODUCT_EVIDENCE_KEYS | {"capability_eval"},
}
STANDALONE_EVIDENCE_KEYS = {
    PATCH_VL_RELEASE: {
        "chat_protocol",
        "http_control_plane",
    },
    "1.5.1-native-vl.4": {
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

for _patch_release, _patch_identity in PATCH_VL_IDENTITIES.items():
    if _patch_release == PATCH_VL_RELEASE:
        continue
    RELEASE_RECORDS[_patch_release] = {
        key: Path(str(path).replace(PATCH_VL_RELEASE, _patch_release))
        for key, path in RELEASE_RECORDS[PATCH_VL_RELEASE].items()
    }
    RELEASE_RECORDS[_patch_release]["archive_checksum"] = Path(
        "benchmarks/results/" + _patch_identity["archive_name"] + ".sha256"
    )
    RELEASE_RECORDS[_patch_release]["baseline_runtime_manifest"] = (
        RELEASE_RECORDS[PATCH_VL_RELEASE]["archive_manifest"]
    )
    PRODUCT_EVIDENCE_KEYS[_patch_release] = (
        PRODUCT_EVIDENCE_KEYS[PATCH_VL_RELEASE].copy()
    )
    STANDALONE_EVIDENCE_KEYS[_patch_release] = (
        STANDALONE_EVIDENCE_KEYS[PATCH_VL_RELEASE].copy()
    )
    if _patch_release == "1.5.1-native-vl.7":
        STANDALONE_EVIDENCE_KEYS[_patch_release].update(
            {"prefix", "one_owner", "text_matrix"}
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evidence_tree(
    path: Path,
    *,
    virtual_components: dict[Path, tuple[str, int]] | None = None,
) -> dict[str, Any]:
    digest = hashlib.sha256()
    records = {
        candidate.relative_to(path): (sha256(candidate), candidate.stat().st_size)
        for candidate in path.rglob("*")
        if candidate.is_file()
    }
    for relative, component in (virtual_components or {}).items():
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"virtual evidence component escapes tree: {relative}")
        records.setdefault(relative, component)

    total_bytes = sum(size for _, size in records.values())
    for relative in sorted(records):
        component_digest, _ = records[relative]
        line = f"{component_digest}  {relative.as_posix()}\n"
        digest.update(line.encode("utf-8"))
    return {
        "path": path.as_posix(),
        "file_count": len(records),
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
            recorded_run = recorded_relative.parts[0]
            recorded_tail = recorded_relative.parts[1:]
            owner_run = owner_relative.parts[0]
            if owner_run == recorded_run or owner_run.startswith(
                recorded_run + "-"
            ):
                rebased = runs_root / owner_run
                rebased = rebased.joinpath(*recorded_tail)
                if rebased.is_file():
                    return rebased
            candidates = sorted(
                run_directory.joinpath(*recorded_tail)
                for run_directory in runs_root.glob(recorded_run + "-*")
                if run_directory.joinpath(*recorded_tail).is_file()
            )
            if len(candidates) == 1:
                return candidates[0]
            if len(candidates) > 1:
                owner_parts = owner_run.split("-")

                def suffix_score(candidate: Path) -> int:
                    candidate_run = candidate.relative_to(runs_root).parts[0]
                    candidate_parts = candidate_run.split("-")
                    score = 0
                    for owner_part, candidate_part in zip(
                        reversed(owner_parts), reversed(candidate_parts)
                    ):
                        if owner_part != candidate_part:
                            break
                        score += 1
                    return score

                scores = {candidate: suffix_score(candidate) for candidate in candidates}
                maximum = max(scores.values())
                matches = [
                    candidate
                    for candidate, score in scores.items()
                    if score == maximum and score > 0
                ]
                if len(matches) == 1:
                    return matches[0]
        return exact

    relative: Path
    if value.startswith("raw/"):
        relative = Path(value)
    elif value.startswith("${AIMA_OUTPUT_DIR}/"):
        relative = Path(value.removeprefix("${AIMA_OUTPUT_DIR}/"))
    elif value.startswith(("${", "/")):
        return None
    else:
        relative = Path(value)

    # Raw reports often repeat paths rooted at the qualification output while
    # also embedding those same paths in nested JSON reports. Resolve from the
    # closest enclosing evidence root instead of blindly duplicating `raw/` at
    # the nested report's directory.
    owner_parent = owner.parent.resolve()
    root = root.resolve()
    for base in (owner_parent, *owner_parent.parents):
        try:
            base.relative_to(root)
        except ValueError:
            continue
        candidate = (base / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            continue
        if candidate.is_file():
            return candidate
    return owner_parent / relative


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


def _matches_archive_only_component(
    root: Path,
    path: Path,
    expected_digest: str,
    expected_bytes: Any,
    archive_only_components: dict[Path, tuple[str, int]] | None,
) -> bool:
    if archive_only_components is None:
        return False
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    component = archive_only_components.get(relative)
    if component is None or component[0] != expected_digest:
        return False
    return expected_bytes is None or expected_bytes == component[1]


def _virtual_components_for_tree(
    root: Path,
    tree: Path,
    archive_only_components: dict[Path, tuple[str, int]] | None,
) -> dict[Path, tuple[str, int]]:
    if archive_only_components is None:
        return {}
    root = root.resolve()
    tree = tree.resolve()
    virtual: dict[Path, tuple[str, int]] = {}
    for repository_relative, component in archive_only_components.items():
        try:
            tree_relative = (root / repository_relative).resolve().relative_to(tree)
        except ValueError:
            continue
        virtual[tree_relative] = component
    return virtual


def _verify_recorded_artifacts(
    root: Path,
    owner: Path,
    value: Any,
    *,
    archive_only_components: dict[Path, tuple[str, int]] | None = None,
) -> list[str]:
    root = root.resolve()
    owner = owner.resolve()
    errors: list[str] = []
    if isinstance(value, list):
        for item in value:
            errors.extend(
                _verify_recorded_artifacts(
                    root,
                    owner,
                    item,
                    archive_only_components=archive_only_components,
                )
            )
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
                if not _matches_archive_only_component(
                    root,
                    path,
                    component_digest,
                    value.get("bytes"),
                    archive_only_components,
                ):
                    errors.append(
                        f"missing evidence artifact: {path.relative_to(root)}"
                    )
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
        elif path_key == "response" and not (
            isinstance(paths, list)
            and all(isinstance(path, str) for path in paths)
        ):
            # HTTP evidence uses response/response_sha256 for an inline JSON
            # object and its canonical-content digest as well as for response
            # artifact paths. Only the latter belongs to the file verifier.
            continue
        elif not isinstance(paths, list):
            errors.append(f"{owner}: malformed {path_key}/{digest_key} pair")
            continue
        if isinstance(digests, str):
            digests = [digests]
        if not isinstance(digests, list) or len(paths) != len(digests):
            errors.append(f"{owner}: malformed {path_key}/{digest_key} pair")
            continue
        # Length equality is checked immediately above; avoid Python 3.10's
        # ``zip(strict=...)`` so evidence verification also runs on Python 3.9.
        for recorded_path, expected in zip(paths, digests):
            if not isinstance(recorded_path, str) or not isinstance(expected, str):
                errors.append(f"{owner}: non-string {path_key}/{digest_key} value")
                continue
            path = _resolve_recorded_path(root, owner, recorded_path)
            if path is None:
                continue
            if not path.is_file():
                if not _matches_archive_only_component(
                    root,
                    path,
                    expected,
                    None,
                    archive_only_components,
                ):
                    errors.append(
                        f"missing evidence artifact: {path.relative_to(root)}"
                    )
            elif sha256(path) != expected:
                errors.append(f"evidence hash mismatch: {path.relative_to(root)}")
    for nested in value.values():
        errors.extend(
            _verify_recorded_artifacts(
                root,
                owner,
                nested,
                archive_only_components=archive_only_components,
            )
        )
    return errors


def _release_record(release: str) -> dict[str, Path] | None:
    return RELEASE_RECORDS.get(release)


def verify_release_evidence(
    root: Path,
    release: str = DEFAULT_RELEASE,
    *,
    require_archived_components: bool = False,
) -> list[str]:
    """Verify sealed evidence, optionally requiring archive-only raw components.

    A Git source checkout can validate the completed native-VL evidence using
    exact digest-bound projections for the six historical ``*.log`` components.
    Evidence archive creation sets ``require_archived_components`` so the files
    themselves must be mounted and pass their recorded hashes.
    """
    root = root.resolve()
    release_record = _release_record(release)
    if release_record is None:
        return [f"unsupported release evidence: {release}"]

    archive_only_components = (
        NATIVE_VL_ARCHIVE_ONLY_COMPONENTS
        if release == NATIVE_VL_RELEASE and not require_archived_components
        else None
    )

    public_result = _load(root / release_record["product_result"])
    bundle_result = _load(root / release_record["bundle_result"])
    provenance = _load(root / release_record["provenance"])
    errors: list[str] = []

    if release in SEALED_RELEASES:
        for label, value in (
            ("product result", public_result),
            ("portable bundle result", bundle_result),
            ("release provenance", provenance),
        ):
            for error in verify_manifest_integrity(value):
                errors.append(f"{label} integrity failed: {error}")
        if release == NATIVE_VL_RELEASE:
            integrity_paths = [
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
            sidecar_paths = integrity_paths
        else:
            integrity_paths = [
                release_record["product_result"],
                release_record["bundle_result"],
                release_record["provenance"],
                release_record["package_input"],
                release_record["chat_protocol"],
                release_record["http_control_plane"],
            ]
            sidecar_paths = [
                *integrity_paths,
                release_record["archive_manifest"],
            ]
        for relative in sidecar_paths:
            path = root / relative
            sidecar = path.with_name(path.name + ".sha256")
            expected = f"{sha256(path)}  {path.name}\n" if path.is_file() else ""
            if not sidecar.is_file() or sidecar.read_text(
                encoding="utf-8"
            ) != expected:
                errors.append(f"sealed record sidecar differs: {relative}")
            if path.is_file() and relative in integrity_paths:
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
    if release in PATCH_VL_IDENTITIES:
        identity = PATCH_VL_IDENTITIES[release]
        source = public_result.get("source", {})
        package_input = _load(root / release_record["package_input"])
        manifest = _load(root / release_record["archive_manifest"])
        manifest_files = {
            str(item.get("path")): item
            for item in manifest.get("files", [])
            if isinstance(item, dict)
        }
        expected_source = {
            "native_source_commit": identity["native_source_commit"],
            "native_source_dirty": False,
            "release_commit": identity["release_commit"],
            "release_tag": f"v{release}",
        }
        if source != expected_source:
            errors.append("patch release source identity differs")
        if (
            public_result.get("archive", {}).get("sha256")
            != identity["archive_sha256"]
            or bundle_result.get("archive", {}).get("sha256")
            != identity["archive_sha256"]
        ):
            errors.append("patch release archive identity differs")
        if (
            public_result.get("candidate", {}).get("native_engine_sha256")
            != identity["engine_sha256"]
            or package_input.get("components", {})
            .get("native_engine", {})
            .get("sha256")
            != identity["engine_sha256"]
            or manifest_files.get("libexec/aima-engine.real", {}).get("sha256")
            != identity["engine_sha256"]
        ):
            errors.append("patch release engine identity differs")
        if (
            public_result.get("decision", {}).get("patch_release_promoted")
            is not True
            or manifest.get("complete") is not True
            or manifest.get("release") != release
            or manifest.get("source", {}).get("commit")
            != identity["release_commit"]
            or manifest.get("source", {}).get("native_commit")
            != identity["native_source_commit"]
            or manifest.get("source", {}).get("dirty") is not False
        ):
            errors.append("patch release promotion or manifest differs")
        checksum = root / release_record["archive_checksum"]
        expected_checksum = (
            f"{identity['archive_sha256']}  {identity['archive_name']}\n"
        )
        if (
            not checksum.is_file()
            or checksum.read_text(encoding="utf-8") != expected_checksum
        ):
            errors.append("patch release archive checksum differs")
        if release in ("1.5.1-native-vl.6", "1.5.1-native-vl.7"):
            baseline_errors = verify_release_evidence(
                root,
                PATCH_VL_RELEASE,
                require_archived_components=require_archived_components,
            )
            errors.extend(
                f"inherited patch baseline: {error}" for error in baseline_errors
            )
            baseline_manifest = _load(
                root / release_record["baseline_runtime_manifest"]
            )

            def runtime_files(value: dict[str, Any]) -> dict[str, Any]:
                return {
                    item["path"]: item
                    for item in value.get("files", [])
                    if item["path"].startswith(
                        ("lib/", "amdgcn/", "share/hip/", "share/certs/")
                    )
                    or item["path"] == "bin/aima-engine"
                }

            if runtime_files(manifest) != runtime_files(baseline_manifest):
                errors.append(
                    "patch portable userspace differs from its frozen baseline"
                )
            chat = _load(root / release_record["chat_protocol"])
            if (
                chat.get("checks", {}).get(
                    "vl_default_thinking_stream_nonstream_parity"
                )
                is not True
                or public_result.get("bundle_checks", {}).get(
                    "pinned_userspace_and_launcher_unchanged"
                )
                is not True
            ):
                errors.append(
                    "patch default VL reasoning or pinned userspace gate is missing"
                )

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
    if release in PATCH_VL_IDENTITIES:
        for key in (
            "chat_protocol",
            "http_control_plane",
            "archive_manifest",
            "archive_checksum",
            "baseline_g5",
            "baseline_package_input",
        ):
            expected_immutable[key] = release_record[key]
        if "baseline_runtime_manifest" in release_record:
            expected_immutable["baseline_runtime_manifest"] = release_record[
                "baseline_runtime_manifest"
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
                    _verify_recorded_artifacts(
                        root,
                        path,
                        _load(path),
                        archive_only_components=archive_only_components,
                    )
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

    if release == "1.5.1-native-vl.7":
        errors.extend(_verify_checkpoint_validation(
            root, package_input, public_evidence, PATCH_VL_IDENTITIES[release]
        ))

    for key in product_evidence_keys:
        provenance_record = public_evidence.get(key)
        result_record = public_result.get("evidence", {}).get(key)
        if not isinstance(provenance_record, dict) or not isinstance(result_record, dict):
            errors.append(f"public evidence record is missing: {key}")
            continue
        recorded_path = result_record.get("path")
        if release in PATCH_VL_IDENTITIES and isinstance(recorded_path, str):
            provenance_path = provenance_record.get("path")
            path_matches = isinstance(provenance_path, str) and Path(
                recorded_path
            ).name == Path(provenance_path).name
        else:
            mirrored = (
                _public_mirror_path(recorded_path)
                if isinstance(recorded_path, str)
                else None
            )
            path_matches = (
                mirrored is not None
                and mirrored.as_posix() == provenance_record.get("path")
            )
        if not path_matches:
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
            errors.extend(
                _verify_recorded_artifacts(
                    root,
                    path,
                    _load(path),
                    archive_only_components=archive_only_components,
                )
            )
        tree_path = record.get("tree_path")
        directory = (
            root / tree_path
            if isinstance(tree_path, str)
            else path.parent
        )
        if directory.is_dir():
            for raw_json in sorted(directory.rglob("*.json")):
                errors.extend(
                    _verify_recorded_artifacts(
                        root,
                        raw_json,
                        json.loads(raw_json.read_text(encoding="utf-8")),
                        archive_only_components=archive_only_components,
                    )
                )

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
            actual = evidence_tree(
                directory,
                virtual_components=_virtual_components_for_tree(
                    root,
                    directory,
                    archive_only_components,
                ),
            )
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
    if release in SEALED_RELEASES:
        provenance_sidecar = provenance_path.with_name(
            provenance_path.name + ".sha256"
        )
        if provenance_sidecar.is_file():
            paths.append(provenance_sidecar)
    for key, record in provenance["immutable_records"].items():
        path = root / record["path"]
        paths.append(path)
        if release in SEALED_RELEASES:
            sidecar = path.with_name(path.name + ".sha256")
            if sidecar.is_file():
                paths.append(sidecar)
            if (
                release == NATIVE_VL_RELEASE
                and key in NATIVE_VL_RAW_IMMUTABLE_KEYS
                and path.is_file()
            ):
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
    if release in ("1.5.1-native-vl.6", "1.5.1-native-vl.7"):
        paths.extend(evidence_paths(root, PATCH_VL_RELEASE))
    return list(dict.fromkeys(paths))


def _verify_checkpoint_validation(
    root: Path,
    package_input: dict[str, Any],
    public_evidence: dict[str, Any],
    identity: dict[str, str],
) -> list[str]:
    """Bind new checkpoint measurements separately from inherited kernel gates."""
    errors = []
    try:
        runtime = expected_binding(identity["engine_sha256"], root / (
            "benchmarks/results/native-portable-manifest-v1.5.1-native-vl.5.json"))
    except (OSError, ValueError, KeyError, TypeError):
        return ["safe-prefix pinned runtime baseline is missing or invalid"]
    # Historical verification uses the producer sealed into the package input,
    # not whichever verifier implementation happens to be installed today.
    runtime["verifier_sha256"] = package_input.get("inputs", {}).get(
        "qualification_runtime_verifier", {}).get("sha256")
    if not isinstance(runtime["verifier_sha256"], str) or len(runtime["verifier_sha256"]) != 64:
        errors.append("safe-prefix runtime verifier input is missing")
    for gate in ("exact_safe_prefix_generation_logits", "exact_safe_prefix_one_owner",
                 "exact_text_19_cell_matrix"):
        if package_input.get("gates", {}).get(gate) is not True:
            errors.append(f"safe-prefix release gate is missing: {gate}")
    for name, capacity in (("prefix", 32768), ("one_owner", 262144), ("text_matrix", None)):
        record = public_evidence.get(name, {})
        summary = (root / str(record.get("path", ""))).resolve()
        if not summary.is_relative_to(root.resolve()) or not summary.is_file():
            errors.append(f"safe-prefix measurement is missing: {name}")
            continue
        expected = {
            "path": f"candidate-validation/{name}/{summary.name}",
            "sha256": sha256(summary), "bytes": summary.stat().st_size,
        }
        if package_input.get("candidate_validation", {}).get(name) != expected:
            errors.append(f"safe-prefix package input binding differs: {name}")
        result = _load(summary)
        if result.get("runtime_binding") != runtime:
            errors.append(f"safe-prefix pinned runtime binding differs: {name}")
        engine = (result.get("engine", {}).get("sha256") if capacity is None
                  else result.get("engine_sha256"))
        if (result.get("complete") is not True or result.get("qualified") is not True
            or engine != identity["engine_sha256"]):
            errors.append(f"safe-prefix candidate measurement differs: {name}")
        if capacity is not None:
            if (result.get("release_eligible") is not True
                or result.get("build_info", {}).get("source_commit") != identity["native_source_commit"]
                or result.get("configuration", {}).get("cache_capacity") != capacity
                or verify_manifest_integrity(result)):
                errors.append(f"safe-prefix source/capacity/integrity differs: {name}")
            sidecar = summary.with_name(summary.name + ".sha256")
            if (not sidecar.is_file()
                or sidecar.read_text() != f"{sha256(summary)}  {summary.name}\n"):
                errors.append(f"safe-prefix checksum sidecar differs: {name}")
            artifacts = result.get("artifacts", [])
            if not artifacts:
                errors.append(f"safe-prefix raw artifacts are missing: {name}")
            for artifact in artifacts:
                path = (summary.parent / artifact["path"]).resolve()
                if (not path.is_relative_to(summary.parent) or not path.is_file()
                    or path.stat().st_size != artifact["bytes"]
                    or sha256(path) != artifact["sha256"]):
                    errors.append(f"safe-prefix raw artifact differs: {name}/{artifact['path']}")
        else:
            cells = result.get("cells", [])
            if len(cells) != 19 or not all(cell.get("pass") is True for cell in cells):
                errors.append("safe-prefix 19-cell matrix is incomplete")
            for cell in cells:
                reports, digests = cell.get("reports", []), cell.get("report_sha256", [])
                if len(reports) != len(digests) or len(reports) < 2:
                    errors.append("safe-prefix matrix raw bindings are incomplete")
                for report, digest in zip(reports, digests):
                    path = (summary.parent / report).resolve()
                    if (not path.is_relative_to(summary.parent) or not path.is_file()
                        or sha256(path) != digest):
                        errors.append(f"safe-prefix matrix raw report differs: {report}")
                    elif _load(path).get("qualification", {}).get("runtime_binding") != runtime:
                        errors.append(f"safe-prefix matrix raw runtime differs: {report}")
    return errors
