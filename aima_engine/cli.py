"""Operator CLI for the released AMD395 Qwen3.6 engine.

The control plane intentionally uses only the Python standard library. The
model runtime lives in a separately prepared ROCm/vLLM environment and is
selected explicitly with ``--runtime-python`` or ``AIMA_RUNTIME_PYTHON``.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any, Iterable
from urllib import error as urlerror
from urllib import request as urlrequest

from . import __version__


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "engine" / "production-runtime-config.json"
DEFAULT_ENDPOINT = "http://127.0.0.1:8000"
HASH_CHUNK_BYTES = 8 * 1024 * 1024


class UserError(RuntimeError):
    """An actionable configuration or environment error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(HASH_CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise UserError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise UserError(f"JSON root must be an object: {path}")
    return value


def load_json_value(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise UserError(f"cannot read JSON {path}: {exc}") from exc


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_json(path)
    if config.get("schema") != "aima-amd395-qwen36/production-runtime-config/v1":
        raise UserError(f"unsupported runtime config schema: {path}")
    if config.get("release_version") != __version__:
        raise UserError(
            f"runtime config version {config.get('release_version')!r} does not match CLI {__version__}"
        )
    return config


def component_records(config: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any]]]:
    yield "server", config["server"]
    yield "engine", config["engine"]
    yield "shape_manifest", config["shape_manifest"]
    for name, record in config["runtime_dependencies"].items():
        yield name, record
    for name, record in config["native"].items():
        yield name, record
    direct_plan = config["direct_checkpoint"]["plan"]
    yield "checkpoint_load_plan", direct_plan
    striped_template = config["striped_image"]["template"]
    if striped_template["path"] != direct_plan["path"]:
        yield "striped_image_template", striped_template


def verify_components(config: dict[str, Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for name, record in component_records(config):
        path = repo_path(str(record["path"]))
        actual = sha256_file(path) if path.is_file() else None
        expected = str(record["sha256"])
        results.append(
            {
                "name": name,
                "path": str(path),
                "expected_sha256": expected,
                "actual_sha256": actual,
                "passed": actual == expected,
            }
        )
    return results


def configured_path(
    explicit: str | None,
    environment_name: str,
    config_value: str | None,
    label: str,
    *,
    resolve_symlinks: bool = True,
) -> Path:
    value = explicit or os.environ.get(environment_name) or config_value
    if not value:
        raise UserError(f"{label} is required; pass it explicitly or set {environment_name}")
    path = Path(value).expanduser()
    return path.resolve() if resolve_symlinks else path.absolute()


def runtime_python(args: argparse.Namespace, config: dict[str, Any]) -> Path:
    path = configured_path(
        getattr(args, "runtime_python", None),
        "AIMA_RUNTIME_PYTHON",
        config.get("runtime_python"),
        "runtime Python",
        resolve_symlinks=False,
    )
    if not path.is_file():
        raise UserError(f"runtime Python does not exist: {path}")
    return path


def model_dir(args: argparse.Namespace, config: dict[str, Any]) -> Path:
    path = configured_path(
        getattr(args, "model_dir", None),
        "AIMA_MODEL_DIR",
        config.get("default_model_dir"),
        "model directory",
    )
    if not path.is_dir():
        raise UserError(f"model directory does not exist: {path}")
    return path


def image_manifest(args: argparse.Namespace) -> Path:
    return configured_path(
        getattr(args, "image_manifest", None),
        "AIMA_IMAGE_MANIFEST",
        None,
        "striped-image manifest",
    )


def direct_worker_count(args: argparse.Namespace, config: dict[str, Any]) -> int:
    explicit = getattr(args, "load_workers", None)
    count = int(config["direct_checkpoint"]["workers"] if explicit is None else explicit)
    if count <= 0 or count > 16:
        raise UserError("--load-workers must be between 1 and 16")
    return count


def validate_model(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    required = [path / "config.json", path / "model.safetensors.index.json"]
    missing = [str(item) for item in required if not item.is_file()]
    if missing:
        raise UserError(f"model directory is incomplete; missing: {', '.join(missing)}")
    index_path = required[1]
    actual = sha256_file(index_path)
    expected = str(config["direct_checkpoint"]["checkpoint_index_sha256"])
    if actual != expected:
        raise UserError(
            "checkpoint index does not match the qualified Qwen3.6-35B-A3B BF16 layout: "
            f"expected {expected}, got {actual}"
        )
    return {"model_dir": str(path), "checkpoint_index_sha256": actual, "passed": True}


def validate_direct_checkpoint(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    model = validate_model(path, config)
    direct = config["direct_checkpoint"]
    plan_path = repo_path(str(direct["plan"]["path"]))
    plan = load_json(plan_path)
    index = load_json(path / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    entries = plan.get("entries")
    if not isinstance(weight_map, dict):
        raise UserError("checkpoint index does not contain a weight_map object")
    if not isinstance(entries, list) or len(entries) != int(direct["tensor_count"]):
        raise UserError("direct checkpoint plan does not contain the qualified tensor set")
    if plan.get("layout", {}).get("payload_bytes") != int(direct["payload_bytes"]):
        raise UserError("direct checkpoint plan payload total does not match the release contract")

    names: set[str] = set()
    shard_names: set[str] = set()
    payload_total = 0
    normalized: list[tuple[str, str, int, int]] = []
    for position, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise UserError(f"direct checkpoint entry {position} is not an object")
        name = entry.get("name")
        shard = entry.get("source_shard")
        if not isinstance(name, str) or not name or name in names:
            raise UserError(f"invalid or duplicate tensor name at direct checkpoint entry {position}")
        if (
            not isinstance(shard, str)
            or not shard
            or Path(shard).name != shard
            or weight_map.get(name) != shard
        ):
            raise UserError(f"checkpoint shard mapping mismatch for tensor {name}")
        if entry.get("dtype") != "BF16":
            raise UserError(f"unsupported direct checkpoint dtype for tensor {name}")
        offset = int(entry.get("source_offset_bytes", -1))
        payload = int(entry.get("payload_bytes", 0))
        if offset < 0 or payload <= 0:
            raise UserError(f"invalid direct checkpoint byte geometry for tensor {name}")
        names.add(name)
        shard_names.add(shard)
        payload_total += payload
        normalized.append((name, shard, offset, payload))
    if not names.issubset(weight_map):
        raise UserError("direct checkpoint plan contains tensors absent from checkpoint weight_map")
    if payload_total != int(direct["payload_bytes"]):
        raise UserError("direct checkpoint entry payload sum mismatch")

    shard_results: list[dict[str, Any]] = []
    shard_sizes: dict[str, int] = {}
    for shard in sorted(shard_names):
        shard_path = path / shard
        if not shard_path.is_file():
            raise UserError(f"checkpoint shard is missing: {shard_path}")
        shard_sizes[shard] = shard_path.stat().st_size
        shard_results.append(
            {"name": shard, "path": str(shard_path), "bytes": shard_sizes[shard]}
        )
    for name, shard, offset, payload in normalized:
        if offset + payload > shard_sizes[shard]:
            raise UserError(f"tensor payload exceeds checkpoint shard: {name}")
    return {
        **model,
        "plan": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
        "tensor_count": len(entries),
        "payload_bytes": payload_total,
        "shard_count": len(shard_results),
        "shard_bytes": sum(item["bytes"] for item in shard_results),
        "shards": shard_results,
        "passed": True,
    }


def validate_image_manifest(
    path: Path,
    config: dict[str, Any],
    *,
    deep: bool,
) -> dict[str, Any]:
    manifest = load_json(path)
    image = config["striped_image"]
    template = load_json(repo_path(image["template"]["path"]))
    entries = manifest.get("entries")
    lanes = manifest.get("lanes")
    checks: dict[str, bool] = {
        "schema": manifest.get("schema") == image["manifest_schema"],
        "complete": manifest.get("complete") is True,
        "entries_match_release": isinstance(entries, list) and entries == template.get("entries"),
        "lane_count": isinstance(lanes, list) and len(lanes) == 2,
        "checkpoint_index": manifest.get("inputs", {})
        .get("checkpoint_index", {})
        .get("sha256")
        == image["checkpoint_index_sha256"],
        "layout_fingerprint": manifest.get("layout", {}).get("layout_fingerprint_sha256")
        == image["layout_fingerprint_sha256"],
        "aligned_bytes": manifest.get("layout", {}).get("aligned_bytes")
        == image["aligned_bytes"],
        "payload_bytes": manifest.get("layout", {}).get("payload_bytes")
        == image["payload_bytes"],
    }
    lane_results: list[dict[str, Any]] = []
    if isinstance(lanes, list) and len(lanes) == 2:
        for index, lane in enumerate(lanes):
            expected = image[f"lane{index}"]
            template_lane = template["lanes"][index]
            lane_path = Path(str(lane.get("image_path", ""))).expanduser().resolve()
            exists = lane_path.is_file()
            size = lane_path.stat().st_size if exists else None
            actual_sha = sha256_file(lane_path) if exists and deep else None
            lane_result = {
                "lane": index,
                "path": str(lane_path),
                "exists": exists,
                "expected_bytes": expected["bytes"],
                "actual_bytes": size,
                "size_passed": size == expected["bytes"],
                "expected_sha256": expected["sha256"],
                "actual_sha256": actual_sha,
                "sha256_passed": actual_sha == expected["sha256"] if deep else None,
            }
            lane_results.append(lane_result)
            checks[f"lane{index}_size"] = lane_result["size_passed"]
            checks[f"lane{index}_geometry"] = all(
                lane.get(key) == template_lane.get(key)
                for key in (
                    "aggregate_base_offset_bytes",
                    "image_bytes",
                    "image_gib",
                    "lane",
                    "tensor_count",
                )
            )
            if deep:
                checks[f"lane{index}_sha256"] = bool(lane_result["sha256_passed"])
    passed = all(checks.values())
    return {
        "path": str(path),
        "deep": deep,
        "checks": checks,
        "lanes": lane_results,
        "passed": passed,
    }


def materialize_manifest(
    *,
    output_path: Path,
    model_path: Path,
    lane_paths: list[Path],
    config: dict[str, Any],
) -> dict[str, Any]:
    template_path = repo_path(config["striped_image"]["template"]["path"])
    manifest = copy.deepcopy(load_json(template_path))
    manifest["inputs"]["checkpoint_index"]["path"] = str(
        (model_path / "model.safetensors.index.json").resolve()
    )
    for index, lane_path in enumerate(lane_paths):
        manifest["lanes"][index]["image_path"] = str(lane_path.resolve())
        manifest["lanes"][index]["filesystem_role"] = f"configured-lane{index}"
        manifest["lanes"][index]["physical_device"] = f"dev:{lane_path.stat().st_dev}"
    manifest["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["build_status"] = {
        "engine_source_mutated": False,
        "images_written": False,
        "model_loaded": False,
        "portable_manifest_materialized": True,
    }
    write_json(output_path, manifest)
    return manifest


def register_images(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config).resolve())
    model_path = model_dir(args, config)
    validate_model(model_path, config)
    lane_paths = [Path(args.lane0).expanduser().resolve(), Path(args.lane1).expanduser().resolve()]
    image = config["striped_image"]
    for index, lane_path in enumerate(lane_paths):
        if not lane_path.is_file():
            raise UserError(f"lane{index} image does not exist: {lane_path}")
        expected = image[f"lane{index}"]
        if lane_path.stat().st_size != expected["bytes"]:
            raise UserError(f"lane{index} size mismatch: {lane_path}")
        actual = sha256_file(lane_path)
        if actual != expected["sha256"]:
            raise UserError(
                f"lane{index} SHA-256 mismatch: expected {expected['sha256']}, got {actual}"
            )
    output_path = Path(args.output_manifest).expanduser().resolve()
    materialize_manifest(
        output_path=output_path,
        model_path=model_path,
        lane_paths=lane_paths,
        config=config,
    )
    result = validate_image_manifest(output_path, config, deep=False)
    if not result["passed"]:
        raise UserError(f"materialized image manifest failed validation: {result['checks']}")
    print(json.dumps({"registered": True, "image_manifest": str(output_path)}, indent=2))
    return 0


def write_lane_plan(
    path: Path,
    entries: list[dict[str, Any]],
    lane: int,
    total_bytes: int,
    model_path: Path,
) -> None:
    selected = sorted(
        (item for item in entries if int(item["lane"]) == lane),
        key=lambda item: int(item["lane_offset_bytes"]),
    )
    lines = [f"v1\t{total_bytes}\t{len(selected)}"]
    expected_offset = 0
    for item in selected:
        offset = int(item["lane_offset_bytes"])
        if offset != expected_offset:
            raise UserError(f"lane{lane} manifest offsets are not contiguous")
        source = model_path / str(item["source_shard"])
        if not source.is_file():
            raise UserError(f"checkpoint shard is missing: {source}")
        aligned = int(item["aligned_bytes"])
        lines.append(
            "\t".join(
                [
                    str(offset),
                    str(aligned),
                    str(int(item["payload_bytes"])),
                    str(int(item["source_offset_bytes"])),
                    str(source.resolve()),
                ]
            )
        )
        expected_offset += aligned
    if expected_offset != total_bytes:
        raise UserError(f"lane{lane} plan total mismatch")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def require_disk_space(paths: list[Path], byte_requirements: list[int]) -> None:
    by_device: dict[int, dict[str, Any]] = {}
    for path, required in zip(paths, byte_requirements):
        path.mkdir(parents=True, exist_ok=True)
        device = path.stat().st_dev
        record = by_device.setdefault(device, {"path": path, "required": 0})
        record["required"] += required
    reserve = 1024**3
    for record in by_device.values():
        free = shutil.disk_usage(record["path"]).free
        required = int(record["required"]) + reserve
        if free < required:
            raise UserError(
                f"not enough free space on {record['path']}: need {required} bytes, have {free}"
            )


def build_striped_images(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config).resolve())
    model_path = model_dir(args, config)
    validate_model(model_path, config)
    lane0_dir = Path(args.lane0_dir).expanduser().resolve()
    lane1_dir = Path(args.lane1_dir).expanduser().resolve()
    lane_paths = [lane0_dir / "lane0.bin", lane1_dir / "lane1.bin"]
    image = config["striped_image"]
    require_disk_space(
        [lane0_dir, lane1_dir],
        [int(image["lane0"]["bytes"]), int(image["lane1"]["bytes"])],
    )
    for lane_path in lane_paths:
        if lane_path.exists():
            if not args.force:
                raise UserError(f"refusing to overwrite existing image: {lane_path}")
            lane_path.unlink()

    state_dir = Path(args.state_dir).expanduser().resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    builder = ROOT / "build" / "striped_image_builder"
    source = ROOT / "benchmarks" / "shape-lab" / "native" / "src" / "striped_image_builder.cpp"
    if not builder.is_file() or builder.stat().st_mtime_ns < source.stat().st_mtime_ns:
        builder.parent.mkdir(parents=True, exist_ok=True)
        command = [
            args.cxx,
            "-O3",
            "-std=c++17",
            "-pthread",
            str(source),
            "-o",
            str(builder),
        ]
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode != 0:
            raise UserError(f"striped-image builder compilation failed with {completed.returncode}")

    template = load_json(repo_path(image["template"]["path"]))
    entries = template.get("entries")
    if not isinstance(entries, list) or len(entries) != 693:
        raise UserError("striped-image template does not contain the qualified 693 tensors")
    plans = [state_dir / "lane0.plan.tsv", state_dir / "lane1.plan.tsv"]
    write_lane_plan(plans[0], entries, 0, int(image["lane0"]["bytes"]), model_path)
    write_lane_plan(plans[1], entries, 1, int(image["lane1"]["bytes"]), model_path)
    build_result = state_dir / "image-build-result.json"
    command = [
        str(builder),
        str(plans[0]),
        str(plans[1]),
        str(lane_paths[0]),
        str(lane_paths[1]),
        str(build_result),
        str(image["chunk_bytes"]),
    ]
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode != 0:
        raise UserError(
            f"striped-image build failed with {completed.returncode}; inspect {build_result}"
        )
    result = load_json(build_result)
    if result.get("complete") is not True:
        raise UserError(f"striped-image build did not complete: {build_result}")

    output_manifest = Path(args.output_manifest).expanduser().resolve()
    for index, lane_path in enumerate(lane_paths):
        actual = sha256_file(lane_path)
        expected = image[f"lane{index}"]["sha256"]
        if actual != expected:
            raise UserError(f"lane{index} SHA-256 mismatch after build: expected {expected}, got {actual}")
    materialize_manifest(
        output_path=output_manifest,
        model_path=model_path,
        lane_paths=lane_paths,
        config=config,
    )
    print(
        json.dumps(
            {
                "built": True,
                "lane0": str(lane_paths[0]),
                "lane1": str(lane_paths[1]),
                "image_manifest": str(output_manifest),
                "build_result": str(build_result),
            },
            indent=2,
        )
    )
    return 0


def runtime_probe(python: Path, config: dict[str, Any]) -> dict[str, Any]:
    qualified = config["qualified_runtime"]
    code = """
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
import sys

names = ["torch", "vllm", "triton", "transformers", "safetensors"]
spec = importlib.util.find_spec("torch")
torch_root = Path(spec.origin).resolve().parent if spec and spec.origin else None
library = Path(os.environ["AIMA_AOTRITON_LIBRARY"]).resolve() if os.environ.get("AIMA_AOTRITON_LIBRARY") else (torch_root / "lib" / os.environ["AIMA_AOTRITON_NAME"] if torch_root else None)
digest = None
if library and library.is_file():
    value = hashlib.sha256()
    with library.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    digest = value.hexdigest()
print(json.dumps({
    "python_major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
    "versions": {name: metadata.version(name) for name in names},
    "aotriton_library": str(library) if library else None,
    "aotriton_sha256": digest,
}))
"""
    env = os.environ.copy()
    env["AIMA_AOTRITON_NAME"] = str(qualified["aotriton_library"])
    completed = subprocess.run(
        [str(python), "-c", code],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
        env=env,
    )
    if completed.returncode != 0:
        return {
            "passed": False,
            "returncode": completed.returncode,
            "stderr": completed.stderr.strip(),
        }
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {"passed": False, "stdout": completed.stdout.strip()}
    value["checks"] = {
        "python": value.get("python_major_minor") == qualified["python_major_minor"],
        "packages": value.get("versions") == qualified["packages"],
        "aotriton": value.get("aotriton_sha256") == qualified["aotriton_sha256"],
    }
    value["passed"] = all(value["checks"].values())
    return value


def doctor(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config).resolve())
    checks: list[dict[str, Any]] = []
    checks.append(
        {
            "name": "linux_x86_64",
            "passed": platform.system() == "Linux" and platform.machine() == "x86_64",
            "value": platform.platform(),
        }
    )
    rocminfo = shutil.which("rocminfo") or "/opt/rocm/bin/rocminfo"
    if Path(rocminfo).is_file():
        completed = subprocess.run([rocminfo], capture_output=True, text=True, check=False, timeout=30)
        gpu_passed = completed.returncode == 0 and "gfx1151" in completed.stdout
    else:
        gpu_passed = False
    checks.append({"name": "gfx1151", "passed": gpu_passed, "value": rocminfo})
    components = verify_components(config)
    checks.append(
        {
            "name": "release_components",
            "passed": all(item["passed"] for item in components),
            "details": components,
        }
    )
    try:
        python = runtime_python(args, config)
        probe = runtime_probe(python, config)
        checks.append({"name": "runtime", "passed": probe.get("passed") is True, "details": probe})
    except UserError as exc:
        checks.append({"name": "runtime", "passed": False, "error": str(exc)})
    model_path: Path | None = None
    try:
        model_path = model_dir(args, config)
        checks.append({"name": "model", "passed": True, "details": validate_model(model_path, config)})
    except UserError as exc:
        checks.append({"name": "model", "passed": False, "error": str(exc)})
    if args.load_mode == "direct":
        try:
            if model_path is None:
                raise UserError("model validation must pass before direct loading can be checked")
            direct_result = validate_direct_checkpoint(model_path, config)
            direct_result["reader_workers"] = direct_worker_count(args, config)
            checks.append(
                {
                    "name": "direct_safetensors",
                    "passed": direct_result["passed"],
                    "details": direct_result,
                }
            )
        except UserError as exc:
            checks.append({"name": "direct_safetensors", "passed": False, "error": str(exc)})
    else:
        try:
            manifest_path = image_manifest(args)
            image_result = validate_image_manifest(manifest_path, config, deep=args.deep)
            checks.append(
                {"name": "striped_images", "passed": image_result["passed"], "details": image_result}
            )
        except UserError as exc:
            checks.append({"name": "striped_images", "passed": False, "error": str(exc)})
    result = {"ready": all(item["passed"] for item in checks), "checks": checks}
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        for item in checks:
            print(f"{'PASS' if item['passed'] else 'FAIL'}  {item['name']}")
            if not item["passed"] and item.get("error"):
                print(f"      {item['error']}")
        print("READY" if result["ready"] else "NOT READY")
    return 0 if result["ready"] else 1


def serve(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config).resolve())
    component_checks = verify_components(config)
    failed = [item["name"] for item in component_checks if not item["passed"]]
    if failed:
        raise UserError(f"release component integrity failed: {', '.join(failed)}")
    python = runtime_python(args, config)
    probe = runtime_probe(python, config)
    if probe.get("passed") is not True:
        raise UserError(f"runtime contract mismatch: {probe}")
    model_path = model_dir(args, config)
    validate_model(model_path, config)
    direct_result: dict[str, Any] | None = None
    image_manifest_path: Path | None = None
    image_result: dict[str, Any] | None = None
    if args.load_mode == "direct":
        direct_result = validate_direct_checkpoint(model_path, config)
    else:
        image_manifest_path = image_manifest(args)
        image_result = validate_image_manifest(image_manifest_path, config, deep=False)
        if not image_result["passed"]:
            raise UserError(f"striped-image validation failed: {image_result['checks']}")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (
            Path(args.output_root).expanduser().resolve()
            if args.output_root
            else ROOT / "output"
        )
        / f"service-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    if output_dir.exists():
        raise UserError(f"output directory already exists: {output_dir}")
    ready_file = (
        Path(args.ready_file).expanduser().resolve()
        if args.ready_file
        else output_dir / "ready.json"
    )
    server_path = repo_path(config["server"]["path"])
    shape_manifest = repo_path(config["shape_manifest"]["path"])
    env = os.environ.copy()
    internal_load_environment = (
        "AIMA_DIRECT_CHECKPOINT_HELPER",
        "AIMA_DIRECT_CHECKPOINT_PLAN",
        "AIMA_DIRECT_CHECKPOINT_LOADER",
        "AIMA_DIRECT_CHECKPOINT_NATIVE_REPORT",
        "AIMA_DIRECT_CHECKPOINT_REPORT",
        "AIMA_DIRECT_CHECKPOINT_INDEX_SHA256",
        "AIMA_DIRECT_CHECKPOINT_EXPECTED_XOR",
        "AIMA_DIRECT_CHECKPOINT_EXPECTED_SUM",
        "AIMA_DIRECT_CHECKPOINT_PAYLOAD_BYTES",
        "AIMA_DIRECT_CHECKPOINT_TENSOR_COUNT",
        "AIMA_DIRECT_CHECKPOINT_CHUNK_BYTES",
        "AIMA_DIRECT_CHECKPOINT_WORKERS",
        "AMD395_STRIPED_IMAGE_CHUNK_BYTES",
        "AMD395_STRIPED_IMAGE_EXPECTED_SUM",
        "AMD395_STRIPED_IMAGE_EXPECTED_XOR",
        "AMD395_STRIPED_IMAGE_LANE0_SHA256",
        "AMD395_STRIPED_IMAGE_LANE1_SHA256",
        "AMD395_STRIPED_IMAGE_LOADER",
        "AMD395_STRIPED_IMAGE_MANIFEST",
        "AMD395_STRIPED_IMAGE_NATIVE_REPORT",
    )
    for name in internal_load_environment:
        env.pop(name, None)
    env.update(
        {
            "AIMA_LOAD_MODE": args.load_mode,
            "HIP_VISIBLE_DEVICES": str(args.device),
            "PYTHONNOUSERSITE": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTORCH_TUNABLEOP_ENABLED": "0",
            "PYTORCH_TUNABLEOP_TUNING": "0",
            "TOKENIZERS_PARALLELISM": "false",
            "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "1",
        }
    )
    load_event: dict[str, Any]
    if args.load_mode == "direct":
        direct = config["direct_checkpoint"]
        load_workers = direct_worker_count(args, config)
        direct_plan = repo_path(str(direct["plan"]["path"]))
        direct_loader = repo_path(config["native"]["direct_checkpoint_loader"]["path"])
        direct_helper = repo_path(
            config["runtime_dependencies"]["direct_checkpoint_helper"]["path"]
        )
        env.update(
            {
                "AIMA_DIRECT_CHECKPOINT_HELPER": str(direct_helper),
                "AIMA_DIRECT_CHECKPOINT_PLAN": str(direct_plan),
                "AIMA_DIRECT_CHECKPOINT_LOADER": str(direct_loader),
                "AIMA_DIRECT_CHECKPOINT_NATIVE_REPORT": str(
                    output_dir / "direct-checkpoint-native.json"
                ),
                "AIMA_DIRECT_CHECKPOINT_REPORT": str(
                    output_dir / "direct-checkpoint-loader.json"
                ),
                "AIMA_DIRECT_CHECKPOINT_INDEX_SHA256": str(
                    direct["checkpoint_index_sha256"]
                ),
                "AIMA_DIRECT_CHECKPOINT_EXPECTED_XOR": str(direct["expected_xor"]),
                "AIMA_DIRECT_CHECKPOINT_EXPECTED_SUM": str(direct["expected_sum"]),
                "AIMA_DIRECT_CHECKPOINT_PAYLOAD_BYTES": str(direct["payload_bytes"]),
                "AIMA_DIRECT_CHECKPOINT_TENSOR_COUNT": str(direct["tensor_count"]),
                "AIMA_DIRECT_CHECKPOINT_CHUNK_BYTES": str(direct["chunk_bytes"]),
                "AIMA_DIRECT_CHECKPOINT_WORKERS": str(load_workers),
            }
        )
        load_event = {
            "load_mode": "direct",
            "checkpoint_plan": str(direct_plan),
            "checkpoint_shards": direct_result["shard_count"] if direct_result else None,
            "load_workers": load_workers,
            "extra_weight_copy_bytes": 0,
        }
    else:
        if image_result is None or image_manifest_path is None:
            raise UserError("striped-image validation did not produce a load contract")
        image = config["striped_image"]
        native_loader = repo_path(config["native"]["striped_image_loader"]["path"])
        lanes = image_result["lanes"]
        env.update(
            {
                "AMD395_STRIPED_IMAGE_CHUNK_BYTES": str(image["chunk_bytes"]),
                "AMD395_STRIPED_IMAGE_EXPECTED_SUM": str(image["expected_sum"]),
                "AMD395_STRIPED_IMAGE_EXPECTED_XOR": str(image["expected_xor"]),
                "AMD395_STRIPED_IMAGE_LANE0_SHA256": str(image["lane0"]["sha256"]),
                "AMD395_STRIPED_IMAGE_LANE1_SHA256": str(image["lane1"]["sha256"]),
                "AMD395_STRIPED_IMAGE_LOADER": str(native_loader),
                "AMD395_STRIPED_IMAGE_MANIFEST": str(image_manifest_path),
                "AMD395_STRIPED_IMAGE_NATIVE_REPORT": str(
                    output_dir / "striped-native-loader.json"
                ),
            }
        )
        load_event = {
            "load_mode": "striped",
            "image_manifest": str(image_manifest_path),
            "lane_paths": [item["path"] for item in lanes],
        }
    policy = config["fixed_policy"]
    argv = [
        str(python),
        str(server_path),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--output-dir",
        str(output_dir),
        "--model-dir",
        str(model_path),
        "--manifest",
        str(shape_manifest),
        "--device",
        str(policy["device"]),
        "--ready-file",
        str(ready_file),
        "--resident-native-decode-hotset-layers",
        str(policy["resident_native_decode_hotset_layers"]),
        "--exact-prefix-cache-max-entries",
        str(policy["exact_prefix_cache_max_entries"]),
        "--exact-prefix-cache-max-tokens",
        str(policy["exact_prefix_cache_max_tokens"]),
        "--max-requests",
        str(args.max_requests),
        "--admitted-context-policy",
        "--exact-prefix-cache",
        "--execute",
    ]
    event = {
        "event": "exec_aima_engine",
        "version": __version__,
        "host": args.host,
        "port": args.port,
        "model_dir": str(model_path),
        "output_dir": str(output_dir),
        "ready_file": str(ready_file),
        "engine_sha256": config["engine"]["sha256"],
        **load_event,
    }
    print(json.dumps(event, sort_keys=True), flush=True)
    os.chdir(ROOT)
    os.execve(str(python), argv, env)
    return 0


def http_json(
    method: str,
    endpoint: str,
    path: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if body is not None else {}
    request = urlrequest.Request(
        endpoint.rstrip("/") + path,
        data=body,
        headers=headers,
        method=method,
    )
    try:
        with urlrequest.urlopen(request, timeout=timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
    except (urlerror.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise UserError(f"request failed: {exc}") from exc
    if not isinstance(value, dict):
        raise UserError("server response is not a JSON object")
    return value


def http_sse(
    endpoint: str,
    path: str,
    payload: dict[str, Any],
    timeout: float,
) -> Iterable[dict[str, Any]]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urlrequest.Request(
        endpoint.rstrip("/") + path,
        data=body,
        headers={
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlrequest.urlopen(request, timeout=timeout) as response:
            content_type = response.headers.get_content_type()
            if content_type != "text/event-stream":
                raise UserError(
                    f"server returned {content_type}, expected text/event-stream"
                )
            for wire_line in response:
                line = wire_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                value = json.loads(data)
                if not isinstance(value, dict):
                    raise UserError("stream event is not a JSON object")
                if "error" in value:
                    message = value.get("error", {}).get("message", value)
                    raise UserError(f"stream failed: {message}")
                yield value
    except (urlerror.URLError, TimeoutError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise UserError(f"stream request failed: {exc}") from exc


def status(args: argparse.Namespace) -> int:
    print(json.dumps(http_json("GET", args.endpoint, "/health", timeout=args.timeout), indent=2))
    return 0


def models(args: argparse.Namespace) -> int:
    print(json.dumps(http_json("GET", args.endpoint, "/v1/models", timeout=args.timeout), indent=2))
    return 0


def chat(args: argparse.Namespace) -> int:
    if args.messages_json:
        messages = load_json_value(Path(args.messages_json).expanduser())
        if not isinstance(messages, list) or not messages:
            raise UserError("--messages-json must contain a non-empty array")
        if args.prompt is not None or args.system:
            raise UserError("--messages-json cannot be combined with prompt or --system")
    else:
        prompt = args.prompt
        if prompt is None:
            prompt = sys.stdin.read()
        if not prompt.strip():
            raise UserError("prompt is empty")
        messages = []
        if args.system:
            messages.append({"role": "system", "content": args.system})
        messages.append({"role": "user", "content": prompt})
    payload = {
        "model": args.model,
        "messages": messages,
        "temperature": 0,
        "top_p": 1,
        "n": 1,
        "stream": args.stream,
        "max_tokens": args.max_tokens,
    }
    if args.tools_json:
        tools = load_json_value(Path(args.tools_json).expanduser())
        if not isinstance(tools, list):
            raise UserError("--tools-json must contain an array")
        payload["tools"] = tools
        if args.tool_choice in {"auto", "none", "required"}:
            payload["tool_choice"] = args.tool_choice
        else:
            payload["tool_choice"] = {
                "type": "function",
                "function": {"name": args.tool_choice},
            }
        payload["parallel_tool_calls"] = args.parallel_tool_calls
    elif args.tool_choice != "auto":
        raise UserError("--tool-choice requires --tools-json")
    if args.stream:
        payload["stream_options"] = {"include_usage": True}
        tool_calls: dict[int, dict[str, Any]] = {}
        wrote_content = False
        for event in http_sse(
            args.endpoint,
            "/v1/chat/completions",
            payload,
            timeout=args.timeout,
        ):
            if args.json:
                print(json.dumps(event, ensure_ascii=False), flush=True)
                continue
            for choice in event.get("choices", []):
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if content:
                    print(content, end="", flush=True)
                    wrote_content = True
                for call in delta.get("tool_calls") or []:
                    index = int(call["index"])
                    target = tool_calls.setdefault(
                        index,
                        {
                            "id": call.get("id"),
                            "type": call.get("type", "function"),
                            "function": {"name": "", "arguments": ""},
                        },
                    )
                    function = call.get("function") or {}
                    target["function"]["name"] += function.get("name", "")
                    target["function"]["arguments"] += function.get("arguments", "")
        if not args.json:
            if wrote_content:
                print()
            if tool_calls:
                print(
                    json.dumps(
                        {"tool_calls": [tool_calls[index] for index in sorted(tool_calls)]},
                        ensure_ascii=False,
                    )
                )
        return 0
    response = http_json(
        "POST",
        args.endpoint,
        "/v1/chat/completions",
        payload,
        timeout=args.timeout,
    )
    if args.json:
        print(json.dumps(response, indent=2, ensure_ascii=False))
    else:
        try:
            message = response["choices"][0]["message"]
            if message.get("tool_calls"):
                print(
                    json.dumps(
                        {"tool_calls": message["tool_calls"]},
                        ensure_ascii=False,
                    )
                )
            else:
                print(message["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise UserError("server response is missing an assistant message") from exc
    return 0


def shutdown(args: argparse.Namespace) -> int:
    print(json.dumps(http_json("POST", args.endpoint, "/shutdown", timeout=args.timeout), indent=2))
    return 0


def verify_release(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config).resolve())
    components = verify_components(config)
    required_files = [
        ROOT / "LICENSE",
        ROOT / "NOTICE",
        ROOT / "THIRD_PARTY_NOTICES.md",
        ROOT / "README.md",
        ROOT / "SECURITY.md",
        ROOT / "CONTRIBUTING.md",
        ROOT / "CODE_OF_CONDUCT.md",
    ]
    required_checks = [
        {"path": str(path), "passed": path.is_file() and path.stat().st_size > 0}
        for path in required_files
    ]
    private_markers = [
        "/home/" + "quings",
        "/data/home/" + "quings",
        "aima-" + "hidden-20260622",
        "qujing" + "#$@21",
    ]
    hygiene_findings: list[dict[str, str]] = []
    excluded_parts = {".git", "__pycache__", "output", "build", "state"}
    text_suffixes = {
        "",
        ".c",
        ".cc",
        ".cpp",
        ".h",
        ".hip",
        ".json",
        ".md",
        ".py",
        ".sh",
        ".toml",
        ".txt",
        ".yml",
        ".yaml",
    }
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in excluded_parts for part in path.parts):
            continue
        if path.suffix.lower() not in text_suffixes and path.name not in {"Makefile", "NOTICE"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for marker in private_markers:
            if marker in text:
                hygiene_findings.append({"path": str(path.relative_to(ROOT)), "marker": marker})
    result = {
        "version": __version__,
        "components_passed": all(item["passed"] for item in components),
        "required_files_passed": all(item["passed"] for item in required_checks),
        "hygiene_passed": not hygiene_findings,
        "components": components,
        "required_files": required_checks,
        "hygiene_findings": hygiene_findings,
    }
    result["passed"] = (
        result["components_passed"]
        and result["required_files_passed"]
        and result["hygiene_passed"]
    )
    if args.json or not result["passed"]:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"release v{__version__}: PASS")
        print(f"engine sha256: {config['engine']['sha256']}")
    return 0 if result["passed"] else 1


def add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--runtime-python")
    parser.add_argument("--model-dir")
    parser.add_argument(
        "--load-mode",
        choices=("direct", "striped"),
        default=os.environ.get("AIMA_LOAD_MODE", "direct"),
        help="load standard Safetensors directly (default) or use optional striped images",
    )
    parser.add_argument(
        "--load-workers",
        type=int,
        help="direct Safetensors reader workers; defaults to the qualified release value",
    )
    parser.add_argument(
        "--image-manifest",
        help="required only with --load-mode striped",
    )


def add_http_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--timeout", type=float, default=30.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aima-engine",
        description="AIMA batch-1 Qwen3.6-35B-A3B BF16 engine for AMD395 Linux",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify_parser = subparsers.add_parser("verify", help="verify release files and hashes")
    verify_parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    verify_parser.add_argument("--json", action="store_true")
    verify_parser.set_defaults(handler=verify_release)

    doctor_parser = subparsers.add_parser("doctor", help="check host, runtime and model loading")
    add_runtime_arguments(doctor_parser)
    doctor_parser.add_argument(
        "--deep",
        action="store_true",
        help="with --load-mode striped, also SHA-256 both 32 GiB lanes",
    )
    doctor_parser.add_argument("--json", action="store_true")
    doctor_parser.set_defaults(handler=doctor)

    register_parser = subparsers.add_parser(
        "register-images", help="validate existing startup image lanes and write a portable manifest"
    )
    register_parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    register_parser.add_argument("--model-dir")
    register_parser.add_argument("--lane0", required=True)
    register_parser.add_argument("--lane1", required=True)
    register_parser.add_argument("--output-manifest", required=True)
    register_parser.set_defaults(handler=register_images)

    prepare_parser = subparsers.add_parser(
        "prepare-images", help="build two startup image lanes from licensed local model weights"
    )
    prepare_parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    prepare_parser.add_argument("--model-dir")
    prepare_parser.add_argument("--lane0-dir", required=True)
    prepare_parser.add_argument("--lane1-dir", required=True)
    prepare_parser.add_argument("--state-dir", required=True)
    prepare_parser.add_argument("--output-manifest", required=True)
    prepare_parser.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
    prepare_parser.add_argument("--force", action="store_true")
    prepare_parser.set_defaults(handler=build_striped_images)

    serve_parser = subparsers.add_parser("serve", help="start the resident OpenAI-compatible server")
    add_runtime_arguments(serve_parser)
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--device", type=int, default=0)
    serve_parser.add_argument("--output-dir")
    serve_parser.add_argument(
        "--output-root",
        help="parent directory for a new timestamped service output directory",
    )
    serve_parser.add_argument("--ready-file")
    serve_parser.add_argument("--max-requests", type=int, default=0)
    serve_parser.set_defaults(handler=serve)

    status_parser = subparsers.add_parser("status", help="read server health")
    add_http_arguments(status_parser)
    status_parser.set_defaults(handler=status)

    models_parser = subparsers.add_parser("models", help="list served models")
    add_http_arguments(models_parser)
    models_parser.set_defaults(handler=models)

    chat_parser = subparsers.add_parser("chat", help="send one deterministic chat completion")
    add_http_arguments(chat_parser)
    chat_parser.add_argument("prompt", nargs="?")
    chat_parser.add_argument("--system")
    chat_parser.add_argument(
        "--messages-json",
        help="JSON file containing a complete user/assistant/tool message history",
    )
    chat_parser.add_argument(
        "--tools-json",
        help="JSON file containing an array of OpenAI function tools",
    )
    chat_parser.add_argument(
        "--tool-choice",
        default="auto",
        help="auto, none, required, or a function name",
    )
    chat_parser.add_argument(
        "--parallel-tool-calls",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    chat_parser.add_argument("--stream", action="store_true")
    chat_parser.add_argument("--model", default="aima-amd395-qwen36-35b")
    chat_parser.add_argument("--max-tokens", type=int, default=128)
    chat_parser.add_argument("--json", action="store_true")
    chat_parser.set_defaults(handler=chat)

    shutdown_parser = subparsers.add_parser("shutdown", help="stop the resident server cleanly")
    add_http_arguments(shutdown_parser)
    shutdown_parser.set_defaults(handler=shutdown)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        result = args.handler(args)
    except UserError as exc:
        parser.exit(2, f"error: {exc}\n")
    raise SystemExit(int(result or 0))
