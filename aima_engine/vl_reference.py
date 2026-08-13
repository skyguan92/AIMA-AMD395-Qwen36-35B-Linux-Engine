"""Hash-bound reference identity helpers for the native VL qualification."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import shutil
import stat
import subprocess
import sys
from typing import Any


REFERENCE_SCHEMA = "aima-amd395-qwen36/vl-reference-manifest/v1"
LAUNCH_SCHEMA = "aima-amd395-qwen36/vl-reference-launch/v1"
CAPABILITY_SCHEMA = "aima-amd395-qwen36/vl-capability-manifest/v1"

BASELINE_RELEASE = "v1.5.1"
BASELINE_COMMIT = "6f3e669ac897eaabfeceb7f193a5e02708a4d95e"
BASELINE_NATIVE_COMMIT = "65c198415709dad6d046c247acab3dc9df2a95a0"
MODEL_REPOSITORY = "Qwen/Qwen3.6-35B-A3B"
MODEL_REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"
REFERENCE_MAX_BATCHED_TOKENS = 16_384
REFERENCE_MEDIA_LIMITS = {"image": 16, "video": 21}
REFERENCE_ATTENTION_BACKEND = "TRITON_ATTN"

PINNED_PACKAGES = {
    "vllm": "0.19.1rc1.dev300+g29e5d1020",
    "torch": "2.10.0+git8514f05",
    "transformers": "4.57.6",
}
MEDIA_PACKAGES = (
    "numpy",
    "pillow",
    "av",
    "opencv-python",
    "opencv-python-headless",
    "requests",
    "aiohttp",
)
REQUIRED_MODEL_FILES = (
    "config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
)
REQUIRED_TIMING_PHASES = (
    "media_fetch",
    "media_decode",
    "processor",
    "vision_encode",
    "llm_prefill",
    "decode",
    "ttft",
    "total_latency",
)
REQUIRED_SOURCE_MODULES = (
    "vllm.model_executor.models.qwen3_5",
    "vllm.model_executor.models.qwen3_vl",
    "vllm.multimodal.media.connector",
    "vllm.multimodal.media.video",
    "transformers.models.qwen3_vl.processing_qwen3_vl",
    "transformers.models.qwen3_vl.video_processing_qwen3_vl",
    "transformers.models.qwen2_vl.image_processing_qwen2_vl_fast",
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SENSITIVE_ENVIRONMENT_KEY = re.compile(
    r"(?:PASSWORD|PASSWD|SECRET|TOKEN|CREDENTIAL|API_KEY|ACCESS_KEY)", re.IGNORECASE
)


class ReferenceManifestError(ValueError):
    """Raised when a reference input cannot satisfy the frozen goal."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256_bytes(payload)


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReferenceManifestError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReferenceManifestError(f"JSON root must be an object: {path}")
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
    digest = sha256_bytes(payload)
    sidecar = path.with_name(path.name + ".sha256")
    sidecar_temporary = sidecar.with_name(sidecar.name + ".tmp")
    sidecar_temporary.write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    sidecar_temporary.replace(sidecar)
    return digest


def file_component(path: Path, logical_path: str) -> dict[str, Any]:
    if not path.is_file():
        raise ReferenceManifestError(f"required file is missing: {path}")
    return {
        "path": logical_path,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _option_value(argv: Sequence[str], option: str) -> str | None:
    values: list[str] = []
    for index, item in enumerate(argv):
        if item == option:
            if index + 1 >= len(argv):
                return None
            values.append(argv[index + 1])
        elif item.startswith(option + "="):
            values.append(item[len(option) + 1 :])
    if len(values) != 1:
        return None
    return values[0]


def _option_values(argv: Sequence[str], option: str) -> list[str] | None:
    try:
        start = argv.index(option) + 1
    except ValueError:
        return None
    values: list[str] = []
    for item in argv[start:]:
        if item.startswith("--"):
            break
        values.append(item)
    return values


def _json_option(argv: Sequence[str], option: str, errors: list[str]) -> Any:
    raw = _option_value(argv, option)
    if raw is None:
        errors.append(f"launch argv must contain exactly one {option} value")
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        errors.append(f"{option} must be canonical JSON")
        return None


def validate_launch_config(config: Mapping[str, Any]) -> list[str]:
    """Return all violations that would make a VL reference ambiguous or weak."""

    errors: list[str] = []
    if config.get("schema") != LAUNCH_SCHEMA:
        errors.append(f"launch schema must be {LAUNCH_SCHEMA}")

    argv_value = config.get("argv")
    if not isinstance(argv_value, list) or not argv_value or not all(
        isinstance(item, str) and item for item in argv_value
    ):
        return errors + ["launch argv must be a non-empty string array"]
    argv: list[str] = argv_value
    if argv[:3] != [
        "${AIMA_VLLM_PYTHON}",
        "-m",
        "vllm.entrypoints.openai.api_server",
    ]:
        errors.append("launch argv must use the frozen vLLM OpenAI entrypoint")

    forbidden = {
        "--language-model-only": "language-model-only is not a VL reference",
        "--skip-mm-profiling": "skip-mm-profiling is not a VL reference",
    }
    for flag, message in forbidden.items():
        if flag in argv or any(item.startswith(flag + "=") for item in argv):
            errors.append(message)
    for flag in (
        "--enable-chunked-prefill",
        "--enable-auto-tool-choice",
        "--enforce-eager",
        "--no-async-scheduling",
        "--no-enable-prefix-caching",
        "--no-language-model-only",
        "--no-skip-mm-profiling",
    ):
        if flag not in argv:
            errors.append(f"launch argv must explicitly contain {flag}")

    expected_scalars = {
        "--model": "${AIMA_MODEL_DIR}",
        "--dtype": "bfloat16",
        "--max-model-len": "262144",
        "--max-num-batched-tokens": str(REFERENCE_MAX_BATCHED_TOKENS),
        "--max-num-seqs": "1",
        "--attention-backend": REFERENCE_ATTENTION_BACKEND,
        "--mm-encoder-attn-backend": REFERENCE_ATTENTION_BACKEND,
        "--gdn-prefill-backend": "triton",
        "--tool-call-parser": "qwen3_xml",
        "--load-format": "safetensors",
        "--tensor-parallel-size": "1",
    }
    for option, expected in expected_scalars.items():
        actual = _option_value(argv, option)
        if actual != expected:
            errors.append(f"{option} must be {expected!r}, got {actual!r}")
    if not _option_value(argv, "--served-model-name"):
        errors.append("--served-model-name must be explicit")

    environment_policy = config.get("environment_policy")
    if not isinstance(environment_policy, dict):
        errors.append("environment_policy must be an object")
    else:
        if environment_policy.get("inherit") is not False:
            errors.append("reference launch must use a clean non-inherited environment")
        variables = environment_policy.get("variables")
        if not isinstance(variables, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in variables.items()
        ):
            errors.append("environment_policy.variables must be a string map")
        else:
            sensitive = sorted(
                key for key in variables if _SENSITIVE_ENVIRONMENT_KEY.search(key)
            )
            if sensitive:
                errors.append(
                    "reference environment must not embed credential variables: "
                    + ", ".join(sensitive)
                )
            for required in (
                "HF_HUB_OFFLINE",
                "TRANSFORMERS_OFFLINE",
                "PYTHONHASHSEED",
                "PYTORCH_ROCM_ARCH",
                "VLLM_IMAGE_FETCH_TIMEOUT",
                "VLLM_VIDEO_FETCH_TIMEOUT",
            ):
                if required not in variables:
                    errors.append(f"reference environment must freeze {required}")

    media_policy = config.get("media_policy")
    if not isinstance(media_policy, dict):
        errors.append("media_policy must be an object")
    else:
        limits = media_policy.get("limits")
        if not isinstance(limits, dict):
            errors.append("media_policy.limits must be an object")
        else:
            for modality in ("image", "video"):
                count = limits.get(modality)
                if not isinstance(count, int) or isinstance(count, bool) or count < 2:
                    errors.append(
                        f"media_policy.limits.{modality} must be at least 2"
                    )
            if limits != REFERENCE_MEDIA_LIMITS:
                errors.append(
                    "media_policy.limits must equal the processor-derived "
                    f"reference limits {REFERENCE_MEDIA_LIMITS}"
                )
        parsed_limits = _json_option(argv, "--limit-mm-per-prompt", errors)
        if parsed_limits is not None and parsed_limits != limits:
            errors.append("--limit-mm-per-prompt must equal media_policy.limits")

        local_paths = media_policy.get("allowed_local_media_paths")
        if not isinstance(local_paths, list) or not local_paths or not all(
            isinstance(item, str) and item for item in local_paths
        ):
            errors.append("media_policy.allowed_local_media_paths must be non-empty")
        elif len(local_paths) != 1 or _option_value(
            argv, "--allowed-local-media-path"
        ) != local_paths[0]:
            errors.append(
                "--allowed-local-media-path must equal the single frozen local path"
            )

        domains = media_policy.get("allowed_media_domains")
        if not isinstance(domains, list) or not domains or not all(
            isinstance(item, str) and item for item in domains
        ):
            errors.append("media_policy.allowed_media_domains must be non-empty")
        elif _option_values(argv, "--allowed-media-domains") != domains:
            errors.append("--allowed-media-domains must equal the frozen domain list")

        for field, option in (
            ("media_io_kwargs", "--media-io-kwargs"),
            ("mm_processor_kwargs", "--mm-processor-kwargs"),
        ):
            expected = media_policy.get(field)
            if not isinstance(expected, dict):
                errors.append(f"media_policy.{field} must be an object")
                continue
            parsed = _json_option(argv, option, errors)
            if parsed is not None and parsed != expected:
                errors.append(f"{option} must equal media_policy.{field}")

        media_io = media_policy.get("media_io_kwargs")
        video_io = media_io.get("video") if isinstance(media_io, dict) else None
        if not isinstance(video_io, dict) or video_io.get("video_backend") not in {
            "opencv",
            "opencv_dynamic",
        }:
            errors.append("video_backend must be frozen to a supported OpenCV path")
        if not isinstance(video_io, dict) or not any(
            key in video_io for key in ("fps", "num_frames")
        ):
            errors.append("video fps or frame-count sampling must be explicit")

        cache_gb = media_policy.get("processor_cache_gb")
        if not isinstance(cache_gb, (int, float)) or isinstance(cache_gb, bool):
            errors.append("media_policy.processor_cache_gb must be numeric")
        else:
            raw = _option_value(argv, "--mm-processor-cache-gb")
            try:
                parsed_cache = float(raw) if raw is not None else None
            except ValueError:
                parsed_cache = None
            if parsed_cache != float(cache_gb):
                errors.append(
                    "--mm-processor-cache-gb must equal media_policy.processor_cache_gb"
                )

        pruning_rate = media_policy.get("video_pruning_rate")
        if pruning_rate != 0:
            errors.append("video_pruning_rate must be 0 for original-model semantics")
        raw_pruning = _option_value(argv, "--video-pruning-rate")
        try:
            parsed_pruning = float(raw_pruning) if raw_pruning is not None else None
        except ValueError:
            parsed_pruning = None
        if parsed_pruning != 0.0:
            errors.append("--video-pruning-rate must explicitly be 0")

    boundary = config.get("measurement_boundary")
    if not isinstance(boundary, dict):
        errors.append("measurement_boundary must be an object")
    else:
        for phase in REQUIRED_TIMING_PHASES:
            if boundary.get(phase) is not True:
                errors.append(f"measurement boundary must include {phase}")

    capability = config.get("capability_manifest")
    if not isinstance(capability, dict):
        errors.append("capability_manifest binding must be an object")
    else:
        if capability.get("schema") != CAPABILITY_SCHEMA:
            errors.append(f"capability_manifest.schema must be {CAPABILITY_SCHEMA}")
        if not isinstance(capability.get("path"), str) or not capability.get("path"):
            errors.append("capability_manifest.path must be explicit")
        if not isinstance(capability.get("sha256"), str) or not _SHA256.fullmatch(
            capability.get("sha256", "")
        ):
            errors.append("capability_manifest.sha256 must be a SHA-256 digest")

    return errors


def require_valid_launch_config(config: Mapping[str, Any]) -> None:
    errors = validate_launch_config(config)
    if errors:
        raise ReferenceManifestError("invalid VL reference launch:\n- " + "\n- ".join(errors))


def _safe_model_member(model_dir: Path, relative: str) -> Path:
    if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ReferenceManifestError(f"unsafe model member path: {relative!r}")
    root = model_dir.resolve()
    candidate = (model_dir / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ReferenceManifestError(
            f"model member resolves outside the model directory: {relative!r}"
        ) from exc
    return candidate


def model_file_components(model_dir: Path) -> dict[str, dict[str, Any]]:
    """Hash all identity/processor files and every shard named by the index."""

    components: dict[str, dict[str, Any]] = {}
    for name in REQUIRED_MODEL_FILES:
        path = _safe_model_member(model_dir, name)
        components[name] = file_component(path, name)

    index = load_json_object(model_dir / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ReferenceManifestError("checkpoint index has no non-empty weight_map")
    shard_values = list(weight_map.values())
    if not all(isinstance(item, str) for item in shard_values):
        raise ReferenceManifestError("checkpoint weight_map contains non-string shards")
    shard_names = sorted(set(shard_values))
    for name in shard_names:
        path = _safe_model_member(model_dir, name)
        components[name] = file_component(path, name)
    return components


def verify_components(
    components: Mapping[str, Any], root: Path, *, label: str
) -> list[str]:
    errors: list[str] = []
    for name, record in components.items():
        if not isinstance(name, str) or not isinstance(record, dict):
            errors.append(f"{label} contains a malformed component")
            continue
        try:
            path = _safe_model_member(root, name)
        except ReferenceManifestError as exc:
            errors.append(str(exc))
            continue
        if not path.is_file():
            errors.append(f"{label} file is missing: {name}")
            continue
        if record.get("bytes") != path.stat().st_size:
            errors.append(f"{label} size mismatch: {name}")
        if record.get("sha256") != sha256_file(path):
            errors.append(f"{label} SHA-256 mismatch: {name}")
    return errors


def _distribution_component(name: str) -> dict[str, Any] | None:
    try:
        distribution = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return None
    files: dict[str, dict[str, Any]] = {}
    for candidate_name in ("METADATA", "RECORD", "direct_url.json"):
        candidates = [
            item
            for item in (distribution.files or ())
            if item.name == candidate_name and ".dist-info" in str(item)
        ]
        if not candidates:
            continue
        candidate = candidates[0]
        path = Path(distribution.locate_file(candidate))
        if path.is_file():
            files[candidate_name.lower().replace(".", "_")] = file_component(
                path, str(candidate)
            )
    return {
        "name": distribution.metadata.get("Name", name),
        "version": distribution.version,
        "metadata_files": files,
    }


def _version_matches(installed: str, expected: str) -> bool:
    return installed == expected or installed.startswith(expected + ".")


def runtime_components() -> dict[str, Any]:
    packages: dict[str, Any] = {}
    for name in (*PINNED_PACKAGES, *MEDIA_PACKAGES):
        component = _distribution_component(name)
        if component is not None:
            packages[name] = component

    errors: list[str] = []
    for name, expected in PINNED_PACKAGES.items():
        actual = packages.get(name, {}).get("version")
        if not isinstance(actual, str) or not _version_matches(actual, expected):
            errors.append(f"{name} must be {expected}, got {actual!r}")
    if errors:
        raise ReferenceManifestError("runtime pin mismatch:\n- " + "\n- ".join(errors))

    modules: dict[str, dict[str, Any]] = {}
    for name in REQUIRED_SOURCE_MODULES:
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None:
            raise ReferenceManifestError(f"cannot locate reference source module: {name}")
        path = Path(spec.origin).resolve()
        modules[name] = file_component(path, name.replace(".", "/") + path.suffix)

    executable = Path(sys.executable).resolve()
    return {
        "python": file_component(executable, "${AIMA_VLLM_PYTHON}"),
        "python_version": platform.python_version(),
        "packages": packages,
        "source_modules": modules,
    }


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    return str(value)


def _processor_component(processor: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    attributes = (
        "size",
        "patch_size",
        "temporal_patch_size",
        "merge_size",
        "image_mean",
        "image_std",
        "fps",
        "min_frames",
        "max_frames",
        "do_resize",
        "do_rescale",
        "rescale_factor",
        "do_normalize",
        "do_convert_rgb",
    )

    def describe(value: Any) -> dict[str, Any]:
        return {
            "class": f"{type(value).__module__}.{type(value).__name__}",
            "parameters": {
                name: _json_safe(getattr(value, name, None)) for name in attributes
            },
        }

    tokenizer = processor.tokenizer
    chat_template = getattr(tokenizer, "chat_template", None)
    special_names = (
        "image_token",
        "image_token_id",
        "video_token",
        "video_token_id",
        "vision_start_token",
        "vision_end_token",
    )
    return {
        "class": f"{type(processor).__module__}.{type(processor).__name__}",
        "tokenizer_class": f"{type(tokenizer).__module__}.{type(tokenizer).__name__}",
        "tokenizer_vocab_size": len(tokenizer),
        "chat_template_sha256": (
            sha256_bytes(chat_template.encode("utf-8"))
            if isinstance(chat_template, str)
            else None
        ),
        "special_tokens": {
            name: _json_safe(getattr(processor, name, None)) for name in special_names
        }
        | {
            name: _json_safe(config.get(name))
            for name in (
                "vision_start_token_id",
                "vision_end_token_id",
                "image_token_id",
                "video_token_id",
            )
        },
        "image": describe(processor.image_processor),
        "video": describe(processor.video_processor),
    }


def processor_identity(model_dir: Path) -> dict[str, Any]:
    try:
        from transformers import AutoProcessor
    except ImportError as exc:
        raise ReferenceManifestError("Transformers is required to capture processor identity") from exc
    processor = AutoProcessor.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=False,
    )
    config = load_json_object(model_dir / "config.json")
    identity = _processor_component(processor, config)
    if not identity["class"].endswith(".Qwen3VLProcessor"):
        raise ReferenceManifestError(
            f"unexpected processor class: {identity['class']}"
        )
    return identity


def _command_fingerprint(command: Sequence[str]) -> dict[str, Any]:
    executable = shutil.which(command[0])
    if executable is None:
        return {"command": list(command), "available": False}
    completed = subprocess.run(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=False,
        check=False,
        timeout=30,
    )
    output = completed.stdout
    return {
        "command": list(command),
        "available": True,
        "executable": file_component(Path(executable).resolve(), Path(executable).name),
        "returncode": completed.returncode,
        "output_bytes": len(output),
        "output_sha256": sha256_bytes(output),
    }


def _device_node(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "present": False}
    metadata = path.stat()
    return {
        "path": str(path),
        "present": True,
        "character_device": stat.S_ISCHR(metadata.st_mode),
        "major": os.major(metadata.st_rdev),
        "minor": os.minor(metadata.st_rdev),
        "mode": oct(stat.S_IMODE(metadata.st_mode)),
    }


def host_identity(label: str) -> dict[str, Any]:
    os_release: dict[str, str] = {}
    release_path = Path("/etc/os-release")
    if release_path.is_file():
        for line in release_path.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition("=")
            if separator:
                os_release[key] = value.strip().strip('"')

    mem_total_kib: int | None = None
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        match = re.search(r"^MemTotal:\s+(\d+)\s+kB$", meminfo.read_text(), re.MULTILINE)
        if match:
            mem_total_kib = int(match.group(1))

    cpu_model: str | None = None
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        match = re.search(r"^model name\s*:\s*(.+)$", cpuinfo.read_text(), re.MULTILINE)
        if match:
            cpu_model = match.group(1).strip()

    rocminfo = _command_fingerprint(("rocminfo",))
    if rocminfo.get("available"):
        completed = subprocess.run(
            ["rocminfo"], capture_output=True, text=True, check=False, timeout=30
        )
        rocminfo["gpu_architectures"] = sorted(
            set(re.findall(r"\bgfx[0-9a-f]+\b", completed.stdout, re.IGNORECASE))
        )
    rocm_smi = _command_fingerprint(
        ("rocm-smi", "--showproductname", "--showdriverversion", "--showvbios")
    )
    version_paths = (
        Path("/opt/rocm/.info/version"),
        Path("/opt/rocm/.info/version-dev"),
    )
    rocm_versions = {
        str(path): path.read_text(encoding="utf-8").strip()
        for path in version_paths
        if path.is_file()
    }
    return {
        "label": label,
        "hostname": platform.node(),
        "kernel": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "os_release": {
            key: os_release.get(key)
            for key in ("ID", "VERSION_ID", "PRETTY_NAME")
            if key in os_release
        },
        "cpu_model": cpu_model,
        "logical_cpu_count": os.cpu_count(),
        "mem_total_kib": mem_total_kib,
        "device_nodes": [
            _device_node(Path("/dev/kfd")),
            _device_node(Path("/dev/dri/renderD128")),
        ],
        "rocm_versions": rocm_versions,
        "rocminfo": rocminfo,
        "rocm_smi": rocm_smi,
        "ffmpeg": _command_fingerprint(("ffmpeg", "-version")),
        "ffprobe": _command_fingerprint(("ffprobe", "-version")),
    }


def git_identity(root: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise ReferenceManifestError(
                f"git {' '.join(arguments)} failed: {completed.stderr.strip()}"
            )
        return completed.stdout.strip()

    status = run("status", "--porcelain", "--untracked-files=normal")
    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(status),
        "status_sha256": sha256_bytes(status.encode("utf-8")),
    }


def seal_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    if "integrity" in payload:
        raise ReferenceManifestError("cannot seal a manifest that already has integrity")
    sealed = dict(payload)
    sealed["integrity"] = {
        "algorithm": "sha256",
        "canonical_payload_sha256": canonical_json_sha256(payload),
    }
    return sealed


def verify_manifest_integrity(manifest: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    integrity = manifest.get("integrity")
    if not isinstance(integrity, dict):
        return ["manifest has no integrity object"]
    expected = integrity.get("canonical_payload_sha256")
    payload = {key: value for key, value in manifest.items() if key != "integrity"}
    actual = canonical_json_sha256(payload)
    if integrity.get("algorithm") != "sha256":
        errors.append("manifest integrity algorithm must be sha256")
    if expected != actual:
        errors.append("manifest canonical payload SHA-256 mismatch")
    return errors


def build_reference_manifest(
    *,
    root: Path,
    model_dir: Path,
    launch_config_path: Path,
    capability_manifest_path: Path,
    product_contract_path: Path,
    goal_document_path: Path,
    host_label: str,
    captured_at: str | None = None,
) -> dict[str, Any]:
    launch_config = load_json_object(launch_config_path)
    require_valid_launch_config(launch_config)
    capability = load_json_object(capability_manifest_path)
    capability_binding = launch_config["capability_manifest"]
    if capability.get("schema") != CAPABILITY_SCHEMA:
        raise ReferenceManifestError(
            f"capability manifest schema must be {CAPABILITY_SCHEMA}"
        )
    if capability.get("complete") is not True:
        raise ReferenceManifestError("capability manifest is not complete")
    if capability.get("qualified") is not True:
        raise ReferenceManifestError("capability manifest is not qualified")
    capability_integrity_errors = verify_manifest_integrity(capability)
    if capability_integrity_errors:
        raise ReferenceManifestError(
            "capability manifest integrity failed:\n- "
            + "\n- ".join(capability_integrity_errors)
        )
    capability_sha256 = sha256_file(capability_manifest_path)
    if capability_binding["sha256"] != capability_sha256:
        raise ReferenceManifestError("launch config capability SHA-256 mismatch")

    product_contract = load_json_object(product_contract_path)
    if product_contract.get("release") != BASELINE_RELEASE.removeprefix("v"):
        raise ReferenceManifestError("product contract is not the frozen v1.5.1 baseline")
    model_contract = product_contract.get("model")
    if not isinstance(model_contract, dict):
        raise ReferenceManifestError("product contract has no model identity")
    if model_contract.get("source_revision") != MODEL_REVISION:
        raise ReferenceManifestError("product contract model revision drifted")

    model_files = model_file_components(model_dir)
    for field, filename in (
        ("config_sha256", "config.json"),
        ("checkpoint_index_sha256", "model.safetensors.index.json"),
        ("tokenizer_sha256", "tokenizer.json"),
        ("tokenizer_config_sha256", "tokenizer_config.json"),
    ):
        if model_contract.get(field) != model_files[filename]["sha256"]:
            raise ReferenceManifestError(f"frozen model hash mismatch: {filename}")

    source = git_identity(root)
    if source["dirty"]:
        raise ReferenceManifestError(
            "reference capture source is dirty; commit the capture inputs before oracle use"
        )

    checkpoint_index = load_json_object(model_dir / "model.safetensors.index.json")
    weight_map = checkpoint_index["weight_map"]
    payload: dict[str, Any] = {
        "schema": REFERENCE_SCHEMA,
        "complete": True,
        "qualified_for_oracle_capture": True,
        "captured_at": captured_at
        or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "baseline": {
            "release": BASELINE_RELEASE,
            "release_commit": BASELINE_COMMIT,
            "native_source_commit": BASELINE_NATIVE_COMMIT,
            "product_contract": file_component(
                product_contract_path, "native/product-contract-v1.5.1.json"
            ),
            "goal_document": file_component(
                goal_document_path, "docs/NATIVE_VL_GOAL.md"
            ),
        },
        "capture_source": source,
        "model": {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "dtype": "bfloat16",
            "model_dir": "${AIMA_MODEL_DIR}",
            "tensor_count": len(weight_map),
            "checkpoint_payload_bytes": checkpoint_index.get("metadata", {}).get(
                "total_size"
            ),
            "files": model_files,
        },
        "processor": processor_identity(model_dir),
        "reference_runtime": runtime_components(),
        "host": host_identity(host_label),
        "launch": launch_config,
        "capability_manifest": file_component(
            capability_manifest_path, capability_binding["path"]
        ),
        "capture_inputs": {
            "launch_config": file_component(
                launch_config_path, str(launch_config_path.relative_to(root))
            )
        },
    }
    architectures = payload["host"]["rocminfo"].get("gpu_architectures", [])
    if "gfx1151" not in architectures:
        raise ReferenceManifestError(
            f"reference host is not the frozen gfx1151 target: {architectures}"
        )
    return seal_manifest(payload)


def verify_reference_manifest(
    manifest: Mapping[str, Any], *, model_dir: Path
) -> list[str]:
    errors = verify_manifest_integrity(manifest)
    if manifest.get("schema") != REFERENCE_SCHEMA:
        errors.append(f"manifest schema must be {REFERENCE_SCHEMA}")
    if manifest.get("complete") is not True:
        errors.append("manifest is not complete")
    if manifest.get("qualified_for_oracle_capture") is not True:
        errors.append("manifest is not qualified for oracle capture")
    model = manifest.get("model")
    if not isinstance(model, dict) or not isinstance(model.get("files"), dict):
        errors.append("manifest has no model file components")
    else:
        errors.extend(verify_components(model["files"], model_dir, label="model"))
    return errors
