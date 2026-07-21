#!/usr/bin/env python3
"""Admitted request-local context policy for the resident AMD395 service."""

from __future__ import annotations

import contextlib
import ctypes
import hashlib
import importlib.util
import os
import socket
import struct
import threading
from pathlib import Path
from typing import Any, Iterator


SOURCE_LIBRARY_NAME = "libaotriton_v2.so.0.11.1"
SOURCE_LIBRARY_SHA256 = "e0638806efa5d35cef04fd7fb02c62cd038b3a38727ecb5d87a49045aa1b9aa5"
LUT_CALL_VIRTUAL_ADDRESS = 0x345AF2
LUT_CALL_ORIGINAL_BYTES = bytes.fromhex("e8c9990300")
SELECTED_INDEX = 7
EXACT_ADMITTED_POLICY: dict[int, tuple[str, int | None]] = {
    8192: ("seq", None),
    16384: ("seq", None),
    32768: ("grouped", None),
    65536: ("seq", SELECTED_INDEX),
    131072: ("seq", SELECTED_INDEX),
}
_PROCESS_PATCH_LOCK = threading.Lock()


def source_library() -> Path:
    override = os.environ.get("AIMA_AOTRITON_LIBRARY")
    if override:
        return Path(override).expanduser().resolve()
    spec = importlib.util.find_spec("torch")
    if spec is None or spec.origin is None:
        raise RuntimeError("cannot locate the installed torch package")
    return Path(spec.origin).resolve().parent / "lib" / SOURCE_LIBRARY_NAME


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def select_policy(
    *,
    prompt_tokens: int,
    enabled: bool,
    exact_prefix_cache: bool,
    exact_prefix_cache_max_tokens: int,
    fallback_layout: str,
) -> dict[str, Any]:
    """Select only an exact measured cold policy; never interpolate by threshold."""
    require(prompt_tokens > 0, "prompt_tokens must be positive")
    require(exact_prefix_cache_max_tokens >= 0, "prefix-cache token bound must be non-negative")
    require(fallback_layout in {"seq", "grouped"}, "unsupported fallback KV layout")
    fixed = EXACT_ADMITTED_POLICY.get(prompt_tokens)
    prefix_bypassed = not exact_prefix_cache or prompt_tokens > exact_prefix_cache_max_tokens
    active = bool(enabled and prefix_bypassed and fixed is not None)
    layout, schedule_index = fixed if active and fixed is not None else (fallback_layout, None)
    if not enabled:
        reason = "explicit_opt_out"
    elif fixed is None:
        reason = "unmeasured_prompt_length"
    elif not prefix_bypassed:
        reason = "prefix_cache_eligible"
    else:
        reason = "exact_cold_admitted_length"
    return {
        "schema": "aima-amd395-qwen36/context-policy-selection/v1",
        "enabled": bool(enabled),
        "active": active,
        "reason": reason,
        "prompt_tokens": prompt_tokens,
        "exact_length_match": fixed is not None,
        "cold_prefix_bypass": prefix_bypassed,
        "exact_prefix_cache": bool(exact_prefix_cache),
        "exact_prefix_cache_max_tokens": exact_prefix_cache_max_tokens,
        "fallback_layout": fallback_layout,
        "kv_layout": layout,
        "schedule_index": schedule_index,
        "policy_source": "v1.0.0 exact cold-context admission",
    }


def mapped_library_records() -> list[dict[str, Any]]:
    source = str(source_library())
    rows: list[dict[str, Any]] = []
    for line in Path("/proc/self/maps").read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 6 or fields[-1] != source:
            continue
        start_text, end_text = fields[0].split("-")
        rows.append(
            {
                "start": int(start_text, 16),
                "end": int(end_text, 16),
                "perms": fields[1],
                "offset": int(fields[2], 16),
                "path": fields[-1],
                "raw": line,
            }
        )
    return rows


def _location() -> tuple[list[dict[str, Any]], dict[str, Any], int, int]:
    records = mapped_library_records()
    require(records, "installed AOTriton mapping absent")
    for row in records:
        base = int(row["start"]) - int(row["offset"])
        address = base + LUT_CALL_VIRTUAL_ADDRESS
        if int(row["start"]) <= address < int(row["end"]) and "x" in str(row["perms"]):
            require(str(row["perms"]).endswith("p"), f"mapping is not private: {row['perms']}")
            return records, row, base, address
    raise RuntimeError("executable mapping for LUT call absent")


def current_state() -> dict[str, Any]:
    library = source_library()
    records, mapping, base, address = _location()
    return {
        "host": socket.gethostname(),
        "library": str(library),
        "library_sha256": sha256(library),
        "base_address": f"0x{base:x}",
        "runtime_address": f"0x{address:x}",
        "virtual_address": f"0x{LUT_CALL_VIRTUAL_ADDRESS:x}",
        "runtime_bytes_hex": ctypes.string_at(address, len(LUT_CALL_ORIGINAL_BYTES)).hex(),
        "mapping": mapping,
        "mapping_count": len(records),
    }


def assert_unpatched() -> dict[str, Any]:
    state = current_state()
    require(state["library_sha256"] == SOURCE_LIBRARY_SHA256, "installed library sha mismatch")
    require(state["runtime_bytes_hex"] == LUT_CALL_ORIGINAL_BYTES.hex(), "LUT mapping is not original")
    return state


class Binding:
    def __init__(self, *, address: int, page: int, page_size: int, libc: Any, info: dict[str, Any]) -> None:
        self._address = address
        self._page = page
        self._page_size = page_size
        self._libc = libc
        self.info = info
        self._restored = False

    def restore(self) -> dict[str, Any]:
        require(not self._restored, "schedule binding already restored")
        prot_rwx = 1 | 2 | 4
        prot_rx = 1 | 4
        require(
            self._libc.mprotect(
                ctypes.c_void_p(self._page), ctypes.c_size_t(self._page_size), prot_rwx
            )
            == 0,
            f"restore mprotect RWX failed errno={ctypes.get_errno()}",
        )
        ctypes.memmove(self._address, LUT_CALL_ORIGINAL_BYTES, len(LUT_CALL_ORIGINAL_BYTES))
        require(
            self._libc.mprotect(
                ctypes.c_void_p(self._page), ctypes.c_size_t(self._page_size), prot_rx
            )
            == 0,
            f"restore mprotect RX failed errno={ctypes.get_errno()}",
        )
        self._restored = True
        state = assert_unpatched()
        return {
            "restored": True,
            "restored_bytes_hex": state["runtime_bytes_hex"],
            "disk_library_sha256": state["library_sha256"],
        }


def install(index: int = SELECTED_INDEX) -> Binding:
    require(index == SELECTED_INDEX, f"only preregistered index{SELECTED_INDEX} is allowed")
    require(
        os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL") == "1",
        "AOTriton experimental environment is absent",
    )
    import torch

    require(torch.cuda.is_available(), "ROCm device unavailable")
    require(torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx1151"), "gfx1151 mismatch")
    library = source_library()
    require(sha256(library) == SOURCE_LIBRARY_SHA256, "installed library sha mismatch")
    before = assert_unpatched()
    _, mapping, base, address = _location()
    patch = bytes([0xB8]) + struct.pack("<I", index)
    page_size = os.sysconf("SC_PAGE_SIZE")
    page = address & ~(page_size - 1)
    libc = ctypes.CDLL(None, use_errno=True)
    prot_rwx = 1 | 2 | 4
    prot_rx = 1 | 4
    require(
        libc.mprotect(ctypes.c_void_p(page), ctypes.c_size_t(page_size), prot_rwx) == 0,
        f"mprotect RWX failed errno={ctypes.get_errno()}",
    )
    ctypes.memmove(address, patch, len(patch))
    require(
        libc.mprotect(ctypes.c_void_p(page), ctypes.c_size_t(page_size), prot_rx) == 0,
        f"mprotect RX restore failed errno={ctypes.get_errno()}",
    )
    active = ctypes.string_at(address, len(patch))
    require(active == patch, "process-private patch verification failed")
    disk_sha = sha256(library)
    require(disk_sha == SOURCE_LIBRARY_SHA256, "installed library changed during mapping patch")
    info = {
        "selected_index": index,
        "arch_module": 5,
        "function_number": 354,
        "isolation": "MAP_PRIVATE process-local copy-on-write executable mapping",
        "library": str(library),
        "library_sha256": disk_sha,
        "base_address": f"0x{base:x}",
        "runtime_address": f"0x{address:x}",
        "virtual_address": f"0x{LUT_CALL_VIRTUAL_ADDRESS:x}",
        "mapping": mapping,
        "before_bytes_hex": before["runtime_bytes_hex"],
        "patch_bytes_hex": patch.hex(),
        "active_bytes_hex": active.hex(),
        "page_size": page_size,
    }
    return Binding(address=address, page=page, page_size=page_size, libc=libc, info=info)


@contextlib.contextmanager
def bind_for_request(selection: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Install index7 only for the selected request and restore it unconditionally."""
    index = selection.get("schedule_index")
    evidence: dict[str, Any] = {
        "requested_schedule_index": index,
        "installed": False,
        "restored": index is None,
        "lock_scope": "process",
    }
    if index is None:
        yield evidence
        return
    require(selection.get("active") is True, "schedule binding requires active policy")
    require(index == SELECTED_INDEX, "only admitted index7 may be bound")
    with _PROCESS_PATCH_LOCK:
        binding = install(index)
        evidence["installed"] = True
        evidence["install"] = binding.info
        try:
            yield evidence
        finally:
            evidence["restore"] = binding.restore()
            evidence["restored"] = True
