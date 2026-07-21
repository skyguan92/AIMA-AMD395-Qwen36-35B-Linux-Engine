"""Automatic exact-q8192 binding for the strict-passing CK-FMHA component."""

from __future__ import annotations

import ctypes
import hashlib
from pathlib import Path
from typing import Any

from packed_gqa_mha_kernel import packed_gqa_attention
from partial_persistent_tilequeue_d256_kernel import (
    PERSISTENT_PROGRAMS as PARTIAL_PERSISTENT_PROGRAMS,
    QUERY_ROWS_PER_TILE as PARTIAL_QUERY_ROWS_PER_TILE,
    SUPPORTED_SHAPES as PARTIAL_SUPPORTED_SHAPES,
    partial_persistent_tilequeue_d256_attention,
)
from persistent_tilequeue_d256_kernel import (
    PERSISTENT_PROGRAMS,
    TOTAL_QUERY_TILES,
    persistent_tilequeue_d256_attention,
)


_TOKENS = 8192
_Q_FEATURES = 4096
_CK_LAYERS = frozenset(range(3, 39, 4))
_Q16_CK_LAYERS = _CK_LAYERS
_COMPONENTS = frozenset(("selected_moe", "ck_fmha", "fused_gdn"))
_ACTIVE_COMPONENTS = frozenset(("ck_fmha",))
_CK_LIBRARY = Path(__file__).resolve().parent / "native/libqrt_ck_fmha_q8192_provider.so"
_CK_LIBRARY_SHA256 = "5d03c4ea9491af3fc5505b0ce8fc052c33055254e514dd6c674f5d5ccf7fa7c9"
_Q16_CK_LIBRARY = (
    Path(__file__).resolve().parent
    / "native/libqrt_ck_fmha_q8192_kv16384_bottom_right_provider.so"
)
_Q16_CK_LIBRARY_SHA256 = "22132b23c649daaa8c85799cb41705a1cd4b00cc1088143d5ad6ae6dc1618865"
_PACKED_KERNEL = Path(__file__).resolve().parent / "packed_gqa_mha_kernel.py"
_PACKED_KERNEL_SHA256 = "00d2d507d0f14ef855ba593f9737188284b47fba3b124e9d582e602f713b5085"

_PERSISTENT_KERNEL = Path(__file__).resolve().parent / "persistent_tilequeue_d256_kernel.py"
_PERSISTENT_KERNEL_SHA256 = "0aa7bfd289ba4c7450d913bf1fb83906e56fe58d6f42f084b7e738ebb404a075"
_PERSISTENT_CACHE_ENDS = frozenset(range(24576, 253953, 8192))
_PARTIAL_PERSISTENT_KERNEL = (
    Path(__file__).resolve().parent / "partial_persistent_tilequeue_d256_kernel.py"
)
_PARTIAL_PERSISTENT_KERNEL_SHA256 = (
    "6365d607bd71bdeb1ea9900fe12a6e15cb986f6cc30892a7e10ac5e0fbde1573"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pointer(tensor: Any) -> ctypes.c_void_p:
    if not tensor.is_cuda or not tensor.is_contiguous():
        raise RuntimeError("q8192 CK-provider tensors must be contiguous ROCm tensors")
    return ctypes.c_void_p(int(tensor.data_ptr()))


def _stream_pointer(torch: Any) -> ctypes.c_void_p:
    return ctypes.c_void_p(int(torch.cuda.current_stream().cuda_stream))


class Q8192CompoundProvider:
    """Preserve the existing engine API while owning only admitted CK math."""

    def __init__(self) -> None:
        self._prepared = False
        self._ck: Any | None = None
        self._q16_prepared = False
        self._q16_ck: Any | None = None
        self._workspaces: dict[str, Any] = {}
        self._counts = {"selected_moe": 0, "ck_fmha": 0, "fused_gdn": 0}
        self._model_layers = {"selected_moe": [], "ck_fmha": [], "fused_gdn": []}
        self._q16_hybrid_calls = 0
        self._q16_hybrid_model_layers: list[int] = []
        self._persistent_tilequeue_prepared = False
        self._persistent_tilequeue_calls = 0
        self._persistent_tilequeue_model_layers: list[int] = []
        self._persistent_tilequeue_cache_ends: list[int] = []
        self._persistent_tilequeue_layouts: list[str] = []
        self._partial_persistent_tilequeue_prepared = False
        self._partial_persistent_tilequeue_calls = 0
        self._partial_persistent_tilequeue_model_layers: list[int] = []
        self._partial_persistent_tilequeue_query_tokens: list[int] = []
        self._partial_persistent_tilequeue_cache_ends: list[int] = []
        self._partial_persistent_tilequeue_layouts: list[str] = []

    @staticmethod
    def enabled() -> bool:
        return True

    @staticmethod
    def active_components() -> frozenset[str]:
        return _ACTIVE_COMPONENTS

    @classmethod
    def component_enabled(cls, name: str) -> bool:
        if name not in _COMPONENTS:
            raise RuntimeError(f"unknown q8192 provider component: {name}")
        return name in cls.active_components()

    @staticmethod
    def exact_request_active(*, mode: str, tokens: int, position_start: int) -> bool:
        return mode == "prefill" and int(tokens) == _TOKENS and int(position_start) == 0

    @staticmethod
    def library_contract() -> dict[str, Any]:
        return {
            "path": str(_CK_LIBRARY),
            "sha256": _CK_LIBRARY_SHA256,
            "module_relative": True,
        }

    def prepare(self) -> None:
        if self._prepared:
            return
        if not _CK_LIBRARY.is_file():
            raise RuntimeError(f"bundled CK-FMHA provider is missing: {_CK_LIBRARY}")
        actual_sha256 = _sha256(_CK_LIBRARY)
        if actual_sha256 != _CK_LIBRARY_SHA256:
            raise RuntimeError(
                "bundled CK-FMHA provider hash mismatch: "
                f"expected {_CK_LIBRARY_SHA256}, got {actual_sha256}"
            )

        ck = ctypes.CDLL(str(_CK_LIBRARY), mode=ctypes.RTLD_LOCAL)
        ck.qrt_ck_fmha_q8192_prepare.argtypes = []
        ck.qrt_ck_fmha_q8192_prepare.restype = ctypes.c_int
        ck.qrt_ck_fmha_q8192_bf16_launch.argtypes = [ctypes.c_void_p] * 5
        ck.qrt_ck_fmha_q8192_bf16_launch.restype = ctypes.c_int
        ck.qrt_ck_fmha_q8192_release.argtypes = []
        ck.qrt_ck_fmha_q8192_release.restype = ctypes.c_int
        if ck.qrt_ck_fmha_q8192_prepare() != 0:
            raise RuntimeError("bundled CK-FMHA prepare failed")
        self._ck = ck
        self._prepared = True

    def prepare_q16_hybrid(self) -> None:
        if self._q16_prepared:
            return
        immutable = (
            (_Q16_CK_LIBRARY, _Q16_CK_LIBRARY_SHA256),
            (_PACKED_KERNEL, _PACKED_KERNEL_SHA256),
        )
        for path, expected in immutable:
            if not path.is_file():
                raise RuntimeError(f"q16 hybrid provider input is missing: {path}")
            actual = _sha256(path)
            if actual != expected:
                raise RuntimeError(
                    f"q16 hybrid provider hash mismatch for {path.name}: "
                    f"expected {expected}, got {actual}"
                )
        ck = ctypes.CDLL(str(_Q16_CK_LIBRARY), mode=ctypes.RTLD_LOCAL)
        ck.qrt_ck_fmha_q8192_kv16384_bottom_right_prepare.argtypes = []
        ck.qrt_ck_fmha_q8192_kv16384_bottom_right_prepare.restype = ctypes.c_int
        ck.qrt_ck_fmha_q8192_kv16384_bottom_right_bf16_launch.argtypes = [
            ctypes.c_void_p
        ] * 5
        ck.qrt_ck_fmha_q8192_kv16384_bottom_right_bf16_launch.restype = ctypes.c_int
        ck.qrt_ck_fmha_q8192_kv16384_bottom_right_release.argtypes = []
        ck.qrt_ck_fmha_q8192_kv16384_bottom_right_release.restype = ctypes.c_int
        if ck.qrt_ck_fmha_q8192_kv16384_bottom_right_prepare() != 0:
            raise RuntimeError("q16 CK-FMHA prepare failed")
        self._q16_ck = ck
        self._q16_prepared = True

    def _workspace(self, name: str, shape: tuple[int, ...], dtype: Any, torch: Any) -> Any:
        value = self._workspaces.get(name)
        if value is None:
            value = torch.empty(shape, device="cuda", dtype=dtype)
            self._workspaces[name] = value
        if tuple(value.shape) != shape or value.dtype != dtype:
            raise RuntimeError(f"q8192 CK-provider workspace drift: {name}")
        return value

    def launch_ck_fmha(self, *, q: Any, k: Any, value: Any, model_layer: int) -> Any:
        if model_layer not in _CK_LAYERS:
            raise RuntimeError(f"model layer {model_layer} is outside the q8192 CK scope")
        import torch

        self.prepare()
        if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or value.dtype != torch.bfloat16:
            raise RuntimeError("CK-FMHA q/k/v inputs must be BF16")
        output = self._workspace("ck_output_f32", (_TOKENS, _Q_FEATURES), torch.float32, torch)
        status = self._ck.qrt_ck_fmha_q8192_bf16_launch(
            _pointer(q),
            _pointer(k),
            _pointer(value),
            _pointer(output),
            _stream_pointer(torch),
        )
        if status != 0:
            raise RuntimeError(f"CK-FMHA launch failed: hip_status={status}")
        self._counts["ck_fmha"] += 1
        self._model_layers["ck_fmha"].append(int(model_layer))
        return output

    def launch_q16_hybrid_attention(
        self,
        *,
        q: Any,
        k: Any,
        value: Any,
        model_layer: int,
    ) -> Any:
        if model_layer not in _Q16_CK_LAYERS:
            raise RuntimeError(f"model layer {model_layer} is outside the q16 hybrid scope")
        import torch

        self.prepare_q16_hybrid()
        expected_q = (_TOKENS, 16, 256)
        expected_kv = (2 * _TOKENS, 2, 256)
        if tuple(q.shape) != expected_q or tuple(k.shape) != expected_kv:
            raise RuntimeError(f"q16 hybrid q/k shape drift: {tuple(q.shape)} / {tuple(k.shape)}")
        if tuple(value.shape) != expected_kv:
            raise RuntimeError(f"q16 hybrid value shape drift: {tuple(value.shape)}")
        if q.dtype != torch.bfloat16 or k.dtype != q.dtype or value.dtype != q.dtype:
            raise RuntimeError("q16 hybrid q/k/v inputs must be BF16")
        if not q.is_contiguous() or not k.is_contiguous() or not value.is_contiguous():
            raise RuntimeError("q16 hybrid q/k/v inputs must be contiguous BSHD storage")

        q_bshd = q.unsqueeze(0)
        k_bshd = k.unsqueeze(0)
        v_bshd = value.unsqueeze(0)
        ck_output = self._workspace(
            "q16_ck_output_f32", (1, _TOKENS, 16, 256), torch.float32, torch
        )
        packed_output = self._workspace(
            "q16_packed_output_bf16", (1, _TOKENS, 16, 256), torch.bfloat16, torch
        )
        hybrid_output = self._workspace(
            "q16_hybrid_output_bf16", (1, _TOKENS, 16, 256), torch.bfloat16, torch
        )
        status = self._q16_ck.qrt_ck_fmha_q8192_kv16384_bottom_right_bf16_launch(
            _pointer(q_bshd),
            _pointer(k_bshd),
            _pointer(v_bshd),
            _pointer(ck_output),
            _stream_pointer(torch),
        )
        if status != 0:
            raise RuntimeError(f"q16 CK-FMHA launch failed: hip_status={status}")
        packed_gqa_attention(
            q_bshd,
            k_bshd,
            v_bshd,
            packed_output,
            softmax_scale=0.0625,
        )
        hybrid_output.copy_(packed_output)
        hybrid_output[:, :, 14:, :].copy_(ck_output[:, :, 14:, :])
        self._q16_hybrid_calls += 1
        self._q16_hybrid_model_layers.append(int(model_layer))
        return hybrid_output.view(_TOKENS, _Q_FEATURES)


    def launch_persistent_tilequeue_attention(
        self,
        *,
        q: Any,
        k: Any,
        value: Any,
        model_layer: int,
    ) -> Any:
        if model_layer not in _CK_LAYERS:
            raise RuntimeError(f"model layer {model_layer} is outside the persistent-tilequeue scope")
        import torch

        if not self._persistent_tilequeue_prepared:
            if not _PERSISTENT_KERNEL.is_file():
                raise RuntimeError(f"persistent-tilequeue kernel is missing: {_PERSISTENT_KERNEL}")
            actual = _sha256(_PERSISTENT_KERNEL)
            if actual != _PERSISTENT_KERNEL_SHA256:
                raise RuntimeError(
                    "persistent-tilequeue kernel hash mismatch: "
                    f"expected {_PERSISTENT_KERNEL_SHA256}, got {actual}"
                )
            self._persistent_tilequeue_prepared = True

        if k.ndim != 3 or tuple(k.shape) != tuple(value.shape):
            raise RuntimeError(
                f"persistent-tilequeue matching rank-3 k/v required: {tuple(k.shape)} / {tuple(value.shape)}"
            )
        if k.shape[1:] != (2, 256):
            raise RuntimeError(f"persistent-tilequeue direct-seq BSHD k/v required: {tuple(k.shape)}")
        cache_end = int(k.shape[0])
        if cache_end not in _PERSISTENT_CACHE_ENDS:
            raise RuntimeError(f"persistent-tilequeue cache-end drift: {cache_end}")
        if tuple(q.shape) != (_TOKENS, 16, 256):
            raise RuntimeError(f"persistent-tilequeue q shape drift: {tuple(q.shape)}")
        if q.dtype != torch.bfloat16 or k.dtype != q.dtype or value.dtype != q.dtype:
            raise RuntimeError("persistent-tilequeue q/k/v inputs must be BF16")
        if not q.is_contiguous() or not k.is_contiguous() or not value.is_contiguous():
            raise RuntimeError("persistent-tilequeue q/k/v inputs must be contiguous direct-seq BSHD")

        output = self._workspace(
            "persistent_tilequeue_output_bf16",
            (1, _TOKENS, 16, 256),
            torch.bfloat16,
            torch,
        )
        persistent_tilequeue_d256_attention(
            q.unsqueeze(0), k.unsqueeze(0), value.unsqueeze(0), output
        )
        self._persistent_tilequeue_calls += 1
        self._persistent_tilequeue_model_layers.append(int(model_layer))
        self._persistent_tilequeue_cache_ends.append(cache_end)
        self._persistent_tilequeue_layouts.append("BSHD")
        return output.view(_TOKENS, _Q_FEATURES)

    def launch_partial_persistent_tilequeue_attention(
        self,
        *,
        q: Any,
        k: Any,
        value: Any,
        model_layer: int,
    ) -> Any:
        if model_layer not in _CK_LAYERS:
            raise RuntimeError(
                f"model layer {model_layer} is outside the partial persistent-tilequeue scope"
            )
        import torch

        if not self._partial_persistent_tilequeue_prepared:
            if not _PARTIAL_PERSISTENT_KERNEL.is_file():
                raise RuntimeError(
                    f"partial persistent-tilequeue kernel is missing: {_PARTIAL_PERSISTENT_KERNEL}"
                )
            actual = _sha256(_PARTIAL_PERSISTENT_KERNEL)
            if actual != _PARTIAL_PERSISTENT_KERNEL_SHA256:
                raise RuntimeError(
                    "partial persistent-tilequeue kernel hash mismatch: "
                    f"expected {_PARTIAL_PERSISTENT_KERNEL_SHA256}, got {actual}"
                )
            self._partial_persistent_tilequeue_prepared = True

        if k.ndim != 3 or tuple(k.shape) != tuple(value.shape):
            raise RuntimeError(
                "partial persistent-tilequeue matching rank-3 k/v required: "
                f"{tuple(k.shape)} / {tuple(value.shape)}"
            )
        if k.shape[1:] != (2, 256):
            raise RuntimeError(
                f"partial persistent-tilequeue direct-seq BSHD k/v required: {tuple(k.shape)}"
            )
        query_tokens = int(q.shape[0]) if q.ndim == 3 else -1
        cache_end = int(k.shape[0])
        if (query_tokens, cache_end) not in PARTIAL_SUPPORTED_SHAPES:
            raise RuntimeError(
                f"partial persistent-tilequeue shape drift: q{query_tokens}/kv{cache_end}"
            )
        if tuple(q.shape) != (query_tokens, 16, 256):
            raise RuntimeError(f"partial persistent-tilequeue q shape drift: {tuple(q.shape)}")
        if q.dtype != torch.bfloat16 or k.dtype != q.dtype or value.dtype != q.dtype:
            raise RuntimeError("partial persistent-tilequeue q/k/v inputs must be BF16")
        if not q.is_contiguous() or not k.is_contiguous() or not value.is_contiguous():
            raise RuntimeError(
                "partial persistent-tilequeue q/k/v inputs must be contiguous direct-seq BSHD"
            )

        output = self._workspace(
            f"partial_persistent_tilequeue_output_bf16_q{query_tokens}",
            (1, query_tokens, 16, 256),
            torch.bfloat16,
            torch,
        )
        partial_persistent_tilequeue_d256_attention(
            q.unsqueeze(0), k.unsqueeze(0), value.unsqueeze(0), output
        )
        self._partial_persistent_tilequeue_calls += 1
        self._partial_persistent_tilequeue_model_layers.append(int(model_layer))
        self._partial_persistent_tilequeue_query_tokens.append(query_tokens)
        self._partial_persistent_tilequeue_cache_ends.append(cache_end)
        self._partial_persistent_tilequeue_layouts.append("BSHD")
        return output.view(query_tokens, _Q_FEATURES)

    def launch_gdn(self, **_: Any) -> tuple[Any, Any]:
        raise RuntimeError("fused-GDN is not admitted in the production q8192 provider")

    def launch_selected_moe(self, **_: Any) -> Any:
        raise RuntimeError("selected-MoE is not admitted in the production q8192 provider")

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled(),
            "active_components": sorted(self.active_components()),
            "prepared": self._prepared,
            "call_counts": dict(self._counts),
            "model_layers": {key: list(value) for key, value in self._model_layers.items()},
            "q16_hybrid": {
                "prepared": self._q16_prepared,
                "calls": self._q16_hybrid_calls,
                "model_layers": list(self._q16_hybrid_model_layers),
                "packed_heads": 14,
                "ck_heads": 2,
                "policy": "exact-q16384 second-q8192 nonterminal full-attention layers",
                "ck_library_sha256": _Q16_CK_LIBRARY_SHA256,
                "packed_kernel_sha256": _PACKED_KERNEL_SHA256,
            },
            "persistent_tilequeue": {
                "prepared": self._persistent_tilequeue_prepared,
                "calls": self._persistent_tilequeue_calls,
                "model_layers": list(self._persistent_tilequeue_model_layers),
                "cache_ends": list(self._persistent_tilequeue_cache_ends),
                "layouts": list(self._persistent_tilequeue_layouts),
                "policy": "automatic q32/q64/q128 direct-seq suffixes on nonterminal full-attention layers",
                "kernel_sha256": _PERSISTENT_KERNEL_SHA256,
                "persistent_programs": PERSISTENT_PROGRAMS,
                "logical_query_tiles": TOTAL_QUERY_TILES,
                "score_bytes": 0,
                "cross_launch_state_bytes": 0,
            },
            "partial_persistent_tilequeue": {
                "prepared": self._partial_persistent_tilequeue_prepared,
                "calls": self._partial_persistent_tilequeue_calls,
                "model_layers": list(self._partial_persistent_tilequeue_model_layers),
                "query_tokens": list(self._partial_persistent_tilequeue_query_tokens),
                "cache_ends": list(self._partial_persistent_tilequeue_cache_ends),
                "layouts": list(self._partial_persistent_tilequeue_layouts),
                "policy": "automatic q256 final 7168/7680 direct-seq suffix on nonterminal full-attention layers",
                "kernel_sha256": _PARTIAL_PERSISTENT_KERNEL_SHA256,
                "persistent_programs": PARTIAL_PERSISTENT_PROGRAMS,
                "query_rows_per_tile": PARTIAL_QUERY_ROWS_PER_TILE,
                "supported_shapes": [list(shape) for shape in sorted(PARTIAL_SUPPORTED_SHAPES)],
                "score_bytes": 0,
                "cross_launch_state_bytes": 0,
            },
            "workspace_bytes": int(
                sum(value.numel() * value.element_size() for value in self._workspaces.values())
            ),
            "scratch_bytes": (
                {"selected_moe": 0, "fused_gdn": 0} if self._prepared else None
            ),
            "library": self.library_contract(),
        }

    def release(self) -> dict[str, Any]:
        import torch

        if self._prepared:
            torch.cuda.synchronize()
            status = self._ck.qrt_ck_fmha_q8192_release()
            if status != 0:
                raise RuntimeError(f"CK-FMHA release failed: hip_status={status}")
        if self._q16_prepared:
            torch.cuda.synchronize()
            status = self._q16_ck.qrt_ck_fmha_q8192_kv16384_bottom_right_release()
            if status != 0:
                raise RuntimeError(f"q16 CK-FMHA release failed: hip_status={status}")
        self._ck = None
        self._prepared = False
        self._q16_ck = None
        self._q16_prepared = False
        self._persistent_tilequeue_prepared = False
        self._partial_persistent_tilequeue_prepared = False
        self._workspaces.clear()
        torch.cuda.empty_cache()
        return self.stats()


_PROVIDER = Q8192CompoundProvider()


def q8192_compound_provider() -> Q8192CompoundProvider:
    return _PROVIDER


def q8192_compound_provider_stats() -> dict[str, Any]:
    return _PROVIDER.stats()


def q8192_compound_provider_release() -> dict[str, Any]:
    return _PROVIDER.release()
