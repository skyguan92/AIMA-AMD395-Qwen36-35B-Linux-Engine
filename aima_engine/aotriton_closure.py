"""Validate the frozen AOTriton runtime closure used by native qualifiers."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from aima_engine.vl_reference import sha256_file


AOTRITON_RUNTIME_SONAME = "libaotriton_v2.so.0.11.1"
AOTRITON_IMAGE_RELATIVE = Path(
    "aotriton.images/amd-gfx11xx/flash/attn_fwd/"
    "FONLY__＊bf16@16_256_F_F_3_0___gfx11xx.aks2"
)
FROZEN_AOTRITON_RUNTIME_SHA256 = (
    "e0638806efa5d35cef04fd7fb02c62cd038b3a38727ecb5d87a49045aa1b9aa5"
)
FROZEN_AOTRITON_IMAGE_SHA256 = (
    "0f3a6a2f9dee6620443ee2145ee1f8257bde65a378589952840d99bf3d485c10"
)


@dataclass(frozen=True)
class AotritonClosure:
    """Content-bound provider-adjacent AOTriton dependencies."""

    provider: Path
    runtime: Path
    image: Path


def resolve_aotriton_closure(fmha_provider: Path) -> AotritonClosure:
    """Return the frozen closure or fail before a native workload starts."""

    provider = fmha_provider.resolve()
    if not provider.is_file():
        raise RuntimeError("AOTriton FMHA provider is missing")
    runtime = provider.parent / AOTRITON_RUNTIME_SONAME
    image = provider.parent / AOTRITON_IMAGE_RELATIVE
    if not runtime.is_file():
        raise RuntimeError(
            "AOTriton runtime is missing beside the FMHA provider: "
            f"{AOTRITON_RUNTIME_SONAME}"
        )
    if sha256_file(runtime) != FROZEN_AOTRITON_RUNTIME_SHA256:
        raise RuntimeError("AOTriton runtime differs from the frozen artifact")
    if not image.is_file():
        raise RuntimeError(
            "AOTriton gfx1151 image is missing from the FMHA provider closure"
        )
    if sha256_file(image) != FROZEN_AOTRITON_IMAGE_SHA256:
        raise RuntimeError("AOTriton gfx1151 image differs from the frozen artifact")

    image_root = provider.parent / "aotriton.images"
    images = sorted(path.resolve() for path in image_root.rglob("*.aks2"))
    if images != [image.resolve()]:
        raise RuntimeError(
            "AOTriton FMHA provider closure must contain exactly the frozen image"
        )
    return AotritonClosure(provider=provider, runtime=runtime, image=image)


def require_aotriton_closure(fmha_provider: Path) -> AotritonClosure:
    """Resolve a qualifier input with a concise command-line failure."""

    try:
        return resolve_aotriton_closure(fmha_provider)
    except RuntimeError as error:
        raise SystemExit(f"invalid AOTriton qualification closure: {error}") from None
