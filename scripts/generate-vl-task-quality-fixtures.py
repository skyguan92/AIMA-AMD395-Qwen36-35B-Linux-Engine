#!/usr/bin/env python3
"""Generate deterministic synthetic image/video task-quality fixtures."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aima_engine.vl_reference import (  # noqa: E402
    atomic_json,
    file_component,
    load_json_object,
    seal_manifest,
    sha256_file,
)
from aima_engine.vl_task_quality import (  # noqa: E402
    FIXTURE_SCHEMA,
    TASK_CASES,
    validate_fixture_manifest,
)


DEFAULT_OUTPUT = ROOT / "benchmarks/fixtures/vl-task-quality-v0.1.0"
IMAGE_SIZE = (512, 384)
VIDEO_SIZE = (384, 288)
VIDEO_FRAMES = 16
VIDEO_FPS = 4

COLORS = {
    "white": (255, 255, 255),
    "black": (8, 8, 8),
    "red": (220, 32, 32),
    "green": (24, 170, 70),
    "blue": (35, 90, 220),
    "yellow": (245, 205, 35),
    "orange": (238, 126, 28),
    "purple": (130, 55, 190),
}

PIXEL_FONT = {
    " ": ("00000",) * 7,
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("11111", "00100", "00100", "00100", "00100", "00100", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "5": ("11111", "10000", "10000", "11110", "00001", "00001", "11110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00001", "11110"),
}


def pillow_modules():
    try:
        from PIL import Image, ImageDraw
    except ImportError as error:
        raise RuntimeError("Pillow is required to generate task fixtures") from error
    return Image, ImageDraw


def new_canvas(size: tuple[int, int]):
    Image, ImageDraw = pillow_modules()
    image = Image.new("RGB", size, COLORS["white"])
    return image, ImageDraw.Draw(image)


def draw_circle(draw: Any, center: tuple[int, int], radius: int, color: str) -> None:
    x, y = center
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=COLORS[color])


def draw_square(draw: Any, center: tuple[int, int], half: int, color: str) -> None:
    x, y = center
    draw.rectangle((x - half, y - half, x + half, y + half), fill=COLORS[color])


def draw_triangle(draw: Any, center: tuple[int, int], radius: int, color: str) -> None:
    x, y = center
    draw.polygon(
        ((x, y - radius), (x - radius, y + radius), (x + radius, y + radius)),
        fill=COLORS[color],
    )


def arrow_points(size: tuple[int, int], direction: str) -> list[tuple[int, int]]:
    width, height = size
    right = [
        (width // 6, height * 2 // 5),
        (width * 3 // 5, height * 2 // 5),
        (width * 3 // 5, height // 4),
        (width * 5 // 6, height // 2),
        (width * 3 // 5, height * 3 // 4),
        (width * 3 // 5, height * 3 // 5),
        (width // 6, height * 3 // 5),
    ]
    if direction == "right":
        return right
    if direction == "left":
        return [(width - x, y) for x, y in right]
    if direction == "up":
        center_x = width // 2
        center_y = height // 2
        return [
            (center_x + y - center_y, center_y - x + center_x)
            for x, y in right
        ]
    raise ValueError(f"unsupported arrow direction: {direction}")


def draw_pixel_text(
    draw: Any,
    text: str,
    *,
    origin: tuple[int, int],
    scale: int,
) -> None:
    cursor_x, cursor_y = origin
    for character in text:
        glyph = PIXEL_FONT[character]
        for row, bits in enumerate(glyph):
            for column, bit in enumerate(bits):
                if bit != "1":
                    continue
                left = cursor_x + column * scale
                top = cursor_y + row * scale
                draw.rectangle(
                    (left, top, left + scale - 1, top + scale - 1),
                    fill=COLORS["black"],
                )
        cursor_x += 6 * scale


def pixel_text_dimensions(text: str, scale: int) -> tuple[int, int]:
    if not text or scale <= 0:
        raise ValueError("pixel text and scale must be nonempty and positive")
    return (len(text) * 6 - 1) * scale, 7 * scale


def save_png(path: Path, render: Callable[[Any], None]) -> None:
    image, draw = new_canvas(IMAGE_SIZE)
    render(draw)
    image.save(path, format="PNG", compress_level=9)


def generate_images(root: Path) -> list[dict[str, Any]]:
    specs: list[tuple[str, Callable[[Any], None]]] = [
        (
            "image-central-red-circle.png",
            lambda draw: draw_circle(draw, (256, 192), 120, "red"),
        ),
        (
            "image-spatial-shapes.png",
            lambda draw: (
                draw_square(draw, (150, 192), 78, "blue"),
                draw_triangle(draw, (365, 192), 88, "yellow"),
            ),
        ),
        (
            "image-count-shapes.png",
            lambda draw: (
                draw_circle(draw, (110, 105), 46, "green"),
                draw_circle(draw, (256, 105), 46, "green"),
                draw_circle(draw, (402, 105), 46, "green"),
                draw_square(draw, (175, 275), 48, "red"),
                draw_square(draw, (337, 275), 48, "red"),
            ),
        ),
        (
            "image-text-aima-395.png",
            lambda draw: draw_pixel_text(
                draw, "AIMA 395", origin=(42, 157), scale=10
            ),
        ),
        (
            "image-two-color-bands.png",
            lambda draw: (
                draw.rectangle((0, 0, 511, 191), fill=COLORS["orange"]),
                draw.rectangle((0, 192, 511, 383), fill=COLORS["purple"]),
            ),
        ),
        (
            "image-arrow-right.png",
            lambda draw: draw.polygon(
                arrow_points(IMAGE_SIZE, "right"), fill=COLORS["black"]
            ),
        ),
    ]
    records: list[dict[str, Any]] = []
    for name, render in specs:
        path = root / name
        save_png(path, render)
        records.append(
            {
                "path": name,
                "modality": "image",
                "format": "png",
                "width": IMAGE_SIZE[0],
                "height": IMAGE_SIZE[1],
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def video_frame(case: str, index: int):
    image, draw = new_canvas(VIDEO_SIZE)
    width, height = VIDEO_SIZE
    if case == "video-red-circle-right.mp4":
        x = 55 + index * (width - 110) // (VIDEO_FRAMES - 1)
        draw_circle(draw, (x, height // 2), 42, "red")
    elif case == "video-blue-square-down.mp4":
        y = 52 + index * (height - 104) // (VIDEO_FRAMES - 1)
        draw_square(draw, (width // 2, y), 40, "blue")
    elif case == "video-circle-blue-green.mp4":
        draw_circle(
            draw,
            (width // 2, height // 2),
            62,
            "blue" if index < VIDEO_FRAMES // 2 else "green",
        )
    elif case == "video-count-one-to-four.mp4":
        count = min(4, index // 4 + 1)
        positions = ((90, 85), (294, 85), (90, 205), (294, 205))
        for position in positions[:count]:
            draw_circle(draw, position, 36, "yellow")
    elif case == "video-shape-order.mp4":
        if index < 5:
            draw_circle(draw, (width // 2, height // 2), 62, "red")
        elif index < 10:
            draw_square(draw, (width // 2, height // 2), 58, "blue")
        else:
            draw_triangle(draw, (width // 2, height // 2), 68, "green")
    elif case == "video-arrow-left-up.mp4":
        direction = "left" if index < VIDEO_FRAMES // 2 else "up"
        draw.polygon(arrow_points(VIDEO_SIZE, direction), fill=COLORS["black"])
    else:
        raise ValueError(f"unknown video fixture: {case}")
    return image


def encode_video(ffmpeg: str, output: Path) -> None:
    frames = b"".join(
        video_frame(output.name, index).tobytes() for index in range(VIDEO_FRAMES)
    )
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{VIDEO_SIZE[0]}x{VIDEO_SIZE[1]}",
        "-framerate",
        str(VIDEO_FPS),
        "-i",
        "pipe:0",
        "-frames:v",
        str(VIDEO_FRAMES),
        "-an",
        "-map_metadata",
        "-1",
        "-threads",
        "1",
        "-c:v",
        "libx264",
        "-preset",
        "veryslow",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-fflags",
        "+bitexact",
        "-flags:v",
        "+bitexact",
        "-movflags",
        "+faststart",
        "-y",
        str(output),
    ]
    completed = subprocess.run(
        command,
        input=frames,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {output.name}: "
            + completed.stderr.decode("utf-8", errors="replace")
        )


def generate_videos(root: Path, ffmpeg: str) -> list[dict[str, Any]]:
    names = (
        "video-red-circle-right.mp4",
        "video-blue-square-down.mp4",
        "video-circle-blue-green.mp4",
        "video-count-one-to-four.mp4",
        "video-shape-order.mp4",
        "video-arrow-left-up.mp4",
    )
    records: list[dict[str, Any]] = []
    for name in names:
        path = root / name
        encode_video(ffmpeg, path)
        records.append(
            {
                "path": name,
                "modality": "video",
                "format": "mp4",
                "width": VIDEO_SIZE[0],
                "height": VIDEO_SIZE[1],
                "frame_count": VIDEO_FRAMES,
                "fps": VIDEO_FPS,
                "duration_seconds": VIDEO_FRAMES / VIDEO_FPS,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def expected_fixture_names() -> tuple[str, ...]:
    return tuple(case["fixture"] for case in TASK_CASES)


def validate_existing(root: Path) -> list[str]:
    manifest_path = root / "fixtures-manifest.json"
    if not manifest_path.is_file():
        return ["task-quality fixture manifest is missing"]
    payload = load_json_object(manifest_path)
    errors = validate_fixture_manifest(payload, root)
    sidecar_path = manifest_path.with_name(manifest_path.name + ".sha256")
    expected_sidecar = f"{sha256_file(manifest_path)}  {manifest_path.name}\n"
    if (
        not sidecar_path.is_file()
        or sidecar_path.read_text(encoding="utf-8") != expected_sidecar
    ):
        errors.append("task-quality fixture manifest sidecar changed")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.check:
        errors = validate_existing(output)
        if errors:
            raise SystemExit("invalid task-quality fixtures:\n- " + "\n- ".join(errors))
        print("VL task-quality fixtures: PASS")
        return 0
    if output.exists():
        raise SystemExit("task-quality fixture output must not exist")
    ffmpeg = shutil.which(args.ffmpeg)
    if ffmpeg is None:
        raise SystemExit("ffmpeg is required to generate video task fixtures")
    output.mkdir(parents=True)
    fixtures = generate_images(output) + generate_videos(output, ffmpeg)
    ordered = {record["path"]: record for record in fixtures}
    fixtures = [ordered[name] for name in expected_fixture_names()]
    version = subprocess.run(
        [ffmpeg, "-version"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()[0]
    manifest = seal_manifest(
        {
            "schema": FIXTURE_SCHEMA,
            "complete": True,
            "generator": file_component(
                Path(__file__).resolve(),
                "scripts/generate-vl-task-quality-fixtures.py",
            ),
            "encoder": {
                "ffmpeg_version": version,
                "video_codec": "libx264",
                "pixel_format": "yuv420p",
            },
            "fixtures": fixtures,
        }
    )
    digest = atomic_json(output / "fixtures-manifest.json", manifest)
    print(
        json.dumps(
            {"fixtures": len(fixtures), "manifest_sha256": digest},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
