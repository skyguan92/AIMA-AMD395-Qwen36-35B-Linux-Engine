#!/usr/bin/env python3
"""Generate deterministic media for native VL execution-envelope qualification."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aima_engine.vl_reference import (  # noqa: E402
    atomic_json,
    file_component,
    seal_manifest,
)


SCHEMA = "aima-amd395-qwen36/vl-envelope-fixtures/v1"


IMAGE_SPECS = (
    ("image-minimum-1x1.png", 1, 1, "image.resize.minimum_source"),
    ("image-portrait-256x1024.png", 256, 1024, "image.resize.portrait"),
    ("image-landscape-1024x256.png", 1024, 256, "image.resize.landscape"),
    ("image-maximum-4096x4096.png", 4096, 4096, "image.resize.maximum_pixels"),
    (
        "image-above-maximum-8192x8192.png",
        8192,
        8192,
        "image.resize.above_maximum_pixels",
    ),
    (
        "image-rejected-aspect-201x1.png",
        201,
        1,
        "image.resize.aspect_ratio_over_200",
    ),
)


VIDEO_SPECS = (
    {
        "path": "video-minimum-2f-2fps-32x32.mp4",
        "width": 32,
        "height": 32,
        "frames": 2,
        "fps": 2,
        "boundary": "video.resize.temporal_factor+spatial_factor",
    },
    {
        "path": "video-typical-4f-2fps-256x256.mp4",
        "width": 256,
        "height": 256,
        "frames": 4,
        "fps": 2,
        "boundary": "video.resize.typical",
    },
    {
        "path": "video-maximum-2f-2fps-4096x3072.mp4",
        "width": 4096,
        "height": 3072,
        "frames": 2,
        "fps": 2,
        "boundary": "video.resize.maximum_feature_shape",
    },
    {
        "path": "video-rejected-temporal-1f-2fps-32x32.mp4",
        "width": 32,
        "height": 32,
        "frames": 1,
        "fps": 2,
        "boundary": "video.resize.below_temporal_factor",
    },
    {
        "path": "video-rejected-spatial-2f-2fps-32x31.avi",
        "width": 32,
        "height": 31,
        "frames": 2,
        "fps": 2,
        "boundary": "video.resize.below_spatial_factor",
        "container": "avi",
        "codec": "mjpeg",
        "pixel_format": "yuvj444p",
    },
    {
        "path": "video-rejected-aspect-2f-2fps-6432x32.mp4",
        "width": 6432,
        "height": 32,
        "frames": 2,
        "fps": 2,
        "boundary": "video.resize.aspect_ratio_over_200",
    },
    {
        "path": "video-sampling-minimum-48f-24fps-256x256.mp4",
        "width": 256,
        "height": 256,
        "frames": 48,
        "fps": 24,
        "boundary": "video.sampling.minimum_frames",
    },
    {
        "path": "video-sampling-typical-240f-24fps-256x256.mp4",
        "width": 256,
        "height": 256,
        "frames": 240,
        "fps": 24,
        "boundary": "video.sampling.typical_fps",
    },
    {
        "path": "video-sampling-maximum-9216f-24fps-256x256.mp4",
        "width": 256,
        "height": 256,
        "frames": 9216,
        "fps": 24,
        "boundary": "video.sampling.maximum_frames",
    },
    {
        "path": "video-sampling-above-maximum-18432f-24fps-256x256.mp4",
        "width": 256,
        "height": 256,
        "frames": 18432,
        "fps": 24,
        "boundary": "video.sampling.above_maximum_frames",
    },
)


def image_fixture(
    root: Path, filename: str, width: int, height: int, boundary: str,
) -> dict[str, Any]:
    color = (
        (width * 3 + height * 5) % 251,
        (width * 7 + height * 11) % 251,
        (width * 13 + height * 17) % 251,
    )
    image = Image.new("RGB", (width, height), color)
    draw = ImageDraw.Draw(image)
    marker_width = min(width, 64)
    marker_height = min(height, 64)
    if marker_width > 1 and marker_height > 1:
        draw.rectangle(
            (0, 0, marker_width - 1, marker_height - 1),
            fill=(255, 127, 31),
        )
        draw.line(
            (0, marker_height - 1, marker_width - 1, 0),
            fill=(0, 63, 255),
            width=max(1, min(marker_width, marker_height) // 16),
        )
    path = root / filename
    image.save(path, format="PNG", compress_level=9, optimize=False)
    image.close()
    return {
        **file_component(path, filename),
        "fixture_id": filename,
        "modality": "image",
        "format": "png",
        "width": width,
        "height": height,
        "boundary": boundary,
    }


def video_frame(width: int, height: int, offset: int) -> np.ndarray:
    frame = np.empty((height, width, 3), dtype=np.uint8)
    frame[:, :, 0] = (width * 3 + height * 5 + offset) % 251
    frame[:, :, 1] = (width * 7 + height * 11 + offset * 3) % 251
    frame[:, :, 2] = (width * 13 + height * 17 + offset * 5) % 251
    return frame


def write_video(ffmpeg: str, path: Path, spec: dict[str, Any]) -> list[str]:
    width = int(spec["width"])
    height = int(spec["height"])
    frames = int(spec["frames"])
    fps = int(spec["fps"])
    container = str(spec.get("container", "mp4"))
    codec = str(spec.get("codec", "mpeg4"))
    pixel_format = str(spec.get("pixel_format", "yuv420p"))
    codec_args = (
        ["-c:v", "mjpeg", "-q:v", "3", "-pix_fmt", pixel_format]
        if codec == "mjpeg"
        else ["-c:v", "mpeg4", "-q:v", "2", "-pix_fmt", pixel_format]
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
        f"{width}x{height}",
        "-framerate",
        str(fps),
        "-i",
        "pipe:0",
        "-frames:v",
        str(frames),
        "-an",
        "-map_metadata",
        "-1",
        "-threads",
        "1",
        "-fflags",
        "+bitexact",
        "-flags:v",
        "+bitexact",
        *codec_args,
    ]
    if container == "mp4":
        command.extend(["-movflags", "+faststart"])
    command.extend(["-y", str(path)])
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.stdin is None:
        raise RuntimeError("ffmpeg input pipe is unavailable")
    frame = video_frame(width, height, frames % 251)
    marker_height = min(height, 16)
    marker_width = min(width, 16)
    try:
        for frame_index in range(frames):
            frame[:marker_height, :marker_width] = (
                frame_index & 255,
                (frame_index >> 8) & 255,
                (frame_index >> 16) & 255,
            )
            process.stdin.write(frame.tobytes(order="C"))
        process.stdin.close()
        stdout = process.stdout.read() if process.stdout is not None else b""
        stderr = process.stderr.read() if process.stderr is not None else b""
        return_code = process.wait()
    except Exception:
        process.kill()
        process.wait()
        raise
    if return_code != 0:
        raise RuntimeError(
            f"ffmpeg failed for {path.name}: "
            + stderr.decode("utf-8", errors="replace")
            + stdout.decode("utf-8", errors="replace")
        )
    return ["${AIMA_FIXTURE_FFMPEG}", *command[1:-1], path.name]


def video_fixture(
    root: Path, ffmpeg: str, spec: dict[str, Any],
) -> dict[str, Any]:
    filename = str(spec["path"])
    path = root / filename
    command = write_video(ffmpeg, path, spec)
    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
    try:
        metadata = next(reader)
    finally:
        reader.close()
    expected_size = (int(spec["width"]), int(spec["height"]))
    actual_size = tuple(metadata.get("size") or ())
    actual_fps = float(metadata.get("fps") or 0.0)
    actual_duration = float(metadata.get("duration") or 0.0)
    expected_duration = int(spec["frames"]) / int(spec["fps"])
    inferred_frames = round(actual_duration * actual_fps)
    if (
        actual_size != expected_size
        or actual_fps != float(spec["fps"])
        or actual_duration != expected_duration
        or inferred_frames != int(spec["frames"])
    ):
        raise RuntimeError(
            f"generated video metadata drifted for {filename}: {metadata!r}"
        )
    return {
        **file_component(path, filename),
        "fixture_id": filename,
        "modality": "video",
        "format": str(spec.get("container", "mp4")),
        "codec": str(spec.get("codec", "mpeg4")),
        "pixel_format": str(spec.get("pixel_format", "yuv420p")),
        "width": int(spec["width"]),
        "height": int(spec["height"]),
        "frame_count": int(spec["frames"]),
        "fps": int(spec["fps"]),
        "duration_seconds": expected_duration,
        "boundary": str(spec["boundary"]),
        "generator_command": command,
    }


def package_version(name: str) -> str:
    return importlib.metadata.version(name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generated-at")
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("fixture output directory must be absent or empty")
    output.mkdir(parents=True, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    version_line = subprocess.run(
        [ffmpeg, "-version"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()[0]

    fixtures = [
        image_fixture(output, filename, width, height, boundary)
        for filename, width, height, boundary in IMAGE_SPECS
    ]
    for spec in VIDEO_SPECS:
        print(f"GENERATE {spec['path']}", flush=True)
        fixtures.append(video_fixture(output, ffmpeg, dict(spec)))

    generated_at = args.generated_at or datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    )
    manifest = seal_manifest(
        {
            "schema": SCHEMA,
            "complete": True,
            "generated_at": generated_at,
            "generator": file_component(
                Path(__file__).resolve(),
                "scripts/generate-vl-envelope-fixtures.py",
            ),
            "runtime": {
                "python": sys.version.split()[0],
                "numpy": package_version("numpy"),
                "pillow": package_version("pillow"),
                "imageio_ffmpeg": package_version("imageio-ffmpeg"),
                "ffmpeg": version_line,
            },
            "fixtures": sorted(fixtures, key=lambda item: item["fixture_id"]),
        }
    )
    digest = atomic_json(output / "fixtures-manifest.json", manifest)
    print(
        json.dumps(
            {
                "fixtures": len(fixtures),
                "output": str(output),
                "sha256": digest,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
