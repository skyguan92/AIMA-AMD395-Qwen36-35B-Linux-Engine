#!/usr/bin/env python3
"""Generate deterministic image/video fixtures for VL capability discovery."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

import numpy as np
from PIL import Image


SCHEMA = "aima-amd395-qwen36/vl-capability-fixtures/v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pattern(height: int, width: int, *, offset: int = 0) -> np.ndarray:
    y, x = np.indices((height, width), dtype=np.uint32)
    return np.stack(
        (
            (x * 3 + y * 5 + offset) % 256,
            (x * 7 + y * 11 + offset * 3) % 256,
            (x * 13 + y * 17 + offset * 5) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)


def record(path: Path, root: Path, **metadata: Any) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        **metadata,
    }


def save_images(root: Path) -> list[dict[str, Any]]:
    fixtures: list[dict[str, Any]] = []

    rgb = pattern(256, 256, offset=1)
    png = root / "image-rgb-256.png"
    Image.fromarray(rgb, "RGB").save(png, format="PNG", compress_level=9)
    fixtures.append(
        record(png, root, modality="image", format="png", width=256, height=256)
    )

    jpeg = root / "image-landscape-512x192.jpg"
    Image.fromarray(pattern(192, 512, offset=2), "RGB").save(
        jpeg,
        format="JPEG",
        quality=91,
        subsampling=0,
        optimize=False,
        progressive=False,
    )
    fixtures.append(
        record(jpeg, root, modality="image", format="jpeg", width=512, height=192)
    )

    rgba_source = pattern(320, 160, offset=3)
    alpha = np.linspace(0, 255, 160, dtype=np.uint8)[None, :].repeat(320, axis=0)
    rgba = np.concatenate((rgba_source, alpha[..., None]), axis=-1)
    transparent = root / "image-transparent-160x320.png"
    Image.fromarray(rgba, "RGBA").save(
        transparent, format="PNG", compress_level=9
    )
    fixtures.append(
        record(
            transparent,
            root,
            modality="image",
            format="png",
            mode="RGBA",
            width=160,
            height=320,
        )
    )

    webp = root / "image-portrait-192x512.webp"
    Image.fromarray(pattern(512, 192, offset=4), "RGB").save(
        webp, format="WEBP", lossless=True, method=6, exact=True
    )
    fixtures.append(
        record(webp, root, modality="image", format="webp", width=192, height=512)
    )

    invalid_aspect = root / "image-invalid-aspect-201x1.png"
    Image.fromarray(pattern(1, 201, offset=5), "RGB").save(
        invalid_aspect, format="PNG", compress_level=9
    )
    fixtures.append(
        record(
            invalid_aspect,
            root,
            modality="image",
            format="png",
            width=201,
            height=1,
            expected="processor-reject-aspect-ratio-over-200",
        )
    )
    return fixtures


def video_frames(
    *, frame_count: int, height: int, width: int, offset: int
) -> bytes:
    frames = []
    for frame_index in range(frame_count):
        frame = pattern(height, width, offset=offset + frame_index * 19)
        marker_width = min(width, 8 + frame_index * 3)
        frame[:8, :marker_width] = (255, frame_index * 17 % 256, 0)
        frames.append(frame)
    return np.stack(frames).tobytes(order="C")


def run_ffmpeg(
    ffmpeg: str,
    output: Path,
    *,
    frame_count: int,
    width: int,
    height: int,
    fps: int,
    offset: int,
    encoder_args: list[str],
) -> None:
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
        str(frame_count),
        "-an",
        "-map_metadata",
        "-1",
        "-threads",
        "1",
        *encoder_args,
        "-y",
        str(output),
    ]
    completed = subprocess.run(
        command,
        input=video_frames(
            frame_count=frame_count,
            height=height,
            width=width,
            offset=offset,
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {output.name}: "
            + completed.stderr.decode("utf-8", errors="replace")
        )


def run_opencv(
    output: Path,
    *,
    frame_count: int,
    width: int,
    height: int,
    fps: int,
    offset: int,
    fourcc: str,
) -> None:
    import cv2

    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*fourcc),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not create {output.name} with {fourcc}")
    try:
        raw = video_frames(
            frame_count=frame_count,
            height=height,
            width=width,
            offset=offset,
        )
        frames = np.frombuffer(raw, dtype=np.uint8).reshape(
            frame_count, height, width, 3
        )
        for frame in frames:
            writer.write(frame[..., ::-1])
    finally:
        writer.release()


def save_videos(root: Path) -> list[dict[str, Any]]:
    ffmpeg = shutil.which("ffmpeg")

    specs = [
        {
            "name": "video-8f-4fps-128.mp4",
            "frame_count": 8,
            "width": 128,
            "height": 128,
            "fps": 4,
            "offset": 31,
            "format": "mp4",
            "opencv_fourcc": "mp4v",
            "encoder_args": [
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
            ],
        },
        {
            "name": "video-12f-6fps-192x128.avi",
            "frame_count": 12,
            "width": 192,
            "height": 128,
            "fps": 6,
            "offset": 73,
            "format": "avi",
            "opencv_fourcc": "MJPG",
            "encoder_args": ["-c:v", "mjpeg", "-q:v", "3", "-pix_fmt", "yuvj420p"],
        },
    ]
    fixtures: list[dict[str, Any]] = []
    for spec in specs:
        output = root / str(spec["name"])
        common = {
            "frame_count": int(spec["frame_count"]),
            "width": int(spec["width"]),
            "height": int(spec["height"]),
            "fps": int(spec["fps"]),
            "offset": int(spec["offset"]),
        }
        if ffmpeg is not None:
            run_ffmpeg(
                ffmpeg,
                output,
                **common,
                encoder_args=list(spec["encoder_args"]),
            )
            generator = "ffmpeg"
        else:
            run_opencv(
                output,
                **common,
                fourcc=str(spec["opencv_fourcc"]),
            )
            generator = "opencv-video-writer"
        fixtures.append(
            record(
                output,
                root,
                modality="video",
                format=spec["format"],
                width=spec["width"],
                height=spec["height"],
                frame_count=spec["frame_count"],
                fps=spec["fps"],
                duration_seconds=spec["frame_count"] / spec["fps"],
                generator=generator,
            )
        )
    return fixtures


def write_manifest(root: Path, fixtures: list[dict[str, Any]]) -> None:
    corrupt_image = root / "corrupt-image.png"
    corrupt_image.write_bytes(b"not-a-png\x00aima-vl-fixture\n")
    fixtures.append(
        record(
            corrupt_image,
            root,
            modality="image",
            format="corrupt",
            expected="decode-reject",
        )
    )
    corrupt_video = root / "corrupt-video.mp4"
    corrupt_video.write_bytes(b"not-an-mp4\x00aima-vl-fixture\n")
    fixtures.append(
        record(
            corrupt_video,
            root,
            modality="video",
            format="corrupt",
            expected="decode-reject",
        )
    )

    payload = {
        "schema": SCHEMA,
        "generator": "scripts/generate-vl-capability-fixtures.py",
        "fixtures": sorted(fixtures, key=lambda item: item["path"]),
    }
    manifest = root / "fixtures-manifest.json"
    manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    sidecar = root / "fixtures-manifest.json.sha256"
    sidecar.write_text(
        f"{sha256_file(manifest)}  {manifest.name}\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    fixtures = save_images(output)
    fixtures.extend(save_videos(output))
    write_manifest(output, fixtures)
    print(output / "fixtures-manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
