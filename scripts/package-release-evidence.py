#!/usr/bin/env python3
"""Build a deterministic, checksum-bound archive of public release evidence."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aima_engine.public_hygiene import scan_bytes
from aima_engine.release_evidence import (
    DEFAULT_RELEASE,
    evidence_paths,
    verify_release_evidence,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--release", default=DEFAULT_RELEASE)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument("--source-date-epoch", default="1784678400")
    args = parser.parse_args()
    root = args.root.resolve()
    requested_output = args.output or Path(
        f"dist/aima-engine-v{args.release}-public-evidence.tar.zst"
    )
    output = (
        requested_output
        if requested_output.is_absolute()
        else root / requested_output
    )
    checksum = output.with_name(output.name + ".sha256")
    if output.exists() or checksum.exists():
        raise SystemExit(f"refusing to replace existing evidence asset: {output}")

    errors = verify_release_evidence(root, release=args.release)
    if errors:
        raise SystemExit("\n".join(errors))
    inputs = evidence_paths(root, release=args.release)
    for entry in inputs:
        candidates = entry.rglob("*") if entry.is_dir() else (entry,)
        for path in candidates:
            if path.is_file():
                findings = scan_bytes(str(path.relative_to(root)), path.read_bytes())
                if findings:
                    rules = ", ".join(sorted({finding.rule for finding in findings}))
                    raise SystemExit(f"public evidence hygiene failed: {path}: {rules}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    relative_inputs = sorted({str(path.relative_to(root)) for path in inputs})
    command = [
        "tar",
        "--sort=name",
        f"--mtime=@{args.source_date_epoch}",
        "--owner=0",
        "--group=0",
        "--numeric-owner",
        "--format=posix",
        "--pax-option=delete=atime,delete=ctime",
        "--zstd",
        "-cf",
        str(temporary),
        "-C",
        str(root),
        *relative_inputs,
    ]
    try:
        subprocess.run(command, check=True)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    checksum.write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    print(output)
    print(checksum)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
