#!/usr/bin/env python3
"""Verify a published release's evidence summaries and referenced raw reports."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aima_engine.release_evidence import DEFAULT_RELEASE, verify_release_evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--release", default=DEFAULT_RELEASE)
    parser.add_argument(
        "--require-archived-components",
        action="store_true",
        help="require raw components stored only in the public evidence archive",
    )
    args = parser.parse_args()
    errors = verify_release_evidence(
        args.root,
        release=args.release,
        require_archived_components=args.require_archived_components,
    )
    if errors:
        for error in errors:
            print(error)
        print(f"release evidence: FAIL ({len(errors)} error(s))")
        return 1
    print(f"release evidence {args.release}: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
