#!/usr/bin/env python3
"""Verify every published v1.3 evidence summary and referenced raw report."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aima_engine.release_evidence import verify_release_evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    errors = verify_release_evidence(args.root)
    if errors:
        for error in errors:
            print(error)
        print(f"release evidence: FAIL ({len(errors)} error(s))")
        return 1
    print("release evidence: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
