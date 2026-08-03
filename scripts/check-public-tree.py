#!/usr/bin/env python3
"""Fail when tracked public files contain likely credentials or private host data."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aima_engine.public_hygiene import scan_public_tree


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    findings = scan_public_tree(root)
    if findings:
        for finding in findings:
            print(f"{finding.path}:{finding.line}: {finding.rule}")
        print(f"public tree hygiene: FAIL ({len(findings)} finding(s))")
        return 1
    print("public tree hygiene: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
