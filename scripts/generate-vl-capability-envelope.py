#!/usr/bin/env python3
"""Generate the min/typical/max VL envelope from frozen reference evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aima_engine.vl_envelope import (  # noqa: E402
    build_envelope,
    validate_envelope,
)
from aima_engine.vl_reference import (  # noqa: E402
    ReferenceManifestError,
    atomic_json,
    file_component,
    load_json_object,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processor-probe", type=Path, required=True)
    parser.add_argument("--capability-manifest", type=Path, required=True)
    parser.add_argument("--generated-at")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    processor_path = args.processor_probe.resolve()
    capability_path = args.capability_manifest.resolve()
    module_path = ROOT / "aima_engine/vl_envelope.py"
    script_path = Path(__file__).resolve()
    bindings = {
        "processor_probe": file_component(
            processor_path,
            "benchmarks/results/vl-processor-capability-v0.1.0.json",
        ),
        "api_capability_manifest": file_component(
            capability_path,
            "benchmarks/results/vl-capability-manifest.json",
        ),
        "derivation_module": file_component(
            module_path, "aima_engine/vl_envelope.py"
        ),
        "generator": file_component(
            script_path, "scripts/generate-vl-capability-envelope.py"
        ),
    }
    generated_at = args.generated_at or datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    )
    try:
        result = build_envelope(
            load_json_object(processor_path),
            load_json_object(capability_path),
            bindings,
            generated_at,
        )
    except ReferenceManifestError as exc:
        parser.error(str(exc))
    errors = validate_envelope(result)
    if errors:
        parser.error("generated VL envelope is invalid:\n- " + "\n- ".join(errors))
    print(atomic_json(args.output.resolve(), result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
