from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
SCRIPT = SCRIPTS / "qualify-native-surfaces.py"
SPEC = importlib.util.spec_from_file_location("native_surfaces", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
surfaces = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(surfaces)


class NativeSurfacesQualificationTest(unittest.TestCase):
    def test_defaults_match_native_vl_goal(self) -> None:
        self.assertEqual(surfaces.STARTUP_CEILING_MS, 44_900.0)
        self.assertEqual(surfaces.MINIMUM_PREFIX_TTFT_SPEEDUP, 2637.0)
        self.assertEqual(surfaces.MINIMUM_PREFIX_DECODE_RETENTION, 1.0003)

    def test_public_paths_bind_external_candidate_and_output(self) -> None:
        engine = Path("/tmp/candidate/build/aima-engine-native")
        model = Path("/srv/private/model")
        output = Path("/tmp/evidence/surfaces")
        value = {
            "command": [str(engine), "--model-dir", str(model)],
            "report": str(output / "raw" / "run.json"),
        }
        self.assertEqual(
            surfaces.publicize(
                value,
                engine=engine,
                model_dir=model,
                output_dir=output,
            ),
            {
                "command": ["${AIMA_ENGINE}", "--model-dir", "${AIMA_MODEL_DIR}"],
                "report": "${AIMA_OUTPUT_DIR}/raw/run.json",
            },
        )


if __name__ == "__main__":
    unittest.main()
