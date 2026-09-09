from pathlib import Path
import json
import tempfile
import unittest

from aima_engine.qualification_runtime import bind_runtime, expected_binding, require_runtime_binding, sha256


class QualificationRuntimeTest(unittest.TestCase):
    def fixture(self, root):
        capsule = root / "capsule"
        records = []
        for name in ("bin/aima-engine", "lib/loader", "libexec/aima-engine.real"):
            path = capsule / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(name.encode())
            path.chmod(0o755)
            records.append({"path": name, "type": "file", "bytes": path.stat().st_size, "sha256": sha256(path)})
        (capsule / "lib/alias").symlink_to("loader")
        records.append({"path": "lib/alias", "type": "symlink", "target": "loader"})
        manifest = {"complete": True, "files": records}
        (capsule / "manifest.json").write_text(json.dumps(manifest))
        baseline = root / "baseline.json"
        baseline.write_text(json.dumps(manifest))
        engine = root / "engine"
        engine.write_bytes((capsule / "libexec/aima-engine.real").read_bytes())
        return capsule, baseline, engine

    def test_binds_launcher_and_payload_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            capsule, baseline, engine = self.fixture(Path(directory))
            launcher, binding = bind_runtime(engine, capsule, baseline)
            self.assertEqual(launcher, (capsule / "bin/aima-engine").resolve())
            self.assertEqual(binding, expected_binding(sha256(engine), baseline))
            require_runtime_binding(binding, sha256(engine), baseline)
            self.assertEqual(bind_runtime(engine, None), (engine, None))
            for changed in (None, {}, dict(binding, engine_sha256="0" * 64),
                            dict(binding, execution="host-loader")):
                with self.assertRaises(ValueError):
                    require_runtime_binding(changed, sha256(engine), baseline)

    def test_rejects_modified_unlisted_and_escaping_runtime(self):
        for mutation in ("dependency", "unlisted", "escape", "engine", "manifest", "launcher"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                capsule, baseline, engine = self.fixture(Path(directory))
                if mutation == "dependency":
                    (capsule / "lib/loader").write_bytes(b"modified")
                elif mutation == "unlisted":
                    (capsule / "lib/extra.so").write_bytes(b"extra")
                elif mutation == "escape":
                    (capsule / "lib/alias").unlink()
                    (capsule / "lib/alias").symlink_to(engine)
                elif mutation == "engine":
                    engine.write_bytes(b"other engine")
                elif mutation == "launcher":
                    (capsule / "bin/aima-engine").chmod(0o644)
                else:
                    manifest = json.loads((capsule / "manifest.json").read_text())
                    manifest["files"].append(manifest["files"][0])
                    (capsule / "manifest.json").write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    bind_runtime(engine, capsule, baseline)
