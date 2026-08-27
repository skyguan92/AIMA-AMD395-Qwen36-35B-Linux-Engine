from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]


def git_blob(commit: str, path: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{commit}:{path}"],
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"could not resolve evidence source blob {commit}:{path}"
        )
    return completed.stdout


def assert_component_at_commit(
    test: unittest.TestCase, component: dict, commit: str
) -> None:
    payload = git_blob(commit, component["path"])
    test.assertEqual(len(payload), component["bytes"], component["path"])
    test.assertEqual(
        hashlib.sha256(payload).hexdigest(),
        component["sha256"],
        component["path"],
    )


def assert_components_at_commit(
    test: unittest.TestCase, components: list[dict], commit: str
) -> None:
    for component in components:
        assert_component_at_commit(test, component, commit)
