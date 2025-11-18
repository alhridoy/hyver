"""Code execution rule verifier (simple sandbox)."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Mapping, Tuple


class PythonUnitTestRule:
    """Executes candidate code with inline unittest-based tests."""

    def __init__(self, *, timeout: float = 3.0, python_bin: str = "python3") -> None:
        self.timeout = timeout
        self.python_bin = python_bin

    def __call__(self, candidate: str, metadata: Mapping[str, object]) -> Tuple[bool, dict]:
        tests_code = metadata.get("tests_code")
        if not isinstance(tests_code, str) or not tests_code.strip():
            raise ValueError("PythonUnitTestRule requires 'tests_code' in metadata")

        script = self._compose_script(candidate, tests_code)
        with tempfile.TemporaryDirectory() as tmp_dir:
            script_path = Path(tmp_dir) / "candidate_tests.py"
            script_path.write_text(script)
            proc = subprocess.run(
                [self.python_bin, str(script_path)],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        passed = proc.returncode == 0
        return passed, {"stdout": proc.stdout, "stderr": proc.stderr, "returncode": proc.returncode}

    def _compose_script(self, candidate: str, tests_code: str) -> str:
        runner = """
import unittest, sys

def _run():
    loader = unittest.defaultTestLoader
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=0).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)

if __name__ == "__main__":
    _run()
"""
        return "\n".join([candidate.strip(), tests_code.strip(), runner])
