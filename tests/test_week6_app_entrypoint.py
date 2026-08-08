"""Regression tests for launching the Week6 Streamlit script directly."""

from pathlib import Path
import subprocess
import sys
import unittest


class Week6AppEntrypointTest(unittest.TestCase):
    def test_app_resolves_project_imports_in_isolated_mode(self) -> None:
        root = Path(__file__).resolve().parents[1]
        app = root / "src" / "visualization" / "week6_app.py"
        probe = (
            "import runpy; "
            f"runpy.run_path({str(app)!r}, run_name='week6_app_import_probe')"
        )
        result = subprocess.run(
            [sys.executable, "-I", "-c", probe],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("No module named 'src'", result.stderr)


if __name__ == "__main__":
    unittest.main()
