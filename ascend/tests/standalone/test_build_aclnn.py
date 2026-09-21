# SPDX-License-Identifier: Apache-2.0
"""Exercise ACLNN staging with shell fixtures only; no CANN, torch or compiler.

Run directly with ``python -B ascend/tests/standalone/test_build_aclnn.py -v``.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

BUILD_ACLNN = Path(__file__).resolve().parents[2] / "csrc/build_aclnn.sh"


class BuildAclnnTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="aclnn-script-test-")
        self.addCleanup(temporary.cleanup)
        # A private source copy starts with csrc only; spaces check path quoting.
        self.root = Path(temporary.name) / "aclnn source"
        self.csrc = self.root / "csrc"
        (self.csrc / "third_party/catlass/include").mkdir(parents=True)
        self.install_dir = self.root / "vllm_ascend/_cann_ops_custom"
        self.installer_called = self.csrc / "installer-called"
        self.build_script = self.csrc / "build.sh"
        self.installer = self.csrc / "installer-fixture.sh"
        self.build_script.write_text(
            """#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 4 && "$1" == "-n" && "$3" == "-c" && "$4" == "ascend910b" ]]
mkdir build output
cp installer-fixture.sh output/CANN-custom_ops-test.run
""",
            encoding="utf-8",
        )
        self.installer.write_text(
            """#!/usr/bin/env bash
set -euo pipefail
: > installer-called
[[ $# -eq 1 && "$1" == --install-path=* ]]
install_dir="${1#--install-path=}"
if [[ ! -d "$install_dir" ]]; then
    echo "Install directory must exist before invoking the installer" >&2
    exit 23
fi
printf '%s\\n' "$install_dir" > "$install_dir/installed-path.txt"
""",
            encoding="utf-8",
        )

    def run_script(self):
        env = os.environ.copy()
        # Do not run a caller's shell startup hooks during a standalone test.
        env.pop("BASH_ENV", None)
        env.pop("ENV", None)
        return subprocess.run(
            ["bash", str(BUILD_ACLNN), str(self.root), "ascend910b3"],
            cwd=self.root,
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

    def assert_installed(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(
            (self.install_dir / "installed-path.txt").read_text(encoding="utf-8"),
            str(self.install_dir) + "\n",
        )

    def test_creates_missing_parent_and_install_directory(self):
        self.assertFalse(self.install_dir.parent.exists())
        self.assert_installed(self.run_script())

    def test_creates_install_directory_under_existing_parent(self):
        self.install_dir.parent.mkdir()
        self.assert_installed(self.run_script())

    def test_preserves_existing_install_directory_contents(self):
        self.install_dir.mkdir(parents=True)
        sentinel = self.install_dir / "existing.txt"
        sentinel.write_text("keep this fixture\n", encoding="utf-8")
        self.assert_installed(self.run_script())
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep this fixture\n")

    def test_directory_creation_failure_prevents_installer_execution(self):
        self.install_dir.parent.write_text("not a directory\n", encoding="utf-8")
        result = self.run_script()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.installer_called.exists())
        self.assertEqual(
            self.install_dir.parent.read_text(encoding="utf-8"), "not a directory\n"
        )

    def test_propagates_installer_failure(self):
        self.install_dir.mkdir(parents=True)
        self.installer.write_text(
            "#!/usr/bin/env bash\n: > installer-called\nexit 17\n",
            encoding="utf-8",
        )
        result = self.run_script()
        self.assertEqual(result.returncode, 17)
        self.assertTrue(self.installer_called.exists())

    def test_build_failure_prevents_installation(self):
        self.build_script.write_text("exit 19\n", encoding="utf-8")
        result = self.run_script()
        self.assertEqual(result.returncode, 19)
        self.assertFalse(self.install_dir.exists())
        self.assertFalse(self.installer_called.exists())

    def test_stale_build_is_rejected_without_removing_it(self):
        stale = self.csrc / "build"
        stale.mkdir()
        sentinel = stale / "existing.txt"
        sentinel.write_text("keep this fixture\n", encoding="utf-8")
        result = self.run_script()
        self.assertEqual(result.returncode, 2)
        self.assertIn("Use a fresh ACLNN build source copy", result.stderr)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep this fixture\n")
        self.assertFalse(self.install_dir.exists())
        self.assertFalse(self.installer_called.exists())


if __name__ == "__main__":
    unittest.main()
