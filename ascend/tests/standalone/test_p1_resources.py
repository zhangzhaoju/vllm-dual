# SPDX-License-Identifier: Apache-2.0
"""Execute resource registration against regular and strict-editable fixtures."""

import ast
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[2] / "vllm_ascend/platform.py"


class ResourcePaths(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="p1-resource-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def register(self, linked: bool) -> tuple[dict, str]:
        """Load only the real registration method, with no framework/device imports."""
        source = self.root / "source/platform.py"
        source.parent.mkdir()
        source.write_text("# Python source fixture\n")
        installed = self.root / "link-tree/platform.py" if linked else source
        if linked:
            installed.parent.mkdir()
            installed.symlink_to(source)
        vendor = installed.parent / "_cann_ops_custom/vendors/vllm-ascend"
        vendor.mkdir(parents=True)
        tree = ast.parse(SOURCE.read_text())
        cls = next(
            item
            for item in tree.body
            if isinstance(item, ast.ClassDef) and item.name == "NPUPlatform"
        )
        method = next(
            item
            for item in cls.body
            if isinstance(item, ast.FunctionDef) and item.name == "import_kernels"
        )
        method.decorator_list = []
        namespace = {
            "os": os,
            "__file__": str(installed),
            "_CUSTOM_OP_REGISTERED": False,
        }
        exec(
            compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), "exec"),
            namespace,
        )
        namespace["import_kernels"](None)
        return namespace, str(vendor)

    def test_regular_wheel_resource_path_is_preserved(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            _, vendor = self.register(False)
            self.assertEqual(os.environ["ASCEND_CUSTOM_OPP_PATH"], vendor)

    def test_strict_editable_does_not_resolve_back_to_raw_source(self) -> None:
        with patch.dict(
            os.environ, {"ASCEND_CUSTOM_OPP_PATH": "/existing/vendor"}, clear=True
        ):
            namespace, vendor = self.register(True)
            self.assertEqual(
                os.environ["ASCEND_CUSTOM_OPP_PATH"], vendor + ":/existing/vendor"
            )
            self.assertFalse((self.root / "source/_cann_ops_custom").exists())
            namespace["import_kernels"](None)
            self.assertEqual(
                os.environ["ASCEND_CUSTOM_OPP_PATH"], vendor + ":/existing/vendor"
            )


if __name__ == "__main__":
    unittest.main()
