# SPDX-License-Identifier: Apache-2.0
"""Host-only P1 build/install contracts; no backend, compiler, pip install or NPU.

Run directly: python -B tests/standalone/test_p1_development.py -v
Native build commands in this suite are mocks producing synthetic file fixtures.
"""

from __future__ import annotations

# Standard
import contextlib
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile
from importlib import util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

# Third Party
from setuptools import Distribution

ROOT = Path(__file__).resolve().parents[2]


def load(path: Path, name: str) -> object:
    """Load a build helper, never the inference framework."""
    spec = util.spec_from_file_location(name, path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILD = load(ROOT / "p1_build.py", "p1_build_test")
with patch.dict(sys.modules, {"p1_build": BUILD}):
    DEV = load(ROOT / "p1_dev.py", "p1_dev_test")


class DevelopmentContracts(unittest.TestCase):
    """Exercise both distribution layouts through their independent helpers."""

    def setUp(self) -> None:
        """Create only disposable synthetic source/material/artifact files."""
        temporary = tempfile.TemporaryDirectory(prefix="p1-contract-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.primary, self.version = DEV.project()
        self.addon = self.primary + "_ascend"
        for module in (BUILD, DEV):
            self.enterContext(patch.object(module, "ROOT", self.root))
        self.dist = Distribution({"name": self.primary, "version": self.version})
        self.command = BUILD.P1BuildExt(self.dist)
        self.command.build_lib = str(self.root / "pip-temporary/lib")
        self.source = self.root / "source"
        self.source.mkdir()
        self.staging = self.root / "retained-native/install"
        self.info = {"cann_version": "8.5.1", "use_hixl": True, "build_mooncake": False}
        (self.root / "pyproject.toml").write_text(
            f'[project]\nname = "{self.primary}"\nversion = "{self.version}"\n'
            '[build-system]\nrequires = ["setuptools>=77.0.3,<81"]\n'
        )
        (self.root / "requirements").mkdir()
        (self.root / "requirements/ascend.txt").write_text("torch==2.9.0\n")

    def populate(self, base: Path, mode: str = "wheel") -> list[str]:
        """Create synthetic native bytes and real generated metadata."""
        paths = []
        for namespace, patterns in BUILD.required_artifacts(
            self.primary, self.info
        ).items():
            for pattern in patterns:
                relative = namespace + "/" + pattern.replace("*", ".fixture")
                path = base / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"SYNTHETIC TEST DATA, NOT AN ELF LIBRARY")
                paths.append(relative)
        for namespace in (self.primary, self.addon):
            path = base / namespace / "__init__.py"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# synthetic package\n")
        BUILD.write_build_metadata(
            base,
            self.primary,
            self.addon,
            self.version,
            {**self.info, "install_mode": mode},
        )
        return paths

    def test_build_names_are_unique_without_creating_directories(self) -> None:
        first, second = BUILD.P1Build(self.dist), BUILD.P1Build(self.dist)
        self.assertNotEqual(first.build_base, second.build_base)
        self.assertTrue(Path(first.build_base).is_relative_to(self.root / "build"))
        self.assertFalse((self.root / "build").exists())

    def test_editable_only_maps_python_without_environment_probe(self) -> None:
        command = BUILD.P1BuildPy(self.dist)
        command.editable_mode = True
        with patch.object(BUILD, "check_environment") as check:
            command.run()
        check.assert_not_called()

    def test_editable_enforces_strict_mode(self) -> None:
        command = BUILD.P1EditableWheel(self.dist)
        with patch.object(BUILD.editable_wheel, "run") as backend:
            command.run()
            self.assertEqual(command.mode, "strict")
            backend.assert_called_once()
        for mode in ("lenient", "compat"):
            command.mode = mode
            with self.assertRaisesRegex(RuntimeError, "strict"):
                command.run()

    def test_editable_maps_all_resources_outside_pip_temporary_tree(self) -> None:
        self.populate(self.staging, "strict-editable")
        self.command.editable_mode = True
        self.command.publish_outputs(self.staging)
        mapping = self.command.get_output_mapping()
        self.assertEqual(set(self.command.get_outputs()), set(mapping))
        self.assertGreaterEqual(len(mapping), 8)
        self.assertFalse(Path(self.command.build_lib).exists())
        for output, source in mapping.items():
            self.assertTrue(Path(output).is_relative_to(Path(self.command.build_lib)))
            self.assertTrue(Path(source).is_relative_to(self.staging))
            self.assertTrue(Path(source).is_file())
        self.assertTrue(any("_build_info.py" in path for path in mapping))
        self.assertTrue(any("p1_build_info.json" in path for path in mapping))
        self.assertFalse((self.root / self.primary).exists())
        self.assertFalse((self.root / "ascend" / self.addon).exists())

    def test_wheel_copies_every_native_resource(self) -> None:
        self.populate(self.staging)
        self.command.publish_outputs(self.staging)
        self.assertEqual(self.command.get_output_mapping(), {})
        for output in self.command.get_outputs():
            relative = Path(output).relative_to(self.command.build_lib)
            self.assertEqual(
                Path(output).read_bytes(), (self.staging / relative).read_bytes()
            )

    def test_external_artifact_symlink_is_rejected(self) -> None:
        self.staging.mkdir(parents=True)
        outside = self.root / "outside.so"
        outside.write_bytes(b"not an artifact")
        (self.staging / "escape.so").symlink_to(outside)
        with self.assertRaisesRegex(RuntimeError, "escapes"):
            self.command.publish_outputs(self.staging)

    def test_material_inventory_ignores_git_control_but_not_extra_payload(self) -> None:
        (self.source / "kernel.cpp").write_text("// fixture")
        expected = BUILD.material_inventory(self.source)
        (self.source / ".git").write_text("gitdir: /irrelevant/control")
        self.assertEqual(BUILD.material_inventory(self.source), expected)
        (self.source / "unexpected.o").write_bytes(b"old executable")
        self.assertNotEqual(BUILD.material_inventory(self.source), expected)

    def test_pip_plans_never_fetch_or_resolve_dependencies(self) -> None:
        output = self.root / "not-created"
        for action in ("build", "editable", "install"):
            plan = DEV.pip_plan(action, output, self.root / "candidate.whl")
            self.assertIn("--no-index", plan)
            self.assertIn("--no-deps", plan)
            self.assertIn("--no-cache-dir", plan)
            if action != "install":
                self.assertIn("--no-build-isolation", plan)
        self.assertIn("editable_mode=strict", DEV.pip_plan("editable", output))
        self.assertFalse(output.exists())

    def test_dry_run_does_not_check_environment_or_create_outputs(self) -> None:
        output = self.root / "not-created"
        with (
            patch.object(DEV, "doctor") as check,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            code = DEV.main(["editable", "--output", str(output), "--dry-run"])
        self.assertEqual(code, 0)
        check.assert_not_called()
        self.assertFalse(output.exists())

    def test_mutating_install_requires_dedicated_environment_confirmation(self) -> None:
        with self.assertRaisesRegex(ValueError, "isolated-env"):
            DEV.check_install_target(False)

    def test_existing_baseline_plugin_or_wrong_version_is_rejected(self) -> None:
        for name in ("vllm-ascend", "lmcache-ascend", "vllm", "lmcache"):

            def installed(candidate: str, name: str = name) -> str:
                if candidate == name:
                    return "old-baseline-version"
                raise DEV.metadata.PackageNotFoundError(candidate)

            with patch.object(DEV.metadata, "version", side_effect=installed):
                with self.assertRaisesRegex(RuntimeError, "Old/conflicting"):
                    DEV.check_install_target(True)

    def test_command_failure_preserves_log_and_exit_status(self) -> None:
        output = self.root / "logs"
        output.mkdir()
        # Harmless interpreter fixture, not a build or an installation.
        command = [
            sys.executable,
            "-B",
            "-c",
            "print('fixture failure'); raise SystemExit(7)",
        ]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(DEV.run_logged(command, output), 7)
        self.assertIn("fixture failure", (output / "command.log").read_text())
        self.assertEqual(
            json.loads((output / "command-result.json").read_text())["returncode"], 7
        )
        with self.assertRaises(FileExistsError):
            DEV.run_logged(command, output)

    def make_wheel(self, path: Path, *, omit: str = "", mode: str = "wheel") -> None:
        """Write a synthetic ZIP; never invoke a wheel backend."""
        self.populate(self.staging, mode)
        metadata_dir = f"{self.primary}-{self.version}.dist-info"
        with zipfile.ZipFile(path, "w") as wheel:
            for item in self.staging.rglob("*"):
                if item.is_file() and item.name != omit:
                    wheel.write(item, str(item.relative_to(self.staging)))
            wheel.writestr(
                metadata_dir + "/METADATA",
                f"Metadata-Version: 2.4\nName: {self.primary}\n"
                f"Version: {self.version}\n",
            )
            wheel.writestr(
                metadata_dir + "/WHEEL",
                "Root-Is-Purelib: false\nTag: cp311-cp311-linux_aarch64\n",
            )

    def test_wheel_identity_and_resources_are_checked_without_loading(self) -> None:
        wheel = self.root / "fixture.whl"
        self.make_wheel(wheel)
        report = DEV.wheel_info(wheel)
        self.assertEqual(report["distribution"], self.primary)
        self.assertFalse(report["ABI_tested"])

    def test_missing_wheel_resource_and_editable_snapshot_are_rejected(self) -> None:
        wheel = self.root / "fixture.whl"
        self.make_wheel(wheel, omit="_build_info.py")
        with self.assertRaisesRegex(ValueError, "Missing wheel resource"):
            DEV.wheel_info(wheel)
        self.make_wheel(wheel, mode="strict-editable")
        with self.assertRaisesRegex(ValueError, "regular wheel"):
            DEV.wheel_info(wheel)

    def test_metadata_failure_does_not_import_torch_or_run_compilers(self) -> None:
        with (
            patch.object(
                DEV.metadata,
                "version",
                side_effect=DEV.metadata.PackageNotFoundError("torch"),
            ),
            patch.object(BUILD, "check_environment") as check,
            patch.object(DEV.subprocess, "run") as process,
        ):
            report = DEV.doctor()
        self.assertFalse(report["passed"])
        check.assert_not_called()
        process.assert_not_called()

    def test_mocked_native_rebuild_uses_fresh_work_and_install_directories(
        self,
    ) -> None:
        # Cover both the device-object relink and private ACLNN source paths.
        for relative, _ in BUILD.MATERIALS.values():
            material = self.root / "ascend" / relative
            (material / "include").mkdir(parents=True)
            (material / "CMakeLists.txt").write_text("# fixture")
        cann = self.root / "sdk"
        ini = cann / "aarch64-linux/data/platform_config/Ascend910B3.ini"
        ini.parent.mkdir(parents=True)
        ini.write_text("[version]\nAIC_version=AscendC-220\n")
        info = {
            **self.info,
            "cann": str(cann),
            "torch_npu_path": "/fixture/npu",
            "torch": {"path": "/fixture/torch", "cmake": "/fixture/cmake", "abi": 1},
        }
        directories = []
        install = None
        selected = None

        def native(command: list[str], **kwargs: object) -> None:
            nonlocal install, selected
            if command[0] == "bash":
                resource = (
                    Path(command[2])
                    / "vllm_ascend/_cann_ops_custom/vendors/vllm-ascend"
                    / "op_api/lib/fixture.so"
                )
                resource.parent.mkdir(parents=True)
                resource.write_bytes(b"SYNTHETIC ACLNN")
            if command[:2] == ["cmake", "-S"]:
                prefix = next(
                    value.split("=", 1)[1]
                    for value in command
                    if value.startswith("-DCMAKE_INSTALL_PREFIX=")
                )
                install = Path(prefix).parent
                selected = Path(prefix).name.removesuffix("_ascend")
                directories.append((command[command.index("-B") + 1], selected))
            if command[:2] == ["cmake", "--install"]:
                for namespace, patterns in BUILD.required_artifacts(
                    selected, info
                ).items():
                    for pattern in patterns:
                        path = install / namespace / pattern.replace("*", ".fixture")
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(b"SYNTHETIC")

        with (
            patch.object(BUILD, "check_environment", side_effect=lambda: dict(info)),
            patch.object(BUILD, "verify_materials", return_value={"files": 1}),
            patch.object(BUILD.platform, "machine", return_value="aarch64"),
            patch.object(BUILD.metadata, "version", return_value=BUILD.TRITON_VERSION),
            patch.object(
                BUILD.subprocess, "check_output", return_value="/fixture/pybind"
            ),
            patch.object(BUILD.subprocess, "run", side_effect=native),
            patch.dict(
                os.environ, {"MAX_JOBS": "1", "BUILD_MOONCAKE": "0"}, clear=True
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            for primary in ("lmcache", "vllm"):
                for _ in range(2):
                    command = BUILD.P1BuildExt(
                        Distribution(
                            {"name": primary, "version": DEV.VERSIONS[primary]}
                        )
                    )
                    command.build_lib = str(self.root / "wheel-lib")
                    command.run()
        self.assertEqual(len({directory for directory, _ in directories}), 4)
        for directory, primary in directories:
            for namespace, patterns in BUILD.required_artifacts(primary, info).items():
                for pattern in patterns:
                    self.assertTrue(
                        list((Path(directory) / "install" / namespace).glob(pattern))
                    )
        for relative, _ in BUILD.MATERIALS.values():
            self.assertFalse((self.root / "ascend" / relative / "build").exists())

    def test_invalid_job_count_fails_before_native_commands(self) -> None:
        with (
            patch.object(BUILD, "check_environment", return_value={}),
            patch.dict(os.environ, {"MAX_JOBS": "0"}),
            patch.object(BUILD.subprocess, "run") as process,
        ):
            with self.assertRaisesRegex(RuntimeError, "MAX_JOBS"):
                self.command.run()
        process.assert_not_called()

    def test_native_failure_does_not_publish_or_reuse_old_payload(self) -> None:
        relative, _ = BUILD.MATERIALS["lmcache"]
        material = self.root / "ascend" / relative
        material.mkdir(parents=True)
        (material / "CMakeLists.txt").write_text("# fixture")
        ini = self.root / "sdk/aarch64-linux/data/platform_config/Ascend910B3.ini"
        ini.parent.mkdir(parents=True)
        ini.write_text("[version]\nAIC_version=AscendC-220\n")
        info = {
            **self.info,
            "cann": str(self.root / "sdk"),
            "torch_npu_path": "/fixture/npu",
            "torch": {"path": "/fixture/torch", "cmake": "/fixture/cmake", "abi": 1},
        }
        command = BUILD.P1BuildExt(
            Distribution({"name": "lmcache", "version": "0.4.3+ascend.p1"})
        )
        command.build_lib = str(self.root / "lib")
        with (
            patch.object(BUILD, "check_environment", return_value=info),
            patch.object(BUILD, "verify_materials", return_value={"files": 1}),
            patch.object(BUILD.platform, "machine", return_value="aarch64"),
            patch.object(
                BUILD.subprocess, "check_output", return_value="/fixture/pybind"
            ),
            patch.object(
                BUILD.subprocess,
                "run",
                side_effect=subprocess.CalledProcessError(2, ["cmake"]),
            ),
            patch.dict(
                os.environ, {"MAX_JOBS": "1", "BUILD_MOONCAKE": "0"}, clear=True
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            with self.assertRaises(subprocess.CalledProcessError):
                command.run()
        self.assertEqual(command.native_output_mapping, {})
        self.assertFalse(Path(command.build_lib).exists())
        self.assertEqual(len(list((self.root / "build/p1-native").glob("run-*"))), 1)

    def test_verify_accepts_strict_link_tree_and_rejects_wrong_mode(self) -> None:
        tree = self.root / "build/__editable__.fixture"
        self.populate(tree, "strict-editable")
        distribution = SimpleNamespace(
            version=self.version,
            read_text=lambda _: json.dumps({"dir_info": {"editable": True}}),
        )
        with (
            patch.object(DEV, "check_install_target"),
            patch.object(DEV.metadata, "distribution", return_value=distribution),
            patch.object(
                DEV.PathFinder,
                "find_spec",
                side_effect=lambda name: SimpleNamespace(
                    origin=str(tree / name / "__init__.py")
                ),
            ),
        ):
            self.assertTrue(DEV.verify("editable")["passed"])
            with self.assertRaisesRegex(RuntimeError, "mode"):
                DEV.verify("wheel")

    def test_material_snapshot_registers_only_pinned_archive(self) -> None:
        relative, commit = BUILD.MATERIALS[self.primary]
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w") as stream:
            item = tarfile.TarInfo("kernel.cpp")
            item.size = len(b"// fixture")
            stream.addfile(item, io.BytesIO(b"// fixture"))

        def query(source: Path, *args: str) -> str:
            return (
                {("--show-toplevel",): str(self.source)}.get(args[1:], commit)
                if args[0] == "rev-parse"
                else ""
            )

        with (
            patch.object(DEV, "git", side_effect=query),
            patch.object(
                DEV.subprocess, "check_output", return_value=archive.getvalue()
            ),
        ):
            report = DEV.materials(self.source)
        self.assertEqual(report["commit"], commit)
        self.assertEqual(report["files"], 1)
        self.assertEqual(
            (self.root / "ascend" / relative / "kernel.cpp").read_text(), "// fixture"
        )
        self.assertEqual(BUILD.verify_materials(self.primary), report)

    def test_install_path_verification_rejects_source_shadowing(self) -> None:
        paths = {
            namespace: self.root / "site-packages" / namespace
            for namespace in (self.primary, self.addon)
        }
        self.populate(self.root / "site-packages")
        distribution = SimpleNamespace(
            version=self.version,
            read_text=lambda _: "{}",
            locate_file=lambda name: paths[name],
        )
        with (
            patch.object(DEV, "check_install_target"),
            patch.object(DEV.metadata, "distribution", return_value=distribution),
            patch.object(
                DEV.PathFinder,
                "find_spec",
                side_effect=lambda name: SimpleNamespace(
                    origin=str(paths[name] / "__init__.py")
                ),
            ),
        ):
            self.assertTrue(DEV.verify("wheel")["passed"])
        with (
            patch.object(DEV, "check_install_target"),
            patch.object(DEV.metadata, "distribution", return_value=distribution),
            patch.object(
                DEV.PathFinder,
                "find_spec",
                return_value=SimpleNamespace(
                    origin=str(self.root / self.primary / "__init__.py")
                ),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "shadowed"):
                DEV.verify("wheel")


if __name__ == "__main__":
    unittest.main()
