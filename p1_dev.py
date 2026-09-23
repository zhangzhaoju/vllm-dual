# SPDX-License-Identifier: Apache-2.0
"""Standalone P1 intranet materials, build and isolated-install entry point.

No automatic downloads or dependency installation. Use --dry-run to print pip
commands without compiling, installing or creating directories.
"""

from __future__ import annotations

# Standard
import argparse
import fnmatch
import hashlib
import io
import json
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from email.parser import BytesParser
from importlib import metadata
from importlib.machinery import PathFinder
from pathlib import Path

import tomllib

# Third Party
from packaging.requirements import Requirement

# Local
import p1_build as builder

ROOT = Path(__file__).resolve().parent
VERSIONS = {"vllm": "0.18.0+ascend.p1", "lmcache": "0.4.3+ascend.p1"}


def project() -> tuple[str, str]:
    """Return this checkout's distribution identity without importing frameworks."""
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    return config["name"], config["version"]


def digest(path: Path) -> str:
    """Return a file SHA-256 for local artifact provenance."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_report(path: Path, report: dict) -> None:
    """Create a new JSON report, refusing to overwrite an earlier result."""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def git(source: Path, *args: str) -> str:
    """Run a read-only Git query on an explicitly selected material checkout."""
    return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()


def materials(source: Path) -> dict:
    """Snapshot a clean pinned submodule, or register an identical initialized one.

    Existing payloads are never overwritten. Raises on drift, unsafe links,
    nested submodules, or an unexpected source commit.
    """
    primary, _ = project()
    relative, commit = builder.MATERIALS[primary]
    target = ROOT / "ascend" / relative
    manifest = ROOT / "ascend/submodule-materials.json"
    if target.is_symlink():
        raise ValueError("Material destination must not be a symlink")
    if manifest.exists():
        return builder.verify_materials(primary)
    source = source.resolve()
    if Path(git(source, "rev-parse", "--show-toplevel")).resolve() != source:
        raise ValueError(f"Not an initialized submodule checkout: {source}")
    if git(source, "rev-parse", "HEAD") != commit or git(
        source, "status", "--porcelain"
    ):
        raise ValueError(f"Expected clean material checkout at {commit}")
    if any(
        line.startswith("160000 ")
        for line in git(source, "ls-tree", "-r", "HEAD").splitlines()
    ):
        raise ValueError("Nested submodules require explicit material review")
    if source != target.resolve() and (
        target.is_file() or (target.exists() and any(target.iterdir()))
    ):
        raise FileExistsError(f"Refusing to overwrite material: {target}")
    archive = subprocess.check_output(["git", "-C", str(source), "archive", commit])
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="p1-material-", dir=target.parent
    ) as temporary:
        payload = Path(temporary) / "payload"
        payload.mkdir()
        with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
            stream.extractall(payload, filter="data")
        files = builder.material_inventory(payload)
        if not files:
            raise ValueError("Empty material archive")
        if source == target.resolve():
            if builder.material_inventory(target) != files:
                raise ValueError(
                    "Initialized submodule contains extra/changed payload files"
                )
        else:
            payload.rename(target)
        write_report(
            manifest,
            {
                "schema_version": 1,
                "path": relative,
                "commit": commit,
                "archive_sha256": hashlib.sha256(archive).hexdigest(),
                "files": files,
            },
        )
    return builder.verify_materials(primary)


def doctor(*, building: bool = True) -> dict:
    """Check candidate dependencies/materials; never install packages or probe NPU.

    A torch-only subprocess reads build paths/ABI after metadata checks pass.
    Installation checks do not require compiler tools or source materials.
    """
    primary, _ = project()
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    requirements = set(builder.runtime_requirements())
    if building:
        requirements.update(config["build-system"]["requires"])
    checks, errors = [], []
    for raw in sorted(requirements):
        req = Requirement(raw)
        if req.marker and not req.marker.evaluate():
            continue
        try:
            installed = metadata.version(req.name)
        except metadata.PackageNotFoundError:
            installed = None
        passed = installed is not None and req.specifier.contains(
            installed, prereleases=True
        )
        checks.append({"requirement": raw, "installed": installed, "passed": passed})
        if not passed:
            errors.append(f"Missing/mismatched {raw}: {installed}")
    if sys.version_info[:2] != (3, 11) or platform.machine() != "aarch64":
        errors.append("P1 candidate requires Python 3.11 / aarch64")
    if building:
        for command in ("cmake", "g++", "gcc", "make", "bash"):
            if shutil.which(command) is None:
                errors.append(f"Missing tool: {command}")
        try:
            builder.verify_materials(primary)
        except (OSError, ValueError, RuntimeError) as exc:
            errors.append(str(exc))
    info = None
    if not errors:
        try:
            info = builder.check_environment()
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            errors.append(str(exc))
    return {
        "scope": "dependencies_and_build_inputs_not_NPU_acceptance",
        "requirements": checks,
        "environment": info,
        "errors": errors,
        "passed": not errors,
    }


def check_install_target(isolated: bool) -> None:
    """Require dedicated-environment confirmation; reject old four-pack installs."""
    if not isolated:
        raise ValueError(
            "Pass --isolated-env only in a dedicated P1 container/interpreter"
        )
    for name in ("vllm-ascend", "lmcache-ascend", "vllm", "lmcache"):
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
        if name not in VERSIONS or installed != VERSIONS[name]:
            raise RuntimeError(
                f"Old/conflicting distribution {name}=={installed}; "
                "prepare a clean container"
            )


def wheel_info(path: Path) -> dict:
    """Validate this project's native wheel identity, resources and build provenance."""
    primary, version = project()
    addon = primary + "_ascend"
    with zipfile.ZipFile(path) as wheel:
        names = wheel.namelist()
        metas = [name for name in names if name.endswith(".dist-info/METADATA")]
        if len(metas) != 1 or len(names) != len(set(names)) or wheel.testzip():
            raise ValueError("Invalid wheel metadata or duplicate/corrupt entries")
        if any(Path(name).is_absolute() or ".." in Path(name).parts for name in names):
            raise ValueError("Unsafe wheel member")
        meta = BytesParser().parsebytes(wheel.read(metas[0]))
        if meta["Name"].lower() != primary or meta["Version"] != version:
            raise ValueError("Wheel identity does not match this P1 checkout")
        dist_info = metas[0].rsplit("/", 1)[0]
        tags = BytesParser().parsebytes(wheel.read(dist_info + "/WHEEL"))
        if tags["Root-Is-Purelib"] != "false" or tags.get_all("Tag") != [
            "cp311-cp311-linux_aarch64"
        ]:
            raise ValueError("Expected a native cp311-cp311-linux_aarch64 wheel")
        info = json.loads(wheel.read(addon + "/p1_build_info.json"))
        if info.get("install_mode") != "wheel":
            raise ValueError(
                "Only a regular wheel is accepted here, not editable metadata"
            )
        required = builder.required_artifacts(primary, info)
        required.setdefault(primary, []).extend(["__init__.py", "_version.py"])
        required.setdefault(addon, []).extend(
            ["__init__.py", "_version.py", "_build_info.py"]
        )
        for namespace, patterns in required.items():
            for pattern in patterns:
                if not any(
                    fnmatch.fnmatchcase(name, namespace + "/" + pattern)
                    for name in names
                ):
                    raise ValueError(f"Missing wheel resource: {namespace}/{pattern}")
    return {
        "distribution": primary,
        "version": version,
        "wheel": str(path),
        "sha256": digest(path),
        "ABI_tested": False,
    }


def pip_plan(action: str, output: Path, wheel: Path | None = None) -> list[str]:
    """Build a no-network/no-dependency-update command; perform no side effects."""
    command = [
        sys.executable,
        "-B",
        "-m",
        "pip",
        "--disable-pip-version-check",
        "--no-cache-dir",
    ]
    if action == "build":
        return [
            *command,
            "wheel",
            "--verbose",
            "--no-index",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(output / "wheels"),
            str(ROOT),
        ]
    if action == "editable":
        return [
            *command,
            "install",
            "--verbose",
            "--no-index",
            "--no-deps",
            "--no-build-isolation",
            "--force-reinstall",
            "--config-settings",
            "editable_mode=strict",
            "--editable",
            str(ROOT),
        ]
    if action == "install" and wheel is not None:
        return [
            *command,
            "install",
            "--verbose",
            "--no-index",
            "--no-deps",
            "--force-reinstall",
            str(wheel),
        ]
    raise ValueError("Install requires an explicit wheel")


def run_logged(command: list[str], output: Path) -> int:
    """Run one explicitly requested pip action and retain complete logs/exit status."""
    with (output / "command.log").open("x", encoding="utf-8") as log:
        with subprocess.Popen(
            command,
            cwd=output,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        ) as process:
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                print(line, end="", flush=True)
            code = process.wait()
    write_report(output / "command-result.json", {"argv": command, "returncode": code})
    return code


def verify(mode: str) -> dict:
    """Check distribution/import paths and native files without loading an NPU."""
    primary, version = project()
    addon = primary + "_ascend"
    check_install_target(True)
    distribution = metadata.distribution(primary)
    if distribution.version != version:
        raise RuntimeError("Unexpected installed distribution version")
    direct = json.loads(distribution.read_text("direct_url.json") or "{}")
    editable = bool(direct.get("dir_info", {}).get("editable"))
    if editable != (mode == "editable"):
        raise RuntimeError("Installed wheel/editable mode does not match --mode")
    paths = {}
    for namespace in (primary, addon):
        spec = PathFinder.find_spec(namespace)
        if spec is None or spec.origin is None:
            raise RuntimeError(f"Missing namespace: {namespace}")
        directory = Path(spec.origin).absolute().parent
        expected = (
            ROOT / "build"
            if editable
            else Path(distribution.locate_file(namespace)).resolve()
        )
        if (editable and not directory.is_relative_to(expected)) or (
            not editable and directory.resolve() != expected
        ):
            raise RuntimeError(
                f"Import shadowed by another checkout: {namespace}: {directory}"
            )
        paths[namespace] = directory
    info = json.loads((paths[addon] / "p1_build_info.json").read_text())
    expected_mode = "strict-editable" if editable else "wheel"
    if info.get("install_mode") != expected_mode:
        raise RuntimeError("Generated build metadata does not match installed mode")
    for namespace, patterns in builder.required_artifacts(primary, info).items():
        for pattern in patterns:
            if not any(path.is_file() for path in paths[namespace].glob(pattern)):
                raise RuntimeError(
                    f"Missing installed native resource: {namespace}/{pattern}"
                )
    for namespace in (primary, addon):
        if not (paths[namespace] / "_version.py").is_file():
            raise RuntimeError(f"Missing generated version: {namespace}")
    if not (paths[addon] / "_build_info.py").is_file():
        raise RuntimeError("Missing generated Ascend build metadata")
    return {
        "scope": "installation_paths_and_files_not_ABI_or_NPU",
        "distribution": primary,
        "version": version,
        "mode": mode,
        "namespaces": {key: str(value) for key, value in paths.items()},
        "passed": True,
    }


def main(argv: list[str] | None = None) -> int:
    """Execute one intranet action, or show its plan; failures keep a nonzero status."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser(
        "materials", help="register/copy a pinned local Git submodule"
    )
    prep.add_argument("--from-submodule", required=True, type=Path)
    check = sub.add_parser("doctor", help="read-only dependency/material gate")
    check.add_argument("--output", type=Path)
    installed = sub.add_parser(
        "verify", help="read-only installation path/resource check"
    )
    installed.add_argument("--mode", choices=("wheel", "editable"), required=True)
    installed.add_argument("--output", type=Path)
    for action in ("build", "editable", "install"):
        command = sub.add_parser(action)
        command.add_argument(
            "--output", required=True, type=Path, help="new result directory"
        )
        command.add_argument("--dry-run", action="store_true")
        if action != "build":
            command.add_argument(
                "--isolated-env",
                action="store_true",
                help="confirm dedicated P1 environment, not a serving baseline",
            )
        if action == "install":
            command.add_argument("--wheel", required=True, type=Path)
    args = parser.parse_args(argv)
    # The script directory is not an installed package location. Preserve all
    # other entries so explicit PYTHONPATH contamination still fails verification.
    if sys.path and Path(sys.path[0]).resolve() == ROOT:
        sys.path.pop(0)
    try:
        if args.action == "materials":
            print(json.dumps(materials(args.from_submodule), indent=2))
            return 0
        if args.action in ("doctor", "verify"):
            report = doctor() if args.action == "doctor" else verify(args.mode)
            if args.output:
                write_report(args.output.resolve(), report)
            print(json.dumps(report, indent=2))
            return 0 if report["passed"] else 1
        output = args.output.resolve()
        wheel = args.wheel.resolve() if args.action == "install" else None
        command = pip_plan(args.action, output, wheel)
        if args.dry_run:
            print(
                json.dumps(
                    {"argv": command, "output": str(output), "executed": False},
                    indent=2,
                )
            )
            return 0
        if args.action != "build":
            check_install_target(args.isolated_env)
        if output.is_relative_to(ROOT / "ascend") or output.is_relative_to(
            ROOT / project()[0]
        ):
            raise ValueError(
                "Reports must not be written into package/material directories"
            )
        output.mkdir(parents=True, exist_ok=False)
        if args.action == "install":
            write_report(output / "wheel-input.json", wheel_info(wheel))
        preflight = doctor(building=args.action != "install")
        write_report(output / "preflight.json", preflight)
        if not preflight["passed"]:
            print(json.dumps(preflight, indent=2))
            return 1
        if args.action == "build":
            (output / "wheels").mkdir()
        code = run_logged(command, output)
        if code:
            return code
        if args.action == "build":
            wheels = list((output / "wheels").glob("*.whl"))
            if len(wheels) != 1:
                raise RuntimeError("Expected exactly one newly built wheel")
            write_report(output / "artifact.json", wheel_info(wheels[0]))
        return 0
    except (
        OSError,
        ValueError,
        RuntimeError,
        KeyError,
        subprocess.SubprocessError,
        zipfile.BadZipFile,
    ) as exc:
        print(f"P1 STOP: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
