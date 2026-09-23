# SPDX-License-Identifier: Apache-2.0
"""P1's Ascend-only build commands; imported without torch or device probing.

The two repositories carry independent copies of this build helper. Native
builds are performed only by intranet operators. Metadata/sdist preparation
does not execute a compiler, initialize an NPU, or fetch dependencies.
"""

from __future__ import annotations

# Standard
import configparser
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from importlib import metadata
from pathlib import Path
from uuid import uuid4

# Third Party
from setuptools import Extension, find_packages
from setuptools.command.build import build
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
from setuptools.command.editable_wheel import editable_wheel

ROOT = Path(__file__).resolve().parent
TORCH_VERSION = "2.9.0"
TORCH_NPU_VERSION = "2.9.0.post2"
TRITON_VERSION = "3.2.0.dev20260322"
MATERIALS = {
    "vllm": ("csrc/third_party/catlass", "716fd7baa7fb7f6cac0488bb628fd1dd0e875641"),
    "lmcache": ("third_party/kvcache-ops", "9f18d2339bc58a43429f7d5bdaef1628c820eff5"),
}


def material_inventory(directory: Path) -> dict:
    """Hash material payload files; exclude only the top-level Git control entry."""
    actual = {}
    for path in sorted(directory.rglob("*")):
        if path.relative_to(directory).parts[0] == ".git":
            continue
        if path.is_symlink():
            link = str(path.readlink())
            if not path.resolve().is_relative_to(directory.resolve()):
                raise RuntimeError(f"Material symlink escapes its tree: {path}")
            sha = hashlib.sha256(link.encode()).hexdigest()
        elif path.is_file():
            link = None
            with path.open("rb") as stream:
                sha = hashlib.file_digest(stream, "sha256").hexdigest()
        else:
            continue
        actual[str(path.relative_to(directory))] = {"sha256": sha, "symlink": link}
    return actual


def verify_materials(primary: str) -> dict:
    """Check every material file, including builds from an sdist without .git."""
    relative, commit = MATERIALS[primary]
    manifest = ROOT / "ascend/submodule-materials.json"
    if not manifest.is_file():
        raise RuntimeError("Run python -B p1_dev.py materials --help first")
    expected = json.loads(manifest.read_text())
    if expected.get("commit") != commit or expected.get("path") != relative:
        raise RuntimeError("Submodule material does not match the pinned commit")
    actual = material_inventory(ROOT / "ascend" / relative)
    if not actual or actual != expected.get("files"):
        raise RuntimeError("Submodule payload is missing or changed; do not build")
    return {"path": relative, "commit": commit, "files": len(actual)}


def runtime_requirements() -> list[str]:
    return [
        line.split("#", 1)[0].strip()
        for line in (ROOT / "requirements/ascend.txt").read_text().splitlines()
        if line.split("#", 1)[0].strip()
    ]


def package_names(primary: str, addon: str) -> list[str]:
    return find_packages(str(ROOT), include=[primary, primary + ".*"]) + find_packages(
        str(ROOT / "ascend"), include=[addon, addon + ".*"]
    )


def check_environment() -> dict:
    """Validate explicit build inputs; called by build commands, never metadata."""
    if os.environ.get("SOC_VERSION", "").lower() != "ascend910b3":
        raise RuntimeError("P1 candidate requires explicit SOC_VERSION=ascend910b3")
    if os.environ.get("USE_MINDSPORE", "").lower() not in ("", "0", "false", "off"):
        raise RuntimeError("P1 builds the PyTorch Ascend path only")
    if os.environ.get("BUILD_WITH_HIP", "0") != "0":
        raise RuntimeError("HIP builds are not part of P1")
    if os.environ.get("VLLM_TARGET_DEVICE", "ascend") != "ascend":
        raise RuntimeError(
            "P1 target is ascend; remove the old empty/cpu/cuda override"
        )
    if os.environ.get("VLLM_USE_PRECOMPILED", "0") != "0":
        raise RuntimeError("Precompiled upstream wheels are not P1 build inputs")
    if os.environ.get("COMPILE_CUSTOM_KERNELS", "1") != "1":
        raise RuntimeError("A P1 wheel must contain the required Ascend kernels")
    if sys.version_info[:2] != (3, 11) or platform.machine() != "aarch64":
        raise RuntimeError("P1 candidate requires Python 3.11 on aarch64")
    if metadata.version("torch").split("+", 1)[0] != TORCH_VERSION:
        raise RuntimeError("Reuse the approved torch 2.9.0 environment; do not upgrade")
    if metadata.version("torch-npu") != TORCH_NPU_VERSION:
        raise RuntimeError("torch-npu differs from the approved P1 candidate")
    cann_value = os.environ.get("ASCEND_HOME_PATH")
    if not cann_value or not Path(cann_value).is_dir():
        raise RuntimeError(
            "Set ASCEND_HOME_PATH and source the approved CANN environment"
        )
    cann = Path(cann_value).resolve()
    info_paths = (
        cann / f"{platform.machine()}-linux/ascend_toolkit_install.info",
        cann / "ascend_toolkit_install.info",
    )
    cann_version = None
    for path in info_paths:
        if path.is_file():
            match = re.search(r"(?m)^version\s*=\s*(.+)$", path.read_text())
            if match:
                cann_version = match.group(1).strip()
                break
    if cann_version is None or tuple(
        map(int, re.findall(r"\d+", cann_version)[:3])
    ) != (8, 5, 1):
        raise RuntimeError("Cannot verify CANN 8.5.1 from ascend_toolkit_install.info")
    # torch is imported only in the build subprocess, never while generating metadata.
    torch_info = json.loads(
        subprocess.check_output(
            [
                sys.executable,
                "-B",
                "-c",
                "import json, torch; print(json.dumps({'path':torch.__path__[0],"
                "'cmake':torch.utils.cmake_prefix_path,"
                "'abi':int(torch._C._GLIBCXX_USE_CXX11_ABI)}))",
            ],
            text=True,
        )
    )
    npu_path = Path(
        metadata.distribution("torch-npu").locate_file("torch_npu")
    ).resolve()
    if not (npu_path / "include").is_dir():
        raise RuntimeError("The current interpreter cannot locate torch_npu headers")
    use_hixl = os.environ.get("USE_HIXL", "1").lower() in ("1", "true", "on")
    if os.environ.get("USE_HIXL", "1").lower() not in (
        "1",
        "true",
        "on",
        "0",
        "false",
        "off",
    ):
        raise RuntimeError("USE_HIXL must be 0 or 1")
    return {
        "soc": "ascend910b3",
        "cann": str(cann),
        "cann_version": cann_version,
        "torch": torch_info,
        "torch_npu_path": str(npu_path),
        "use_hixl": use_hixl,
        "python": sys.executable,
    }


def write_build_metadata(
    build_lib: Path, primary: str, addon: str, version: str, info: dict
) -> None:
    version_tuple = tuple(map(int, version.split("+", 1)[0].split(".")))
    for name in (primary, addon):
        directory = build_lib / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "_version.py").write_text(
            "# Generated from the single distribution version.\n"
            f"__version__ = version = {version!r}\n"
            f"__version_tuple__ = version_tuple = {version_tuple!r}\n"
        )
    if addon == "vllm_ascend":
        text = "# Generated for the approved 910B3 build.\n__device_type__ = 'A2'\n"
    else:
        text = (
            "# Generated for the approved 910B3 build.\n"
            "__soc_version__ = 'Ascend910B3'\n__framework_name__ = 'pytorch'\n"
            f"__cann_version__ = {info['cann_version']!r}\n"
            "def cann_version_tuple():\n"
            "    import re\n"
            "    return tuple(int(p) for p in re.findall(r'\\d+', __cann_version__))\n"
        )
    (build_lib / addon / "_build_info.py").write_text(text)
    (build_lib / addon / "p1_build_info.json").write_text(
        json.dumps(info, indent=2) + "\n"
    )


def required_artifacts(primary: str, info: dict) -> dict[str, list[str]]:
    """Return required native resources for the selected compiled feature set."""
    if primary == "vllm":
        return {
            "vllm_ascend": [
                "vllm_ascend_C*.so",
                "libvllm_ascend_kernels.so",
                "_cann_ops_custom/vendors/vllm-ascend/op_api/lib/*.so",
            ]
        }
    channels = (
        ["hixl_npu_comms*.so", "hcomm_onesided*.so"]
        if info["use_hixl"]
        else ["hccl_npu_comms*.so"]
    )
    host = ["native_storage_ops*.so", "lmcache_fs*.so", "lmcache_redis*.so"]
    if info.get("build_mooncake", False):
        host.append("lmcache_mooncake*.so")
    return {
        "lmcache_ascend": ["c_ops*.so", "libcache_kernels.so", *channels],
        "lmcache": host,
    }


class P1Build(build):
    """Give every wheel build a fresh Python/native staging namespace."""

    def initialize_options(self) -> None:
        """Choose a unique path without creating directories during metadata hooks."""
        super().initialize_options()
        self.build_base = str(ROOT / "build" / ("p1-" + uuid4().hex))


class P1BuildPy(build_py):
    def run(self) -> None:
        """Copy wheel Python files, or let strict editable map them to source."""
        if self.editable_mode:
            return
        info = check_environment()
        super().run()
        primary = self.distribution.get_name()
        addon = primary + "_ascend"
        write_build_metadata(
            Path(self.build_lib), primary, addon, self.distribution.get_version(), info
        )


class P1EditableWheel(editable_wheel):
    """Expose both namespaces and all native resources through a strict link tree."""

    def run(self) -> None:
        """Build a development installation; regular wheels remain the exit gate."""
        if self.mode not in (None, "strict"):
            raise RuntimeError("P1 editable installs require editable_mode=strict")
        self.mode = "strict"
        super().run()


class P1BuildExt(build_ext):
    def initialize_options(self) -> None:
        """Initialize the complete native resource mapping used by PEP 660."""
        super().initialize_options()
        self.native_output_mapping = {}

    def get_outputs(self) -> list[str]:
        """Report all generated files, including host libraries and CANN resources."""
        return sorted(self.native_output_mapping) or super().get_outputs()

    def get_output_mapping(self) -> dict[str, str]:
        """Map strict-editable outputs to persistent, successful build artifacts."""
        return dict(self.native_output_mapping) if self.editable_mode else {}

    def publish_outputs(self, staging: Path) -> None:
        """Copy wheel artifacts or map editable files without changing source files."""
        build_lib = Path(self.build_lib).resolve()
        mapping = {}
        for source in sorted(staging.rglob("*")):
            if not source.is_file():
                continue
            if not source.resolve().is_relative_to(staging.resolve()):
                raise RuntimeError(
                    f"Native artifact escapes its staging tree: {source}"
                )
            destination = build_lib / source.relative_to(staging)
            if not self.editable_mode:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
            mapping[str(destination)] = str(source)
        self.native_output_mapping = mapping

    def run(self) -> None:
        """Compile into a fresh retained directory; publish only validated outputs."""
        info = check_environment()
        jobs = int(os.environ.get("MAX_JOBS", str(os.cpu_count() or 1)))
        if jobs < 1:
            raise RuntimeError("MAX_JOBS must be positive")
        if self.inplace and not self.editable_mode:
            raise RuntimeError("Use pip install -e . instead of build_ext --inplace")
        primary = self.distribution.get_name()
        addon = primary + "_ascend"
        info["submodule"] = verify_materials(primary)
        if primary == "vllm" and metadata.version("triton-ascend") != TRITON_VERSION:
            raise RuntimeError("triton-ascend differs from the approved P1 candidate")
        native = ROOT / "ascend"
        needed = (
            native / "csrc/third_party/catlass/include"
            if primary == "vllm"
            else native / "third_party/kvcache-ops/CMakeLists.txt"
        )
        if not needed.exists():
            raise RuntimeError(
                f"Missing pinned submodule material: {needed}; no automatic fetch"
            )
        # Retain failure diagnostics even when pip removes its temporary build dir.
        native_runs = ROOT / "build/p1-native"
        native_runs.mkdir(parents=True, exist_ok=True)
        build_dir = Path(tempfile.mkdtemp(prefix="run-", dir=native_runs))
        print(f"[P1] Fresh native build directory: {build_dir}", flush=True)
        staging = build_dir / "install"
        package_dir = staging / addon
        package_dir.mkdir(parents=True, exist_ok=True)
        pybind = subprocess.check_output(
            [sys.executable, "-B", "-m", "pybind11", "--cmakedir"], text=True
        ).strip()
        args = [
            "cmake",
            "-S",
            str(ROOT),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_INSTALL_PREFIX={package_dir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DPython3_EXECUTABLE={sys.executable}",
            f"-DPYTHON_INCLUDE_PATH={sysconfig.get_path('include')}",
            f"-DCMAKE_PREFIX_PATH={pybind};{info['torch']['cmake']}",
            f"-DTORCH_PATH={info['torch']['path']}",
            f"-DTORCH_NPU_PATH={info['torch_npu_path']}",
            f"-DASCEND_HOME_PATH={info['cann']}",
            f"-DASCEND_CANN_PACKAGE_PATH={info['cann']}",
            f"-DGLIBCXX_USE_CXX11_ABI={info['torch']['abi']}",
            "-DFETCHCONTENT_FULLY_DISCONNECTED=ON",
        ]
        for env, option in (("CC", "CMAKE_C_COMPILER"), ("CXX", "CMAKE_CXX_COMPILER")):
            if os.environ.get(env):
                args.append(f"-D{option}={os.environ[env]}")
        if primary == "vllm":
            args.append("-DSOC_VERSION=ascend910b3")
            # Run upstream's custom-op generator in a fresh private source copy.
            aclnn_root = build_dir / "aclnn-source"
            if aclnn_root.exists():
                raise RuntimeError(
                    "Use a fresh build directory; refusing stale ACLNN artifacts"
                )
            shutil.copytree(
                native / "csrc",
                aclnn_root / "csrc",
                ignore=shutil.ignore_patterns(".git", "build", "output", "__pycache__"),
            )
            subprocess.run(
                [
                    "bash",
                    str(native / "csrc/build_aclnn.sh"),
                    str(aclnn_root),
                    "ascend910b3",
                ],
                cwd=aclnn_root,
                check=True,
            )
            shutil.copytree(
                aclnn_root / "vllm_ascend/_cann_ops_custom",
                package_dir / "_cann_ops_custom",
                dirs_exist_ok=True,
            )
        else:
            soc = "Ascend910B3"
            ini = (
                Path(info["cann"])
                / f"{platform.machine()}-linux/data/platform_config/{soc}.ini"
            )
            config = configparser.ConfigParser()
            if not config.read(ini):
                raise RuntimeError(f"Missing SoC platform configuration: {ini}")
            aicore = config.get("version", "AIC_version").split("-")[-1]
            flag = "ON" if info["use_hixl"] else "OFF"
            args.extend(
                [
                    f"-DSOC_VERSION={soc}",
                    f"-DASCEND_AICORE_ARCH={aicore}",
                    f"-DARCH={platform.machine()}",
                    "-DUSE_ASCEND=1",
                    "-DUSE_MINDSPORE=OFF",
                    f"-DUSE_HIXL={flag}",
                    f"-DUSE_HCOMM_ONESIDED={flag}",
                    f"-DP1_HOST_INSTALL_DIR={staging / primary}",
                ]
            )
            mooncake = os.environ.get("BUILD_MOONCAKE", "0")
            if mooncake not in ("0", "1"):
                raise RuntimeError("BUILD_MOONCAKE must be 0 or 1")
            info["build_mooncake"] = mooncake == "1"
            args.append(f"-DBUILD_MOONCAKE={'ON' if mooncake == '1' else 'OFF'}")
            for key in ("MOONCAKE_INCLUDE_DIR", "MOONCAKE_LIB_DIR"):
                if os.environ.get(key):
                    args.append(f"-D{key}={os.environ[key]}")
        subprocess.run(args, check=True)
        subprocess.run(
            ["cmake", "--build", str(build_dir), "--parallel", str(jobs), "--verbose"],
            check=True,
        )
        subprocess.run(["cmake", "--install", str(build_dir)], check=True)
        for namespace, patterns in required_artifacts(primary, info).items():
            for pattern in patterns:
                if not any(
                    path.is_file() for path in (staging / namespace).glob(pattern)
                ):
                    raise RuntimeError(
                        f"Required native artifact not installed: {namespace}/{pattern}"
                    )
        info["build_directory"] = str(build_dir)
        info["install_mode"] = "strict-editable" if self.editable_mode else "wheel"
        write_build_metadata(
            staging, primary, addon, self.distribution.get_version(), info
        )
        self.publish_outputs(staging)


def setup_arguments(primary: str) -> dict:
    addon = primary + "_ascend"
    module = "vllm_ascend_C" if primary == "vllm" else "c_ops"
    return {
        "packages": package_names(primary, addon),
        "package_dir": {primary: primary, addon: f"ascend/{addon}"},
        "install_requires": runtime_requirements(),
        "include_package_data": True,
        "package_data": {
            name: [
                "py.typed",
                "**/*.pyi",
                "**/*.json",
                "**/*.yaml",
                "**/*.yml",
                "**/*.jinja",
                "**/*.jinja2",
                "**/*.txt",
                "**/*.js",
                "**/*.css",
            ]
            for name in (primary, addon)
        },
        "ext_modules": [Extension(f"{addon}.{module}", sources=[])],
        "cmdclass": {
            "build": P1Build,
            "build_ext": P1BuildExt,
            "build_py": P1BuildPy,
            "editable_wheel": P1EditableWheel,
        },
    }
