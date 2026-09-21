# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parse the P1 CMake entry without configuring, compiling or probing devices.

Run directly with ``python -B tests/standalone/test_p1_cmake.py -v``.
Only CMake is required; project/package/subdirectory commands are test doubles.
"""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT_ENTRY = Path(__file__).resolve().parents[2] / "CMakeLists.txt"
PARSE_HARNESS = """
cmake_minimum_required(VERSION 3.26.1)

# Never enable languages, search dependencies or enter the native build here.
function(project)
  if(NOT "${ARGV}" STREQUAL "vllm_ascend_unified;LANGUAGES;C;CXX")
    message(FATAL_ERROR "Unexpected root project: ${ARGV}")
  endif()
  set(P1_PROJECT_SEEN TRUE PARENT_SCOPE)
endfunction()

function(find_package)
  if(NOT "${ARGV}" STREQUAL
      "Python3;COMPONENTS;Interpreter;Development.Module;REQUIRED")
    message(FATAL_ERROR "Unexpected root dependency: ${ARGV}")
  endif()
  set(P1_PYTHON_SEEN TRUE PARENT_SCOPE)
endfunction()

function(add_compile_definitions)
  set(P1_ABI_DEFINITION "${ARGV}" PARENT_SCOPE)
endfunction()

function(add_subdirectory)
  if(NOT "${ARGV}" STREQUAL "ascend" OR P1_ASCEND_SEEN)
    message(FATAL_ERROR "Expected exactly one Ascend subdirectory: ${ARGV}")
  endif()
  set(P1_ASCEND_SEEN TRUE PARENT_SCOPE)
endfunction()

function(execute_process)
  message(FATAL_ERROR "Subprocesses are forbidden in the parser-only test")
endfunction()

function(enable_language)
  message(FATAL_ERROR "Compiler probes are forbidden in the parser-only test")
endfunction()

include("${P1_ROOT_ENTRY}")
if(NOT P1_PROJECT_SEEN OR NOT P1_PYTHON_SEEN OR NOT P1_ASCEND_SEEN)
  message(FATAL_ERROR "Incomplete unified build entry")
endif()
if(NOT "${P1_ABI_DEFINITION}" STREQUAL
    "_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_USE_CXX11_ABI}")
  message(FATAL_ERROR "The supplied torch ABI was not forwarded")
endif()
if(NOT "${CMAKE_CXX_STANDARD}" STREQUAL "17" OR
    NOT CMAKE_CXX_STANDARD_REQUIRED OR NOT PYBIND11_FINDPYTHON)
  message(FATAL_ERROR "Missing C++/Python build settings")
endif()
message(STATUS "P1 CMake entry parsed without native configuration")
"""


class P1CMakeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="p1-cmake-parse-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.harness = self.directory / "parse.cmake"
        self.harness.write_text(PARSE_HARNESS, encoding="utf-8")

    def parse_entry(self, abi=None, entry=ROOT_ENTRY):
        args = ["cmake", f"-DP1_ROOT_ENTRY={entry}"]
        if abi is not None:
            args.append(f"-DGLIBCXX_USE_CXX11_ABI={abi}")
        result = subprocess.run(
            [*args, "-P", str(self.harness)],
            cwd=self.directory,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        self.assertFalse((self.directory / "CMakeCache.txt").exists())
        self.assertFalse((self.directory / "CMakeFiles").exists())
        return result

    def test_root_entry_parses_and_preserves_torch_abi_zero(self):
        result = self.parse_entry(abi=0)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_root_entry_parses_and_preserves_torch_abi_one(self):
        result = self.parse_entry(abi=1)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_root_entry_rejects_missing_torch_abi(self):
        result = self.parse_entry()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Pass the ABI of the installed torch", result.stderr)

    def test_root_entry_has_no_gpu_build_tail(self):
        source = ROOT_ENTRY.read_text(encoding="utf-8")
        self.assertNotRegex(source, r"(?i)CUDA|HIP|ROCM|MARLIN|CUTLASS|VLLM_GPU")
        self.assertTrue(source.rstrip().endswith("add_subdirectory(ascend)"))

    def test_parser_rejects_an_unmatched_endif(self):
        malformed = self.directory / "malformed.cmake"
        malformed.write_text(
            ROOT_ENTRY.read_text(encoding="utf-8") + "\nendif()\n",
            encoding="utf-8",
        )
        result = self.parse_entry(abi=1, entry=malformed)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Flow control statements are not properly nested", result.stderr)


if __name__ == "__main__":
    unittest.main()
