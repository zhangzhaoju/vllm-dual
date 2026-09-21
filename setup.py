# SPDX-License-Identifier: Apache-2.0
"""Single P1 build entry: Ascend extensions and both temporary namespaces."""

import sys
from pathlib import Path

from setuptools import setup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p1_build import setup_arguments

setup(**setup_arguments("vllm"))
