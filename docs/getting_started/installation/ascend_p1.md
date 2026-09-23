# P1 Ascend build and development install

This fork builds one `vllm` distribution containing `vllm` and `vllm_ascend`.
Use the root entry point; do not install the upstream Ascend plugin separately.
Native build/install validation must run in the intranet candidate environment.

## Prerequisites

Use a dedicated container, not an active baseline service: aarch64, Python 3.11,
Ascend910B3, CANN 8.5.1, torch 2.9.0, torch_npu
`2.9.0.post1+gitee7ba04`, triton-ascend `3.2.0.dev20260322`.
Prepare `requirements/build.txt` and `requirements/ascend.txt` from approved
materials without replacing the existing torch candidate.

Source the approved CANN `set_env.sh`. Set `ASCEND_HOME_PATH`,
`SOC_VERSION=ascend910b3`, `VLLM_TARGET_DEVICE=ascend`, `USE_MINDSPORE=0`,
`BUILD_WITH_HIP=0`, `COMPILE_CUSTOM_KERNELS=1`, `VLLM_USE_PRECOMPILED=0`
and a positive `MAX_JOBS` (start with 8). Remove old framework paths while
preserving the CANN SDK paths.

## Commands

Set `P1_REPO` to this checkout, `P1_MATERIAL` to a clean initialized CATLASS
checkout, and `P1_RUN` to a new result directory outside the Python packages.
CATLASS must be exactly `716fd7baa7fb7f6cac0488bb628fd1dd0e875641`.
An initialized `ascend/csrc/third_party/catlass` in this repo is also accepted.
The helper never fetches materials or installs dependencies.

```bash
cd "$P1_RUN"
python -B "$P1_REPO/p1_dev.py" materials --from-submodule "$P1_MATERIAL"
python -B "$P1_REPO/p1_dev.py" doctor --output "$P1_RUN/doctor.json"
python -B "$P1_REPO/tests/standalone/test_p1_development.py" -v
python -B "$P1_REPO/ascend/tests/standalone/test_p1_resources.py" -v
python -B "$P1_REPO/p1_dev.py" build --output "$P1_RUN/wheel"
python -B "$P1_REPO/p1_dev.py" install --isolated-env \
  --wheel "$P1_RUN/wheel/wheels/vllm-0.18.0+ascend.p1-cp311-cp311-linux_aarch64.whl" \
  --output "$P1_RUN/install"
python -B "$P1_REPO/p1_dev.py" verify --mode wheel --output "$P1_RUN/paths.json"
```

Alternatively, in a dedicated development container, replace the final three
commands with strict editable installation (the first install still compiles):

```bash
python -B "$P1_REPO/p1_dev.py" editable --isolated-env --output "$P1_RUN/editable"
python -B "$P1_REPO/p1_dev.py" verify --mode editable --output "$P1_RUN/paths.json"
```

`--isolated-env` is the operator's confirmation, not automatic isolation.
Old four-package distributions are rejected; stop dedicated debug processes
before reinstalling even the same P1 version. Install the companion LMCache
P1 repo too. Standard `pip install --no-index --no-deps --no-build-isolation
--config-settings editable_mode=strict -e PATH` is supported after preparation.

## Debugging and acceptance

Existing Python file changes take effect after restarting the debug process.
Reinstall after adding/renaming files or changing native code/dependencies.
Do not move the checkout or delete its `build/` while editable-installed:
the strict link tree uses retained `build/p1-native/run-*` artifacts.
Run inference outside both repository roots to avoid source shadowing.

Every compile uses a fresh native directory. Retry with a new `--output`;
do not remove the previous failure diagnostics. Add `--dry-run` to build,
install or editable to inspect commands without side effects. Preserve
`preflight.json`, `command.log`, `command-result.json` and artifact hashes.

`verify` checks paths/resources only, not ELF loading, ABI or NPU execution.
Formal P1 acceptance still requires regular wheels, independent sdist rebuild,
extension loading, and GLM-5.2 DSA/MTP/C8-off baseline comparisons in both
TP8/DP2 and TP4/DP4 layouts. See the separately delivered
`design/p1/intranet-next-steps.md` and `development-build-install.md`.
