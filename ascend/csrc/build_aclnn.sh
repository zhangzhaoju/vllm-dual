#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build only in a fresh private source copy; never fetch or modify Git config.
set -euo pipefail
ROOT_DIR=$(realpath -- "$1")
SOC_VERSION=$2
[[ "$SOC_VERSION" == "ascend910b3" ]] || {
    echo "P1 custom ops require SOC_VERSION=ascend910b3" >&2
    exit 2
}
CATLASS_PATH="$ROOT_DIR/csrc/third_party/catlass/include"
[[ -d "$CATLASS_PATH" ]] || {
    echo "Missing pinned CATLASS headers; prepare material before building" >&2
    exit 2
}
[[ ! -e "$ROOT_DIR/csrc/build" && ! -e "$ROOT_DIR/csrc/output" ]] || {
    echo "Use a fresh ACLNN build source copy" >&2
    exit 2
}
export CPATH="$CATLASS_PATH${CPATH:+:$CPATH}"
CUSTOM_OPS="moe_grouped_matmul;grouped_matmul_swiglu_quant_weight_nz_tensor_list;lightning_indexer_vllm;sparse_flash_attention;matmul_allreduce_add_rmsnorm;moe_init_routing_custom;moe_gating_top_k;add_rms_norm_bias;apply_top_k_top_p_custom;transpose_kv_cache_by_block;copy_and_expand_eagle_inputs;causal_conv1d;lightning_indexer_quant;"
cd "$ROOT_DIR/csrc"
bash build.sh -n "$CUSTOM_OPS" -c ascend910b
shopt -s nullglob
installers=(output/CANN-custom_ops*.run)
[[ ${#installers[@]} -eq 1 ]] || {
    echo "Expected exactly one freshly built custom-op installer" >&2
    exit 2
}
bash "${installers[0]}" --install-path="$ROOT_DIR/vllm_ascend/_cann_ops_custom"
