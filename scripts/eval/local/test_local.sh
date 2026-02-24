#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash scripts/eval/local/test_local.sh <model_path> <hf_model_path> [tp_size]
EOF
}

if [[ $# -lt 2 ]]; then
    usage
    exit 1
fi

MODEL_PATH="$1"
HF_MODEL_PATH="$2"
TP_SIZE="${3:-1}"

bash scripts/eval/local/sc_local.sh \
    "${MODEL_PATH}" \
    1 \
    16 \
    "${TP_SIZE}" \
    "--prompt_type instruct_default --max_new_tokens 16384 --hf_model_path ${HF_MODEL_PATH}"
