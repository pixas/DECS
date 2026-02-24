#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash scripts/eval/local/prolong_gen_test_local.sh <model_path> <hf_model_path> [sc_size] [prolong_length]
EOF
}

if [[ $# -lt 2 ]]; then
    usage
    exit 1
fi

MODEL_PATH="$1"
HF_MODEL_PATH="$2"
SC_SIZE="${3:-128}"
PROLONG_LENGTH="${4:-32768}"

bash scripts/eval/local/prolong_gen_local.sh \
    "${MODEL_PATH}" \
    "${SC_SIZE}" \
    "${PROLONG_LENGTH}" \
    "--max_new_tokens ${PROLONG_LENGTH} --hf_model_path ${HF_MODEL_PATH}"
