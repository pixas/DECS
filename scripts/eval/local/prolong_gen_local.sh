#!/usr/bin/env bash
set -euo pipefail

# Resolve repository root and run all relative paths from there.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/eval/local/prolong_gen_local.sh <model_name_or_path> [sc_size] [prolong_length] [extra_overrides...]

Env:
  OUTPUT_ROOT default: ./results
  DATASETS    default: "aime2024 aime2025 amc23" (supports whitespace or comma-separated list)
EOF
}

if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-./results}"
DATASETS_RAW="${DATASETS:-aime2024 aime2025 amc23}"

model_name_or_path="$1"
sc_size="${2:-16}"
prolong_length="${3:-32768}"

derive_model_name() {
    local path="$1"
    local model_name
    local last_part
    local second_last_part
    local global_step

    if [[ "${path}" == *"/"* ]]; then
        last_part="$(basename "${path}")"
        if [[ "${last_part}" == checkpoint* ]]; then
            second_last_part="$(basename "$(dirname "${path}")")"
            model_name="${second_last_part}-${last_part}"
        elif [[ "${last_part}" == actor* ]]; then
            global_step="$(basename "$(dirname "${path}")")"
            second_last_part="$(basename "$(dirname "$(dirname "${path}")")")"
            model_name="${second_last_part}_${global_step}"
        else
            model_name="${last_part}"
        fi
    else
        model_name="${path}"
    fi
    printf '%s\n' "${model_name}"
}

model_name="$(derive_model_name "${model_name_or_path}")"

datasets=()
if [[ "${DATASETS_RAW}" == *","* ]]; then
    IFS=',' read -r -a datasets <<< "${DATASETS_RAW}"
else
    read -r -a datasets <<< "${DATASETS_RAW}"
fi
if [[ "${#datasets[@]}" -eq 0 ]]; then
    datasets=("aime2024" "aime2025" "amc23")
fi

extra_args=()
if [[ "$#" -gt 3 ]]; then
    for raw in "${@:4}"; do
        if [[ -n "${raw// }" ]]; then
            parts=()
            read -r -a parts <<< "${raw}"
            extra_args+=("${parts[@]}")
        fi
    done
fi

for dataset in "${datasets[@]}"; do
    dataset="$(echo "${dataset}" | xargs)"
    if [[ -z "${dataset}" ]]; then
        continue
    fi

    input_dir="${OUTPUT_ROOT}/${dataset}/${model_name}_sc${sc_size}"
    output_dir="${OUTPUT_ROOT}/${dataset}/${model_name}_sc${sc_size}_prolong${prolong_length}"
    log_file="${output_dir}/infer.log"

    mkdir -p "${output_dir}"
    python3 -u evaluation/prolonged_gen.py \
        --data_path "${input_dir}" \
        --output_path "${output_dir}" \
        --model_name_or_path "${model_name_or_path}" \
        --temperature 0.6 \
        "${extra_args[@]}" 2>&1 | tee -a "${log_file}"

    sleep 2
done
