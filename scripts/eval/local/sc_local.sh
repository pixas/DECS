#!/usr/bin/env bash
set -euo pipefail

# Resolve repository root and run all relative paths from there.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/eval/local/sc_local.sh <model_name_or_path> [chunk_num] [sc_size] [tp_size] [extra_overrides...]

Env:
  DATA_ROOT   default: data
  OUTPUT_ROOT default: ./results
  DATASETS    default: math (supports whitespace or comma-separated list)
EOF
}

if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

DATA_ROOT="${DATA_ROOT:-data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./results}"
DATASETS_RAW="${DATASETS:-math}"

model_name_or_path="$1"
chunk_num="${2:-1}"
sc_size="${3:-16}"
tp_size="${4:-1}"

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
    datasets=("math")
fi

if [[ "${model_name_or_path,,}" == *"32b"* ]]; then
    bs=2
else
    bs=4
fi

extra_args=("--tp_size" "${tp_size}")
if [[ "$#" -gt 4 ]]; then
    for raw in "${@:5}"; do
        if [[ -n "${raw// }" ]]; then
            parts=()
            read -r -a parts <<< "${raw}"
            extra_args+=("${parts[@]}")
        fi
    done
fi

run_single_eval() {
    local data_path="$1"
    local output_path="$2"
    local log_file="$3"

    mkdir -p "${output_path}"
    python3 -u evaluation/test.py \
        --data_path "${data_path}" \
        --output_path "${output_path}" \
        --model_name_or_path "${model_name_or_path}" \
        --batch "${bs}" \
        --resume \
        --sample_num "${sc_size}" \
        --temperature 0.6 \
        --use_vllm \
        "${extra_args[@]}" 2>&1 | tee -a "${log_file}"
}

for dataset in "${datasets[@]}"; do
    dataset="$(echo "${dataset}" | xargs)"
    if [[ -z "${dataset}" ]]; then
        continue
    fi

    if [[ "${chunk_num}" != "1" ]]; then
        for ((i=0; i<chunk_num; i++)); do
            output_path="${OUTPUT_ROOT}/${dataset}/${model_name}_sc${sc_size}/chunk_${i}"
            log_file="${output_path}/infer.log"
            run_single_eval "${DATA_ROOT}/${dataset}/test.parquet" "${output_path}" "${log_file}"
            sleep 2
        done
    else
        output_path="${OUTPUT_ROOT}/${dataset}/${model_name}_sc${sc_size}"
        log_file="${output_path}/infer.log"
        run_single_eval "${DATA_ROOT}/${dataset}/test.parquet" "${output_path}" "${log_file}"
        sleep 2
    fi
done
