DATA_ROOT=data/
# DATA_ROOT=/mnt/hwfile/medai/LLMModels/MedicalDatas/train
OUTPUT_ROOT=./results

model_name_or_path="$1"
# 判断是否包含 / 字符
if [[ $model_name_or_path == *"/"* ]]; then
    # 提取最后一个 / 后的内容
    last_part=$(basename "$model_name_or_path")
    
    # 判断最后一个部分是否以 "checkpoint" 开头
    if [[ $last_part == "checkpoint"* ]]; then
        # 提取倒数第二个 / 和最后一个 / 之间的内容
        second_last_part=$(basename "$(dirname "$model_name_or_path")")
        
        # 组合成新的 model_name
        model_name="${second_last_part}-${last_part}"
    else
        # 如果不以 "checkpoint" 开头，直接使用最后一个部分
        # 检查最后一个是不是以global_step开头，如果是，那么取倒数第三个
        # 否则直接使用最后一个
        if [[ $last_part == "actor"* ]]; then
            # 提取倒数第三个 / 和最后一个 / 之间的内容
            global_step=$(basename "$(dirname "$model_name_or_path")")
            second_last_part=$(basename "$(dirname "$(dirname "$model_name_or_path")")")
            model_name="${second_last_part}_${global_step}"
        else
            # 直接使用最后一个部分
            model_name="$last_part"
        fi
    fi
else
    # 如果不包含 /，直接使用原字符串
    model_name="$model_name_or_path"
fi
# DATASETS=(math aime2024 aime2025 amc23 olympiad_bench gpqa_diamond)
DATASETS=(aime2024 aime2025 amc23)
SC_SIZE=${2:-16}
PROLONG_LENGTH=${3:-32768}

for DATASET in "${DATASETS[@]}"; do 
    INPUT_DIR=${OUTPUT_ROOT}/${DATASET}/${model_name}_sc${SC_SIZE}
    OUTPUT_DIR=${OUTPUT_ROOT}/${DATASET}/${model_name}_sc${SC_SIZE}_prolong${PROLONG_LENGTH}

    mkdir -p $OUTPUT_DIR
  
    LOG_FILE=$OUTPUT_DIR/infer.log
    # if [ ! -f "${OUTPUT_ROOT}/${DATASET}/${model_name}_sc${SC_SIZE}/cache.jsonl" ]; then
    job_id=$(sbatch -o $LOG_FILE ./scripts/eval/slurm/prolong_gen.sh \
        $INPUT_DIR \
        $OUTPUT_DIR \
        ${model_name_or_path} "${@:4}" | awk '{print $4}')
    echo "Submitted batch job $job_id"


    sleep 2
done