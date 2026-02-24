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
# DATASETS=(math aime2024 aime2025 amc23)
DATASETS=(math)
CHUNK_NUM=${2:-1}

for DATASET in "${DATASETS[@]}"; do 
    LOG_FILE=${OUTPUT_ROOT}/${DATASET}/${model_name}_cot/infer.log

    mkdir -p ${OUTPUT_ROOT}/${DATASET}/${model_name}_cot
    
    # if [ -f "${OUTPUT_ROOT}/${DATASET}/${model_name}_cot/result.json" ]; then
    #     echo "Skip ${DATASET} as it exists."
    #     continue
    # fi
    if  [[ $CHUNK_NUM != 1 ]]; then 
        for i in $(seq 0 $((CHUNK_NUM-1))); do
            OUTPUT_PATH=${OUTPUT_ROOT}/${DATASET}/${model_name}_cot/chunk_${i}
            mkdir -p $OUTPUT_PATH
            LOG_FILE=${OUTPUT_PATH}/infer.log
            sbatch -o $LOG_FILE ./scripts/eval/slurm/cot.sh \
                ${DATA_ROOT}/${DATASET}/test.parquet \
                ${OUTPUT_PATH} \
                ${model_name_or_path} "--prompt_type $prompt_type --chunk_num $CHUNK_NUM --chunk_idx $i" ${@:4}
            sleep 2
        done
    else
        LOG_FILE=${OUTPUT_ROOT}/${DATASET}/${model_name}_cot/infer.log
        if [ ! -f "${OUTPUT_ROOT}/${DATASET}/${model_name}_cot/cache.jsonl" ]; then
            job_id=$(sbatch -o $LOG_FILE ./scripts/eval/slurm/cot.sh \
                ${DATA_ROOT}/${DATASET}/test.parquet \
                ${OUTPUT_ROOT}/${DATASET}/${model_name}_cot \
                ${model_name_or_path} "${@:3}" | awk '{print $4}')
            echo "Submitted batch job $job_id"
            # sbatch -o ${OUTPUT_ROOT}/${DATASET}/${model_name}_cot/eval.log --dependency=afterok:"$job_id" ./scripts/eval/slurm/score.sh ${OUTPUT_ROOT}/${DATASET}/${model_name}_cot/cache.jsonl
        # else 
        #     sbatch -o ${OUTPUT_ROOT}/${DATASET}/${model_name}_cot/eval.log ./scripts/eval/slurm/score.sh ${OUTPUT_ROOT}/${DATASET}/${model_name}_cot/cache.jsonl
        fi
        
    fi
    sleep 2
done