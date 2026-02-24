
MODEL_PATH="$1"
HF_MODEL_PATH="$2"
TP_SIZE="${3:-1}"

bash scripts/eval/bash/sc.sh $MODEL_PATH 1 16 $TP_SIZE "--prompt_type instruct_default --max_new_tokens 16384 --hf_model_path ${HF_MODEL_PATH}"
