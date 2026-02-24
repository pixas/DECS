# Tested with 1 & 4 GPUs
set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: run_gen_qwen05.sh <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

nproc_per_node=$1
save_path=$2
infer_tp=${3:-2}  # Default tensor parallel size to 2

# Shift the arguments so $@ refers to the rest
shift 2
# export VLLM_ATTENTION_BACKEND=XFORMERS
srun python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=$nproc_per_node \
    data.path=data/aime2024/test.parquet \
    data.prompt_key=prompt \
    data.n_samples=2 \
    data.output_path=$save_path \
    model.path=/mnt/hwfile/medai/LLMModels/Model/Qwen2.5-3B-Instruct \
    +model.trust_remote_code=True \
    rollout.temperature=1.0 \
    rollout.top_p=0.7 \
    rollout.prompt_length=1600 \
    rollout.response_length=6400 \
    rollout.tensor_model_parallel_size=$infer_tp \
    rollout.gpu_memory_utilization=0.6
