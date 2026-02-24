#!/bin/bash
#SBATCH --partition=medai_llm
#SBATCH -N1
#SBATCH --quotatype=auto
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1    
#SBATCH --mem-per-cpu=6G  
#SBATCH --time=72:00:00
###SBATCH --kill-on-bad-exit=1

DATA_PATH="$1"
OUTPUT_PATH="$2"
MODEL="$3"

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip nodes_array: $nodes_array
srun bash -c 'echo $SLURMD_NODENAME-$SLURM_JOB_GPUS' # 打印出不同机器上分配的显卡编号


mkdir -p ${OUTPUT_PATH}

if [[ $MODEL == *"32b"* ]]; then 
    bs=4
else
    bs=16
fi



# srun python3 -m verl.trainer.main_generation \
#     trainer.nnodes=1 \
#     trainer.n_gpus_per_node=1 \
#     data.path=$DATA_PATH \
#     data.prompt_key=prompt \
#     data.n_samples=1 \
#     data.output_path=$OUTPUT_PATH/cache.jsonl \
#     model.path=$MODEL \
#     +model.trust_remote_code=True \
#     rollout.temperature=0 \
#     rollout.top_k=-1 \
#     rollout.top_p=1 \
#     rollout.enforce_eager=False \
#     rollout.free_cache_engine=False \
#     rollout.max_num_batched_tokens=18000 \
#     rollout.prompt_length=1600 \
#     rollout.response_length=16384 \
#     rollout.tensor_model_parallel_size=1 \
#     rollout.gpu_memory_utilization=0.8 ${@:4}


srun --jobid $SLURM_JOBID python evaluation/test.py \
    --data_path ${DATA_PATH} \
    --output_path ${OUTPUT_PATH} \
    --model ${MODEL} \
    --batch $bs \
    --resume \
    --temperature 0.0 \
    --use_vllm ${@:4}