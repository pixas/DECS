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
SC_SIZE="$4"

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip nodes_array: $nodes_array
srun bash -c 'echo $SLURMD_NODENAME-$SLURM_JOB_GPUS' # 打印出不同机器上分配的显卡编号


mkdir -p ${OUTPUT_PATH}

if [[ $MODEL == *"32b"* ]]; then 
    bs=2
else
    bs=4
fi



export VLLM_USE_V1=0

srun --jobid $SLURM_JOBID python -m baselines.alphaone.main  \
    --data_path ${DATA_PATH} \
    --output_path ${OUTPUT_PATH} \
    --model ${MODEL} \
    --batch $bs \
    --resume \
    --sample_num $SC_SIZE \
    --temperature 0.6 \
    --use_vllm ${@:5}