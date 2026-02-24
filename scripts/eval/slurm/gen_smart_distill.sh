#!/bin/bash
#SBATCH --partition=medai_llm_p
#SBATCH -N1
#SBATCH --quotatype=auto
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1    
#SBATCH --mem-per-cpu=6G  
#SBATCH --time=72:00:00
###SBATCH --kill-on-bad-exit=1



NUM_CHUNKS="$1"
CHUNK_IDX="$2"

SAVE_PATH=/mnt/petrelfs/jiangshuyang/datasets/smart_split_disllation_data_${NUM_CHUNKS}_${CHUNK_IDX}.jsonl

srun python -u -m utils.smart_loc_distill \
    --save_file $SAVE_PATH \
    --num_chunks $NUM_CHUNKS \
    --chunk_idx $CHUNK_IDX 