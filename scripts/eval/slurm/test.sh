#!/bin/bash
#SBATCH --partition=medai_llm
#SBATCH -N1
#SBATCH --quotatype=auto
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1    
#SBATCH --mem-per-cpu=6G  
#SBATCH --time=72:00:00
###SBATCH --kill-on-bad-exit=1



srun python utils/probe_param.py \
    --base_path /mnt/hwfile/medai/LLMModels/Model/Qwen2.5-Math-7B \
    --finetuned_path /mnt/hwfile/medai/LLMModels/Model/DeepSeek-R1-Distill-Qwen-7B