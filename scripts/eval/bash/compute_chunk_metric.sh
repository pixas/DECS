tokenizer=/mnt/petrelfs/jiangshuyang/checkpoints/DeepSeek-R1-Distill-Qwen-7B
dataset=(aime2024 aime2025 amc23)
# ckpts=("LCR1_1.5B_sc16" "DeepSeek-R1-Distill-Qwen-1.5B-thinkprune-iter2k_sc16" "deepscaler_mix_r1_distill_qwen1.5b_rloo_tlmre_global_step_179_sc16" "r1distill-qwen-1.5b_sc16")
ckpts=("deepscaler_7b_r1_distill_qwen7b_grpo_proc_length_a001_b001_c001_n16_nozeroadv_fm0_constant_dapo_adp_global_step_265_sc16")
ckpts=("Phi-4-mini-reasoning_sc16" "Qwen3-4B_sc128")


for d in "${dataset[@]}"; do
    for ckpt in "${ckpts[@]}"; do
        data_path=/mnt/petrelfs/jiangshuyang/repo/new/ReflectPRM/results/${d}/${ckpt}/cache.jsonl
        echo "Processing dataset: $d, checkpoint: $ckpt"
        srun -p medai_llm_p --cpus-per-task=8 --quotatype=spot python show_reasoning_behavior.py \
            --data_path $data_path \
            --tokenizer $tokenizer 

        sleep 2
    done
done
