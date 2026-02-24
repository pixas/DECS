MODEL_NAME=qwen317b
# MODEL_NAME=llama3_8b_medical
# MODEL_NAME=r1_distill_qwen32b
# MODEL_NAME=phi4_mini
# MODEL_NAME=deepscaler_1.5b
MODEL_NAME=r1_distill_qwen1.5b
# MODEL_NAME=r1_distill_qwen7b
# MODEL_NAME=r1_distill_llama8b
# MODEL_NAME=medsins
# 声明一个关联数组来存储映射
declare -A model_mapping

# 添加 MODEL_NAME 到 ckpt 的映射
model_mapping=(
    ["llama3.2"]="Llama-3.2-3B-Instruct"
    ["llama3_8b_medical"]="Meta-Llama-3-8B-Instruct"
    ["llama3.1_8b"]="Meta-Llama-3.1-8B-Instruct-ysl"
    ["qwen2.5_3b"]="Qwen2.5-3B-Instruct"
    ["qwen2.5_math_7b_base"]="Qwen2.5-Math-7B"
    ["qwen2.5_3b_base"]="Qwen2.5-3B"
    ["qwen2_7b"]="Qwen2-7B-Instruct"
    ["r1_distill_qwen1.5b"]="DeepSeek-R1-Distill-Qwen-1.5B"
    ["r1_distill_qwen7b"]="DeepSeek-R1-Distill-Qwen-7B"
    # ["r1_distill_llama8b"]="DeepSeek-R1-Distill-Llama-8B"
    ["r1_distill_llama8b"]="deepseek_r1_distill_llama8b_step_82"
    ["r1_distill_qwen32b"]="DeepSeek-R1-Distill-Qwen-32B"
    ["deepscaler_1.5b"]="DeepScaleR-1.5B-Preview"
    ["phi4_mini"]="Phi-4-mini-reasoning"
    ["medsins"]="MMedS-Llama3-3-8B"
    ["qwen34b"]="deepscaler_qwen34b_global_step_163"
    ["qwen317b"]="Qwen3-1.7B"
)


# 初始化 ckpt 变量
ckpt=""

# 遍历映射，检查 MODEL_NAME 是否包含某个键
for key in "${!model_mapping[@]}"; do
    if [[ $MODEL_NAME == "$key" ]]; then
        ckpt=${model_mapping[$key]}
        break
    fi
done

MODEL_PATH=/mnt/petrelfs/jiangshuyang/checkpoints/${ckpt}
# MODEL_PATH=/mnt/petrelfs/jiangshuyang/checkpoints/deepscaler_mix_r1_distill_qwen1.5b_grpo_proc_length_a001_b001_c001_n16_nozeroadv_fm0_2_step250

data_name="deepscaler_mix"
# adv=grpo_hybrid
adv=grpo_proc_length
# save_name=${MODEL_NAME}_${adv}_a045_b03_c045_fm1_oriadv1_onlyminusentropy
# save_name=${MODEL_NAME}_${adv}_a001_b001_c001_fm1_filter_80highentropy_n16_onlyright
save_name=${MODEL_NAME}_${adv}_a001_b001_c001_n16_nozeroadv_fm0_invert_dapo_adp_lr02_16k_debug
# save_name=${MODEL_NAME}_grpo_chunk_a06_b04_c08_normal
# save_name=${MODEL_NAME}_${adv}_a001_b001_c001_onlyentropy_fm0_allow_incorrect

other_configs="data.is_base=False trainer.total_epochs=3 actor_rollout_ref.actor.clip_ratio_high=0.2 trainer.val_before_train=False data.max_response_length=16384 data.prompt_type=default actor_rollout_ref.rollout.max_num_batched_tokens=20000 reward_model.reward_manager=chunk reward_model.format_score=0 "
# train 4card 
# replace space to underscore in data_name
if [[ $data_name == *" "* ]]; then
    IFS=' ' read -ra ADDR <<<"$data_name"
    save_data_name=${ADDR[0]} 
    for i in "${ADDR[@]:1}"; do
        save_data_name="${save_data_name}_${i}"
    done
else 
    save_data_name=${data_name}
    ADDR=($data_name)
fi

log_path=logs/${save_data_name}/${save_name}
mkdir -p ${log_path}

# rollout_n=32
# dynamic_configs="actor_rollout_ref.reflect_think.enable=True data.weighted=True actor_rollout_ref.reflect_think.reflect_n=16 actor_rollout_ref.reflect_think.reflect_bonus=0.1 actor_rollout_ref.reflect_think.reflect_type=replace actor_rollout_ref.reflect_think.adjust_old_logprobs=0 actor_rollout_ref.reflect_think.dynamic_bonus=True actor_rollout_ref.reflect_think.both_correct_bonus=0.1"
# dynamic_configs="actor_rollout_ref.reflect_think.enable=True actor_rollout_ref.reflect_think.mask_manual=False actor_rollout_ref.reflect_think.reflect_n=8 actor_rollout_ref.reflect_think.short_type=last actor_rollout_ref.reflect_think.dynamic_bonus=True actor_rollout_ref.reflect_think.both_correct_bonus=0.1 actor_rollout_ref.reflect_think.reflect_type=replace trainer.total_steps="
# obtain the below script's jobid
# sbatch -o $log_path/train.log scripts/train/slurm/train_rl.sh "${data_name}" $adv $MODEL_PATH $save_name ${rollout_n} "${other_configs}" "${dynamic_configs}"

rollout_n=16
dynamic_configs="chunk_config.enable=True chunk_config.judge_model=/mnt/petrelfs/jiangshuyang/checkpoints/smart_split_1.5b_v2 chunk_config.judge_url=10.140.54.40:10041 chunk_config.ori_adv_factor=0 chunk_config.only_entropy_token=False chunk_config.only_minus_entropy_token=False"

# remote_split_configs="confidence_reward.split_model=/mnt/petrelfs/jiangshuyang/checkpoints/smart_split_1.5b_v2 confidence_reward.smart_split_url=10.140.54.16:10042"
remote_split_configs=" chunk_config.use_entropy=False chunk_config.alpha=0.001 chunk_config.beta=0.001 chunk_config.gamma=0.001  chunk_config.filter_by_entropy=False chunk_config.only_right=True chunk_config.high_entropy_quantile=0.8 chunk_config.no_zero_adv=True chunk_config.as_return=False chunk_config.reward_func=constant"



dapo_configs=" algorithm.filter_groups.enable=True algorithm.filter_groups.max_num_gen_batches=10 algorithm.filter_groups.enable_kappa=True algorithm.filter_groups.kappa_lr=0.2 data.gen_batch_size=128"
# chunk_config.high_token_weight=0 chunk_config.skip_last_chunk=True
# remote_split_configs=" chunk_config.use_entropy=False chunk_config.alpha=0.16 chunk_config.beta=0.12 chunk_config.gamma=0.1 chunk_config.force=0.1"
# dynamic_configs="actor_rollout_ref.reflect_think.enable=True actor_rollout_ref.reflect_think.mask_manual=False actor_rollout_ref.reflect_think.reflect_n=8 actor_rollout_ref.reflect_think.short_type=last actor_rollout_ref.reflect_think.dynamic_bonus=True actor_rollout_ref.reflect_think.both_correct_bonus=0.1 actor_rollout_ref.reflect_think.reflect_type=replace trainer.total_steps=40"
# obtain the below script's jobid
# sbatch -o $log_path/train.log scripts/train/slurm/train_rl.sh "${data_name}" $adv $MODEL_PATH $save_name ${rollout_n} "${other_configs}" "${dynamic_configs}" "${remote_split_configs}"
# resume_configs="trainer.resume_mode=resume_path trainer.resume_from_path=s3://syj_new/checkpoints/verl_math/deepscaler_mix_r1_distill_qwen1.5b_grpo_proc_length_a001_b001_c001_n16_nozeroadv_fm0_2/global_step_258"

sbatch -o $log_path/train.log scripts/train/slurm/train_rl_chunk.sh "${data_name}" $adv $MODEL_PATH $save_name ${rollout_n} "${other_configs}" "${dynamic_configs}" "${remote_split_configs}" "${dapo_configs}"

# sbatch -o $log_path/train.log scripts/train/slurm/train_rl_chunk.sh "${data_name}" $adv $MODEL_PATH $save_name ${rollout_n} "${other_configs}" "${dynamic_configs}" "${remote_split_configs}"
# sbatch -o $log_path/train.log scripts/train/slurm/train_singlenode_8card.sh "${data_name}" $adv $MODEL_PATH $save_name ${rollout_n} "src_valid.main_ppo" "${other_configs}" "${dynamic_configs}" "${remote_split_configs}" 
# sbatch -o $log_path/train.log scripts/train/slurm/train_multinode_4card.sh "${data_name}" $adv $MODEL_PATH $save_name ${rollout_n} "src_valid.main_ppo" "${other_configs}" "${dynamic_configs}"

# bash scripts/train/slurm/train_rl_debug.sh "${data_name}" $adv $MODEL_PATH $save_name ${rollout_n} "${other_configs}" "${dynamic_configs}"
# exit
# job_id=$(sbatch -o $log_path/train.log scripts/train/slurm/train_4card.sh "${data_name}" $adv $MODEL_PATH $save_name "${other_configs}" | awk '{print $4}')

# echo "Submitted batch job $job_id"

# obtain the job id of the above sbatch job

# submit a job dependent on the above job
# another_configs="reward_model.overlong_buffer.enable=True reward_model.overlong_buffer.len=128 actor_rollout_ref.actor.clip_ratio_high=0.28 data.is_base=True data.max_response_length=6400 trainer.total_epochs=1 trainer.val_before_train=False"


# process_configs="process_reward.enable=True process_reward.step_type=9 process_reward.step_reward=True process_reward.guide_prompt=order process_reward.warmup_ratio=0 process_reward.filter=True process_reward.as_reward=False"

# sbatch -o $log_path/train.log  scripts/train/slurm/train_4card.sh "${data_name}" $adv $MODEL_PATH "${save_name}" "${another_configs}" "${process_configs}"
# bash  scripts/train/slurm/train_4card.sh "${data_name}" $adv $MODEL_PATH "${save_name}" "${another_configs}" "${process_configs}"
# sbatch -o $log_path/train_long.log  scripts/train/slurm/train_4card.sh "${data_name}" $adv $MODEL_PATH $save_name "${another_configs}"
# submit a dependency job
# sbatch -o $log_path/train_long.log --dependency=afterok:"$job_id" scripts/train/slurm/train_4card.sh "${data_name}" $adv $MODEL_PATH $save_name "${another_configs}"




# sbatch -o $log_path/train_long.log  scripts/train/slurm/train_8card.sh "${data_name}" $adv $MODEL_PATH $save_name "${another_configs}"