#!/usr/bin/env bash
set -euo pipefail

# Resolve repository root and run all relative paths from there.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

# Anonymous defaults: provide values via env vars in your own environment.
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints}"
LENGTH="${LENGTH:-4000}"
RUN_NAME="${RUN_NAME:-DeepSeek-R1-Distill-Qwen-1.5B-${LENGTH}}"
MODEL_PATH="${MODEL_PATH:-${CHECKPOINT_ROOT}/DeepSeek-R1-Distill-Qwen-1.5B}"

data_name="${DATA_NAME:-past_aime_amc}"
adv_estimator="${ADV_ESTIMATOR:-grpo}"
rollout_n="${ROLLOUT_N:-16}"
entry_module="${ENTRY_MODULE:-src_valid.main_ppo}"

other_configs="${OTHER_CONFIGS:-actor_rollout_ref.actor.use_kl_loss=True trainer.total_epochs=10 reward_model.reward_manager=thinkprune}"

if [[ "${data_name}" == *" "* ]]; then
    IFS=' ' read -r -a datasets <<< "${data_name}"
    data_tag="${datasets[0]}"
    for item in "${datasets[@]:1}"; do
        data_tag="${data_tag}_${item}"
    done
else
    data_tag="${data_name}"
    datasets=("${data_name}")
fi

log_path="${LOG_PATH:-logs/${data_tag}/${RUN_NAME}}"
mkdir -p "${log_path}"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    num_gpus="$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")"
else
    num_gpus="$(nvidia-smi -L | wc -l | tr -d ' ')"
fi

if [[ "${num_gpus}" -lt 1 ]]; then
    echo "No GPU detected. Set CUDA_VISIBLE_DEVICES or check nvidia-smi." >&2
    exit 1
fi

train_files="["
for item in "${datasets[@]}"; do
    cur_train_path="data/${item}/train.parquet"
    train_files="${train_files}'${cur_train_path}',"
done
train_files="${train_files%?}]"

aime_test_path="data/aime2024/test.parquet"
aime25_test_path="data/aime2025/test.parquet"
amc_test_path="data/amc23/test.parquet"
test_files="[${aime_test_path},${aime25_test_path},${amc_test_path}]"

extra_args=()
if [[ -n "${other_configs// }" ]]; then
    cfg_parts=()
    read -r -a cfg_parts <<< "${other_configs}"
    extra_args+=("${cfg_parts[@]}")
fi
if [[ "$#" -gt 0 ]]; then
    extra_args+=("$@")
fi

python3 -u -m "${entry_module}" \
    algorithm.adv_estimator="${adv_estimator}" \
    data.train_files="${train_files}" \
    data.val_files="${test_files}" \
    data.train_batch_size=128 \
    data.max_prompt_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n="${rollout_n}" \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.top_p=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.n=16 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    critic.optim.lr=5e-6 \
    critic.model.use_remove_padding=True \
    critic.optim.lr_warmup_steps_ratio=0.01 \
    critic.model.path="${MODEL_PATH}" \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=8 \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_math' \
    trainer.experiment_name="${data_tag}_${RUN_NAME}" \
    trainer.n_gpus_per_node="${num_gpus}" \
    trainer.nnodes=1 \
    trainer.save_freq=1 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.max_critic_ckpt_to_keep=1 \
    trainer.test_freq=5 \
    trainer.log_val_generations=10 \
    "${extra_args[@]}" 2>&1 | tee -a "${log_path}/train.log"
