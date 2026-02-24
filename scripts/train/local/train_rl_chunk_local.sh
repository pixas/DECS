#!/usr/bin/env bash
set -euo pipefail

# Resolve repository root and run all relative paths from there.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

# Anonymous defaults: provide values via env vars in your own environment.
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints}"
MODEL_NAME="${MODEL_NAME:-r1_distill_qwen1.5b}"

declare -A model_mapping=(
    
    ["r1_distill_qwen1.5b"]="DeepSeek-R1-Distill-Qwen-1.5B"
    ["r1_distill_qwen7b"]="DeepSeek-R1-Distill-Qwen-7B"
    ["r1_distill_llama8b"]="deepseek_r1_distill_llama8b_step_82"
    ["deepscaler_1.5b"]="DeepScaleR-1.5B-Preview"
    ["qwen34b"]="Qwen3-4B"

)

ckpt=""
for key in "${!model_mapping[@]}"; do
    if [[ "${MODEL_NAME}" == "${key}" ]]; then
        ckpt="${model_mapping[$key]}"
        break
    fi
done

if [[ -z "${ckpt}" ]]; then
    echo "Unsupported MODEL_NAME: ${MODEL_NAME}" >&2
    exit 1
fi

MODEL_PATH="${MODEL_PATH:-${CHECKPOINT_ROOT}/${ckpt}}"

data_name="${DATA_NAME:-deepscaler_mix}"
adv_estimator="${ADV_ESTIMATOR:-grpo_proc_length}"
save_name="${SAVE_NAME:-${MODEL_NAME}_${adv_estimator}_a001_b001_c001_n16_nozeroadv_fm0_invert_dapo_adp_lr02_16k}"
rollout_n="${ROLLOUT_N:-16}"

chunk_judge_model="${CHUNK_JUDGE_MODEL:-${CHECKPOINT_ROOT}/smart_split_1.5b_v2}"
chunk_judge_url="${CHUNK_JUDGE_URL:-127.0.0.1:10041}"

other_configs="${OTHER_CONFIGS:-data.is_base=False trainer.total_epochs=3 actor_rollout_ref.actor.clip_ratio_high=0.2 trainer.val_before_train=False data.max_response_length=16384 data.prompt_type=default actor_rollout_ref.rollout.max_num_batched_tokens=20000 reward_model.reward_manager=chunk reward_model.format_score=0}"
dynamic_configs="${DYNAMIC_CONFIGS:-chunk_config.enable=True chunk_config.judge_model=${chunk_judge_model} chunk_config.judge_url=${chunk_judge_url} chunk_config.ori_adv_factor=0 chunk_config.only_entropy_token=False chunk_config.only_minus_entropy_token=False}"
remote_split_configs="${REMOTE_SPLIT_CONFIGS:-chunk_config.use_entropy=False chunk_config.alpha=0.001 chunk_config.beta=0.001 chunk_config.gamma=0.001 chunk_config.filter_by_entropy=False chunk_config.only_right=True chunk_config.high_entropy_quantile=0.8 chunk_config.no_zero_adv=True chunk_config.as_return=False chunk_config.reward_func=constant}"
dapo_configs="${DAPO_CONFIGS:-algorithm.filter_groups.enable=True algorithm.filter_groups.max_num_gen_batches=10 algorithm.filter_groups.enable_kappa=True algorithm.filter_groups.kappa_lr=0.2 data.gen_batch_size=128}"

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

log_path="${LOG_PATH:-logs/${data_tag}/${save_name}}"
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

actor_ppo_max_token_len="${ACTOR_PPO_MAX_TOKEN_LEN:-20000}"
infer_ppo_max_token_len="${INFER_PPO_MAX_TOKEN_LEN:-80000}"

extra_args=()
for cfg in "${other_configs}" "${dynamic_configs}" "${remote_split_configs}" "${dapo_configs}"; do
    if [[ -n "${cfg// }" ]]; then
        cfg_parts=()
        read -r -a cfg_parts <<< "${cfg}"
        extra_args+=("${cfg_parts[@]}")
    fi
done
if [[ "$#" -gt 0 ]]; then
    extra_args+=("$@")
fi

python3 -u -m src_valid.main_ppo \
    algorithm.adv_estimator="${adv_estimator}" \
    data.train_files="${train_files}" \
    data.val_files="${test_files}" \
    data.train_batch_size=64 \
    data.max_prompt_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${actor_ppo_max_token_len}" \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${infer_ppo_max_token_len}" \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${infer_ppo_max_token_len}" \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n="${rollout_n}" \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.top_p=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.n=16 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_math' \
    trainer.experiment_name="${data_tag}_${save_name}" \
    trainer.n_gpus_per_node="${num_gpus}" \
    trainer.nnodes=1 \
    trainer.save_freq=1 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.max_critic_ckpt_to_keep=1 \
    trainer.test_freq=5 \
    trainer.log_val_generations=10 \
    "${extra_args[@]}" 2>&1 | tee -a "${log_path}/train.log"
