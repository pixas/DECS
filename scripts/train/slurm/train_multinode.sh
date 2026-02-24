#!/bin/bash
#SBATCH --partition=medai_llm_p
#SBATCH -N2
#SBATCH -n2
#SBATCH --quotatype=spot
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1    
#SBATCH --mem=900G
###SBATCH --kill-on-bad-exit=1
# load necessary modules

# replace these information with your own

# apptainer_image_path=/path/to/verl-ngc.sif
# replace these information with your own
# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
# 正确创建数组的方式
IFS=$'\n' read -d '' -r -a nodes_array <<< "$nodes"

head_node=${nodes_array[0]}
echo "Job started on $(date)"
echo "Head node: $head_node"
echo "All nodes: ${nodes_array[@]}"
echo "Number of nodes: ${#nodes_array[@]}"
# 打印时间


head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi

port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

# make sure we set environment variables before Ray initialization
# export VLLM_ATTENTION_BACKEND=XFORMERS
http_proxy=""
https_proxy=""
export http_proxy
export https_proxy

MEMORY_SIZE=100000000000
# 随机10000-10100之间的数
DASHBOARD_PORT=$(shuf -i 10000-10100 -n 1)
echo "${SLURM_CPUS_PER_TASK}"
echo "${SLURM_GPUS_PER_TASK}"
echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
        ray start --head --node-ip-address="$head_node_ip" --port=$port \
        --dashboard-port=$DASHBOARD_PORT \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_TASK}" --block --object-store-memory="${MEMORY_SIZE}" --dashboard-host=0.0.0.0 --min-worker-port 10101 --max-worker-port 19999 --temp-dir=/tmp/jiangshuyang  &
# optional, though may be useful in certain versions of Ray < 1.0.
sleep 10

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$ip_head" \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_TASK}" --block --object-store-memory="${MEMORY_SIZE}" --min-worker-port 10101 --max-worker-port 19999  &
        # apptainer run --nv --bind $verl_workdir $apptainer_image_path \
    sleep 5
done




data_name="$1"
adv_estimator="$2"

if [[ $adv_estimator == *"reinforce_plus_plus"* ]]; then 
    save_estimator=rpp
else 
    save_estimator=$adv_estimator
fi
model_name_or_path="$3"

# model_name=qwen253b_process_maxactor_total
# model_name=qwen253b_process_hardactor_step
save_name="$4"
rollout_n="$5"
entry_name="$6"

if [[ -z $entry_name ]]; then
    entry_name="verl.trainer.main_ppo"
fi

# check if data_name is separated by space, if so, obtain each part and save to a new list
if [[ $data_name == *" "* ]]; then
    IFS=' ' read -ra ADDR <<<"$data_name"
    data_name=${ADDR[0]}
    for i in "${ADDR[@]:1}"; do
        data_name="${data_name}_${i}"
    done
else 
    data_name=${data_name}
    ADDR=($data_name)
fi


train_files="["
# test_files="["
# add each train and test file to train_files and test_files 
for i in "${ADDR[@]}"; do
    cur_train_path=data/${i}/train.parquet
    # cur_test_path=data/${i}/test.parquet
    train_files="${train_files}'$cur_train_path',"
    # test_files="${test_files}'$cur_test_path',"
done
train_files="${train_files%?}]"

# always add aime2024 to test_files
aime_test_path=data/aime2024/test.parquet
# math_test_path=data/math/test.parquet
aime25_test_path=data/aime2025/test.parquet
amc_test_path=data/amc23/test.parquet
test_files="[${aime_test_path},${aime25_test_path},${amc_test_path}]"
# test_files="[${aime_test_path},${math_test_path},${aime25_test_path},${amc_test_path}]"



HYDRA_FULL_ERROR=1
RAY_ADDRESS="http://${head_node_ip}:${DASHBOARD_PORT}" ray job submit -- python3 -u -m ${entry_name} \
    algorithm.adv_estimator=$adv_estimator \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=128 \
    data.max_prompt_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$model_name_or_path \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=$rollout_n \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.n=16 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.9 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_math' \
    trainer.experiment_name="${data_name}_${save_name}" \
    trainer.n_gpus_per_node=$SLURM_GPUS_PER_TASK \
    trainer.nnodes=$SLURM_JOB_NUM_NODES \
    trainer.save_freq=5 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.max_critic_ckpt_to_keep=1 \
    trainer.test_freq=5 \
    trainer.log_val_generations=15  ${@:7} 