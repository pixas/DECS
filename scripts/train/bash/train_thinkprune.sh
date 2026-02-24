LENGTH=4000
RUN_NAME=DeepSeek-R1-Distill-Qwen-1.5B-${LENGTH}
MODEL=/mnt/hwfile/medai/LLMModels/Model/DeepSeek-R1-Distill-Qwen-1.5B

N_GPUS=8
TP=1
MODEL_DIR=~/checkpoints/${RUN_NAME}
DATA_DIR=data/past_aime_amc/length${LENGTH}

BATCH_SIZE=64
ROLLOUT_BS=128
ROLLOUT_N=16

DATA_NAME=past_aime_amc
adv=grpo

other_configs="actor_rollout_ref.actor.use_kl_loss=True trainer.total_epochs=10 reward_model.reward_manager=thinkprune"
sbatch scripts/train/slurm/train_rl.sh $DATA_NAME $adv $MODEL $RUN_NAME $ROLLOUT_N 
