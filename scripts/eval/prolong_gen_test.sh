# MODEL_PATH=qwen2.5-3b
# MODEL_PATH=/mnt/petrelfs/jiangshuyang/checkpoints/
# MODEL_PATH=r1distill-qwen-1.5b
# MODEL_PATH=r1distill-qwen-7b
# MODEL_PATH=s3://syj_test/checkpoints/verl_math/deepscaler_r1_distill_qwen1.5b_grpo_new/global_step_314/actor
# MODEL_PATH=/mnt/petrelfs/jiangshuyang/checkpoints/DeepSeek-R1-Distill-Qwen-7B_best05
# MODEL_PATH=/mnt/petrelfs/jiangshuyang/checkpoints/deepscaler_r1_distill_qwen1.5b_reinforce_plus_plus_new_nomask_onlylast_global_step_80
MODEL_PATH="$1"
# MODEL_PATH=/mnt/petrelfs/jiangshuyang/checkpoints/deepscaler_r1_distill_qwen1.5b_reinforce_plus_plus_new_nomask_onlylast_highpenalty_global_step_40
# MODEL_PATH=/mnt/petrelfs/jiangshuyang/checkpoints/tmp_af9db1b123386e68
# MODEL_PATH=/mnt/petrelfs/jiangshuyang.p/checkpoints/verl_math/gsm8k_math_qwen2.5_3b_reinforce_plus_plus/global_step_210/actor
# MODEL_PATH=/mnt/petrelfs/jiangshuyang.p/checkpoints/verl_math/gsm8k_math_qwen2.5_3b_base_reinforce_plus_plus/global_step_210/actor
# MODEL_PATH=/mnt/petrelfs/jiangshuyang.p/checkpoints/verl_math/orz_original_qwen2.5_3b_base_reinforce_plus_plus_mathpt_process_asreturn/global_step_222/actor


# bash scripts/eval/bash/prolong_gen.sh $MODEL_PATH 128 "--prompt_type instruct_default --max_new_tokens 16384"
# bash scripts/eval/bash/prolong_gen.sh $MODEL_PATH 128 32768 "--max_new_tokens 32768 --hf_model_path /mnt/petrelfs/jiangshuyang/checkpoints/DeepSeek-R1-Distill-Qwen-1.5B"

bash scripts/eval/bash/prolong_gen.sh $MODEL_PATH 128 32768 "--max_new_tokens 32768 --hf_model_path /mnt/petrelfs/jiangshuyang/checkpoints/DeepSeek-R1-Distill-Qwen-7B"


# results