# DECS
Official codebase for the ICLR 2026 Oral paper:
**Overthinking Reduction with Decoupled Rewards and Curriculum Data Scheduling**

🤗 [Hugging Face (DECS 1.5B release target)](https://huggingface.co/pixas/DECS_1.5B)  
🤗 [Hugging Face (DECS 7B release target)](https://huggingface.co/pixas/DECS_7B)  
📄 [Paper (arXiv:2509.25827)](https://arxiv.org/abs/2509.25827)

## Project Overview
DECS is a training and evaluation framework for reasoning models, with two core ideas:

- **Decoupled Rewards** to reduce inefficient overthinking behaviors during RL training.
- **Curriculum Data Scheduling** to improve stability and generalization by progressively controlling training data difficulty.



## Repository Structure (Reproducibility-Relevant)
- `scripts/train/local/train_rl_chunk_local.sh`: main DECS RL training script (chunk reward + decoupled configs)
- `scripts/train/local/train_thinkprune_local.sh`: thinkprune training script
- `scripts/eval/local/sc_local.sh`: standard self-consistency inference/evaluation
- `scripts/eval/local/prolong_gen_local.sh`: prolonged generation for long trajectories
- `scripts/eval/local/test_local.sh`: quick wrapper for `sc_local.sh`
- `scripts/eval/local/prolong_gen_test_local.sh`: quick wrapper for `prolong_gen_local.sh`


## Environment Setup


First create a virtual environment:

```bash
conda create -n verl python=3.10
conda activate verl
```

Then install `vllm==0.8.5.post1` via 
```bash
export VLLM_VERSION=0.8.5.post1
export CUDA_VERSION=126 # or other
export CPU_ARCH=$(uname -m) # x86_64 or aarch64
uv pip install https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu${CUDA_VERSION}-cp38-abi3-manylinux_2_35_${CPU_ARCH}.whl --extra-index-url https://download.pytorch.org/whl/cu${CUDA_VERSION}
```

After that, install verl via 
```
pip install -e .
```

## Data Layout
Training scripts expect `data/<dataset>/train.parquet`, and evaluation scripts expect `data/<dataset>/test.parquet`.

Minimal example:

```text
data/
  deepscaler/
    train.parquet
  aime2024/
    test.parquet
  aime2025/
    test.parquet
  amc23/
    test.parquet
  math/
    test.parquet
  
```

## End-to-End Pipeline: Training to Inference

### Step 1. Set Basic Environment Variables

```bash
# Select GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Root directory for model checkpoints
export CHECKPOINT_ROOT=checkpoints
```

### Step 2. Run Main DECS Training 

Download the NRP DETECTOR from `https://huggingface.co/pixas/DECS_NRP_DETECTOR`, put it at `checkpoints` directory and deploy it via 
```
vllm serve --model checkpoints/DECS_NRP_DETECTOR --port 10041 
```

```bash
bash scripts/train/local/train_rl_chunk_local.sh
```

Common overrides:

```bash
MODEL_NAME=r1_distill_qwen1.5b \
DATA_NAME="deepscaler" \
ROLLOUT_N=16 \
CHUNK_JUDGE_URL=127.0.0.1:10041 \
bash scripts/train/local/train_rl_chunk_local.sh
```

Training logs are written to:
- `logs/<data_tag>/<save_name>/train.log`

Checkpoints are saved under (controlled by `src_valid/config/ppo_trainer.yaml`):
- `checkpoints/verl_math/<experiment_name>/global_step_*/actor`


### Step 3. Run Standard Inference/Evaluation (SC)

Assume your trained checkpoint is:
`checkpoints/verl_math/<exp>/global_step_xxx/actor`

```bash
MODEL_CKPT="checkpoints/verl_math/<exp>/global_step_xxx/actor"
HF_MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

bash scripts/eval/local/sc_local.sh \
  "${MODEL_CKPT}" \
  1 \
  16 \
  1 \
  "--prompt_type instruct_default --max_new_tokens 16384 --hf_model_path ${HF_MODEL_PATH}"
```

Parameter meanings:
- arg2 = `chunk_num`
- arg3 = `sc_size`
- arg4 = `tp_size`

Default dataset in `sc_local.sh` is `math`. To evaluate multiple datasets:

```bash
export DATASETS="aime2024 aime2025 amc23 math"
```

### Step 4. Run Prolonged Generation Evaluation

```bash
bash scripts/eval/local/prolong_gen_local.sh \
  "${MODEL_CKPT}" \
  16 \
  32768 \
  "--max_new_tokens 32768 --hf_model_path ${HF_MODEL_PATH}"
```

### Step 5. Check Output Files

Standard SC outputs:
- `results/<dataset>/<model_name>_sc<k>/cache.jsonl`
- `results/<dataset>/<model_name>_sc<k>/result.json`

Prolonged-generation outputs:
- `results/<dataset>/<model_name>_sc<k>_prolong<length>/cache.jsonl`
- `results/<dataset>/<model_name>_sc<k>_prolong<length>/result.json`

## Quick Entry Scripts

### SC Quick Test

```bash
bash scripts/eval/local/test_local.sh <model_ckpt_path> <hf_model_path> [tp_size]
```

### Prolong Quick Test

```bash
bash scripts/eval/local/prolong_gen_test_local.sh <model_ckpt_path> <hf_model_path> [sc_size] [prolong_length]
```


## Citation (BibTeX)

If you find our work useful, please cite our work as:
```bibtex
@inproceedings{jiang2026decs,
  title     = {Overthinking Reduction with Decoupled Rewards and Curriculum Data Scheduling},
  author    = {Jiang, Shuyang and Tao, Xiaofeng and Zhang, Kui and Xiao, Yanghua},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  note      = {Oral},
  url       = {https://arxiv.org/abs/2509.25827}
}
```
