import os
import inspect
import sys 

# Add the current directory to the beginning of sys.path to prioritize local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
# Ensure the parent directory is also in sys.path if evaluation is a subdirectory
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import re
# Get parameter names

import sys
import time
import json
import argparse
import random
from itertools import chain
from tqdm import tqdm



from evaluation.models import get_model

from evaluation.utils.data_manager import DataManager

from evaluation.utils.prompt import get_prepare_func
# from rollout.vllm_rollout import VLLMRollout
# model = VLLMRollout("r1distill-qwen-1.5b", response_length=8192)
# exit(0)
from tqdm import tqdm

try:
    import torch_npu

    torch_npu.npu.set_compile_mode(jit_compile=False)
except:
    pass

def parse_args():
    parser = argparse.ArgumentParser(prog="Math Reforcement Learning")

    # global args
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", type=str, default=None)

    # data args
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--chunk_num", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--resume", action="store_true", default=False)

    # model args
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--hf_model_path", type=str, default=None, help="if not None, will convert the fsdp checkpoints to hf models")
    parser.add_argument("--peft_path", type=str, default=None)
    parser.add_argument("--use_vllm", action="store_true", default=False)

    # inference args
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--sample_num", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    
    parser.add_argument('--tp_size', type=int, default=1)
    

    
    # prompt args
    parser.add_argument("--prompt_type", type=str, required=True, choices=["base_default", "base_think", "base_zero", "instruct_default", "instruct_think", "instruct_verl", "instruct_answer", 'instruct_math', 'base_orz', 'raw'], help="base_default: a default cot prompt without <think> tag; base_think: a default cot prompt with <think> tag; base_zero: a default cot prompt with <think></think><answer></answer> tag; instruct_default: a default instruct prompt without <think> tag; instruct_think: a default instruct prompt with <think> tag; instruct_verl: a verl-based instruct prompt without <think>; instruct_answer: a instruct prompt with 'The answer is' format")

    # log args
    parser.add_argument(
        "--cache_file", default="cache.jsonl", help="name of the cache file"
    )
    parser.add_argument(
        "--result_file", default="result.json", help="name of the results file"
    )
    args = parser.parse_args()

    if not os.path.exists(os.path.join(args.output_path)):
        os.makedirs(os.path.join(args.output_path), exist_ok=True)

    print("====Input Arguments====")
    print(json.dumps(vars(args), indent=4, sort_keys=False))

    return args


def get_option_num(question):
    pattern = r"(?<=\n)[A-Z]\..*?(?=\n|$)"
    options = re.findall(pattern, question, flags=re.DOTALL | re.MULTILINE)
    # 返回选项的数量
    return len(options)


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)


    dataset = DataManager(args)
    data_source = dataset.data_source
    print("data_source:", data_source)
    prepare_func = get_prepare_func(args.model_name_or_path, args.prompt_type, data_source)

    # eval begin
    if len(dataset) == 0:
        # already process, we can score
        dataset.score()
        exit(0)
    
    Model = get_model(args.model_name_or_path, args.peft_path, args.use_vllm, max_new_tokens=args.max_new_tokens, temperature=args.temperature, hf_model_path=args.hf_model_path, tensor_parallel_size=args.tp_size)
    tokenizer = Model.tokenizer 

    for examples in tqdm(
        dataset, total=len(dataset), desc=f"Inference with k={args.sample_num}"
    ):
        start_time = time.time()
        inputs = [prepare_func(example)["prompt"] for example in examples]
        inputs = [p if isinstance(p, str) else p[0]['content'] for p in inputs]
        # input_texts = [tokenizer.apply_chat_template(pt, tokenize=False, add_generation_prompt=True) for pt in inputs]
        # if len(inputs) == 0:
        #     continue
        # trajectories = Model(
        #     inputs,
        #     sample_num=args.sample_num,
        #     temperature=args.temperature if args.sample_num > 1 else 0,
        #     max_new_tokens=args.max_new_tokens,
        # )  # trajectories is a list of length b x S
        # input_ids = tokenizer(input_texts, padding=True, padding_side='left').input_ids
        # repeat the inputs by args.sample_num
        repeated_inputs = []
        for input_ in inputs:
            repeated_inputs.extend([input_] * args.sample_num)
        inputs = repeated_inputs
        trajectories = Model.rollout(
            inputs,
            do_sample=True if args.sample_num > 1 else False,
            wrap_chat=True if "instruct" in args.prompt_type else False,
            n=1
        )['response_text']
        # for example, [a1, a2, a3, a4, b1, b2, b3, b4, ...], where ai is the output from the same prompt
        # for vllm, the trajectories is a list of list
        # if isinstance(trajectories, list):
        # predictions = [traj for trajectory in trajectories for traj in trajectory]
        # trajectories = [traj for trajectory in trajectories for traj in trajectory]
        # inputs = [inputs[i // args.sample_num] for i in range(len(trajectories))]

        # # extract the answer with logit bias
        # get_prediction_inputs = [
        #     f"""{input_}{trajectory}{'' if trajectory.endswith('.') else '.'} Therefore, the answer is """
        #     for input_, trajectory in zip(inputs, trajectories)
        # ]

        # option_num = max(get_option_num(example["input"]) for example in examples)
        
        # predictions = Model(
        #     get_prediction_inputs,
        #     max_new_tokens=1 if option_num > 0 else 20,
        #     logit_bias=option_num,
        # )
        # predictions = [pred for prediction in predictions for pred in prediction]
        end_time = time.time()
        # post process of trajectories, if the prompt type is base_orz, extract the content between <answer></answer>
        # and wrap it with \\boxed{}
        # and reconstruct the answer with the trajectory
        if args.prompt_type == 'base_orz':
            new_trajectories = []
            for traj in trajectories:
                # traj = traj.replace("<answer>", "\\boxed{").replace("</answer>", "}")
                traj = re.sub(r"<answer>(.*?)</answer>", r"<answer>\\boxed{\1}</answer>", traj)
                # I need to re-add the <answer></answer> tag but with \\boxed{}
                # traj
                new_trajectories.append(traj)
            trajectories = new_trajectories
        # record the evaluation process
        sample_logs = [
            {
                "task": example,
                "trajectory": [
                    f"""{trajectory}"""
                    for trajectory in trajectories[
                                                batch_id
                                                * args.sample_num : (batch_id + 1)
                                                * args.sample_num
                                            ]
                ],
                "time": (end_time - start_time) / len(examples),
                "tokens": [
                    len(
                        Model.tokenizer(trajectory, add_special_tokens=False)[
                            "input_ids"
                        ]
                    )
                    for trajectory in trajectories[
                        batch_id * args.sample_num : (batch_id + 1) * args.sample_num
                    ]
                ],
            }
            for batch_id, example in enumerate(examples)
        ]
        dataset.update_cache(sample_logs)
