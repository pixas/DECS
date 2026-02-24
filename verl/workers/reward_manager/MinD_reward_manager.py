# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import torch
from collections import defaultdict


try:
    from math_verify.metric import math_metric
    from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig
    from math_verify.errors import TimeoutException
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")
from functools import wraps
import dill

import re 


def mind_compute_score(solution_str: str, ground_truth: str, extra_info=None, data_source=None, format_score: float=0, timeout_score: float = 0, consider_format=True) -> bool:
    model_output = solution_str
    if not model_output.startswith("<think>"):
        model_output = "<think>\n" + model_output
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    ret_score = format_score

    # Wrap the ground truth in \boxed{} format for verification
    ground_truth_boxed = "\\boxed{" + ground_truth + "}" if "boxed" not in ground_truth else ground_truth
    
    # check the model_output's format is <think>xxx</think>xxx<think>xxx</think>
    # check the number of <think> and </think>
    # check the content between <think> and </think> is not empty
    # check the special token after <think> should be </think> and the token after </think> should be <think> except for the condition where </think> is the last special token
    number_check = model_output.count("</think>") == model_output.count("<think>")
    if not number_check:
        ret_score = -1
    else:
        ret_score = 1
        content_check = re.findall(r"<think>(.*?)</think>", model_output, re.DOTALL)
        # check all the pattern 's content should not be empty
        if not content_check or any(not content.strip() for content in content_check):
            ret_score = -1
        if any("<think>" in content or "</think>" in content for content in content_check):
            ret_score = -1
        
    
    
    try:
        model_answer_subseq = model_output.split("</think>")[-1] if consider_format else model_output
        result, _ = verify_func([ground_truth_boxed], [model_answer_subseq])
        if result == True:
            ret_score = ret_score + 2
        else:
            ret_score = ret_score - 2
        # compactness reward 
        # extract the first <think></think> content 
        first_think_content = re.search(r"<think>(.*?)</think>", model_output, re.DOTALL)
        if "double-check" in first_think_content:
            ret_score -= 0.3
        else:
            ret_score = ret_score 
    except Exception as e:
        pass
    except TimeoutException as e:
        print(model_output[-100:], e)
        ret_score = timeout_score

    return ret_score


class MinDRewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, reward_config=None, compute_score=None, reward_fn_key='data_source',max_resp_len=-1, format_score=0) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or mind_compute_score
        self.compute_score = partial(self.compute_score, format_score=format_score)
        self.reward_fn_key = reward_fn_key
        self.reward_config = reward_config
        self.max_resp_len = max_resp_len

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch['rm_scores']}
            else:
                return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            # print(prompt_str)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score
            if self.reward_config is not None:
                if self.reward_config.enable:
                    overlong_buffer_len = self.reward_config.len
                    expected_len = self.max_resp_len - overlong_buffer_len
                    exceed_len = valid_response_length - expected_len
                    overlong_penalty_factor = self.reward_config.penalty_factor
                    overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
                    if self.reward_config.log:
                        reward_extra_info['orignal_reward'].append(reward)
                        reward_extra_info["overlong_reward"].append(overlong_reward)
                        reward_extra_info["overlong"].append(overlong_reward < 0)
                    reward += overlong_reward

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print(f"[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor