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
# from .reflect_reward_func import compute_score as reflect_compute_score
import torch
from collections import defaultdict
import json

class ADLPRewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, reward_config=None, compute_score=None, reward_fn_key='data_source',max_resp_len=-1, format_score=0, ref_acc_file=None, eval=False) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        # self.compute_score = partial(self.compute_score, eval=eval)
        self.reward_fn_key = reward_fn_key
        self.reward_config = reward_config

        self.max_resp_len = max_resp_len
        if ref_acc_file is not None:
            print("Read reference accuracy file from", ref_acc_file)
            with open(ref_acc_file, 'r') as f:
                ref_acc = json.load(f)
                self.problem_acc_mapping = {item['problem'].strip(): item['metrics']['avg_acc_thinking'] for item in ref_acc}
        else:
            self.problem_acc_mapping = None
                
        if eval:
            self.lambda_t = 0
            self.eta = 0
        else:
            self.lambda_t = 1e-3
            self.eta = 1e-3
        self.eval = eval

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
        
        # response_uid2score = dict()

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']
            # response_uid = data_item.non_tensor_batch['response_uid']
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            
            # enforce_nothinking = data_item.batch['enforce_nothinking'].item()
            # if enforce_nothinking < valid_response_length:
            #     manual_response = True 
            # else:
            #     manual_response = False
            # manual_response = data_item.batch['should_modify'].item()
            # modify_loc = data_item.batch['modify_loc'].item()
            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
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
            if self.problem_acc_mapping is not None:
                problem = prompt_str.split('<｜User｜>')[1].split('<｜Assistant｜>')[0].strip()
                acc = self.problem_acc_mapping[problem]
                
            else:
                # print("Warning: ADLP method requries to access a reference acc file")
                acc = None
            # if acc is not None:
            #     reward_extra_info['ref_acc'].append(acc)
            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score
            
            if acc is not None:
                cur_acc = max(reward, 0) # [clip to 0, 1 which is required by adlp]
                self.lambda_t = max(0, self.lambda_t + self.eta * (cur_acc - acc))
                reward = cur_acc - self.lambda_t * valid_response_length
                reward_extra_info['lambda'].append(self.lambda_t)
                # a correct reward 
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


        
        batch_size = len(data)
        keys_to_pad = [
            'lambda'
        ]
        for key in keys_to_pad:
            if key in reward_extra_info:
                reward_extra_info[key] = reward_extra_info[key] + [-1] * (batch_size - len(reward_extra_info[key]))

        
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor