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

import numpy as np
from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import torch
from collections import defaultdict


def split_based_on_reflect(sentence, predefined_reflect_word=["\nWait", "\nBut wait", "\nBut", "\nAlternatively", "\nHmm"]):
    """
    Splits the sentence based on the predefined reflect words.
    """
    first_response = sentence
    splitted_response = []
    # Only proceed if first_response has content. If first_response is an empty string, 
    # splitted_response should remain empty, which is the default.
    if first_response:
        all_split_points = []
        for word in predefined_reflect_word:
            # Ensure the predefined word itself is not empty to avoid issues with find()
            if not word:
                continue
            
            start_search_index = 0
            while True:
                idx = first_response.find(word, start_search_index)
                if idx == -1:
                    break
                all_split_points.append(idx)
                # Move search start to after the beginning of the found word to find subsequent occurrences
                start_search_index = idx + 1 

        # Get unique split points and sort them
        unique_sorted_split_points = sorted(list(set(all_split_points)))

        if not unique_sorted_split_points:
            # No reflect words found, the whole response is a single segment
            # we should cut the contents until </think> tokens 
            cut_first_response = first_response[:first_response.find("</think>")]
            splitted_response.append(cut_first_response)
        else:
            last_cut = 0
            for point in unique_sorted_split_points:
                # Add the segment before the current reflect word's start index
                # This check also handles cases where a reflect word is at the beginning (point == last_cut)
                if point > last_cut:
                    splitted_response.append(first_response[last_cut:point])
                last_cut = point # The next segment will start from this point
            
            # Add the final segment, from the start of the last reflect word to the end of the string
            if last_cut < len(first_response):
                splitted_response.append(first_response[last_cut:])
            # If last_cut == len(first_response), it means the string ended exactly at a split point,
            # or the last segment added was up to the end.
            # For example, if first_response = "text\nReflectWord", and "\nReflectWord" is a split point.
            # last_cut will be the index of "\nReflectWord". The segment "\nReflectWord" is added.
            # If first_response = "\nReflectWord", last_cut will be 0. The segment "\nReflectWord" is added.
    # for the lask segment, we need to cut until </think>
            if splitted_response and splitted_response[-1].find("</think>") != -1:
                splitted_response[-1] = splitted_response[-1][:splitted_response[-1].find("</think>")]
    # for extremely short response (only one line), we need to merge it to the next segment with multi-lines
    # there will be a case that consecutive segments contain only one line, cache them until a segment with multi lines 
    clean_splitted_response = []
    merge_cache = []
    for i, response in enumerate(splitted_response):
        lines = [line.strip() for line in response.split("\n") if line.strip()]
        
        if len(lines) == 1 and i < len(splitted_response) - 1 and len(merge_cache) < 1:
            # merge it with the next segment
            merge_cache.append(response)
        else:
            tobe_merged_contents = "".join(merge_cache)
            clean_splitted_response.append(tobe_merged_contents + response)
            merge_cache = []
    return clean_splitted_response

class EMAStepPredictor:
    def __init__(self):
        self.ema_correct_step = 10
        self.ema_error_step = 20
        self.alpha = 1
        self.correct_factor = 0.3
        self.error_factor = -0.02 
        self.correct_steps_history = [] 
        self.error_steps_history = []
        
    
    # def update(self, trajectories, correctness):
    #     reasoning_steps = [split_based_on_reflect(trajectory) for trajectory in trajectories]
    #     for i, trajectory in enumerate(trajectories):
    #         if correctness[i]:
    #             self.ema_correct_step = self.alpha * len(reasoning_steps[i]) + (1 - self.alpha) * self.ema_correct_step
    #         else:
    #             self.ema_error_step = self.alpha * len(reasoning_steps[i]) + (1 - self.alpha) * self.ema_error_step
    
    def update_one(self, reasoning_steps, correctness):
        # reasoning_steps = split_based_on_reflect(trajectory)
        if correctness:
            self.ema_correct_step = self.alpha * len(reasoning_steps) + (1 - self.alpha) * self.ema_correct_step
        else:
            self.ema_error_step = self.alpha * len(reasoning_steps) + (1 - self.alpha) * self.ema_error_step
        
        if correctness:
            self.correct_steps_history.append(len(reasoning_steps))
            if len(self.correct_steps_history) > 2048:
                self.correct_steps_history.pop(0)
        else:
            self.error_steps_history.append(len(reasoning_steps))
            if len(self.error_steps_history) > 2048:
                self.error_steps_history.pop(0) 
    
    def get_efficiency(self, total_steps, is_correct):
        correct_benchmark = np.percentile(self.correct_steps_history, 30)  # 30分位数
        error_benchmark = np.percentile(self.error_steps_history, 90)      # 90分位数
        efficiency_reward = (
            self.correct_factor * max(0, correct_benchmark - total_steps) if is_correct
            else self.error_factor * max(0, total_steps - error_benchmark)
        )
        return efficiency_reward

class ConfColdRewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, reward_config=None, compute_score=None, reward_fn_key='data_source',max_resp_len=-1, format_score=0, eval=False) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.compute_score = partial(self.compute_score, format_score=format_score)
        self.reward_fn_key = reward_fn_key
        self.reward_config = reward_config
        self.max_resp_len = max_resp_len
        self.eval = eval
        # self.max_non_confident_thr = 2
        # self.step_penalty = 0.05
        self.ema_step_predictor = EMAStepPredictor()
        

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch['rm_scores']}
            else:
                return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        efficiency_reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        total_num_steps = []
        total_response_len = []
        total_correctness = []
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
            
            reward_extra_info['original_reward'].append(reward)
            
            reasoning_steps = split_based_on_reflect(response_str)
            self.ema_step_predictor.update_one(reasoning_steps, reward > 0)
            total_num_steps.append(len(reasoning_steps))
            total_response_len.append(valid_response_length)
            total_correctness.append(reward > 0)
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
                
        for i in range(len(data)):
            efficiency_reward = self.ema_step_predictor.get_efficiency(
                total_steps=total_num_steps[i],
                is_correct=total_correctness[i]
            )
            reward_extra_info['efficiency_reward'].append(efficiency_reward)
            reward_tensor[i, total_response_len[i] - 1] += efficiency_reward
            
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor