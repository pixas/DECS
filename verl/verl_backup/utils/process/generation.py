from functools import partial
import math
import warnings
import torch
import re
from enum import Enum
from collections import defaultdict
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from itertools import chain

from verl.utils.torch_functional import logprobs_from_logits
from .tensor_helper import TensorHelper, TensorConfig
# from search_r1.utils import set_seed
# from search_r1.utils.plot import (
#     save_trajectory_to_output,
#     parse_llm_output
# )
from verl import DataProto
from verl.utils.tracking import Tracking
import shutil
import requests
from verl.utils.prompt import *


def split_response_into_steps(response):
    # Use a regex pattern to split by numbered steps (1., 2., etc.)
    # The pattern captures the step number and the "." following it as part of the split
    step_pattern = r'^\d+\.\s+\*\*'
    
    # Find all matches
    step_starts = [m.start() for m in re.finditer(step_pattern, response, re.MULTILINE)]
    
    steps = []
    # Extract each step
    for i in range(len(step_starts)):
        start = step_starts[i]
        if i == 0:
            if start > 0:
                # If this is the first step, include everything before it
                steps.append(response[:start])

        # If this is not the last step, end at the next step
        if i < len(step_starts) - 1:
            end = step_starts[i + 1]
            steps.append(response[start:end])
        else:
            # For the last step, include everything to the end
            steps.append(response[start:])
    if not steps:
        return []
    # Verify that combining the steps gives the original response
    if ''.join(steps) != response: 
        raise ValueError("The split steps don't reconstruct the original response, original response: {}, split steps: {}".format(response, ''.join(steps)))

    
    return steps

def compute_log_prob(model, batch):
    input_ids = batch.batch['input_ids']
    attention_mask = batch.batch['attention_mask']
    position_ids = batch.batch['position_ids']
    response = batch.batch['responses']
    response_length = response.size(-1)
    output = model(input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False)
    logits = output.logits
    logits.div_(1.0)
    logits = logits[:, -response_length - 1:-1, :]
    log_probs = logprobs_from_logits(logits, response)
    return DataProto.from_dict({"ref_log_prob": log_probs})

@dataclass
class GenerationConfig:
    max_prompt_length: int 
    max_response_length: int
    # logging: dict
    num_gpus: int
    gamma: float
    

class ProcessGuideType(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """
    Direct = "direct"
    Chunk = "chunk"
    Order = "order"
    

class ProcessRewardManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
        # logger: Tracking,
        is_validation: bool = False,
        process_reward_config: Dict[str, Any] = None
    ):

        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        # self.logger = logger
        self.is_validation = is_validation
        self.process_reward_config = process_reward_config
        
        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
        ))
        self.guide_prompt = self.process_reward_config.get("guide_prompt", "direct")
        if self.guide_prompt == ProcessGuideType.Direct or self.guide_prompt == ProcessGuideType.Order:
            self.process_guide_prompt = process_guide_prompt_direct
        elif self.guide_prompt == ProcessGuideType.Chunk:
            self.process_guide_prompt = process_guide_prompt_chunk
        else:
            raise ValueError(f"Invalid process guide prompt type {self.process_reward_config.guide_prompt}")
        
    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']
        
    def adaptive_forward(self):
        # return partial(compute_log_prob, self.actor_rollout_wg)
        if self.process_reward_config.policy == 'ref':
            return self.actor_rollout_wg.compute_ref_log_prob
        elif self.process_reward_config.policy == 'actor':
            return self.actor_rollout_wg.compute_log_prob
        else:
            raise ValueError("Invalid policy type, only support 'ref' and 'actor'")
    
    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
        
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)

        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)
        
        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output

    def _forward_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.adaptive_forward()(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        if remainder == 0:
            return self.adaptive_forward()(active_batch)
        
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)

        # Generate with padded batch
        padded_output = self.adaptive_forward()(padded_active_batch)
        
        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output
    
    def compute_batch_basic_info(self, data: DataProto):
        prompt_ids = data.batch['prompts']
        prompt_length = prompt_ids.shape[-1]
        # prompt_strings = self.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)

        response_ids = data.batch['responses']
        valid_response_lengths = data.batch['attention_mask'][:, prompt_length:].sum(dim=-1)
        # sequences_strings = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        # data_sources = data.non_tensor_batch['data_source']
        # extra_info = data.non_tensor_batch.get('extra_info', [None] * len(data_sources))
        ground_truth = [data_item.non_tensor_batch['reward_model']['ground_truth'] for data_item in data]
        
        return prompt_ids, response_ids, valid_response_lengths, ground_truth
    
    def _construct_batch_from_response(self, sub_responses, ground_truth, meta_info):

        sub_response_result = self.tokenizer(sub_responses, 
                                             return_tensors='pt', 
                                             padding=True, 
                                             truncation=True, 
                                             max_length=self.config.max_prompt_length + self.config.max_response_length + 50,
                                             padding_side='left')
        sub_response_input_ids = sub_response_result['input_ids']
        sub_response_attention_mask = sub_response_result['attention_mask']
        # 获得\nThe answer is后的input_ids，作为新的batch中的response内容，去计算log_prob
        new_response_ids = self.tokenizer(f" {ground_truth}", return_tensors='pt')['input_ids']
        new_batch = DataProto.from_dict(
            tensors={
                "responses": torch.cat([new_response_ids] * sub_response_input_ids.shape[0]),
                "attention_mask": sub_response_attention_mask,
                "input_ids": sub_response_input_ids,
                "position_ids": self.tensor_fn.create_position_ids(sub_response_attention_mask),
            },
            non_tensors={},
            meta_info=meta_info
        )
        return new_batch

    def compute_ref_log_prob_reward(self, batch: DataProto):
        """Compute the reference log probability reward for each step in the reasoning process."""
        reward_tensor = torch.zeros_like(batch.batch['responses'], dtype=torch.float32)
        
        # Check if batch is empty
        if len(batch) == 0:
            return reward_tensor

        batch_prompt_ids, batch_response_ids, batch_valid_response_lengths, ground_truths = self.compute_batch_basic_info(batch)
        
        for i in range(len(batch)):
            data_item = batch[i]
            prompt_ids = batch_prompt_ids[i]
            response_ids = batch_response_ids[i]
            valid_response_length = batch_valid_response_lengths[i]
            outcome_reward = data_item.batch['original_token_level_score']
            ground_truth = ground_truths[i]
            if self.process_reward_config.filter and outcome_reward.sum() == -1:
                # -1 means a bad format
                continue
            # Skip processing if no valid response
            if valid_response_length <= 0:
                continue
                
            # Extract prompt and response text
            prompt_str, response_str = self._extract_prompt_response_text(data_item, prompt_ids, response_ids)
            
            # Process response into reasoning steps
            sub_responses = self._extract_reasoning_steps(response_str, response_ids)
            
            # Skip if not enough steps for meaningful analysis
            if len(sub_responses) <= 1:
                continue
                
            # Create processed prompts for each reasoning step
            processed_prompts = self._create_step_prompts(prompt_str, sub_responses, ground_truth)
            
            # Calculate split locations for step rewards if needed
            token_split_location = None
            if self.process_reward_config.step_reward:
                token_split_location = self._calculate_split_locations(response_str, sub_responses, valid_response_length)
            
            # Create batch for computing log probabilities
            new_batch = self._construct_batch_from_response(
                processed_prompts, 
                ground_truth, 
                {"temperature": data_item.meta_info['temperature']}
            )

            # Compute and apply rewards
            self._compute_and_apply_logp_step_rewards(
                new_batch, reward_tensor, i, valid_response_length, 
                token_split_location, len(sub_responses)
            )
            
        return reward_tensor
        
    def _extract_prompt_response_text(self, data_item, prompt_ids, response_ids):
        """Extract the text from prompt and response tokens."""
        prompt_length = prompt_ids.shape[-1]
        
        # Extract valid prompt
        valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]
        prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
        
        # Extract valid response
        valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]
        response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        
        return prompt_str, response_str
        
    def _extract_reasoning_steps(self, response_str, response_ids):
        """Extract reasoning steps from response text."""
        if self.guide_prompt == ProcessGuideType.Direct:
            # Split by double newlines, then by single newlines
            sub_responses = response_str.split("\n\n")
            sub_responses = [x.strip() for x in sub_responses]
            temp = [x.split("\n") for x in sub_responses]
            sub_responses = list(chain(*temp))
            
            # Filter and deduplicate while preserving order
            sub_responses = [x.strip() for x in sub_responses if len(x.strip()) > 0]
            sub_responses = list(dict.fromkeys(sub_responses))
            
            # Filter out steps that are too long to tokenize
            sub_responses = [s for s in sub_responses if len(self.tokenizer.encode(s)) <= 500]
        elif self.guide_prompt == ProcessGuideType.Order:
            sub_responses = split_response_into_steps(response_str)
        elif self.guide_prompt == ProcessGuideType.Chunk:
            # split the response_ids into chunks
            # sub_responses are the decoded string
            sub_responses = []
            chunk_size = self.process_reward_config.chunk_size 
            i = 0

            while i < len(response_ids):
            # for i in range(0, len(response_ids), chunk_size):
                # sub_responses.append(response_ids[i:i+chunk_size])
                chunk_response = response_ids[i:i+chunk_size]
                string = self.tokenizer.decode(chunk_response, skip_special_tokens=True)
                if string:
                    # 判断有没有乱码

                    # Check if all characters are valid (not garbled)
                    while string not in response_str and len(chunk_response) > 0:
                        chunk_response = chunk_response[:-1]
                        string = self.tokenizer.decode(chunk_response, skip_special_tokens=True)
                    if len(chunk_response) == 0:
                        i += 1
                        continue
                    sub_responses.append(string)
                    i += len(chunk_response)
                    

                else:
                    i += chunk_size
            for sub_response in sub_responses:
                assert sub_response in response_str, f"sub_response: {sub_response}, response_str: {response_str}"
        
        return sub_responses
        
    def _calculate_split_locations(self, response_str, sub_responses, valid_response_length):
        """Calculate token indices where reasoning steps split."""
        # Find string indices of each step in the response
        split_location = []
        for sub_response in sub_responses[1:]:
            try:
                split_location.append(response_str.index(sub_response))
            except:
                print("original response:", response_str)
                print("failed sub_response:", sub_response)
                exit(-1)
        
        # Convert to token indices
        token_split_location = [max(0, len(self.tokenizer.encode(response_str[:loc])) - 1) 
                               for loc in split_location] 
        
        return token_split_location
        
    def _create_step_prompts(self, prompt_str, sub_responses, ground_truth):
        """Create prompts for each reasoning step."""
        processed_prompts = []
        for j in range(len(sub_responses)):
            if self.guide_prompt == ProcessGuideType.Direct:
                prefix = '\n'.join(sub_responses[:j])
            elif self.guide_prompt == ProcessGuideType.Chunk or self.guide_prompt == ProcessGuideType.Order:
                prefix = ''.join(sub_responses[:j])
            else:
                raise ValueError(f"Invalid process guide prompt type {self.guide_prompt}")
            processed_prompts.append(f"{prompt_str}{prefix}{self.process_guide_prompt}{ground_truth}")
        return processed_prompts
        
    def _compute_and_apply_logp_step_rewards(self, new_batch, reward_tensor, batch_idx, 
                                      valid_response_length, token_split_location, num_steps):
        """Compute and apply rewards for each reasoning step."""
        # Get log probabilities
        answer_log_prob = self._forward_with_gpu_padding(new_batch)
        tensor_key = 'ref_log_prob' if 'ref_log_prob' in answer_log_prob.batch else 'old_log_probs'
        tensor_answer_log_prob = answer_log_prob.batch[tensor_key]
        
        # Sum over response length
        tensor_answer_log_prob = tensor_answer_log_prob.sum(-1)
        
        if self.process_reward_config.score_type == 'rel':
            self._compute_rel_logp_step_rewards(tensor_answer_log_prob, reward_tensor, batch_idx,
                                                valid_response_length, token_split_location)
        elif self.process_reward_config.score_type == 'abs':
            self._compute_abs_logp_step_rewards(tensor_answer_log_prob, reward_tensor, batch_idx,
                                                valid_response_length, token_split_location)
            pass
    
    def _compute_rel_logp_step_rewards(self, tensor_answer_log_prob, reward_tensor, batch_idx,
                                      valid_response_length, token_split_location):
        """Compute and apply rewards for each reasoning step."""
        # Calculate differences between consecutive steps
        tensor_answer_log_prob_diff = tensor_answer_log_prob[1:] - tensor_answer_log_prob[:-1]
        # the first step has no relative reward, append 0 to the start of tensor_answer_log_prob_diff
        # tensor_answer_log_prob_diff = torch.cat([torch.tensor([0.0]).to(tensor_answer_log_prob_diff.device), tensor_answer_log_prob_diff])
        
        # Apply appropriate reward transformation
        # process_reward = self._transform_step_rewards(tensor_answer_log_prob_diff)
        bound = torch.max(torch.abs(tensor_answer_log_prob_diff)) - 1
        process_reward = (torch.abs(tensor_answer_log_prob_diff) - 1) / bound 
        # take the larger value between process_reward and 0, element wise
        process_reward = torch.where(process_reward > 0, process_reward, torch.tensor(0.0).to(process_reward.device))
        
        
        # Apply rewards to tensor
        if self.process_reward_config.step_reward and token_split_location:
            self._apply_step_rewards(reward_tensor, batch_idx, valid_response_length, 
                                process_reward, token_split_location, tensor_answer_log_prob_diff)
        else:
            # Use proportion of positive steps as final reward
            one_count = (tensor_answer_log_prob_diff > 0) & (process_reward > 0)
            
            one_count = one_count.sum().item()
            reward_tensor[batch_idx, valid_response_length - 1] = one_count / len(process_reward)
    
    def _compute_abs_logp_step_rewards(self, tensor_answer_log_prob, reward_tensor, batch_idx,
                                      valid_response_length, token_split_location):
        """Compute and apply rewards for each reasoning step."""
        process_reward = torch.exp(tensor_answer_log_prob) # [step,] represent the probability to generate answer tokens
        
        
        # Apply rewards to tensor
        if self.process_reward_config.step_reward and token_split_location:
            self._apply_step_rewards(reward_tensor, batch_idx, valid_response_length, 
                                process_reward, token_split_location, None)
        else:
            raise ValueError("Absolute reward should have step_reward enabled")
            # Use proportion of positive steps as final reward
            # one_count = (process_reward > 0).sum().item()
            # reward_tensor[batch_idx, valid_response_length - 1] = one_count / len(process_reward)
    
    def _transform_step_rewards(self, reward_diff):
        """Transform raw reward differences using configured method."""
        if self.process_reward_config.use_soft:
            if self.process_reward_config.use_soft == 'tanh':
                return torch.tanh(reward_diff)
            elif self.process_reward_config.use_soft == 'max':
                return reward_diff / (reward_diff.abs().max())
            elif self.process_reward_config.use_soft == 'sigmoid':
                return torch.sigmoid(reward_diff)
            else:
                raise NotImplementedError(f"Soft reward type {self.process_reward_config.use_soft} is not implemented")
        else:
            # Binary reward: 1 for positive, -1 for negative
            return torch.where(reward_diff > 0, 1, -1).float()
            
    def _apply_step_rewards(self, reward_tensor, batch_idx, valid_response_length, 
                          process_reward, token_split_location, reward_diff):
        """Apply rewards at specific token positions based on step locations."""
        last_step_loc = 0
        assert len(token_split_location) == len(process_reward), f"Length of split locations and rewards do not match, split location {len(token_split_location)}, rewards {len(process_reward)}"
        for j, step_loc in enumerate(token_split_location):
            if step_loc >= valid_response_length:
                warnings.warn(f"Split location {step_loc} is out of bound {valid_response_length}")
                continue
                
            # Calculate division factor based on step type configuration
            division_factor = self._calculate_division_factor(
                j, step_loc, valid_response_length, reward_diff
            )
            if self.process_reward_config.token_reward:

                reward_tensor[batch_idx, last_step_loc:step_loc + 1] = process_reward[j] * division_factor

                last_step_loc = step_loc + 1
            else:
                # Apply reward at the step location
                reward_tensor[batch_idx, step_loc] = process_reward[j] * division_factor
            
    def _calculate_division_factor(self, step_idx, step_loc, valid_response_length, reward_diff=None):
        """Calculate division factor based on step type configuration."""
        step_type = self.process_reward_config.get("step_type", 1)
        
        if step_type == 1:
            return 1 / (valid_response_length - step_loc)
        elif step_type == 2:
            return 1 / (step_loc + 1)
        elif step_type == 3:
            return 1
        elif step_type == 4:
            return 1 / math.sqrt(step_idx + 1)
        elif step_type == 5:
            return math.pow(self.config.gamma, (valid_response_length - step_loc))
        elif step_type == 6:
            tau_plus = torch.quantile(reward_diff, 0.75)
            tau_minus = torch.quantile(reward_diff, 0.25)
            return 1 if reward_diff[step_idx] > tau_plus else 0.05 if reward_diff[step_idx] < tau_minus else 0
        elif step_type == 7:
            return 1 if reward_diff[step_idx] > 0 else 0
        elif step_type == 8:
            return 1 if reward_diff[step_idx] > 0 else 0.05
        elif step_type == 9:
            return 1 if reward_diff[step_idx] > 0 else -1
        return 1  # Default case