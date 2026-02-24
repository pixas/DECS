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
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

from itertools import chain
import json
from math import e
import os
import random
import uuid
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from tqdm import tqdm
from typing import Type, Dict
import copy
from copy import deepcopy
from collections import defaultdict
from functools import partial
from tqdm import tqdm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

import ray
import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
# from verl.utils.process.generation import GenerationConfig, ProcessRewardManager


from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics, reduce_metrics, bootstrap_metric, calc_maj_val, process_validation_metrics
from verl.utils.proxy import send_feishu_message
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn


from src_valid.tracking import ValidationGenerationsLogger, TrainGenerationsLogger
from torch.utils.data import Dataset, RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from verl.utils.checkpoint.s3_client import client 

import verl.utils.torch_functional as verl_F
WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """
    GAE = 'gae'
    GRPO = 'grpo'
    GRPO_PROC = 'grpo_proc'
    GRPO_PROC_LENGTH = 'grpo_proc_length'
    RPP_PROC_LENGTH = 'rpp_proc_length'
    GRPO_CHUNK = 'grpo_chunk'
    GRPO_HYBRID = 'grpo_hybrid'
    REINFORCE_PLUS_PLUS = 'reinforce_plus_plus'
    REINFORCE_WITH_REFLECT = 'reinforce_with_reflect'
    REINFORCE_PLUS_PLUS_BASELINE = 'reinforce_plus_plus_baseline'
    REMAX = 'remax'
    RLOO = 'rloo'


def compute_chunk_metrics(batch: DataProto):
    metrics = {}

    # response_mask = batch.batch['response_mask']  # [B, L]
    chunk_aware_adv = batch.batch.get('chunk_aware_adv', None)  # [B, L]
    base_advantages = batch.batch.get('base_advantages', None)

    if chunk_aware_adv is None :
        return {}
    bsz = chunk_aware_adv.shape[0]
    
    
    total_count = bsz

    # metrics['chunk_aware/base_all_correct'] = all_correct / total_count
    if base_advantages is not None:
        with torch.no_grad():
            lower_correct = (chunk_aware_adv.abs() > base_advantages.abs()).any(dim=-1).sum().item() 
        metrics['chunk_aware/dominated_by_chunk'] = lower_correct / total_count
    # exit(-1)
    # chunk_aware_adv
    # based on response mask to select all values of chunk_aware_adv 
    chunk_mask = chunk_aware_adv != 0
    valid_adv = torch.masked_select(chunk_aware_adv, chunk_mask.bool())
    if valid_adv.numel() != 0:
    
        metrics['chunk_aware/token_adv_mean'] = valid_adv.mean().item()
        metrics['chunk_aware/token_adv_min'] = valid_adv.min().item()
        metrics['chunk_aware/token_adv_max'] = valid_adv.max().item()
        metrics['chunk_aware/token_adv_std'] = valid_adv.std().item()
    else:
        metrics['chunk_aware/token_adv_mean'] = 0.0
        metrics['chunk_aware/token_adv_min'] = 0.0
        metrics['chunk_aware/token_adv_max'] = 0.0
        metrics['chunk_aware/token_adv_std'] = 0.0
    if 'first_correct_chunk_ratio' in batch.non_tensor_batch:
        first_correct_chunk_ratio = batch.non_tensor_batch['first_correct_chunk_ratio']
        
        metrics['chunk_aware/first_correct_chunk_ratio_mean'] = (first_correct_chunk_ratio.sum() / (first_correct_chunk_ratio != 0).sum())
        metrics['chunk_aware/first_correct_chunk_ratio_min'] = first_correct_chunk_ratio[first_correct_chunk_ratio != 0].min()
        metrics['chunk_aware/first_correct_chunk_ratio_max'] = first_correct_chunk_ratio[first_correct_chunk_ratio != 0].max()
        
        total_chunk_count = batch.non_tensor_batch['total_chunk_count']
        
        metrics['chunk_aware/total_chunk_mean'] = total_chunk_count.mean()
        metrics['chunk_aware/total_chunk_min'] = total_chunk_count.min()
        metrics['chunk_aware/total_chunk_max'] = total_chunk_count.max()
    
    # metrics['chunk_aware/']

    print(f'\n\nchunk_aware_advantage_METRICS\n{metrics}\n\n')
    return metrics


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get('GPU', 0) for node, node_info in node_available_resources.items()}

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}")

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(
                    f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes} cannot be satisfied in this ray cluster"
                )


import torch
from verl.utils.torch_functional import masked_mean, masked_var


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                kl_penalty=kl_penalty)  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'actor/reward_kl_penalty': current_kl, 'actor/reward_kl_penalty_coeff': beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    responses = data.batch['responses']
    response_length = responses.size(1)
    attention_mask = data.batch['attention_mask']
    return attention_mask[:, -response_length:]

import torch

def scale_advantages_vectorized(chunk_aware_adv, base_advantages, factor=2, only_positive=True):
    """
    向量化实现缩放chunk_aware_adv，使其大于0的元素不超过base_advantages/2，同时保持大小关系
    
    参数:
        chunk_aware_adv: 需要缩放的tensor (任意形状)
        base_advantages: 基准tensor，形状应与chunk_aware_adv相同
        
    返回:
        缩放后的chunk_aware_adv
    """
    # 创建原始tensor的副本以避免修改原始数据
    scaled_adv = chunk_aware_adv.clone()
    
    # 1. 只处理大于0的元素
    mask_pos = chunk_aware_adv > 0 
    
    # 2. 计算上限 (base_advantages / 2)
    upper_bound = base_advantages / factor
    
    # 3. 找出需要缩放的元素 (大于对应位置的upper_bound)
    need_scale_mask = mask_pos & (chunk_aware_adv > upper_bound)
    
    # 4. 对于需要缩放的元素，计算缩放因子
    # 缩放因子 = upper_bound / original_value
    scale_factors = torch.where(need_scale_mask, upper_bound / chunk_aware_adv, 1.0)
    
    # 5. 对每行(或任意最后一维以外的维度)找出最小缩放因子
    # 首先将不需要缩放的位置设为inf，这样min不会选中它们
    inverted_scale = torch.where(need_scale_mask, scale_factors, float('inf'))
    
    # 沿着最后一维取最小值，保持其他维度
    min_scale = inverted_scale.min(dim=-1, keepdim=True).values
    
    # 6. 应用缩放 (保持大小关系)
    # 将inf替换为1.0 (无缩放)
    min_scale = torch.where(min_scale == float('inf'), 1.0, min_scale)
    
    # 只对正元素应用缩放
    scaled_adv = torch.where(mask_pos, chunk_aware_adv * min_scale, chunk_aware_adv)
    
    return scaled_adv

# 全局状态
ema_gamma = 0.5
alpha_history = []

def get_alpha_from_gamma(current_gamma, alpha_max=0.15, alpha_min=0.02):
    global ema_gamma
    ema_gamma = 0.9 * ema_gamma + 0.1 * current_gamma
    
    # 映射：gamma ∈ [0.5, 0.9] → alpha ∈ [alpha_max, alpha_min]
    alpha = np.interp(ema_gamma, [0.5, 0.9], [alpha_max, alpha_min])
    return np.clip(alpha, alpha_min, alpha_max)


def compute_grpo_proc_length_advantage_return(token_level_rewards: torch.Tensor,
                                              chunk_aware_adv: torch.Tensor,
                                              response_mask: torch.Tensor, 
                                              index,
                                              chunk_config):
    epsilon = 1e-6
    token_level_scores = token_level_rewards.sum(-1) 
    response_length = response_mask.sum(-1) # [B, ]
    max_seq_len = response_mask.shape[-1] 
    bsz = token_level_rewards.shape[0]
    
    
    
    with torch.no_grad():

        token_correctness_scores = token_level_scores.unsqueeze(-1).repeat(1, token_level_rewards.shape[1])

            
        final_token_rewards = token_correctness_scores.clone() * (1 + chunk_config.length_reward_beta)
            
        # detect chunk_aware_adv the first negative value index
        length_reward = torch.zeros_like(token_level_rewards)
        for i in range(bsz):
            if token_level_scores[i] > 0:
                cur_seq_len = response_length[i]
                # invert
                # if chunk_config.reward_func == 'invert':
                #     length_reward[i, :cur_seq_len] = chunk_config.length_reward_beta / (cur_seq_len - torch.arange(0, cur_seq_len, dtype=length_reward.dtype))
                # elif chunk_config.reward_func == 'exp':
                
                #     # exponential
                #     # l_metrics = cur_seq_len - torch.arange(0, cur_seq_len) / max_seq_len
                #     length_reward[i, :cur_seq_len] = -torch.exp((cur_seq_len - torch.arange(0, cur_seq_len)) / max_seq_len) / e * chunk_config.length_reward_beta
                # elif chunk_config.reward_func == 'linear':
                #     length_reward[i, :cur_seq_len] = (1+(torch.arange(0, cur_seq_len) - cur_seq_len) / max_seq_len) * chunk_config.length_reward_beta
                # elif chunk_config.reward_func == 'constant':
                length_reward[i, :cur_seq_len] = (1-cur_seq_len / max_seq_len) * chunk_config.length_reward_beta
                

                cur_chunk_adv = chunk_aware_adv[i]
                # find the first index whose cur_chunk_adv is negative, if found, all values after this index should add corresponding length_reward, if not found, remain the final_token_rewards as token_correctness_scores
                negative_index = (cur_chunk_adv < 0).nonzero(as_tuple=True)[0]
                if negative_index.numel() > 0:
                    first_negative_index = negative_index[0].item()
                    final_negative_index = negative_index[-1].item() + 1
                    final_token_rewards[i, first_negative_index: final_negative_index] = token_level_scores[i] + length_reward[i, first_negative_index: final_negative_index]
        
        final_token_advantages = torch.zeros_like(final_token_rewards)
        # we should separate to compute the baseline 
        
        id2score = defaultdict(list)
        id2response_mask = defaultdict(list)
        id2mean = {}
        id2std = {}
        
        
        with torch.no_grad():
            bsz = final_token_rewards.shape[0]
            for i in range(bsz):
                id2score[index[i]].append(final_token_rewards[i])
                id2response_mask[index[i]].append(response_mask[i])
            for idx in id2score:
                if len(id2score[idx]) == 1:
                    id2mean[idx] = torch.tensor(0.0)
                    id2std[idx] = torch.tensor(1.0)
                else:

                    current_batch_token_rewards = torch.stack(id2score[idx], 0)
                    current_batch_token_rewards_mean = current_batch_token_rewards.mean(dim=0)
                    current_batch_token_rewards_sigma = current_batch_token_rewards.std(dim=0)
                        
                    
                    id2mean[idx] = current_batch_token_rewards_mean
                    id2std[idx] = current_batch_token_rewards_sigma 
            
            for i in range(bsz):
                final_token_advantages[i] = (final_token_rewards[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            final_token_advantages = final_token_advantages * response_mask
        
        return final_token_advantages, final_token_rewards        

def compute_rpp_proc_length_advantage_return(token_level_rewards: torch.Tensor,
                                              chunk_aware_adv: torch.Tensor,
                                              response_mask: torch.Tensor, 
                                              index,
                                              chunk_config):
    epsilon = 1e-6
    token_level_scores = token_level_rewards.sum(-1)
    response_length = response_mask.sum(-1) # [B, ]
    max_seq_len = response_mask.shape[-1] 
    bsz = token_level_rewards.shape[0]
    
    
    
    with torch.no_grad():
        if chunk_config.no_zero_adv:
            token_correctness_scores = token_level_scores.unsqueeze(-1).repeat(1, token_level_rewards.shape[1])
        else:
            token_correctness_scores = token_level_scores.unsqueeze(-1) * response_mask
            
        final_token_rewards = token_correctness_scores.clone()
            
        # detect chunk_aware_adv the first negative value index
        length_reward = torch.zeros_like(token_level_rewards)
        for i in range(bsz):
            if token_level_scores[i] > 0:
                cur_seq_len = response_length[i]
                length_reward[i, :cur_seq_len] = chunk_config.length_reward_beta * (torch.arange(0, cur_seq_len, dtype=length_reward.dtype)-cur_seq_len) / max_seq_len

                cur_chunk_adv = chunk_aware_adv[i]
                # find the first index whose cur_chunk_adv is negative, if found, all values after this index should add corresponding length_reward, if not found, remain the final_token_rewards as token_correctness_scores
                negative_index = (cur_chunk_adv < 0).nonzero(as_tuple=True)[0]
                if negative_index.numel() > 0:
                    first_negative_index = negative_index[0].item()
                    final_negative_index = negative_index[-1].item() + 1
                    final_token_rewards[i, first_negative_index: final_negative_index] += length_reward[i, first_negative_index: final_negative_index]
        
        final_token_advantages = torch.zeros_like(final_token_rewards)
        # we should separate to compute the baseline 
        
        id2score = defaultdict(list)
        id2response_mask = defaultdict(list)
        id2mean = {}
        id2std = {}

        bsz = final_token_rewards.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(final_token_rewards[i])
            id2response_mask[index[i]].append(response_mask[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            else:
                if chunk_config.no_zero_adv:
                    current_batch_token_rewards = torch.stack(id2score[idx], 0)
                    current_batch_token_rewards_mean = current_batch_token_rewards.mean(dim=0)
                    # current_batch_token_rewards_sigma = current_batch_token_rewards.std(dim=0)
                    
                else:
                    current_batch_token_rewards = torch.stack(id2score[idx], 0)
                    current_batch_response_mask = torch.stack(id2response_mask[idx], 0)
                    current_batch_token_rewards_mean = verl_F.masked_mean(current_batch_token_rewards, current_batch_response_mask, axis=0)
                    # manually compute the std, using the std formula: \sigma^2 = E(x^2) - E(x)^2
                    # current_batch_token_rewards_square = current_batch_token_rewards.pow(2)
                    # current_batch_token_rewards_square_mean = verl_F.masked_mean(current_batch_token_rewards_square, current_batch_response_mask, axis=0)
                    # current_batch_token_rewards_sigma = torch.sqrt(current_batch_token_rewards_square_mean - current_batch_token_rewards_mean.pow(2) + epsilon)
                id2mean[idx] = current_batch_token_rewards_mean
                # id2std[idx] = current_batch_token_rewards_sigma 
        
        for i in range(bsz):
            final_token_advantages[i] = (final_token_rewards[i] - id2mean[index[i]])
        final_token_advantages = verl_F.masked_whiten(final_token_advantages, response_mask)
        
        return final_token_advantages, final_token_rewards         
    




def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, beta=0.1, chunk_config=None):
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch['response_mask'] = compute_response_mask(data)
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == AdvantageEstimator.GAE:
        values = data.batch['values']
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch['token_level_rewards'],
            values=data.batch['values'],
            response_mask=data.batch['response_mask'],
            gamma=gamma,
            lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.GRPO:
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch['token_level_rewards'],
            response_mask=data.batch['response_mask'],
            index=data.non_tensor_batch['uid'])
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    
    elif adv_estimator == AdvantageEstimator.GRPO_PROC_LENGTH:    
        advantages, returns = compute_grpo_proc_length_advantage_return(
            token_level_rewards=data.batch['token_level_rewards'],
            chunk_aware_adv= data.batch['chunk_aware_adv'],
            response_mask=data.batch['response_mask'],
            index=data.non_tensor_batch['uid'],
            chunk_config=chunk_config
            )
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns

    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE:
        advantages, returns = core_algos.compute_reinforce_plus_plus_baseline_outcome_advantage(
            token_level_rewards=data.batch['token_level_rewards'],
            response_mask=data.batch['response_mask'],
            index=data.non_tensor_batch['uid'])
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        if "values" in data.batch:
            advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_process_advantage(
                token_level_rewards=data.batch['token_level_rewards'],
                process_token_level_rewards=data.batch['values'],
                response_mask=data.batch['response_mask'],
                gamma=gamma,
                lam=beta)
            
        else:
            advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
                token_level_rewards=data.batch['token_level_rewards'],
                response_mask=data.batch['response_mask'],
                gamma=gamma)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_WITH_REFLECT:
        # TODO : implement REINFORCE_WITH_REFLECT
        pass
    elif adv_estimator == AdvantageEstimator.REMAX:
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=data.batch['token_level_rewards'],
            reward_baselines=data.batch['reward_baselines'],
            response_mask=data.batch['response_mask'])

        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=data.batch['token_level_rewards'],
            response_mask=data.batch['response_mask'],
            index=data.non_tensor_batch['uid'])
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data




@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    if name not in timing_raw:
        
        timing_raw[name] = timer.last
    else:
        timing_raw[name] += timer.last


class ValidRayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 processor=None,
                 reward_fn=None,
                 val_reward_fn=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        if not self.config.trainer.default_local_dir.startswith("s3://"):
            self.config.trainer.default_local_dir = os.path.expanduser(self.config.trainer.default_local_dir)

        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'
        # print(Role.ActorRollout) # Role.ActorRollout
        # print(role_worker_mapping.keys()) # dict_keys([<Role.ActorRollout: 2>, <Role.Critic: 3>])
        # print(Role.ActorRollout in role_worker_mapping)
        
        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping

        
        self.ray_worker_group_cls = ray_worker_group_cls
        self.validation_generations_logger = ValidationGenerationsLogger()
        self.train_generations_logger = TrainGenerationsLogger()
        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
                AdvantageEstimator.GRPO, AdvantageEstimator.GRPO_CHUNK, AdvantageEstimator.REINFORCE_PLUS_PLUS, AdvantageEstimator.REMAX, AdvantageEstimator.GRPO_HYBRID, AdvantageEstimator.GRPO_PROC, AdvantageEstimator.GRPO_PROC_LENGTH, AdvantageEstimator.RPP_PROC_LENGTH,
                AdvantageEstimator.RLOO, AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader()
        self.kappa = 0.0
        self.last_mean_nrp_ratio = 0.5

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")
        
    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with _timer("dump_rollout_generations", timing_raw):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )
            
    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, \
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(
                        f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(
                        f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. "
                        f"Please remove '{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
                    )

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.actor.ppo_micro_batch_size,
                                     config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.actor")

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                                         config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                                         "actor_rollout_ref.ref")

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.rollout")

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu,
                                     "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu,
                                     "reward_model")

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean", "seq-mean-token-sum", "seq-mean-token-mean"
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print(f"NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get('ulysses_sequence_parallel_size', 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == 'fsdp':
            if config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1) > 1 or \
                    config.actor_rollout_ref.ref.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.actor_rollout_ref.model.use_remove_padding, \
                    "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == 'fsdp':
            if config.critic.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.critic.model.use_remove_padding, \
                    "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get('val_batch_size', None) is not None:
            print(
                f"WARNING: val_batch_size is deprecated. Validation datasets are sent to inference engines as a whole batch, which will schedule the memory themselves."
            )

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, \
                "validation gen temperature should be greater than 0 when enabling do_sample"

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self):
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.utils.import_utils import load_extern_type
        if "custom_cls" in self.config.data and self.config.data.custom_cls.get("path", None) is not None:
            dataset_cls = load_extern_type(self.config.data.custom_cls.path, self.config.data.custom_cls.name)
            if not issubclass(dataset_cls, Dataset):
                raise TypeError(f"The custom dataset class '{self.config.data.custom_cls.name}' from "
                                f"'{self.config.data.custom_cls.path}' must inherit from torch.utils.data.Dataset")
        else:
            dataset_cls = RLHFDataset

        self.train_dataset = dataset_cls(
            data_files=self.config.data.train_files,
            tokenizer=self.tokenizer,
            processor=self.processor,
            config=self.config.data,
        )

        # use sampler for better ckpt resume
        

        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get('seed', 1))
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = StatefulDataLoader(dataset=self.train_dataset,
                                                   batch_size=self.config.data.get('gen_batch_size',
                                                                                   self.config.data.train_batch_size),
                                                   num_workers=8,
                                                   drop_last=True,
                                                   collate_fn=collate_fn,
                                                   sampler=sampler)

        self.val_dataset = dataset_cls(
            data_files=self.config.data.val_files,
            tokenizer=self.tokenizer,
            processor=self.processor,
            config=self.config.data,
        )
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            # Validation datasets are sent to inference engines as a whole batch,
            # which will schedule the memory themselves.
            batch_size=len(self.val_dataset),
            num_workers=8,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn)

        assert len(self.train_dataloader) >= 1
        assert len(
            self.val_dataloader
        ) == 1, "Validation dataloader must have a single batch, which inference engines will schedule the memory themselves."

        print(f'Size of train dataloader: {len(self.train_dataloader)}')

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)
    
    def _maybe_log_train_generations(self, inputs, outputs, scores):
        generations_to_log = self.config.trainer.log_train_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.train_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        

        
        
        # record the acc about one problem, and corresponding response length
        id2score = defaultdict(list)
        id2response_length = defaultdict(list)
        
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            test_batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(test_batch.batch))],
                                                             dtype=object)
            # repeat test batch
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n,
                                           interleave=True)

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                return {}

            # Store original inputs
            input_ids = test_batch.batch['input_ids']
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            if 'multi_modal_inputs' in test_batch.non_tensor_batch.keys():
                test_gen_batch = test_batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs'],
                )
            else:
                test_gen_batch = test_batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids'],
                )

            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                'validate': True,
            }
            print(f'test_gen_batch meta info: {test_gen_batch.meta_info}')

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print('validation generation end')

            # Store generated outputs
            output_ids = test_output_gen_batch.batch['responses']
            attention_mask = test_output_gen_batch.batch['attention_mask']
            response_length = attention_mask.sum(-1).tolist()
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    if len(lst) != 0:
                        reward_extra_infos_dict[key].extend(lst)
            reward_extra_infos_dict['response_length'].extend(response_length)
            
            # record the scores for the same uid
            for uid, score, response_len in zip(test_batch.non_tensor_batch['uid'], scores, response_length):
                id2score[uid].append(score)
                id2response_length[uid].append(response_len)

            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))

        
        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if var_name == core_var and any(
                            metric_name.startswith(pfx)
                            for pfx in ["mean", "std", "maj", "best"]) and f"@{n_max}/" in metric_name:
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        return metric_dict
    
    def log_train(self, batch):
        data = batch
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            
            # score = data_item.batch['original_token_level_score'].sum(-1).detach().cpu().item()
            score = data_item.batch['token_level_scores'].sum(-1).detach().cpu().item() 
            # enforce_nothinking = data_item.batch['enforce_nothinking'].item()
            # if enforce_nothinking < valid_response_length - 1:
            #     manual_response = True 
            # else:
            #     manual_response = False

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            sample_inputs.append(prompt_str)
            sample_outputs.append(response_str)
            
            sample_scores.append(score)
        
        self._maybe_log_train_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)   

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool,
                                                ray_cls_with_init=worker_dict_cls,
                                                **wg_kwargs)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir,
                                                f'global_step_{self.global_steps}')

        print(f'local_global_step_folder: {local_global_step_folder}')
        os.makedirs(local_global_step_folder, exist_ok=True)
        # save kappa to os.path.join(local_global_step_folder, 'kappa.txt')
        utils_info = {
            "kappa": self.kappa,
            "last_mean_nrp_ratio": self.last_mean_nrp_ratio
        }
        client.write_json(utils_info, os.path.join(local_global_step_folder, 'utils_info.json'))
        actor_local_path = os.path.join(local_global_step_folder, 'actor')

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'actor')

        remove_previous_ckpt_in_save = self.config.trainer.get('remove_previous_ckpt_in_save', False)
        if remove_previous_ckpt_in_save:
            print(
                'Warning: remove_previous_ckpt_in_save is deprecated, set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead'
            )
        max_actor_ckpt_to_keep = self.config.trainer.get('max_actor_ckpt_to_keep',
                                                         None) if not remove_previous_ckpt_in_save else 1
        max_critic_ckpt_to_keep = self.config.trainer.get('max_critic_ckpt_to_keep',
                                                          None) if not remove_previous_ckpt_in_save else 1

        self.actor_rollout_wg.save_checkpoint(actor_local_path,
                                              actor_remote_path,
                                              self.global_steps,
                                              max_ckpt_to_keep=max_actor_ckpt_to_keep)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, 'critic')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'critic')
            self.critic_wg.save_checkpoint(critic_local_path,
                                           critic_remote_path,
                                           self.global_steps,
                                           max_ckpt_to_keep=max_critic_ckpt_to_keep)

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, 'data.pt')
        dataloader_state_dict = self.train_dataloader.state_dict()
        client.save_checkpoint(dataloader_state_dict, dataloader_local_path)
        # torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir,
                                                           'latest_checkpointed_iteration.txt')
        client.write_text(str(self.global_steps), local_latest_checkpointed_iteration)
       


    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == 'disable':
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError('load from hdfs is not implemented yet')
        else:
            if not self.config.trainer.default_local_dir.startswith("s3://"):
                checkpoint_folder = os.path.expanduser(self.config.trainer.default_local_dir)  # TODO: check path
                if not os.path.isabs(checkpoint_folder):
                    working_dir = os.getcwd()
                    checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            else:
                checkpoint_folder = self.config.trainer.default_local_dir

            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == 'auto':
            if global_step_folder is None:
                print('Training from scratch')
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert 'global_step_' in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_from_path
                if not global_step_folder.startswith("s3://"):
                    if not os.path.isabs(global_step_folder):
                        working_dir = os.getcwd()
                        global_step_folder = os.path.join(working_dir, global_step_folder)
                
                
        print(f'Load from checkpoint folder: {global_step_folder}')
        utils_info = client.read_json(os.path.join(global_step_folder, 'utils_info.json'))
        self.kappa = utils_info['kappa']
        self.last_mean_nrp_ratio = utils_info['last_mean_nrp_ratio']
        # self.kappa = float(client.read_txt(os.path.join(global_step_folder, 'kappa.txt')))
        # set global step
        self.global_steps = int(global_step_folder.split('global_step_')[-1])

        print(f'Setting global step to {self.global_steps}')
        print(f'Resuming from {global_step_folder}')

        actor_path = os.path.join(global_step_folder, 'actor')
        critic_path = os.path.join(global_step_folder, 'critic')
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path,
                                              del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path,
                                           del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, 'data.pt')
        
        # if os.path.exists(dataloader_local_path):
        if client.exists(dataloader_local_path):
            dataloader_state_dict = client.load_checkpoint(dataloader_local_path, weights_only=False)
            # dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict,)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch['attention_mask'].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)
    

    def maybe_replace(self, batch: DataProto, metrics):
        if (not self.config.actor_rollout_ref.reflect_think.enable) or (self.config.actor_rollout_ref.reflect_think.reflect_type != 'replace'):
            return batch
        # group response based on batch.non_tensor_batch['response_uid']
        
        token_level_scores = batch.batch['token_level_rewards'].sum(-1)
        should_modify = batch.batch['should_modify']
        # modify_loc = batch.batch['modify_loc']
        
        # ref_acc = batch.non_tensor_batch['ref_acc']
        bsz = token_level_scores.shape[0]
        # uids = batch.non_tensor_batch['uid']
        
        # id2score = batch.meta_info['id2score']
        # manual_id2score = batch.meta_info['manual_id2score']
        
        uid2idx = defaultdict(list)
        uid_group_scores = defaultdict(list)
        
        
        
        for i in range(bsz):
            uid = batch.non_tensor_batch['response_uid'][i]
            uid2idx[uid].append(i)
            uid_group_scores[uid].append(token_level_scores[i])
        # print("uid2idx:", uid2idx)
        # print("group token_level_scores:", uid_group_scores)
        # for each group, select the idx whose token_level_scores[idx] is the largest among the sequence.
        # if all scores are equal, select the idx such that should_modify[idx] = False
        with torch.no_grad():
            new_batch_indices = []
            for uid, idxs in uid2idx.items():
                assert len(idxs) == 2, f"It should only contain two responses for each uid, got {len(idxs)} responses."
                
                
                scores = token_level_scores[idxs]
                should_modify_idxs = should_modify[idxs]
                assert should_modify_idxs.sum() == 1, f"Only one response should be modified, got {should_modify_idxs.sum()} responses to modify."
                # max_score = scores.max()
                for i, idx in enumerate(idxs):
                    if should_modify_idxs[i]:
                        new_batch_indices.append(idx)
                        # normal_idx = idxs[len(idxs) - i - 1]
                        # normal_score = scores[normal_score]
                        
                        
        
        selection_indices = torch.tensor(new_batch_indices, dtype=torch.long)
        new_batch = batch.select_idxs(selection_indices)
        # print how many responses are should_modify
        # print("Current batch size: ", new_batch.batch['responses'].shape[0])
        
        # print("Number of normal responses: ", (~new_batch.batch['should_modify']).sum().item())
        # print("Number of should_modify responses: ", new_batch.batch['should_modify'].sum().item())
        if self.config.trainer.balance_batch:
            self._balance_batch(new_batch, metrics=metrics)
        # assert enforce_nothinking.shape[0] == 2 * selection_indices.shape[0], f"The new batch {selection_indices.shape[0]} should have exactly half of the original batch {enforce_nothinking.shape[0]}."
        return new_batch
    
    def may_compute_normal_score(self, batch: DataProto):
        if self.config.algorithm.adv_estimator not in [AdvantageEstimator.GRPO_REFLECT, AdvantageEstimator.GRPO_TWICE]:
            return 
        id2score = defaultdict(list)
        manual_id2score = defaultdict(list)
        index = batch.non_tensor_batch['uid']
        should_modify = batch.batch['should_modify']
        scores = batch.batch["token_level_rewards"].sum(dim=-1)
        
        
        # print(scores)
        with torch.no_grad():
            bsz = scores.shape[0]
            for i in range(bsz):
                if not should_modify[i]:
                    id2score[index[i]].append(max(0.0,scores[i]))
                else:
                    manual_id2score[index[i]].append(max(0.0,scores[i]))
        batch.meta_info['id2score'] = id2score
        batch.meta_info['manual_id2score'] = manual_id2score
        pass

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        # total_training_steps = self.total_training_steps - self.global_steps
        self.global_steps += 1
        last_val_metrics = None


        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        
        for epoch in range(self.config.trainer.total_epochs):
            pprint(f'Training Epoch {epoch + 1}/{self.config.trainer.total_epochs}')
            cur_batch_kept_num = 0
            for batch_dict in self.train_dataloader:
                if self.config.trainer.total_steps is not None and self.global_steps > self.config.trainer.total_steps:
                    pprint(f'Reached total steps: {self.global_steps}, stopping training.')
                    pprint(f'Final validation metrics: {val_metrics}')
                    feishu_message = {"exp_name": self.config.trainer.experiment_name, **val_metrics}
                    send_feishu_message(feishu_message)
                    progress_bar.close()
                    return
                pprint(f'Current progress: {self.global_steps}/{self.total_training_steps}')
                metrics = {}
                # timing_raw = {}

                # batch: DataProto = DataProto.from_single_dict(batch_dict)
                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1
                
                # pop those keys for generation
                if 'multi_modal_inputs' in new_batch.non_tensor_batch.keys():
                    gen_batch = new_batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs'],
                    )
                else:
                    gen_batch = new_batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['raw_prompt_ids'],
                    )

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer('step', timing_raw):
                    # generate a batch
                    with _timer('gen', timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer('gen_max', timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info['do_sample'] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            new_batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(new_batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            new_batch.batch['reward_baselines'] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    # batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                    #                                          dtype=object)
                    # # repeat to align with repeated responses in rollout
                    # batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    # batch = batch.union(gen_batch_output)
                    new_batch.non_tensor_batch['uid'] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object)
                    # repeat to align with repeated responses in rollout
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)
                    with _timer('reward', timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        try:
                            reward_result = self.reward_fn(new_batch, return_dict=True)
                            reward_tensor = reward_result['reward_tensor']
                            reward_extra_infos_dict = reward_result['reward_extra_info']
                        except Exception as e:
                            print(f'Error in reward_fn: {e}')
                            reward_tensor = self.reward_fn(new_batch)
                            reward_extra_infos_dict = {}

                        new_batch.batch['token_level_scores'] = reward_tensor

                        print(f'{list(reward_extra_infos_dict.keys())=}')
                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update({
                                k: np.array(v) for k, v in reward_extra_infos_dict.items()
                            })

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            new_batch, kl_metrics = apply_kl_penalty(new_batch,
                                                                     kl_ctrl=self.kl_ctrl_in_reward,
                                                                     kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(
                                kl_metrics)  # TODO: This will be cleared if we use multiple genenration batches
                        else:
                            new_batch.batch['token_level_rewards'] = new_batch.batch['token_level_scores']
                    
                    
                    if not self.config.algorithm.filter_groups.enable:
                        batch = new_batch
                    else:  # NOTE: When prompts after filtering is less than train batch size, we skip to the next generation batch
                        metric_name = self.config.algorithm.filter_groups.metric
                        # if metric_name == "seq_final_reward":
                        #     # Turn to numpy for easier filtering
                        #     new_batch.non_tensor_batch["seq_final_reward"] = new_batch.batch['token_level_scores'].sum(
                        #         dim=-1).numpy()
                        # elif metric_name == "seq_reward":
                        new_batch.non_tensor_batch["seq_reward"] = new_batch.batch['token_level_scores'].sum(
                                dim=-1).numpy()

                        # Collect the sequence reward for each trajectory
                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(new_batch.non_tensor_batch['uid'],
                                                   new_batch.non_tensor_batch[metric_name]):
                            prompt_uid2metric_vals[uid].append(metric_val)
                        if not self.config.algorithm.filter_groups.enable_kappa:
                            prompt_uid2metric_std = {}
                            for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                                prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

                            kept_prompt_uids = [
                                uid for uid, std in prompt_uid2metric_std.items()
                                if std > 0 or len(prompt_uid2metric_vals[uid]) == 1
                            ]
                            
                        else:
                            prompt_uid2metric_std = {}
                            for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                                prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)
                            prompt_uid2metric_mean = {}
                            for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                                prompt_uid2metric_mean[prompt_uid] = np.mean(metric_vals)
                            kept_prompt_uids = [
                                uid for uid in prompt_uid2metric_std.keys() 
                                if prompt_uid2metric_std[uid] > 0
                            ]
                            
                            all_correct_prompt_uids = [
                                uid for uid in prompt_uid2metric_std.keys() 
                                if prompt_uid2metric_mean[uid] > 0 and prompt_uid2metric_std[uid] == 0
                            ]
                            if self.kappa > 0:
                                cur_batch_kept_num += len(kept_prompt_uids) 
                                
                                num_could_append = int(self.kappa * cur_batch_kept_num / (1 - self.kappa)) # 容易导致无法得到num_could_append的数量，
                                # 应该要计算kep_prompt_uid的迄今为止的数量，若此值一直为0，则一直累加
                                # num_could_append = int(len(list(prompt_uid2metric_vals.keys())) * self.kappa)
                                if num_could_append > 0:
                                    cur_batch_kept_num = 0
                                
                                num_could_append = min(num_could_append, len(all_correct_prompt_uids))
                            

                                kept_prompt_uids.extend(random.sample(all_correct_prompt_uids, k=num_could_append))
                                print("Append all correct samples: ", num_could_append)
                        random.shuffle(kept_prompt_uids)
                            

                        
                        num_prompt_in_batch += len(kept_prompt_uids)

                        kept_traj_idxs = []
                        for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch['uid']):
                            if traj_from_prompt_uid in kept_prompt_uids:
                                kept_traj_idxs.append(idx)

                        new_batch = new_batch[kept_traj_idxs]
                    
                        
                        if batch is None:
                            batch = new_batch
                        else:
                            batch = DataProto.concat([batch, new_batch])

                        prompt_bsz = self.config.data.train_batch_size
                        if num_prompt_in_batch < prompt_bsz:
                            print(f'{num_prompt_in_batch=} < {prompt_bsz=}')
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f'{num_gen_batches=}. Keep generating...')
                                continue
                            else:
                                raise ValueError(
                                    f'{num_gen_batches=} >= {max_num_gen_batches=}. Generated too many. Please check your data.'
                                )
                        else:
                            # Align the batch
                            traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                            batch = batch[:traj_bsz]

                    batch.batch['original_token_level_score'] = batch.batch['token_level_scores'].sum(-1, keepdim=True)

                    

                    
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # recompute old_log_probs
                    with _timer('old_log_prob_entropy', timing_raw):
                        old_log_prob_entropy = self.actor_rollout_wg.compute_log_prob_entropy(batch)
                        # old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        batch = batch.union(old_log_prob_entropy)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)
                    
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                    with _timer('adv', timing_raw):
                        
                        batch.batch['response_mask'] = compute_response_mask(batch)
                        chunk_info = self.reward_fn.compute_chunk_adv(batch)
                        batch.batch['chunk_aware_adv'] = chunk_info['chunk_aware_adv']
                        batch.non_tensor_batch.update({k: np.array(v) for k, v in chunk_info.items() if k != 'chunk_aware_adv'})
                        self.log_train(batch)

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n,
                                                  chunk_config=self.config.chunk_config)
                        metrics.update(compute_chunk_metrics(batch=batch,))
                        if self.last_mean_nrp_ratio != 0:
                            
                            self.kappa = self.kappa + (metrics['chunk_aware/first_correct_chunk_ratio_mean'] - self.last_mean_nrp_ratio) * self.config.algorithm.filter_groups.kappa_lr
                            
                        self.last_mean_nrp_ratio = metrics['chunk_aware/first_correct_chunk_ratio_mean']
                            # self.last_mean_nrp_ratio = 
                            # self.kappa = max(self.kappa, 0)

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        (is_last_step or  (self.global_steps % self.config.trainer.test_freq == 0 and self.global_steps > 1)):
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if is_last_step or (self.config.trainer.save_freq > 0 and \
                            (self.global_steps % self.config.trainer.save_freq == 0 and self.global_steps > 0)):
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)  # clear timing

                metrics["train/num_gen_batches"] = num_gen_batches
                metrics['train/kappa'] = self.kappa
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0
                cur_batch_kept_num = 0

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f'Final validation metrics: {last_val_metrics}')
                    # feishu_message = {**last_val_metrics, "exp_name": self.config.trainer.experiment_name,}
                    # send_feishu_message(feishu_message)
                    progress_bar.close()
                    return 

                progress_bar.update(1)
                self.global_steps += 1
