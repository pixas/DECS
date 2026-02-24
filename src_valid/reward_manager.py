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
from itertools import chain

from verl import DataProto
from verl.utils.reward_score import _default_compute_score
from evaluation.models.base_model import APIModel
import torch
from collections import defaultdict
from transformers import AutoTokenizer 

def split_based_on_tokens(response_ids: list[int], predefined_split_tokens, newline_tokens, star_tokens):
    """
    Splits the sentence based on the predefined reflect words.
    """
    first_response = response_ids
    splitted_response = []
    # Only proceed if first_response has content. If first_response is an empty string, 
    # splitted_response should remain empty, which is the default.
    if first_response:
        all_split_points = []
        # Create a set for O(1) lookup
        split_tokens_set = set(predefined_split_tokens)
        
        # Single pass through first_response
        for i in range(len(first_response)):
            for token in split_tokens_set:
                
                if first_response[i] == token:
                    all_split_points.append(i)
                    break  # Found a match at this position, no need to check other tokens

        # Get unique split points and sort them
        unique_sorted_split_points = sorted(list(set(all_split_points)))

        if not unique_sorted_split_points:
            # No reflect words found, the whole response is a single segment
            # we should cut the contents until ** tokens 
            try:
                end_thinking_loc = first_response.index(star_tokens)  # ** token
            except:
                end_thinking_loc = -1
            if end_thinking_loc != -1:
                cut_first_response = first_response[:first_response.index(star_tokens)]
            else:
                cut_first_response = first_response
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
    # for the lask segment, we need to cut until **, otherwise the final answer will also be punished, which is bad
            end_thinking_loc = -1 
            try:
                end_thinking_loc = splitted_response[-1].index(star_tokens)
            except ValueError:
                pass
            if splitted_response and end_thinking_loc != -1:
                splitted_response[-1] = splitted_response[-1][:end_thinking_loc]
    # for extremely short response (only one line), we need to merge it to the next segment with multi-lines
    # there will be a case that consecutive segments contain only one line, cache them until a segment with multi lines 
    # return splitted_response
    clean_splitted_response = []
    merge_cache = []

    for i, response in enumerate(splitted_response):
        # see if the response contains \n token or \n\n token that is not in the start or end of the response 
        # strip_response = [ for x in response]
        start_line = True 
        multi_line = False 
        for token in response:
            if token in newline_tokens:
                if start_line:
                    continue 
                else:
                    multi_line = True 
                    break
            else:
                start_line=False 
        end_line = True 
        end_multi_line = False 
        for j, token in enumerate(response[::-1]):
            if token in newline_tokens:
                if end_line:
                    continue 
                else:
                    end_multi_line = True 
                    break
            else:
                end_line = False 
        actual_multi_line = multi_line and end_multi_line
        
        
        if not actual_multi_line and i < len(splitted_response) - 1:
            # merge it with the next segment
            merge_cache.append(response)
        else:
            tobe_merged_contents = list(chain(*merge_cache))
            clean_splitted_response.append(tobe_merged_contents + response)
            merge_cache = []
    return clean_splitted_response


    
class ChunkAwareGRPO:
    judge_prompt_template = "Given a math problem and a segement of a long reasoning process to solve the problem, your task is to identify whether this segment has presented a correct final answer. If the segment contains information that can serve as the final answer to the problem and the answer is semantically correct when referring to the ground truth, simply explain the reason and output \\boxed{{yes}}. Otherwise, directly output \\boxed{{no}}.\n**Problem**:\n{problem}\n**Reasoning segment**:\n{segment}\n**Ground Truth**: {answer}"
    def __init__(self, split_tokens, newline_tokens, star_tokens, chunk_config, 
                 token_weights={"high_entropy": 1.5, "default": 0.8},
                 pos_decay=0.5, ):
        """
        初始化 Chunk-Aware GRPO 算法
        
        参数:
        alpha (float): a类chunk的奖励系数 (首次正确)
        beta (float): b类chunk的奖励系数 (必要尝试)
        gamma (float): c/d类chunk的惩罚系数 (冗余反思)
        token_weights (dict): 不同token类型的权重
        pos_decay (float): chunk内位置衰减系数
        """
        self.chunk_config = chunk_config 
        
        
        
        self.alpha = chunk_config.alpha
        self.beta = chunk_config.beta
        self.gamma = chunk_config.gamma
        self.token_weights = token_weights
        self.pos_decay = pos_decay
        self.split_tokens = split_tokens
        all_judge_url = chunk_config.judge_url.split(";")
        self.judge_model_tokenizer = AutoTokenizer.from_pretrained(chunk_config.judge_model)
        self.judge_model_list = [APIModel(chunk_config.judge_model, url) for url in all_judge_url]
        self.allow_incorrect = chunk_config.allow_incorrect
        self.newline_tokens = newline_tokens
        self.star_tokens = star_tokens
        # self.judge_model = APIModel(judge_model, judge_url)
        
        
        # 缓存计算结果
        self.chunk_cache = {}
        self.class_cache = {}
        
    def split_into_chunks(self, response_ids):
        splitted_chunks = split_based_on_tokens(response_ids, self.split_tokens, self.newline_tokens, self.star_tokens)
        splitted_indices = []
        last = 0
        for chunk in splitted_chunks:
            splitted_indices.append(len(chunk)+last)
            last = len(chunk)+last
        return splitted_chunks, splitted_indices
    
    def contains_correct_answer(self, tokenizer, chunk, problem, true_answer):
        
        
        segment = tokenizer.decode(chunk, skip_special_tokens=True)
        prompt = self.judge_prompt_template.format(problem=problem, segment=segment, answer=true_answer)
        
        result = self.judge_model_list[0](prompt, max_new_tokens=5)[0][0]
        return "yes" in result 
    
    
    def _process_chunk(self, args: tuple) -> list:
        """
        处理一个数据块的进程函数
        :param args: 元组 (model_idx, prompts, max_new_tokens)
        :return: 该进程处理的结果列表
        """
        model_idx, prompts, max_new_tokens = args
        return self.judge_model_list[model_idx](prompts, max_new_tokens=max_new_tokens)
    
    def batch_classify_all_chunks(self, tokenizer, all_chunks, problems, true_answers, is_correct):
        """,
        Batched version of classify_all_chunks that processes multiple responses at once
        
        Args:
            tokenizer: Tokenizer instance
            all_chunks: List of lists, where each inner list contains chunks for one response
            problems: List of problem strings (one per response)
            true_answers: List of true answer strings (one per response)
            is_correct: List of booleans indicating if the response is correct
        
        Returns:
            List of lists containing chunk types for each response
        """
        # Flatten all chunks while keeping track of which response they belong to
        flat_chunks = []
        response_indices = []
        chunk_indices = []
        
        for resp_idx, chunks in enumerate(all_chunks):
            if not is_correct[resp_idx] and (not self.allow_incorrect):
                continue
            for chunk_idx, chunk in enumerate(chunks):
                flat_chunks.append(chunk)
                response_indices.append(resp_idx)
                chunk_indices.append(chunk_idx)
        
        # Batch decode all chunks at once
        all_segments = tokenizer.batch_decode(flat_chunks, skip_special_tokens=True)
        
        # Create all prompts
        all_prompts = [
            self.judge_prompt_template.format(
                problem=problems[resp_idx],
                segment=segment,
                answer=true_answers[resp_idx]
            )
            for resp_idx, segment in zip(response_indices, all_segments)
        ]
        
        
        # Single API call for all chunks

        try:
            try:
                enc = self.judge_model_tokenizer(all_prompts, add_special_tokens=False)
                ids = enc.input_ids if hasattr(enc, 'input_ids') else enc['input_ids']
                prompt_token_lens = [len(x) for x in ids]
            except Exception as e:
                print(e)
                prompt_token_lens = [len(tokenizer.encode(p, add_special_tokens=False)) for p in all_prompts]
        except Exception as e2:
            print(e2)
            prompt_token_lens = [0] * len(all_prompts)

        max_input_tokens = 17990
        if max_input_tokens is None:
            max_input_tokens = getattr(self.judge_model_list[0], 'max_input_tokens', 0) or 8192
        # print("Current prompt lengths")
        # print(prompt_token_lens)
        valid_indices = [i for i, L in enumerate(prompt_token_lens) if L <= max_input_tokens]
        
        results = [['no']] * len(all_prompts)

        if valid_indices:
            valid_prompts = [all_prompts[i] for i in valid_indices]
            judged = self.judge_model_list[0](valid_prompts, max_new_tokens=5)
            # judged = [j[0] for j in judged]
            for idx, res in zip(valid_indices, judged):
                results[idx] = res

        all_result = results
        # all_result = self.judge_model_list[0](all_prompts, max_new_tokens=5)

        
        #
        all_result = [i[0] for i in all_result]
        
        # Reconstruct results per response
        results_per_response = {}
        for resp_idx, chunk_idx, result in zip(response_indices, chunk_indices, all_result):
            if resp_idx not in results_per_response:
                results_per_response[resp_idx] = {}
            results_per_response[resp_idx][chunk_idx] = result
        
        # Process each response separately to determine chunk types
        all_chunk_types = []
        
        for resp_idx in range(len(all_chunks)):
            chunks = all_chunks[resp_idx]
            if not is_correct[resp_idx] and (not self.allow_incorrect):
                # 如果响应不正确且不允许不正确的响应，直接标记所有chunk为b类
                chunk_types = ['b'] * len(chunks)
                all_chunk_types.append(chunk_types)
                continue
            chunk_results = results_per_response[resp_idx]
            
            # Find first correct chunk for this response
            first_correct_idx = None
            chunk_types = [None] * len(chunks)
            
            
                
            
            for chunk_idx in range(len(chunks)):
                result = chunk_results[chunk_idx]
                
                if "yes" in result and first_correct_idx is None:
                    first_correct_idx = chunk_idx
                
                if "yes" in result and first_correct_idx == chunk_idx:
                    chunk_types[chunk_idx] = 'a'  # 首次正确
                elif "no" in result and first_correct_idx is None:
                    chunk_types[chunk_idx] = 'b'
                elif "no" in result and first_correct_idx is not None:
                    chunk_types[chunk_idx] = 'c'
                elif "yes" in result and first_correct_idx is not None and chunk_idx > first_correct_idx:
                    chunk_types[chunk_idx] = 'd'
                else:
                    chunk_types[chunk_idx] = 'b'
            
            all_chunk_types.append(chunk_types)
        
        return all_chunk_types
                
    
    def get_delta(self, chunk_type, weight=1, entropy=None, correct=True):
        """
        获取chunk的delta值 (优势修正因子)
        
        参数:
        chunk_type (str): chunk类型 ('a', 'b', 'c', 'd')
        response_entropy (float): 响应的平均熵 (可选)
        
        返回:
        float: delta值
        """
        # 如果没有提供熵值，使用默认参数
        alpha = self.alpha 
        beta = self.beta if correct else 0.0  # 如果不正确，b类chunk不鼓励
        gamma = self.gamma * weight
        
        if entropy is not None:
            alpha = alpha * (1 + entropy)
            beta = beta * (1 + entropy)
            gamma = gamma / (1 + entropy)

        # 根据chunk类型返回delta
        if chunk_type == 'a':
            return alpha  # 强烈鼓励
        elif chunk_type == 'b':
            return beta   # 弱鼓励
        elif chunk_type in ('c', 'd'):
            return -gamma  # 抑制
        else:
            return 0.0  # 未知类型
    
    def position_weight(self, token_idx, chunk_start, chunk_length):
        """
        计算位置权重
        
        参数:
        token_idx (int): token在序列中的全局索引
        chunk_start (int): chunk在序列中的起始位置
        chunk_length (int): chunk长度
        token (str): token内容
        
        返回:
        float: 位置权重
        """
        # 计算token在chunk内的相对位置
        chunk_pos = token_idx - chunk_start
        relative_pos = chunk_pos / max(1, chunk_length - 1)  # 0到1之间
        
        # 位置衰减：开头权重高，结尾权重低
        pos_weight = 1.0 - self.pos_decay * relative_pos
        
        # 高熵token额外加权
        is_high_entropy = chunk_start != 0 and chunk_pos == 0
        # is_high_entropy = any(het.lower() in token.lower() for het in HIGH_ENTROPY_TOKENS)
        token_weight = self.token_weights["high_entropy"] if is_high_entropy else self.token_weights["default"]
        
        return token_weight * pos_weight
    

    def batch_compute_chunk_adv(self, response_ids, response_mask, chunk_aware_adv, ground_truths, response_entropies, problems, tokenizer, is_correct):
        """
        Batched version of chunk-aware adversarial computation
        
        Args:
            response_ids: [batch_size, max_seq_len] tensor of token IDs
            response_mask: [batch_size, max_seq_len] boolean mask of valid tokens
            chunk_aware_adv: [batch_size, max_seq_len] tensor to store results
            ground_truths: list of ground truth strings (length batch_size)
            response_entropies: [batch_size] tensor of response entropies
            problems: list of problem strings (length batch_size)
            tokenizer: tokenizer instance
            is_correct: [batch_size] tensor to store whether the response is correct
        """
        device = response_ids.device
        batch_size, max_seq_len = response_ids.shape
        
        # Step 1: Process each sequence to get chunks and split indices
        all_chunks = []
        all_split_indices = []
        all_valid_lengths = []
        
        # Pre-compute valid lengths and prepare chunk data
        for i in range(batch_size):
            valid_length = response_mask[i].sum().item()
            valid_tokens = response_ids[i, :valid_length].tolist()
            chunks, split_indices = self.split_into_chunks(valid_tokens)
            
            all_chunks.append(chunks)
            all_split_indices.append(split_indices)
            all_valid_lengths.append(valid_length)
        
        # Step 2: Batch classify all chunks across all sequences
        batch_chunk_types = self.batch_classify_all_chunks(tokenizer, all_chunks, problems, ground_truths, is_correct)
    
    
        # Step 3: Prepare batched tensors
        # Create token-to-chunk mapping and other properties
        token_to_chunk_id = torch.zeros((batch_size, max_seq_len), dtype=torch.long, device=device)
        token_delta = torch.zeros((batch_size, max_seq_len), device=device)
        # token_chunk_type = torch.zeros((batch_size, max_seq_len), dtype=torch.long, device=device)
        chunk_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool, device=device)
        
        first_correct_chunk_ratio = torch.zeros(batch_size, dtype=torch.float, device=device)  # Track first correct chunk index
        total_chunk_count = torch.zeros(batch_size, dtype=torch.float, device=device)  # Track total chunk count
        
        for i in range(batch_size):
            valid_length = all_valid_lengths[i]
            split_indices = all_split_indices[i]
            chunk_types = batch_chunk_types[i]
            correct = is_correct[i].item()
            
            if not split_indices:
                continue  # no chunks in this sequence
                
            chunk_starts = [0] + split_indices[:-1]
            chunk_ends = split_indices
            num_chunks = len(chunk_starts)
            
            # Create token-to-chunk mapping
            for chunk_id, (start, end) in enumerate(zip(chunk_starts, chunk_ends)):
                token_to_chunk_id[i, start:end] = chunk_id
            
            # Set chunk mask
            chunk_mask[i, :chunk_ends[-1]] = True
            
            # Get deltas for each chunk

            # chunk_num = len(chunk_types)
            total_chunk_count[i] = num_chunks
            
            # first_a_chunk_idx = None 
            weights_list = [1 for _ in range(num_chunks)]

                    
                    
            chunk_deltas = torch.tensor([self.get_delta(ct, weights_list[j], entropy=None) 
                                    for j, ct in enumerate(chunk_types)], device=device)
            if not correct and (not self.allow_incorrect):
                first_correct_chunk_ratio[i] = 0.0 
                # if not correct, we do not need to spread, currently
            else:
                first_correct_chunk_id = next((j for j, ct in enumerate(chunk_types) if ct == 'a'), -1)
                first_correct_chunk_ratio[i] = (first_correct_chunk_id + 1) / num_chunks
                if first_correct_chunk_id != -1:

                    # Spread chunk properties to tokens
                    for chunk_id in range(num_chunks):
                        if self.chunk_config.skip_last_chunk and chunk_id == num_chunks - 1:
                            continue
                        start, end = chunk_starts[chunk_id], chunk_ends[chunk_id]
                        end = max(start, end - self.chunk_config.skip_last_tokens ) if chunk_deltas[chunk_id] < 0 else end
                    
                        token_delta[i, start:end] = chunk_deltas[chunk_id] 
                        

                else:
                    # although we allow incorrect responses, but there is no correct answer among it
                    # we should not encourage anything, as the whole sequence contains no correct information
                    pass

        # Step 4: Compute relative positions and weights
        token_indices = torch.arange(max_seq_len, device=device).expand(batch_size, -1)
        chunk_starts = torch.zeros_like(token_to_chunk_id, device=device)
        chunk_lengths = torch.zeros_like(token_to_chunk_id, device=device)
        
        # Compute chunk starts and lengths for each token
        for i in range(batch_size):
            valid_length = all_valid_lengths[i]
            split_indices = all_split_indices[i]
            
            if not split_indices:
                continue
                
            chunk_starts_tensor = torch.tensor([0] + split_indices[:-1], device=device)
            chunk_ends_tensor = torch.tensor(split_indices, device=device)
            chunk_lengths_tensor = chunk_ends_tensor - chunk_starts_tensor
            
            chunk_starts[i, :valid_length] = chunk_starts_tensor[token_to_chunk_id[i, :valid_length]]
            chunk_lengths[i, :valid_length] = chunk_lengths_tensor[token_to_chunk_id[i, :valid_length]]
        
        # Compute relative positions
        relative_pos = (token_indices - chunk_starts).float()

        
        
        
        # High entropy token detection (first token of non-first chunks)
        is_first_token = (relative_pos == 0)
        is_not_first_chunk = (chunk_starts != 0)
        is_high_entropy = is_first_token & is_not_first_chunk
        

        pos_weight = 1 
        # pos_weight = is_high_entropy.float()
        
        # Token weights

        default_weight = torch.tensor(self.token_weights["default"], device=device)
        high_entropy_weight = torch.tensor(self.token_weights["high_entropy"], device=device)
        token_weight = torch.where(is_high_entropy, high_entropy_weight, default_weight)

        # set token_weight so that if there only exist positive values of token deltas, 
        # the final token_weight should be zero (i.e. no punishment and no encouragement for already good responses)
        # only_positive_deltas = (token_delta >= 0).all(dim=1, keepdim=True).repeat(1, token_weight.shape[1]) # [B, 1]
        # token_weight = torch.where(only_positive_deltas, torch.zeros_like(token_weight), token_weight)
        
        # Final weights
        final_weight = token_weight * pos_weight
        
        # Apply chunk mask and compute adversarial values
        chunk_aware_adv[:] = (token_delta * final_weight) * chunk_mask.float()
        
        return first_correct_chunk_ratio, total_chunk_count

class ChunkRewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, reward_config=None, compute_score=None, reward_fn_key='data_source',max_resp_len=-1, format_score=0, chunk_config=None, eval=False) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.compute_score = partial(self.compute_score, format_score=format_score)
        self.reward_fn_key = reward_fn_key
        self.reward_config = reward_config
        self.max_resp_len = max_resp_len
        self.eval = eval
        self.chunk_config = chunk_config
        # high entropy token's id 
        self.split_tokens = self.initialize_split_tokens()
        if self.chunk_config is not None:
            newline_words = ["\n", " \n", "\n\n", " \n\n", ".\n", ".\n\n"]
            newline_tokens = self.tokenizer(newline_words, add_special_tokens=False).input_ids 
            newline_tokens = [x[0] for x in newline_tokens]
            star_tokens = self.tokenizer("</think>", add_special_tokens=False).input_ids[0]
            self.chunk_aware_grpo = ChunkAwareGRPO(self.split_tokens, newline_tokens, star_tokens, self.chunk_config, 
                                                   token_weights={"high_entropy": self.chunk_config.high_token_weight, "default": self.chunk_config.default_token_weight})
        
    
    def initialize_split_tokens(self):
        split_words = ['Wait', ' Wait', 'But', ' But', 'Alternatively', ' Alternatively', 'Hmm', ' Hmm', 'Let', ' Let', 'Therefore', ' Therefore']
        split_tokens = self.tokenizer(split_words, add_special_tokens=False).input_ids 
        split_tokens = [x[0] for x in split_tokens]
        return split_tokens 

    def __call__(self, data: DataProto,  return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch['rm_scores']}
            else:
                return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        # chunk_aware_adv = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        ground_truth_list = []
        # problem_list = []

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            
            # entropy = data_item.batch.get("old_entropy", 0)
            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get('extra_info', None)
            
            # below to record the chunk aware advantage 
            ground_truth_list.append(ground_truth)


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
            
            reward_extra_info['original_score'].append(reward)
            
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
                # "chunk_aware_adv": chunk_aware_adv,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
    
    def compute_chunk_adv(self, data: DataProto):
        response_ids = data.batch['responses']
        response_mask = data.batch['response_mask']
        reward_tensor = data.batch['token_level_scores'] 
        
        is_correct = reward_tensor.sum(-1) > 0
        chunk_aware_adv = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        ground_truth_list = []
        # prompts = data.batch['prompts']
        problem_list = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            # below to record the chunk aware advantage 
            ground_truth_list.append(ground_truth)

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            if "qwen3" in self.tokenizer.name_or_path.lower():
                # print(prompt_str)
                problem = prompt_str.split("user\n")[1].split("assistant\n")[0].strip()
            elif "Phi" in self.tokenizer.name_or_path:
                # "<|system|>Your name is Phi, an AI math expert developed by Microsoft.<|end|><|user|>What's your name?<|end|><|assistant|>
                problem = prompt_str.split("<|user|>")[1].split("<|end|><|assistant|>")[0].strip()
            else:
                problem = prompt_str.split('<｜User｜>')[1].split('<｜Assistant｜>')[0].strip()
            problem_list.append(problem)
        first_correct_chunk_ratio, total_chunk_count = self.chunk_aware_grpo.batch_compute_chunk_adv(response_ids, response_mask,
                                                         chunk_aware_adv, ground_truth_list,
                                                         response_entropies=None,
                                                            problems=problem_list,
                                                            tokenizer=self.tokenizer,
                                                            is_correct=is_correct)
        correct_mask = is_correct.unsqueeze(-1).repeat(1, response_ids.shape[1])
        chunk_aware_adv = chunk_aware_adv * correct_mask.float()
        return {
            "chunk_aware_adv": chunk_aware_adv,
            "first_correct_chunk_ratio": first_correct_chunk_ratio.tolist(),
            "total_chunk_count": total_chunk_count.tolist()
        }
           
        

        
