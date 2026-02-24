import os
import re
import pdb
import openai
from openai import OpenAI
import time
import requests
import concurrent.futures
import yaml
import torch
import torch.nn.functional as F

from peft import PeftModel, LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList, LogitsProcessor, LogitsProcessorList
from verl.utils.checkpoint.s3_client import client

from transformers import AutoConfig, AutoModelForTokenClassification, AutoModelForCausalLM, AutoModelForVision2Seq
from tempfile import TemporaryDirectory
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple
from torch.distributed._tensor import DTensor, Shard, Placement
from pathlib import Path
from shutil import copytree


def logprobs_from_logits_v2(logits: torch.FloatTensor, labels):
    """
    A memory efficient implementation of logprobs_from_logits
    """
    if logits.dtype in [torch.float32, torch.float64]:
        logits_labels = torch.gather(logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        # loop to reduce peak mem consumption
        logsumexp_values = torch.stack([torch.logsumexp(l, dim=-1) for l in logits])
        logprobs_labels = logits_labels - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficent approach
        logprobs_labels = []
        for row_logits, row_labels in zip(logits, labels):  # loop to reduce peak mem consumption
            row_logprobs = F.log_softmax(row_logits, dim=-1)
            row_logprobs_labels = row_logprobs.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            logprobs_labels.append(row_logprobs_labels)
        logprobs_labels = torch.stack(logprobs_labels)
    return logprobs_labels

def manage_http_proxy(func):
    def wrapper(*args, **kwargs):
        original_proxy = os.environ.get('http_proxy', '')
        original_https_proxy = os.environ.get('https_proxy', '')
        
        # Set proxy to empty for the duration of the function call
        os.environ['http_proxy'] = ''
        os.environ['https_proxy'] = ''
        
        try:
            result = func(*args, **kwargs)
        finally:
            # Restore original proxy settings
            if original_proxy is not None:
                os.environ['http_proxy'] = original_proxy
            else:
                # If it wasn't set before, remove it
                if 'http_proxy' in os.environ:
                    del os.environ['http_proxy']
            
            if original_https_proxy is not None:
                os.environ['https_proxy'] = original_https_proxy
            else:
                if 'https_proxy' in os.environ:
                    del os.environ['https_proxy']
        return result
    return wrapper

current_dir = os.path.dirname(os.path.abspath(__file__))
def merge_by_placement(tensors: List[torch.Tensor], placement: Placement):
    if placement.is_replicate():
        return tensors[0]
    elif placement.is_partial():
        raise NotImplementedError("Partial placement is not supported yet")
    elif placement.is_shard():
        return torch.cat(tensors, dim=placement.dim).contiguous()
    else:
        raise ValueError(f"Unsupported placement: {placement}")

def convert_fsdp_checkpoints_to_hfmodels(local_dir, hf_path, hf_model_path):
    

    # copy rank zero to find the shape of (dp, fsdp)
    rank = 0
    world_size = 0
    # for filename in os.listdir(local_dir):
    for filename in client.listdir(local_dir):
        match = re.match(r"model_world_size_(\d+)_rank_0\.pt", filename)
        if match:
            world_size = match.group(1)  
            break  
    assert world_size, "No model file with the proper format"
    
    path = os.path.join(local_dir, f'model_world_size_{world_size}_rank_{rank}.pt')
    state_dict = client.load_checkpoint(path, map_location='cpu')
    # state_dict = torch.load(os.path.join(local_dir, f'model_world_size_{world_size}_rank_{rank}.pt'), map_location='cpu')
    pivot_key = sorted(list(state_dict.keys()))[0]
    weight = state_dict[pivot_key]
    assert isinstance(weight, torch.distributed._tensor.DTensor)
    # get sharding info
    device_mesh = weight.device_mesh
    mesh = device_mesh.mesh
    mesh_dim_names = device_mesh.mesh_dim_names

    print(f'Got device mesh {mesh}, mesh_dim_names {mesh_dim_names}')

    assert mesh_dim_names in (
        ('fsdp',),
    ), f'Unsupported mesh_dim_names {mesh_dim_names}'

    if 'tp' in mesh_dim_names:
        # fsdp * tp
        total_shards = mesh.shape[-1] * mesh.shape[-2]
        mesh_shape = (mesh.shape[-2], mesh.shape[-1])
    else:
        # fsdp
        total_shards = mesh.shape[-1]
        mesh_shape = (mesh.shape[-1],)

    print(f'Processing model shards with {total_shards} {mesh_shape} in total')

    model_state_dict_lst = []
    model_state_dict_lst.append(state_dict)
    model_state_dict_lst.extend([""] * (total_shards - 1))

    def process_one_shard(rank):
        model_path = os.path.join(local_dir, f'model_world_size_{world_size}_rank_{rank}.pt')
        state_dict = client.load_checkpoint(model_path, map_location='cpu', weights_only=False)
        # state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
        model_state_dict_lst[rank] = state_dict
        return state_dict

    with ThreadPoolExecutor(max_workers=min(32, os.cpu_count())) as executor:
        for rank in range(1, total_shards):
            executor.submit(process_one_shard, rank)
    state_dict = {}
    param_placements: Dict[str, List[Placement]] = {}
    keys = set(model_state_dict_lst[0].keys())
    for key in keys:
        state_dict[key] = []
        for model_state_dict in model_state_dict_lst:
            try:
                tensor = model_state_dict.pop(key)
            except:
                print("-"*30)
                print(model_state_dict)
            if isinstance(tensor, DTensor):
                state_dict[key].append(tensor._local_tensor.bfloat16())
                placements = tuple(tensor.placements)
                # replicated placement at dp dimension can be discarded
                if mesh_dim_names[0] == 'dp':
                    placements = placements[1:]
                if key not in param_placements:
                    param_placements[key] = placements
                else:
                    assert param_placements[key] == placements
            else:
                state_dict[key] = tensor.bfloat16()

    del model_state_dict_lst

    for key in sorted(state_dict):
        if not isinstance(state_dict[key], list):
            print(f"No need to merge key {key}")
            continue
        # merge shards
        placements: Tuple[Shard] = param_placements[key]
        if len(mesh_shape) == 1:
            # 1-D list, FSDP without TP
            assert len(placements) == 1
            shards = state_dict[key]
            state_dict[key] = merge_by_placement(shards, placements[0])
        else:
            # 2-D list, FSDP + TP
            raise NotImplementedError("FSDP + TP is not supported yet")

    print('Writing to local disk')

    config = AutoConfig.from_pretrained(hf_model_path)

    if 'ForTokenClassification' in config.architectures[0]:
        auto_model = AutoModelForTokenClassification
    elif 'ForCausalLM' in config.architectures[0]:
        auto_model = AutoModelForCausalLM
    elif 'ForConditionalGeneration' in config.architectures[0]:
        auto_model = AutoModelForVision2Seq
    else:
        raise NotImplementedError(f'Unknown architecture {config["architectures"]}')

    with torch.device('meta'):
        model = auto_model.from_config(config, torch_dtype=torch.bfloat16)
    model.to_empty(device='cpu')

    print(f'Saving model to {hf_path}')
    model.save_pretrained(hf_path, state_dict=state_dict)
    del state_dict
    del model

with open(os.path.join(current_dir, 'config.yaml'), 'r') as f:
    # 将YAML内容转换为字典
    LOCAL_MODEL_PATHS = yaml.safe_load(f)

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

class Base_Model:
    def __init__(self, model_name):
        self.model_path = self.get_model_path(model_name)
        if os.path.exists(os.path.join(self.model_path, 'tokenizer.json')):
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False, trust_remote_code=True)
        else:
            
            self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(self.model_path, 'huggingface'), use_fast=False, trust_remote_code=True)
        # self.processor = AutoProcessor.from_pretrained(self.model_path, use_fast=False, trust_remote_code=True)

    def __call__(self, inputs, sample_num=1, temperature=0.0, max_new_tokens=2048, logit_bias=False):
        raise NotImplementedError
    
    def prepare_inputs(self, inputs):
        if isinstance(inputs, str) or isinstance(inputs[0], str):
            return inputs
        
        elif isinstance(inputs, list) or isinstance(inputs[0], list):
            return self.tokenizer.apply_chat_template(inputs, add_generation_prompt=True, tokenize=False)
        
        else:
            raise NotImplementedError(f"""The inputs type {type(inputs)} not in (str, list[str], dict, list[dict])""")
    
    def get_logit_bias(self, state_num=5):
        state_list = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        logit_bias = {}
        # pdb.set_trace()
        for i in range(state_num):
            logit_bias[self.tokenizer(state_list[i], add_special_tokens=False)["input_ids"][0]] = 100

        return logit_bias
    
    def get_model_path(self, model_name):
        return model_name if model_name not in LOCAL_MODEL_PATHS else LOCAL_MODEL_PATHS[model_name]

class Local_Model(Base_Model):
    def __init__(self, model_name, peft_path=None):
        super().__init__(model_name)

        disable_torch_init()
        from accelerate import PartialState
        device_type = PartialState().default_device.type
        if device_type == 'cuda':
            self.device = "cuda:0"
        else:
            self.device = "npu:0"
        # self.device = "npu:0" if torch.npu.is_available() else "cuda:0"
        # if "gemma-3" in self.model_path:
        #     from transformers import Gemma3ForConditionalGeneration
        #     model_cls = Gemma3ForConditionalGeneration
        # else:
        #     model_cls = AutoModelForCausalLM
        # self.model = AutoModelForCausalLM.from_pretrained(self.model_path,
        #     torch_dtype=torch.float16, trust_remote_code=True).to(self.device)
        self.init_model()
        self.tokenizer.padding_side = "left" 
        
        if peft_path is not None:
            lora_config = LoraConfig.from_pretrained(peft_path)
            self.model = PeftModel.from_pretrained(self.model, peft_path, config=lora_config)
            print(f"Merging weights")
            self.model = self.model.merge_and_unload()
            print('Convert to FP16...')
            self.model.to(torch.float16)
        self.model.eval()
    
    def init_model(self):
        model = self.model_path
        if os.path.exists(os.path.join(model, "tokenizer.json")):
            model_path = model 
        else:
            tmpobj = TemporaryDirectory()
            tmpdir = tmpobj.name
            
            # convert the fsdp checkpoints to hf models
            hf_model_path = os.path.join(model, "huggingface")
            convert_fsdp_checkpoints_to_hfmodels(model, tmpdir, hf_model_path=hf_model_path)
            # copy all the files in the os.path.join(model, 'huggingface') to the tmp dir
            for file in os.listdir(hf_model_path):
                src = os.path.join(hf_model_path, file)
                dst = os.path.join(tmpdir, file)
                if os.path.isdir(src):
                    copytree(src, dst)
                else:
                    Path(dst).write_text(Path(src).read_text())
            # print all the files in the tmp dir
            print(f"Files in the tmp dir: {os.listdir(tmpdir)}")
            model_path = tmpdir
        self.model = AutoModelForCausalLM.from_pretrained(model_path,
            torch_dtype=torch.bfloat16, trust_remote_code=True).to(self.device)
        
        
        # check if model_path is a tmp dir
        # if so, delete the tmp dir
        if "tmp" in model_path:
            tmpobj.cleanup()
    
    def __call__(self, inputs, sample_num=1, temperature=0.0, max_new_tokens=2048, logit_bias=False, **kwargs):
        inputs = self.prepare_inputs(inputs)
        inputs = self.tokenizer(inputs, padding=True, truncation=True, padding_side='left', return_tensors="pt")
        
        if logit_bias:
            logits_processor_list = LogitsProcessorList([
                LogitBiasLogitsProcessor(self.get_logit_bias(logit_bias)),
            ])

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=inputs.input_ids.to(self.device),
                attention_mask=inputs.attention_mask.to(self.device),
                num_return_sequences=sample_num,
                do_sample=False if temperature == 0.0 else True,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                tokenizer=self.tokenizer,
                logits_processor=logits_processor_list if logit_bias else None,
                **kwargs
            )
            
        final_outputs = output_ids[:, len(inputs["input_ids"][0]):]
        outputs = self.tokenizer.batch_decode(final_outputs, skip_special_tokens=True)
        outputs = [outputs[batch_id*sample_num:(batch_id+1)*sample_num] for batch_id in range(len(inputs.input_ids))]

        return outputs

    def compute_logp(self, inputs, ground_truths, temperature=0.6):
        """
        Compute the log probability of the labels given the input_ids.
        """
        if isinstance(inputs, str):
            inputs = [inputs]
        # inputs are the text, transform to input_ids, labels and attention_mask
        # input_ids = self.tokenizer(inputs, add_special_tokens=False)['input_ids']  # [list of list[int]]
        prompt_results = self.tokenizer(inputs, add_special_tokens=False, return_tensors='pt', padding=True, padding_side='left')
        prompt_ids, prompt_attention_mask = prompt_results['input_ids'], prompt_results['attention_mask']
        
        response_results = self.tokenizer(ground_truths, add_special_tokens=False, return_tensors='pt', padding=True, padding_side='right')
        response_ids, response_attention_mask = response_results['input_ids'], response_results['attention_mask']
        response_length = response_results['input_ids'].shape[1]
        
        prompt_ids = prompt_ids.to(self.device)
        prompt_attention_mask = prompt_attention_mask.to(self.device)
        response_ids = response_ids.to(self.device)
        response_attention_mask = response_attention_mask.to(self.device)
        
        
        final_input_ids = torch.cat([prompt_ids, response_ids], dim=-1).to(self.device)  # [B, T]
        attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=-1).to(self.device)  # [B, T]
        with torch.no_grad():
            outputs = self.model(final_input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            logits.div_(temperature)
            logits = logits[:, -response_length - 1:-1, :]  # (bsz, response_length, vocab_size)
            log_probs = logprobs_from_logits_v2(logits, response_ids) # [B, T]
            
            # based on the response_attention_mask, compute the log_probs
            log_probs = log_probs * response_attention_mask  # [B, T]
            true_response_length = response_attention_mask.sum(dim=-1)  # [B]
            log_probs = log_probs.sum(dim=-1)  # [B]
            log_probs = log_probs / true_response_length  # [B]
            
            

        return log_probs

class LogitBiasLogitsProcessor(LogitsProcessor):
    def __init__(self, logit_bias):
        self.logit_bias = logit_bias

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):

        for index in self.logit_bias.keys():
            scores[:, index] += self.logit_bias[index]
        return scores


class APIModel(Base_Model):
    def __init__(self, model_name, url=None, peft_path=None):
        self.model_name = model_name
        if "deepseek" in model_name:
            base_url = "https://api.deepseek.com"
            if "proxy_url" in os.environ:
                os.environ['http_proxy'] = os.environ['https_proxy'] = os.environ['proxy_url']
            else:
                print("You are using DeepSeek API, please ensure that you have prepared necessary network variables.")
            api_key = os.environ.get("DEEPSEEK_API_KEY", None)
        elif url is not None:
            # vllm server
            base_url = f"http://{url}/v1"
            api_key = None
            os.environ['http_proxy'] = os.environ['https_proxy'] = ''
        else:
            base_url = "https://api.openai.com/v1"
            if "GPT_PROXY" in os.environ:
                os.environ['http_proxy'] = os.environ['https_proxy'] = os.environ['GPT_PROXY']
            else:
                print("You are using OpenAI API, please ensure that you have prepared necessary network variables.")
            # os.environ['http_proxy'] = os.environ['https_proxy'] = 
            api_key = os.environ.get("OPENAI_API_KEY", None)
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        
        assert peft_path is None, "API model does not support PEFT model"

   

    @manage_http_proxy
    def __call__(self, inputs, sample_num=1, temperature=0.0, max_new_tokens=2048, wrap_chat=True, logit_bias=False):
        if isinstance(inputs, str):
            inputs = [inputs]
        if len(inputs) == 1 and isinstance(inputs[0], list):
            # 
            return [self.call_chatgpt(inputs[0], temperature=temperature, top_p=0.95, max_tokens=max_new_tokens, n=sample_num, wrap_chat=wrap_chat)]
        elif len(inputs) == 1 and isinstance(inputs[0], dict):
            return [self.call_chatgpt(inputs, temperature=temperature, top_p=0.95, max_tokens=max_new_tokens, n=sample_num, wrap_chat=wrap_chat)]
        else:
            results = [None] * len(inputs)
            def task(instruction):
                return self.call_chatgpt(instruction, temperature=temperature, top_p=0.95, max_tokens=max_new_tokens, n=sample_num, wrap_chat=wrap_chat)
            # 使用ThreadPoolExecutor来处理多线程
            # Get number of CPU cores for maximum thread workers
            # max_workers = min(32, os.cpu_count() or 1)  # Limit to 32 threads max
            max_workers = os.cpu_count()
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有任务，并保留每个future的索引
                futures = {executor.submit(task, instruction): i for i, instruction in enumerate(inputs)}
                
                # 获取所有任务的结果，并根据索引放入results中
                for future in concurrent.futures.as_completed(futures):
                    try:
                        # 按照原始索引顺序存储结果
                        index = futures[future]
                        results[index] = future.result()
                    except Exception as e:
                        print(f"Error processing instruction: {e}")
            
            return results
        
    
    
    def call_chatgpt(self, prompt, **kwargs):
        success = False
        re_try_count = 15
        ans = ''
        while not success and re_try_count >= 0:
            re_try_count -= 1
            try:
                ans = None
                # while ans is None:
                ans = self.get_oai_completion(prompt, **kwargs)
                # time.sleep(1)
                success = True if ans is not None else False
            except Exception as e:
                print(f"Error processing instruction: {e}")
                time.sleep(5)
                print('retry for sample:', prompt)
        return ans if ans is not None else ''
    
    def get_oai_completion(self, prompt, **kwargs):
        temperature = kwargs.get("temperature", 1)
        top_p = kwargs.get("top_p", 0.95)
        frequency_penalty = kwargs.get("frequency_penalty", 0)
        presence_penalty = kwargs.get("presence_penalty", 0)
        stop = kwargs.get("stop", None)
        wrap_chat = kwargs.pop("wrap_chat", True)
        n = kwargs.get("n", 1)
        gen_kwargs = {
            "n": n,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "stop": stop,
            "temperature": temperature
        }
        if self.model_name == 'gpt-3.5-turbo':
            gen_kwargs['max_tokens'] = kwargs.get("max_tokens", 4096)
        elif self.model_name == 'o3-mini' or self.model_name == 'deepseek-reasoner':
            gen_kwargs['max_completion_tokens'] = kwargs.get("max_tokens", 16384)
            gen_kwargs.pop("top_p")
        else:
            gen_kwargs['max_tokens'] = kwargs.get("max_tokens", 8192)
        
        try: 
            if "deepseek" in self.model_name:
                # deepseek model not support n arg
                # gen_kwargs['n'] = 1
                res = []
                # for i in range(n):
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        # {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    **gen_kwargs
                )
                for i in range(gen_kwargs['n']):
                    res.append(completion.choices[i].message.content)
                
            else:
                # openai supports n 
                if wrap_chat:
                    completion = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                                # {"role": "system", "content": "You are a helpful assistant."},
                                {"role": "user", "content": prompt},
                            
                            ],
                        **gen_kwargs
                    )
                    
                    res = [x.message.content for x in completion.choices]
                else:
                    completion = self.client.completions.create(
                        model=self.model_name,
                        prompt=prompt,
                        **gen_kwargs
                    )
                    
                    res = [x.text for x in completion.choices]
                
                # res = [x.message.content for x in completion.choices]
        
            gpt_output = res
            return gpt_output
        except requests.exceptions.Timeout:
            # Handle the timeout error here
            print("The OpenAI API request timed out. Please try again later.")
            
            return None
        except openai.BadRequestError as e:
            # Handle the invalid request error here
            print(f"The OpenAI API request was invalid: {e}")
            return None
        except openai.APIError as e:
            if "The operation was timeout" in str(e) or "Request timed out" in str(e) or "Connection error" in str(e):
                # Handle the timeout error here
                print(f"Error: {e}. The OpenAI API request timed out. Please try again later.")
                time.sleep(1)
                return None        
            else:
                # Handle other API errors here
                print(f"The OpenAI API returned an error: {e}")
                return None
        except openai.RateLimitError as e:
            return self.get_oai_completion(prompt, **kwargs)


if __name__ == "__main__":
    model = APIModel("/mnt/petrelfs/jiangshuyang/checkpoints/DeepSeek-R1-Distill-Qwen-1.5B", url="10.140.54.16:10041")
    outputs = model([{"role": "user", "content": "what is your name"}], temperature=0.6, wrap_chat=True, max_new_tokens=10)
    print(outputs)