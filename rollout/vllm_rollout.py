from contextlib import contextmanager
from pathlib import Path
from shutil import copytree
from tempfile import TemporaryDirectory
from time import sleep
from typing import Union
from transformers import AutoConfig
from vllm import LLM, SamplingParams 
import os 
from verl.utils.checkpoint.s3_client import client 
import yaml
import torch
import shutil


current_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(current_dir, 'config.yaml'), 'r') as f:
    # 将YAML内容转换为字典
    LOCAL_MODEL_PATHS = yaml.safe_load(f)


def pad_2d_list_to_length(response, pad_token_id, max_length=None):
    """
    pad a 2D list (e.g. responses, logprobs) to a 2D tensor.
    """
    response_length = max(len(sub_list) for sub_list in response)
    if max_length is not None and max_length > response_length:
        target_length = max_length
    else:
        target_length = response_length
    padded_response = [tuple(sub_list) + (pad_token_id,) * (target_length - len(sub_list)) for sub_list in response]
    tensor = torch.tensor(padded_response)
    return tensor


class VLLMRollout:
    def __init__(self, model_path: str, gpu_memory_utilization=0.9, hf_model_path=None, **kwargs):
        """_summary_

        Args:
            model_path (str): _description_
            gpu_memory_utilization (float, optional): _description_. Defaults to 0.9.
            kwargs contain the following arguments:
            max_seq_len_to_capture (int): max sequence length to capture, default is 16384
            max_model_len (int): max model length, default is 16384
            response_length (int): max response length, default is 8192
        """
        model_path = self.get_model_path(model_path)
        self.hf_model_path = hf_model_path 
        self.response_length = kwargs.get("response_length", 8192)
        
        self.llm = self.init_model(model=model_path, hf_model_path=hf_model_path, gpu_memory_utilization=gpu_memory_utilization, **kwargs)
        
        self.tokenizer = self.llm.get_tokenizer()
        self.pad_token_id = self.tokenizer.pad_token_id 
        
        sample_kwargs = dict(
            n=1,
            logprobs=0,
            max_tokens=self.response_length
        )
        for k in kwargs.keys():
            if hasattr(SamplingParams(), str(k)):
                sample_kwargs[k] = kwargs.get(k)
        self.sampling_params = SamplingParams(**sample_kwargs)
    
    def init_model(self, *args, **kwargs):

        model = kwargs.pop("model", None)
        gpu_memory_utilization = kwargs.pop("gpu_memory_utilization", 0.9)  
        hf_model_path = kwargs.pop("hf_model_path", None)
        
        # if os.path.exists(os.path.join(model, "tokenizer.json")):
        print(os.path.join(model, "tokenizer.json"))
        if client.exists(os.path.join(model, "tokenizer.json")):
            self.config = AutoConfig.from_pretrained(model)
            model_path = model
        else:
            # tmpdir = '/mnt/petrelfs/jiangshuyang/checkpoints/tmp_af9db1b123386e68'
            # continue
            from evaluation.models.base_model import convert_fsdp_checkpoints_to_hfmodels
            # # 创建一个随机字符串
            
            random_str = "tmp_" + os.urandom(8).hex()
            tmpdir = os.path.join(os.path.expanduser("~/checkpoints"), random_str)
            os.makedirs(tmpdir)
            # # tmpobj = TemporaryDirectory()
            # # tmpdir = tmpobj.name
            
            # # convert the fsdp checkpoints to hf models
            if hf_model_path is not None:
                hf_model_path = self.hf_model_path 
            else:
                hf_model_path = os.path.join(model, "huggingface")
            convert_fsdp_checkpoints_to_hfmodels(model, tmpdir, hf_model_path=hf_model_path)
            # copy all the files in the os.path.join(model, 'huggingface') to the tmp dir
            # for file in os.listdir(hf_model_path):
            # for file in client.listdir(hf_model_path):
            for file in ['config.json', 'tokenizer.json', 'generation_config.json', 'tokenizer_config.json']:
                src = os.path.join(hf_model_path, file)
                dst = os.path.join(tmpdir, file)
                # print(src, dst)
                # if os.path.isdir(src):
                # if client.is
                if os.path.isdir(src):
                    copytree(src, dst)
                else:
                    Path(dst).write_text(Path(src).read_text())
            # print all the files in the tmp dir
            print(f"Files in the tmp dir: {os.listdir(tmpdir)}")
            model_path = tmpdir
            self.config = AutoConfig.from_pretrained(model_path)
            sleep(2)
        max_model_len = min(self.config.max_position_embeddings, kwargs.pop("max_model_len", 16384))
        max_model_len = max(max_model_len, self.response_length+1024)
        max_seq_len_to_capture = min(max(kwargs.get("max_seq_len_to_capture", 16384), self.response_length+1024), max_model_len)
        tensor_parallel_size = kwargs.pop("tensor_parallel_size", 1)
        
        self.model = LLM(
            model=model_path,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
            max_seq_len_to_capture=max_seq_len_to_capture,
            max_num_batched_tokens=20000,
            enable_prefix_caching=True,
            max_model_len=max_model_len,
            enable_chunked_prefill=True,
            enable_sleep_mode=True,
            tensor_parallel_size=tensor_parallel_size
            
        )

        # check if model_path is a tmp dir
        # if so, delete the tmp dir
        if "tmp" in model_path:
            shutil.rmtree(tmpdir)
            # remove the tmpdir
            
        return self.model
    
    def get_model_path(self, model_name):
        return model_name if model_name not in LOCAL_MODEL_PATHS else LOCAL_MODEL_PATHS[model_name]
    
    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)
            
    def rollout_ids(self, input_ids, do_sample=True, n=1, temperature=0.6):
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }
        else:
            kwargs = {
                "n": n,
                "temperature": temperature,
                'top_p': 0.95
            }
        batch_size = len(input_ids)

        with self.update_sampling_params(**kwargs):
            outputs = self.llm.generate(
                prompt_token_ids=input_ids,
                sampling_params=self.sampling_params,
                use_tqdm=False
            )

            response = []
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    response.append(output.outputs[sample_id].token_ids)
            
            response = pad_2d_list_to_length(response, self.pad_token_id,
                                             max_length=self.response_length).to('cuda:0')

            texts = self.tokenizer.batch_decode(response, skip_special_tokens=True)
        return {
            "response_ids": response,
            "response_text": texts
        }
    
    def rollout(self, prompts: list[str], wrap_chat=True, do_sample=True, n=1, temperature=0.6, **gen_args):
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }
        else:
            kwargs = {
                "n": n,
                "temperature": temperature,
                "top_p": 0.95
            }
        kwargs.update(gen_args)
        if isinstance(prompts, str):
            prompts = [prompts]
        batch_size = len(prompts)
        if wrap_chat:
            message = [[{"role": "user", "content": pt}] for pt in prompts]
            input_prefix = self.tokenizer.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True
            )
        else:
            input_prefix = prompts
        with self.update_sampling_params(**kwargs):
            outputs = self.llm.generate(
                prompts=input_prefix,
                sampling_params=self.sampling_params,
                use_tqdm=False
            )

            response = []
            response_text = []
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    response.append(output.outputs[sample_id].token_ids)
                    response_text.append(output.outputs[sample_id].text)
            # response_ids = response
            
            # response = pad_2d_list_to_length(response, self.pad_token_id,
            #                                  max_length=self.response_length).to('cuda:0')

            # texts = self.tokenizer.batch_decode(response, skip_special_tokens=True)
        return {
            "response_ids": response,
            "response_text": response_text
        }
            
