import os
import torch
import re
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple
from torch.distributed._tensor import DTensor, Shard, Placement
from tempfile import TemporaryDirectory
from pathlib import Path
from shutil import copytree
from verl.utils.checkpoint.s3_client import client  

from evaluation.models.base_model import Base_Model, convert_fsdp_checkpoints_to_hfmodels
from transformers import AutoConfig, AutoModelForTokenClassification, AutoModelForCausalLM, AutoModelForVision2Seq

# os.environ["VLLM_INSTALL_PUNICA_KERNELS"] = 1



class VLLM_Model(Base_Model):
    def __init__(self, model_name, hf_model_path=None, peft_path=None):
        super().__init__(model_name)
        config_path = self.model_path if os.path.exists(os.path.join(self.model_path, 'config.json')) else os.path.join(self.model_path, "huggingface")
        model_config = AutoConfig.from_pretrained(config_path, use_fast=False, trust_remote_code=True)
        self.peft_path = peft_path
        max_len = getattr(model_config, "max_position_embeddings", 16384)
        self.hf_model_path = hf_model_path
        if peft_path is None:
            self.init_model(model=self.model_path, trust_remote_code=True, max_model_len=min(32000, max_len), max_seq_len_to_capture=32000, gpu_memory_utilization=0.9, enable_prefix_caching=True,) # gpu_memory_utilization=0.8
            # self.model = LLM(model=self.model_path, trust_remote_code=True, max_model_len=min(32000, model_config.max_position_embeddings), max_seq_len_to_capture=32000, gpu_memory_utilization=0.9) # gpu_memory_utilization=0.8
        else:
            self.init_model(model=self.model_path, trust_remote_code=True, max_model_len=min(32000, max_len), max_seq_len_to_capture=32000, gpu_memory_utilization=0.9, enable_lora=True, enable_prefix_caching=True) # gpu_memory_utilization=0.8
            # self.model = LLM(model=self.model_path, trust_remote_code=True, max_model_len=min(32000, model_config.max_position_embeddings), max_seq_len_to_capture=32000, gpu_memory_utilization=0.9, enable_lora=True) # gpu_memory_utilization=0.8
    
    def init_model(self, *args, **kwargs):
        max_model_len = kwargs.pop("max_model_len", 32000)
        model = kwargs.pop("model", None)
        # if os.path.exists(os.path.join(model, "tokenizer.json")):
        if client.exists(model, "tokenizer.json"):
            model_path = model
        else:
            tmpobj = TemporaryDirectory()
            tmpdir = tmpobj.name
            
            # convert the fsdp checkpoints to hf models
            if self.hf_model_path is not None:
                hf_model_path = self.hf_model_path 
            else:
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
        
        self.model = LLM(*args, model=model_path, max_model_len=max_model_len,
            skip_tokenizer_init=False,
            enable_chunked_prefill=True,
            enable_sleep_mode=True,
            **kwargs)
        # check if model_path is a tmp dir
        # if so, delete the tmp dir
        if "tmp" in model_path:
            tmpobj.cleanup()
    
    
    
    def model_forward(self, inputs, sampling_params):
        if self.peft_path:
            outputs = self.model.generate(prompts=inputs, sampling_params=sampling_params, lora_request=LoRARequest("lora", 1, self.peft_path), use_tqdm=False)
        else:
            outputs = self.model.generate(prompts=inputs, sampling_params=sampling_params, use_tqdm=False)
        return outputs

    def __call__(self, inputs_, sample_num=1, temperature=0.0, max_new_tokens=2048, logit_bias=False, return_logprobs=False, **kwargs):
        if isinstance(inputs_, list) and isinstance(inputs_[0], dict):
            # a conversation style single data, we batch it
            inputs_ = [inputs_]
        inputs = self.prepare_inputs(inputs_)
        sampling_params = SamplingParams(n=sample_num, temperature=temperature, max_tokens=max_new_tokens, logit_bias=None if not logit_bias else self.get_logit_bias(logit_bias), **kwargs)
        
        outputs = self.model_forward(inputs, sampling_params)
        
        text_outputs = [[output.text.strip() for output in outputs[i].outputs] for i in range(len(inputs_))]
        if return_logprobs:
            logits = [[output.logprobs for output in outputs[i].outputs] for i in range(len(inputs_))]
            return text_outputs, logits
        return text_outputs
