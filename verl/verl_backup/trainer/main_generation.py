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
Generate responses given a dataset of prompts
"""
import ray
import numpy as np
import hydra
import os
from tempfile import TemporaryDirectory
from pathlib import Path
from shutil import copytree
from tqdm import trange
from evaluation.models.base_model import convert_fsdp_checkpoints_to_hfmodels
import yaml
from verl.utils.prompt import base_prompt_map

os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

from verl.utils.model import compute_position_id_with_mask

import pandas as pd

from transformers import AutoTokenizer

from verl import DataProto
from verl.utils.model import maybe_apply_chat_template
from verl.utils.fs import copy_to_local
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.utils.hdfs_io import makedirs
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup

with open("evaluation/models/config.yaml", 'r') as f:
    # 将YAML内容转换为字典
    LOCAL_MODEL_PATHS = yaml.safe_load(f)

@hydra.main(config_path='config', config_name='generation', version_base=None)
def main(config):
    run_generation(config)


def run_generation(config) -> None:

    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))


@ray.remote(num_cpus=1)
def main_task(config):
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)
    if config.data.resume and os.path.exists(config.data.output_path):
        print(f"Output file {config.data.output_path} already exists, skip generation.")
        return
    config.model.path = LOCAL_MODEL_PATHS[config.model.path] if config.model.path in LOCAL_MODEL_PATHS else config.model.path
    config.model.path = config.model.path[:-1] if config.model.path.endswith('/') else config.model.path
    local_path = copy_to_local(config.model.path)
    from verl.utils import hf_tokenizer

    trust_remote_code = config.data.get('trust_remote_code', False)
    if os.path.exists(os.path.join(local_path, 'tokenizer.json')):
        
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
    else:
        
        tokenizer = hf_tokenizer(os.path.join(local_path, 'huggingface'), trust_remote_code=trust_remote_code)
        tmpobj = TemporaryDirectory()
        tmpdir = tmpobj.name
        
        # convert the fsdp checkpoints to hf models
        hf_model_path = os.path.join(local_path, "huggingface")
        convert_fsdp_checkpoints_to_hfmodels(local_path, tmpdir, hf_model_path=hf_model_path)
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
        config.model.path = model_path

    if config.rollout.temperature == 0.:
        assert config.data.n_samples == 1, 'When temperature=0, n_samples must be 1.'

    # read dataset. Note that the dataset should directly contain chat template format (e.g., a list of dictionary)
    dataset = pd.read_parquet(config.data.path)
    chat_lst = dataset[config.data.prompt_key].tolist()
    data_sources = dataset['data_source'].tolist()

    chat_lst = [chat.tolist() for chat in chat_lst]

    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role='rollout')
    resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
    wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
    wg.init_model()

    total_samples = len(dataset)
    # real_batch_size = data.batch['input_ids'].shape[0]
    config_batch_size = config.data.batch_size
    dispatch_dp_size = wg.world_size
    num_batch = -(-total_samples // config_batch_size)
    output_lst = [[] for _ in range(config.data.n_samples)]

    for batch_idx in trange(num_batch):
        print(f'[{batch_idx+1}/{num_batch}] Start to process.')
        batch_chat_lst = chat_lst[batch_idx * config_batch_size:(batch_idx + 1) * config_batch_size]
        batch_data_sources = data_sources[batch_idx * config_batch_size:(batch_idx + 1) * config_batch_size]
        if config.data.is_base:
            batch_chat_lst = [base_prompt_map(data_src, config.data.prompt_type).format(prompt=chat) for data_src, chat in zip(batch_data_sources, batch_chat_lst)]

        inputs = maybe_apply_chat_template(batch_chat_lst, tokenizer,
                                               add_generation_prompt=True,
                                               padding=True,
                                               truncation=True,
                                               max_length=config.rollout.prompt_length,
                                               return_tensors='pt',
                                               return_dict=True,
                                               tokenize=True)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        position_ids = compute_position_id_with_mask(attention_mask)

        batch_dict = {'input_ids': input_ids, 'attention_mask': attention_mask, 'position_ids': position_ids}

        data = DataProto.from_dict(batch_dict)
        real_batch_size = data.batch['input_ids'].shape[0]
        if real_batch_size % dispatch_dp_size != 0:
            dummy_data_size = dispatch_dp_size - real_batch_size % dispatch_dp_size
            if dummy_data_size <= real_batch_size:
                dummy_data = data[:dummy_data_size]
            else:
                dummy_data = data.repeat(-(-dummy_data_size // real_batch_size))[:dummy_data_size]
            data = DataProto.concat([data, dummy_data])
            print(
                f'real_batch_size {real_batch_size} is not divisible by dispatch_dp_size {dispatch_dp_size}, add {dummy_data_size} dummy data'
            )

        batch_size = data.batch['input_ids'].shape[0]
        assert batch_size % dispatch_dp_size == 0, f'batch_size {batch_size} is not divisible by dispatch_dp_size {dispatch_dp_size}'

        print(f'[{batch_idx+1}/{num_batch}] Start to generate.')
        # START TO GENERATE FOR n_samples TIMES
        for i in range(config.data.n_samples):
            output = wg.generate_sequences(data)
            # remove dummy data
            output = output[:real_batch_size]
            output_text = tokenizer.batch_decode(output.batch['input_ids'][:, -config.rollout.response_length:],
                                                 skip_special_tokens=False)

            # remove the padding
            pad_token = tokenizer.pad_token
            output_text_unpad = []
            for text in output_text:
                output_text_unpad.append(text.replace(pad_token, ''))

            output_lst[i].extend(output_text_unpad)

    # convert output_lst from (n_samples, n_data) to (n_data, n_sampels)
    output_lst = np.array(output_lst, dtype=object)
    output_lst = np.transpose(output_lst, axes=(1, 0)).tolist()

    # add to the data frame
    dataset[f'responses'] = output_lst

    # write to a new parquet
    output_dir = os.path.dirname(os.path.abspath(config.data.output_path))
    makedirs(output_dir, exist_ok=True)
    
    if config.data.output_path.endswith('.parquet'):
        dataset.to_parquet(config.data.output_path)
    elif config.data.output_path.endswith('.json'):
        dataset.to_json(config.data.output_path, orient='records', lines=False)
    elif config.data.output_path.endswith('.jsonl'):
        dataset.to_json(config.data.output_path, orient='records', lines=True)
    # return output_text
    if "tmp" in config.model.path:
        tmpobj.cleanup()

if __name__ == '__main__':
    main()
