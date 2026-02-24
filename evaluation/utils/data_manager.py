import os
import re
import sys
import json
import math
from tqdm import tqdm
from evaluation.eval.score import score_task

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

class DataManager:
    def __init__(self, args):
        self.args = args
        for key, value in vars(args).items():
            setattr(self, key, value)
        
        # arguments
        self.cache_file = os.path.join(self.output_path, self.cache_file)
        self.result_file = os.path.join(self.output_path, self.result_file)
        
        assert self.chunk_idx < self.chunk_num

        self.cache = []
        self.dataset = self.data_path.rsplit("/", 1)[1].rsplit(".", 1)[0]
        self.examples, self.pids = self.load_data()
        
    
    def convert_parquet_to_dict(self, parquet_file):
        import pyarrow.parquet as pq
        import pandas as pd

        # Read the Parquet file
        table = pq.read_table(parquet_file)
        df = table.to_pandas()

        # Convert to python dict
        new_items = []
        for index, row in df.iterrows():
            new_item = {
                "id": f"{row['data_source']}_{index}",
                "dataset": row['data_source'],
                "input": row['prompt'][-1]['content'],
                "instruction": None,
                "eval": {
                    "answer": row["reward_model"]['ground_truth']
                }
            }
            new_items.append(new_item)
        print(f"Converting {parquet_file} to list[dict], total {len(new_items)} items")
        return new_items
        

    def load_data(self):
        # load test data
        if self.data_path.endswith("parquet"):
            examples = self.convert_parquet_to_dict(self.data_path)
            self.data_source = examples[0]['dataset']
        else:
            examples = json.load(open(self.data_path, "r"))
            self.data_source = examples[0]['dataset']
        examples = get_chunk(examples, self.chunk_num, self.chunk_idx)

        pids = [example["id"] for example in examples]

        if self.resume:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, "r") as f:
                    for line in tqdm(f.readlines(), ncols=60):
                        task_log = json.loads(line)

                        if task_log["task"]["id"] in pids:
                            self.score_task(task_log)
                            self.cache.append(task_log)
                            task_id_pos = pids.index(task_log["task"]["id"])
                            pids.pop(task_id_pos)
                        else:
                            continue
                self.save_cache()

        examples = {example["id"]: example for example in examples if example["id"] in pids}
        return examples, pids
    
    def update_cache(self, task_logs):
        for task_log in task_logs:
            self.score_task(task_log)
            self.cache.append(task_log)

            with open(self.cache_file, "a") as f:
                f.write(json.dumps(task_log, ensure_ascii=False, separators=(',', ': ')) + "\n")
        
        self.score()
    
    def save_cache(self):
        with open(self.cache_file, "w") as f:
            for task_log in self.cache:
                f.write(json.dumps(task_log, ensure_ascii=False, separators=(',', ': ')) + "\n")      
    
    def __getitem__(self, items):
        if items * self.batch >= len(self.pids):
            raise IndexError("Index out of range")
        
        example_ids = self.pids[items * self.batch : (items + 1) * self.batch]
        batch_examples = [self.examples[example_id] for example_id in example_ids]

        example_datas = [{
            "id": example["id"],
            "dataset": example["dataset"] if "dataset" in example else self.dataset,
            "instruction": example["instruction"] if example["instruction"] is not None else "",
            "input": example["input"] if example["input"] is not None else None,
            "eval": example["eval"],
        } for example in batch_examples]

        return example_datas
    
    def score_task(self, task_log):
        task_log["score"] = score_task(task_log)
        return task_log["score"]
    
    def score(self):
        count = len(self.cache)
        score = {matric:0 for matric in self.cache[0]["score"]}
        for task_log in self.cache:
            for metric in score:
                score[metric] += task_log["score"][metric]
        
        for metric in score:
            score[metric] /= count

        avg_time = sum([cache["time"] for cache in self.cache]) / count
        total_time = sum([cache["time"] for cache in self.cache])
        
        avg_tokens = sum([sum(cache["tokens"]) / len(cache['tokens']) for cache in self.cache]) / count if count > 0 else 0

        result = {'score': score, 'count': count, "time": {"avg": avg_time, "total": total_time}, "tokens": {"avg": avg_tokens}, 'args': vars(self.args)}
        # print("score: ", score)
        with open(self.result_file, 'w') as f:
            json.dump(result, f, indent=2, separators=(',', ': '))
    

    def __len__(self):
        if len(self.pids) % self.batch == 0:
            return len(self.pids) // self.batch
        else:
            return len(self.pids) // self.batch + 1


def tree2list(root):
    def dfs(node, current_path, all_paths):
        # 添加当前节点的值到路径
        current_path.append([node["state"], node["value"], node["rollout_value"]])
        # 如果是叶子节点，保存路径
        if not node["children"]:  # children 为空时
            all_paths.append(current_path[:])  # 深拷贝保存路径
        else:
            # 遍历子节点
            for child in node["children"]:
                dfs(child, current_path, all_paths)
        
        # 回溯：移除当前节点
        current_path.pop()

    # 初始化
    all_paths = []
    dfs(root, [], all_paths)
    return all_paths