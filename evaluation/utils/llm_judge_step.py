from copy import copy
from typing import Union

from tqdm import tqdm, trange
from evaluation.models import get_model 
import argparse 
import os 
import json 
from pydantic import BaseModel
from enum import Enum
from vllm.sampling_params import GuidedDecodingParams
class Score(BaseModel):
    step_scores: list[float]
    final_answer_match: bool
    explanation: str

json_schema = Score.model_json_schema()

judge_template = """任务：请根据以下内容判断模型推理步骤的正确性，给出评分（0到1之间）和解释。
你可以从几个方面考虑：
1. 单步的推理是否正确（是否使用了错误的知识），这是soundness评分
2. 单步的推理是否有助于回答问题，这是completeness评分
3. 推理的最终结果是否正确，这是correctness评分

问题：{question}
标准答案：{ground_truth}
模型推理步骤：
{steps}

输出格式：
```json
{{
    "soundness": ,
    "completeness": ,
    "correctness": ,
}}

```

"""
no_ref_prompt = """
作为严谨的评估专家，请分析模型的推理步骤是否符合以下要求：
1. **逻辑正确性**：每个步骤的推导是否符合逻辑规则
2. **事实正确性**：涉及的事实/数据/公式是否准确
3. **结论支持度**：所有步骤是否共同支持最终答案"{ground_truth}"
4. **必要性**：是否存在冗余或无关步骤

【评估规则】
- 对每个步骤独立打分（0/0.5/1）：
  1分：完全正确且必要
  0.5分：部分正确但存在次要问题
  0分：错误或无关步骤
- 最终答案错误时，最高总分为0.8（即使步骤看似合理）

【输入问题】
{question}

【模型推理步骤】
{model_steps}

请按JSON格式输出：
{{
  "step_scores": [按顺序的得分列表],
  "final_answer_match": true/false,
  "explanation": "..."
}}
禁止使用Markdown，直接输出纯文本符合格式的JSON。
"""

def parse_score(raw_scores):
    # parse to dict
    try:
        scores = json.loads(raw_scores)
    except json.decoder.JSONDecodeError as e:
        print(raw_scores)
        print(e)
        exit(-1)
    # step scores
    step_scores = scores['step_scores']
    final_answer_match = scores['final_answer_match']
    total_score = sum(step_scores) * 0.8 / len(step_scores) + 0.2 * int(final_answer_match)
    scores['total_score'] = total_score
    return scores

def test_one_item(model, data, args, test_id: Union[int, list] = 1):
    if isinstance(test_id, int):
        test_id = [test_id]

    # batch operation
    test_items = [data[idx] for idx in test_id]
    questions = [item['task']["input"] for item in test_items]
    ground_truths = [item['task']['eval']['answer_idx'] + ". " + item['task']['eval']['answer'] for item in test_items]
    trajectories = [item['trajectory'][0] for item in test_items]
    trajectories = [trajectory.split("Therefore, the answer is")[0] for trajectory in trajectories]
    reasoning_steps = [trajectory.split("\n") for trajectory in trajectories]
    reasoning_steps = [[x for x in steps if x] for steps in reasoning_steps]
    reasoning_steps = [list(dict.fromkeys(steps)) for steps in reasoning_steps]
    reasoning_steps = [[step for step in steps if len(model.tokenizer.encode(step)) < 500] for steps in reasoning_steps]
    # query = judge_template.format(question=question, ground_truth=ground_truth, steps=trajectory)
    queries = [no_ref_prompt.format(question=question, model_steps="\n".join(steps), ground_truth=ground_truth) for question, steps, ground_truth in zip(questions, reasoning_steps, ground_truths)]
    inputs_ = [[{"role": "user", "content": query}] for query in queries]
    outputs = model(inputs_, temperature=0, max_new_tokens=2048, guided_decoding=GuidedDecodingParams(
        json=json_schema,
        backend='lm-format-enforcer'
    ))
    scores = [parse_score(output[0]) for output in outputs]
    return scores


def process_all_items(model, data, args):
    results = {}
    if args.resume:
        if os.path.exists(args.output_path):
            with open(args.output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    results[item['task']['id']] = item
                print(f"Loaded {len(results)} results from {args.output_path}")
            data = [item for item in data if item['task']['id'] not in results]
        write_file_handler = open(args.output_path, 'a', encoding='utf-8')
    else:
        write_file_handler = open(args.output_path, 'w', encoding='utf-8')
    
    batch_size = args.batch_size
    # for item_idx, item in tqdm(enumerate(data), total=len(data), desc="Processing items"):
    #     new_item = copy(item)
    #     scores = test_one_item(model, data, args, test_id=item_idx)
    #     new_item['scores'] = scores
    #     write_file_handler.write(json.dumps(new_item) + '\n')
    # batch inference
    for i in trange(0, len(data), batch_size):
        batch_data = data[i:i+batch_size]
        scores = test_one_item(model, data, args, test_id=list(range(i, i+batch_size)))
        for item, score in zip(batch_data, scores):
            item['scores'] = score
            write_file_handler.write(json.dumps(item, ensure_ascii=False) + '\n')
            write_file_handler.flush()
    write_file_handler.close()        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default="sft_model")
    parser.add_argument("--data_path", type=str, default="data.jsonl")
    parser.add_argument("--output_path", type=str, default="output.jsonl")
    parser.add_argument("--resume", default=False, action="store_true")
    parser.add_argument('--batch_size', type=int, default=2)
    args = parser.parse_args()

    model = get_model(args.model_path, use_vllm=True)
    data = [json.loads(line) for line in open(args.data_path)]
    process_all_items(model, data, args)