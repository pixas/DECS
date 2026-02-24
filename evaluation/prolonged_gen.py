import argparse 
from copy import deepcopy
import os 
import json 
import sys 
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
# Ensure the parent directory is also in sys.path if evaluation is a subdirectory
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from evaluation.models import get_model
from evaluation.eval.score import score_task
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--hf_model_path", type=str, default=None, help="if not None, will convert the fsdp checkpoints to hf models")
    
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--sample_num", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument(
        "--cache_file", default="cache.jsonl", help="name of the cache file"
    )
    parser.add_argument(
        "--result_file", default="result.json", help="name of the results file"
    )
    args = parser.parse_args()
    if not os.path.exists(os.path.join(args.output_path)):
        os.makedirs(os.path.join(args.output_path), exist_ok=True)

    print("====Input Arguments====")
    print(json.dumps(vars(args), indent=4, sort_keys=False))
    return args





if __name__ == "__main__":
    args = parse_args()
    # original_data = 
    original_cache_path = os.path.join(args.data_path, args.cache_file)
    original_cache_data = [json.loads(line) for line in open(original_cache_path)]
    # original_result_data = 
    Model = get_model(args.model_name_or_path, use_vllm=True, max_new_tokens=args.max_new_tokens, temperature=args.temperature, hf_model_path=args.hf_model_path)
    tokenizer = Model.tokenizer 
    
    
    should_continue = []
    should_continue_ori_response = []
    should_continue_eval_info = []
    should_continue_location = []
    for i in range(len(original_cache_data)):
        item = original_cache_data[i]
        eval_info = item['task']['eval']
        problem = item['task']['input']
        prompt_input = tokenizer.apply_chat_template(
            [{"role": "user", "content": problem}],
            add_generation_prompt=True, tokenize=False
        )
        for j, trajectory in enumerate(item['trajectory']):
            cur_gen_token_count = item['tokens'][j]
            if cur_gen_token_count > 15000:
                should_continue.append(prompt_input + trajectory)
                should_continue_ori_response.append(trajectory)
                should_continue_eval_info.append(item)
                should_continue_location.append((i, j))
    
    print(f"Collecting {len(should_continue)} trajectories to prolong")
    
    remaining_trajectories = Model.rollout(
        should_continue,
        do_sample=True,
        wrap_chat=False,
        n=1
    )['response_text']
    
    complete_should_continue_response = [x + y for x, y in zip(should_continue_ori_response, remaining_trajectories)]
    new_data = deepcopy(original_cache_data)
    for idx, (i, j) in enumerate(should_continue_location):
        new_data[i]['trajectory'][j] = complete_should_continue_response[idx]
        new_data[i]['tokens'][j] = len(tokenizer(complete_should_continue_response[idx], add_special_tokens=False)['input_ids'])
    
    score_info = []
    
    for task in new_data:
        task_score = score_task(task)
        task['score'] = task_score
        # score_info.append(task_score)
    # score_task()
    count = len(new_data)
    score = {matric:0 for matric in new_data[0]["score"]}
    for task_log in new_data:
        for metric in score:
            score[metric] += task_log["score"][metric]
    
    for metric in score:
        score[metric] /= count

    avg_time = sum([cache["time"] for cache in new_data]) / count
    total_time = sum([cache["time"] for cache in new_data])
    
    avg_tokens = sum([sum(cache["tokens"]) / len(cache['tokens']) for cache in new_data]) / count if count > 0 else 0

    result = {'score': score, 'count': count, "time": {"avg": avg_time, "total": total_time}, "tokens": {"avg": avg_tokens}, 'args': vars(args)}
    # print("score: ", score)
    with open(os.path.join(args.output_path, args.result_file), 'w') as f:
        json.dump(result, f, indent=2, separators=(',', ': '))
    
    with open(os.path.join(args.output_path, args.cache_file), "w") as f:
        for task_log in new_data:
            f.write(json.dumps(task_log, ensure_ascii=False, separators=(',', ': ')) + "\n")


    
    
