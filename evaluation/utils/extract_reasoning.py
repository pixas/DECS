import json
import argparse
import os
import re
# import nltk
import torch
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

from evaluation.utils.prompt import get_prepare_func 
from evaluation.models import get_model

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load data from a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def save_jsonl(data: List[Dict[str, Any]], file_path: str) -> None:
    """Save data to a JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def extract_reasoning_steps(trajectory: str, model_type: str) -> List[str]:
    """Extract reasoning steps based on model type (sft or rl)."""
    if not trajectory:
        return []
    
    # Join all trajectory elements
    full_text = trajectory
    # model_type argument is deprecated now use "\n" and "\n\n" to split the text
    reasoning_chain = []
    for line in full_text.split('\n'):
        if line.strip():
            if len(line.strip().split("\n\n")) > 1:
                for sub_line in line.strip().split("\n\n"):
                    reasoning_chain.append(sub_line.strip())
            else:
                reasoning_chain.append(line.strip())
    return reasoning_chain
    # if model_type.lower() == "sft":
    #     # For SFT: split by sentences
    #     # try:
    #     #     nltk.data.find('tokenizers/punkt')
    #     # except LookupError:
    #     #     nltk.download('punkt', quiet=True)
    #     # return nltk.sent_tokenize(full_text)
    #     return [line.strip() for line in full_text.split('\n\n') if line.strip()]
    # elif model_type.lower() == "rl":
    #     # For RL: split by newlines
    #     return [line.strip() for line in full_text.split('\n') if line.strip()]
    # else:
    #     raise ValueError(f"Unknown model type: {model_type}. Expected 'sft' or 'rl'.")

def extract_final_answer(text: str) -> str:
    """Extract the final answer from the text."""
    # Look for patterns like "The answer is X" or "Therefore, X"
    patterns = [
        r"Therefore,\s+([A-E])\.",
        r"Therefore, the answer is ([A-E])\.",
        r"The answer is ([A-E])\.",
        r"Thus, the answer is ([A-E])\."
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    
    # If no pattern match, look for the last occurrence of A, B, C, D, or E
    matches = re.findall(r"([A-E])\.", text)
    if matches:
        return matches[-1]
    
    return ""

def process_data(model, prepare_func, input_file: str, output_prefix: str, model_type: str, resume: bool) -> None:
    """Process data and generate files with various reasoning step counts."""
    try:
        data = load_jsonl(input_file)
        print(f"Loaded {len(data)} items from {input_file}")
    except Exception as e:
        print(f"Error loading input file: {e}")
        return
    
    output_dir = os.path.dirname(output_prefix) or "."
    os.makedirs(output_dir, exist_ok=True)
    # base_name = os.path.basename(output_prefix)
    output_file = f"{output_prefix}.jsonl"
    results = {}
    if resume:
        # remove those items that have been processed in output_file
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    results[item['task']['id']] = item
            data = [item for item in data if item['task']['id'] not in results]
        
        write_file_handler = open(output_file, 'a', encoding='utf-8')
    else:
        write_file_handler = open(output_file, 'w', encoding
                                 ='utf-8')
    new_items = []
    for item_idx, item in enumerate(tqdm(data, total=len(data), desc="Processing items")):
        trajectory = item.get("trajectory", [])
        if not trajectory:
            continue
        trajectory = trajectory[0]
        trajectory = trajectory.split("Therefore, the answer is")[0]
        reasoning_steps = extract_reasoning_steps(trajectory, model_type)
        # remove duplicate reasonin steps
        # must be order invariant
        reasoning_steps = list(dict.fromkeys(reasoning_steps))
        # remove extremely long reasoning steps as it can be a noise
        reasoning_steps = [step for step in reasoning_steps if len(model.tokenizer.encode(step)) < 500]
        # Skip items with no reasoning steps
        if not reasoning_steps:
            print(f"Skipping item {item['task']['id']} with no reasoning steps")
            continue
            
        answer = item['task']['eval']['answer_idx']
        new_item = item.copy()
        new_item['trajectory_answer_prob'] = []
        
        # Process each step count individually to avoid memory issues
        # Create all truncated steps versions at once
        all_inputs = []
        for step_count in range(len(reasoning_steps) + 1):
            # Create a version with only the first 'step_count' steps
            truncated_steps = reasoning_steps[:step_count]
            
            # For step_count=0, just keep final answer; otherwise keep truncated reasoning + answer
            if model_type.lower() == "sft":
                truncated_trajectory = "\n\n".join(truncated_steps)
            else:
                truncated_trajectory = "\n".join(truncated_steps)
            truncated_trajectory += f" Therefore, the answer is {answer}"
            prompt = prepare_func(item['task'])["prompt"]
            
            if isinstance(prompt, str):
                input_text = prompt + truncated_trajectory
            elif isinstance(prompt, list):
                input_text = model.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True) + truncated_trajectory
            else:
                raise ValueError(f"Unknown prompt type: {type(prompt)}")
            all_inputs.append(input_text)
        
        # Process in micro-batches
        batch_size = 8  # Adjust based on your GPU memory
        new_item['trajectory_answer_prob'] = []
        
        for i in range(0, len(all_inputs), batch_size):
            batch_inputs = all_inputs[i:i+batch_size]
            
            # Tokenize the batch
            input_tensors = model.tokenizer(batch_inputs, return_tensors="pt", padding=True, padding_side='left')
            input_tensors = {k: v.to(model.device) for k, v in input_tensors.items()}
            
            with torch.no_grad():
                outputs = model.model(**input_tensors).logits
            
            response_length = 1
            # compute log_prob for the second last token 
            logits = outputs[:, -response_length - 1: -1, :] # [B, 1, V]
            labels = input_tensors['input_ids'][:, -response_length:] # [B, 1]
            prob = torch.nn.functional.softmax(logits, dim=-1).gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1) # [B, 1]
            prob = prob.sum(dim=-1) # [B]
            new_item['trajectory_answer_prob'].extend(prob.tolist())
            # Process each item in the batch
            
            # Free up memory
            del input_tensors, outputs
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        write_file_handler.write(json.dumps(new_item, ensure_ascii=False) + '\n')
        write_file_handler.flush()
        new_items.append(new_item)
        # Add to results for this step count

        # results[step_count].append(new_item)

    
    
    
def main():
    parser = argparse.ArgumentParser(description='Extract reasoning steps from model outputs.')
    parser.add_argument('--input_file', help='Path to the input JSONL file')
    parser.add_argument('--output_prefix', help='Prefix for output files')
    parser.add_argument('--model_type', choices=['sft', 'rl'], default='sft', 
                        help='Model type: sft (sentence-level) or rl (newline-level)')
    parser.add_argument('--prompt_type', choices=['base_default', 'base_think', 'base_zero', 'instruct_default', 'instruct_think', 'instruct_verl', 'instruct_answer'], required=True)
    parser.add_argument('--model_name_or_path', type=str, default=None)
    parser.add_argument('--resume', action='store_true', help='Resume processing from existing output file')
    args = parser.parse_args()
    
    if not args.output_prefix:
        base_name, ext = os.path.splitext(args.input_file)
        args.output_prefix = f"{base_name}_processed"
    if args.model_name_or_path is None:
        # read the result.json file to get the model_name_or_path
        dir_of_input_file = os.path.dirname(args.input_file)
        result_file = os.path.join(dir_of_input_file, "result.json")
        with open(result_file) as f:
            result = json.load(f)
        args.model_name_or_path =  result['args']['model_name_or_path']
    print(f"Loading {args.model_name_or_path}")
    model = get_model(args.model_name_or_path, use_vllm=False)
    prepare_func = get_prepare_func(args.model_name_or_path, args.prompt_type)
    process_data(model, prepare_func, args.input_file, args.output_prefix, args.model_type, args.resume)

if __name__ == "__main__":
    main()