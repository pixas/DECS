from tqdm import trange
from evaluation.models import get_model
import argparse
import json 
import re 
import os

prompt_template = """I will give you a list of medical knowledge items, your task is to combine two or three of them to construct logically meaningful problems.
Such problems just need a simple logic combination and do not require complex reasoning.
The problems are to test whether a well-trained medical large language models learn medical knowledge and its simple combination.

Knowledge items:
{knowledge}

Your output format should be a list, where each element is a dict object.
Each dict has three keys, where "problem" which is the newly constructed problem; "involved knowledge items" which is a list containing involving knowledge; "answer" which is the gold answer of your proposed problem."""

def postprocess(text):
    # extract the content of text
    # I only want the content wrapped by ``` and ```
    # the first ``` may follow by some words in the same line, remove them
    # for example ```json\n[the content]\n```, I only want [the content]
    # use re to extract 

    # Extract the content between triple backticks
    match = re.search(r'```(?:.*?\n)?([\s\S]*?)```', text)
    if match:
        try:
            clean_text = match.group(1).strip().replace('\\', '\\\\')
            if "=" in clean_text:
                clean_text = clean_text.split('=')[1].strip()
            return json.loads(clean_text)
        except Exception as e:
            print(f"Error loading json: {e}")
            print(match.group(1).strip())
            return match.group(1).strip()
    return text


def generate_2hop_knowledge(model, items):
    convs = [[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt_template.format(knowledge="\n".join(item['knowledge']))}
    ] for item in items]
    outputs = model(convs, temperature=1, max_new_tokens=4096)
    outputs = [postprocess(output[0]) for output in outputs]
    outputs = [x for x in outputs if x is not None]
    return outputs


def chunk_data(data, num_chunks, chunk_idx):
    chunk_size = len(data) // num_chunks
    start_idx = chunk_idx * chunk_size
    end_idx = (chunk_idx + 1) * chunk_size
    if chunk_idx == num_chunks - 1:
        end_idx = len(data)
    return data[start_idx:end_idx]

def main(model, data, save_fp):
    batch_size = 4
    for i in trange(0, len(data), batch_size, desc="Generating 2-hop knowledge"):
        batch = data[i:i+batch_size]
        outputs = generate_2hop_knowledge(model, batch)
        for i, output in enumerate(outputs):
            new_item = {
                "id": batch[i]['id'],
                "dataset": batch[i]['dataset'],
                "gen_output": output
            }
            save_fp.write(json.dumps(new_item) + '\n')
            save_fp.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default="qwen2.5-72b")
    parser.add_argument('--num_chunks', type=int, default=1)
    parser.add_argument('--chunk_idx', type=int, default=0)
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    
    args = parser.parse_args()
    
    model = get_model(args.model_name_or_path, use_vllm=True)
    # chunk input data
    data = [json.loads(line) for line in open(args.input_file)]
    chunked_data = chunk_data(data, args.num_chunks, args.chunk_idx)
    
    # resume from args.output_file
    if os.path.exists(args.output_file):
        saved_contents = [json.loads(line) for line in open(args.output_file)]
        saved_ids = {item['id'] for item in saved_contents}
        chunked_data = [item for item in chunked_data if item['id'] not in saved_ids]
        save_fp = open(args.output_file, 'a')
    else:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        save_fp = open(args.output_file, 'w')
        
    main(model, chunked_data, save_fp)