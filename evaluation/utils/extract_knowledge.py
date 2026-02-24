import re
from tqdm import tqdm
from evaluation.models.base_model import APIModel
import json
import argparse 
import os
from concurrent.futures import ThreadPoolExecutor
import threading
import datetime 
from datetime import timedelta
import sys

def check_time_decorator(func):
    def wrapper(*args, **kwargs):
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        beijing_time = now_utc + timedelta(hours=8)
        deadline = beijing_time.replace(hour=8, minute=30, second=0, microsecond=0)
        if beijing_time >= deadline:
            print("已超过北京时间8:30，程序退出", flush=True)
            sys.exit(0)
        return func(*args, **kwargs)
    return wrapper


prompt = """I will give you a medical related problem and its gold answer as well as the process directing to the answer. Please try your best to extract the involving medical knowledge items.
A medical knowledge item is a simple fact sentence that does not involve complicated reasoning and needs as atomic as possible. 

Input Question: {input}

Output Response: {output}

Please list the medical knowledge problem one by one with a newline to separate them
"""

def extract_knowledge(args):
    with open(args.input_data, 'r') as f:
        data = json.load(f)
    
    # Check if output file exists to resume processing
    result_data = []
    if os.path.exists(args.output_data):
        result_data = [json.loads(line) for line in open(args.output_data, 'r')]
        print(f"Found existing output file with {len(result_data)} processed items. Resuming...")
        # Create a map of processed items by input
        processed_map = {item["input"]: item for item in result_data}
    else:
        processed_map = {}
    write_handler = open(args.output_data, 'a')
    # Process all items, skip those already processed
    result_data = []
    # Create a list for items that need processing
    items_to_process = []
    for item in data:
        if item["input"] in processed_map:
            # Use the already processed item
            result_data.append(processed_map[item["input"]])
        else:
            items_to_process.append(item)
    
    # Process items in parallel batches
    
    write_lock = threading.Lock()  # For thread-safe writing
    batch_size = 16  # Adjust based on resources and model capabilities
    
    def process_item(item):
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        beijing_time = now_utc + timedelta(hours=8)
        deadline = beijing_time.replace(hour=8, minute=30, second=0, microsecond=0)
        if beijing_time >= deadline:
            print("已超过北京时间8:30，程序退出", flush=True)
            sys.exit(0)
        inputs = item["input"]
        outputs = item["output"]
        instruction = prompt.format(input=inputs, output=outputs)
        results = model(instruction, sample_num=1, temperature=1.0)[0][0]
        
        # Process knowledge items
        all_knowledge = results.split("\n")
        for i, knowledge in enumerate(all_knowledge):
            if re.match(r"^\d+\.", knowledge):
                try:
                    all_knowledge[i] = re.findall(r"\d+\.(.*)", knowledge)[0].strip()
                except:
                    all_knowledge[i] = knowledge.strip()
            else:
                all_knowledge[i] = knowledge.strip()
        
        item["knowledge"] = all_knowledge
        return item
    if batch_size == 1:
        for item in tqdm(items_to_process, desc="Extracting Knowledge"):
            result_data.append(process_item(item))
            write_handler.write(json.dumps(item, ensure_ascii=False) + "\n")
            write_handler.flush()
    else:
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = []
            for item in items_to_process:
                futures.append(executor.submit(process_item, item))
            
            # Use tqdm to track progress
            for future in tqdm(futures, desc="Extracting Knowledge"):
                try:
                    item = future.result()
                    with write_lock:
                        write_handler.write(json.dumps(item, ensure_ascii=False) + "\n")
                        write_handler.flush()
                        result_data.append(item)
                except Exception as e:
                    print(f"Error processing item: {e}")
    
    return result_data
        
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  
    parser.add_argument('--input_data', type=str, required=True)
    parser.add_argument('--output_data', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='deepseek-chat')
    
    
    args = parser.parse_args()
    model = APIModel(args.model_name)
    knowledge_data = extract_knowledge(args)
    # with open(args.output_data, 'w') as f:
    #     json.dump(knowledge_data, f, indent=2, ensure_ascii=False)
        
    