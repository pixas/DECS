import argparse
import os
import json
import torch.nn.functional as F
import tqdm
import numpy as np
from evaluation.models import get_model
import torch

def compute_single_knowledge(model, data):
    """compute loss for the model on the data. Each item of data contains a key 'knowledge' which is a list containing the involving necessary knowledge of the problem. Use batch to compute the loss to speed up. All the knowledge items of a single problem form a batch

    Args:
        model (Local_Model): the model to be evaluated. Use model.model() to call the model
        data (List[Dict]): the data items
    """
    new_data = []
    for item in tqdm.tqdm(data, total=len(data), desc="Computing knowledge"):
        knowledge = item['knowledge']
        if len(knowledge) == 0:
            continue
        inputs = model.tokenizer(knowledge, padding=True, truncation=True, return_tensors='pt', add_special_tokens=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        logits = model.model(**inputs).logits
        # compute the negative log likelihood loss
        labels = inputs['input_ids']


        # Shift the labels to compute loss (predict next token)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten the logits and labels
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)

        # Compute loss using CrossEntropyLoss, ignoring padding tokens
        loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=model.tokenizer.pad_token_id)

        # Store the loss in the item
        new_data.append({
            "id": item["id"],
            "dataset": item["dataset"],
            "knowledge": item['knowledge'],
            "knowledge_loss": loss.item()
        })
    return new_data

def compute_2hop_knowledge(model, data):
    new_data = []
    prompt = """A conversation between User and Assistant. The User asks a question and the Assistant answers with concise response.\nUser: {question}\nAssistant: {answer}"""
    assistant_tokens = model.tokenizer("Assistant:", add_special_tokens=False).input_ids
    for item in tqdm.tqdm(data, total=len(data), desc="Computing 2-hop knowledge"):
        twohop_knowledge = item['gen_output']
        if isinstance(twohop_knowledge, str):
            print(f"skip id={item['id']} for error parsing json")
            continue
        if isinstance(twohop_knowledge, dict):
            twohop_knowledge = [twohop_knowledge]
        qa_pairs = []
        for qa in twohop_knowledge:
            keys = list(qa.keys())
            if "involved_knowledge" in keys:
                keys.remove("involved_knowledge")
            if "answer" in keys:
                answer = qa["answer"]
                keys.remove("answer")
                problem = qa[keys[0]]
                qa_pairs.append(prompt.format(question=problem, answer=answer))
            else:
                continue

        inputs = model.tokenizer(qa_pairs, padding=True, return_tensors='pt', add_special_tokens=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        logits = model.model(**inputs).logits
        labels = inputs['input_ids']
        # we only want to compute the loss of the answer
        # mask labels to only compute the loss of the answer
        # Find the position of "Assistant: " in each sequence
        
        batch_size, seq_length = labels.shape
        mask = torch.ones_like(labels).to(labels.device)

        for i in range(batch_size):
            # Find the position where "Assistant:" starts
            # for j in range(seq_length - len(assistant_tokens) + 1):
            for j in np.where(labels[i].cpu().numpy() == assistant_tokens[0])[0]:
                if torch.all(labels[i, j:j+len(assistant_tokens)] == torch.tensor(assistant_tokens).to(labels.device)):
                    # Mask out everything before "Assistant:" + its length
                    mask[i, :j+len(assistant_tokens)] = 0
                    break

        # Apply mask to labels - replace masked tokens with pad token id
        labels = torch.where(mask == 1, labels, model.tokenizer.pad_token_id)
        
        # Shift the labels to compute loss (predict next token)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten the logits and labels
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)

        # Compute loss using CrossEntropyLoss, ignoring padding tokens
        loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=model.tokenizer.pad_token_id)
        # Store the loss in the item
        new_data.append({
            "id": item["id"],
            "knowledge_loss": loss.item()
        })
    return new_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', required=True, type=str)
    parser.add_argument('--data', required=True, type=str)
    parser.add_argument('--hop', default=1, type=int, choices=[1,2])
    parser.add_argument('--output_file', required=True, type=str)
    args = parser.parse_args()
    # obtain the name of args.model_name_or_path by fetching the element after the last /
    model_name = args.model_name_or_path.split('/')[-1]

        
        
    model = get_model(args.model_name_or_path, use_vllm=False)
    data = [json.loads(line) for line in open(args.data)]
    if args.hop == 1:
        new_data = compute_single_knowledge(model, data)
    elif args.hop == 2:
        new_data = compute_2hop_knowledge(model, data)
    else:
        raise ValueError("Invalid hop value")
    # print(data)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)
    # output the average loss of current model
    avg_loss = sum([item['knowledge_loss'] for item in new_data]) / len(new_data)
    print(f"Average loss of model {model_name}: {avg_loss}")
