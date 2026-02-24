from transformers import AutoTokenizer, AutoModelForCausalLM 
import torch 
from itertools import chain

from verl.protocol import DataProto
from verl.utils.model import compute_position_id_with_mask
from verl.utils.prompt import BASE_INFER_PROMPT
from verl.utils.torch_functional import logprobs_from_logits
import verl.utils.torch_functional as verl_F

torch.set_default_device("cuda")

model_path = "/mnt/hwfile/medai/LLMModels/Model/Qwen2.5-3B"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

def compute_ref_log_prob(model, batch):
    input_ids = batch.batch['input_ids']
    attention_mask = batch.batch['attention_mask']
    position_ids = batch.batch['position_ids']
    response_length = batch.batch['responses'].size(-1)
    temperature = 1.0
    output = model(input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False)  # prevent model thinks we are generating
    logits = output.logits
    logits.div_(temperature)
    logits = logits[:, -response_length - 1:-1, :]  # (bsz, response_length, vocab_size)
    log_probs = logprobs_from_logits(logits, batch.batch['responses'])
    # entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
    output = DataProto.from_dict(tensors={'ref_log_prob': log_probs},
                                         meta_info={'temperature': 1})
    return output

query = """A junior orthopaedic surgery resident is completing a carpal tunnel repair with the department chairman as the attending physician. During the case, the resident inadvertently cuts a flexor tendon. The tendon is repaired without complication. The attending tells the resident that the patient will do fine, and there is no need to report this minor complication that will not harm the patient, as he does not want to make the patient worry unnecessarily. He tells the resident to leave this complication out of the operative report. Which of the following is the correct next action for the resident to take?\nA. Disclose the error to the patient but leave it out of the operative report\nB. Disclose the error to the patient and put it in the operative report\nC. Tell the attending that he cannot fail to disclose this mistake\nD. Report the physician to the ethics committee\nE. Refuse to dictate the operative report"""
response = "1. The resident is a junior orthopaedic surgery resident, and the attending physician is the department chairman.\n2. The resident inadvertently cuts a flexor tendon during the carpal tunnel repair.\n3. The tendon is repaired without complication.\n4. The attending tells the resident that there is no need to report this minor complication as it will not harm the patient.\n5. The attending does not want to make the patient worry unnecessarily.\n6. The resident is asked to leave this complication out of the operative report.\n7. The resident must decide on the appropriate next action to take.\n8. Option B, \"Disclose the error to the patient and put it in the operative report,\" is the correct next action for the resident to take.\n9. The resident should disclose the error to the patient and include it in the operative report to ensure transparency and accountability.\n\nThe answer is B."
prompt = BASE_INFER_PROMPT.format(prompt = query)
prompt_ids = tokenizer(prompt, return_tensors='pt')['input_ids']
total_sequence = prompt + response 
total_info = tokenizer(total_sequence, return_tensors='pt')
response_ids = tokenizer(response, return_tensors='pt')['input_ids']
total_input_ids, attention_mask = total_info['input_ids'], total_info['attention_mask']

batch = DataProto.from_dict(
    tensors={
        "prompts": torch.cat([prompt_ids] * 2),
        "responses": torch.cat([response_ids] * 2),
        "attention_mask": torch.cat([attention_mask] * 2),
    },
    
)

reward_tensor = torch.zeros_like(batch.batch['responses'])
# Check if batch is empty

for i in range(len(batch)):
    data_item = batch[i]
    # response_str = 
    prompt_ids = data_item.batch['prompts']

    prompt_length = prompt_ids.shape[-1]

    valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
    valid_prompt_ids = prompt_ids[-valid_prompt_length:]

    response_ids = data_item.batch['responses']
    valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
    valid_response_ids = response_ids[:valid_response_length]
    
    
    prompt_str = tokenizer.decode(valid_prompt_ids)
    response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
    
    
    ground_truth = "C. Tell the attending that he cannot fail to disclose this mistake"
    # 对response_str按\n切分，去重（保留相对位置），并且去除self.tokenizer.encode()超过500的sub response
    sub_responses = response_str.split("\n")
    sub_responses = [x.split("\n\n") for x in sub_responses]
    sub_responses = list(chain(*sub_responses))
    sub_responses = [x for x in sub_responses if len(x) > 0]
    # 去重，保留相对位置
    sub_responses = list(dict.fromkeys(sub_responses))
    if len(sub_responses) == 1:
        # only one step, skip 
        continue
    
    sub_responses = [sub_response for sub_response in sub_responses if len(tokenizer.encode(sub_response)) <= 500]
    if len(sub_responses) == 1:
        continue
    # 拼接prompt，sub_response以及"\nThe answer is {ground_truth}"
    sub_responses = [f"{prompt_str}{sub_response}\nThe answer is {ground_truth}" for sub_response in sub_responses]

    chunk_size = 4
        
    # chunk_size = if self.device_mesh is not None:
    # sp_size = self.device_mesh['sp'].size()
    if chunk_size is not None and len(sub_responses) % chunk_size != 0:
        # pad to be divisible by dp_size
        pad_size = chunk_size - len(sub_responses) % chunk_size
        sub_responses += [sub_responses[-1]] * pad_size
    else:
        pad_size = 0
    sub_response_result = tokenizer(sub_responses, return_tensors='pt', padding=True, truncation=True, max_length=3092)
    sub_response_input_ids = sub_response_result['input_ids']
    sub_response_attention_mask = sub_response_result['attention_mask']
    # 获得\nThe answer is后的input_ids，作为新的batch中的response内容，去计算log_prob
    new_response_ids = tokenizer(f" {ground_truth}", return_tensors='pt')['input_ids']
    # new_item = copy.deepcopy(data_item)
    # new_item['responses'] = new_response_ids
    # pprint(batch.meta_info)
    new_batch = DataProto.from_dict(
        tensors={
            "responses": torch.cat([new_response_ids] * len(sub_response_input_ids)),
            "attention_mask": sub_response_attention_mask,
            "input_ids": sub_response_input_ids,
            "position_ids": compute_position_id_with_mask(sub_response_attention_mask),
        },
        non_tensors={},
        meta_info={"temperature": 1}
    )
    # pprint(new_batch)
    try:
        answer_log_prob = compute_ref_log_prob(model, new_batch)
    except Exception as e:
        print(response_str)
        print("-" * 80)
        print(new_batch)

        print(e)
        exit(-1)

    tensor_answer_log_prob = answer_log_prob.batch['ref_log_prob']

    # Check if there are enough responses to compute differences
    if len(tensor_answer_log_prob) > 1:
        if valid_response_length > 0:  # Ensure valid index
            batch_count = len(sub_responses)
            tensor_answer_log_prob_diff = tensor_answer_log_prob[1:batch_count-pad_size] - tensor_answer_log_prob[:batch_count-pad_size-1]
            # tensor_answer_log_prob_diff = tensor_answer_log_prob[1:] - tensor_answer_log_prob[:-1]
            # compute average considering pad_size
            
            average_score = tensor_answer_log_prob_diff.mean()
            normalized_score = torch.sigmoid(average_score)
            # if valid_response_length > 0:  # Ensure valid index
            reward_tensor[i, valid_response_length - 1] = normalized_score