
# from evaluation.models.base_model import LOCAL_MODEL_PATHS
from functools import partial
DEFAULT_PROMPT = "You are a helpful assistant."
INSTRUCT_MATH = "Think the math problem step by step and output the final answer within \\boxed{}"
# SYSTEM_PROMPT = """You are a helpful assistant at solving complex medical problems.
# You should first think about the reasoning process in the mind and then directly put the answer in the <answer></answer> tags. The reasoning process and answer should be enclosed within <think> </think> and
# <answer> </answer> tags, respectively."""

SYSTEM_PROMPT = """You are a helpful assistant at solving complex medical problems.
You should first think about the reasoning process in the mind and then summarize the answer by stating 'The answer is' in the end. The reasoning process should be enclosed within <think> </think> tages. After </think> tag, output your summarized answer."""

# SYSTEM_PROMPT = "Please think step by step and output the final answer as 'The answer is '."

VERL_SYSTEM_PROMPT = "Please think step by step and output the final answer as 'The answer is '."

THE_ANSWER_IS_PROMPT = """You are a helpful assistant at solving complex medical problems.
You should first think about the reasoning process in the mind and then summarize the answer by stating 'The answer is' in the end.
"""

BASE_PROMPT = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer, ensuring that the final result in the answer is given as the answer index directly. The reasoning process and answer are enclosed within '<think>' '</think>' and '<answer>' '</answer>' tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {prompt}\nAssistant: """

BASE_INFER_PROMPT = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The assistant thinks deeply and outputs the final answer as 'The answer is '. \nUser: {prompt}\nAssistant: """


BASE_GSM8K_PROMPT = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The assistant thinks deeply and output the final answer after \"####\". \nUser: {prompt}\nAssistant: """



BASE_MATH_PROMPT = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The assistant thinks deeply and output the final answer within \\boxed{{}}. \nUser: {prompt}\nAssistant: """

RAW_PROMPT = """{prompt}\n"""

THINK_BASE_INFER_PROMPT = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process is enclosed within '<think>' '</think>'. After </think> tag, the assistant outputs the summarized answer. \nUser: {prompt}\nAssistant: """

ORZ_PROMPT = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user
with the answer. The reasoning process and answer are enclosed within <think> </think> and
<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>. User: You must put your answer inside <answer> </answer> tags, i.e.,
<answer> answer here </answer>. And your final answer will be extracted automatically by the \\boxed{{}} tag.
{prompt}
Assistant: <think>"""

prompt_mapping = {
    "base_default": BASE_INFER_PROMPT,
    "base_think": THINK_BASE_INFER_PROMPT,
    "base_zero": BASE_PROMPT,
    "base_orz": ORZ_PROMPT,
    "base_math": BASE_MATH_PROMPT,
    "instruct_default": DEFAULT_PROMPT,
    "instruct_think": SYSTEM_PROMPT,
    "instruct_verl": VERL_SYSTEM_PROMPT,
    "instruct_answer": THE_ANSWER_IS_PROMPT,
    "instruct_math": INSTRUCT_MATH,
    "raw": RAW_PROMPT
}

def prepare_inputs_for_instruct(sample, prompt):
    input_ = {
            "prompt": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": sample["input"] if "context" not in sample else f"""{sample["context"]}\n\n{sample["input"]}""" },    
            ],
            "completion": [
                {"role": "assistant", "content": sample["output"] if "output" in sample else ""},    
            ],
            "label": sample["eval"]
        }
    if prompt == DEFAULT_PROMPT:
        input_['prompt'] = input_['prompt'][1:]
    return input_

def prepare_inputs_for_instruct_sft(sample, prompt):
    input_ = {
            "prompt": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": sample["input"]},    
            ],
            "completion": [
                {"role": "assistant", "content": sample["output"] if "output" in sample else ""},    
            ],
            "label": sample["eval"]
        }
    
    return input_


def prepare_inputs_for_gsm8k_sft(sample, prompt=None):
    input_ = {
            "prompt": [
                {"role": "user", "content": sample["input"] + " Let's think step by step and output the final answer after \"####\"."},    
            ],
            "completion": [
                {"role": "assistant", "content": sample["output"] if "output" in sample else ""},    
            ],
            "label": sample["eval"]
        }
    
    return input_

def prepare_inputs_for_math_sft(sample, prompt=None):
    input_ = {
            "prompt": [
                {"role": "user", "content": sample["input"] + " Let's think step by step and output the final answer within \\boxed{}."},    
            ],
            "completion": [
                {"role": "assistant", "content": sample["output"] if "output" in sample else ""},    
            ],
            "label": sample["eval"]
        }
    
    return input_

def prepare_inputs_for_medqa_sft(sample, prompt):
    input_ = {
            "prompt": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": sample["input"]},    
            ],
            "completion": [
                {"role": "assistant", "content": sample["output"] if "output" in sample else ""},    
            ],
            "label": sample["eval"]
        }
    
    return input_
    
    
def prepare_inputs_for_base(sample, prompt):
    input_ = {
            "prompt": prompt.format(prompt=sample["input"]),
            "completion": sample["output"] if "output" in sample else "",
            "label": sample["eval"]
        }
    
    return input_



# def get_prepare_func(model_name_or_path, data_format = None):
#     # data format can be three things: 
#     # None: for inference
#     # think: for rl or r1-distill sft or r1-distill model inference
#     # raw: instruct/base sft/rl
#     if "verl" in model_name_or_path:
#         # this is the verl-based rl checkpoint
#         # either based on instruct or base model to rl
#         global SYSTEM_PROMPT
#         global BASE_PROMPT
#         SYSTEM_PROMPT = VERL_SYSTEM_PROMPT
#         BASE_PROMPT = BASE_INFER_PROMPT
    
#     if model_name_or_path in LOCAL_MODEL_PATHS:
#         if "base" in model_name_or_path.lower():
#             # base model inference or base model sft (we only conduct rl on verl)
#             # detect data_format
#             if data_format is None:
#                 # base inference
#                 return prepare_inputs_for_base 
#             elif data_format == "raw":
#                 # base sft
#                 return prepare_inputs_for_base
#             elif data_format == "think":
#                 # base sft on r1-distill data
#                 # global BASE_PROMPT
#                 BASE_PROMPT = BASE_INFER_PROMPT
#                 return prepare_inputs_for_base
        
#         else:
#             # instruct model inference or instruct model sft
#             if data_format is None:
                
#                 return prepare_inputs_for_instruct
#             elif data_format == "raw":
#                 return prepare_inputs_for_instruct
#             elif data_format == "think":
#                 # instruct model rl on trl 
#                 return prepare_inputs_for_instruct_sft
    
#     else:
#         if "base" in model_name_or_path.lower():
#             # this is a base fine-tuned model
#             # detect data_format to change the BASE_PROMPT
#             if data_format == "think":
#                 # global BASE_PROMPT
#                 BASE_PROMPT = THINK_BASE_INFER_PROMPT
#             elif data_format == "raw":
#                 # global BASE_PROMPT
#                 BASE_PROMPT = BASE_INFER_PROMPT
            
#             return prepare_inputs_for_base
#         else:
#             # this is an instruct fine-tuned model
#             # SFT or RL
#             # global SYSTEM_PROMPT 
#             # SYSTEM_PROMPT = DEFAULT_PROMPT
#             return prepare_inputs_for_instruct_sft
#         # return prepare_inputs_for_instruct

def get_prepare_func(model_name_or_path, prompt_type, data_source=None):
    prompt = prompt_mapping.get(prompt_type, None)
    if prompt is None:
        raise ValueError(f"Prompt type '{prompt_type}' not found.")

    # Handle specific data sources with appropriate prompts
    if data_source:
        if "gsm8k" in data_source.lower():
            # For GSM8K, use the specialized prompt or the default one
            if prompt_type == "base_default" or prompt_type == "base_think":
                return partial(prepare_inputs_for_base, prompt=BASE_GSM8K_PROMPT)
            else:
                return prepare_inputs_for_gsm8k_sft
        elif "math" in data_source.lower() or "aime" in data_source.lower() or 'amc' in data_source.lower():
            # For Math, use the specialized math prompt
            if prompt_type == 'base_orz':
                return partial(prepare_inputs_for_base, prompt=prompt)
            if "base" in prompt_type:
                return partial(prepare_inputs_for_base, prompt=BASE_MATH_PROMPT)
            if 'raw' in prompt_type:
                return partial(prepare_inputs_for_base, prompt=RAW_PROMPT)
            else:
                return prepare_inputs_for_math_sft
        elif data_source == "medqa":
            return partial(prepare_inputs_for_medqa_sft, prompt=prompt)

    
    # Default behavior (when data_source is None)
    if "base" in prompt_type:
        return partial(prepare_inputs_for_base, prompt=prompt)
    else:
        return partial(prepare_inputs_for_instruct, prompt=prompt)
