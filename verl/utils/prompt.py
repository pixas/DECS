
BASE_INFER_PROMPT = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The assistant thinks deeply and outputs the final answer as 'The answer is '. \nUser: {prompt}\nAssistant: """

BASE_GSM8K_PROMPT = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The assistant thinks deeply and output the final answer after \"####\". \nUser: {prompt}\nAssistant: """



BASE_MATH_PROMPT = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The assistant thinks deeply and output the final answer within \\boxed{{}}. \nUser: {prompt}\nAssistant: """

ORZ_PROMPT = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user
with the answer. The reasoning process and answer are enclosed within <think> </think> and
<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>. User: You must put your answer inside <answer> </answer> tags, i.e.,
<answer> answer here </answer>. And your final answer will be extracted automatically by the \\boxed{{}} tag.
{prompt}
Assistant: <think>"""

VERL_SYSTEM_PROMPT = "Please think step by step and output the final answer as 'The answer is '."

process_guide_prompt_direct = "\nThe answer is "
process_guide_prompt_chunk = "\n...Wait, I have no time. The answer is "

def base_prompt_map(data_source, prompt_type):
    if prompt_type == 'orz':
        return ORZ_PROMPT
    elif prompt_type == 'default':
        if "gsm8k" in data_source.lower():
            return BASE_GSM8K_PROMPT
        elif "math" in data_source.lower():
            return BASE_MATH_PROMPT
        elif 'orz' in data_source.lower():
            return BASE_MATH_PROMPT
            # return ORZ_PROMPT
        elif "aime" in data_source.lower():
            return BASE_MATH_PROMPT
        elif 'amc' in data_source.lower():
            return BASE_MATH_PROMPT