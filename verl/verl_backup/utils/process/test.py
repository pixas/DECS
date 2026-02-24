from dataclasses import dataclass

import torch

from verl.protocol import DataProto
from verl.utils.process.generation import GenerationConfig, ProcessRewardManager
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.set_default_device("cuda")

@dataclass
class ProcessConfig:
    enable: bool = True
    policy: str = 'ref' # could be ref and actor
    use_soft: str = None 
    step_reward: bool = True 
    factor: float = 0.3


config = ProcessConfig(enable=True, policy='actor', use_soft='tanh', step_reward=True, factor=0.3)
gen_config = GenerationConfig(
            1024,
            max_response_length=1024,
            num_gpus=1
        )
model_name_or_path = "/mnt/hwfile/medai/LLMModels/Model/Qwen2.5-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

manager = ProcessRewardManager(tokenizer, model,  gen_config, False, config)


prompt_str = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The assistant thinks deeply and outputs the final answer as 'The answer is '. \nUser: A 67-year-old man with transitional cell carcinoma of the bladder comes to the physician because of a 2-day history of ringing sensation in his ear. He received this first course of neoadjuvant chemotherapy 1 week ago. Pure tone audiometry shows a sensorineural hearing loss of 45 dB. The expected beneficial effect of the drug that caused this patient's symptoms is most likely due to which of the following actions?\nA. Inhibition of thymidine synthesis\nB. Inhibition of proteasome\nC. Hyperstabilization of microtubules\nD. Generation of free radicals\nE. Cross-linking of DNA\nAssistant: "

response_str = """1. The patient has a 67-year-old male with transitional cell carcinoma of the bladder who received neoadjuvant chemotherapy 1 week ago.\n2. The patient presents with a 2-day history of ringing sensation in his ear and a sensorineural hearing loss of 45 dB.\n3. Pure tone audiometry is used to assess hearing loss, and the results indicate a sensorineural hearing loss.\n4. The drug that caused the patient's symptoms is likely to have affected the auditory system.\n5. The options provided are related to different mechanisms of action of drugs.\n6. Inhibition of thymidine synthesis (option A) would not cause hearing loss.\n7. Inhibition of proteasome (option B) is not associated with hearing loss.\n8. Hyperstabilization of microtubules (option C) is not related to hearing loss.\n9. Generation of free radicals (option D) can cause oxidative stress and damage to cells, including auditory cells.\n10. Cross-linking of DNA (option E) is not directly related to hearing loss.\n11. The expected beneficial effect of the drug that caused the patient's symptoms is most likely due to its ability to generate free radicals.\n12. The drug that caused the patient's symptoms is likely to be a chemotherapeutic agent that generates free radicals as a side effect.\n\nThe answer is D."""

overall_str = prompt_str + response_str

prompt_id = tokenizer(prompt_str, return_tensors="pt", add_special_tokens=False)['input_ids']
response_id = tokenizer(response_str, return_tensors="pt", add_special_tokens=False)['input_ids']

overall_id = tokenizer(overall_str, return_tensors="pt", add_special_tokens=False)

batch = DataProto.from_dict(
    {
        "prompts": torch.cat([prompt_id, prompt_id]),
        "responses": torch.cat([response_id, response_id]),
        "attention_mask": torch.cat([overall_id['attention_mask'], overall_id['attention_mask']]),
    },
    {
        "reward_model": [{
            "ground_truth": "E. Cross-linking of DNA",
        },{
            "ground_truth": "E. Cross-linking of DNA",
        }], 
    },
    {
        "temperature": 1.0
    }
)
# print(batch[0])
manager.compute_ref_log_prob_reward(batch)



    