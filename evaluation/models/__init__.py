

def get_model(model_name, peft_path=None, use_vllm=False, max_new_tokens=8192, temperature=0.0, hf_model_path=None, **kwargs):
    
    # api model
    # if model_name == "gpt-3.5-turbo":
    #     return OpenAI_Model(model_type="gpt-3.5-turbo")
    
    # elif model_name == "gpt-4o":
    #     return OpenAI_Model(model_type="gpt-4o")
    
    # elif model_name == "gpt-4o-mini":
    #     return OpenAI_Model(model_type="gpt-4o-mini")

    # else:
    if use_vllm:
        from evaluation.models.vllm_models import VLLM_Model
        from rollout.vllm_rollout import VLLMRollout
        return VLLMRollout(model_name, response_length=max_new_tokens, temperature=temperature, hf_model_path=hf_model_path, **kwargs)
        # return VLLM_Model(model_name, peft_path=peft_path)
    
    else:
        from evaluation.models.base_model import Local_Model
        return Local_Model(model_name, peft_path=peft_path)