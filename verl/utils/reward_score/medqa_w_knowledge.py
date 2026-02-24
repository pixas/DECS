
import re
import torch
import numpy as np
import requests
from verl.utils.proxy import remove_proxy
# torch.set_default_device("cuda:0")
def extract_solution(solution_str, method='strict'):
    assert method in ['strict', 'flexible']

    if method == 'strict':
        match = re.search(r'(.*)[T|t]he answer is (.*?)(\.|$)', solution_str, re.DOTALL)
        if match:
            final_answer = match.group(2)
            
        else:
            final_answer = None
        
    return final_answer


def get_embedding(retriever, retriever_tokenizer, query):
    if isinstance(query, str):
        if len(query.split("\n")) > 1:
            # we can split the query so that each sentence is an atomic reasoning step
            query = query.split("\n")
        else:
            query = [query]
        batch = False
    elif isinstance(query, list):
        # 单个query内需要处理重复的内容
        # 过长的内容也要去除
        # 先去除重复的内容
        query = [list(set(x.split("\n"))) if len(x.split("\n")) > 1 else [x] for x in query]
        # 由于过长的内容会被截断，所以不需要处理
        
        query_length = [len(x) for x in query]
        query = [item for sublist in query for item in sublist]
        batch = True
        
    with torch.no_grad():
        encoded = retriever_tokenizer(
            query,
            truncation=True,
            padding=True,
            return_tensors='pt',
            max_length=512
        )
        encoded = {k: v.to(retriever.device) for k, v in encoded.items()}
        micro_batch_size = 16
        embeds = []
        for i in range(0, encoded['input_ids'].shape[0], micro_batch_size):
            micro_encoded = {k: v[i:i+micro_batch_size] for k, v in encoded.items()}
            micro_embeds = retriever(**micro_encoded).last_hidden_state[:, 0, :]
            embeds.append(micro_embeds)
        embeds = torch.cat(embeds, dim=0)
        
        # embeds = retriever(**encoded).last_hidden_state[:, 0, :] #[ k, d ] k is the reasoning step
    if not batch:
        # single query, embeds is just the embedding of each reasoning step of query
        return embeds
    else:
        # multiple queries, embeds is a list of tensors, each tensor is the embedding of each reasoning step of query
        # format the embeds as [b, k, d]
        output = []
        start = 0
        for length in query_length:
            output.append(embeds[start:start+length])
            start += length
        return output



@remove_proxy()
def get_embedding_api(api_path, query):
    if isinstance(query, str):
        if len(query.split("\n")) > 1:
            # we can split the query so that each sentence is an atomic reasoning step
            query = query.split("\n")
        else:
            query = [query]
        batch = False
    elif isinstance(query, list):
        # 单个query内需要处理重复的内容
        # 过长的内容也要去除
        # 先去除重复的内容
        new_query = []
        for x in query:
            sub_query = [new_query for new_query in x.split("\n") if new_query]
            if len(sub_query) > 1:
                new_query.append(list(dict.fromkeys(sub_query)))
            else:
                new_query.append(sub_query)
        query = new_query
        # 由于过长的内容会被截断，所以不需要处理
        query = [x for x in query if len(x) > 0]
        query_length = [len(x) for x in query]
        query = [item for sublist in query for item in sublist]
        batch = True
    payload = {
        "queries": query
    }
    response = requests.post(api_path, json=payload)
    response.raise_for_status()
    retrieved_data = response.json()
    embeds = torch.from_numpy(np.stack(retrieved_data['embeds'], axis=0))
    if not batch:
        # single query, embeds is just the embedding of each reasoning step of query
        return embeds
    else:
        # multiple queries, embeds is a list of tensors, each tensor is the embedding of each reasoning step of query
        # format the embeds as [b, k, d]
        output = []
        start = 0
        for length in query_length:
            output.append(embeds[start:start+length])
            start += length
        return output
        # embeds = [ for i, embed in enumerate(embeds)]


def compute_sim_score(response_embeds, knowledge_embeds):
    # response_embeds: [ k, d]
    # knowledge_embeds: [ m, d]
    # 
    response_norm = torch.nn.functional.normalize(response_embeds, p=2, dim=-1).float()
    knowledge_norm = torch.nn.functional.normalize(knowledge_embeds, p=2, dim=-1).float()
    sim = torch.einsum("kd,md->km", response_norm, knowledge_norm)
    # return the mean similarity score [b, ]
    return sim.mean().item()

def compute_score(data_source, solution_str, ground_truth, extra_info=None, method='strict', format_score=0., score=1.):
    """The scoring function for MedQA. Considers both format, accuracy and knowledge occurence

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str, method=method)
    if answer is None:
        return -1
    else:
        # if retriever is not None:
        #     knowledge_embeds = torch.from_numpy(np.stack(extra_info['knowledge_embeds'], axis=0)).to(retriever.device)
        #     response_embeds = get_embedding(retriever, retriever_tokenizer, solution_str)
        #     # Normalize the embeddings
        #     response_norm = torch.nn.functional.normalize(response_embeds, p=2, dim=1).float()
        #     knowledge_norm = torch.nn.functional.normalize(knowledge_embeds, p=2, dim=1).float()

        #     # Compute pairwise cosine similarities efficiently (shape: [a, b])
        #     sim = torch.matmul(response_norm, knowledge_norm.transpose(0, 1))
        #     # sim is a tensor of size [k, num_knowledge]
        #     knowledge_score = sim.mean().item()
        # else:
        #     if api_path is not None:

        #         knowledge_embeds = torch.from_numpy(np.stack(extra_info['knowledge_embeds'], axis=0))
        #         response_embeds = get_embedding_api(api_path, solution_str)
        #         # Normalize the embeddings
        #         response_norm = torch.nn.functional.normalize(response_embeds, p=2, dim=1).float()
        #         knowledge_norm = torch.nn.functional.normalize(knowledge_embeds, p=2, dim=1).float()
                
        #         sim = torch.matmul(response_norm, knowledge_norm.transpose(0, 1))
        #         knowledge_score = sim.mean().item()
        #     else:
                
        #         knowledge_score = 0
        if len(answer) == 1 and answer == ground_truth[0]:
            
            return score 
        if ground_truth[0] + "." in answer:
            return score 
        if ground_truth.split(".")[1].strip().lower() in answer.lower():
            return score 
        return format_score 

if __name__ == "__main__":
    embeds = get_embedding_api("http://10.140.0.170:10044/retrieve", ["Slow reacting substance of anaphylaxis (SRS-A) is a mix of compounds produced during allergic reactions and asthma.", ""])
    print(embeds.shape)
    