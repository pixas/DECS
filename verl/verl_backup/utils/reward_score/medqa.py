
import re


def extract_solution(solution_str, method='strict'):
    assert method in ['strict', 'flexible']

    if method == 'strict':
        match = re.search(r'(.*)[T|t]he answer is (.*?)(\.|$)', solution_str, re.DOTALL)
        if match:
            final_answer = match.group(2)
            
        else:
            final_answer = None
        
    return final_answer

def compute_score(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """The scoring function for MedQA.

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
        return 0
    else:
        if len(answer) == 1 and answer == ground_truth[0]:
            return score 
        if ground_truth[0] + "." in answer:
            return score 
        if ground_truth.split(".")[1].strip().lower() in answer.lower():
            return score 
        return format_score
