import re
import pdb
import json
import numpy as np
from collections import Counter
from math_verify import parse, ExprExtractionConfig, LatexExtractionConfig
from sympy import simplify

def extracted_postprocess(answer):
    return answer.strip(" \n*{")

def detect_answer_in_response(prediction):
    match = re.search(r'(.*)[T|t]he answer is (.*?)(\.|$)', prediction, re.DOTALL)
    return True if match else False

def extract_predicted_answer_util_end(prediction):
    match = re.search(r'(.*)[T|t]he answer is (.*?)$', prediction, re.DOTALL)

    return extracted_postprocess(match.group(2)) if match else prediction[-100:]

def extract_predicted_answer(prediction):
    match = re.search(r'(.*)[T|t]he answer is (.*?)(\.|$)', prediction, re.DOTALL)

    return extracted_postprocess(match.group(2)) if match else prediction[-100:]

def extract_gsm8k(prediction):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", prediction)
    if solution is None:
        return None 
    else:
        final_answer = solution.group(0)
        final_answer = final_answer.split('#### ')[1].replace(',', '').replace('$', '')
        return final_answer

def extract_math_answer(prediction, source):
    if "gsm8k" in source.lower():
        prediction = extract_gsm8k(prediction)
    

    pred = parse(prediction, )

    if isinstance(pred, list):
        if pred:
            pred = str(pred[0])
        else:
            pred = ""
    return pred

def single_choice_score(prediction, eval_info):
    prediction = extract_predicted_answer(prediction)
    if prediction is None:
        return False

    answer = eval_info["answer"]
    answer_idx = eval_info["answer_idx"]

    if answer.lower() in prediction.lower() or (answer_idx + ".") in prediction:
        return True
    
    elif len(prediction) == 1 and prediction == answer_idx:
        return True
    
    else:
        return False

def text_gen_score(prediction, eval_info):
    # prediction_idx = prediction.index("</think>") 
    # prediction = prediction[prediction_idx + len("</think>"):]
    prediction = extract_predicted_answer(prediction)
    if prediction is None:
        return False

    answer = eval_info["answer"]
    # answer_idx = eval_info["answer_idx"]

    if answer.lower() in prediction.lower():
        return True
    
    
    else:
        return False

def entity_match_score(prediction, eval_info):
    prediction = extract_predicted_answer_util_end(prediction)
    if prediction is None:
        return False
    
    answer = eval_info["answer"]

    if answer.lower() in prediction.lower():
        return True
    
    else:
        return False

def gsm8k_score(prediction, eval_info):
    answer = eval_info["answer"]
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", prediction)
    if solution is None:
        box_score = math_score(prediction, eval_info)
        return box_score
        
    else:
        final_answer = solution.group(0)
        final_answer = final_answer.split('#### ')[1].replace(',', '').replace('$', '')
        if final_answer == answer:
            return True
        else:
            return False

import signal
from contextlib import contextmanager

@contextmanager
def timeout(duration):
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Function execution exceeded {duration} seconds")
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    try:
        yield
    finally:
        signal.alarm(0)

def math_score(prediction, eval_info, debug=True):
    from verl.utils.reward_score.math_verify import compute_score
    try:
        # with timeout(10):
        score = compute_score(prediction, eval_info["answer"], consider_format=False)
    except (TimeoutError, Exception) as e:
        
        if debug:
            print(e)
            print(f"Error in computing math score for prediction: {prediction}, eval_info: {eval_info}")
        score = 0
    
    return bool(score)





def gpqa_score(prediction, eval_info) -> float:
    solution_str = prediction 
    ground_truth = eval_info["answer"]
    ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?"
    match = re.search(ANSWER_PATTERN_MULTICHOICE, solution_str)
    extracted_answer = match.group(1) if match else None
    if extracted_answer is None:
        # try to extract by \boxed{}
        extracted_answer = re.search(r"\\boxed{(.*?)}", solution_str)
        if extracted_answer:
            extracted_answer = extracted_answer.group(1)
        else:
            extracted_answer = None
    score = 1.0 if extracted_answer == ground_truth else 0.0
    return score

def estimate_pass_at_k(num_sample, num_correct, k):
    """Estimates pass@k of each problem and returns them in an array."""

    def estimator(n: int, c: int, k: int) -> float:
        """Calculates 1 - comb(n - c, k) / comb(n, k)."""
        if n - c < k:
            return 1.0
        return (1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))

    return estimator(int(num_sample), int(num_correct), k)

SCORE_FUNC = {
    # "math": math_score,
    # "gsm8k": gsm8k_score,
    "medqa": single_choice_score,
    "medmcqa": single_choice_score,
    "mmlu": single_choice_score,
    "pubmedqa": single_choice_score,
    "bioasq": single_choice_score,
    "medqa_5op": single_choice_score,
    "mmlu_medcare": single_choice_score,
    "medmcqa": single_choice_score,
    "medsins": entity_match_score,
    "medqa_open": entity_match_score,
    "medical_o1": entity_match_score,
    "ddxplus": text_gen_score,
    "gsm8k": gsm8k_score,
    "math": math_score,
    "olympiadbench": math_score,
    "aime": math_score,
    "amc": math_score,
    "gpqa": gpqa_score,
}

def score_task(task: dict, task_name: str = None):
    score = {}

    # find score func
    task_name = task_name if task_name is not None else task["task"]["dataset"]
    if task_name.lower() in SCORE_FUNC:
        score_func = SCORE_FUNC[task_name]

    elif any(dataset_name in task_name.lower() for dataset_name in SCORE_FUNC):
        for dataset_name in SCORE_FUNC:
            if dataset_name in task_name.lower():
                score_func = SCORE_FUNC[dataset_name]
                break
    else:
        print(f"Not found matched dataset name {task_name} in task name, use default score func")
        assert "answer_idx" in task["task"]["eval"] # default is the multiple-choice question
        score_func = single_choice_score
    
    
    # get answer acc
    assert "trajectory" in task
    if isinstance(task["trajectory"], str):
        score["acc"] = score_func(task["trajectory"], eval_info=task["task"]["eval"])
        if not hasattr(task, "prediction"):
            task["prediction"] = extract_predicted_answer(task["trajectory"])
         
    elif (isinstance(task["trajectory"], list) and len(task["trajectory"]) == 1):
        score["acc"] = score_func(task["trajectory"][0], eval_info=task["task"]["eval"])
        if not hasattr(task, "prediction"):
            task["prediction"] = extract_predicted_answer(task["trajectory"][0])
        
    elif isinstance(task["trajectory"], list):
        accuracy = [score_func(trajectory, eval_info=task["task"]["eval"]) for trajectory in task["trajectory"]]
        # predictions = [extract_predicted_answer(trajectory) for trajectory in task["trajectory"]]
        # print(accuracy, flush=True)
        
        predictions = [extract_math_answer(trajectory, task["task"]["dataset"]) for trajectory in task["trajectory"]]
        # print(predictions, flush=True)
        simplified_predictions = []
        for pred in predictions:
            if isinstance(pred, str):
                try:
                    with timeout(10):
                        pred = simplify(pred)
                except:
                    pred = pred 
            try:
                pred = str(pred)
            except:
                pred = ""
            simplified_predictions.append(pred)
        # print(simplified_predictions, flush=True)
        predictions = simplified_predictions
        # predictions = [str(simplify(pred)) if isinstance(pred, str) else pred for pred in predictions]
        # if len(predictions) == 0:
        #     import pdb
        #     pdb.set_trace()
            
        vote_prediction = Counter(predictions).most_common()[0][0]
        vote_index = predictions.index(vote_prediction)
        vote_acc = accuracy[vote_index]
        # vote_prediction = "</think>\\boxed{" + vote_prediction + "}"
        # vote_acc = score_func("</think>\\boxed{" + vote_prediction + "}", eval_info=task["task"]["eval"])
        # vote_acc = score_func(vote_prediction, eval_info=task["task"]["eval"])
        score["acc"] = vote_acc
        score["avg_acc"] = sum(accuracy) / len(accuracy)
        score["least_acc"] = float(sum(accuracy) > 0)
        task["prediction"] = vote_prediction
        
        task['predictions'] = predictions
        task['vote_token'] = task['tokens'][vote_index]
        task['all_acc'] = accuracy
        task['average_tokens'] = sum(task['tokens']) / len(task['tokens'])
    
    else:
        raise NotImplementedError
    
    # get answer pass@k
    if "trajectory" in task and len(task["trajectory"]) > 1:
        sample_num = len(task["trajectory"])
        correct_num = sum(accuracy)
        for k in range(1, sample_num+1):
            score[f"pass@{k}"] = estimate_pass_at_k(sample_num, correct_num, k)
        
    return score

