# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

try:
    from math_verify.metric import math_metric
    from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig
    from math_verify.errors import TimeoutException
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")
from functools import wraps
import dill

def make_pickleable(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    wrapper._pickle_dump = dill.dumps(func)
    return wrapper

import re 

@make_pickleable
def think_compute_score(data_source, solution_str, ground_truth, extra_info=None, format_score=0) -> bool:
    # first check whether model_output is in the format of <think>xxx</think>new line or space or no content<answer>answer</answer>
    model_output = "<think>" + solution_str
    if model_output.count("<think>") != 1 or model_output.count("</think>") != 1 :
        return format_score
    res = re.search(r"<think>(.*?)</think>\s*<answer>(.*?)</answer>", model_output, re.DOTALL)
    if not res:
        return format_score 
    answer_content = res.group(2).strip()
    if "boxed" not in answer_content:
        answer_content = f"\\boxed{{{answer_content}}}"
    return compute_score(answer_content, ground_truth, 0)
    



def compute_score(model_output: str, ground_truth: str, format_score: float=0, timeout_score: float = 0, consider_format=True) -> bool:
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    ret_score = format_score

    # Wrap the ground truth in \boxed{} format for verification
    ground_truth_boxed = "\\boxed{" + ground_truth + "}" if "boxed" not in ground_truth else ground_truth
    # model_output = "<think>" + model_output
    if consider_format and (model_output.count("</think>") != 1 ):
        return ret_score 
    try:
        model_answer_subseq = model_output.split("</think>")[-1] if consider_format else model_output
        ret_score, _ = verify_func([ground_truth_boxed], [model_answer_subseq])

    except Exception as e:
        pass
    except TimeoutException as e:
        print(model_output[-100:], e)
        ret_score = timeout_score

    return ret_score
