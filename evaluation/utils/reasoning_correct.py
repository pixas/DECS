import os 
import json
import argparse

# we define a reasoning trajectory is correct if all adjacent reasoning steps
# follow the increasing order
# for each item[trajectory_answer_prob], we check item[trajectory_answer_prob][i] < item[trajectory_answer_prob][i+1]
def is_trajectory_correct(trajectory_answer_prob):
    """
    Check if a reasoning trajectory is correct based on whether 
    the probabilities are in increasing order.
    
    Args:
        trajectory_answer_prob (list): A list of probabilities representing the reasoning trajectory.
        
    Returns:
        float: 1 if each probability is smaller than the next one for all pairs, otherwise a [0,1) float.
    """
    if len(trajectory_answer_prob) <= 1:
        return True
    
    total_pairs = len(trajectory_answer_prob) - 1
    correct_pairs = sum(1 for i in range(len(trajectory_answer_prob) - 1) if trajectory_answer_prob[i] <= trajectory_answer_prob[i + 1])
    return correct_pairs / total_pairs

def check_reasoning_trajectories(data):
    """
    Analyze reasoning trajectories in a collection of data items.
    
    Args:
        data: Can be a single item, a list of items, or a file path to a JSON file.
             Each item should contain a 'trajectory_answer_prob' key.
        
    Returns:
        dict: Statistics about correct and incorrect trajectories.
    """
    items = []
    
    # Handle different input types
    if isinstance(data, str) and os.path.exists(data):
        items = [json.loads(line) for line in open(data, 'r')]
    elif isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = [data]
    else:
        raise ValueError("Input must be a file path, a list of items, or a single item dictionary")
    
    results = []
    for item in items:
        trajectory = item.get('trajectory_answer_prob', [])
        correct_score = is_trajectory_correct(trajectory)
        results.append({
            'trajectory': trajectory,
            'correct_score': correct_score
        })
    
    # Compute statistics
    overall_score = sum(item['correct_score'] for item in results) / len(results) if results else 0
    
    return {
        "source": data if isinstance(data, str) else "input",
        'total': len(results),
        "overall_score": overall_score,
        'detailed_results': results
    }
    
if __name__ == "__main__":
    # Test the function with a sample data
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_data", type=str, default="sample_data.jsonl")
    args = parser.parse_args()
    results = check_reasoning_trajectories(args.sample_data)
    # save the result to the same dir with sample_data with the name 'trajectory_score.json'
    with open(args.sample_data.replace(".jsonl", "_score.json"), 'w') as f:
        json.dump(results, f, indent=2)
    