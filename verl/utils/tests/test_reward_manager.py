from verl.workers.reward_manager.naive import NaiveRewardManager
import pytest
import torch
from verl import DataProto
import numpy as np

from verl.utils.reward_score.medqa_w_knowledge import compute_score
from tensordict import TensorDict

class TestNaiveRewardManager:
    @pytest.fixture
    def reward_manager(self):
        # Mock tokenizer
        class MockTokenizer:
            def decode(self, ids):
                if 7 in ids:
                    return "The answer is test"
                else:
                    return "test text"
                # return "test text"
                
        tokenizer = MockTokenizer()
        return NaiveRewardManager(tokenizer=tokenizer, num_examine=1, compute_score=compute_score)

    def test_call_with_rm_scores(self, reward_manager):
        # Test when rm_scores already exists
        batch = {
            'rm_scores': torch.tensor([0.5, 0.7])
        }
        data = DataProto(batch=batch)
        
        result = reward_manager(data)
        assert torch.equal(result, batch['rm_scores'])

    def test_call_compute_scores(self, reward_manager):
        # Test computing reward scores
        batch = {
            'prompts': torch.tensor([[1, 2, 3]]),
            'responses': torch.tensor([[4, 5, 6]]),
            'attention_mask': torch.tensor([1, 1, 1, 1, 1, 1])
        }
        non_tensor_batch = {
            'reward_model': {'ground_truth': 'test'},
            'data_source': 'test_source'
        }
        data = DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
        
        result = reward_manager(data)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == batch['responses'].shape
        assert result.dtype == torch.float32

    def test_call_with_api(self, reward_manager):
        # Test with API path
        reward_manager.api_path = "http://10.140.0.170:10044/retrieve"
        
        batch = TensorDict({
            'prompts': torch.tensor([[1, 2, 3], [2, 3, 4]]),
            'responses': torch.tensor([[4, 5, 6], [7, 8, 9]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]),
        }, batch_size=2)
        non_tensor_batch = {
            'reward_model': np.array([{'ground_truth': 'A.test'}, {"ground_truth": 'A.test'}]),
            'data_source': np.array(['test_source', 'test_source'], dtype=object),
            'extra_info': np.array([{'knowledge_embeds': torch.randn(2, 768).tolist()}, {'knowledge_embeds': torch.randn(2, 768).tolist()}]) 
        }
        # for key, val in non_tensor_batch.items():
        #     print(type(val), val.dtype)
        data = DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
        score = reward_manager(data)
        print(score)
        with pytest.raises(Exception):
            # Should raise exception when API is not reachable
            score = reward_manager(data)
            print(score)