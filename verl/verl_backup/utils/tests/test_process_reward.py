from verl.trainer.ppo.ray_trainer import RayPPOTrainer
import pytest
import torch
from verl import DataProto
from unittest.mock import MagicMock, patch

class TestComputeProcessReward:

    @pytest.fixture
    def trainer(self):
        # Mock trainer with required attributes
        config = MagicMock()
        tokenizer = MagicMock()
        role_worker_mapping = {}
        resource_pool_manager = MagicMock()

        # Create mock tokenizer methods
        tokenizer.decode.side_effect = lambda x: "test prompt" if len(x) == 1 else "test response" 
        tokenizer.encode.return_value = [1, 2, 3]
        tokenizer.return_value = {
            'input_ids': torch.tensor([[1,2,3]]),
            'attention_mask': torch.tensor([[1,1,1]]),
        }

        trainer = RayPPOTrainer(config, tokenizer, role_worker_mapping, resource_pool_manager)

        # Mock actor_rollout_wg
        trainer.actor_rollout_wg = MagicMock()
        trainer.actor_rollout_wg.compute_log_prob.return_value = DataProto(
            batch={'ref_log_prob': torch.tensor([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]])},
            non_tensor_batch={},
            meta_info={}
        )
        
        return trainer

    def test_compute_process_reward_ref_log_prob(self, trainer):
        # Create test batch
        batch = DataProto(
            batch={
                'prompts': torch.tensor([[1]]),
                'responses': torch.tensor([[2, 3, 4]]),
                'attention_mask': torch.tensor([1, 1, 1, 1])
            },
            non_tensor_batch={
                'reward_model': {
                    'ground_truth': 'test'
                }
            },
            meta_info={}
        )

        # Call function
        reward_tensor = trainer.compute_process_reward('ref_log_prob', batch)

        # Verify output
        assert isinstance(reward_tensor, torch.Tensor)
        assert reward_tensor.shape == batch.batch['responses'].shape
        assert reward_tensor.dtype == torch.float32
        
        # Verify reward is only placed at end of sequence
        assert torch.all(reward_tensor[:,:-1] == 0)
        assert reward_tensor[:,-1] != 0

    def test_compute_process_reward_empty_response(self, trainer):
        # Test with empty response after splitting
        batch = DataProto(
            batch={
                'prompts': torch.tensor([[1]]),
                'responses': torch.tensor([[2]]), 
                'attention_mask': torch.tensor([1, 1])
            },
            non_tensor_batch={
                'reward_model': {
                    'ground_truth': 'test'  
                }
            },
            meta_info={}
        )

        reward_tensor = trainer.compute_process_reward('ref_log_prob', batch)
        
        assert torch.all(reward_tensor == 0)

    def test_compute_process_reward_multiple_responses(self, trainer):
        # Test with multiple sub-responses after splitting
        trainer.tokenizer.decode.side_effect = lambda x: "response1\nresponse2"
        
        batch = DataProto(
            batch={
                'prompts': torch.tensor([[1]]),
                'responses': torch.tensor([[2, 3, 4]]),
                'attention_mask': torch.tensor([1, 1, 1, 1]) 
            },
            non_tensor_batch={
                'reward_model': {
                    'ground_truth': 'test'
                }
            },
            meta_info={}
        )

        reward_tensor = trainer.compute_process_reward('ref_log_prob', batch)

        assert isinstance(reward_tensor, torch.Tensor)
        assert reward_tensor.shape == batch.batch['responses'].shape
        assert torch.all(reward_tensor[:,:-1] == 0)
        assert reward_tensor[:,-1] != 0