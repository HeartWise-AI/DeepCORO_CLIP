import unittest
from utils.config.clip_config import ClipConfig


class TestClipConfigSimple(unittest.TestCase):
    """Simple test cases for ClipConfig class to cover __post_init__ method."""
    
    def test_post_init_with_none_values(self):
        """Test that __post_init__ sets default values when recall_k and ndcg_k are None."""
        
        # Create minimal config with required fields and None values for lists
        config = ClipConfig(
            lr=0.001,
            batch_size=32,
            num_workers=4,
            debug=False,
            temperature=0.1,
            max_grad_norm=1.0,
            data_filename='test.csv',
            root='/tmp',
            target_label='label',
            datapoint_loc_label='location',
            frames=16,
            stride=1,
            multi_video=False,
            num_videos=1,
            groupby_column='group',
            shuffle_videos=False,
            aggregate_videos_tokens=False,
            per_video_pool=False,
            model_name='test_model',
            pretrained=True,
            video_freeze_ratio=0.0,
            text_freeze_ratio=0.0,
            dropout=0.1,
            num_heads=8,
            aggregator_depth=2,
            optimizer='Adam',
            scheduler_name='cosine',
            lr_step_period=10,
            factor=0.1,
            video_weight_decay=1e-4,
            text_weight_decay=1e-4,
            gradient_accumulation_steps=1,
            num_warmup_percent=0.1,
            num_hard_restarts_cycles=0,
            warm_restart_tmult=2,
            use_amp=False,
            period=10,
            loss_name='contrastive',
            recall_k=None,  # This should trigger default assignment
            ndcg_k=None,    # This should trigger default assignment
            rand_augment=False,
            resize=224,
            apply_mask=False,
            save_best='loss',
            resume_training=False,
            checkpoint=None,
            topk=5,
            text_embeddings_path='/tmp/embeddings.pkl',
            metadata_path='/tmp/metadata.json',
            inference_results_path='/tmp/results.json',
            epochs=10,
            pipeline_project='DeepCORO_clip',
            base_checkpoint_path='/tmp/checkpoint.pth',
            run_mode='train',
            seed=42,
            name='test_run',
            project='test_project',
            entity='test_entity',
            tag='test_tag',
            use_wandb=False,
            world_size=1,
            device=0,
            is_ref_device=True,
        )
        
        # Check that default values were set by __post_init__
        self.assertEqual(config.recall_k, [1, 5])
        self.assertEqual(config.ndcg_k, [5])


if __name__ == '__main__':
    unittest.main() 