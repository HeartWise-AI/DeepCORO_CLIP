import unittest
import argparse
from unittest.mock import patch, MagicMock, mock_open
import sys
import json

# Import modules to test
from utils.parser import BaseParser, ClipParser, LinearProbingParser, HeartWiseParser
from utils.registry import ParserRegistry
from utils.config.heartwise_config import HeartWiseConfig
from utils.parser_typing import str2bool, parse_list, parse_optional_int, parse_optional_str


class TestBaseParser(unittest.TestCase):
    """Tests for BaseParser class"""

    def test_init(self):
        """Test initialization of BaseParser"""
        description = "Test Parser"
        parser = BaseParser(description)
        
        self.assertIsInstance(parser.parser, argparse.ArgumentParser)
        self.assertEqual(parser.parser.description, description)
        
        # Check if base arguments were added
        args, _ = parser.parser.parse_known_args(['--base_config', 'test.yaml'])
        self.assertEqual(args.base_config, 'test.yaml')
        
    def test_add_base_arguments(self):
        """Test adding base arguments"""
        parser = BaseParser("Test")
        
        # Check base arguments
        args, _ = parser.parser.parse_known_args(['--base_config', 'test.yaml', '--use_wandb', 'True', 
                                                 '--name', 'test_run', '--project', 'test_proj', 
                                                 '--entity', 'test_entity'])
        
        self.assertEqual(args.base_config, 'test.yaml')
        self.assertTrue(args.use_wandb)
        self.assertEqual(args.name, 'test_run')
        self.assertEqual(args.project, 'test_proj')
        self.assertEqual(args.entity, 'test_entity')
    
    def test_add_common_arguments(self):
        """Test adding common arguments"""
        parser = BaseParser("Test")
        
        # Check common arguments
        args, _ = parser.parser.parse_known_args(['--base_config', 'test.yaml', '--seed', '42', 
                                                 '--device', 'cuda', '--world_size', '2'])
        
        self.assertEqual(args.base_config, 'test.yaml')
        self.assertEqual(args.seed, 42)
        self.assertEqual(args.device, 'cuda')
        self.assertEqual(args.world_size, 2)


class TestClipParser(unittest.TestCase):
    """Tests for ClipParser class"""

    def setUp(self):
        """Setup for tests"""
        self.clip_parser = ClipParser()
    
    def test_initialization(self):
        """Test initialization of ClipParser"""
        self.assertIsInstance(self.clip_parser, BaseParser)
        self.assertEqual(self.clip_parser.parser.description, "Train DeepCORO_CLIP model (Contrastive Pipeline)")
        self.assertTrue(self.clip_parser.parser.add_help)
        
    def test_add_clip_arguments(self):
        """Test adding clip-specific arguments"""
        # Test training parameters
        args, _ = self.clip_parser.parser.parse_known_args([
            '--base_config', 'test.yaml',
            '--lr', '0.001',
            '--batch_size', '32',
            '--num_workers', '4',
            '--debug', 'True',
            '--temperature', '0.07',
            '--base_checkpoint_path', '/path/to/checkpoint'
        ])
        
        self.assertEqual(args.lr, 0.001)
        self.assertEqual(args.batch_size, 32)
        self.assertEqual(args.num_workers, 4)
        self.assertTrue(args.debug)
        self.assertEqual(args.temperature, 0.07)
        self.assertEqual(args.base_checkpoint_path, '/path/to/checkpoint')
        
        # Test data parameters
        args, _ = self.clip_parser.parser.parse_known_args([
            '--base_config', 'test.yaml',
            '--data_filename', 'data.csv',
            '--root', '/data/root',
            '--target_label', 'description',
            '--datapoint_loc_label', 'video_path',
            '--frames', '16',
            '--stride', '2',
            '--multi_video', 'True',
            '--num_videos', '5',
            '--groupby_column', 'patient_id',
            '--shuffle_videos', 'True'
        ])
        
        self.assertEqual(args.data_filename, 'data.csv')
        self.assertEqual(args.root, '/data/root')
        self.assertEqual(args.target_label, 'description')
        self.assertEqual(args.datapoint_loc_label, 'video_path')
        self.assertEqual(args.frames, 16)
        self.assertEqual(args.stride, 2)
        self.assertTrue(args.multi_video)
        self.assertEqual(args.num_videos, 5)
        self.assertEqual(args.groupby_column, 'patient_id')
        self.assertTrue(args.shuffle_videos)
        
        # Test model parameters
        args, _ = self.clip_parser.parser.parse_known_args([
            '--base_config', 'test.yaml',
            '--model_name', 'resnet50',
            '--pretrained', 'True',
            '--video_freeze_ratio', '0.5',
            '--text_freeze_ratio', '0.3',
            '--dropout', '0.2',
            '--num_heads', '2',
            '--aggregator_depth', '3'
        ])
        
        self.assertEqual(args.model_name, 'resnet50')
        self.assertTrue(args.pretrained)
        self.assertEqual(args.video_freeze_ratio, 0.5)
        self.assertEqual(args.text_freeze_ratio, 0.3)
        self.assertEqual(args.dropout, 0.2)
        self.assertEqual(args.num_heads, 2)
        self.assertEqual(args.aggregator_depth, 3)
        
        # Test optimization parameters
        args, _ = self.clip_parser.parser.parse_known_args([
            '--base_config', 'test.yaml',
            '--max_grad_norm', '1.0',
            '--optimizer', 'AdamW',
            '--scheduler_name', 'cosine',
            '--lr_step_period', '10',
            '--factor', '0.1',
            '--video_weight_decay', '0.01',
            '--text_weight_decay', '0.01',
            '--gradient_accumulation_steps', '2',
            '--num_warmup_percent', '0.1',
            '--num_hard_restarts_cycles', '3',
            '--warm_restart_tmult', '2'
        ])
        
        self.assertEqual(args.max_grad_norm, 1.0)
        self.assertEqual(args.optimizer, 'AdamW')
        self.assertEqual(args.scheduler_name, 'cosine')
        self.assertEqual(args.lr_step_period, 10)
        self.assertEqual(args.factor, 0.1)
        self.assertEqual(args.video_weight_decay, 0.01)
        self.assertEqual(args.text_weight_decay, 0.01)
        self.assertEqual(args.gradient_accumulation_steps, 2)
        self.assertEqual(args.num_warmup_percent, 0.1)
        self.assertEqual(args.num_hard_restarts_cycles, 3)
        self.assertEqual(args.warm_restart_tmult, 2)
        
        # Test system parameters
        args, _ = self.clip_parser.parser.parse_known_args([
            '--base_config', 'test.yaml',
            '--use_amp', 'True',
            '--period', '5'
        ])
        
        self.assertTrue(args.use_amp)
        self.assertEqual(args.period, 5)
        
        # Test metrics parameters
        args, _ = self.clip_parser.parser.parse_known_args([
            '--base_config', 'test.yaml',
            '--loss_name', 'CLIPLoss',
            '--recall_k', '[1,5,10]',
            '--ndcg_k', '[1,5,10]'
        ])
        
        self.assertEqual(args.loss_name, 'CLIPLoss')
        self.assertEqual(args.recall_k, [1, 5, 10])
        self.assertEqual(args.ndcg_k, [1, 5, 10])
        
        # Test augmentation parameters
        args, _ = self.clip_parser.parser.parse_known_args([
            '--base_config', 'test.yaml',
            '--rand_augment', 'true',
            '--resize', '224',
            '--apply_mask', 'false'
        ])
        
        self.assertEqual(args.rand_augment, True)
        self.assertEqual(args.resize, 224)
        self.assertEqual(args.apply_mask, False)
        
        # Test checkpoint parameters
        args, _ = self.clip_parser.parser.parse_known_args([
            '--base_config', 'test.yaml',
            '--save_best', 'recall@1',
            '--resume_training', 'True',
            '--checkpoint', 'checkpoint.pt'
        ])
        
        self.assertEqual(args.save_best, 'recall@1')
        self.assertTrue(args.resume_training)
        self.assertEqual(args.checkpoint, 'checkpoint.pt')
        
    @patch('utils.config.heartwise_config.HeartWiseConfig.update_config_with_args')
    def test_parse_args_and_update(self, mock_update_config):
        """Test parsing arguments and updating config"""
        # Setup mock
        mock_config = MagicMock()
        mock_update_config.return_value = mock_config
        
        # Define args
        args_list = ['--base_config', 'test.yaml', '--lr', '0.001']
        
        # Call the method
        result = self.clip_parser.parse_args_and_update(mock_config, args_list)
        
        # Check the result
        self.assertEqual(result, mock_config)
        mock_update_config.assert_called_once()


class TestLinearProbingParser(unittest.TestCase):
    """Tests for LinearProbingParser class"""

    def setUp(self):
        """Setup for tests"""
        self.lp_parser = LinearProbingParser()
    
    def test_initialization(self):
        """Test initialization of LinearProbingParser"""
        self.assertIsInstance(self.lp_parser, BaseParser)
        self.assertEqual(self.lp_parser.parser.description, "Run Linear Probing Evaluation")
        self.assertFalse(self.lp_parser.parser.add_help)
        
        # Check supported unknown args
        self.assertIn('head_lr', self.lp_parser.supported_unknown_args)
        self.assertIn('head_weight_decay', self.lp_parser.supported_unknown_args)
        self.assertEqual(self.lp_parser.supported_unknown_args['head_lr'], float)
        self.assertEqual(self.lp_parser.supported_unknown_args['head_weight_decay'], float)
        
    def test_add_linear_probing_arguments(self):
        """Test adding linear probing arguments"""
        # Test training parameters
        args, _ = self.lp_parser.parser.parse_known_args([
            '--base_config', 'test.yaml',
            '--scheduler_name', 'cosine',
            '--lr_step_period', '10',
            '--factor', '0.1',
            '--optimizer', 'AdamW',
            '--video_encoder_weight_decay', '0.01',
            '--use_amp', 'True',
            '--gradient_accumulation_steps', '2',
            '--num_warmup_percent', '0.1',
            '--num_hard_restarts_cycles', '3',
            '--warm_restart_tmult', '2'
        ])
        
        self.assertEqual(args.scheduler_name, 'cosine')
        self.assertEqual(args.lr_step_period, 10)
        self.assertEqual(args.factor, 0.1)
        self.assertEqual(args.optimizer, 'AdamW')
        self.assertEqual(args.video_encoder_weight_decay, 0.01)
        self.assertTrue(args.use_amp)
        self.assertEqual(args.gradient_accumulation_steps, 2)
        self.assertEqual(args.num_warmup_percent, 0.1)
        self.assertEqual(args.num_hard_restarts_cycles, 3)
        self.assertEqual(args.warm_restart_tmult, 2)
        
        # Test data parameters
        args, _ = self.lp_parser.parser.parse_known_args([
            '--base_config', 'test.yaml',
            '--data_filename', 'data.csv',
            '--num_workers', '4',
            '--batch_size', '32',
            '--datapoint_loc_label', 'video_path',
            '--target_label', 'label',
            '--rand_augment', 'True',
            '--resize', '224',
            '--frames', '16',
            '--stride', '2'
        ])
        
        self.assertEqual(args.data_filename, 'data.csv')
        self.assertEqual(args.num_workers, 4)
        self.assertEqual(args.batch_size, 32)
        self.assertEqual(args.datapoint_loc_label, 'video_path')
        self.assertEqual(args.target_label, 'label')
        self.assertTrue(args.rand_augment)
        self.assertEqual(args.resize, 224)
        self.assertEqual(args.frames, 16)
        self.assertEqual(args.stride, 2)
        
        # Test model parameters
        args, _ = self.lp_parser.parser.parse_known_args([
            '--base_config', 'test.yaml',
            '--model_name', 'resnet50',
            '--aggregator_depth', '3',
            '--num_heads', '2',
            '--video_freeze_ratio', '0.5',
            '--dropout', '0.2',
            '--pretrained', 'True',
            '--video_encoder_checkpoint_path', '/path/to/encoder',
            '--video_encoder_lr', '0.0001'
        ])
        
        self.assertEqual(args.model_name, 'resnet50')
        self.assertEqual(args.aggregator_depth, 3)
        self.assertEqual(args.num_heads, 2)
        self.assertEqual(args.video_freeze_ratio, 0.5)
        self.assertEqual(args.dropout, 0.2)
        self.assertTrue(args.pretrained)
        self.assertEqual(args.video_encoder_checkpoint_path, '/path/to/encoder')
        self.assertEqual(args.video_encoder_lr, 0.0001)
        
        # Test linear probing parameters
        args, _ = self.lp_parser.parser.parse_known_args([
            '--base_config', 'test.yaml',
            '--head_structure', '{"head1": 128, "head2": 64}',
            '--loss_structure', '{"head1": "CE", "head2": "BCE"}',
            '--head_weights', '{"head1": 1.0, "head2": 0.5}',
            '--head_dropout', '{"head1": 0.2, "head2": 0.1}',
            '--head_task', '{"head1": "classification", "head2": "binary"}'
        ])
        
        self.assertEqual(args.head_structure, '{"head1": 128, "head2": 64}')
        self.assertEqual(args.loss_structure, '{"head1": "CE", "head2": "BCE"}')
        self.assertEqual(args.head_weights, '{"head1": 1.0, "head2": 0.5}')
        self.assertEqual(args.head_dropout, '{"head1": 0.2, "head2": 0.1}')
        self.assertEqual(args.head_task, '{"head1": "classification", "head2": "binary"}')
    
    @patch('utils.config.heartwise_config.HeartWiseConfig.update_config_with_args')
    def test_parse_args_and_update_known_args(self, mock_update_config):
        """Test parsing known arguments and updating config"""
        # Setup mock
        mock_config = MagicMock()
        mock_config.pipeline_project = 'DeepCORO_video_linear_probing'
        mock_update_config.return_value = mock_config
        
        # Mock sys.argv
        with patch('sys.argv', ['test.py', '--base_config', 'test.yaml']):
            # Mock argparse.ArgumentParser.parse_known_args
            with patch.object(argparse.ArgumentParser, 'parse_known_args') as mock_parse_known_args:
                mock_args = MagicMock()
                mock_args.base_config = 'test.yaml'
                mock_parse_known_args.return_value = (mock_args, [])
                
                # Call the method
                result = self.lp_parser.parse_args_and_update(mock_config)
                
                # Check the result
                self.assertEqual(result, mock_config)
                mock_update_config.assert_called_once()
    
    @patch('utils.config.heartwise_config.HeartWiseConfig.update_config_with_args')
    def test_parse_args_and_update_unknown_args(self, mock_update_config):
        """Test parsing unknown arguments with dot notation"""
        # Setup mock
        mock_config = MagicMock()
        mock_config.pipeline_project = 'DeepCORO_video_linear_probing'
        mock_config.head_lr = {'Value': 0.0, 'Other': 0.0}
        mock_update_config.return_value = mock_config
        
        # Mock sys.argv
        with patch('sys.argv', ['test.py', '--base_config', 'test.yaml', '--head_lr.Value=0.001']):
            # Mock argparse.ArgumentParser.parse_known_args
            with patch.object(argparse.ArgumentParser, 'parse_known_args') as mock_parse_known_args:
                mock_args = MagicMock()
                mock_args.base_config = 'test.yaml'
                mock_parse_known_args.return_value = (mock_args, ['--head_lr.Value=0.001'])
                
                # Call the method
                result = self.lp_parser.parse_args_and_update(mock_config)
                
                # Check the result
                self.assertEqual(result, mock_config)
                self.assertEqual(mock_config.head_lr['Value'], 0.001)
                mock_update_config.assert_called_once()
    
    @patch('utils.config.heartwise_config.HeartWiseConfig.update_config_with_args')
    def test_parse_args_and_update_unsupported_args(self, mock_update_config):
        """Test parsing unsupported unknown arguments"""
        # Setup mock
        mock_config = MagicMock()
        mock_config.pipeline_project = 'DeepCORO_video_linear_probing'
        mock_update_config.return_value = mock_config
        
        # Mock sys.argv
        with patch('sys.argv', ['test.py', '--base_config', 'test.yaml', '--unsupported.Value=0.001']):
            # Mock argparse.ArgumentParser.parse_known_args
            with patch.object(argparse.ArgumentParser, 'parse_known_args') as mock_parse_known_args:
                mock_args = MagicMock()
                mock_args.base_config = 'test.yaml'
                mock_parse_known_args.return_value = (mock_args, ['--unsupported.Value=0.001'])
                
                # Call the method
                with self.assertRaises(ValueError):
                    self.lp_parser.parse_args_and_update(mock_config)
    
    @patch('utils.config.heartwise_config.HeartWiseConfig.update_config_with_args')
    def test_parse_args_and_update_attribute_not_dict(self, mock_update_config):
        """Test parsing when attribute is not a dictionary"""
        # Setup mock
        mock_config = MagicMock()
        mock_config.pipeline_project = 'DeepCORO_video_linear_probing'
        mock_config.head_lr = "not_a_dict"  # Attribute exists but is not a dict
        mock_update_config.return_value = mock_config
        
        # Mock sys.argv
        with patch('sys.argv', ['test.py', '--base_config', 'test.yaml', '--head_lr.Value=0.001']):
            # Mock argparse.ArgumentParser.parse_known_args
            with patch.object(argparse.ArgumentParser, 'parse_known_args') as mock_parse_known_args:
                mock_args = MagicMock()
                mock_args.base_config = 'test.yaml'
                mock_parse_known_args.return_value = (mock_args, ['--head_lr.Value=0.001'])
                
                # Call the method
                with self.assertRaises(TypeError):
                    self.lp_parser.parse_args_and_update(mock_config)
    
    @patch('utils.config.heartwise_config.HeartWiseConfig.update_config_with_args')
    def test_parse_args_and_update_head_not_in_dict(self, mock_update_config):
        """Test parsing when head is not in dictionary"""
        # Setup mock
        mock_config = MagicMock()
        mock_config.pipeline_project = 'DeepCORO_video_linear_probing'
        mock_config.head_lr = {'OtherKey': 0.0}  # Dictionary missing the key
        mock_update_config.return_value = mock_config
        
        # Mock sys.argv
        with patch('sys.argv', ['test.py', '--base_config', 'test.yaml', '--head_lr.Value=0.001']):
            # Mock argparse.ArgumentParser.parse_known_args
            with patch.object(argparse.ArgumentParser, 'parse_known_args') as mock_parse_known_args:
                mock_args = MagicMock()
                mock_args.base_config = 'test.yaml'
                mock_parse_known_args.return_value = (mock_args, ['--head_lr.Value=0.001'])
                
                # Call the method
                with self.assertRaises(ValueError):
                    self.lp_parser.parse_args_and_update(mock_config)


class TestHeartWiseParser(unittest.TestCase):
    """Tests for HeartWiseParser class"""
    
    @patch('utils.config.heartwise_config.HeartWiseConfig.from_yaml')
    @patch('utils.registry.ParserRegistry.get')
    def test_get_pipeline_parser(self, mock_get_parser, mock_from_yaml):
        """Test getting the pipeline parser"""
        # Setup mocks
        mock_config = MagicMock()
        mock_config.pipeline_project = 'DeepCORO_clip'
        mock_from_yaml.return_value = mock_config
        
        mock_parser_cls = MagicMock()
        mock_parser_instance = MagicMock()
        mock_parser_cls.return_value = mock_parser_instance
        mock_get_parser.return_value = mock_parser_cls
        
        # Mock sys.argv for ArgumentParser
        with patch('sys.argv', ['test.py', '--base_config', 'test.yaml']):
            # Call the method
            parser_instance, config = HeartWiseParser._get_pipeline_parser()
            
            # Check the result
            self.assertEqual(parser_instance, mock_parser_instance)
            self.assertEqual(config, mock_config)
            mock_from_yaml.assert_called_once_with('test.yaml')
            mock_get_parser.assert_called_once_with('DeepCORO_clip')
    
    @patch('utils.config.heartwise_config.HeartWiseConfig.from_yaml')
    def test_get_pipeline_parser_missing_pipeline(self, mock_from_yaml):
        """Test error when pipeline_project is missing"""
        # Setup mock
        mock_config = MagicMock(spec=[])  # No attributes
        mock_from_yaml.return_value = mock_config
        
        # Mock sys.argv for ArgumentParser
        with patch('sys.argv', ['test.py', '--base_config', 'test.yaml']):
            # Call the method
            with self.assertRaises(ValueError) as context:
                HeartWiseParser._get_pipeline_parser()
            
            # Check the error message
            self.assertIn("pipeline_project", str(context.exception))
    
    @patch('utils.config.heartwise_config.HeartWiseConfig.from_yaml')
    @patch('utils.registry.ParserRegistry.list_registered')
    def test_get_pipeline_parser_unknown_pipeline(self, mock_list_registered, mock_from_yaml):
        """Test error when pipeline_project is not registered"""
        # Setup mocks
        mock_config = MagicMock()
        mock_config.pipeline_project = 'unknown_pipeline'
        mock_from_yaml.return_value = mock_config
        
        mock_list_registered.return_value = {'DeepCORO_clip': None}
        
        # Mock sys.argv for ArgumentParser
        with patch('sys.argv', ['test.py', '--base_config', 'test.yaml']):
            # Call the method
            with self.assertRaises(ValueError) as context:
                HeartWiseParser._get_pipeline_parser()
            
            # Check the error message
            self.assertIn("Unknown pipeline_project", str(context.exception))
    
    @patch('utils.parser.HeartWiseParser._get_pipeline_parser')
    @patch('utils.config.heartwise_config.HeartWiseConfig.set_gpu_info_in_place')
    def test_parse_config(self, mock_set_gpu_info, mock_get_pipeline_parser):
        """Test parse_config method"""
        # Setup mocks
        mock_config = MagicMock()
        mock_parser = MagicMock()
        mock_parser.parse_args_and_update.return_value = mock_config
        mock_get_pipeline_parser.return_value = (mock_parser, mock_config)
        
        # Call the method
        result = HeartWiseParser.parse_config()
        
        # Check the result
        self.assertEqual(result, mock_config)
        mock_parser.parse_args_and_update.assert_called_once_with(mock_config)
        mock_set_gpu_info.assert_called_once_with(mock_config)


class TestParserTyping(unittest.TestCase):
    """Tests for parser_typing.py helper functions"""
    
    def test_str2bool(self):
        """Test str2bool function"""
        # Test True values
        self.assertTrue(str2bool(True))
        self.assertTrue(str2bool('yes'))
        self.assertTrue(str2bool('true'))
        self.assertTrue(str2bool('t'))
        self.assertTrue(str2bool('y'))
        self.assertTrue(str2bool('1'))
        
        # Test False values
        self.assertFalse(str2bool(False))
        self.assertFalse(str2bool('no'))
        self.assertFalse(str2bool('false'))
        self.assertFalse(str2bool('f'))
        self.assertFalse(str2bool('n'))
        self.assertFalse(str2bool('0'))
        
        # Test invalid value
        with self.assertRaises(argparse.ArgumentTypeError):
            str2bool('invalid')
    
    def test_parse_list(self):
        """Test parse_list function"""
        # Test list input
        self.assertEqual(parse_list([1, 2, 3]), [1, 2, 3])
        
        # Test string input
        self.assertEqual(parse_list('1,2,3'), [1, 2, 3])
        self.assertEqual(parse_list('[1,2,3]'), [1, 2, 3])
        self.assertEqual(parse_list('  1, 2, 3  '), [1, 2, 3])
        
        # Test empty input
        self.assertEqual(parse_list(''), [])
        self.assertEqual(parse_list('[]'), [])
        
        # Test invalid input
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_list('1,a,3')
    
    def test_parse_optional_int(self):
        """Test parse_optional_int function"""
        # Test None values
        self.assertIsNone(parse_optional_int('none'))
        self.assertIsNone(parse_optional_int('None'))
        self.assertIsNone(parse_optional_int(''))
        
        # Test valid int values
        self.assertEqual(parse_optional_int('123'), 123)
        self.assertEqual(parse_optional_int('-456'), -456)
        
        # Test invalid input
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_optional_int('abc')
    
    def test_parse_optional_str(self):
        """Test parse_optional_str function"""
        # Test None values
        self.assertIsNone(parse_optional_str('none'))
        self.assertIsNone(parse_optional_str('None'))
        self.assertIsNone(parse_optional_str(''))
        
        # Test valid string values
        self.assertEqual(parse_optional_str('hello'), 'hello')
        self.assertEqual(parse_optional_str('123'), '123')
        self.assertEqual(parse_optional_str('true'), 'true')


if __name__ == '__main__':
    unittest.main() 