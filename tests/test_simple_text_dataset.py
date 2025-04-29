import unittest
import torch
from torch.utils.data import DataLoader
from dataloaders.simple_text_dataset import SimpleTextDataset
from tests.templates import DatasetTestsMixin
from transformers import AutoTokenizer


class TestSimpleTextDataset(DatasetTestsMixin, unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test method."""
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
        )
        
        # Create test data
        self.test_texts = [
            "This is a test report for cardiology.",
            "Patient shows normal cardiac function.",
            "Evidence of mild cardiomegaly on chest x-ray."
        ]
        
        # Create dataset
        self.dataset = SimpleTextDataset(
            texts=self.test_texts,
            tokenizer=self.tokenizer
        )
        
        # Expected values for tests
        self.expected_size = len(self.test_texts)
        self.expected_shapes = {
            "input_ids": torch.Size([512]),
            "attention_mask": torch.Size([512])
        }
        self.expected_batch_shapes = {
            "input_ids": torch.Size([4, 512]),
            "attention_mask": torch.Size([4, 512])
        }
        
    def test_getitem(self):
        """Test if __getitem__ returns correctly formatted data."""
        for i in range(len(self.dataset)):
            item = self.dataset[i]
            
            # Check if the item has the expected keys
            self.assertIn("input_ids", item)
            self.assertIn("attention_mask", item)
            
            # Check if tensor types are correct
            self.assertIsInstance(item["input_ids"], torch.Tensor)
            self.assertIsInstance(item["attention_mask"], torch.Tensor)
            
            # Check that attention_mask aligns with input_ids
            # (non-zero attention mask where input_ids are non-zero)
            pad_token_id = self.tokenizer.pad_token_id
            non_pad_tokens = (item["input_ids"] != pad_token_id)
            self.assertTrue(torch.all(item["attention_mask"][non_pad_tokens] == 1))
            
    def test_tokenization(self):
        """Test if tokenization works as expected."""
        # Check first sample manually
        expected_encoded = self.tokenizer(
            self.test_texts[0],
            padding="max_length", 
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        expected_encoded = {k: v.squeeze(0) for k, v in expected_encoded.items()}
        
        actual_encoded = self.dataset[0]
        
        # Compare input_ids
        self.assertTrue(torch.all(expected_encoded["input_ids"] == actual_encoded["input_ids"]))
        # Compare attention_mask
        self.assertTrue(torch.all(expected_encoded["attention_mask"] == actual_encoded["attention_mask"]))
        
    def test_batch_collation(self):
        """Test if items can be properly collated into batches."""
        # Override parent test to use a batch size matching our data size
        dataloader = DataLoader(
            self.dataset,
            batch_size=len(self.test_texts),  # Use all samples
            shuffle=False,
            num_workers=0
        )
        
        batch = next(iter(dataloader))
        
        # Check batch shapes
        self.assertEqual(batch["input_ids"].shape, torch.Size([len(self.test_texts), 512]))
        self.assertEqual(batch["attention_mask"].shape, torch.Size([len(self.test_texts), 512]))
        
    def test_multi_worker_loading(self):
        """Test if dataset works with multiple workers."""
        # Override the parent test to handle partial batches
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=2,  # Use smaller batch size to avoid partial batch issues
            shuffle=True,
            num_workers=2,
            drop_last=False  # Don't drop the last batch even if it's smaller
        )
        
        # Try loading a few batches
        for i, batch in enumerate(dataloader):
            if i >= 3:  # Check first 3 batches
                break
            
            # Check shapes - we need to handle different batch sizes
            actual_batch_size = batch["input_ids"].size(0)
            self.assertLessEqual(actual_batch_size, 2)  # Should be at most our batch size
            self.assertEqual(batch["input_ids"].shape, torch.Size([actual_batch_size, 512]))
            self.assertEqual(batch["attention_mask"].shape, torch.Size([actual_batch_size, 512]))


if __name__ == '__main__':
    unittest.main() 