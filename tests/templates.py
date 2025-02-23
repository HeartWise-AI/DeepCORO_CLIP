import unittest
import torch


class ModelTestsMixin:
    """Mixin class providing common model tests."""
    
    @torch.no_grad()
    def test_shape_consistency(self):
        """Test if model outputs maintain expected shapes."""
        if isinstance(self.test_inputs, tuple):
            outputs = self.model(*self.test_inputs)
        else:
            outputs = self.model(self.test_inputs)
        self.assertEqual(self.expected_output_shape, outputs.shape)

    @torch.no_grad()
    @unittest.skipUnless(torch.cuda.is_available(), 'No GPU was detected')
    def test_device_moving(self):
        """Test if model can be moved between devices while maintaining output consistency."""
        model_on_gpu = self.model.to('cuda:0')
        model_back_on_cpu = model_on_gpu.cpu()

        # Handle tuple inputs for text encoder
        if isinstance(self.test_inputs, tuple):
            inputs_gpu = tuple(t.to('cuda:0') for t in self.test_inputs)
            torch.manual_seed(42)
            outputs_cpu = self.model(*self.test_inputs)
            torch.manual_seed(42)
            outputs_gpu = model_on_gpu(*inputs_gpu)
            torch.manual_seed(42)
            outputs_back_on_cpu = model_back_on_cpu(*self.test_inputs)
        else:
            torch.manual_seed(42)
            outputs_cpu = self.model(self.test_inputs)
            torch.manual_seed(42)
            outputs_gpu = model_on_gpu(self.test_inputs.to('cuda:0'))
            torch.manual_seed(42)
            outputs_back_on_cpu = model_back_on_cpu(self.test_inputs)

        self.assertAlmostEqual(0., torch.sum(outputs_cpu - outputs_gpu.cpu()).item())
        self.assertAlmostEqual(0., torch.sum(outputs_cpu - outputs_back_on_cpu).item())

    def test_batch_independence(self):
        """Test if samples in a batch are processed independently."""
        if isinstance(self.test_inputs, tuple):
            input_ids, attention_mask = self.test_inputs
            input_ids = input_ids.clone()
            attention_mask = attention_mask.clone()
            input_ids.requires_grad = True
            
            # Compute forward pass in eval mode
            self.model.eval()
            outputs = self.model(input_ids, attention_mask)
            self.model.train()
            
            batch_size = input_ids.shape[0]
        else:
            inputs = self.test_inputs.clone()
            inputs.requires_grad = True
            
            # Compute forward pass in eval mode
            self.model.eval()
            outputs = self.model(inputs)
            self.model.train()
            
            batch_size = inputs.shape[0]

        # Mask loss for certain samples in batch
        mask_idx = torch.randint(0, batch_size, ())
        mask = torch.ones_like(outputs)
        mask[mask_idx] = 0
        outputs = outputs * mask

        # Compute backward pass
        loss = outputs.mean()
        loss.backward()

        # Check gradients
        if isinstance(self.test_inputs, tuple):
            grad = input_ids.grad
        else:
            grad = inputs.grad

        # Check if gradient exists and is zero for masked samples
        for i in range(batch_size):
            if i == mask_idx:
                self.assertTrue(torch.all(grad[i] == 0).item())
            else:
                self.assertFalse(torch.all(grad[i] == 0).item())


class DatasetTestsMixin:
    """Mixin class providing common dataset tests."""
    
    def test_dataset_length(self):
        """Test if dataset length matches expected size."""
        self.assertEqual(self.expected_size, len(self.dataset))

    def test_item_shapes(self):
        """Test if dataset items have correct shapes."""
        item = self.dataset[0]
        for key, expected_shape in self.expected_shapes.items():
            self.assertEqual(expected_shape, item[key].shape)

    def test_batch_collation(self):
        """Test if items can be properly collated into batches."""
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0
        )
        batch = next(iter(dataloader))
        
        for key, expected_shape in self.expected_batch_shapes.items():
            self.assertEqual(expected_shape, batch[key].shape)

    def test_multi_worker_loading(self):
        """Test if dataset works with multiple workers."""
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=4,
            shuffle=True,
            num_workers=2
        )
        
        # Try loading a few batches
        for i, batch in enumerate(dataloader):
            if i >= 3:  # Check first 3 batches
                break
            
            for key, expected_shape in self.expected_batch_shapes.items():
                self.assertEqual(expected_shape, batch[key].shape) 