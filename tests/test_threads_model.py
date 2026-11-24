import unittest
import torch

def add_parent_to_sys_path():
    import sys
    from pathlib import Path
    current_dir = Path(__file__).resolve()
    parent_dir = current_dir.parent.parent
    sys.path.insert(0, str(parent_dir))

add_parent_to_sys_path()
from threadsmodel.inference import create_model, create_model_from_pretrained


class TestModelLoadingAndInference(unittest.TestCase):
    
    def setUp(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.patch_features = torch.randn(1, 500, 768)  # (batch_size, num_patches, feature_dim)
    
    def test_model_loading_with_pretrained(self):
        # Test loading with pretrained weights
        model, precision, _ = create_model_from_pretrained(
            checkpoint_path="hf_hub:MahmoodLab/threads", 
            cache_dir="./model_assets",
        )
        self.assertIsNotNone(model, "Model with pretrained weights failed to load")
        self.precision = precision  

    def test_model_loading_without_pretrained(self):
        # Test loading without pretrained weights
        model_no_weights, precision, _ = create_model(
            checkpoint_path="hf_hub:MahmoodLab/threads",  
            cache_dir="./model_assets",
        )
        self.assertIsNotNone(model_no_weights, "Model without pretrained weights failed to load")
        self.precision = precision  
    
    def run_inference_and_check_output(self, model):
        # Move model to device and check inference
        model.to(self.device)
        with torch.inference_mode(), torch.amp.autocast(
            self.device, dtype=self.precision, enabled=self.precision in [torch.bfloat16, torch.float16]
        ):
            pooled_features = model(self.patch_features.to(self.device))
        
        # Assertions
        self.assertIsNotNone(pooled_features, "Model did not produce output")
        self.assertEqual(
            pooled_features.shape[0], self.patch_features.shape[0],
            "Batch size of output does not match input"
        )
        print(f'Success: Model produced output features of shape {pooled_features.shape}')

    def test_inference_with_pretrained(self):
        # Test inference for model loaded with pretrained weights
        model, precision, _ = create_model_from_pretrained(
            checkpoint_path="hf_hub:MahmoodLab/threads",
            cache_dir="./model_assets",
        )
        self.precision = precision
        self.run_inference_and_check_output(model)

    def test_inference_without_pretrained(self):
        # Test inference for model loaded without pretrained weights
        model_no_weights, precision, _ = create_model(
            checkpoint_path="hf_hub:MahmoodLab/threads",
            cache_dir="./model_assets",
        )
        self.precision = precision
        self.run_inference_and_check_output(model_no_weights)

if __name__ == '__main__':
    unittest.main()
