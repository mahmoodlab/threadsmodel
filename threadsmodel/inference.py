import torch
import json
from typing import Optional, Tuple

from threadsmodel.ThreadsModel import ThreadsModel


def create_model(
    checkpoint_path: Optional[str] = None,
    cfg_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    hf_auth_token: Optional[str] = None
) -> Tuple[torch.nn.Module, torch.dtype, int]:
    """
    Creates a ThreadsModel, defined with configuration detailes in config.
    
    Args:
        checkpoint_path (str, optional): Path to the checkpoint, supports Hugging Face Hub paths prefixed with 'hf_hub:'.
        cfg_path (str, optional): Path to the configuration file if not loading from Hugging Face Hub. Override if providing hf_hub:
        cache_dir (str, optional): Directory to cache files when downloading from Hugging Face Hub.
        hf_auth_token (str, optional): Authentication token for Hugging Face Hub.

    Returns:
        Tuple[torch.nn.Module, torch.dtype, int]: The model, its precision, and embedding dimension.
    """
    import os

    # get config and set checkpoint path 
    if checkpoint_path and checkpoint_path.startswith("hf_hub:"):
        from huggingface_hub import hf_hub_download

        # download config.json
        cfg_path = hf_hub_download(
            checkpoint_path[len("hf_hub:"):], 
            cache_dir=cache_dir,
            filename="config.json",
            token=hf_auth_token
        )

    with open(cfg_path) as f:
        config = json.load(f)

    # Initialize model 
    model = ThreadsModel(config=config)
    model.eval()
    # Get precision from config
    if config['precision'] in ['float32', 'fp32']:
        precision = torch.float32
    elif config['precision'] in ['float16', 'fp16']:
        precision = torch.float16
    elif config['precision'] in ['bfloat16', 'bf16']:
        precision = torch.bfloat16
    else:
        raise ValueError(f"Unsupported precision {config['precision']}")
    embedding_dim = 1024

    return model, precision, embedding_dim
    
def create_model_from_pretrained(
    checkpoint_path: Optional[str] = None,
    cfg_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    hf_auth_token: Optional[str] = None
) -> Tuple[torch.nn.Module, torch.dtype, int]:
    """
    Creates and loads pretrained ThreadsModel from checkpoint.

    Args:
        checkpoint_path (str, optional): Path to the checkpoint, supports Hugging Face Hub paths prefixed with 'hf_hub:'.
        cfg_path (str, optional): Path to the configuration file if not loading from Hugging Face Hub.
        cache_dir (str, optional): Directory to cache files when downloading from Hugging Face Hub.
        hf_auth_token (str, optional): Authentication token for Hugging Face Hub.

    Returns:
        Tuple[torch.nn.Module, torch.dtype, int]: The model, its precision, and embedding dimension.
    """

    # Create model and retrieve precision and embedding dimension
    model, precision, embedding_dim = create_model(
        checkpoint_path=checkpoint_path,
        cfg_path=cfg_path,
        cache_dir=cache_dir,
        hf_auth_token=hf_auth_token
    )

    # Load checkpoint if provided
    if checkpoint_path and checkpoint_path.startswith("hf_hub:"):
        from huggingface_hub import hf_hub_download

        # Download checkpoint from Hugging Face Hub
        checkpoint_path = hf_hub_download(
            checkpoint_path[len("hf_hub:"):], 
            cache_dir=cache_dir,
            filename="threads_model.pt",
            token=hf_auth_token
        )
    
    # Load the state dictionary, removing specific prefixes and entries
    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    state_dict = {k: v for k, v in state_dict.items() if 'molecular_encoders' not in k}

    # Load the model state
    model.load_state_dict(state_dict, strict=True)
    print(f"\033[92mLoaded model weights from {checkpoint_path}\033[0m")

    return model, precision, embedding_dim


if __name__ == "__main__":

    # Test loading with pretrained weights
    model, precision, _ = create_model_from_pretrained(
        checkpoint_path="hf_hub:MahmoodLab/threads",  # or provide a local path
        cache_dir="./model_assets",
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    patch_features = torch.randn(1, 500, 768) # (batch_size, num_patches, feature_dim)
    with torch.inference_mode(), torch.amp.autocast(device, dtype = precision, enabled = precision in [torch.bfloat16, torch.float16]):
        pooled_features = model(patch_features.to(device))
        print(f'Success: Model produced output features of shape {pooled_features.shape}')
