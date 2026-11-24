from typing import Dict
import torch 
from torch import nn
from threadsmodel.components.ABMILEmbedder import ABMILEmbedder
from threadsmodel.components.ABMILEmbedder_MH import ABMILEmbedder_MH

"""
This file contains the Threads model architecture.
"""

class ThreadsModel(nn.Module):
    def __init__(self, config: Dict):
        '''
        Initializes the Threads model.
        
        Args:
            config (dict): The model configuration. Must contain the following keys:
                - wsi_encoder (str): The WSI encoder to use: "abmil_mh" (released THREADS weights) or "abmil".
                - hidden_dim (int): The hidden dimension of the WSI encoder.
                - embedding_dim (int): The input dimension of the WSI encoder.
                - dtype (str): Precision to use for model pretraining/finetuning.
                - num_wsi_embedder_hidden_layers (int): The number of hidden layers in the WSI encoder.
                - n_heads (int): The number of heads to use in the multi-headed attention mechanism.
                - activation (str): The activation function to use in the WSI encoder.
        '''
        super().__init__()
        self.config = config

        # Initialize WSI embedder
        self.wsi_embedder = nn.ModuleDict()
        wsi_encoder = self.config["wsi_encoder"]
        if wsi_encoder == "abmil":
            self._init_abmil_embedder()
        elif wsi_encoder == "abmil_mh":
            self._init_abmil_mh_embedder()
        else:
            raise NotImplementedError(f"WSI encoder '{wsi_encoder}' not implemented")

    def _init_abmil_embedder(self):
        """Initialize the ABMIL WSI embedder."""
        n_heads = self.config["n_heads"]
        if n_heads != 1:
            raise ValueError("ABMIL must have exactly 1 head. Use ABMIL_MH for multi-head attention.")

        pre_params = {
            "input_dim": self.config["embedding_dim"],
            "hidden_dim": self.config["hidden_dim"],
            "num_hidden_layers": self.config["num_wsi_embedder_hidden_layers"],
        }
        attention_params = {
            "model": "ABMIL",
            "params": {
                "input_dim": self.config["hidden_dim"],
                "hidden_dim": self.config["hidden_dim"],
                "dropout": True,
                "activation": self.config["activation"],
                "n_classes": 1,
            },
        }

        self.wsi_embedder['wsi'] = nn.ModuleDict({
            'embedder': ABMILEmbedder(
                pre_attention_params=pre_params,
                attention_params=attention_params,
            ),
        })

    def _init_abmil_mh_embedder(self):
        '''
        Initialize the multi-headed ABMIL WSI embedder.
        '''
        n_heads = self.config["n_heads"]
        if n_heads is None or n_heads <= 1:
            raise ValueError("ABMIL_MH must have more than 1 head")

        pre_params = {
            "input_dim": self.config["embedding_dim"],
            "hidden_dim": self.config["hidden_dim"],
            "num_hidden_layers": self.config["num_wsi_embedder_hidden_layers"],
        }
        attention_params = {
            "model": "ABMIL",
            "params": {
                "input_dim": self.config["hidden_dim"],
                "hidden_dim": self.config["hidden_dim"],
                "dropout": True,
                "activation": self.config["activation"],
                "n_heads": n_heads,
                "n_classes": 1,
            },
        }

        self.wsi_embedder['wsi'] = nn.ModuleDict({
            'embedder': ABMILEmbedder_MH(
                pre_attention_params=pre_params,
                attention_params=attention_params,
            ),
        })

    def forward(self, patch_features: torch.Tensor, return_raw_attention=False):
        """
        Processes histology data and returns embeddings.

            patch_features (torch.Tensor): The input patch features. Shape (batch_size, num_patches, patch_dim).
            return_raw_attention (bool): If the raw attention (before softmax) should be returned. Defaults to False.

        """
        # Pass through the embedder
        embedder = self.wsi_embedder['wsi']['embedder']
        if return_raw_attention:
            pooled_features, raw_attention = embedder(patch_features, return_raw_attention)
        else:
            pooled_features = embedder(patch_features)

        if return_raw_attention:
            return pooled_features, raw_attention

        return pooled_features
