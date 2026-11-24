from typing import List
from torch import nn

"""
This file contains a basic block.
Used in pre-attention layers of ABMILEmbedder_MH.py
"""

def build_block(in_dim: int, out_dim: int, dropout: float, activation: nn.Module) -> List[nn.Module]:
    """
    Build the default block.

    Args:
        in_dim (int): The number of input features.
        out_dim (int): The number of output features.
        dropout (float): The dropout rate.
        activation (torch.nn.Module): The activation function.

    Returns:
        list: A list containing the layers of the block.
    """
    return [
            nn.Linear(in_features=in_dim, out_features=out_dim),
            nn.LayerNorm(out_dim),
            activation,
            nn.Dropout(dropout),
    ]
