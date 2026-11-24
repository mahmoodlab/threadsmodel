from typing import Dict
import torch
from torch import nn
from einops import rearrange
from threadsmodel.components.Block import build_block
from threadsmodel.components.BatchedABMIL import BatchedABMIL

"""
This file contains a multiheaded-attention ABMIL module.
Used in ThreadsModel.py
"""

class ABMILEmbedder_MH(nn.Module):
    def __init__(self, pre_attention_params: Dict = None, attention_params: Dict = None):
        """
        Initializes ABMILEmbedder_MH.

        Args:
            pre_attention_params (Optional[Dict]]): 
                Configuration for the pre-attention module, including:
                - 'input_dim' (int): Input feature dimension.
                - 'hidden_dim' (int): Hidden layer dimension.
                - 'num_hidden_layers' (int): Number of hidden layers.

            attention_params (Optional[Dict]): 
                Configuration for the attention module, including:
                - 'model' (str): Name of the attention model (e.g., "ABMIL").
                - 'params' (Dict): Parameters for the attention model.

            aggregation (str): 
                Type of aggregation after applying attention. 
                Supported options: 'regular' (default).
        """
        super().__init__()

        # Pre-attention layer
        self.n_heads = attention_params["params"]["n_heads"]
        self.pre_attention_params = pre_attention_params
        if pre_attention_params is not None:
            self._build_pre_attention_layers(params=pre_attention_params)

        # Attention layer
        self.attention_params = attention_params
        if attention_params is not None:
            self._build_attention_layers(attention_params)

        # Projection head
        self.proj_multihead = nn.Linear(
            in_features=pre_attention_params["hidden_dim"] * self.n_heads,
            out_features=pre_attention_params["hidden_dim"],
        )

    def _build_pre_attention_layers(self, params):
        '''
        Build pre-attention network.
        
        Args:
            params (dict): Parameters for the pre-attention network. Must contain the following keys:
                - input_dim (int): The input dimension of the network.
                - hidden_dim (int): The hidden dimension of the network.
                - num_hidden_layers (int): The number of hidden layers in the network.
        '''
        layers = []
        
        # Initial layer
        layers.extend(build_block(in_dim=params["input_dim"], out_dim=params["hidden_dim"], dropout=0.1, activation=nn.GELU()))
        
        # Hidden layers 
        for _ in range(params["num_hidden_layers"]):
            layers.extend(build_block(in_dim=params["hidden_dim"], out_dim=params["hidden_dim"], dropout=0.1, activation=nn.GELU()))
            
        # Output layer
        layers.extend(build_block(in_dim=params["hidden_dim"], out_dim=params["hidden_dim"] * self.n_heads, dropout=0.1, activation=nn.GELU()))
        
        self.pre_attn = nn.Sequential(*layers)

    def _build_attention_layers(self, params=None):
        """
        Build attention network.
        
        Args:
            attn_model (str): The attention model to use. Currently only "ABMIL" is supported.
        """
        if params['model'] == "ABMIL":
            self.attn = nn.ModuleList([BatchedABMIL(**params['params']) for _ in range(self.n_heads)])
        else:
            raise NotImplementedError(f"Model type {params['model']} not implemented.")

    def forward(self, x, return_raw_attention=False):
        """
        Foward pass.

        Args:
            x (torch.Tensor): Batched representation of the input. Shape (batch_size, num_patches, patch_dim).
        Returns:
                torch.Tensor or tuple:
                    - If return_raw_attention is False, returns a tensor of shape (batch_size, hidden_dim).
                    - If return_raw_attention is True, returns a tuple (output, attention_weights), where
                    output has shape (batch_size, hidden_dim) and attention_weights is a tensor
                    containing the model's raw attention scores.
        """

        # Pre_attn common to all heads
        if self.pre_attention_params is not None:
            embeddings = self.pre_attn(x)
        else:
            embeddings = x

        # Split output into heads
        embeddings = rearrange(embeddings, "b n (d h) -> b n d h", h=self.n_heads)

        # Get attention for each head
        attention = []
        raw_attention = []
        for i, attn_net in enumerate(self.attn):
            c_attention, c_raw_attention = attn_net(embeddings[:, :, :, i], return_raw_attention=True)
            attention.append(c_attention)
            raw_attention.append(c_raw_attention)
        
        # Return post softmax attention
        attention = torch.stack(attention, dim=-1)
        raw_attention = torch.stack(raw_attention, dim=-1)

        # Compute weighted sum of embeddings
        embeddings = embeddings * attention
        slide_embeddings = torch.sum(embeddings, dim=1)

        # Project to final dimension
        slide_embeddings = rearrange(slide_embeddings, "b d h -> b (d h)")
        slide_embeddings = self.proj_multihead(slide_embeddings)

        if return_raw_attention:
            return slide_embeddings, raw_attention

        return slide_embeddings
