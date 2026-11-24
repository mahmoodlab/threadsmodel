from typing import Union, Tuple
import torch
from torch import nn
import torch.nn.functional as F

"""
This file contains an ABMIL network that supports batched inputs.
Used in attention layers of ABMILEmbedder_MH.py
"""

class BatchedABMIL(nn.Module):

    def __init__(
            self,
            input_dim: int = 1024,
            hidden_dim: int = 256,
            dropout: bool = False,
            n_classes: int = 1,
            activation: str = 'softmax',
            **kwargs
        ):
        """
        Attention Network with Sigmoid Gating (3 fc layers). Supports batching.
        args:
            input_dim (int): input feature dimension
            hidden_dim (int): hidden layer dimension
            dropout (bool): whether to use dropout (p = 0.25)
            n_classes (int): number of classes
            activation (str): activation function
        """
        super().__init__()

        self.activation = activation
        self.attention_a = [
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh()]

        self.attention_b = [nn.Linear(input_dim, hidden_dim),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(hidden_dim, n_classes)

    def forward(
        self, 
        x: torch.Tensor, 
        return_raw_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_patches, patch_dim).
            return_raw_attention (bool, optional): If True, returns both activated attention 
                and raw attention scores. Defaults to False.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                - If `return_raw_attention` is False, returns the activated attention tensor 
                of shape (batch_size, n_classes).
                - If `return_raw_attention` is True, returns a tuple:
                (activated_attention, raw_attention), where:
                - `activated_attention`: Activated attention tensor.
                - `raw_attention`: Raw attention scores before activation.
        """
        # gated attention
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes

        if self.activation == 'softmax':
            activated_A = F.softmax(A, dim=1)
        elif self.activation == 'leaky_relu':  # enable "counting"
            activated_A = F.leaky_relu(A)
        elif self.activation == 'relu':
            activated_A = F.relu(A)
        elif self.activation == 'sigmoid':  # enable "counting"
            activated_A = torch.sigmoid(A)
        else:
            raise NotImplementedError('Activation not implemented.')

        if return_raw_attention:
            return activated_A, A

        return activated_A
