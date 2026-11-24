from typing import Dict
import torch
from torch import nn
from threadsmodel.components.Block import build_block
from threadsmodel.components.BatchedABMIL import BatchedABMIL


class ABMILEmbedder(nn.Module):

    def __init__(
        self,
        pre_attention_params: Dict = None,
        attention_params: Dict = None,
        aggregation: str = 'regular',
    ) -> None:
        """
        Initializes ABMILEmbedder.

        Args:
            pre_attention_params (Optional[Dict[str, Union[int, float]]]): 
                Configuration for the pre-attention module, including:
                - 'input_dim' (int): Input feature dimension.
                - 'hidden_dim' (int): Hidden layer dimension.
                - 'num_hidden_layers' (int): Number of hidden layers.

            attention_params (Optional[Dict[str, Union[str, Dict]]]): 
                Configuration for the attention module, including:
                - 'model' (str): Name of the attention model (e.g., "ABMIL").
                - 'params' (Dict): Parameters for the attention model.

            aggregation (str): 
                Type of aggregation after applying attention. 
                Supported options: 'regular' (default).
        """
        super(ABMILEmbedder, self).__init__()

        # 1- build pre-attention params 
        self.pre_attention_params = pre_attention_params
        if pre_attention_params is not None:
            self._build_pre_attention_params(params=pre_attention_params)

        # 2- build attention params
        self.attention_params = attention_params
        if attention_params is not None:
            self._build_attention_params(
                attn_model=attention_params['model'],
                params=attention_params['params']
            )

        # 3- set aggregation type 
        self.agg_type = aggregation  # Option are: mean, regular, additive, mean_additive

    def _build_pre_attention_params(self, params):
        """
        Build pre-attention params 
        """
        
        # build basic block
        layers = []
        
        # initial layer 
        layers.extend(build_block(in_dim=params['input_dim'], out_dim=params['hidden_dim'], dropout=0.1, activation=nn.GELU()))
                
        # add hidden layers
        for _ in range(params["num_hidden_layers"]):
            layers.extend(build_block(in_dim=params['hidden_dim'], out_dim=params['hidden_dim'], dropout=0.1, activation=nn.GELU()))
        
        # change to sequential 
        self.pre_attn = nn.Sequential(*layers)
        
    def _build_attention_params(self, attn_model: str = 'ABMIL', params: Dict = None):
        """
        Build attention params 
        """
        if attn_model == 'ABMIL':
            self.attn = BatchedABMIL(**params)
        else:
            raise NotImplementedError('Attention model not implemented -- Options are ABMIL, PatchGCN and TransMIL.')
        

    def forward(
        self,
        bags: torch.Tensor,
        return_attention: bool = False, 
    ) -> torch.tensor:
        """
        Foward pass.

        Args:
            bags (torch.Tensor): batched representation of the tokens 
            return_attention (bool): if attention weights should be returned (raw attention)
        Returns:
            torch.tensor: Model output.
        """

        # pre-attention
        if self.pre_attention_params is not None:
            embeddings = self.pre_attn(bags)
        else:
            embeddings = bags

        # compute attention weights  
        if self.attention_params is not None:
            if return_attention:
                attention, raw_attention = self.attn(embeddings, return_raw_attention=True)
            else:
                attention = self.attn(embeddings)  # return post softmax attention

        if self.agg_type == 'regular':
            embeddings = embeddings * attention
            if self.attention_params["params"]["activation"] == "sigmoid":
                slide_embeddings = torch.mean(embeddings, dim=1)
            else:
                slide_embeddings = torch.sum(embeddings, dim=1)

        else:
            raise NotImplementedError('Agg type not supported. Options are "additive" or "regular".')

        if return_attention:
            return slide_embeddings, raw_attention
        
        return slide_embeddings