import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(
            self, *,
            d_model:int, 
            num_heads:int, 
            dropout:float=0.1
        ):
        super(SelfAttention, self).__init__()

        self.mha = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs: torch.Tensor, mask:torch.Tensor=None):
        # inputs has the shape [B, S, D]
        # Given mask has values 0/1, 1 meaning it should be attended to and zero means mask
        # In the line below we translate it back to what torch accepts, in which, true means mask it
        device = inputs.device
        if mask is not None and mask == 'causal':
            seq_len = inputs.shape[1]
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
            mask = mask.to(torch.bool)
        elif mask is not None:
            mask = (1 - mask).to(torch.bool)
        x, _ = self.mha(query=inputs, key=inputs, value=inputs, attn_mask=mask)
        x = self.layer_norm(x + inputs)
        return x
    

class FeedForward(nn.Module):
    def __init__(
            self, *, 
            d_model:int, 
            ff_factor:int=4, 
            dropout:float=0.1
        ):
        super(FeedForward, self).__init__()

        self.expand = nn.Linear(d_model, d_model * ff_factor)
        self.collapse = nn.Linear(d_model * ff_factor, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)


    def forward(self, inputs: torch.Tensor):
        x = self.expand(inputs)
        x = F.gelu(x)
        x = self.collapse(x)
        x = self.layer_norm(x + inputs)
        x = self.dropout(x)
        return x