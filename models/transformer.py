import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.attention import SelfAttention, FeedForward
from transformers import get_inverse_sqrt_schedule
import lightning as L


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x:torch.Tensor):
        # x: [B, S, D]
        S = x.size(1)
        x = x + self.pe[:, :S, :]
        return x


class DecoderLayer(nn.Module):
    def __init__(
            self, *,
            d_model:int, 
            num_heads:int, 
            dropout:float=0.1,
            ff_factor:int=4
        ):
        super(DecoderLayer, self).__init__()

        self.self_attention = SelfAttention(
            d_model=d_model, 
            num_heads=num_heads, 
            dropout=dropout
        )

        self.feedforward = FeedForward(
            d_model=d_model, 
            ff_factor=ff_factor
        )

    def forward(self, inputs:torch.Tensor, mask:torch.Tensor=None):
        x = self.self_attention(inputs, mask=mask)
        x = self.feedforward(x)
        return x
    

class Decoder(nn.Module):
    def __init__(
        self, *,
        d_model:int,
        num_layers:int,
        num_heads:int,
        vocab_size:int,
        padding_idx:int=0,
        ff_factor:int=4,
        dropout:float=0.1,
        max_len:int=5000
    ):
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model

        self.embedding = nn.Embedding(
            vocab_size, 
            embedding_dim=d_model, 
            padding_idx=padding_idx
        )
        self.pos_encoding = PositionalEncoding(d_model=d_model, max_len=max_len)
        self.embedding_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            DecoderLayer(
                d_model=d_model, 
                num_heads=num_heads, 
                dropout=dropout, 
                ff_factor=ff_factor
            ) for _ in range(num_layers)
        ])

        self.classifier = nn.Linear(d_model, vocab_size)

    def forward(
            self, 
            inputs:torch.Tensor, 
            mask:torch.Tensor=None
        ):
        device = inputs.device
        x = self.embedding(inputs)
        x = x * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32, device=device))
        x = self.pos_encoding(x)
        x = self.embedding_dropout(x)

        for i in range(self.num_layers):
            x = self.layers[i](x, mask)
        
        x = self.classifier(x)
        return x
    

class DecoderLightning(L.LightningModule):
    def __init__(
            self, 
            model:nn.Module,
            label_smoothing:float=0.1,
            learning_rate:float=1e-3,
            warmup_steps:int=4000,
            betas:list=[0.99, 0.99],
            eps:float=1e-8,
        ):
        super(DecoderLightning, self).__init__()
        self.save_hyperparameters(ignore=['model'])

        self.model = model
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.betas = betas
        self.eps = eps
        self.label_smoothing = label_smoothing

    def forward(
            self, 
            inputs:torch.Tensor, 
            mask:torch.Tensor=None
        ):
        x = self.model(inputs, mask)
        return x
    
    def training_step(
            self, 
            batch:torch.Tensor, 
            batch_idx:int
        ):
        inputs = batch['inputs']
        labels = batch['labels']

        x = self.forward(inputs, mask='causal')
        x = torch.reshape(x, (-1, x.shape[-1]))
        labels = torch.reshape(labels, (-1,))
        loss = F.cross_entropy(x, labels, label_smoothing=self.label_smoothing)

        self.log('train/loss', loss, on_step=True, on_epoch=True)
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('train/lr', lr, on_step=True, on_epoch=False)
        return loss
    
    def validation_step(
            self, 
            batch:torch.Tensor, 
            batch_idx:int):
        
        inputs = batch['inputs']
        labels = batch['labels']

        x = self.forward(inputs, mask='causal')
        x = torch.reshape(x, (-1, x.shape[-1]))
        labels = torch.reshape(labels, (-1,))
        loss = F.cross_entropy(x, labels, label_smoothing=self.label_smoothing)

        self.log('validation/loss', loss, on_step=True, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.learning_rate, 
            betas=[self.betas[0], self.betas[1]], 
            eps=self.eps
        )

        scheduler = get_inverse_sqrt_schedule(optimizer, num_warmup_steps=self.warmup_steps)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            }
        }
                