import torch
import torch.nn as nn

class SBERTGenerator(nn. Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        
        self.mlp = nn.Sequential(
            nn.Linear(
                cfg.style_dim+cfg.content_dim+cfg.residual_dim,
                cfg.gene_dim
            ),
            nn.GELU(),
            nn.Linear(cfg.gene_dim, cfg.bert_dim)
        )

    def forward(self, style_embed, content_embed, residual_embed):
        x = torch.cat(
            (style_embed, content_embed, residual_embed),
            dim=-1
        )
        return self.mlp(x)