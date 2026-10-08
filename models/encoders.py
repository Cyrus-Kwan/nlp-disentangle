import torch
import torch.nn as nn

class StyleEncoder(nn. Module):
    def __init__(self, cfg):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(cfg.bert_dim, cfg.style_dim),
            nn.GELU(),
            nn.Linear(cfg.style_dim, cfg.style_dim)
        )

    def forward(self, x):
        return self.mlp(x)


class ContentEncoder(nn. Module):
    def __init__(self, cfg):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(cfg.bert_dim, cfg.content_dim),
            nn.GELU(),
            nn.Linear(cfg.content_dim, cfg.content_dim)
        )

    def forward(self, x):
        return self.mlp(x)


class ResidualEncoder(nn. Module):
    def __init__(self, cfg):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(cfg.bert_dim, cfg.residual_dim),
            nn.GELU(),
            nn.Linear(cfg.residual_dim, cfg.residual_dim)
        )

    def forward(self, x):
        return self.mlp(x)