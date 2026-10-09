import torch
import torch.nn as nn

class SBERTDiscriminator(nn. Module):
    def __init__(self, cfg):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(cfg.bert_dim, cfg.disc_dim),
            nn.PReLU(cfg.disc_dim),
            nn.Linear(cfg.disc_dim, cfg.disc_dim)
        )

        self.classifier = nn.Linear(cfg.disc_dim, 1)

    def forward(self, x):
        embed = self.mlp(x)
        return self.classifier(embed)