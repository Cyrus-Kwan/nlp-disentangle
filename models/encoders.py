import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super().__init__()

        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.fc2 = nn.Linear(latent_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        h = F.gelu(self.fc2(z))
        return self.fc_out(h)  # Returns logits

    def forward(self, x, sample=True):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar) if sample else mu
        recon = self.decode(z)
        return recon, mu, logvar, z

class StyleEncoder(nn. Module):
    def __init__(self, cfg):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(cfg.bert_dim, cfg.style_hidden),
            nn.GELU(),
            nn.Linear(cfg.style_hidden, cfg.style_hidden),
            nn.GELU(),
            nn.Linear(cfg.style_hidden, cfg.style_hidden),
            nn.GELU(),
            nn.Linear(cfg.style_hidden, cfg.style_dim)
        )

    def forward(self, x):
        return self.mlp(x)


class ContentEncoder(nn. Module):
    def __init__(self, cfg):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(cfg.bert_dim, 32),
            nn.GELU(),
            nn.Linear(32, cfg.content_hidden),
            nn.GELU(),
            nn.Linear(cfg.content_hidden, cfg.content_hidden),
            nn.GELU(),
            nn.Linear(cfg.content_hidden, cfg.content_dim)
        )

    def forward(self, x):
        return self.mlp(x)


class ContextEncoder(nn. Module):
    def __init__(self, cfg):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(cfg.bert_dim, 32),
            nn.GELU(),
            nn.Linear(32, cfg.content_hidden),
            nn.GELU(),
            nn.Linear(cfg.content_hidden, cfg.content_hidden),
            nn.GELU(),
            nn.Linear(cfg.content_hidden, cfg.context_dim)
        )

    def forward(self, x):
        return self.mlp(x)