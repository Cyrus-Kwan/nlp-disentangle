import torch
import torch.nn as nn

from models.encoders import StyleEncoder, ContentEncoder, ContextEncoder
from models.generator import SBERTGenerator
from models.discriminator import SBERTDiscriminator

class SBERTDisentangler(nn. Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.style_encoder      = StyleEncoder(cfg)
        self.content_encoder    = ContentEncoder(cfg)
        self.context_encoder    = ContextEncoder(cfg)

        self.generator      = SBERTGenerator(cfg)
        self.discriminator  = SBERTDiscriminator(cfg)

        self.style_classifier   = nn.Linear(cfg.style_dim, 1)
        self.content_regressor  = nn.Linear(cfg.content_dim, cfg.content_dim)
        self.context_regressor  = nn.Linear(cfg.context_dim, cfg.question_dim)

    def generator_parameters(self):
        for module in (
            self.style_encoder,
            self.content_encoder, 
            self.context_encoder,
            self.generator,
            self.style_classifier,
            self.content_regressor,
            self.context_regressor
        ):
            yield from module.parameters()

    def discriminator_parameters(self):
        return self.discriminator.parameters()
    
    def encode(self, x):
        style_embed     = self.style_encoder(x)
        content_embed   = self.content_encoder(x)
        context_embed   = self.context_encoder(x)

        return style_embed, content_embed, context_embed

    def reconstruct(self, x):
        style_embed, content_embed, context_embed = self.encode(x)
        bert_recon = self.generator(style_embed, content_embed, context_embed)
        return bert_recon, style_embed, content_embed, context_embed

    def generate(self, style_embed, content_embed, context_embed):
        return self.generator(style_embed, content_embed, context_embed)

    def classify(self, x):
        return self.style_classifier(self.style_encoder(x))