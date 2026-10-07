import torch
import torch.nn as nn

from models.encoders import StyleEncoder, ContentEncoder, ResidualEncoder
from models.generator import SBERTGenerator
from models.discriminator import SBERTDiscriminator

class Disentangler(nn. Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.style_encoder      = StyleEncoder(cfg)
        self.content_encoder    = ContentEncoder(cfg)
        self.residual_encoder   = ResidualEncoder(cfg)

        self.generator      = SBERTGenerator(cfg)
        self.discriminator  = SBERTDiscriminator(cfg)

        self.style_classifier   = nn.Linear()
        self.content_regressor  = nn.Linear()