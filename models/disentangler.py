import torch
import torch.nn as nn

from models.encoders import StyleEncoder, ContentEncoder, ContextEncoder, VAE
from models.generator import SBERTGenerator
from models.discriminator import SBERTDiscriminator
from models.probes import Probe

class SBERTDisentangler(nn. Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.style_encoder      = StyleEncoder(cfg)
        self.content_encoder    = ContentEncoder(cfg)
        self.context_encoder    = ContextEncoder(cfg)
        # self.style_encoder      = VAE(cfg.bert_dim, cfg.style_hidden, cfg.style_dim)
        # self.content_encoder    = VAE(cfg.bert_dim, cfg.content_hidden, cfg.content_dim)
        # self.context_encoder    = VAE(cfg.bert_dim, cfg.context_hidden, cfg.context_dim)

        self.generator      = SBERTGenerator(cfg)
        self.discriminator  = SBERTDiscriminator(cfg)

        self.style_classifier   = nn.Linear(cfg.style_dim, 1)
        self.content_regressor  = nn.Linear(cfg.content_dim, cfg.content_dim)
        self.context_regressor  = nn.Linear(cfg.context_dim, cfg.question_dim)

        self.probes = nn.ModuleDict()

        in_dims = {
            "style": cfg.style_dim,
            "content": cfg.content_dim,
            "context": cfg.context_dim
        }
        out_dims = {
            "style": cfg.style_out,
            "content": cfg.content_out,
            "context": cfg.context_out
        }

        for i in in_dims:
            for o in out_dims:
                if i == o: continue
                self.probes[f"{i}_{o}"] = Probe(in_dims[i], out_dims[o])

    def probe_parameters(self):
        return self.probes.parameters()

    def generator_parameters(self):
        for module in (
            self.style_encoder,
            self.content_encoder, 
            self.context_encoder,
            self.generator,
            self.style_classifier,
            self.content_regressor,
            self.context_regressor,
        ):
            yield from module.parameters()

    def discriminator_parameters(self):
        return self.discriminator.parameters()
    
    def encode(self, x, sample=True):
        style   = self.style_encoder(x)
        content = self.content_encoder(x)
        context = self.context_encoder(x)
        # style_mu, style_logvar     = self.style_encoder.encode(x)
        # content_mu, content_logvar = self.content_encoder.encode(x)
        # context_mu, context_logvar = self.context_encoder.encode(x)

        if sample:
            style   = self.style_encoder(x)
            content = self.content_encoder(x)
            context = self.context_encoder(x)
            # style   = self.style_encoder.reparameterize(style_mu, style_logvar)
            # content = self.content_encoder.reparameterize(content_mu, content_logvar)
            # context = self.context_encoder.reparameterize(context_mu, context_logvar)
        else:
            style, content, context = (style, content, context)
            # style, content, context = (style, content, context_mu)
            # style, content, context = (style, content_mu, context_mu)
            # style, content, context = (style_mu, content_mu, context_mu)

        return style, content, context

    def reconstruct(self, x, sample=True):
        style, content, context = self.encode(x, sample=sample)
        bert_recon = self.generator(style, content, context)
        return bert_recon, style, content, context

    def generate(self, style_embed, content_embed, context_embed):
        return self.generator(style_embed, content_embed, context_embed)

    def classify(self, x):
        style = self.style_encoder(x)
        return self.style_classifier(style)