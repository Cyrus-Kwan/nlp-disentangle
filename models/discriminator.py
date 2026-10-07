import torch
import torch.nn as nn

class SBERTDiscriminator(nn. Module):
    def __init__(self, cfg):
        super().__init__()