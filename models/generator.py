import torch
import torch.nn as nn

class SBERTGenerator(nn. Module):
    def __init__(self, cfg):
        super().__init__()