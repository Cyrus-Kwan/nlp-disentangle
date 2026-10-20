import torch
import torch.nn as nn
from torch.autograd import Function

class Probe(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.mlp(x)

class _GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None
