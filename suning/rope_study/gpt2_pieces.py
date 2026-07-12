"""Classic GPT-2 building blocks (gelu, GPT2MLP), vendored so this experiment is
self-contained. An ablation freezes the exact code that produced its figures, so
each experiment owns its own copy rather than sharing — vendoring, not DRY.
Borrowed from the GPT-2 architecture (minGPT gpt-nano)."""
import math

import torch
import torch.nn as nn


def gelu(x):
    # minGPT's tanh approximation of GELU — the 2019 default.
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


class GPT2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)      # bias=True (GPT-2)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)    # bias=True

    def forward(self, x):
        return self.c_proj(gelu(self.c_fc(x)))
