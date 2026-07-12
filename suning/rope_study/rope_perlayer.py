"""rope_perlayer.py — Setup 2, per-layer: EACH layer learns its OWN RoPE frequencies.

Where rope_learnable.RoPELearnableTrunk shares one log-residual delta across all layers,
here every block owns its own delta[N] (init 0). This asks a structural question that the
shared version can't: do different layers WANT different frequency profiles (early = local
/ high-freq, late = global / low-freq)? The payoff is the readout — the per-layer learned
profiles — not the loss (which, per Setup 1 + the noise floor, is expected to stay flat).

Structural note: shared RoPE already hands every layer the FULL frequency menu (1..base);
a layer uses the subset it needs via attention. Per-layer only lets a layer REALLOCATE
resolution within its band — a marginal move, not a new capability — so expect small
effects at seq_len=512. The interesting regime is long context.

Because each layer has different frequencies, cos/sin can no longer be computed once at the
top — each block computes its own from its own delta, in fp32 (compile-safe; gradients reach
each delta by ordinary autograd). Readout via ROPE_DUMP at process exit (atexit).
"""
import atexit
import json
import os

import torch
import torch.nn as nn

from core.model.gpt import GPTConfig
from gpt2_pieces import GPT2MLP
from rope_trunk import RoPEAttention
from schedules import rope_inv_freq

BASE = 10000.0   # the frozen geometric baseline every layer's delta rides on


class RoPEPerLayerBlock(nn.Module):
    """Pre-LN GPT-2 block that owns its OWN learnable log-residual delta and computes
    its own cos/sin each forward."""

    def __init__(self, config):
        super().__init__()
        head_dim = config.n_embd // config.n_head
        self.N = head_dim // 2
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = RoPEAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = GPT2MLP(config)
        self.delta = nn.Parameter(torch.zeros(self.N))   # this layer's frequency residual

    def current_inv_freq(self, inv_freq_base):
        return inv_freq_base * torch.exp(self.delta)

    def forward(self, x, pos, inv_freq_base):
        inv_freq = self.current_inv_freq(inv_freq_base)
        with torch.autocast(device_type=x.device.type, enabled=False):
            freqs = torch.outer(pos.float(), inv_freq.float())        # [T, N]
            cos, sin = freqs.cos(), freqs.sin()
        cos_sin = (cos.bfloat16()[None, :, None, :], sin.bfloat16()[None, :, None, :])
        x = x + self.attn(self.ln_1(x), cos_sin)
        x = x + self.mlp(self.ln_2(x))
        return x


class RoPEPerLayerTrunk(nn.Module):
    """GPT-2 + RoPE where every layer learns its own frequency profile."""

    Config = GPTConfig

    def __init__(self, config):
        super().__init__()
        self.config = config
        head_dim = config.n_embd // config.n_head
        self.N = head_dim // 2
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            h=nn.ModuleList([RoPEPerLayerBlock(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.register_buffer("inv_freq_base", rope_inv_freq(head_dim, base=BASE), persistent=False)
        self._dump_path = os.environ.get("ROPE_DUMP")
        if self._dump_path:
            atexit.register(self._dump_final)
        print(f"[RoPEPerLayerTrunk] per-layer learnable freqs: {config.n_layer} layers x "
              f"delta[{self.N}] init 0 on geometric base={BASE:g}  (head_dim={head_dim})")

    @property
    def blocks(self):
        return self.transformer.h

    def init_weights(self):
        self.apply(self._init_weights)
        head_dim = self.config.n_embd // self.config.n_head
        self.inv_freq_base = rope_inv_freq(head_dim, base=BASE,
                                           device=self.transformer.wte.weight.device)
        for b in self.transformer.h:            # re-zero every layer's delta (init = baseline)
            torch.nn.init.zeros_(b.delta)
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def _dump_final(self):
        try:
            deltas = [b.delta.detach().float().cpu().tolist() for b in self.transformer.h]
            invs = [b.current_inv_freq(self.inv_freq_base).detach().float().cpu().tolist()
                    for b in self.transformer.h]
            with open(self._dump_path, "w") as f:
                json.dump({"base": BASE, "delta": deltas, "inv_freq": invs,
                           "inv_freq_base": self.inv_freq_base.detach().float().cpu().tolist()}, f)
            print(f"[RoPEPerLayerTrunk] dumped per-layer learned freqs -> {self._dump_path}")
        except Exception as e:
            print(f"[RoPEPerLayerTrunk] dump failed: {e}")

    def estimate_flops(self):
        nparams = sum(p.numel() for p in self.parameters())
        lookup = self.transformer.wte.weight.numel()
        l, h = self.config.n_layer, self.config.n_head
        q, t = self.config.n_embd // self.config.n_head, self.config.sequence_len
        return 6 * (nparams - lookup) + 12 * l * h * q * t

    def get_device(self):
        return self.transformer.wte.weight.device

    def forward(self, idx, token_types=None, kv_cache=None, block_mask=None):
        B, T = idx.shape
        x = self.transformer.wte(idx)
        pos = torch.arange(T, device=idx.device)
        for block in self.transformer.h:
            x = block(x, pos, self.inv_freq_base)
        x = self.transformer.ln_f(x)
        return x
