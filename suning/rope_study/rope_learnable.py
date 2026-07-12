"""rope_learnable.py — Setup 2: LEARNABLE RoPE frequencies.

Same GPT-2 + RoPE body as rope_trunk.RoPETrunk, but the per-pair rotation rates are
learned instead of fixed. Parameterization (from the design discussion):

  * log-RESIDUAL around a frozen geometric baseline:
        inv_freq = inv_freq_base.detach() * exp(delta),   delta = Parameter, init 0
    - init delta=0  => exactly standard RoPE at step 0 (can only match or improve).
    - exp           => frequencies stay positive.
    - AdamW weight_decay pulls delta -> 0 = toward the geometric baseline, so it is a
      PRIOR toward standard RoPE rather than a pathology. delta can therefore ride the
      default matrix param-group (lr_max, wd) with NO core change.
  * SHARED across layers/heads (one delta of shape [N]) — the cleanest first question:
    "is there a single better frequency profile than geometric?"

Structural change vs Setup 1: cos/sin are no longer a cached buffer — they are
recomputed every forward from the live delta, in fp32 (bf16 trig on large angles is
garbage), then cast to bf16 for the rotation. Gradients flow through cos/sin to delta
by ordinary autograd (RoPE is applied OUTSIDE the attention kernel, so SDPA-vs-Flex
is irrelevant). This is all plain tensor ops, so it stays torch.compile-safe.

Readout: if env ROPE_DUMP is set, the final learned inv_freq / delta is written there
at process exit (atexit — no forward side effects, compile-safe).
"""
import atexit
import json
import os

import torch
import torch.nn as nn

from core.model.gpt import GPTConfig
from rope_trunk import RoPEBlock
from schedules import rope_inv_freq

BASE = 10000.0   # the frozen geometric baseline delta rides on


class RoPELearnableTrunk(nn.Module):
    """GPT-2 + RoPE with a shared, learnable log-residual on the geometric frequencies."""

    Config = GPTConfig

    def __init__(self, config):
        super().__init__()
        self.config = config
        head_dim = config.n_embd // config.n_head
        self.N = head_dim // 2
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            h=nn.ModuleList([RoPEBlock(config, i) for i in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        # frozen geometric baseline; delta is the learned log-residual (init 0 = baseline)
        self.register_buffer("inv_freq_base", rope_inv_freq(head_dim, base=BASE), persistent=False)
        self.delta = nn.Parameter(torch.zeros(self.N))
        self._dump_path = os.environ.get("ROPE_DUMP")
        if self._dump_path:
            atexit.register(self._dump_final)
        print(f"[RoPELearnableTrunk] learnable freqs: shared log-residual delta[{self.N}] "
              f"init 0 on geometric base={BASE:g}  (head_dim={head_dim})")

    @property
    def blocks(self):
        return self.transformer.h

    def init_weights(self):
        self.apply(self._init_weights)
        head_dim = self.config.n_embd // self.config.n_head
        # recompute the frozen baseline on the real device; re-zero delta (init = baseline)
        self.inv_freq_base = rope_inv_freq(head_dim, base=BASE,
                                           device=self.transformer.wte.weight.device)
        torch.nn.init.zeros_(self.delta)
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

    def current_inv_freq(self):
        return self.inv_freq_base * torch.exp(self.delta)     # [N], differentiable in delta

    def _dump_final(self):
        try:
            with open(self._dump_path, "w") as f:
                json.dump({"base": BASE,
                           "delta": self.delta.detach().float().cpu().tolist(),
                           "inv_freq": self.current_inv_freq().detach().float().cpu().tolist(),
                           "inv_freq_base": self.inv_freq_base.detach().float().cpu().tolist()}, f)
            print(f"[RoPELearnableTrunk] dumped learned freqs -> {self._dump_path}")
        except Exception as e:   # a readout failure must never crash the run
            print(f"[RoPELearnableTrunk] dump failed: {e}")

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
        dev = idx.device
        x = self.transformer.wte(idx)
        inv_freq = self.current_inv_freq()
        # differentiable cos/sin in fp32 (angles reach ~pos rad; bf16 trig would be garbage),
        # then bf16 for the rotation. delta gets gradient through cos/sin by ordinary autograd.
        with torch.autocast(device_type=dev.type, enabled=False):
            pos = torch.arange(T, dtype=torch.float32, device=dev)
            freqs = torch.outer(pos, inv_freq.float())            # [T, N]
            cos, sin = freqs.cos(), freqs.sin()
        cos_sin = (cos.bfloat16()[None, :, None, :], sin.bfloat16()[None, :, None, :])  # (1,T,1,N)
        for block in self.transformer.h:
            x = block(x, cos_sin)
        x = self.transformer.ln_f(x)
        return x
