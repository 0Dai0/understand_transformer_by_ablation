"""rope_trunk.py — a GPT-2 + RoPE trunk whose RoPE FREQUENCY SCHEDULE is the knob.

The architecture is exactly example_gpt2_vs_modern's GPT2RoPETrunk (classic GPT-2
body — LayerNorm+bias, tanh-GELU, biased linears, plain multi-head — with RoPE for
position). The ONE thing this trunk parameterizes is HOW the RoPE frequencies are
chosen: the study's object.

Setup 1 (this file): the frequencies are FIXED, produced by schedules.rope_inv_freq
from a few dials read from the environment. Setup 2 (learnable frequencies) will
extend this same trunk; see README.

Why env vars and not model config: the text orchestrator builds a fixed GPTConfig
(modalities/text/train_text.py hardcodes the field set) and does not thread extra
model-config through to the trunk. So a project's per-arm knobs ride the subprocess
environment that run.py already sets per arm. Core stays untouched.

    ROPE_BASE   (default 10000.0)   max period / context reach (theta-scaling)
    ROPE_GAMMA  (default 1.0)       warp exponent: >1 local (short), <1 global (long)
    ROPE_PCT    (default 1.0)       fraction of channel-pairs that rotate (rest: theta=0)
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model.gpt import GPTConfig, apply_rotary_emb
from gpt2_pieces import gelu, GPT2MLP  # stable classic-GPT2 pieces
from schedules import rope_inv_freq


def _env_float(name, default):
    v = os.environ.get(name)
    return default if v in (None, "") else float(v)


class RoPEAttention(nn.Module):
    """Classic biased multi-head attention with RoPE on q,k (no QK-norm, no GQA)."""

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)   # bias=True (GPT-2)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)       # bias=True

    def forward(self, x, cos_sin):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        hd = C // self.n_head
        q = q.view(B, T, self.n_head, hd)   # keep (B,T,nh,hd) so RoPE broadcasts over heads
        k = k.view(B, T, self.n_head, hd)
        v = v.view(B, T, self.n_head, hd)
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # SDPA, not FlexAttention
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class RoPEBlock(nn.Module):
    """Pre-LN GPT-2 block, RoPE-attention variant."""

    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = RoPEAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = GPT2MLP(config)

    def forward(self, x, cos_sin=None, kv_cache=None, block_mask=None):
        x = x + self.attn(self.ln_1(x), cos_sin)
        x = x + self.mlp(self.ln_2(x))
        return x


class RoPETrunk(nn.Module):
    """GPT-2 body + RoPE with a configurable (fixed) frequency schedule.

    Trains only (no KV-cache inference path — the comparison is training curves)."""

    Config = GPTConfig

    def __init__(self, config):
        super().__init__()
        self.config = config
        # Frequency-schedule dials (Setup 1: fixed, read once from the env).
        self.rope_base = _env_float("ROPE_BASE", 10000.0)
        self.rope_gamma = _env_float("ROPE_GAMMA", 1.0)
        self.rope_pct = _env_float("ROPE_PCT", 1.0)
        self.rope_single = _env_float("ROPE_SINGLE", None)   # if set: one frequency for all pairs
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),      # token embedding
            h=nn.ModuleList([RoPEBlock(config, i) for i in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),                        # final norm (trunk owns it)
        ))
        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        _sched = (f"single={self.rope_single:g}" if self.rope_single is not None
                  else f"base={self.rope_base:g} gamma={self.rope_gamma:g} rotary_pct={self.rope_pct:g}")
        print(f"[RoPETrunk] freq schedule: {_sched}  (head_dim={head_dim}, N={head_dim // 2})")

    @property
    def blocks(self):
        return self.transformer.h

    def init_weights(self):
        # Meta-init contract (same as GPT2RoPETrunk): init EVERY param (LayerNorm too)
        # AND recompute the rotary buffers (to_empty leaves them garbage).
        self.apply(self._init_weights)
        head_dim = self.config.n_embd // self.config.n_head
        self.cos, self.sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)   # GPT-2 / minGPT init
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        inv_freq = rope_inv_freq(head_dim, base=self.rope_base, gamma=self.rope_gamma,
                                 rotary_pct=self.rope_pct, single=self.rope_single,
                                 device=device)                               # [N] fp32, dial-set
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]     # (1,T,1,N) for broadcast
        return cos, sin

    def estimate_flops(self):
        """FLOPs/token for MFU. wte is a lookup (excluded); no wpe."""
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
        # GPT-2 has no token-type embedding — token_types is ignored on purpose.
        assert T <= self.cos.size(1), f"Sequence length {T} exceeds rotary cache {self.cos.size(1)}"
        cos_sin = self.cos[:, :T], self.sin[:, :T]
        for block in self.transformer.h:
            x = block(x, cos_sin)
        x = self.transformer.ln_f(x)
        return x
