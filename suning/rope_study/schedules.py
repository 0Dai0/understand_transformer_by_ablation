"""schedules.py — RoPE frequency schedules as a function of a few interpretable dials.

The object of the RoPE study: HOW to choose the per-channel-pair rotation rates
(the "frequencies" theta_j) that RoPE uses. Standard RoPE is one specific choice —
a geometric / log-uniform sweep of periods over [1, base]. These dials bend,
rescale, or truncate that curve while keeping it a valid schedule.

Returns inv_freq (theta) of shape [head_dim // 2] — one rate per rotated pair
(apply_rotary_emb pairs channel j with channel j + head_dim//2, so cos/sin carry
head_dim//2 entries, one per pair).
"""
import math

import torch


def rope_inv_freq(head_dim, base=10000.0, gamma=1.0, rotary_pct=1.0, single=None, device=None):
    """RoPE frequencies theta_j, dial-parameterized. Standard RoPE = the defaults.

        period_j = base ** ((j/N) ** gamma),   theta_j = 1 / period_j,   N = head_dim//2

      base       max period / context reach. Standard 10000. Larger -> longer
                 wavelengths reachable (theta-scaling / NTK); sets the TOP endpoint.
      gamma      warp exponent on the normalized channel index j/N. gamma=1 is the
                 standard geometric (log-uniform) sweep. gamma>1 pushes most pairs to
                 SHORT periods (high-freq / local); gamma<1 to LONG periods (low-freq
                 / global). Endpoints stay pinned; only the DENSITY is redistributed.
      rotary_pct fraction of pairs that actually rotate; the rest get theta=0 (no
                 rotation = position-free / pure content). GPT-NeoX partial rotary.
      single     if set, COLLAPSE every pair to one frequency: theta_j = 1/single for
                 all j (period `single`, token-wavelength 2*pi*single). Overrides
                 base/gamma/rotary_pct. `single=1` = only the shortest wavelength.

    Note token wavelength = 2*pi / theta_j = 2*pi * period_j (the 'period' here drops
    the 2*pi; multiply by 2*pi ~ 6.28 for the reach in tokens).

    gamma=1, base=10000, rotary_pct=1 reproduces standard RoPE bit-for-bit: the
    exponent (j/N)**1 = j/N equals the stock 2j/head_dim, so base**(-(j/N)) is the
    stock 1/(base**(2j/head_dim)).
    """
    N = head_dim // 2
    j = torch.arange(N, dtype=torch.float32, device=device)
    if single is not None:                                       # one frequency for all pairs
        return torch.full((N,), 1.0 / float(single), dtype=torch.float32, device=device)
    exponent = (j / N) ** gamma
    inv_freq = torch.exp(-exponent * math.log(base))            # = base ** (-exponent)
    if rotary_pct < 1.0:
        n_rot = max(1, int(round(N * rotary_pct)))
        inv_freq = inv_freq * (j < n_rot).to(inv_freq.dtype)    # zero the non-rotating pairs
    return inv_freq
