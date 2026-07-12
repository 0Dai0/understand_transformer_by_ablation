"""plots.py — every figure for the RoPE frequency study, one function per figure.

    python projects/rope_study/plots.py [name|all]

names: hierarchy · schedules · base_seeds · noise · learned · perlayer · setup2
Figures are written to figures/.  All reads come from results/*.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import spec
from schedules import rope_inv_freq

HERE = Path(__file__).resolve().parent
R = HERE / "results"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)


# ---------- helpers ----------
def _load(fname):
    return json.loads((R / fname).read_text())


def _finals(fname):
    return {a["arm"]: a["trajectory"][-1]["val"] for a in _load(fname)["arms"]}


def _traj(fname, arm=None):
    d = _load(fname)
    a = d["arms"][0] if arm is None else next(x for x in d["arms"] if x["arm"] == arm)
    return (np.array([p["step"] for p in a["trajectory"]]),
            np.array([p["val"] for p in a["trajectory"]]))


# geom is the REFERENCE (black); the others get distinct, non-purple colors so the
# baseline is never confused with gamma=0.5 (both purples in a sequential map).
SCHED_COLORS = {
    "geom · baseline":      "k",
    "gamma=0.5 · global":   "#1f77b4",
    "gamma=2.0 · local":    "#17becf",
    "base=1e3 · short":     "#2ca02c",
    "base=1e5 · long":      "#9467bd",
    "rotary=50% · partial": "#ff7f0e",
}


# ---------- figures ----------
def hierarchy():
    """What actually matters: position / a multi-freq code / the exact schedule."""
    def fbl(fname, label):
        for a in _load(fname)["arms"]:
            if a["arm"] == label:
                return a["trajectory"][-1]["val"]
        raise KeyError(label)

    rows = [
        ("no position\n(1 freq, λ≫ctx)",   fbl("single_sweep.json", "single·longest λ≈54k"), 0),
        ("1 freq · λ≈6\n(shortest)",        fbl("single_sweep.json", "single·shortest λ≈6"),  1),
        ("1 freq · λ≈580\n(≈ context)",     fbl("single_sweep.json", "single·mid λ≈580"),     1),
        ("64 freqs · λ≤49\n(base 8)",       fbl("base_sweep.json", "base=8"),                 2),
        ("64 freqs · standard\n(base 1e4)", fbl("base_sweep.json", "base=10000"),             2),
        ("64 freqs · best\n(base 256)",     fbl("base_sweep.json", "base=256"),               2),
    ]
    labels = [r[0] for r in rows]
    vals = np.array([r[1] for r in rows])
    tier = {0: "#d62728", 1: "#ff7f0e", 2: "#1f77b4"}
    colors = [tier[r[2]] for r in rows]
    y = np.arange(len(rows))[::-1]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.8, 5.6), gridspec_kw={"width_ratios": [1.35, 1]})
    axL.barh(y, vals, color=colors, height=0.62)
    axL.set_yticks(y); axL.set_yticklabels(labels, fontsize=8)
    axL.set_xlim(5.4, 8.7); axL.set_xlabel("final validation CE")
    axL.set_title("full range — POSITION AT ALL is worth ~2.7 CE")
    for yi, v in zip(y, vals):
        axL.text(v + 0.04, yi, f"{v:.3f}", va="center", fontsize=8)
    axL.grid(True, axis="x", ls=":", alpha=0.4)
    axR.barh(y, vals, color=colors, height=0.62)
    axR.set_yticks(y); axR.set_yticklabels([])
    axR.set_xlim(5.53, 5.98); axR.set_xlabel("final validation CE  (zoom)")
    axR.set_title("zoom — a CODE (1→64 freqs) ~0.2;  schedule ~0.03")
    for yi, v in zip(y, vals):
        if v <= 5.98:
            axR.text(v + 0.004, yi, f"{v:.3f}", va="center", fontsize=8)
    axR.axvspan(5.545, 5.579, color="#1f77b4", alpha=0.08)
    axR.grid(True, axis="x", ls=":", alpha=0.4)
    fig.suptitle("What actually matters in RoPE positional encoding  (d6, seq_len=512)", fontsize=13, y=1.00)
    fig.tight_layout(); fig.savefig(FIG / "hierarchy.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def schedules():
    """The 6 Setup-1 schedules: the curves + the period distribution.

    y-axis is period on a LOG scale, so standard RoPE (geom) — period = base^(j/N),
    exponential in j — is a STRAIGHT line (a linear-y wavelength plot shows the same
    curve shooting up at the end). geom is drawn as a bold black reference; gamma warps
    it up (γ<1) or down (γ>1) around it."""
    np.random.seed(0)
    head_dim = spec.HEAD_DIM
    arms = spec.ARMS

    def periods(dials):
        inv = rope_inv_freq(head_dim, base=float(dials.get("ROPE_BASE", 10000.0)),
                            gamma=float(dials.get("ROPE_GAMMA", 1.0)),
                            rotary_pct=float(dials.get("ROPE_PCT", 1.0))).cpu().numpy()
        p = np.full_like(inv, np.nan); nz = inv > 0; p[nz] = 1.0 / inv[nz]
        return p

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.6))
    for label, dials in arms:
        c = SCHED_COLORS[label]
        geom = label.startswith("geom")
        p = periods(dials)
        axL.plot(range(len(p)), p, "-o", color=c, lw=2.6 if geom else 1.6, ms=3.2,
                 zorder=5 if geom else 3,
                 label=label + "  (straight = log-uniform)" if geom else label)
        n_rot = np.sum(~np.isnan(p))
        if n_rot < len(p):
            axL.axvline(n_rot - 0.5, color=c, ls="--", lw=1.0, alpha=0.6)
    axL.set_yscale("log")
    axL.set_xlabel("channel-pair index  j   (0 = highest freq / shortest period)")
    axL.set_ylabel("period  =  1 / θ_j    (LOG axis)")
    axL.set_title(f"RoPE frequency schedules — the curves  (head_dim={head_dim}, N={head_dim // 2})")
    axL.grid(True, which="both", ls=":", alpha=0.4); axL.legend(fontsize=8, loc="upper left")
    for i, (label, dials) in enumerate(arms):
        p = periods(dials); p = p[~np.isnan(p)]
        yv = np.full_like(p, i) + np.random.uniform(-0.18, 0.18, size=p.shape)
        axR.scatter(p, yv, s=16, color=SCHED_COLORS[label], alpha=0.8, edgecolors="none")
    axR.set_xscale("log")
    axR.set_yticks(range(len(arms))); axR.set_yticklabels([lbl for lbl, _ in arms], fontsize=8)
    axR.invert_yaxis(); axR.set_xlabel("period  (log scale)")
    axR.set_title("period distribution — where the 64 pairs sit in [1, base]")
    axR.grid(True, axis="x", which="both", ls=":", alpha=0.4)
    fig.tight_layout(); fig.savefig(FIG / "frequency_schedules.png", dpi=150); plt.close(fig)


def schedule_loss():
    """The Setup-1(a) LOSS data: the 6 schedule arms vs the 5-seed noise band."""
    arms = _load("curves.json")["arms"]
    seeds = np.array([a["trajectory"][-1]["val"] for a in _load("noise_floor.json")["arms"]])
    mu, sd = seeds.mean(), seeds.std(ddof=1)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.6, 5.4))
    for a in arms:
        c = SCHED_COLORS.get(a["arm"], "gray")
        x = [p["step"] for p in a["trajectory"]]; y = [p["val"] for p in a["trajectory"]]
        axL.plot(x, y, "-", color=c, lw=1.7, label=f'{a["arm"]}  ({y[-1]:.3f})')
        axR.plot(x, y, "-o", color=c, lw=1.7, ms=3.5)
    axL.set_xscale("log"); axL.set_xlabel("training step"); axL.set_ylabel("validation cross-entropy")
    axL.set_title(f"schedule sweep — full val-loss curves  (d{spec.DEPTH}, seq_len={spec.SEQ_LEN})")
    axL.grid(True, which="both", ls=":", alpha=0.4); axL.legend(fontsize=7)
    for k in (2, 1):
        axR.axhspan(mu - k * sd, mu + k * sd, color="#d62728", alpha=0.08)
    axR.axhline(mu, color="#d62728", lw=1.0, ls="--", alpha=0.7, label=f"5-seed baseline μ={mu:.3f}±{sd:.3f}")
    max_step = max(p["step"] for a in arms for p in a["trajectory"])
    axR.set_xlim(250, max_step * 1.6); axR.set_ylim(5.51, 5.85); axR.set_xscale("log")
    axR.set_xlabel("training step  (from mid-training)"); axR.set_ylabel("validation cross-entropy")
    axR.set_title("zoom — every schedule sits inside the ±1σ/±2σ noise band")
    axR.grid(True, which="both", ls=":", alpha=0.4); axR.legend(fontsize=8, loc="upper right")
    fig.tight_layout(); fig.savefig(FIG / "schedule_loss.png", dpi=150); plt.close(fig)


def base_seeds():
    """The base sweep with 5 seeds/base: does the trend survive noise?"""
    def load(fname):
        return [(float(a["dials"]["ROPE_BASE"]), a["trajectory"][-1]["val"]) for a in _load(fname)["arms"]]
    rows = load("base_sweep.json")
    if (R / "base_seeds.json").exists():
        rows += load("base_seeds.json")
    N = spec.HEAD_DIM // 2
    bases = sorted(set(b for b, _ in rows))
    reach, mean, std, allpts = [], [], [], []
    for b in bases:
        vals = np.array([v for bb, v in rows if bb == b])
        reach.append(2 * np.pi * b ** ((N - 1) / N)); mean.append(vals.mean())
        std.append(vals.std(ddof=1) if len(vals) > 1 else 0.0); allpts.append(vals)
    reach = np.array(reach); mean = np.array(mean); std = np.array(std)
    n_seed = max(len(v) for v in allpts)

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    for r, vals in zip(reach, allpts):
        ax.scatter([r] * len(vals), vals, s=20, color="0.72", zorder=1)
    ax.errorbar(reach, mean, yerr=std, fmt="-o", color="#1f77b4", lw=1.9, ms=6,
                capsize=4, zorder=3, label=f"mean ± std ({n_seed} seeds)")
    ax.axvline(spec.SEQ_LEN, color="gray", ls="--", lw=1.2, label=f"context = {spec.SEQ_LEN} tokens")
    ax.set_xscale("log")
    ax.set_xlabel("longest wavelength reached (tokens)  =  2π · base^((N-1)/N)")
    ax.set_ylabel("final validation CE")
    ax.set_title(f"RoPE base sweep with seeds  (d{spec.DEPTH}, seq_len={spec.SEQ_LEN})\n"
                 "does the base trend survive seed noise?")
    ax.grid(True, which="both", ls=":", alpha=0.4); ax.legend()
    for r, m, b in zip(reach, mean, bases):
        ax.annotate(f"{b:g}", (r, m), textcoords="offset points", xytext=(0, 10),
                    fontsize=7, ha="center", color="#1f77b4")
    fig.tight_layout(); fig.savefig(FIG / "base_sweep_seeds.png", dpi=150); plt.close(fig)


def noise():
    """Single-seed noise floor vs the schedule/base 'effects'."""
    np.random.seed(0)
    seeds = np.array(list(_finals("noise_floor.json").values()))
    sched = np.array(list(_finals("curves.json").values()))
    base = np.array(list(_finals("base_sweep.json").values()))
    mu, sd = seeds.mean(), seeds.std(ddof=1)

    fig, ax = plt.subplots(figsize=(11, 4.6))
    for k in (2, 1):
        ax.axvspan(mu - k * sd, mu + k * sd, color="#d62728", alpha=0.07)
    ax.axvline(mu, color="#d62728", lw=1.2, ls="--", alpha=0.7)
    ax.axvline(mu - sd, color="#d62728", lw=0.8, ls=":", alpha=0.6)
    ax.axvline(mu + sd, color="#d62728", lw=0.8, ls=":", alpha=0.6)
    rows = [("same config,\n5 seeds  (= NOISE)", seeds, "#d62728"),
            ("6 schedules,\nseed 42", sched, "#1f77b4"),
            ("9 bases,\nseed 42", base, "#2ca02c")]
    for i, (label, vals, c) in enumerate(rows):
        yv = 2 - i; jit = np.random.uniform(-0.10, 0.10, size=vals.shape)
        ax.scatter(vals, np.full_like(vals, yv) + jit, s=46, color=c, alpha=0.85, edgecolors="none", zorder=3)
        ax.text(mu + 2.05 * sd, yv, f"spread {vals.max()-vals.min():.3f}", va="center", fontsize=8, color=c)
    ax.set_yticks([2, 1, 0]); ax.set_yticklabels([r[0] for r in rows], fontsize=9); ax.set_ylim(-0.6, 2.6)
    ax.set_xlabel("final validation CE")
    ax.set_title(f"Is any of it real?  single-seed noise: μ={mu:.3f}, σ={sd:.3f} "
                 f"(range {seeds.max()-seeds.min():.3f})\n"
                 "red bands = ±1σ, ±2σ.  Effects inside the bands are seed lottery.")
    ax.grid(True, axis="x", ls=":", alpha=0.4)
    fig.tight_layout(); fig.savefig(FIG / "noise_floor.png", dpi=150); plt.close(fig)


def learned():
    """Setup-2 shared: the learned profile vs the γ family + the residual δ."""
    d = _load("learned_delta.json")
    inv_learned = np.array(d["inv_freq"]); inv_base = np.array(d["inv_freq_base"]); delta = np.array(d["delta"])
    N = len(delta); head_dim = 2 * N; j = np.arange(N)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.6, 5.4))
    for i, g in enumerate((0.5, 0.75, 1.5, 2.0)):
        p = 1.0 / rope_inv_freq(head_dim, base=d["base"], gamma=g).cpu().numpy()
        axL.plot(j, p, color="0.82", lw=1.0, zorder=1, label="Setup-1 γ family (0.5–2)" if i == 0 else None)
    axL.plot(j, 1.0 / inv_base, "-", color="#1f77b4", lw=2.0, label="geometric baseline (γ=1)", zorder=3)
    axL.plot(j, 1.0 / inv_learned, "-o", color="#d62728", lw=1.8, ms=3, label="learned", zorder=4)
    axL.set_yscale("log"); axL.set_xlabel("channel-pair index  j"); axL.set_ylabel("period  (1 / θ_j)")
    axL.set_title("learned frequency profile vs the γ family")
    axL.grid(True, which="both", ls=":", alpha=0.4); axL.legend(fontsize=8, loc="upper left")
    axR.axhline(0.0, color="#1f77b4", lw=1.5, label="baseline (δ=0)")
    axR.plot(j, delta, "-o", color="#d62728", lw=1.8, ms=3, label="learned δ")
    axR.set_xlabel("channel-pair index  j"); axR.set_ylabel("log-residual  δ_j   (ln of freq multiplier)")
    axR.set_title(f"the learned residual  (max |δ| = {np.abs(delta).max():.3f} "
                  f"→ ×{np.exp(np.abs(delta).max()):.2f} freq)")
    axR.grid(True, which="both", ls=":", alpha=0.4); axR.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "learned_freqs.png", dpi=150); plt.close(fig)


def perlayer():
    """Setup-2 per-layer: did layers specialize? (per-layer period + residual)."""
    d = _load("perlayer_delta.json")
    delta = np.array(d["delta"]); inv = np.array(d["inv_freq"]); inv_base = np.array(d["inv_freq_base"])
    L, N = delta.shape; j = np.arange(N)
    cmap = plt.get_cmap("viridis"); colors = [cmap(l / max(1, L - 1)) for l in range(L)]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.6, 5.4))
    axL.plot(j, 1.0 / inv_base, color="k", lw=2.2, ls="--", label="geometric baseline", zorder=6)
    for l in range(L):
        axL.plot(j, 1.0 / inv[l], "-", color=colors[l], lw=1.5, label=f"layer {l}")
    axL.set_yscale("log"); axL.set_xlabel("channel-pair index  j"); axL.set_ylabel("period  (1 / θ_j)")
    axL.set_title("learned period per layer  (dark=early, bright=late)")
    axL.grid(True, which="both", ls=":", alpha=0.4); axL.legend(fontsize=7, ncol=2)
    axR.axhline(0.0, color="k", lw=1.2, ls="--", label="baseline (δ=0)")
    for l in range(L):
        axR.plot(j, delta[l], "-o", color=colors[l], lw=1.5, ms=2.5, label=f"layer {l}")
    axR.set_xlabel("channel-pair index  j"); axR.set_ylabel("log-residual  δ_j")
    mx = np.abs(delta).max()
    axR.set_title(f"per-layer residual δ  (max |δ| = {mx:.3f} → ×{np.exp(mx):.2f} freq)")
    axR.grid(True, which="both", ls=":", alpha=0.4); axR.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(FIG / "perlayer_freqs.png", dpi=150); plt.close(fig)


def setup2():
    """The Setup-2 LOSS data: shared & per-layer vs baseline + the noise band."""
    gx, gy = _traj("curves.json", "geom · baseline")
    sx, sy = _traj("learn.json")
    px, py = _traj("perlayer.json")
    seeds = np.array([a["trajectory"][-1]["val"] for a in _load("noise_floor.json")["arms"]])
    mu, sd = seeds.mean(), seeds.std(ddof=1)
    series = [("geom baseline (seed 42)", gx, gy, "k"),
              ("shared δ", sx, sy, "#1f77b4"),
              ("per-layer δ", px, py, "#2ca02c")]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.6, 5.4))
    for name, x, y, c in series:
        axL.plot(x, y, "-", color=c, lw=1.7, label=f"{name}  (end {y[-1]:.3f})")
    axL.set_xscale("log"); axL.set_xlabel("training step"); axL.set_ylabel("validation cross-entropy")
    axL.set_title(f"Setup 2 — full val-loss curves  (d{spec.DEPTH}, seq_len={spec.SEQ_LEN})")
    axL.grid(True, which="both", ls=":", alpha=0.4); axL.legend(fontsize=8)
    for k in (2, 1):
        axR.axhspan(mu - k * sd, mu + k * sd, color="#d62728", alpha=0.08)
    axR.axhline(mu, color="#d62728", lw=1.0, ls="--", alpha=0.7, label=f"5-seed baseline μ={mu:.3f}±{sd:.3f}")
    for name, x, y, c in series:
        axR.plot(x, y, "-o", color=c, lw=1.7, ms=4)
        axR.annotate(f"{y[-1]:.3f}", (x[-1], y[-1]), color=c, fontsize=8, va="center",
                     xytext=(6, 0), textcoords="offset points")
    axR.set_xlim(250, gx.max() * 1.6); axR.set_ylim(5.51, 5.85); axR.set_xscale("log")
    axR.set_xlabel("training step  (from mid-training)"); axR.set_ylabel("validation cross-entropy")
    axR.set_title("zoom — red band = baseline ±1σ/±2σ (5 seeds)")
    axR.grid(True, which="both", ls=":", alpha=0.4); axR.legend(fontsize=8, loc="upper right")
    fig.tight_layout(); fig.savefig(FIG / "setup2_loss.png", dpi=150); plt.close(fig)


FIGURES = {
    "hierarchy": hierarchy, "schedules": schedules, "schedule_loss": schedule_loss,
    "base_seeds": base_seeds, "noise": noise, "learned": learned,
    "perlayer": perlayer, "setup2": setup2,
}


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    names = list(FIGURES) if which == "all" else [which]
    for n in names:
        FIGURES[n]()
        print(f"wrote figures/{'hierarchy' if n=='hierarchy' else n}.png" if False else f"[{n}] done")
    print(f"figures in {FIG}")


if __name__ == "__main__":
    main()
