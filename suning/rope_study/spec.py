"""spec.py — the RoPE frequency study. The knobs and the arms.

Base model: GPT-2 + RoPE (rope_trunk.RoPETrunk). Every arm is
the SAME model, data, budget, and recipe; the ONLY thing that varies is the RoPE
frequency schedule, set per-arm by environment dials the trunk reads
(ROPE_BASE / ROPE_GAMMA / ROPE_PCT). Standard geometric RoPE is the baseline arm.

Setup 1 (this spec): fixed schedules. Setup 2 (learnable frequencies) will extend
the same trunk; see README.
"""
DEPTH = 6                  # smoke scale (a couple of minutes per arm on one GPU)
LR_MAX = "3e-4"            # ONE shared recipe (tuned for the modern trunk) — honest
                           #   "swap the schedule, keep the recipe" comparison
SEED = 42

SEQ_LEN, DBS, TBS = 512, 16, 16384
MAX_TOKENS = 20_000_000
WARMUP_STEPS = 100
N_EVALS = 30
EVAL_TOKENS = 131072

ORCHESTRATOR = "modalities.text.train_text"
TRUNK = "rope_trunk.RoPETrunk"                       # Setup 1: fixed schedules
LEARN_TRUNK = "rope_learnable.RoPELearnableTrunk"    # Setup 2: shared learnable
PERLAYER_TRUNK = "rope_perlayer.RoPEPerLayerTrunk"   # Setup 2: per-layer learnable

# d6 geometry, derived exactly as train_text.yaml does (dim = 64·depth; 128-wide heads).
# d6 -> dim 384, n_head 3, head_dim 128, so N = 64 frequency pairs (the profile's x-axis).
DIM = DEPTH * 64
N_HEAD = max(1, (DIM + 127) // 128)
HEAD_DIM = DIM // N_HEAD

# Arms: (label, env-dial dict). Empty dict = standard geometric RoPE (the baseline).
# First batch: bend (gamma), rescale (base), truncate (rotary_pct) — a few curves to eyeball.
ARMS = [
    ("geom · baseline",      {}),                          # gamma=1, base=1e4, pct=1
    ("gamma=0.5 · global",   {"ROPE_GAMMA": "0.5"}),       # bend UP: more long-wavelength pairs
    ("gamma=2.0 · local",    {"ROPE_GAMMA": "2.0"}),       # bend DOWN: more short-wavelength pairs
    ("base=1e3 · short",     {"ROPE_BASE": "1000"}),       # shorter max period (context reach)
    ("base=1e5 · long",      {"ROPE_BASE": "100000"}),     # longer max period
    ("rotary=50% · partial", {"ROPE_PCT": "0.5"}),         # half the pairs carry no position
]

# --- Setup-1 follow-up: the base (theta) sweep ---
# The 6-arm sweep above was flat because base=1e4 puts every wavelength far past the
# 512-token context. Shortening base compresses ALL 64 periods toward 1; once the
# longest period (~base) drops below the positional range the model actually uses,
# loss must degrade. This sweep finds that onset = the model's EFFECTIVE POSITIONAL
# RANGE at 512 context. Only base moves (gamma=1, pct=1). `python run.py base`.
BASE_GRID = (8, 16, 32, 64, 128, 256, 512, 1024, 10000)
BASE_SWEEP = [(f"base={b}", {"ROPE_BASE": str(b)}) for b in BASE_GRID]

# --- Setup-1 follow-up 2: single-frequency RoPE ---
# Ning's recollection: even ONE frequency — just the shortest wavelength — already
# captures most of RoPE's benefit (a little worse than full RoPE, not much). Collapse
# all 64 pairs to a single theta and check, plus a mid and a long single for contrast.
# (The core carries a commented-out fossil of the single-LOWEST-freq variant in
# gpt._precompute_rotary_embeddings.) Period p -> token-wavelength ~ 2*pi*p.
# `python run.py single`.
SINGLE_SWEEP = [
    ("single·shortest λ≈6",  {"ROPE_SINGLE": "1"}),      # theta=1: the shortest wavelength only
    ("single·mid λ≈580",     {"ROPE_SINGLE": "93"}),     # geometric midpoint
    ("single·longest λ≈54k", {"ROPE_SINGLE": "8660"}),   # the single-lowest-freq (core fossil)
]

# --- Setup 2: learnable frequencies ---
# The LEARN_TRUNK learns a shared log-residual delta on the geometric baseline (init 0
# = standard RoPE). No dials; the readout (final learned freqs) is dumped via ROPE_DUMP,
# set by run.py. `python run.py learn`.
LEARNABLE = [("learnable · shared δ", {})]
PERLAYER = [("learnable · per-layer δ", {})]

# --- Noise floor: repeat ONE config across seeds to size single-seed variance ---
# Everything above is single-seed (SEED=42). To know whether the 0.009 schedule spread
# and the 0.034 base bowl are real or noise, repeat the geometric baseline across seeds
# and read the std of the final val CE. `python run.py seeds`.
SEEDS = [42, 43, 44, 45, 46]
NOISE_PROBE = ("geom · baseline", {})   # the config repeated across SEEDS


def train_overrides(max_steps, eval_at, trunk=TRUNK, seed=SEED):
    """Hydra CLI overrides — this study's recipe on the orchestrator's defaults.
    Constant LR (no warmdown) so each curve is genuine loss-vs-step; explicit
    log-spaced eval schedule; no checkpoints. The frequency schedule is NOT here —
    it rides the env (run.py sets it per arm); only `model.trunk_class` varies
    (Setup 1 fixed vs Setup 2 learnable)."""
    ov = {
        "model.depth": DEPTH,
        "model.trunk_class": trunk,
        "optimizer.lr_max": LR_MAX,
        "seed": seed,
        "sequence_len": SEQ_LEN,
        "device_batch_size": DBS,
        "total_batch_size": TBS,
        "max_steps": max_steps,
        "optimizer.scheduler.warmup_steps": WARMUP_STEPS,
        "optimizer.scheduler.warmdown_ratio": 0.0,   # constant LR after warmup
        "optimizer.scheduler.final_lr_frac": 1.0,
        "checkpoint.enabled": "false",
        "evaluation.text.eval_at": "[" + ",".join(map(str, eval_at)) + "]",
        "evaluation.text.eval_tokens": EVAL_TOKENS,
        "logging.log_every": 100,
    }
    return [f"{k}={v}" for k, v in ov.items()]
