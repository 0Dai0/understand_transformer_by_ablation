"""run.py — train each RoPE frequency-schedule arm through the blessed text
orchestrator and collect val-loss trajectories. Same model / data / budget / recipe
for every arm; the ONLY difference is the RoPE frequency schedule, set by per-arm
env dials (ROPE_BASE / ROPE_GAMMA / ROPE_PCT) the trunk reads. Mirrors the
exemplar's scaling.py driving pattern: subprocess the orchestrator, parse the
scheduled val evals from its log.

Needs a FineWeb shard on disk (outputs/base_data/) — fetch a couple first:
    python exemplars/text_pretrain/data/download_shards.py

    python projects/rope_study/run.py      # trains all arms
    python projects/rope_study/plot.py     # -> rope_study.png
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

import spec

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)


def _nanoinfra_checkout(start):
    """The nanoinfra source tree (a dir holding core/ + modalities/) if we're running
    inside one; None when nanoinfra is only pip-installed."""
    p = start
    while p != p.parent:
        if (p / "core").is_dir() and (p / "modalities").is_dir():
            return p
        p = p.parent
    return None


# Portable launch context: this experiment's dir on PYTHONPATH (for its local trunk
# modules), plus the nanoinfra checkout if present; NANOINFRA_BASE_DIR -> FineWeb data.
_NANO = _nanoinfra_checkout(HERE)
_PP = os.pathsep.join([str(HERE)] + ([str(_NANO)] if _NANO else [])
                      + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else []))
BASE_ENV = {**os.environ, "PYTHONPATH": _PP}
BASE_ENV.setdefault("NANOINFRA_BASE_DIR", str(_NANO / "outputs") if _NANO else "./outputs")
BASE_CWD = str(_NANO) if _NANO else str(HERE)

EVAL_RE = re.compile(r"Step\s+(\d+)\s+\|\s+val/text_ce:\s+([\d.]+)")


def eval_schedule(max_steps, n=spec.N_EVALS, first=5):
    """~n log-spaced integer steps in [first, max_steps] (deduped, sorted)."""
    s = np.unique(np.round(np.logspace(np.log10(first), np.log10(max_steps), n)))
    return [int(x) for x in s]


def run_arm(label, env_dials, max_steps, steps, trunk=spec.TRUNK, seed=spec.SEED):
    ov = spec.train_overrides(max_steps, steps, trunk=trunk, seed=seed)
    print(f"[run ] {label}: dials={env_dials or 'standard'} -> {max_steps} steps ...", flush=True)
    env = {**BASE_ENV, **env_dials}
    out = subprocess.run([sys.executable, "-u", "-m", spec.ORCHESTRATOR, *ov],
                         cwd=BASE_CWD, env=env, capture_output=True, text=True)
    text = out.stdout + "\n" + out.stderr
    traj = [{"step": int(s), "val": float(v)} for s, v in EVAL_RE.findall(text)]
    if out.returncode != 0 or len(traj) < 3:
        raise SystemExit(f"arm {label} FAILED (rc={out.returncode}, {len(traj)} evals):\n{text[-3000:]}")
    print(f"[done] {label}: {len(traj)} evals, val {traj[0]['val']:.3f} -> {traj[-1]['val']:.3f}",
          flush=True)
    return {"arm": label, "dials": env_dials, "trajectory": traj}


SWEEPS = {   # name -> (arms thunk, out file, trunk class path, extra per-run env)
    "arms":   (lambda: spec.ARMS,        "curves.json",       spec.TRUNK,       {}),
    "base":   (lambda: spec.BASE_SWEEP,  "base_sweep.json",   spec.TRUNK,       {}),
    "single": (lambda: spec.SINGLE_SWEEP, "single_sweep.json", spec.TRUNK,      {}),
    "learn":  (lambda: spec.LEARNABLE,   "learn.json",        spec.LEARN_TRUNK,
               {"ROPE_DUMP": str(RESULTS / "learned_delta.json")}),
    "perlayer": (lambda: spec.PERLAYER,  "perlayer.json",     spec.PERLAYER_TRUNK,
                 {"ROPE_DUMP": str(RESULTS / "perlayer_delta.json")}),
}


def run_base_seeds():
    """The base sweep repeated across the NON-42 seeds (seed 42 is already in
    base_sweep.json). Combined -> error bars on the base trend."""
    max_steps = int(spec.MAX_TOKENS // spec.TBS)
    steps = eval_schedule(max_steps)
    new_seeds = [s for s in spec.SEEDS if s != spec.SEED]
    arms = []
    for b in spec.BASE_GRID:
        for sd in new_seeds:
            r = run_arm(f"base={b} · seed={sd}", {"ROPE_BASE": str(b)}, max_steps, steps, seed=sd)
            r["seed"], r["base"] = sd, b
            arms.append(r)
    out = {"depth": spec.DEPTH, "head_dim": spec.HEAD_DIM, "max_steps": max_steps,
           "seeds": new_seeds, "arms": arms}
    (RESULTS / "base_seeds.json").write_text(json.dumps(out, indent=2))
    print(f"WROTE {RESULTS / 'base_seeds.json'}")


def run_seeds():
    """Repeat NOISE_PROBE across SEEDS -> the single-seed noise floor."""
    label0, dials0 = spec.NOISE_PROBE
    max_steps = int(spec.MAX_TOKENS // spec.TBS)
    steps = eval_schedule(max_steps)
    arms = [run_arm(f"{label0} · seed={sd}", dials0, max_steps, steps, seed=sd)
            for sd in spec.SEEDS]
    out = {"depth": spec.DEPTH, "head_dim": spec.HEAD_DIM, "max_steps": max_steps,
           "seeds": spec.SEEDS, "arms": arms}
    (RESULTS / "noise_floor.json").write_text(json.dumps(out, indent=2))
    print(f"WROTE {RESULTS / 'noise_floor.json'}")


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "arms"
    if which == "seeds":
        return run_seeds()
    if which == "baseseeds":
        return run_base_seeds()
    arms_fn, out_name, trunk, sweep_env = SWEEPS[which]
    if "ROPE_DUMP" in sweep_env:                       # fresh readout each run
        Path(sweep_env["ROPE_DUMP"]).unlink(missing_ok=True)
    max_steps = int(spec.MAX_TOKENS // spec.TBS)
    steps = eval_schedule(max_steps)
    arms = [run_arm(label, {**sweep_env, **dials}, max_steps, steps, trunk=trunk)
            for label, dials in arms_fn()]
    out = {"depth": spec.DEPTH, "head_dim": spec.HEAD_DIM, "max_steps": max_steps, "arms": arms}
    (RESULTS / out_name).write_text(json.dumps(out, indent=2))
    print(f"WROTE {RESULTS / out_name}")


if __name__ == "__main__":
    main()
