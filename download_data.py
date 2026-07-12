"""download_data.py — fetch a few FineWeb sample-10BT shards so the experiments have
data to train on. Writes into $NANOINFRA_BASE_DIR/base_data (default ./outputs/base_data).
huggingface_hub + pyarrow come with nanoinfra (pip install -r requirements.txt).

    python download_data.py        # a couple of shards (train + val) — enough for the d6 demos
"""
import os
import shutil
import sys
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO = "HuggingFaceFW/fineweb"
BASE = Path(os.environ.get("NANOINFRA_BASE_DIR", "./outputs")).expanduser() / "base_data"
CACHE = BASE / "_hf"


def main(idxs):
    BASE.mkdir(parents=True, exist_ok=True)
    for idx in idxs:
        name = f"{idx}_00000.parquet"
        dst = BASE / f"shard_{idx}_00000.parquet"
        if dst.exists():
            print(f"[skip] {dst}"); continue
        print(f"[download] sample/10BT/{name} ...", flush=True)
        local = hf_hub_download(repo_id=REPO, repo_type="dataset",
                                filename=f"sample/10BT/{name}", local_dir=str(CACHE))
        assert pq.ParquetFile(local).metadata.num_rows > 0, f"empty parquet {local}"
        shutil.move(local, dst)
        print(f"[done] -> {dst}", flush=True)
    print(f"FineWeb ready in {BASE}  (all-but-last shard = train, last = val)")


if __name__ == "__main__":
    main(sys.argv[1:] or ["000", "001"])
