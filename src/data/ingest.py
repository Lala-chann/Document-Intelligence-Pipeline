import hashlib
import json
import argparse
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfApi

DATASET_NAME = "Tobi-Bueck/customer-support-tickets"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_DIR= RAW_DIR / "tickets_raw.parquet"
METADATA_PATH = RAW_DIR / "tickets_raw.meta.json"

def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if OUT_DIR.exists() and not args.force:
        print(f"File {OUT_DIR} already exists. USE --force to update it.")
        return

    info = HfApi().dataset_info(DATASET_NAME, revision=args.revision)
    revision = info.sha
    print(f"Using revision (hash): {revision}")

    ds = load_dataset(DATASET_NAME, revision=revision)

    frames = []
    for split_name, split in ds.items():
        part = split.to_pandas()
        part["hf_split"] = split_name
        frames.append(part)
    df = pd.concat(frames, ignore_index=True)

    RAW_DIR.mkdir(parents=True,exist_ok=True)
    df.to_parquet(OUT_DIR, index=False)





    metadata = {
        "dataset_name": DATASET_NAME,
        "revision": revision,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "n_rows": int(len(df)),
        "columns": df.columns.tolist(),
        "sha256_of_file": sha256_of(OUT_DIR),
    }

    METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"Saved: {OUT_DIR}, {len(df)} rows")
    print(f"Metadata saved: {METADATA_PATH}")


if __name__ == "__main__":
    main()