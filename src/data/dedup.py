import argparse
import hashlib
import json
import re
import pandas as pd

from pathlib import Path
from datasketch import MinHash, MinHashLSH

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IN_PATH = PROJECT_ROOT / "data" / "cleaned" / "tickets_clean.parquet"
OUT_PATH = PROJECT_ROOT / "data" / "cleaned"
REPORT_DIR = PROJECT_ROOT / "reports_for_data"

TOKEN_RE = re.compile(r"\w+")

def exact_key(text: str) -> str:
    norm = " ".join(text.lower().split())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()

def shingles(text: str, k:int=5) -> set:
    tokens = TOKEN_RE.findall(text.lower())
    if len(tokens) < k:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i:i + k]) for i in range(len(tokens) - k + 1)}

def make_minhash(text: str, num_perm: int = 128) -> MinHash:
    mh = MinHash(num_perm=num_perm)
    for sh in shingles(text):
        mh.update(sh.encode("utf-8"))
    return mh

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(IN_PATH))
    parser.add_argument("--threshold", type=float, default=0.85, help="Jaccard similarity threshold for near-duplicate detection (0-1)")
    parser.add_argument("--num-perm",type=int, default=128)
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    report = {"rows_in": int(len(df))}

    # --------- Exact duplicates removal-------
    df["exact_key"] = df["text"].map(exact_key)
    before = len(df)
    df = df.drop_duplicates(subset="exact_key", keep="first").reset_index(drop=True)
    report["exact_duplicates_removed"] = before - len(df)


    # ---------Near duplicates removal using MinHash and LSH-------
    lsh = MinHashLSH(threshold=args.threshold, num_perm=args.num_perm)
    minhashes = {}
    to_drop = set()

    for idx, text in enumerate(df["text"]):
        mh = make_minhash(text, args.num_perm)
        key = str(idx)

        similar = lsh.query(mh)
        if similar:
            to_drop.add(idx)
            continue
        lsh.insert(key, mh)
        minhashes[key] = mh

    report["near_duplicates_removed"] = len(to_drop)
    df = df.drop(columns=["exact_key"])

    report["rows_out"] = int(len(df))
    report["threshold_used"] = args.threshold

    OUT_PATH.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH / "tickets_dedup.parquet", index=False)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "tickets_dedup_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report,indent=2, ensure_ascii=False))
    





if __name__ == "__main__":
    main()
