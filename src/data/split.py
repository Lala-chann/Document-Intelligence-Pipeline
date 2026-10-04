import argparse
import json
import pandas as pd

from pathlib import Path 
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IN_PATH = PROJECT_ROOT / "data" / "cleaned" / "tickets_dedup.parquet"
OUT_PATH = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports_for_data"

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(IN_PATH))
    parser.add_argument("--target", default="priority", help="Target column for stratification")
    parser.add_argument("--train-size", type=float, default=0.70)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()



    assert abs(args.train_size + args.val_size + args.test_size - 1.0) < 1e-6, \
        "train+val+test sizes must sum to 1.0"

    df = pd.read_parquet(args.input)
    report = {"rows_in": int(len(df)), "target": args.target, "seed": args.seed}

    temp_size = args.val_size + args.test_size
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_size,
        random_state=args.seed,
        stratify=df[args.target]
    )

    # train_test_split only makes 2 parts, so split twice:
    # first 70/30 (train/temp), then temp 50/50 -> val 15% / test 15%

    rel_test_size = args.test_size / temp_size
    val_df, test_df = train_test_split(
        temp_df,
        test_size=rel_test_size,
        random_state=args.seed,
        stratify=temp_df[args.target],
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    report["rows_train"] = int(len(train_df))
    report["rows_val"] = int(len(val_df))
    report["rows_test"] = int(len(test_df))

    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        report[f"{name}_{args.target}_distribution"] = (
            part[args.target].value_counts(normalize=True).round(3).to_dict()
        )

    OUT_PATH.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(OUT_PATH / "train.parquet", index=False)
    val_df.to_parquet(OUT_PATH / "val.parquet", index=False)
    test_df.to_parquet(OUT_PATH / "test.parquet", index=False)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "split.report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()