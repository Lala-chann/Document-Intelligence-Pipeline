import json
import pandas as pd

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.parquet"
VAL_PATH = PROJECT_ROOT / "data" / "processed" / "val.parquet"
REPORT_DIR = PROJECT_ROOT / "reports_for_models"

def run_one(x_train, y_train, x_val, y_val, class_weight=None):
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
    xtr = vec.fit_transform(x_train)
    xva = vec.transform(x_val)

    clf = LogisticRegression(max_iter=10000, class_weight=class_weight)
    clf.fit(xtr, y_train)
    preds = clf.predict(xva)

    return classification_report(y_val, preds, output_dict=True), preds

def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    val_df = pd.read_parquet(VAL_PATH)

    x_train,y_train = train_df["text"], train_df["priority"]
    x_val, y_val = val_df["text"], val_df["priority"]

    results = {}
    for label, cw in [("without_balanced", None), ("with_balanced", "balanced")]:
        print(f"\n=== {label} (class_weight={cw}) ===")
        report, preds = run_one(x_train, y_train, x_val, y_val, cw)
        results[label] = report

        cm = confusion_matrix(y_val, preds, labels=["low", "medium", "high"])
        print(pd.DataFrame(cm, index=["actual_low", "actual_medium", "actual_high"],
                            columns=["pred_low", "pred_medium", "pred_high"]))

    summary = {
        label: {
            "accuracy" : round(r["accuracy"], 3),
            "macro_f1" : round(r["macro avg"]["f1-score"], 3),
            "low_recall" : round(r["low"]["recall"],3),
            "low_precision" : round(r["low"]["precision"], 3),
        }
        for label, r in results.items()
    }
    print("\n=== Comparison ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "balanced_comparison.json").write_text(
        json.dumps({"full_reports" : results, "summary": summary},
                   indent=2 , ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nReport Saved: {REPORT_DIR / 'balanced_comparison.json'}")

if __name__ == "__main__":
    main()

