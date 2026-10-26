import argparse
import hashlib
import json
import sys
import joblib
import sklearn

import numpy as np
import pandas as pd

from datetime import datetime, timezone
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.train_final_model import add_queue_prefix


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STORE_DIR = PROJECT_ROOT / "models_store"
VAL_PATH = PROJECT_ROOT / "data" / "processed" / "val.parquet"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.parquet"
RAW_META = PROJECT_ROOT / "data" / "raw" / "tickets_raw.meta.json"
REPORT_PATH = PROJECT_ROOT / "reports_for_models" / "final" / "test_evaluation.json"
LABELS = ["low","medium","high"]


def sha12(path:Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]

def predict_df(df,vec,clf):
    return clf.predict(vec.transform(df.apply(add_queue_prefix,axis=1)))

def summarize(y_true: pd.Series, y_pred) -> dict:
    cm = confusion_matrix(y_true,y_pred,labels=LABELS)
    high = LABELS.index("high")
    n_high = cm[high].sum()

    return {
        "n" : int(len(y_true)),
        "accuracy" : round(float(accuracy_score(y_true,y_pred)),3),
        "macro_f1" : round(float(f1_score(y_true,y_pred,labels=LABELS,average="macro", zero_division=0)),3),
        "high_to_low_rate": round(float(cm[high, LABELS.index("low")] / n_high),3),
        "high_to_medium_rate": round(float(cm[high, LABELS.index("medium")] / n_high),3),
        "majority_baseline_accuracy": round(float(y_true.value_counts(normalize=True).max()),3),
        "confusion_matrix": {"labels" : LABELS, "rows_actual_cols_pred": cm.tolist()},
        "per_class": classification_report(y_true,y_pred,labels=LABELS, output_dict=True, zero_division=0),
    }

def bootstrap_ci(y_true, y_pred, n_resamples: int, seed:int) -> dict:
    rng = np.random.default_rng(seed)
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    n = len(yt)
    f1s, accs = [], []
    for _ in range(n_resamples):
        i = rng.integers(0,n,n)
        f1s.append(f1_score(yt[i], yp[i], labels=LABELS, average="macro", zero_division=0))
        accs.append((yt[i] == yp[i]).mean())
    q=lambda a: [round(float(x),3) for x in np.percentile(a,[2.5, 97.5])]
    return {"macro_f1_95ci": q(f1s), "accuracy_95ci": q(accs)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm-frozen", action="store_true", help="Model has been frozen and is ready for testing")
    ap.add_argument("--force", action="store_true", help="Overwrite the existing output file")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()


    if not args.confirm_frozen:
        sys.exit("STOPPED: --confirm-frozen is missing. Do not run or review the tests unless the model has been frozen.")
    if REPORT_PATH.exists() and not args.force:
        sys.exit(f"STOPPED: {REPORT_PATH} already exists. Looking at the test set a second time makes the results invalid.")


    vec = joblib.load(STORE_DIR / "tfidf_vectorizer.joblib")
    clf = joblib.load(STORE_DIR / "svm_classifier.joblib")
    val_df, test_df = pd.read_parquet(VAL_PATH), pd.read_parquet(TEST_PATH)

    val = summarize(val_df["priority"], predict_df(val_df, vec, clf))
    test_pred = predict_df(test_df, vec, clf)
    test = summarize(test_df["priority"], test_pred)
    test.update(bootstrap_ci(test_df["priority"], test_pred, args.bootstrap, args.seed))


    provenance = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "model_files_sha256_12": {
            "tfidf_vectorizer": sha12(STORE_DIR / "tfidf_vectorizer.joblib"),
            "svm_classifier": sha12(STORE_DIR / "svm_classifier.joblib"),
        },
        "data_revision": json.loads(RAW_META.read_text()).get("revision") if RAW_META.exists() else None,
        "bootstrap_resamples": args.bootstrap,
        "seed": args.seed,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps({"provenance": provenance, "val": val, "test": test}, indent=2))
 
    print("split  n      accuracy  macro_f1  high->low  high->medium")
    for name, s in (("val", val), ("test", test)):
        print(f"{name:6} {s['n']:<6} {s['accuracy']:<9.3f} {s['macro_f1']:<9.3f} "
              f"{s['high_to_low_rate']:<10.1%} {s['high_to_medium_rate']:.1%}")
    print(f"\ntest macro_f1 95% interval : {test['macro_f1_95ci']}")
    print(f"test accuracy 95% interval : {test['accuracy_95ci']}")
    print(f"Baseline (always predicting the most frequent class): accuracy = {test['majority_baseline_accuracy']}")
    print(f"\nReport written: {REPORT_PATH}")
 
 
if __name__ == "__main__":
    main()