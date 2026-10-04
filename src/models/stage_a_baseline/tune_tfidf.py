import sys
import pandas as pd

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT / "src"))
from models.common.eval_utils import evaluate, compare_stages


TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.parquet"
VAL_PATH = PROJECT_ROOT / "data" / "processed" / "val.parquet"
REPORT_PATH = PROJECT_ROOT / "reports_for_models" / "stage_a" / "A1_prosses"

CLASS_WEIGHT = None


CONFIGS = [
    {"name": "A1_base_5k_unigram",   "max_features": 5000,  "ngram_range": (1, 1)},
    {"name": "A1_base_20k_unigram",  "max_features": 20000, "ngram_range": (1, 1)},
    {"name": "A1_20k_bigram",        "max_features": 20000, "ngram_range": (1, 2)},
    {"name": "A1_40k_bigram",        "max_features": 40000, "ngram_range": (1, 2)},
    {"name": "A1_20k_trigram",       "max_features": 20000, "ngram_range": (1, 3)},
]

def add_queue_prefix(row) -> str:
    queue_token = "__queue_" + row["queue"].replace(" ", "_").replace("&", "and") + "__"
    return f"{queue_token} {row['text']}"

def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    val_df = pd.read_parquet(VAL_PATH)

    x_train = train_df.apply(add_queue_prefix, axis=1)
    x_val = val_df.apply(add_queue_prefix, axis=1)
    y_train, y_val = train_df["priority"], val_df["priority"]

    summaries = []

    for cfg in CONFIGS:
        vec = TfidfVectorizer(max_features=cfg["max_features"], ngram_range=cfg["ngram_range"])
        xtr = vec.fit_transform(x_train)
        xva = vec.transform(x_val)

        clf = LogisticRegression(max_iter=1000, class_weight=CLASS_WEIGHT)
        clf.fit(xtr, y_train)
        preds = clf.predict(xva)

        summary = evaluate(y_val, preds, stage_name=cfg["name"], report_dir=REPORT_PATH)
        summaries.append(summary)
    compare_stages(summaries, REPORT_PATH / "A1_comparison.json")

if __name__ == "__main__":
    main()