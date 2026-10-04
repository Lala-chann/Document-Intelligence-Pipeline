import sys
import pandas as pd

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT / "src"))
from models.common.eval_utils import evaluate, compare_stages


TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.parquet"
VAL_PATH = PROJECT_ROOT / "data" / "processed" / "val.parquet"
REPORT_DIR = PROJECT_ROOT / "reports_for_models" / "stage_a" / "A3_prosses"

MAX_FEATURES = 40000
NGRAM_RANGE = (1,2)

def add_queue_prefix(row) -> str:
    queue_token = "__queue_" + row["queue"].replace(" ", "_").replace("&", "and") + "__"
    return f"{queue_token} {row['text']}"

def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    val_df = pd.read_parquet(VAL_PATH)

    x_train = train_df.apply(add_queue_prefix, axis=1)
    x_val = val_df.apply(add_queue_prefix, axis=1)
    y_train, y_val = train_df["priority"], val_df["priority"]

    vec = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE)
    xtr = vec.fit_transform(x_train)
    xva = vec.transform(x_val)

    models = {
        "A3_logistic_regression": LogisticRegression(max_iter=1000, class_weight=None),
        "A3_linear_svm": LinearSVC(class_weight=None, max_iter=5000),
        "A3_naive_bayes": MultinomialNB()

    }
    summaries = []
    for name,clf in models.items():
        clf.fit(xtr, y_train)
        preds = clf.predict(xva)
        summary = evaluate(y_val, preds, stage_name=name, report_dir=REPORT_DIR)
        summaries.append(summary)

    compare_stages(summaries, REPORT_DIR / "A3_comparison.json")

if __name__ == "__main__":
    main()

