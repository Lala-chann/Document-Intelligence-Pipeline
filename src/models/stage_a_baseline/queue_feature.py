import sys
import pandas as pd

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT / "src"))
from models.common.eval_utils import evaluate

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.parquet"
VAL_PATH = PROJECT_ROOT / "data" / "processed" / "val.parquet"
REPORT_PATH = PROJECT_ROOT / "reports_for_models" / "stage_a" / "A2_prosses"

CLASS_WEIGHT = None

def add_queue_prefix(row) -> str:
    queue_token = "__queue_" + row["queue"].replace(" ", "_").replace("&", "and") + "__"
    return f"{queue_token} {row['text']}"


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    val_df = pd.read_parquet(VAL_PATH)

    x_train = train_df.apply(add_queue_prefix, axis=1)
    x_val = val_df.apply(add_queue_prefix, axis=1)
    y_train, y_val = train_df["priority"], val_df["priority"]

    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    xtr = vec.fit_transform(x_train)
    xva = vec.transform(x_val)

    clf = LogisticRegression(max_iter=1000, class_weight=CLASS_WEIGHT)
    clf.fit(xtr, y_train)
    preds = clf.predict(xva)

    evaluate(y_val,preds, stage_name="A2_text_plus_queue", report_dir=REPORT_PATH)

    


if __name__ == "__main__":
    main()