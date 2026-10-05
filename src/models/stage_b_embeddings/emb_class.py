import sys
import pandas as pd
import numpy as np
import argparse

from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT / "src"))
from models.common.eval_utils import evaluate, compare_stages


TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.parquet"
VAL_PATH = PROJECT_ROOT / "data" / "processed" / "val.parquet"
REPORT_PATH = PROJECT_ROOT / "reports_for_models" / "stage_b"
EMBED_CACHE_PATH = PROJECT_ROOT / "data" / "processed" / "_embed_cache"

MODEL_NAME = "all-MiniLM-L6-v2"

def build_input_text(row) -> str:
    return f"Support queue: {row['queue']}. {row['text']}"

def get_embeddings(texts: pd.Series, model: SentenceTransformer, cache_name: str) -> np.ndarray:
    cache_path = EMBED_CACHE_PATH / f"{cache_name}.npy"
    if cache_path.exists():
        print(f"Reading cached embeddings: {cache_path}")
        return np.load(cache_path)
    print(f"Calculating embeddings ({len(texts)} texts)...")
    embeddings = model.encode(texts.tolist(), show_progress_bar=True, batch_size=64)

    EMBED_CACHE_PATH.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeddings)
    print(f"Embeddings cached: {cache_path}")
    return embeddings

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier", choices=["svm", "logreg"], default="svm")
    args = parser.parse_args()

    train_df = pd.read_parquet(TRAIN_PATH)
    val_df = pd.read_parquet(VAL_PATH)

    x_train_text = train_df.apply(build_input_text, axis=1)
    x_val_text = val_df.apply(build_input_text, axis=1)
    y_train, y_val = train_df["priority"], val_df["priority"]

    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    x_train = get_embeddings(x_train_text, model, "train_embeddings")
    x_val = get_embeddings(x_val_text, model, "val_embeddings")

    if args.classifier == "svm":
        clf = LinearSVC(class_weight=None, max_iter = 5000)
        stage_name = "B_embeddings_svm"
    else:
        clf = LogisticRegression(max_iter=1000, class_weight=None)
        stage_name = "B_embeddings_logreg"

    clf.fit(x_train, y_train)
    preds = clf.predict(x_val)

    evaluate(y_val, preds, stage_name=stage_name, report_dir=REPORT_PATH)


if __name__ == "__main__":
    main()
