import sys
import joblib
import pandas as pd

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.parquet"
STORE_DIR = PROJECT_ROOT / "models_store"


MAX_FEATURES = 40000
NGRAM = (1,2)

def add_queue_prefix(row) -> str:
    queue_token = "__queue_" + row["queue"].replace(" ", "_").replace("&", "and") + "__"
    return f"{queue_token} {row['text']}"

def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    X_train = train_df.apply(add_queue_prefix, axis=1)
    y_train = train_df["priority"]
 
    vec = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=NGRAM)
    Xtr = vec.fit_transform(X_train)
 
    clf = LinearSVC(class_weight=None, max_iter=5000)
    clf.fit(Xtr, y_train)
 
    STORE_DIR.mkdir(exist_ok=True)
    joblib.dump(vec, STORE_DIR / "tfidf_vectorizer.joblib")
    joblib.dump(clf, STORE_DIR / "svm_classifier.joblib")
    print(f"Model saxlanildi: {STORE_DIR}/")
 
 
if __name__ == "__main__":
    main()