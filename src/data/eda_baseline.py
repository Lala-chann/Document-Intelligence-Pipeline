import json
import pandas as pd
import argparse

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "tickets_raw.parquet"
REPORT_DIR = PROJECT_ROOT / "reports_for_data"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="priority", choices=["priority", "type", "queue"])
    parser.add_argument("--language", default=None)
    args = parser.parse_args()

    # For ex: python script.py --target type --language en



    df = pd.read_parquet(RAW_PATH)
    if args.language:
        df= df[df["language"] == args.language]

    """
    ⚠️ Argparse reads the arguments every time the program runs. 
    If you set --language en once, it wont automatically stay “en” for the next run. 
    You need to provide the argument again in the terminal each time you execute the program.

    """
    print("--- Data Overview ---")
    for col in ["language", "type", "queue", "priority"]:
        print(f"\n--- {col} ---")
        print(df[col].value_counts(dropna=False))

    print(f"\n--- Length of body ---")
    print(df["body"].str.len().describe())

    df = df.dropna(subset=[args.target, "subject" ,"body"])
    df["text"] = df["subject"].fillna("") + " " + df["body"].fillna("")

    x_train, x_test, y_train, y_test = train_test_split(
        df["text"], df[args.target], 
        test_size=0.2, 
        random_state=42,
        stratify=df[args.target],

    )

    vec = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
    xtr = vec.fit_transform(x_train)
    xte = vec.transform(x_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(xtr, y_train)
    preds = clf.predict(xte)

    report = classification_report(y_test, preds, output_dict=True)
    print(f"\n--- Classification Report = {args.target} ---")
    print(classification_report(y_test, preds))

    """
    ⚠️ The classification_report compares the true labels (y_test) with the predictions (preds). 
    In other words, to measure how well the model performs, we must compare against the real answers. 
    x_test is only the input texts, so it cannot be used for evaluation.

    """

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORT_DIR / f"baseline_{args.target}.json"
    out_path.write_text(json.dumps(report,indent=2, ensure_ascii=False))
    print(f"Classification Report saved to {out_path}")



if __name__ == "__main__":
    main()
