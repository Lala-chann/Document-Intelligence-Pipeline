import argparse
import json
import re
import unicodedata
import pandas as pd

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "tickets_raw.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data" / "cleaned"
REPORT_DIR = PROJECT_ROOT / "reports_for_data"

KEEP_COLS = ["subject", "body", "type", "queue", "priority", "language"]
HTML_RE = re.compile(r"<[^>]+>")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"\+?\d[\d\s().-]{7,}\d")
PLACEHOLDER_RE = re.compile(r"\{\{.*?\}\}|\{[A-Za-z_]+\}|\[[A-Za-z _]+\]")
SPACES_RE = re.compile(r"[ \t]+")
BLANK_LINES_RE = re.compile(r"\n{3,}")


def normalize(text: str, stats: dict) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\\r\\n", "\n").replace("\\n", "\n")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = HTML_RE.sub(" ", text)
 
    stats["emails_masked"] += len(EMAIL_RE.findall(text))
    text = EMAIL_RE.sub("<EMAIL>", text)
 
    stats["phones_masked"] += len(PHONE_RE.findall(text))
    text = PHONE_RE.sub("<PHONE>", text)
 
    stats["placeholder_like_found"] += len(PLACEHOLDER_RE.findall(text))
    text = PLACEHOLDER_RE.sub(" ", text)
    
    text = SPACES_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(RAW_PATH))
    parser.add_argument("--language", default=None, help="mes: en")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    report = {"rows_in": int(len(df))}

    if args.language:
        df=df[df["language"] == args.language]
    report["rows_after_language_filter"] = int(len(df))


    df = df[KEEP_COLS].copy()
    df["subject"] = df["subject"].fillna("")
    df["body"] = df["body"].fillna("")

    stats = {
        "emails_masked": 0,
        "phones_masked": 0,
        "placeholder_like_found": 0,
    }

    has_placeholder = (df["subject"] + " " + df["body"]).str.contains(PLACEHOLDER_RE)
    report["tickets_with_placeholder"] = int(has_placeholder.sum())

    df["subject"] = df["subject"].map(lambda s: normalize(s, stats))
    df["body"] = df["body"].map(lambda s: normalize(s, stats))

    empty_body = df["body"] == ""
    report["dropped_empty_body"] = int(empty_body.sum())
    df = df[~empty_body]

    df["text"] = df["subject"] + "\n\n" + df["body"]
    df = df.reset_index(drop=True)

    report["rows_out"] = int(len(df))
    report.update(stats)
    report["target_nulls"] = {
        c: int(df[c].isna().sum()) for c in ["type", "queue", "priority"]
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_DIR / "tickets_clean.parquet", index=False)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "tickets_clean.report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))





if __name__=="__main__":
    main()

