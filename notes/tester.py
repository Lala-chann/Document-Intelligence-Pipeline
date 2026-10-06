#Figuring out if regex works correctly for email extraction

import hashlib
import re
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# TODO: 1
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
test = "Contact me at john.doe@example.com please"
print(EMAIL_RE.findall(test)) 


#TODO: 2

df = pd.read_parquet(PROJECT_ROOT / "data" / "raw" / "tickets_raw.parquet")
df = df[df["language"] == "en"]
PLACEHOLDER_RE = re.compile(r"\{\{.*?\}\}|\{[A-Za-z_]+\}|\[[A-Za-z _]+\]")

count = 0
for t in df["body"].dropna():
    matches = PLACEHOLDER_RE.findall(t)
    if matches:
        print(matches, "<-", t[:80])
        count += 1
    if count >= 10:
        break


#TODO: 3

df = pd.read_parquet(PROJECT_ROOT / "data" / "cleaned" / "tickets_clean.parquet")
contains_placeholder = df["text"].str.contains(r"\[Your Name\]", regex=True, na=False)
print("Number of '[Your Name]' :", contains_placeholder.sum())

print(df["text"].iloc[0][:300])



#TODO: 4
df = pd.read_parquet(PROJECT_ROOT / "data" / "cleaned" / "tickets_clean.parquet")

def exact_key(text):
    norm = " ".join(text.lower().split())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()

df["_key"] = df["text"].map(exact_key)


grouped = df.groupby("_key")["priority"].nunique()
conflicting = grouped[grouped > 1]

print(f"Number of duplicate groups: {df['_key'].duplicated().sum()}")
print(f"Number of conflicting groups (different priorities): {len(conflicting)}")

# Bir nece ziddiyyetli numuneye bax
if len(conflicting) > 0:
    sample_key = conflicting.index[0]
    print(df[df["_key"] == sample_key][["text", "priority"]].head())

#TODO 5:

import pandas as pd

val_df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "val.parquet")

high_df = val_df[val_df["priority"] == "high"].sample(5, random_state=42)
for t in high_df["text"]:
    print(t[:200], "\n---")

