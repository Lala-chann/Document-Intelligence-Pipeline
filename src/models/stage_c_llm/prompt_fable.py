import argparse
import os
import sys
import time
import pandas as pd

from pathlib import Path
from anthropic import Anthropic, AuthenticationError
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env", override=True)
sys.path.append(str(PROJECT_ROOT / "src"))
from models.common.eval_utils import evaluate, compare_stages

VAL_PATH = PROJECT_ROOT / "data" / "processed" / "val.parquet"
REPORT_PATH = PROJECT_ROOT / "reports_for_models" / "stage_c"

MODEL = "claude-fable-5-1"
LABELS = ["low", "medium", "high"]

SYSTEM_PROMT = (
    "You are an assistant that classifies support tickets by "
    "priority. For each ticket, respond with ONLY one of the following "
    "three words, nothing else: low, medium, high.\n\n"
    "high: urgent, service outage, security issue, "
    "affects multiple users.\n"
    "medium: normal workflow issue, not urgent but needs attention.\n"
    "low: general question, information request, not urgent."
)

def classify_one(client: Anthropic, queue: str, text:str) -> str:
    user_msg = f"Support category: {queue}\n\n Ticket text: {text[:1500]}"

    # Fable always thinks, so max_tokens must leave room for thinking + answer
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=2048,
        system=SYSTEM_PROMT,
        output_config={"effort": "low"},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[{"role": "user", "content": user_msg}]
    )
    if response.stop_reason == "refusal":
        return "medium"

    # content[0] is a thinking block on Fable - collect only the text blocks
    raw = "".join(
        block.text for block in response.content if block.type == "text"
    ).strip().lower()

    for label in LABELS:
        if label in raw:
            return label
    return "medium"


def main():
    parser  = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()


    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not found"
        )
    client = Anthropic(api_key=api_key)
    try:
        client.models.list(limit=1)
    except AuthenticationError:
        sys.exit(
            "API key rejected - create a new key in console.anthropic.com "
            "and update ANTHROPIC_API_KEY in .env"
        )


    val_df = pd.read_parquet(VAL_PATH)
    sample_df, _ = train_test_split(
        val_df,
        train_size = args.sample_size,
        random_state=args.seed,
        stratify=val_df["priority"],
    )
    print(f"Sample size: {len(sample_df)}")

    predictions = []
    for i, (_, row) in enumerate(sample_df.iterrows()):
        pred = classify_one(client, row["queue"], row["text"])
        predictions.append(pred)

        if(i + 1) % 25 == 0:
            print(f"{i + 1}/{len(sample_df)} completed...")
        time.sleep(0.1)

    evaluate(
        sample_df["priority"], pd.Series(predictions),
        stage_name="C_claude_fable_zeroshot", report_dir=REPORT_PATH,
    )


if __name__ == "__main__":
    main()

