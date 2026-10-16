"""
inference.py her iki model you ucun proqnoz mentiqi
"""

import os
import joblib

from pathlib import Path
from anthropic import Anthropic
from dotenv import load_dotenv

from src.shared.text_utils import build_text

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

STORE_DIR = PROJECT_ROOT / "models_store"
CLAUDE_MODEL = "claude-haiku-4-5"
LABELS = ["low","medium","high"]

SYSTEM_PROMT = (
    
    "You are an assistant that classifies customer support tickets by "
    "priority. For each ticket, respond with ONLY one of the following "
    "three words, nothing else: low, medium, high.\n\n"
    "high: urgent, service outage, security issue, "
    "affects multiple users.\n"
    "medium: normal workflow issue, not urgent but needs attention.\n"
    "low: general question, information request, not urgent."
)

_vectorizer = None
_classifier = None
_claude_client  = None

def load_svm_model():
    global _vectorizer, _classifier
    if _vectorizer is None:
        _vectorizer = joblib.load(STORE_DIR / "tfidf_vectorizer.joblib")
        _classifier = joblib.load(STORE_DIR / "svm_classifier.joblib")
    return _vectorizer, _classifier

def get_claude_client() -> Anthropic:
    global _claude_client
    if _claude_client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")

        if not api_key:
            raise RuntimeError("API key is not defined")

        _claude_client = Anthropic(api_key=api_key)
    return _claude_client

def predict_svm(subject: str, body: str, queue: str) -> str:
    vectorizer, classifier = load_svm_model()
    queue_token = "__queue_" + queue.replace(" ", "_").replace("&", "and") + "__"

    text_input = f"{queue_token} {build_text(subject, body)}"

    X = vectorizer.transform([text_input])
    pred = classifier.predict(X)[0]
    return str(pred)


def predict_claude(subject: str, body: str, queue: str) -> str:
    client = get_claude_client()
    text = f"{subject}\n\n{body}"
    user_msg = f"Support category: {queue}\n\n Ticket text: {text[:1500]}"

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=10,
        system=SYSTEM_PROMT,
        messages=[{"role": "user", "content": user_msg}],
    )
    raw = response.content[0].text.strip().lower()

    for label in LABELS:
        if label in raw:
            return label
    return "medium"


def predict(subject:str, body:str, queue:str, model:str) -> str:
    if model == "svm":
        return predict_svm(subject,body,queue)
    elif model == "claude":
        return predict_claude(subject,body,queue)
    raise ValueError(f"Model is not defined: {model}")

