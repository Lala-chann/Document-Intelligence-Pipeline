import json
import pandas as pd

from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports_for_models"

LABELS = ["low", "medium", "high"]
MOST_EXP_ERROR = ("high", "low")

def evaluate(y_true, y_pred, stage_name: str, report_dir: Path = DEFAULT_REPORT_DIR) -> dict:
    report = classification_report(
        y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    cm_df = pd.DataFrame(
        cm,
        index=[f"actual_{lbl}" for lbl in LABELS],
        columns=[f"pred_{lbl}" for lbl in LABELS]
    )
    actual_idx = LABELS.index(MOST_EXP_ERROR[0])
    pred_idx = LABELS.index(MOST_EXP_ERROR[1])
    exp_count = int(cm[actual_idx, pred_idx])
    actual_total = int(cm[actual_idx].sum())
    exp_rate = round(exp_count / actual_total, 3) if actual_total else None

    summary = {
        "stage" : stage_name,
        "accuracy": round(accuracy_score(y_true, y_pred), 3),
        "macro_f1" : round(report["macro avg"]["f1-score"], 3),
        "most_exp_error" : {
            "description" : f"actual={MOST_EXP_ERROR[0]} -> pred={MOST_EXP_ERROR[1]}",
            "count" : exp_count,
            "rate" : exp_rate
        },
    }

    print(f"\n===== {stage_name} =====")
    print(cm_df)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    report_dir.mkdir(parents=True, exist_ok=True)
    out_path = report_dir / f"{stage_name}.json"
    out_path.write_text(json.dumps({
        "summary": summary,
        "full_report" : report,
        "confusion_matrix": cm_df.to_dict()},
        indent=2, ensure_ascii=False
        ), encoding="utf-8")
    print(f"Report saved: {out_path}")

    return summary

def compare_stages(summaries: list[dict], out_path: Path) -> None:
    table = pd.json_normalize(summaries)
    print(table.to_string(index=False))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        table.to_json(orient="records", indent=2, force_ascii=False),
        encoding="utf-8"
    )
    