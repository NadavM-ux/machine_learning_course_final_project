"""
Step 8-B (LLM comparison) — scoring + majority vote.
============================================================================
For each LLM (Claude Opus) run i (1..N):
  * reads  Iteration_i/chatgpt_predictions_iteration_i.csv
  * scores it against llm_test_set_groundtruth.csv on all 3 tasks
  * writes Iteration_i/chatgpt_performance_iteration_i.csv
Then across all runs:
  * majority vote per user per task -> Majority_iterations/chatgpt_predictions_majority_iterations.csv
  * the vote fraction gives a per-class probability, so ROC-AUC is defined for
    the majority result too -> Majority_iterations/chatgpt_performance_majority_iterations.csv

Metrics (macro-averaged over classes, as the tasks are multi-class):
  accuracy, precision, recall, F1, AUC.
AUC per single run is left blank (a hard label has no probability); it is only
computed for the majority result, where the vote fraction is the probability.

Run:  python score_llm_runs.py            # scores every Iteration_* it finds
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)
from sklearn.preprocessing import label_binarize

HERE = Path(__file__).resolve().parent
GT_PATH = HERE / "llm_test_set_groundtruth.csv"
TASKS = ["target_population", "locals_vs_diaspora", "person_vs_organization"]
CLASSES = [0, 1, 2]


def _basic_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def _auc_from_proba(y_true, proba) -> float | str:
    """Macro OVR AUC; needs >=2 classes actually present in y_true."""
    present = sorted(set(y_true))
    if len(present) < 2:
        return ""
    yb = label_binarize(y_true, classes=CLASSES)
    cols = [CLASSES.index(c) for c in present]
    try:
        return roc_auc_score(yb[:, cols], proba[:, cols], average="macro")
    except ValueError:
        return ""


def _iter_run_dirs():
    for d in sorted(HERE.glob("Iteration_*"), key=lambda p: int(p.name.split("_")[1])):
        n = int(d.name.split("_")[1])
        f = d / f"chatgpt_predictions_iteration_{n}.csv"
        if f.exists():
            yield n, d, f


def main() -> None:
    gt = pd.read_csv(GT_PATH).set_index("username")
    order = gt.index.tolist()

    per_run_preds: dict[int, pd.DataFrame] = {}

    for n, d, f in _iter_run_dirs():
        pred = pd.read_csv(f).drop_duplicates("username").set_index("username")
        pred = pred.reindex(order)  # align to ground-truth order
        per_run_preds[n] = pred

        rows = []
        for task in TASKS:
            mask = gt[task].notna() & pred[task].notna()
            yt = gt.loc[mask, task].astype(int)
            yp = pred.loc[mask, task].astype(int)
            m = {"iteration": n, "target_column": task, "n": int(mask.sum())}
            m.update(_basic_metrics(yt, yp))
            m["AUC"] = ""  # single hard-label run has no probability
            rows.append(m)
        out = pd.DataFrame(rows)
        out.to_csv(d / f"chatgpt_performance_iteration_{n}.csv", index=False)
        print(f"scored run {n}: {f.name}")

    if not per_run_preds:
        raise SystemExit("no chatgpt_predictions_iteration_*.csv found yet — run the LLM (Claude) first.")

    # ---- majority vote across runs ----
    runs = sorted(per_run_preds)
    maj = pd.DataFrame(index=order)
    maj.index.name = "username"
    perf_rows = []
    for task in TASKS:
        stacked = pd.concat([per_run_preds[r][task] for r in runs], axis=1)
        # per-class vote fraction -> probability matrix
        proba = np.zeros((len(order), len(CLASSES)))
        for ci, c in enumerate(CLASSES):
            proba[:, ci] = (stacked == c).sum(axis=1).values / stacked.notna().sum(axis=1).values
        maj[task] = [CLASSES[i] for i in proba.argmax(axis=1)]

        mask = gt[task].notna()
        yt = gt.loc[mask, task].astype(int).values
        yp = maj.loc[mask, task].astype(int).values
        row = {"iteration": "majority", "target_column": task, "n": int(mask.sum())}
        row.update(_basic_metrics(yt, yp))
        row["AUC"] = _auc_from_proba(yt, proba[mask.values])
        perf_rows.append(row)

    mdir = HERE / "Majority_iterations"
    mdir.mkdir(exist_ok=True)
    maj.reset_index().to_csv(mdir / "chatgpt_predictions_majority_iterations.csv", index=False)
    pd.DataFrame(perf_rows).to_csv(mdir / "chatgpt_performance_majority_iterations.csv", index=False)

    print(f"\nmajority over runs {runs}")
    print(pd.DataFrame(perf_rows).to_string(index=False))


if __name__ == "__main__":
    main()
