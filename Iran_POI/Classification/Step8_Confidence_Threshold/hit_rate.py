"""
Step 8-A — Hit Rate over the FULL high-confidence target population.
============================================================================
The PDF asks: sample 100 users with confidence >= threshold and manually check
what fraction truly belong to the target population (Hit Rate). Here the whole
high-confidence population is only 173 users, and truth is ALREADY known for
every one of them:

  * 42  were manually validated for step 8 (confidence_validation_labeled.csv),
        and were NOT in the training set  -> an UNBIASED generalization estimate.
  * 131 were labeled earlier during active learning (they sit in the iteration-7
        training set) -> truth known, but the model has SEEN them (optimistic).

So no further manual labeling is possible or needed. This script aggregates the
existing human labels and reports the Hit Rate for three strata plus a literal
random-100 sample (PDF wording), so nothing is hidden:

  unseen_42   conservative, held-out estimate (what generalizes)
  train_131   already-labeled slice of the population
  full_173    the ACTUAL composition of the deployed target population
  sample_100  a random 100 of the 173 (seed 42) — literal PDF ask

Hit categories (predicted target, prob_1 >= threshold):
  true==1 -> hit          (truly target)
  true==0 -> false_pos    (truly non-target)
  true==2 -> unverifiable (suspended / no info / unknown)
  hit_rate_strict     = hits / total          (unknown counts as miss)
  hit_rate_verifiable = hits / (hits+false_pos) (drop unverifiable)

Run:  python hit_rate.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
CLS = HERE.parent
sys.path.insert(0, str(CLS))
import active_learning as al          # noqa: E402

THRESHOLD = 0.80
FINAL_ITER = 7
SEED = 42
PASS_BAR = 0.85


def _rates(truth: pd.Series) -> dict:
    hits = int((truth == 1).sum())
    false_pos = int((truth == 0).sum())
    unver = int((truth == 2).sum())
    total = hits + false_pos + unver
    strict = hits / total if total else float("nan")
    verifiable = hits / (hits + false_pos) if (hits + false_pos) else float("nan")
    return {
        "n": total, "hits_true_target": hits, "false_positives": false_pos,
        "unverifiable": unver, "hit_rate_strict": round(strict, 4),
        "hit_rate_verifiable": round(verifiable, 4), "pass_85pct": bool(strict >= PASS_BAR),
    }


def main() -> None:
    pred = pd.read_csv(HERE / "final_model_predictions.csv")
    hi = pred[pred["prob_1"] >= THRESHOLD].copy()
    hi["k"] = al._norm_user(hi["username"])

    # truth source 1: the 42 manually validated (held out)
    val = pd.read_csv(HERE / "confidence_validation_labeled.csv")
    val["k"] = al._norm_user(val["username"])
    val_truth = val.set_index("k")["true"]
    val_comments = val.set_index("k")["comments"]

    # truth source 2: the iteration-7 training labels (seen)
    lab = pd.read_csv(al.combined_path(FINAL_ITER))
    lab["k"] = al._norm_user(lab["username"])
    lab_truth = pd.to_numeric(lab.set_index("k")["target_population"], errors="coerce")

    # assemble truth + provenance for all 173
    def truth_of(k):
        if k in val_truth.index:
            return val_truth.loc[k], "validation_unseen"
        if k in lab_truth.index:
            return lab_truth.loc[k], "train_labeled"
        return pd.NA, "MISSING"

    tp = hi["k"].map(lambda k: truth_of(k))
    hi["true"] = [t[0] for t in tp]
    hi["source"] = [t[1] for t in tp]
    hi["comments"] = hi["k"].map(val_comments).fillna("")

    missing = hi[hi["source"] == "MISSING"]
    if len(missing):
        raise SystemExit(f"{len(missing)} high-conf users have no known truth: "
                         f"{missing['username'].tolist()[:5]}")
    hi["true"] = hi["true"].astype(int)
    hi["hit"] = (hi["true"] == 1).astype(int)

    out_cols = ["username", "predicted_class", "prob_1", "true", "hit", "source", "comments"]
    hi_sorted = hi.sort_values("prob_1", ascending=False)
    hi_sorted[out_cols].to_csv(HERE / "confidence_validation_full173_labeled.csv", index=False)

    # strata
    strata = {
        "unseen_42": hi[hi["source"] == "validation_unseen"]["true"],
        "train_131": hi[hi["source"] == "train_labeled"]["true"],
        "full_173": hi["true"],
        "sample_100": hi.sample(n=min(100, len(hi)), random_state=SEED)["true"],
    }
    rows = []
    for name, truth in strata.items():
        r = {"stratum": name, "threshold": THRESHOLD}
        r.update(_rates(truth))
        rows.append(r)
    summary = pd.DataFrame(rows)
    summary.to_csv(HERE / "hit_rate_summary.csv", index=False)

    print("=== Step 8-A Hit Rate — full high-confidence population (prob_target >= "
          f"{THRESHOLD}) ===")
    print(summary.to_string(index=False))
    print("\nwrote: hit_rate_summary.csv (4 strata)")
    print("       confidence_validation_full173_labeled.csv (all 173 with truth + source)")
    print("\nHEADLINE: the full 173-user target population is "
          f"{summary.loc[summary.stratum=='full_173','hit_rate_strict'].iat[0]*100:.1f}% strict / "
          f"{summary.loc[summary.stratum=='full_173','hit_rate_verifiable'].iat[0]*100:.1f}% verifiable target.")


if __name__ == "__main__":
    main()
