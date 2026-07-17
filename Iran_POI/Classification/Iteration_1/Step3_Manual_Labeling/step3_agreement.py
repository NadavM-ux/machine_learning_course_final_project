"""
Step 3 — Inter-annotator agreement (Cohen's Kappa) + consensus building.
==========================================================================

RECONSTRUCTED 2026-07-15. The original script that produced
`iteration_1_agreement_report.csv` and `iteration_1_labels_consensus.csv` was
not kept in the repo (only its outputs survived). This script was rebuilt by
reverse-engineering those outputs and was **validated to reproduce both files
byte-for-value exactly** (agreement report: 0.93/1.00/0.96 & kappa
0.8803/1.0000/0.9353; consensus: 93 auto_agree + 7 discussed, 0 label mismatches).

Setup — double annotation (per the PDF, שלב 3):
    The 100 iteration-1 users were each labelled by TWO annotators on three
    label columns (target_population, locals_vs_diaspora, person_vs_organization,
    all encoded 0/1/2). The two label passes sit side-by-side in
    `iteration_1_labels_merged_sidebyside.csv` as columns  *_NADAV  and *_NADAVIL.

What it computes:
    1. Per label column: Percent Agreement + Cohen's Kappa  ->  agreement_report.csv
    2. Consensus label per user:
         - both annotators agree  -> consensus_source = "auto_agree", value = agreed label
         - they disagree          -> consensus_source = "discussed",  value taken from the
                                      resolved `final_*` column in the disagreements file
       -> iteration_1_labels_consensus.csv

Run:
    python step3_agreement.py
"""
from pathlib import Path
import pandas as pd
from sklearn.metrics import cohen_kappa_score

HERE = Path(__file__).resolve().parent

LABEL_COLS = ["target_population", "locals_vs_diaspora", "person_vs_organization"]
ANNOTATOR_A, ANNOTATOR_B = "NADAV", "NADAVIL"          # the two label passes

# Recommended kappa targets from the PDF (for a printed sanity check only).
KAPPA_TARGETS = {
    "target_population": 0.75,
    "locals_vs_diaspora": 0.60,
    "person_vs_organization": 0.60,
}

MERGED_FILE       = HERE / "iteration_1_labels_merged_sidebyside.csv"
DISAGREEMENTS_FILE = HERE / "iteration_1_disagreements.csv"
AGREEMENT_OUT     = HERE / "iteration_1_agreement_report.csv"
CONSENSUS_OUT     = HERE / "iteration_1_labels_consensus.csv"

META_COLS = ["username", "profile_url", "display_name", "description", "location",
             "followers_count", "following_count", "statuses_count", "created_at"]


def agreement_report(merged: pd.DataFrame) -> pd.DataFrame:
    """Percent Agreement + Cohen's Kappa for each label column."""
    rows = []
    for col in LABEL_COLS:
        a = merged[f"{col}_{ANNOTATOR_A}"].astype(int)
        b = merged[f"{col}_{ANNOTATOR_B}"].astype(int)
        rows.append({
            "label_type": col,
            "n_items": len(merged),
            "percent_agreement": round((a == b).mean(), 4),
            "cohens_kappa": round(cohen_kappa_score(a, b), 4),
        })
    return pd.DataFrame(rows)


def build_consensus(merged: pd.DataFrame, disagreements: pd.DataFrame) -> pd.DataFrame:
    """Agreed label where both annotators match; resolved `final_*` where they differ."""
    dis = disagreements.set_index("username")
    out = merged[META_COLS].copy()

    # a row is 'discussed' iff the annotators differ on ANY of the three columns
    agree_all = pd.Series(True, index=merged.index)
    for col in LABEL_COLS:
        agree_all &= (merged[f"{col}_{ANNOTATOR_A}"] == merged[f"{col}_{ANNOTATOR_B}"])

    for col in LABEL_COLS:
        a = merged[f"{col}_{ANNOTATOR_A}"]
        b = merged[f"{col}_{ANNOTATOR_B}"]
        final = a.where(a == b)                       # agreed value, else NaN
        for i, user in enumerate(merged["username"]):
            if pd.isna(final.iloc[i]):                # a disagreement -> take resolved label
                final.iloc[i] = dis.loc[user, f"final_{col}"]
        out[col] = final.astype(int)

    out["consensus_source"] = agree_all.map({True: "auto_agree", False: "discussed"})
    return out


def main() -> None:
    merged = pd.read_csv(MERGED_FILE)
    disagreements = pd.read_csv(DISAGREEMENTS_FILE)

    report = agreement_report(merged)
    report.to_csv(AGREEMENT_OUT, index=False)

    consensus = build_consensus(merged, disagreements)
    consensus.to_csv(CONSENSUS_OUT, index=False)

    print("=== Step 3 — inter-annotator agreement ===")
    for _, r in report.iterrows():
        target = KAPPA_TARGETS[r["label_type"]]
        ok = "✓" if r["cohens_kappa"] >= target else "✗ below target"
        print(f"  {r['label_type']:<24s} agreement={r['percent_agreement']:.2f}  "
              f"kappa={r['cohens_kappa']:.4f}  (target ≥ {target})  {ok}")
    print(f"\n  consensus: {(consensus['consensus_source']=='auto_agree').sum()} auto_agree, "
          f"{(consensus['consensus_source']=='discussed').sum()} discussed")
    print(f"  wrote: {AGREEMENT_OUT.name}, {CONSENSUS_OUT.name}")


if __name__ == "__main__":
    main()
