"""
Step 8-B (LLM comparison) — input builder.
============================================================================
Prepares everything you paste into the LLM (Claude Opus):

  1. llm_test_set_no_answers.csv   -> the test users WITHOUT the answer columns
                                       (this is what you give the model).
  2. llm_test_set_groundtruth.csv  -> the SAME users WITH the 3 human labels,
                                       kept aside for scoring (score_llm_runs.py).
  3. llm_prompt.md                 -> the English prompt: task + the three
                                       step-4 decision flowcharts + output format.

Test set = the 30 hold-out users (Classification/holdout_test_set.csv), the fixed
evaluation set used across the whole project. All 30 also appear in
iteration_7_combined_labeled — so we pull their profile fields, English
translations, and all THREE ground-truth columns from there.

Run:  python build_llm_input.py
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
CLS = HERE.parent / "Classification"

HOLDOUT = CLS / "holdout_test_set.csv"
SOURCE = CLS / "Iteration_7" / "iteration_7_combined_translated.csv"

LABELS = ["target_population", "locals_vs_diaspora", "person_vs_organization"]
# fields the model is allowed to see (mirrors what a human annotator inspects)
PROFILE = [
    "username", "display_name", "display_name_en", "description", "description_en",
    "location", "followers_count", "following_count", "statuses_count", "created_at",
]


def main() -> None:
    holdout = pd.read_csv(HOLDOUT)
    src = pd.read_csv(SOURCE)

    missing = set(holdout["username"]) - set(src["username"])
    if missing:
        raise SystemExit(f"{len(missing)} hold-out users missing from source: {sorted(missing)[:5]}")

    df = src[src["username"].isin(holdout["username"])].copy()
    df = df.drop_duplicates("username").reset_index(drop=True)

    have_profile = [c for c in PROFILE if c in df.columns]

    no_answers = df[have_profile].copy()
    groundtruth = df[["username", *LABELS]].copy()

    no_answers.to_csv(HERE / "llm_test_set_no_answers.csv", index=False)
    groundtruth.to_csv(HERE / "llm_test_set_groundtruth.csv", index=False)

    print(f"test users:              {len(df)}")
    print(f"profile fields exposed:  {have_profile}")
    for c in LABELS:
        print(f"  {c:24s} {dict(groundtruth[c].value_counts().sort_index())}")
    print("\nwrote:")
    print("  llm_test_set_no_answers.csv   (paste this into Claude)")
    print("  llm_test_set_groundtruth.csv  (kept for scoring)")
    print("  llm_prompt.md already written alongside — paste it before the table.")


if __name__ == "__main__":
    main()
