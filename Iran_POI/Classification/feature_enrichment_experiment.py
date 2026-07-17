"""
Feature-enrichment experiment  —  baseline vs Iran-specificity + tweet features.
============================================================================
Proves (and documents for the report) that enriching the feature set with the
Iran-specific lexicon (iran_features.py) + tweet TF-IDF lifts AUC on all three
tasks, after data growth (iterations 1-7) had already saturated at ~0.85.

K-Fold only (LOOCV doesn't change the comparison and would 3x the runtime).
Reuses active_learning.run_one_experiment so metrics match the main sweep.
"""
from __future__ import annotations
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings('ignore')

import active_learning as al
import iran_features as ifx

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / 'Feature_Enrichment'
OUT_DIR.mkdir(exist_ok=True)
FINAL_ITER = 7


def _tfidf(text: pd.Series, min_df: int):
    vec = TfidfVectorizer(max_features=300, lowercase=True, stop_words='english', min_df=min_df)
    return vec.fit_transform(text.fillna('').astype(str))


def build_sets(df: pd.DataFrame) -> dict:
    """Baseline sets + enriched (Iran / tweets) sets, so each pair is comparable."""
    num = sp.csr_matrix(StandardScaler(with_mean=False).fit_transform(df[al.NUMERIC_FEATURES].values))
    iran = sp.csr_matrix(StandardScaler(with_mean=False).fit_transform(df[ifx.FEATURE_COLS].values))
    desc = _tfidf(df['description_en'], 2)
    sets = {
        'numeric':              num,                                   # baseline
        'numeric+iran':         sp.hstack([num, iran]).tocsr(),        # enriched
        'desc+numeric':         sp.hstack([desc, num]).tocsr(),        # baseline
        'desc+numeric+iran':    sp.hstack([desc, num, iran]).tocsr(),  # enriched
        'iran_only':            iran,                                  # signal check
    }
    try:
        tweets = _tfidf(ifx.tweet_blob_series(df), 2)
        sets['tweets+numeric+iran'] = sp.hstack([tweets, num, iran]).tocsr()   # full enrichment
    except ValueError:
        print("[warn] tweet TF-IDF empty (too few tweeters) — skipping tweet set")
    return sets


def main():
    print("=== Feature-enrichment experiment (K-Fold, 640 users) ===")
    df = pd.read_csv(al.combined_path(FINAL_ITER))
    for t in al.TASKS:
        df[t] = pd.to_numeric(df[t], errors='coerce').fillna(2).astype(int)
    # translated text is already cached from the main sweep
    df = al.translate_missing(df)
    df = al.build_numeric(df)
    df['account_age_years'] = df['account_age_years'].fillna(df['account_age_years'].median())
    df = ifx.add_iran_features(df)
    print(f"tweeters: {int(df['has_tweets'].sum())}/{len(df)}")

    feature_sets = build_sets(df)
    print("feature sets: " + ", ".join(f"{k}{v.shape}" for k, v in feature_sets.items()))

    rows = []
    for tname in al.TASKS:
        y_full = df[tname].astype(int).values
        for n_classes in (3, 2):
            if n_classes == 2:
                mask = (y_full != 2); y = y_full[mask]
            else:
                mask, y = None, y_full
            if len(np.unique(y)) < 2:
                continue
            for fname, Xf in feature_sets.items():
                X = Xf[mask] if mask is not None else Xf
                for algo in al.ALGOS:
                    r = al.run_one_experiment(X, y, algo, True, 'K-Fold',
                                              tname, fname, n_classes, FINAL_ITER)
                    rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(OUT_DIR / 'enrichment_experiments.csv', index=False)

    # baseline-vs-enriched summary: best AUC per (task, family)
    pairs = [('numeric', 'numeric+iran'), ('desc+numeric', 'desc+numeric+iran')]
    summ = []
    for tname in al.TASKS:
        d = res[(res['target_column'] == tname) & (res['AUC'].notna())]
        for base, enr in pairs:
            b = d[d['feature_set'] == base]['AUC'].max()
            e = d[d['feature_set'] == enr]['AUC'].max()
            summ.append({'task': tname, 'family': base,
                         'baseline_bestAUC': round(b, 4) if pd.notna(b) else np.nan,
                         'enriched_bestAUC': round(e, 4) if pd.notna(e) else np.nan,
                         'delta': round(e - b, 4) if pd.notna(b) and pd.notna(e) else np.nan})
        full = d[d['feature_set'] == 'tweets+numeric+iran']['AUC'].max()
        if pd.notna(full):
            summ.append({'task': tname, 'family': 'tweets+numeric+iran(full)',
                         'baseline_bestAUC': np.nan, 'enriched_bestAUC': round(full, 4), 'delta': np.nan})
    summary = pd.DataFrame(summ)
    summary.to_csv(OUT_DIR / 'enrichment_summary.csv', index=False)
    print("\n=== BASELINE vs ENRICHED (best K-Fold AUC per task) ===")
    print(summary.to_string(index=False))
    print(f"\nsaved: {OUT_DIR.name}/enrichment_experiments.csv, enrichment_summary.csv")


if __name__ == '__main__':
    main()
