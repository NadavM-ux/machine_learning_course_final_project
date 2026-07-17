"""
Enriched K-Fold sweep across ALL active-learning iterations (1..7).
====================================================================
Answers "can we improve Step-6 accuracy?": folds the Iran-specific lexicon
(iran_features.py) + tweet TF-IDF into the feature sets and re-runs the sweep
for every iteration — K-Fold only (LOOCV inflates AUC on this tiny data and is
not needed for the baseline-vs-enriched comparison; see active_learning.py:534).

Self-contained & internally consistent: baseline (numeric, desc+numeric) and
enriched (…+iran, tweets+numeric+iran) sets are built the SAME way in one run,
so the delta is a clean apples-to-apples measurement.

Outputs (Feature_Enrichment/):
  enriched_kfold_all_iterations.csv   every experiment, all iterations
  enriched_kfold_trend.csv            per-iteration best baseline vs enriched
"""
from __future__ import annotations
import time, warnings
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
OUT = HERE / 'Feature_Enrichment'
OUT.mkdir(exist_ok=True)

BASELINE_SETS = ['numeric', 'desc+numeric']
ENRICHED_SETS = ['numeric+iran', 'desc+numeric+iran', 'tweets+numeric+iran']


def _tfidf(text: pd.Series, min_df: int):
    vec = TfidfVectorizer(max_features=300, lowercase=True, stop_words='english', min_df=min_df)
    return vec.fit_transform(text.fillna('').astype(str))


def build_sets(df: pd.DataFrame) -> dict:
    num = sp.csr_matrix(StandardScaler(with_mean=False).fit_transform(df[al.NUMERIC_FEATURES].values))
    iran = sp.csr_matrix(StandardScaler(with_mean=False).fit_transform(df[ifx.FEATURE_COLS].values))
    desc = _tfidf(df['description_en'], 2)
    sets = {
        'numeric':           num,
        'numeric+iran':      sp.hstack([num, iran]).tocsr(),
        'desc+numeric':      sp.hstack([desc, num]).tocsr(),
        'desc+numeric+iran': sp.hstack([desc, num, iran]).tocsr(),
    }
    try:
        tweets = _tfidf(ifx.tweet_blob_series(df), 2)
        sets['tweets+numeric+iran'] = sp.hstack([tweets, num, iran]).tocsr()
    except ValueError:
        print("   [warn] tweet TF-IDF empty — skipping tweet set this iteration")
    return sets


def load_iter(n: int) -> pd.DataFrame:
    """Translated combined labeled set for iteration n."""
    if n == 1:
        df = pd.read_csv(al.ITER1_CONSENSUS)                       # already translated
    else:
        df = pd.read_csv(HERE / f'Iteration_{n}' / f'iteration_{n}_combined_translated.csv')
    for t in al.TASKS:
        df[t] = pd.to_numeric(df[t], errors='coerce').fillna(2).astype(int)
    df = al.build_numeric(df)
    df['account_age_years'] = df['account_age_years'].fillna(df['account_age_years'].median())
    df = ifx.add_iran_features(df)
    return df


def main():
    all_rows = []
    for n in range(1, 8):
        t0 = time.time()
        df = load_iter(n)
        feature_sets = build_sets(df)
        n_exp = 0
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
                        for balanced in (True, False):
                            r = al.run_one_experiment(X, y, algo, balanced, 'K-Fold',
                                                      tname, fname, n_classes, n)
                            r['n_labeled_users'] = len(df)
                            all_rows.append(r); n_exp += 1
        print(f"iter{n}: {len(df):3d} users, {n_exp} K-Fold experiments, {time.time()-t0:.0f}s")

    res = pd.DataFrame(all_rows)
    res.to_csv(OUT / 'enriched_kfold_all_iterations.csv', index=False)

    # per-iteration trend: best model (by AUC) baseline vs enriched, focus target_population 2-class
    def best(sub, col):
        s = sub[col].dropna()
        return round(float(s.max()), 4) if len(s) else np.nan

    trend = []
    for n in range(1, 8):
        it = res[res['iteration'] == n]
        nlab = int(it['n_labeled_users'].iloc[0])
        row = {'iteration': n, 'n_labeled_users': nlab}
        for tname in al.TASKS:
            for tag, sets in (('baseline', BASELINE_SETS), ('enriched', ENRICHED_SETS)):
                # 2-class headline for target; 3-class stays available in the raw file
                sub = it[(it['target_column'] == tname) & (it['#classes'] == 2)
                         & (it['feature_set'].isin(sets))]
                row[f'{tname}__{tag}_bestAcc'] = best(sub, 'accuracy')
                row[f'{tname}__{tag}_bestAUC'] = best(sub, 'AUC')
        trend.append(row)
    tdf = pd.DataFrame(trend)
    tdf.to_csv(OUT / 'enriched_kfold_trend.csv', index=False)
    print("\n=== TARGET_POPULATION (2-class) — best K-Fold per iteration ===")
    cols = ['iteration', 'n_labeled_users',
            'target_population__baseline_bestAcc', 'target_population__enriched_bestAcc',
            'target_population__baseline_bestAUC', 'target_population__enriched_bestAUC']
    print(tdf[cols].to_string(index=False))
    print(f"\nsaved: {OUT.name}/enriched_kfold_all_iterations.csv, enriched_kfold_trend.csv")


if __name__ == '__main__':
    main()
