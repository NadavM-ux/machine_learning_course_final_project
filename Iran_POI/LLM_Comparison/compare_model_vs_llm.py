"""
Step 8-B — FAIR trained-model vs LLM comparison.
============================================================================
The point of step 8: compare the trained ML model to the LLM. A naive comparison
leaks — the step-8 model was trained on all 640 iteration-7 users, which INCLUDE
the 30 hold-out test users. So here we retrain the same recipe on 640 MINUS the
30 hold-out, then score both the model and the LLM majority on those 30 unseen
users. Primary task = target_population (2-class: target vs non-target).
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)

HERE = Path(__file__).resolve().parent
CLS = HERE.parent / 'Classification'
sys.path.insert(0, str(CLS))
import active_learning as al          # noqa: E402
import iran_features as ifx           # noqa: E402

FEATURES = al.NUMERIC_FEATURES + ifx.FEATURE_COLS
SEED = 42


def _metrics(yt, yp, proba=None):
    m = {'accuracy': round(accuracy_score(yt, yp), 4),
         'precision': round(precision_score(yt, yp, average='macro', zero_division=0), 4),
         'recall': round(recall_score(yt, yp, average='macro', zero_division=0), 4),
         'F1': round(f1_score(yt, yp, average='macro', zero_division=0), 4)}
    try:
        m['AUC'] = round(roc_auc_score(yt, proba), 4) if proba is not None else ''
    except Exception:
        m['AUC'] = ''
    return m


def main() -> None:
    holdout = set(al._norm_user(pd.read_csv(CLS / 'holdout_test_set.csv')['username']))

    lab = pd.read_csv(al.combined_path(7))
    lab['target_population'] = pd.to_numeric(lab['target_population'], errors='coerce').fillna(2).astype(int)
    lab = al.build_numeric(lab); lab = ifx.add_iran_features(lab)
    key = al._norm_user(lab['username'])

    # FAIR: train on everything EXCEPT the hold-out, 2-class target vs non-target
    train_sel = ~key.isin(holdout) & lab['target_population'].isin([0, 1])
    test_sel = key.isin(holdout) & lab['target_population'].isin([0, 1])
    # impute account_age_years with the TRAIN median only (no test leak)
    age_med = lab.loc[train_sel, 'account_age_years'].median()
    lab['account_age_years'] = lab['account_age_years'].fillna(age_med)
    train = lab[train_sel]
    test = lab[test_sel]
    print(f"train (holdout-excluded): {len(train)} | test (certain holdout): {len(test)}")

    sc = StandardScaler().fit(train[FEATURES].values)
    model = CalibratedClassifierCV(
        RandomForestClassifier(n_estimators=300, class_weight='balanced',
                               random_state=SEED, n_jobs=-1),
        method='sigmoid', cv=5).fit(sc.transform(train[FEATURES].values),
                                    train['target_population'].values)
    Xte = sc.transform(test[FEATURES].values)
    yte = test['target_population'].values
    i1 = list(model.classes_).index(1)
    model_row = {'model': 'Trained model (RF+iran, holdout-excluded)',
                 **_metrics(yte, model.predict(Xte), model.predict_proba(Xte)[:, i1])}

    # LLM majority on the SAME certain hold-out users, 2-class view.
    # Hard majority label -> accuracy/precision/recall/F1 (a '2'=unknown counts as a miss).
    # For AUC a hard 0/1/2 label is NOT a valid score, so use the class-1 VOTE FRACTION
    # across the 10 runs as P(target) per user.
    maj = pd.read_csv(HERE / 'Majority_iterations' / 'chatgpt_predictions_majority_iterations.csv')
    maj['k'] = al._norm_user(maj['username'])
    test_keys = al._norm_user(test['username'])
    mj = maj.set_index('k').reindex(test_keys)['target_population'].astype(int).values

    run_files = sorted(HERE.glob('Iteration_*/chatgpt_predictions_iteration_*.csv'),
                       key=lambda p: int(p.parent.name.split('_')[1]))
    votes = []
    for f in run_files:
        r = pd.read_csv(f); r['k'] = al._norm_user(r['username'])
        votes.append(r.set_index('k')['target_population'].reindex(test_keys))
    V = pd.concat(votes, axis=1)
    llm_prob1 = ((V == 1).sum(axis=1) / V.notna().sum(axis=1)).values   # P(target) per user
    llm_row = {'model': 'LLM majority (Claude Opus, 10 runs, blind)',
               **_metrics(yte, mj, llm_prob1)}

    out = pd.DataFrame([model_row, llm_row])
    out.to_csv(HERE / 'trained_model_vs_llm_comparison.csv', index=False)
    print("\n=== FAIR comparison on target_population (certain hold-out users) ===")
    print(out.to_string(index=False))
    print(f"\nsaved: trained_model_vs_llm_comparison.csv  (n_test={len(test)})")


if __name__ == '__main__':
    main()
