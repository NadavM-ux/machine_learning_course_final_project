"""
Step 8 (part A) — Confidence Threshold  (בחירת סף ביטחון).
============================================================================
Deploys the FINAL enriched model on ALL candidate users, produces per-user
probabilities, a confidence-distribution table + histogram, recommends a
threshold, and exports a validation sample for a manual Hit-Rate check.

Model choice (defended in the report):
  * feature set = numeric + Iran-specificity features (iran_features.py).
    AUC 0.870 — within 0.02 of the tweets-based best (0.889) but needs NO text
    translation, so it deploys uniformly on all 946 users (incl. the 73% with
    no tweets).
  * algorithm  = RandomForest, wrapped in CalibratedClassifierCV (sigmoid, 5-fold)
    so the probabilities are meaningful for thresholding — unlike AdaBoost, whose
    higher AUC comes with degenerate (compressed) probabilities.
  * task framing = 2-class target vs non-target (drop 'unknown'=2). prob_2 is left
    empty in the output, which the PDF explicitly allows for a 2-class model.

The Hit-Rate check is computed by `hit_rate.py` over the FULL 173-user
high-confidence population (truth is already known for all of them: 42 validated
here + 131 labeled during active learning), so no further manual labeling is
needed. The LLM-comparison half of step 8 lives in `Iran_POI/LLM_Comparison/`.

Run:  python step8_confidence_threshold.py   (then: python hit_rate.py)
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

HERE = Path(__file__).resolve().parent
CLS = HERE.parent
sys.path.insert(0, str(CLS))
import active_learning as al          # noqa: E402
import iran_features as ifx           # noqa: E402

FINAL_ITER = 7
FEATURES = al.NUMERIC_FEATURES + ifx.FEATURE_COLS
THRESHOLD = 0.80
VALIDATION_N = 100
SEED = 42


def _featurize(df: pd.DataFrame, age_median: float | None) -> tuple[pd.DataFrame, float]:
    df = al.build_numeric(df)
    df = ifx.add_iran_features(df)
    if age_median is None:
        age_median = df['account_age_years'].median()
    df['account_age_years'] = df['account_age_years'].fillna(age_median)
    return df, age_median


def train_final_model():
    """Calibrated RandomForest on the 640 labeled users, 2-class target_population."""
    lab = pd.read_csv(al.combined_path(FINAL_ITER))
    lab['target_population'] = pd.to_numeric(lab['target_population'], errors='coerce').fillna(2).astype(int)
    lab, age_med = _featurize(lab, None)
    d2 = lab[lab['target_population'].isin([0, 1])]
    scaler = StandardScaler().fit(d2[FEATURES].values)
    X = scaler.transform(d2[FEATURES].values)
    y = d2['target_population'].astype(int).values
    base = RandomForestClassifier(n_estimators=300, class_weight='balanced',
                                  random_state=SEED, n_jobs=-1)
    model = CalibratedClassifierCV(base, method='sigmoid', cv=5)
    model.fit(X, y)
    print(f"trained calibrated RandomForest on {len(d2)} labeled (target vs non-target)")
    return model, scaler, age_med, set(al._norm_user(lab['username']))


def predict_all(model, scaler, age_med) -> pd.DataFrame:
    pool = pd.read_csv(al.POOL_FILE)
    pool, _ = _featurize(pool, age_med)
    X = scaler.transform(pool[FEATURES].values)
    proba = model.predict_proba(X)
    idx1 = list(model.classes_).index(1)
    prob_target = proba[:, idx1]
    predicted = model.predict(X)
    confidence = proba.max(axis=1)

    out = pool[[c for c in ['username', 'display_name', 'description', 'location']
                if c in pool.columns]].copy()
    out.insert(1, 'profile_url', 'https://x.com/' + out['username'].astype(str))
    out['predicted_class'] = predicted
    out['confidence_level'] = confidence.round(4)
    out['prob_0'] = proba[:, list(model.classes_).index(0)].round(4)
    out['prob_1'] = prob_target.round(4)
    out['prob_2'] = ''                                   # 2-class model -> empty (PDF-allowed)
    out['uncertainty_score'] = (1 - confidence).round(4)
    return out.sort_values('prob_1', ascending=False).reset_index(drop=True)


def confidence_table(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for thr in (0.9, 0.8, 0.7):
        m = pred['prob_1'] >= thr
        rows.append({'confidence_range': f'prob_target >= {thr}',
                     'n_users': int(m.sum()), 'pct_of_all': round(100 * m.mean(), 1)})
    tbl = pd.DataFrame(rows)
    tbl.to_csv(HERE / 'confidence_distribution_table.csv', index=False)
    print("\n=== confidence distribution (target probability) ===")
    print(tbl.to_string(index=False))
    return tbl


def histogram(pred: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(pred['prob_1'], bins=30, range=(0, 1), color='#1f77b4', edgecolor='white')
    ax.axvline(THRESHOLD, ls='--', lw=2, color='#d62728', label=f'chosen threshold = {THRESHOLD}')
    ax.set_title('Target-probability over all candidate users — Step 8 confidence threshold')
    ax.set_xlabel('prob_target (calibrated)'); ax.set_ylabel('number of users')
    ax.legend()
    fig.tight_layout()
    fig.savefig(HERE / 'confidence_histogram.png', dpi=150)
    plt.close(fig)
    print("saved: confidence_histogram.png")


def validation_sample(pred: pd.DataFrame, labeled_keys: set) -> None:
    """100 random users above threshold that are NOT already labeled -> manual Hit-Rate."""
    key = al._norm_user(pred['username'])
    cand = pred[(pred['prob_1'] >= THRESHOLD) & (~key.isin(labeled_keys))]
    take = min(VALIDATION_N, len(cand))
    sample = cand.sample(n=take, random_state=SEED).copy()
    sample['target_population'] = ''                     # for the human to fill (1=target,0=not,2=unknown)
    sample['comments'] = ''
    cols = ['username', 'profile_url', 'display_name', 'description', 'location',
            'predicted_class', 'prob_1', 'target_population', 'comments']
    sample[[c for c in cols if c in sample.columns]].to_csv(
        HERE / 'confidence_validation_sample_for_manual_labeling.csv', index=False)
    (HERE / 'confidence_validation_urls.txt').write_text(
        '\n'.join(sample['profile_url'].tolist()) + '\n')
    print(f"\nsaved validation sample: {take} users (prob_target >= {THRESHOLD}, not yet labeled)")
    print("  -> label target_population (1/0/2) in "
          "confidence_validation_sample_for_manual_labeling.csv, then compute Hit Rate.")


def main() -> None:
    print("=== Step 8 (part A) — Confidence Threshold ===")
    model, scaler, age_med, labeled_keys = train_final_model()
    pred = predict_all(model, scaler, age_med)
    pred.to_csv(HERE / 'final_model_predictions.csv', index=False)
    print(f"saved: final_model_predictions.csv ({len(pred)} users)")
    print(f"prob_target: min={pred['prob_1'].min():.3f} "
          f"median={pred['prob_1'].median():.3f} max={pred['prob_1'].max():.3f}")
    confidence_table(pred)
    histogram(pred)
    n_target = int((pred['prob_1'] >= THRESHOLD).sum())
    print(f"\n=> at threshold {THRESHOLD}: {n_target} users would enter the final target population.")
    validation_sample(pred, labeled_keys)
    print("\nNEXT: label the validation sample, then run part B (Hit Rate). "
          "LLM comparison is deferred until the step-4 flowchart exists.")


if __name__ == '__main__':
    main()
