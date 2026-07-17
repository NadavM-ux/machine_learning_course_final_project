"""
Step 7 — Stopping Criteria  (בדיקת עצירה).
============================================================================
Checks whether the Active-Learning loop (steps 5–6) reached saturation.

This is PHASE A of step 7 — everything that can run WITHOUT new labels:
  1. Run the FINAL best model (trained on all 600 labeled users from iteration 6)
     on the whole unlabeled pool -> predicted_class, confidence_level,
     prob_0/1/2, uncertainty_score.
  2. Histogram of confidence_level over all unlabeled users (is the model
     "too confident" — most predictions > 0.9 — or still unsure ~0.5?).
  3. Sample 20 users from the UNCERTAIN band [0.45, 0.55] and 20 from the
     CONFIDENT band [0.85, 0.95] -> a file for one more round of manual labeling.
  4. Cross-iteration performance trend (Accuracy + AUC per iteration 1..6) and a
     preliminary stopping decision based on the two PDF rules.

After you label the 40 sampled users, PHASE C (retrain + final conclusion) runs.

Reuses the tested engine in ../active_learning.py (no logic duplicated).
Run:  python stopping_criteria.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
CLS = HERE.parent                                   # …/Classification
sys.path.insert(0, str(CLS))
import active_learning as al                          # reuse the engine

FINAL_ITER = 6                                        # last active-learning iteration
UNCERTAIN_BAND = (0.45, 0.55)
CONFIDENT_BAND = (0.85, 0.95)
N_PER_BAND = 20
SEED = 42


def final_predictions(prev_iter: int = FINAL_ITER + 1) -> pd.DataFrame:
    """Train the final best target_population model on all labeled users and predict
    on the unlabeled pool (same recipe the engine uses in Phase A). `prev_iter` selects
    which labeled set to load: 7 -> iteration-6 combined (600), 8 -> iteration-7 (640)."""
    labeled = al.load_prev_labeled(prev_iter)          # combined set going into prev_iter
    unlabeled = al.load_unlabeled_pool(labeled)
    labeled = al.build_numeric(labeled)
    unlabeled = al.build_numeric(unlabeled)
    med = labeled['account_age_years'].median()
    labeled['account_age_years'] = labeled['account_age_years'].fillna(med)
    unlabeled['account_age_years'] = unlabeled['account_age_years'].fillna(med)

    scaler = StandardScaler()
    X_lab = scaler.fit_transform(labeled[al.NUMERIC_FEATURES].values)
    X_unl = scaler.transform(unlabeled[al.NUMERIC_FEATURES].values)
    y_lab = labeled['target_population'].astype(int).values

    algo, balanced = al.pick_best_target_model(prev_iter)
    model = al.make_model(algo, balanced)
    model.fit(X_lab, y_lab)

    # The best-AUC model may rank well yet emit DEGENERATE probabilities (AdaBoost's
    # predict_proba is famously compressed — here ~0.34–0.40 for every user). Confidence
    # bands and a confidence threshold are meaningless on such output, so if the spread
    # collapses we fall back to calibrated balanced LogReg for the confidence analysis.
    spread = model.predict_proba(X_unl).max(axis=1)
    if spread.max() - spread.min() < 0.15:
        print(f"note: best-AUC model '{algo}' has degenerate probabilities "
              f"(range {spread.min():.2f}–{spread.max():.2f}); using calibrated LogReg "
              f"for the confidence analysis instead.")
        algo, balanced = 'LogReg', True
        model = al.make_model(algo, balanced)
        model.fit(X_lab, y_lab)
    print(f"confidence model: {algo} (balanced={balanced}) trained on {len(labeled)} users")

    proba = model.predict_proba(X_unl)
    predicted = model.predict(X_unl)
    proba_full = np.zeros((proba.shape[0], 3))
    for j, c in enumerate(model.classes_):
        proba_full[:, int(c)] = proba[:, j]
    confidence = proba_full.max(axis=1)

    pred = unlabeled[[c for c in al.COMMON_COLS if c in unlabeled.columns]].copy()
    pred.insert(1, 'profile_url', 'https://x.com/' + pred['username'].astype(str))
    pred['predicted_class'] = predicted
    pred['confidence_level'] = confidence.round(4)
    pred['prob_0'] = proba_full[:, 0].round(4)
    pred['prob_1'] = proba_full[:, 1].round(4)
    pred['prob_2'] = proba_full[:, 2].round(4)
    pred['uncertainty_score'] = (1.0 - confidence).round(4)
    return pred.sort_values('uncertainty_score', ascending=False).reset_index(drop=True)


def confidence_histogram(pred: pd.DataFrame) -> None:
    conf = pred['confidence_level']
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(conf, bins=30, range=(0, 1), color='#3a6ea5', edgecolor='white')
    ax.axvspan(*UNCERTAIN_BAND, color='orange', alpha=0.25, label='uncertain [0.45–0.55]')
    ax.axvspan(*CONFIDENT_BAND, color='green', alpha=0.20, label='confident [0.85–0.95]')
    ax.set_title('Confidence (max class probability) over all unlabeled users — Stopping Criteria')
    ax.set_xlabel('confidence_level'); ax.set_ylabel('number of users'); ax.legend()
    fig.tight_layout()
    fig.savefig(HERE / 'stopping_criteria_confidence_histogram.png', dpi=150)
    plt.close(fig)
    print("saved: stopping_criteria_confidence_histogram.png")


def sample_bands(pred: pd.DataFrame) -> pd.DataFrame:
    rows, summary = [], []
    for name, (lo, hi) in [('uncertain', UNCERTAIN_BAND), ('confident', CONFIDENT_BAND)]:
        band = pred[(pred['confidence_level'] >= lo) & (pred['confidence_level'] <= hi)]
        take = min(N_PER_BAND, len(band))
        s = band.sample(n=take, random_state=SEED) if take else band
        s = s.copy(); s.insert(0, 'prob_group', f'{name} [{lo}-{hi}]')
        rows.append(s)
        summary.append({'prob_group': f'{name} [{lo}-{hi}]',
                        'available_in_band': len(band), 'sampled': take})
    samples = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    for t in al.TASKS:
        samples[t] = ''            # empty — for the human to fill
    samples['comments'] = ''
    samples.to_csv(HERE / 'stopping_criteria_probability_group_samples_for_manual_labeling.csv', index=False)
    pd.DataFrame(summary).to_csv(HERE / 'stopping_criteria_probability_group_summary.csv', index=False)
    print("saved: stopping_criteria_probability_group_samples_for_manual_labeling.csv "
          f"({len(samples)} users to label)")
    return pd.DataFrame(summary)


def performance_trend_and_decision(pred: pd.DataFrame) -> None:
    comp = pd.read_csv(CLS / 'iteration_comparison_summary.csv')
    perf = comp[['iteration', 'n_labeled_users', 'mean_accuracy',
                 'mean_AUC', 'best_AUC_target_population']].copy()
    perf['d_accuracy'] = perf['mean_accuracy'].diff().round(4)
    perf['d_best_AUC_target'] = perf['best_AUC_target_population'].diff().round(4)
    perf.to_csv(HERE / 'stopping_criteria_performance_summary.csv', index=False)

    # trend plot: Accuracy + AUC vs iteration
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(perf['iteration'], perf['mean_accuracy'], 'o-', label='mean Accuracy (all experiments)')
    ax.plot(perf['iteration'], perf['mean_AUC'], 's-', label='mean AUC (all experiments)')
    ax.plot(perf['iteration'], perf['best_AUC_target_population'], '^-', label='best AUC (target_population)')
    ax.set_title('Performance across Active-Learning iterations (Stopping Criteria)')
    ax.set_xlabel('iteration'); ax.set_ylabel('score'); ax.set_ylim(0, 1); ax.legend()
    fig.tight_layout()
    fig.savefig(HERE / 'stopping_criteria_final_accuracy_auc_graph.png', dpi=150)
    plt.close(fig)
    print("saved: stopping_criteria_final_accuracy_auc_graph.png")

    # --- stopping rules (PDF page 17) ---
    # PDF: stop when there is no measurable IMPROVEMENT (gain < 0.5%). We use the
    # SIGNED delta, not |Δ|: a metric that stalled OR declined is "no improvement",
    # so a -0.7% change must count as MET. (An earlier |Δ| version wrongly read the
    # decline as "still changing" and returned stop=False, contradicting the trend.)
    d_acc = float(perf['d_accuracy'].iloc[-1])                 # signed gain in accuracy
    d_auc = float(perf['d_best_AUC_target'].iloc[-1])          # signed gain in best AUC
    THRESH = 0.005                                             # 0.5% improvement bar
    rule1 = (d_acc < THRESH) and (d_auc < THRESH)              # neither metric gained >= 0.5%

    conf = pred['confidence_level']
    frac_extreme = float(((conf > 0.9) | (conf < 0.1)).mean())  # variance-collapse proxy
    rule2 = frac_extreme > 0.5                                 # model "too confident"

    decision = pd.DataFrame([{
        'last_iteration': int(perf['iteration'].iloc[-1]),
        'last_delta_accuracy': round(d_acc, 4),
        'last_delta_best_AUC_target': round(d_auc, 4),
        'improvement_threshold': THRESH,
        'rule1_no_improvement': bool(rule1),
        'frac_unlabeled_conf_gt0.9_or_lt0.1': round(frac_extreme, 4),
        'rule2_confidence_collapsed': bool(rule2),
        'stop_recommended': bool(rule1 or rule2),
    }])
    decision.to_csv(HERE / 'stopping_criteria_decision_summary.csv', index=False)

    print("\n=== STOPPING DECISION ===")
    print(f"  Rule 1 (accuracy gain <{THRESH} AND best-AUC gain <{THRESH}, signed): "
          f"Δacc={d_acc:+.4f}, Δauc={d_auc:+.4f} -> {'MET' if rule1 else 'not met'}")
    print(f"  Rule 2 (>50% of unlabeled at conf>0.9 or <0.1): "
          f"{frac_extreme:.1%} -> {'MET' if rule2 else 'not met'}")
    print(f"  => STOP recommended: {bool(rule1 or rule2)}")
    print("     (final confirmation after labeling the 40 band-samples + retrain — Phase C)")


def phase_a() -> None:
    print("=== Step 7 — Stopping Criteria (Phase A) ===")
    pred = final_predictions()
    pred.to_csv(HERE / 'stopping_criteria_unlabeled_users_predictions.csv', index=False)
    print(f"saved: stopping_criteria_unlabeled_users_predictions.csv ({len(pred)} users)")
    print(f"confidence: min={pred['confidence_level'].min():.3f} "
          f"median={pred['confidence_level'].median():.3f} "
          f"max={pred['confidence_level'].max():.3f}")
    confidence_histogram(pred)
    sample_bands(pred)
    performance_trend_and_decision(pred)
    print("\nNEXT: label the 40 users in "
          "stopping_criteria_probability_group_samples_for_manual_labeling.csv, "
          "then run `python stopping_criteria.py --phase C`.")


def phase_c() -> None:
    """Fold the 40 labeled band-samples in as 'iteration 7', retrain the full sweep
    via the engine, then recompute the stopping decision on iteration 6 -> 7."""
    ITER = FINAL_ITER + 1                                    # 7
    print(f"=== Step 7 — Stopping Criteria (Phase C): retrain as iteration {ITER} ===")

    samples_path = HERE / 'stopping_criteria_probability_group_samples_for_manual_labeling.csv'
    samples = pd.read_csv(samples_path)
    labeled_n = int(pd.to_numeric(samples['target_population'], errors='coerce').notna().sum())
    if labeled_n < len(samples):
        raise ValueError(f"{labeled_n}/{len(samples)} band-samples labeled — "
                         f"finish labeling before Phase C.")
    print(f"band-samples labeled: {labeled_n}/{len(samples)}")

    # Build the engine's to-label file for iteration 7 from the labeled samples.
    iter_dir = CLS / f'Iteration_{ITER}'
    iter_dir.mkdir(exist_ok=True)
    keep = [c for c in (['username', 'profile_url'] + al.COMMON_COLS + al.TASKS + ['comments'])
            if c in samples.columns]
    keep = list(dict.fromkeys(keep))                        # de-dup, preserve order
    to_label_path = iter_dir / f'iteration_{ITER}_users_to_label.csv'
    samples[keep].to_csv(to_label_path, index=False)
    print(f"wrote {to_label_path.relative_to(CLS)} ({len(samples)} users)")

    # Retrain: engine merges 600 ∪ 40 -> 640, runs the full sweep, and refreshes
    # iteration_comparison_summary.csv + the trend / hold-out plots.
    al.phase_c(ITER)

    # Recompute the stopping decision with the NEW (640-user) model's confidence
    # for rule 2, and the freshly-added iteration-7 row for rule 1.
    pred = final_predictions(prev_iter=ITER + 1)            # loads combined_path(7) = 640
    pred.to_csv(HERE / 'stopping_criteria_unlabeled_users_predictions_after_retrain.csv',
                index=False)
    print(f"saved: stopping_criteria_unlabeled_users_predictions_after_retrain.csv "
          f"({len(pred)} users)")
    performance_trend_and_decision(pred)
    print(f"\nDONE Step 7. Decision now reflects iteration {FINAL_ITER} -> {ITER} "
          f"(640 labeled). See stopping_criteria_decision_summary.csv.")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Step 7 — Stopping Criteria.")
    ap.add_argument('--phase', choices=['A', 'C'], default='A',
                    help="A = predict + histogram + band-sampling + preliminary decision; "
                         "C = fold labeled band-samples in, retrain, final decision")
    args = ap.parse_args()
    (phase_a if args.phase == 'A' else phase_c)()


if __name__ == "__main__":
    main()
