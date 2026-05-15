"""Step 5 analysis: verify winners, build summary CSV, generate report plots.

Run from anywhere:
    python Iran_POI/Classification/Iteration_1/Step5_Analysis/step5_analysis.py

Inputs (resolved relative to this script's location):
    ../../experiments_results_iteration_1.csv      (Classification/ root)
    ./iteration_1_consensus_translated.csv         (same folder)

Outputs (all written to this script's folder):
    iteration_1_best_models_summary.csv
    iteration_1_degeneracy_check.csv
    plot_f1_by_algorithm.png
    plot_f1_by_feature_set.png
    plot_kfold_vs_loocv.png
"""
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).parent                        # …/Step5_Analysis
CLASSIFICATION = HERE.parent.parent                  # …/Classification
RESULTS_CSV = CLASSIFICATION / 'experiments_results_iteration_1.csv'
LABELED_CSV = HERE / 'iteration_1_consensus_translated.csv'

results = pd.read_csv(RESULTS_CSV)
labeled = pd.read_csv(LABELED_CSV)

print(f"Loaded {len(results)} experiment rows, {len(labeled)} labeled users\n")

# ============================================================
# A2 — Degeneracy check: does the locals_vs_diaspora AdaBoost
#      "winner" actually predict anything besides "unknown"?
# ============================================================
print("="*60)
print("A2. Degeneracy check: locals_vs_diaspora AdaBoost winner")
print("="*60)

# Recreate the winning combo: AdaBoost + desc+numeric + K-Fold + unbalanced + 3-class
# Build the same feature matrix.
for c in ['followers_count','following_count','statuses_count']:
    labeled[c] = pd.to_numeric(labeled[c], errors='coerce').fillna(0)
labeled['bio_length'] = labeled['description_en'].fillna('').astype(str).str.len()
labeled['followers_following_ratio'] = labeled['followers_count'] / (labeled['following_count']+1)
labeled['created_at_dt'] = pd.to_datetime(labeled['created_at'], format='%B %Y', errors='coerce')
labeled['account_age_years'] = ((pd.Timestamp('2026-05-14') - labeled['created_at_dt']).dt.days/365.25).fillna(0)
labeled['has_description'] = labeled['description'].notna().astype(int)
labeled['has_location'] = labeled['location'].notna().astype(int)
iran_kw = ['iran','iranian','persian','persia','tehran','shiraz','esfahan','isfahan','mashhad','tabriz','kerman','qom','farsi']
def has_iran(t):
    if pd.isna(t): return 0
    return int(any(k in str(t).lower() for k in iran_kw))
labeled['bio_mentions_iran'] = labeled['description_en'].apply(has_iran)
labeled['name_mentions_iran'] = labeled['display_name_en'].apply(has_iran)
labeled['location_mentions_iran'] = labeled['location'].apply(has_iran)
numfeats = ['followers_count','following_count','statuses_count','followers_following_ratio',
            'bio_length','account_age_years','has_description','has_location',
            'bio_mentions_iran','name_mentions_iran','location_mentions_iran']

numeric_scaled = sp.csr_matrix(StandardScaler(with_mean=False).fit_transform(labeled[numfeats].values))
vec_desc = TfidfVectorizer(max_features=300, lowercase=True, stop_words='english', min_df=2)
tfidf_desc = vec_desc.fit_transform(labeled['description_en'].fillna('').astype(str))
X = sp.hstack([tfidf_desc, numeric_scaled]).tocsr()

# Check predictions for top winners
degeneracy_rows = []
for tgt in ['target_population', 'locals_vs_diaspora', 'person_vs_organization']:
    for n_cls in [3, 2]:
        sub = results[(results['target_column']==tgt) & (results['#classes']==n_cls) &
                      (results['algorithm']=='AdaBoost') & (results['feature_set']=='desc+numeric') &
                      (results['training_type']=='K-Fold') & (results['balanced']==False)]
        if sub.empty: continue
        row = sub.iloc[0]

        y_full = labeled[tgt].values
        if n_cls == 2:
            mask = (y_full != 2)
            y = y_full[mask]
            Xs = X[mask]
        else:
            y = y_full; Xs = X

        if len(np.unique(y)) < 2: continue
        cv = StratifiedKFold(5, shuffle=True, random_state=42)
        all_pred, all_true = [], []
        for tr, te in cv.split(np.zeros(len(y)), y):
            m = AdaBoostClassifier(random_state=42)
            m.fit(Xs[tr], y[tr])
            all_pred.extend(m.predict(Xs[te]))
            all_true.extend(y[te])

        pred_dist = pd.Series(all_pred).value_counts(normalize=True).sort_index().to_dict()
        true_dist = pd.Series(all_true).value_counts(normalize=True).sort_index().to_dict()
        degeneracy_rows.append({
            'task': tgt, 'n_classes': n_cls,
            'reported_acc': round(row['accuracy'],3),
            'reported_F1': round(row['F1'],3),
            'reported_AUC': round(row['AUC'],3),
            'true_class_dist': str({k: round(v,2) for k,v in true_dist.items()}),
            'pred_class_dist': str({k: round(v,2) for k,v in pred_dist.items()}),
            'is_degenerate': len(set(all_pred))==1 or max(pred_dist.values())>0.95,
        })

deg_df = pd.DataFrame(degeneracy_rows)
print(deg_df.to_string(index=False))
deg_df.to_csv(HERE / 'iteration_1_degeneracy_check.csv', index=False)
print(f"\n  → saved iteration_1_degeneracy_check.csv\n")


# ============================================================
# A3.1 — Best model per (task, n_classes), excluding degenerate ones
# ============================================================
print("="*60)
print("A3. Best model per task — honest picks")
print("="*60)

# An "honest" model:
#  - accuracy / F1 not driven by majority class only
#  - AUC > 0.60 (better than near-random)
def honest_pick(df_sub):
    candidates = df_sub.dropna(subset=['F1','AUC']).copy()
    # filter to AUC > 0.6 if any qualify; else keep them all
    qualify = candidates[candidates['AUC'] > 0.60]
    pool = qualify if len(qualify) else candidates
    return pool.sort_values(['F1','AUC'], ascending=False).head(1)

summary = []
for tgt in ['target_population', 'locals_vs_diaspora', 'person_vs_organization']:
    for n_cls in [3, 2]:
        sub = results[(results['target_column']==tgt) & (results['#classes']==n_cls)]
        # Naive winner (by F1 only):
        naive = sub.dropna(subset=['F1']).sort_values('F1', ascending=False).head(1)
        # Honest winner:
        honest = honest_pick(sub)
        if naive.empty: continue
        n_row = naive.iloc[0]; h_row = honest.iloc[0] if not honest.empty else n_row
        summary.append({
            'task': tgt,
            'n_classes': n_cls,
            'naive_algo': n_row['algorithm'],
            'naive_feature_set': n_row['feature_set'],
            'naive_F1': round(n_row['F1'],3),
            'naive_AUC': round(n_row['AUC'],3),
            'honest_algo': h_row['algorithm'],
            'honest_feature_set': h_row['feature_set'],
            'honest_training_type': h_row['training_type'],
            'honest_balanced': h_row['balanced'],
            'honest_accuracy': round(h_row['accuracy'],3),
            'honest_precision': round(h_row['precision'],3),
            'honest_recall': round(h_row['recall'],3),
            'honest_F1': round(h_row['F1'],3),
            'honest_AUC': round(h_row['AUC'],3),
        })

summary_df = pd.DataFrame(summary)
print(summary_df.to_string(index=False))
summary_df.to_csv(HERE / 'iteration_1_best_models_summary.csv', index=False)
print(f"\n  → saved iteration_1_best_models_summary.csv\n")


# ============================================================
# A3.2 — Plots for the report
# ============================================================
print("="*60)
print("A3. Plots")
print("="*60)

# Plot 1: F1 by algorithm, per task, n_classes=3 (3-class is what AL will use)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
for ax, tgt in zip(axes, ['target_population','locals_vs_diaspora','person_vs_organization']):
    sub = results[(results['target_column']==tgt) & (results['#classes']==3)]
    means = sub.groupby('algorithm')['F1'].mean().sort_values(ascending=False)
    bars = ax.bar(means.index, means.values, color='steelblue')
    ax.set_title(tgt, fontsize=11)
    ax.set_ylabel('mean F1' if ax is axes[0] else '')
    ax.set_ylim(0, 1.0)
    ax.tick_params(axis='x', rotation=30)
    for b, v in zip(bars, means.values):
        ax.text(b.get_x()+b.get_width()/2, v+0.01, f'{v:.2f}', ha='center', fontsize=8)
fig.suptitle('Mean F1 by algorithm (3-class, across all feature sets + CV strategies + balance modes)', fontsize=11)
plt.tight_layout(rect=[0,0,1,0.95])
plt.savefig(HERE / 'plot_f1_by_algorithm.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  → saved plot_f1_by_algorithm.png")

# Plot 2: F1 by feature set, per task, n_classes=3
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
for ax, tgt in zip(axes, ['target_population','locals_vs_diaspora','person_vs_organization']):
    sub = results[(results['target_column']==tgt) & (results['#classes']==3)]
    means = sub.groupby('feature_set')['F1'].mean().sort_values(ascending=False)
    ax.barh(means.index, means.values, color='coral')
    ax.set_title(tgt, fontsize=11)
    ax.set_xlabel('mean F1')
    ax.set_xlim(0, 1.0)
    ax.invert_yaxis()
    for i, v in enumerate(means.values):
        ax.text(v+0.005, i, f'{v:.2f}', va='center', fontsize=8)
fig.suptitle('Mean F1 by feature set (3-class)', fontsize=11)
plt.tight_layout(rect=[0,0,1,0.95])
plt.savefig(HERE / 'plot_f1_by_feature_set.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  → saved plot_f1_by_feature_set.png")

# Plot 3: K-Fold vs LOOCV — paired scatter per (task, algo, fset, balanced)
fig, ax = plt.subplots(figsize=(6, 6))
for tgt, color in zip(['target_population','locals_vs_diaspora','person_vs_organization'],
                      ['#1f77b4','#ff7f0e','#2ca02c']):
    sub = results[(results['target_column']==tgt) & (results['#classes']==3)]
    pivot = sub.pivot_table(index=['algorithm','feature_set','balanced'],
                             columns='training_type', values='F1').dropna()
    ax.scatter(pivot['K-Fold'], pivot['LOOCV'], alpha=0.5, label=tgt, color=color, s=20)
ax.plot([0,1],[0,1], 'k--', lw=1, alpha=0.5)
ax.set_xlim(0,1); ax.set_ylim(0,1)
ax.set_xlabel('F1 with K-Fold (K=5)')
ax.set_ylabel('F1 with LOOCV')
ax.set_title('K-Fold vs LOOCV agreement (3-class)')
ax.legend(loc='upper left', fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(HERE / 'plot_kfold_vs_loocv.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  → saved plot_kfold_vs_loocv.png")

print("\n✅ Phase A complete.")
