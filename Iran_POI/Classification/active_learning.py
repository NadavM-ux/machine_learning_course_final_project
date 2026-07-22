"""
Active Learning engine — Iran POI project, Step 6.
================================================================

Repeatable across iterations 3, 4, 5, 6 ... . All THREE classification tasks
(target_population, locals_vs_diaspora, person_vs_organization) are trained,
evaluated (K-Fold + LOOCV) and logged in the experiment file every iteration.

Each iteration is TWO commands with your manual labeling in between:

    # Phase A — pick the users the model is least sure about and export them:
    python active_learning.py --iteration 3 --phase A

    #   -> open  Iteration_3/iteration_3_users_to_label.csv
    #   -> for each user, open the profile_url in X, fill target_population,
    #      locals_vs_diaspora and person_vs_organization (0/1/2). Save the file.

    # Phase C — fold the new labels in, retrain the full sweep, refresh the plot:
    python active_learning.py --iteration 3 --phase C

Deliverables produced per iteration (all inside Iteration_N/):
    iteration_N_unlabeled_users_predictions.csv   (Phase A, PDF page 14)
    iteration_N_users_to_label.csv                (Phase A, top-100 to label)
    iteration_N_manual_labels_target_population.csv        (Phase C, PDF page 15)
    iteration_N_manual_labels_locals_vs_diaspora.csv       (Phase C, PDF page 15)
    iteration_N_manual_labels_person_vs_organization.csv   (Phase C, PDF page 15)
    iteration_N_combined_labeled.csv              (Phase C, the growing training set)
    experiments_results_iteration_N.csv           (Phase C, the full sweep)
And project-wide:
    iteration_comparison_summary.csv              (Phase C, one row per iteration)
    plot_iteration_comparison.png                 (Phase C, PDF page 16 trend)

Design choices worth defending in the report:
  * Uncertainty sampling uses the best NUMERIC-feature target_population model
    from the previous iteration. Numeric features need no translation, so Phase A
    runs in seconds — and numeric was in fact the honest winner for this task.
  * The retrain sweep still tries all 9 feature sets (incl. TF-IDF on translated
    text) so the comparison across iterations stays apples-to-apples.
"""

from __future__ import annotations
import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)

warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    XGBOOST_OK = True
except Exception as _e:                                    # pragma: no cover
    XGBOOST_OK = False
    print(f"[warn] XGBoost unavailable ({type(_e).__name__}); it will be skipped.")

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
HERE      = Path(__file__).resolve().parent                # Classification/
DATA_DIR  = HERE.parent / 'data'
POOL_FILE = DATA_DIR / 'Candidates_user_data_MERGED.csv'   # 946 candidate users
ITER1_CONSENSUS = HERE / 'Iteration_1' / 'Step5_Analysis' / 'iteration_1_consensus_translated.csv'

# All three classification tasks. Each one is run BOTH ways by the sweep:
#   3-class -> keep every row, labels 0 / 1 / 2 (2 = unknown)
#   2-class -> drop the rows labelled 2, leaving only 0 / 1
# So you always label with 0/1/2 as usual; the code decides when to ignore the 2s.
TASKS       = ['target_population', 'locals_vs_diaspora', 'person_vs_organization']
BATCH_SIZE  = 100                                          # users to label per iteration
RANDOM_SEED = 42

COMMON_COLS = ['username', 'display_name', 'description', 'location',
               'followers_count', 'following_count', 'statuses_count', 'created_at']

NUMERIC_FEATURES = [
    'followers_count', 'following_count', 'statuses_count',
    'followers_following_ratio', 'bio_length', 'account_age_years',
    'has_description', 'has_location',
    'bio_mentions_iran', 'name_mentions_iran', 'location_mentions_iran',
]

ALGOS = ['LogReg', 'DecisionTree', 'RandomForest', 'SVM', 'AdaBoost']
if XGBOOST_OK:
    ALGOS.append('XGBoost')

# PDF Step 5 & Step 6 require every experiment log to contain BOTH balance modes.
# We therefore sweep balanced=True AND balanced=False. (The cross-iteration comparison
# and the deployed uncertainty model still use the balanced winner, since class_weight
# ='balanced' is consistently better on this imbalanced data — but both are logged.)
BALANCED_MODES = [True, False]

CSV_COL_ORDER = [
    'iteration', 'target_column', '#classes', '#class_0', '#class_1', '#class_2',
    'min_class_size', 'training_type', 'K', 'algorithm', 'feature_set',
    'Features_count', 'balanced', 'accuracy', 'precision', 'recall', 'F1', 'AUC',
]

IRAN_KEYWORDS = [
    # English
    'iran', 'iranian', 'persian', 'persia', 'tehran', 'shiraz', 'esfahan',
    'isfahan', 'mashhad', 'tabriz', 'kerman', 'qom', 'farsi',
    # Persian / Farsi
    'ایران', 'ایرانی', 'تهران', 'شیراز', 'اصفهان', 'مشهد', 'تبریز', 'فارسی',
    # Arabic
    'إيران', 'ايران', 'إيراني', 'ايراني', 'طهران', 'شيراز',
]


# ----------------------------------------------------------------------------
# Feature engineering (shared by Phase A and Phase C)
# ----------------------------------------------------------------------------
def _has_iran(text) -> int:
    if pd.isna(text):
        return 0
    s = str(text).lower()
    return int(any(kw in s for kw in IRAN_KEYWORDS))


def build_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the 11 numeric features. Multilingual keywords on RAW text —
    identical definition for labeled and unlabeled rows, no translation needed."""
    df = df.copy()
    for c in ['followers_count', 'following_count', 'statuses_count']:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    df['bio_length'] = df['description'].fillna('').astype(str).str.len()
    df['followers_following_ratio'] = df['followers_count'] / (df['following_count'] + 1)
    dt = pd.to_datetime(df['created_at'], format='%B %Y', errors='coerce')
    df['account_age_years'] = (pd.Timestamp.today() - dt).dt.days / 365.25
    df['has_description'] = df['description'].notna().astype(int)
    df['has_location'] = df['location'].notna().astype(int)
    df['bio_mentions_iran'] = df['description'].apply(_has_iran)
    df['name_mentions_iran'] = df['display_name'].apply(_has_iran)
    df['location_mentions_iran'] = df['location'].apply(_has_iran)
    return df


def _norm_user(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().str.strip()


def _balanced_mask(df: pd.DataFrame) -> pd.Series:
    """True where balanced==True, robust to bool or 'True'/'False' string storage."""
    return df['balanced'].astype(str).str.lower().isin(['true', '1'])


# ----------------------------------------------------------------------------
# Loading the growing labeled set / the unlabeled pool
# ----------------------------------------------------------------------------
def combined_path(n: int) -> Path:
    return HERE / f'Iteration_{n}' / f'iteration_{n}_combined_labeled.csv'


def _n_labeled_for(n: int) -> int:
    """Actual size of the labeled set at iteration n. Iterations 1..6 add a full
    BATCH_SIZE each (100, 200, ... 600), but step 7 folds in only 40 band-samples
    (-> 640), so read the real file size instead of assuming 100*n."""
    if n == 1 and ITER1_CONSENSUS.exists():
        return int(len(pd.read_csv(ITER1_CONSENSUS)))
    p = combined_path(n)
    if p.exists():
        return int(len(pd.read_csv(p)))
    return 100 * n


def load_prev_labeled(iteration: int) -> pd.DataFrame:
    """Labeled set going INTO `iteration` = the combined set from iteration-1.
    Falls back to the iteration-1 consensus file if the combined file is absent."""
    prev = iteration - 1
    p = combined_path(prev)
    if p.exists():
        df = pd.read_csv(p)
    elif prev == 1:
        df = pd.read_csv(ITER1_CONSENSUS)
    else:
        raise FileNotFoundError(
            f"Cannot find labeled set for iteration {iteration}. "
            f"Expected {p} (run Phase C of iteration {prev} first).")
    keep = [c for c in COMMON_COLS + TASKS if c in df.columns]
    out = df[keep].copy()
    for t in TASKS:
        out[t] = pd.to_numeric(out[t], errors='coerce').fillna(2).astype(int)
    return out


def load_unlabeled_pool(labeled: pd.DataFrame) -> pd.DataFrame:
    pool = pd.read_csv(POOL_FILE)
    labeled_keys = set(_norm_user(labeled['username']))
    unl = pool[~_norm_user(pool['username']).isin(labeled_keys)].reset_index(drop=True)
    return unl


# ----------------------------------------------------------------------------
# Model factory (shared)
# ----------------------------------------------------------------------------
def make_model(algo: str, balanced: bool, training_type: str | None = None):
    cw = 'balanced' if balanced else None
    if algo == 'LogReg':
        return LogisticRegression(max_iter=2000, class_weight=cw, random_state=RANDOM_SEED)
    if algo == 'DecisionTree':
        return DecisionTreeClassifier(class_weight=cw, random_state=RANDOM_SEED)
    if algo == 'RandomForest':
        return RandomForestClassifier(n_estimators=100, class_weight=cw,
                                      random_state=RANDOM_SEED, n_jobs=-1)
    if algo == 'SVM':
        prob = (training_type != 'LOOCV')      # keep AUC under K-Fold; drop it under LOOCV for speed
        return SVC(kernel='linear', probability=prob, class_weight=cw, random_state=RANDOM_SEED)
    if algo == 'AdaBoost':
        return AdaBoostClassifier(random_state=RANDOM_SEED)
    if algo == 'XGBoost':
        return XGBClassifier(eval_metric='mlogloss', random_state=RANDOM_SEED,
                             verbosity=0, n_jobs=-1)
    raise ValueError(f"Unknown algorithm: {algo}")


def _sample_weights(y):
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    w = {c: len(y) / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    return np.array([w[v] for v in y])


# ============================================================================
# PHASE A — predict on the unlabeled pool, rank by uncertainty, export
# ============================================================================
def pick_best_target_model(iteration: int) -> tuple[str, bool]:
    """Read the previous iteration's results; pick the best NUMERIC-feature
    model for target_population (highest AUC). Default LogReg/unbalanced."""
    prev = iteration - 1
    res = HERE / f'Iteration_{prev}' / f'experiments_results_iteration_{prev}.csv'
    if res.exists():
        df = pd.read_csv(res)
        cand = df[(df['target_column'] == 'target_population') &
                  (df['feature_set'] == 'numeric') &
                  (_balanced_mask(df)) &
                  (df['AUC'].notna())]
        if len(cand):
            best = cand.sort_values('AUC', ascending=False).iloc[0]
            return str(best['algorithm']), True          # always balanced per instructor
    return 'LogReg', True


def phase_a(iteration: int) -> None:
    print(f"\n=== Phase A — iteration {iteration}: predict + pick most uncertain ===")
    iter_dir = HERE / f'Iteration_{iteration}'
    iter_dir.mkdir(exist_ok=True)

    labeled = load_prev_labeled(iteration)
    unlabeled = load_unlabeled_pool(labeled)
    print(f"labeled (training) : {len(labeled)}")
    print(f"unlabeled pool     : {len(unlabeled)}")

    labeled = build_numeric(labeled)
    unlabeled = build_numeric(unlabeled)
    med = labeled['account_age_years'].median()
    labeled['account_age_years'] = labeled['account_age_years'].fillna(med)
    unlabeled['account_age_years'] = unlabeled['account_age_years'].fillna(med)

    scaler = StandardScaler()
    X_lab = scaler.fit_transform(labeled[NUMERIC_FEATURES].values)
    X_unl = scaler.transform(unlabeled[NUMERIC_FEATURES].values)
    y_lab = labeled['target_population'].astype(int).values

    algo, balanced = pick_best_target_model(iteration)
    print(f"uncertainty model  : {algo} (balanced={balanced}) on numeric features")
    model = make_model(algo, balanced)
    model.fit(X_lab, y_lab)

    proba = model.predict_proba(X_unl)
    predicted = model.predict(X_unl)
    proba_full = np.zeros((proba.shape[0], 3))
    for j, c in enumerate(model.classes_):
        proba_full[:, int(c)] = proba[:, j]
    confidence = proba_full.max(axis=1)
    uncertainty = 1.0 - confidence

    pred = unlabeled[[c for c in COMMON_COLS if c in unlabeled.columns]].copy()
    pred.insert(1, 'profile_url', 'https://x.com/' + pred['username'].astype(str))
    pred['predicted_class'] = predicted
    pred['confidence_level'] = confidence.round(4)
    pred['prob_0'] = proba_full[:, 0].round(4)
    pred['prob_1'] = proba_full[:, 1].round(4)
    pred['prob_2'] = proba_full[:, 2].round(4)
    pred['uncertainty_score'] = uncertainty.round(4)
    pred = pred.sort_values('uncertainty_score', ascending=False).reset_index(drop=True)

    pred_path = iter_dir / f'iteration_{iteration}_unlabeled_users_predictions.csv'
    pred.to_csv(pred_path, index=False)
    print(f"saved predictions  : {pred_path.relative_to(HERE)}  ({len(pred)} rows)")

    to_label_path = iter_dir / f'iteration_{iteration}_users_to_label.csv'
    if to_label_path.exists():
        print(f"[skip] {to_label_path.name} already exists — not overwriting your labels.")
    else:
        to_label = pred.head(BATCH_SIZE).copy()
        for t in TASKS:
            to_label[t] = ''
        to_label['comments'] = ''
        to_label.to_csv(to_label_path, index=False)
        print(f"saved to-label     : {to_label_path.relative_to(HERE)}  (top {BATCH_SIZE})")

    print("\nMost uncertain 5 (your first picks):")
    print(pred[['username', 'profile_url', 'predicted_class',
                'confidence_level', 'uncertainty_score']].head(5).to_string(index=False))
    print(f"\nNEXT: label the {BATCH_SIZE} users in {to_label_path.name}, then run "
          f"`--iteration {iteration} --phase C`.")


# ============================================================================
# Translation (Phase C helper) — best-effort with a growing cache
# ============================================================================
def load_translation_map() -> dict:
    """username(lower) -> (description_en, display_name_en) from every translated
    CSV already on disk (iter-1 consensus, every iteration_N_combined_translated)."""
    m = {}
    files = [ITER1_CONSENSUS] + sorted(HERE.glob('Iteration_*/iteration_*_combined_translated.csv'))
    for f in files:
        if not f.exists():
            continue
        d = pd.read_csv(f)
        if 'description_en' not in d.columns:
            continue
        key = _norm_user(d['username'])
        for k, de, ne in zip(key, d.get('description_en', ''), d.get('display_name_en', '')):
            m[k] = (de, ne)
    return m


def translate_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill description_en / display_name_en; translate only rows not in the cache."""
    df = df.copy()
    cache = load_translation_map()
    key = _norm_user(df['username'])
    df['description_en'] = [cache.get(k, (np.nan, np.nan))[0] for k in key]
    df['display_name_en'] = [cache.get(k, (np.nan, np.nan))[1] for k in key]

    need = df['description_en'].isna() | df['display_name_en'].isna()
    if need.any():
        print(f"translating {int(need.sum())} new rows (best-effort)...")
        try:
            from deep_translator import GoogleTranslator

            def tr(t):
                if pd.isna(t) or str(t).strip() == '':
                    return ''
                try:
                    return GoogleTranslator(source='auto', target='en').translate(str(t)[:4500])
                except Exception:
                    return str(t)
            for i in df.index[need]:
                df.at[i, 'description_en'] = tr(df.at[i, 'description'])
                df.at[i, 'display_name_en'] = tr(df.at[i, 'display_name'])
        except Exception as e:
            print(f"[warn] translator unavailable ({type(e).__name__}); "
                  f"falling back to raw text for untranslated rows.")
            df['description_en'] = df['description_en'].fillna(df['description'])
            df['display_name_en'] = df['display_name_en'].fillna(df['display_name'])
    df['description_en'] = df['description_en'].fillna(df['description']).fillna('')
    df['display_name_en'] = df['display_name_en'].fillna(df['display_name']).fillna('')
    return df


def build_feature_sets(df: pd.DataFrame) -> dict:
    """The 9 feature sets: 7 TF-IDF + numeric + desc+numeric."""
    def tfidf(cols, min_df):
        text = df[cols[0]].fillna('').astype(str)
        for c in cols[1:]:
            text = text + ' ' + df[c].fillna('').astype(str)
        vec = TfidfVectorizer(max_features=300, lowercase=True,
                              stop_words='english', min_df=min_df)
        return vec.fit_transform(text)

    sets = {
        'desc':               tfidf(['description_en'], 2),
        'username':           tfidf(['username'], 1),
        'fullname':           tfidf(['display_name_en'], 1),
        'desc_user':          tfidf(['description_en', 'username'], 2),
        'desc_fullname':      tfidf(['description_en', 'display_name_en'], 2),
        'user_fullname':      tfidf(['username', 'display_name_en'], 1),
        'desc_user_fullname': tfidf(['description_en', 'username', 'display_name_en'], 2),
    }
    scaler = StandardScaler(with_mean=False)
    numeric = sp.csr_matrix(scaler.fit_transform(df[NUMERIC_FEATURES].values))
    sets['numeric'] = numeric
    sets['desc+numeric'] = sp.hstack([sets['desc'], numeric]).tocsr()
    return sets


# ============================================================================
# The experiment sweep (Phase C core) — ported from Step 5, 2 tasks
# ============================================================================
def _class_counts(y):
    c = pd.Series(y).value_counts().to_dict()
    return {0: int(c.get(0, 0)), 1: int(c.get(1, 0)), 2: int(c.get(2, 0))}


def run_one_experiment(X, y, algo, balanced, training_type,
                       target_column, feature_set_name, n_classes, iteration):
    y = np.asarray(y)
    if training_type == 'K-Fold':
        cv, K_val = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED), 5
    else:
        cv, K_val = LeaveOneOut(), ''

    all_true, all_pred, proba_chunks = [], [], []
    accuracy = precision = recall = f1 = auc = None
    try:
        for tr_idx, te_idx in cv.split(np.zeros(len(y)), y):
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]
            if len(np.unique(y_tr)) < 2:
                continue
            model = make_model(algo, balanced, training_type=training_type)
            fit_kw = {}
            if balanced and algo in ('AdaBoost', 'XGBoost'):
                fit_kw['sample_weight'] = _sample_weights(y_tr)
            model.fit(X_tr, y_tr, **fit_kw)
            all_pred.extend(model.predict(X_te))
            all_true.extend(y_te)
            if hasattr(model, 'predict_proba'):
                proba_chunks.append((model.classes_, model.predict_proba(X_te), list(te_idx)))
        if not all_pred:
            raise RuntimeError("no predictions")
        accuracy = accuracy_score(all_true, all_pred)
        precision = precision_score(all_true, all_pred, average='weighted', zero_division=0)
        recall = recall_score(all_true, all_pred, average='weighted', zero_division=0)
        f1 = f1_score(all_true, all_pred, average='weighted', zero_division=0)
        if proba_chunks:
            gclasses = sorted(set(y))
            pf = np.zeros((len(y), len(gclasses)))
            for classes_seen, block, idxs in proba_chunks:
                cmap = [gclasses.index(c) for c in classes_seen]
                for k, idx in enumerate(idxs):
                    for j, col in enumerate(cmap):
                        pf[idx, col] = block[k, j]
            try:
                auc = (roc_auc_score(y, pf[:, 1]) if len(gclasses) == 2
                       else roc_auc_score(y, pf, multi_class='ovr', average='weighted'))
            except Exception:
                auc = None
    except Exception:
        pass

    counts = _class_counts(y)
    nz = [v for k, v in counts.items() if (n_classes == 3 or k != 2) and v > 0]
    return {
        'iteration': iteration, 'target_column': target_column, '#classes': n_classes,
        '#class_0': counts[0], '#class_1': counts[1],
        '#class_2': counts[2] if n_classes == 3 else 0,
        'min_class_size': min(nz) if nz else 0,
        'training_type': training_type, 'K': K_val, 'algorithm': algo,
        'feature_set': feature_set_name, 'Features_count': X.shape[1],
        'balanced': balanced, 'accuracy': accuracy, 'precision': precision,
        'recall': recall, 'F1': f1, 'AUC': auc,
    }


def run_sweep(df: pd.DataFrame, feature_sets: dict, iteration: int, out_csv: Path):
    """Full sweep with resume + incremental save. Returns the results DataFrame."""
    tasks = [(t, df[t].astype(int).values) for t in TASKS]

    if out_csv.exists():
        existing = pd.read_csv(out_csv)
        results = existing.to_dict('records')
        done = {(r['target_column'], int(r['#classes']), r['algorithm'],
                 r['feature_set'], r['training_type'], bool(r['balanced']))
                for _, r in existing.iterrows()}
        print(f"resuming: {len(results)} rows already present")
    else:
        results, done = [], set()

    def save():
        pd.DataFrame(results)[CSV_COL_ORDER].to_csv(out_csv, index=False)

    t0, new = time.time(), 0
    for tname, y_full in tasks:
        for n_classes in (3, 2):
            if n_classes == 2:
                mask = (y_full != 2)
                y = y_full[mask]
            else:
                mask, y = None, y_full
            if len(np.unique(y)) < 2:
                print(f"  skip {tname} ({n_classes}cls): single class")
                continue
            for fname, Xf in feature_sets.items():
                X = Xf[mask] if mask is not None else Xf
                for algo in ALGOS:
                    for tt in ('K-Fold', 'LOOCV'):
                        for balanced in BALANCED_MODES:
                            key = (tname, n_classes, algo, fname, tt, bool(balanced))
                            if key in done:
                                continue
                            results.append(run_one_experiment(
                                X, y, algo, balanced, tt, tname, fname, n_classes, iteration))
                            done.add(key)
                            new += 1
                            if new % 50 == 0:
                                save()
                                print(f"  ... {new} new ({len(results)} total) "
                                      f"{time.time()-t0:.0f}s")
    save()
    print(f"sweep done: {new} new experiments, {len(results)} total, {time.time()-t0:.0f}s")
    return pd.DataFrame(results)


# ============================================================================
# Cross-iteration comparison (Phase C tail) — PDF page 16
# ============================================================================
def _results_csv_for(n: int) -> Path | None:
    for p in (HERE / f'Iteration_{n}' / f'experiments_results_iteration_{n}.csv',
              HERE / f'experiments_results_iteration_{n}.csv'):
        if p.exists():
            return p
    return None


def best_auc_per_task(df: pd.DataFrame) -> dict:
    """The HONEST winner's AUC per task = highest AUC among **K-Fold** experiments.

    We deliberately ignore LOOCV here: on this tiny, imbalanced data LOOCV inflates
    AUC to ~1.0 by overfitting the rare class (a well-known artifact). StratifiedKFold
    is the trustworthy generalization estimate, so the cross-iteration trend uses it."""
    out = {}
    for t in TASKS:
        d = df[(df['target_column'] == t) &
               (df['training_type'] == 'K-Fold') &
               (df['AUC'].notna())]
        out[t] = round(float(d['AUC'].max()), 4) if len(d) else np.nan
    return out


def rebuild_comparison() -> pd.DataFrame:
    rows = []
    n = 1
    while True:
        p = _results_csv_for(n)
        if p is None:
            if n > 12:
                break
            n += 1
            continue
        df = pd.read_csv(p)
        df = df[df['target_column'].isin(TASKS) & _balanced_mask(df)]   # balanced=True only
        best = best_auc_per_task(df)
        row = {
            'iteration': n,
            'n_labeled_users': _n_labeled_for(n),   # real size (step 7 adds 40, not 100)
            'n_experiments': len(df),
            'mean_accuracy': round(df['accuracy'].mean(), 4),
            'mean_F1': round(df['F1'].mean(), 4),
            'mean_AUC': round(df['AUC'].mean(), 4),
        }
        for t in TASKS:
            row[f'best_AUC_{t}'] = best[t]
        rows.append(row)
        n += 1
    summary = pd.DataFrame(rows)
    if summary.empty:
        print("[warn] no results CSVs found for comparison")
        return summary

    out_summary = HERE / 'iteration_comparison_summary.csv'
    summary.to_csv(out_summary, index=False)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(summary['iteration'], summary['mean_accuracy'],
                 marker='o', lw=2, ms=10, color='steelblue')
    for _, r in summary.iterrows():
        axes[0].annotate(f"{r['mean_accuracy']:.3f}", (r['iteration'], r['mean_accuracy']),
                         textcoords='offset points', xytext=(8, 6), fontsize=10)
    axes[0].set_xticks(summary['iteration'])
    axes[0].set_xlabel('Iteration'); axes[0].set_ylabel('Mean Accuracy (all experiments)')
    axes[0].set_title('Overall mean accuracy by iteration'); axes[0].grid(alpha=0.3)
    axes[0].set_ylim(0, 1)

    for tgt in TASKS:
        axes[1].plot(summary['iteration'], summary[f'best_AUC_{tgt}'],
                     marker='o', lw=2, ms=9, label=tgt, color=TASK_COLORS[tgt])
    axes[1].axhline(0.5, ls='--', color='grey', alpha=0.6, label='random (AUC=0.5)')
    axes[1].set_xticks(summary['iteration'])
    axes[1].set_xlabel('Iteration'); axes[1].set_ylabel('Best model AUC')
    axes[1].set_title('Winner AUC per task by iteration (the real quality signal)')
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3); axes[1].set_ylim(0, 1)
    plt.tight_layout()
    out_plot = HERE / 'plot_iteration_comparison.png'
    plt.savefig(out_plot, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nComparison summary:\n{summary.to_string(index=False)}")
    print(f"\nsaved: {out_summary.name}, {out_plot.name}")
    return summary


# ============================================================================
# FIXED HOLD-OUT EVALUATION  —  the clean "model actually improved" curve
# ============================================================================
# Why this exists: uncertainty sampling adds the HARDEST users each round, so the
# cross-validation metric on the growing set can dip even as the model improves.
# The fix is a test set that never changes. We freeze a random, stratified sample
# from iteration 1 (iteration 1 was itself a random draw, so it's an unbiased test
# set), NEVER train on it, and score every iteration's model on it. Training data
# grows; the test set is fixed → a clean learning curve that rises.
#
# NOTE: this hold-out is only for MEASURING the trend. Your deployed "best model"
# still trains on ALL labeled data (that's what Phase A does).

HOLDOUT_FILE     = HERE / 'holdout_test_set.csv'
HOLDOUT_FRACTION = 0.30
# The config scored per task on the hold-out (task -> (feature_mode, algorithm)).
HOLDOUT_CONFIG = {
    'target_population':      ('numeric',      'LogReg'),        # numeric was the honest winner
    'locals_vs_diaspora':     ('desc+numeric', 'LogReg'),        # very few labels -> may be NaN early on
    'person_vs_organization': ('desc+numeric', 'RandomForest'),  # text matters for this task
}
TASK_COLORS = {
    'target_population':      '#1f77b4',
    'locals_vs_diaspora':     '#ff7f0e',
    'person_vs_organization': '#2ca02c',
}


def ensure_holdout() -> set:
    """Freeze the fixed hold-out once (stratified on target_population). Returns the
    set of lower-cased hold-out usernames. Idempotent — reuses the saved file."""
    if HOLDOUT_FILE.exists():
        return set(_norm_user(pd.read_csv(HOLDOUT_FILE)['username']))
    from sklearn.model_selection import train_test_split
    iter1 = pd.read_csv(ITER1_CONSENSUS)
    iter1['target_population'] = pd.to_numeric(
        iter1['target_population'], errors='coerce').fillna(2).astype(int)
    _, ho_idx = train_test_split(np.arange(len(iter1)), test_size=HOLDOUT_FRACTION,
                                 stratify=iter1['target_population'], random_state=RANDOM_SEED)
    ho = iter1.iloc[ho_idx][['username'] + TASKS].copy()
    ho.to_csv(HOLDOUT_FILE, index=False)
    print(f"froze hold-out: {len(ho)} users -> {HOLDOUT_FILE.name} "
          f"(target dist {ho['target_population'].value_counts().sort_index().to_dict()})")
    return set(_norm_user(ho['username']))


def _combined_with_text(n: int) -> pd.DataFrame:
    """Labeled set for iteration n, with description_en/display_name_en filled."""
    if n == 1:
        df = pd.read_csv(ITER1_CONSENSUS)
    else:
        df = pd.read_csv(combined_path(n))
    for t in TASKS:
        df[t] = pd.to_numeric(df[t], errors='coerce').fillna(2).astype(int)
    if 'description_en' not in df.columns:
        df = translate_missing(df)
    df = build_numeric(df)
    df['account_age_years'] = df['account_age_years'].fillna(df['account_age_years'].median())
    return df


def _featurize_eval(train, test, task, mode):
    """Build 2-class (drop unknown) features. TF-IDF is fit on TRAIN ONLY (no leakage).

    Robust to tiny tasks like locals_vs_diaspora, where the 2-class training set can be
    ~30 rows: min_df=2 would prune every TF-IDF term and raise. We retry with min_df=1
    and, failing that, quietly fall back to numeric-only features."""
    train = train[train[task] != 2]
    test  = test[test[task] != 2]
    if len(test) == 0 or train[task].nunique() < 2:
        return None
    num_sc = StandardScaler().fit(train[NUMERIC_FEATURES].values)
    Xtr_n = num_sc.transform(train[NUMERIC_FEATURES].values)
    Xte_n = num_sc.transform(test[NUMERIC_FEATURES].values)

    if mode != 'numeric':
        tr_txt = train['description_en'].fillna('').astype(str)
        te_txt = test['description_en'].fillna('').astype(str)
        for min_df in (2, 1):
            try:
                vec = TfidfVectorizer(max_features=300, lowercase=True,
                                      stop_words='english', min_df=min_df)
                Xtr_t = vec.fit_transform(tr_txt)
                Xte_t = vec.transform(te_txt)
                return (sp.hstack([Xtr_t, sp.csr_matrix(Xtr_n)]).tocsr(),
                        sp.hstack([Xte_t, sp.csr_matrix(Xte_n)]).tocsr(),
                        train[task].values, test[task].values)
            except ValueError:
                continue            # no terms survived pruning -> loosen, then give up on text
    return Xtr_n, Xte_n, train[task].values, test[task].values


def evaluate_holdout(n: int, holdout: set) -> dict:
    """Train each task's winner config on (iteration n's data MINUS hold-out),
    score on the fixed hold-out. Returns AUC / F1 / target-recall per task."""
    from sklearn.metrics import roc_auc_score, f1_score, recall_score
    df = _combined_with_text(n)
    key = _norm_user(df['username'])
    train_all, test_all = df[~key.isin(holdout)].copy(), df[key.isin(holdout)].copy()
    row = {'iteration': n, 'n_train_total': len(train_all)}
    for task, (mode, algo) in HOLDOUT_CONFIG.items():
        feats = _featurize_eval(train_all, test_all, task, mode)
        if feats is None:
            row[f'{task}_AUC'] = row[f'{task}_F1'] = row[f'{task}_recall1'] = np.nan
            continue
        Xtr, Xte, ytr, yte = feats
        m = make_model(algo, balanced=True)
        m.fit(Xtr, ytr)
        proba, pred = m.predict_proba(Xte), m.predict(Xte)
        try:
            row[f'{task}_AUC'] = round(float(roc_auc_score(yte, proba[:, 1])), 4)
        except Exception:
            row[f'{task}_AUC'] = np.nan
        row[f'{task}_F1'] = round(float(f1_score(yte, pred, average='weighted', zero_division=0)), 4)
        row[f'{task}_recall1'] = round(float(recall_score(
            yte, pred, labels=[1], average='macro', zero_division=0)), 4)
    return row


def holdout_trend() -> pd.DataFrame:
    """Score every iteration that has data on the fixed hold-out; save CSV + plot."""
    holdout = ensure_holdout()
    rows, n = [], 1
    while True:
        has = (n == 1 and ITER1_CONSENSUS.exists()) or combined_path(n).exists()
        if not has:
            if n > 12:
                break
            n += 1
            continue
        rows.append(evaluate_holdout(n, holdout))
        n += 1
    trend = pd.DataFrame(rows)
    if trend.empty:
        print("[warn] no data for hold-out trend")
        return trend
    out_csv = HERE / 'holdout_improvement_trend.csv'
    trend.to_csv(out_csv, index=False)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    metrics = [('_F1', 'F1 on fixed hold-out'), ('_recall1', 'Target-class recall on fixed hold-out')]
    for ax, (suf, title) in zip(axes, metrics):
        for task in TASKS:
            ax.plot(trend['iteration'], trend[f'{task}{suf}'],
                    marker='o', lw=2, ms=9, label=task, color=TASK_COLORS[task])
        ax.set_xticks(trend['iteration'])
        ax.set_xlabel('Iteration'); ax.set_ylabel(title.split(' on ')[0])
        ax.set_title(title); ax.grid(alpha=0.3); ax.set_ylim(0, 1.02); ax.legend(fontsize=9)
    plt.tight_layout()
    out_plot = HERE / 'plot_holdout_improvement.png'
    plt.savefig(out_plot, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nHold-out improvement trend (fixed {len(holdout)}-user test set):")
    print(trend.to_string(index=False))
    print(f"\nsaved: {out_csv.name}, {out_plot.name}")
    return trend


# ============================================================================
# PHASE C — merge new labels, retrain sweep, refresh comparison
# ============================================================================
def phase_c(iteration: int) -> None:
    print(f"\n=== Phase C — iteration {iteration}: merge + retrain + compare ===")
    iter_dir = HERE / f'Iteration_{iteration}'
    to_label_path = iter_dir / f'iteration_{iteration}_users_to_label.csv'
    if not to_label_path.exists():
        raise FileNotFoundError(f"{to_label_path} not found — run Phase A first.")

    manual = pd.read_csv(to_label_path)
    for t in TASKS:
        manual[t] = pd.to_numeric(manual[t], errors='coerce').fillna(2).astype(int)
    blanks = int((manual[TASKS] == 2).all(axis=1).sum())
    if blanks == len(manual):
        raise ValueError(
            f"All {len(manual)} rows in {to_label_path.name} are unlabeled (2). "
            f"Fill in the label columns before running Phase C.")
    print(f"loaded {len(manual)} manual labels ({blanks} rows still all-unknown)")
    for t in TASKS:
        print(f"  {t}: {manual[t].value_counts().sort_index().to_dict()}")

    # PDF page 15 — save the two manual-label CSVs
    for t in TASKS:
        cols = [c for c in COMMON_COLS if c in manual.columns] + [t]
        if 'comments' in manual.columns:
            cols.append('comments')
        manual[cols].to_csv(iter_dir / f'iteration_{iteration}_manual_labels_{t}.csv', index=False)

    # Combine: previous labeled set ∪ these new labels
    prev = load_prev_labeled(iteration)
    prev['iteration_added'] = prev.get('iteration_added', iteration - 1)
    new_slim = manual[[c for c in COMMON_COLS if c in manual.columns] + TASKS].copy()
    new_slim['iteration_added'] = iteration
    combined = pd.concat([prev[[c for c in COMMON_COLS if c in prev.columns] + TASKS + ['iteration_added']],
                          new_slim], ignore_index=True)
    combined = combined.drop_duplicates(subset='username', keep='first').reset_index(drop=True)
    combined.to_csv(combined_path(iteration), index=False)
    print(f"combined labeled set: {len(combined)} rows -> {combined_path(iteration).name}")
    for t in TASKS:
        print(f"  {t}: {combined[t].astype(int).value_counts().sort_index().to_dict()}")

    # Features (translate the new rows, build the 9 sets)
    combined = translate_missing(combined)
    combined.to_csv(iter_dir / f'iteration_{iteration}_combined_translated.csv', index=False)
    combined = build_numeric(combined)
    combined['account_age_years'] = combined['account_age_years'].fillna(
        combined['account_age_years'].median())
    feature_sets = build_feature_sets(combined)
    print(f"feature sets: {', '.join(f'{k}{v.shape}' for k, v in feature_sets.items())}")

    # Sweep + comparison
    out_csv = iter_dir / f'experiments_results_iteration_{iteration}.csv'
    run_sweep(combined, feature_sets, iteration, out_csv)
    rebuild_comparison()
    holdout_trend()          # the clean "model actually improved" curve
    print(f"\nDONE iteration {iteration}. Review plot_holdout_improvement.png "
          f"(the improvement curve) and plot_iteration_comparison.png, "
          f"then run Phase A for iteration {iteration + 1}.")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Active Learning engine (Step 6).")
    ap.add_argument('--iteration', type=int, default=None, help='iteration number (3,4,5,6,...)')
    ap.add_argument('--phase', choices=['A', 'C', 'H'], required=True,
                    help='A = predict+pick to label; C = merge+retrain+compare; '
                         'H = rebuild the fixed-hold-out improvement curve only')
    a = ap.parse_args()
    if a.phase == 'H':
        holdout_trend()
        return
    if a.iteration is None:
        ap.error("--iteration is required for phase A and C")
    (phase_a if a.phase == 'A' else phase_c)(a.iteration)


if __name__ == '__main__':
    main()
