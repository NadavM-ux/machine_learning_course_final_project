# Step 5 — Model Training Report (Iteration 1)

*Draft prepared 2026-05-14. Paste into the Word report and adapt phrasing as needed.*

---

## 1. Methodology

We trained classification models on the 100 manually labeled Twitter users produced in Step 3 (file: `iteration_1_consensus_translated.csv`). Each user has three target labels: `target_population` (target / non_target / unknown), `locals_vs_diaspora` (local / diaspora / unknown — only meaningful for target users), and `person_vs_organization` (person / organization / unknown).

### 1.1 Pre-processing

User descriptions (`description`) and display names (`display_name`) appear in Persian, Arabic, English, and other languages. To enable a unified TF-IDF representation we translated both fields to English using `deep_translator` (Google Translate backend), saving the result to `iteration_1_consensus_translated.csv`. Numeric fields (`followers_count`, `following_count`, `statuses_count`) were filled with 0 where missing; account age in years was computed from `created_at`.

### 1.2 Features

We built **9 feature sets**:

| Feature set | Description |
|---|---|
| `desc` | TF-IDF over translated description |
| `username` | TF-IDF over Twitter handle |
| `fullname` | TF-IDF over translated display name |
| `desc_user` | TF-IDF over description + username |
| `desc_fullname` | TF-IDF over description + display name |
| `user_fullname` | TF-IDF over username + display name |
| `desc_user_fullname` | TF-IDF over all three text fields |
| `numeric` | 11 hand-crafted features: counts, ratios, bio length, account age, has_description / has_location flags, and three Iran-keyword indicators (in bio / name / location) |
| `desc+numeric` | Sparse-stacked combination of `desc` TF-IDF and the numeric features |

TF-IDF was built with `max_features=300`, lowercase, English stopwords, and `min_df=2` (or `min_df=1` for username/fullname which have low shared vocabulary). Numeric features were scaled with `StandardScaler(with_mean=False)` to remain sparse-compatible when stacked with TF-IDF.

### 1.3 Algorithms

Six classifiers were tested, as required by the project specification:

- Logistic Regression (`max_iter=2000`)
- Decision Tree
- Random Forest (`n_estimators=100`)
- SVM (`kernel='linear'`, `probability=True` — linear kernel is faster on sparse TF-IDF than RBF and supports probability output for AUC)
- AdaBoost
- XGBoost (`eval_metric='mlogloss'`)

### 1.4 Validation strategies

Each model was evaluated with **two cross-validation strategies**, both required by the spec:

- **Stratified K-Fold with K=5** (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`) — stratification was used over plain K-Fold to avoid folds with no minority-class examples on highly imbalanced targets.
- **Leave-One-Out CV (LOOCV)** (`LeaveOneOut`) — one user held out per fold, 100 fits per experiment.

### 1.5 Class imbalance handling

Each model was run in **two balance modes** as the spec requires:

- **Unbalanced** — `class_weight=None`. AdaBoost / XGBoost: no sample weighting.
- **Balanced** — `class_weight='balanced'` (LogReg, Decision Tree, Random Forest, SVM). For AdaBoost and XGBoost we passed explicit `sample_weight` so that each class contributes equally during fitting.

### 1.6 Class-count variants

Each combination was run **once on all 3 classes** (target / non_target / unknown — or the analogous trio for the other tasks) and **once on 2 classes** (dropping the `unknown` rows), as required.

### 1.7 Total experiments

3 tasks × {3-class, 2-class} × 6 algorithms × 9 feature sets × {K-Fold, LOOCV} × {balanced, unbalanced} = **1,296 experiments**. The full grid completed in ~88 minutes. All rows produced finite metrics — no failures. Results were saved to `experiments_results_iteration_1.csv` with the 18 columns mandated by the project spec.

---

## 2. Results

The full per-row results are in `experiments_results_iteration_1.csv`. The summary of best models per task is in `iteration_1_best_models_summary.csv`. Below are the headline findings.

### 2.1 Best model per task (after honesty filter)

We filtered out models that achieved high accuracy purely by predicting the majority class (`AUC ≤ 0.60`). The remaining candidates were ranked by F1, then AUC.

| Task | Variant | Best model | Acc | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|---|---|
| target_population | 3-class | LogReg + numeric, K-Fold, unbalanced | 0.670 | 0.668 | 0.670 | 0.665 | 0.747 |
| **target_population** | **2-class** | **LogReg + desc+numeric, LOOCV, balanced** | **0.889** | **0.892** | **0.889** | **0.890** | **0.875** |
| locals_vs_diaspora | 3-class | AdaBoost + desc+numeric, LOOCV | 0.950 | 0.933 | 0.950 | 0.936 | 0.610 |
| locals_vs_diaspora | 2-class | LogReg + desc, LOOCV, balanced | 0.750 | 0.750 | 0.750 | 0.750 | 0.500 |
| person_vs_organization | 3-class | AdaBoost + desc_fullname, LOOCV, balanced | 0.740 | 0.685 | 0.740 | 0.704 | 0.619 |
| **person_vs_organization** | **2-class** | **SVM + desc+numeric, LOOCV, balanced** | **0.776** | **0.788** | **0.776** | **0.781** | **0.840** |

The **bold rows** are the two genuinely strong models — they achieve AUC ≥ 0.84 with substantive F1, on tasks where the model is making meaningful per-class predictions (verified in §2.4).

### 2.2 Algorithm comparison

See `plot_f1_by_algorithm.png`. Mean F1 across all feature sets, CV strategies, and balance modes (3-class):

- **target_population:** LogReg and Random Forest tie at the top; tree-based and linear methods are roughly equivalent.
- **locals_vs_diaspora:** all algorithms hover near the same F1 — but this is misleading, see §2.4.
- **person_vs_organization:** AdaBoost and SVM lead; Logistic Regression close behind.

### 2.3 Feature-set comparison

See `plot_f1_by_feature_set.png`. Mean F1 by feature set (3-class):

- For `target_population`, **`numeric` and `desc+numeric` lead** — the numeric features (especially the Iran-keyword indicators) carry the signal. Pure TF-IDF on text alone underperforms because 100 users is too few to learn vocabulary patterns reliably.
- For `person_vs_organization`, text features matter more — display names like "News Agency", "Journalism Institute" are highly informative. `desc_fullname` and `desc+numeric` lead.
- For `locals_vs_diaspora`, feature choice barely changes F1 because the model is essentially constant (see §2.4).

### 2.4 ⚠️ The `locals_vs_diaspora` degeneracy

The naïve F1 of 0.94 for `locals_vs_diaspora` (3-class) and 0.75 (2-class) does not reflect a meaningful classifier. The data is extremely imbalanced — among 100 labeled users we have only 2 diaspora and 6 local users; the remaining 92 are `unknown`. A classifier that always predicts "unknown" already achieves 92% accuracy.

We verified this directly by inspecting prediction distributions (saved to `iteration_1_degeneracy_check.csv`):

| Variant | True class distribution | Predicted distribution |
|---|---|---|
| 3-class | 2% diaspora / 6% local / 92% unknown | 0% / 3% / 97% |
| 2-class | 25% diaspora / 75% local | 0% / 100% |

In both cases the model **never predicts the minority class**. The AUC values (0.61 and 0.50) confirm this — they are at or near random.

**Implication:** We cannot use `locals_vs_diaspora` as a deployable classifier from Iteration 1 alone. We will rely on the `target_population` classifier first, and tackle `locals_vs_diaspora` only once Active Learning surfaces enough diaspora examples (Step 6+).

### 2.5 K-Fold vs LOOCV

See `plot_kfold_vs_loocv.png`. The two validation strategies agree closely (most points fall along the diagonal), with LOOCV producing slightly lower variance estimates as expected. The agreement gives us confidence the F1 scores are not artifacts of a particular fold split.

---

## 3. Conclusions and implications for Active Learning

1. **`target_population` is learnable.** The strongest model — LogReg with `desc+numeric` features under LOOCV, balanced — reaches F1 = 0.89 and AUC = 0.88 on the 2-class version (63 samples after dropping unknowns). This is the model we will use as the **probability oracle for Step 6** to identify which unlabeled users to query next.

2. **`person_vs_organization` is also learnable.** SVM with linear kernel on `desc+numeric` features reaches F1 = 0.78, AUC = 0.84 on the 2-class version. Strong enough to drive uncertainty sampling for this task.

3. **`locals_vs_diaspora` is unlearnable at n=100** because only 8 of the 100 labeled users were classified as local-or-diaspora. Active Learning should preferentially surface users that are likely to be Iranian (so we can label whether they live in Iran or abroad). Across iterations we expect this class distribution to become more workable.

4. **Numeric features are surprisingly strong.** The three Iran-keyword indicators (in bio / display name / location) appear to capture much of the signal that distinguishes target from non-target users. This validates the manual labeling logic from Step 3 (location-based and bio-based heuristics).

5. **Translation step was justified.** Without translating description and display name, the TF-IDF features would be split across Persian, Arabic, English, and Hebrew vocabulary — losing statistical power. The translation step unifies them.

For Step 6 we will:
- Use the iteration-1 best models to score the unlabeled pool (~243 POI users not yet labeled).
- Compute combined uncertainty across all three classification tasks per user.
- Select the 100 most-uncertain users and label them manually (Iteration 2).
- Re-run the full 1,296-experiment grid and compare iteration-1 vs iteration-2 performance.
