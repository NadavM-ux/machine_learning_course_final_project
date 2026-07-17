# Active Learning — How to run iterations 3 → 6

> This is the **current, authoritative** guide for Step 6. It supersedes the
> iteration-2-specific parts of `STEP6_GUIDE.md`. All THREE tasks are trained and
> logged every iteration (target_population, locals_vs_diaspora, person_vs_organization).

The whole loop is now one script: **`active_learning.py`**. You never edit code
between iterations. Each iteration is two commands with your manual labeling in
the middle.

---

## The three classification tasks

| task | classes |
|---|---|
| `target_population` | 0 = non-target, 1 = target, 2 = unknown |
| `locals_vs_diaspora` | 0 = diaspora, 1 = local, 2 = unknown (only meaningful when target=1) |
| `person_vs_organization` | 0 = organization, 1 = person, 2 = unknown |

All three tasks are trained, evaluated (K-Fold + LOOCV) and logged in every
`experiments_results_iteration_N.csv`. `locals_vs_diaspora` is a small class
(most users are unknown=2), so its metrics are noisier — read them with caution.

---

## Where you are

| iteration | labeled users | status |
|---|---|---|
| 1 | 100 | ✅ done (Step 5) |
| 2 | 200 | ✅ done |
| **3** | **300** | ▶ **to-label file ready — start here** |
| 4 | 400 | pending |
| 5 | 500 | pending |
| 6 | 600 | pending |

The unlabeled pool has 746 users left, so 4 more rounds of 100 fit comfortably.

---

## Run an iteration (this is the whole workflow)

From inside the `Classification/` folder, using the project venv:

```bash
cd Iran_POI/Classification

# --- PHASE A: the model picks the 100 users it's least sure about ---
../../.venv/bin/python active_learning.py --iteration 3 --phase A
```

This writes `Iteration_3/iteration_3_users_to_label.csv` (top-100 most uncertain)
and the full `iteration_3_unlabeled_users_predictions.csv` (PDF page 14).

**Then YOU label (the human-in-the-loop part I can't do):**

1. Open `Iteration_3/iteration_3_users_to_label.csv` in Excel / Google Sheets.
2. For each of the 100 rows, click the `profile_url` (opens `https://x.com/<user>`).
3. Fill the three empty columns per the tables above:
   - `target_population` → 0 / 1 / 2
   - `locals_vs_diaspora` → 0 / 1 / 2
   - `person_vs_organization` → 0 / 1 / 2
   - optional `comments` — a note on why (helps resolve annotator disagreements).
4. Same discipline as Step 3: **only mark what you're 100% sure of; otherwise 2 (unknown).**
   Ideally both partners label independently, then reconcile (Double Annotation).
5. Save the file (keep the same name and location).

```bash
# --- PHASE C: fold the new labels in, retrain the full sweep, refresh the plot ---
../../.venv/bin/python active_learning.py --iteration 3 --phase C
```

Phase C automatically:
- saves the 2 manual-label CSVs (PDF page 15),
- builds `iteration_3_combined_labeled.csv` (= 300 rows),
- translates the new descriptions (cached; falls back to raw text if offline),
- runs all **1,296 experiments** (3 tasks × {3,2} classes × 9 feature sets × 6 algos ×
  {K-Fold, LOOCV} × {balanced, unbalanced}) → `experiments_results_iteration_3.csv`,
- rebuilds `iteration_comparison_summary.csv` and `plot_iteration_comparison.png`
  across **all** iterations so far.

Then repeat for `--iteration 4`, `5`, `6`. Same two commands, bump the number.

> **Runtime warning.** Phase C's sweep is dominated by LOOCV, which does *N* fits per
> experiment. At 300 rows expect ~1.5–2 h; at 600 rows ~3–4 h. It **saves progress
> every 50 experiments and resumes** if interrupted — safe to run in the background
> or stop/restart. If it's too slow on your machine, the honest, defensible shortcut
> is to comment `LOOCV` out of the `('K-Fold', 'LOOCV')` tuple in `run_sweep` for the
> intermediate iterations and keep it only for the final one (note this in the report).

---

## Reading the results the right way (important for your grade)

There are **three** numbers in `iteration_comparison_summary.csv`. They are not
equally trustworthy:

| column | what it is | trust |
|---|---|---|
| `mean_accuracy` / `mean_AUC` | average over **all 1,296** experiments | ⚠ blunt — dragged down by dozens of deliberately-bad configs. The PDF asks for the mean-accuracy plot, so keep it, but don't read quality into it. |
| `best_AUC_*` (per task) | the **K-Fold winner** per task | ✅ this is your real quality signal |
| LOOCV rows with AUC ≈ 0.95–1.0 | overfit artifacts on the tiny minority class | ❌ ignore — the "high-accuracy lie" |

**Why the winner AUC dipped from iteration 1 → 2** (0.90 → 0.79 for target):
that is *expected* and worth explaining in your report. Iteration 1's 100 users
were a **random** sample (many easy, clear-cut accounts). Iteration 2 added the
100 **most ambiguous** users on purpose — so the cross-validation test set is now
harder. The cross-validated AUC can dip even though the model is genuinely learning
the boundary better. The unambiguous win: the rare `target` class grew **13 → 43**
labeled examples, so the model is far less starved.

### ⭐ How to SHOW consistent improvement — the fixed hold-out curve

This is now built in and it is the chart you should lead with. Every Phase C also
runs `holdout_trend()` (or run it any time with `--phase H`), producing:

```
holdout_improvement_trend.csv       ← metrics per iteration on the fixed test set
plot_holdout_improvement.png        ← F1 + target-recall rising across iterations
holdout_test_set.csv                ← the 30 frozen test users (never trained on)
```

How it works and why it rises where the CV metric doesn't:
- We freeze **30 users from iteration 1** as a test set that **never changes** and is
  **never trained on**. Iteration 1 was a random draw, so this is an unbiased test set.
- Every iteration trains on all its labeled data *minus* those 30, then scores on the 30.
- Training data grows 70 → 170 → 270 → … while the test set is fixed ⇒ a clean learning
  curve. On the real iter1→iter2 data it already shows: `target_population` F1 **0.85→0.90**
  and target-class recall **0.75→1.00**.
- Improvement is clearest in **F1** and **target-class recall** (what active learning
  fixes), not raw AUC. Lead your report with these.

> The hold-out is **only for measuring the trend**. Your deployed "best model" (Phase A's
> uncertainty model, and whatever you name as final) still trains on **all** labeled data.

Fallback story if a task's curve is flat/noisy (small 30-user test set is noisy for the
rarer classes): report the `best_AUC_*` K-Fold trend from `iteration_comparison_summary.csv`
alongside it, and note that the definitive gain is the labeled minority class growing
(target: 13 → 43 → …).

---

## Making the model as good as possible (levers, in priority order)

1. **More labels on the minority class.** This is 80% of the gain. When you label,
   don't shy away from marking clear `target=1` / `organization=0` users — those rare
   classes are what's starving the model.
2. **Trust the K-Fold winner, tune it.** The winners so far are numeric-only LogReg/SVM
   for `target_population` and RandomForest on `desc+numeric` for `person_vs_organization`.
   Light hyperparameter tuning of *those* (C for LogReg/SVM, depth/estimators for RF) is
   worth more than adding exotic feature sets.
3. **2-class beats 3-class.** Dropping `unknown` (the 2-class runs) consistently scores
   higher and is a cleaner production model. Your final "best model" is almost certainly
   a 2-class one.
4. **Balanced usually wins, but both are logged.** The PDF requires both balanced and
   unbalanced runs, so the sweep logs both every iteration. Every cross-iteration
   comparison and the deployed winner use `balanced=True` (consistently better on this
   imbalanced data), but the unbalanced rows are on disk as required.
5. **Stop when it plateaus** (that's Step 7, don't do it yet): when the winner AUC stops
   moving > 0.5% between iterations, you've hit saturation.

---

## Files produced per iteration (all inside `Iteration_N/`)

```
iteration_N_unlabeled_users_predictions.csv     ← Phase A, PDF page 14
iteration_N_users_to_label.csv                  ← Phase A, the 100 you label
iteration_N_manual_labels_target_population.csv        ← Phase C, PDF page 15
iteration_N_manual_labels_locals_vs_diaspora.csv       ← Phase C, PDF page 15
iteration_N_manual_labels_person_vs_organization.csv   ← Phase C, PDF page 15
iteration_N_combined_labeled.csv                ← Phase C, the growing training set
iteration_N_combined_translated.csv             ← Phase C, translation cache
experiments_results_iteration_N.csv             ← Phase C, the 1,296-row sweep
```
Project-wide, refreshed every Phase C:
```
iteration_comparison_summary.csv                ← one row per iteration (mean + winner AUC)
plot_iteration_comparison.png                   ← PDF page 16 trend
holdout_improvement_trend.csv                   ← fixed-hold-out metrics per iteration
plot_holdout_improvement.png                    ← ⭐ the clean improvement curve to present
holdout_test_set.csv                            ← the 30 frozen test users
```

---

## Safety notes

- Phase A **won't overwrite** an existing `iteration_N_users_to_label.csv` — your
  manual labels are safe across re-runs.
- Phase C **refuses to run** if the to-label file is still entirely unlabeled (all 2s),
  so you can't accidentally poison the training set with 100 unknowns.
- Don't run Step 7 (stopping criteria) or Step 8 (LLM comparison) yet — finish the
  iterations first, as agreed.
