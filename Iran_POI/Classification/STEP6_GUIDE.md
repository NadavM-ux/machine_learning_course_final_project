# Step 6 — Active Learning: A Learning Guide

> **⚠ UPDATED (current state).** `locals_vs_diaspora` was **dropped** per the instructor —
> the project is now **2 tasks** (`target_population`, `person_vs_organization`) and
> **864 experiments** per iteration (not 1,296). Iterations are no longer hand-coded per
> notebook; they run through **`active_learning.py`**. For the live iteration-3→6 workflow,
> read **`ITERATIONS_HOWTO.md`** — it supersedes the iteration-2-specific parts below.
> The conceptual sections (uncertainty sampling, phases A/B/C) below are still accurate.

> **Read this once end-to-end (~30 min) before building anything.** Then answer the §10 self-check questions. Only then open the notebook.

---

## 1. The big picture: what Step 6 is asking

Your Step 5 model is OK but not great. The bottleneck is **data**: you only have 100 labeled users, and 13 of those are the rare "target" class. The model is starved for examples.

The naive way to improve it: **label more random users**. Pick 100 unlabeled users at random, label them, retrain. Wastes effort — half of them are obvious cases the model already gets right.

The smart way is **Active Learning**: ask your Step 5 model "which users are you LEAST sure about?" and label *those* specifically. Each ambiguous user you label gives the model a maximum-information training example.

In short:

```
Step 5 winner model + 846 unlabeled users
        ↓
       predict probabilities for each
        ↓
       rank by uncertainty (closer to "I have no idea" = higher uncertainty)
        ↓
       hand-label the 100 most uncertain ones
        ↓
       add those 100 labels to the 100 you already had → 200 labeled
        ↓
       re-run the full Step 5 experiment loop
        ↓
       performance should improve. PDF wants a graph showing this.
```

The PDF (page 13) calls this whole thing one **iteration**. Iteration 1 = Step 5 (the 100 you started with). Iteration 2 = Step 6 (you now have 200). Iteration 3 = repeat for another 100 (you now have 300).

---

## 2. Step 6's three phases

| Phase | What you do | Where | Time |
|---|---|---|---|
| A. Predict + pick | Run Step 5 winner on 846 unlabeled users, compute uncertainty, save top 100 | `part6.ipynb` | ~10 min coding, ~1 min run |
| B. Manual labeling | You + partner label the 100 most uncertain users in Twitter (same workflow as Step 3) | Excel/Sheets + browser | ~3–4 hours |
| C. Combine + retrain | Add the 100 new labels to the 100 old → 200 total. Re-run the Step 5 loop with iteration=2. | `part6.ipynb` | ~30 min code + ~30 min loop |

Phase B is the unavoidable manual slog. Phases A and C are quick.

---

## 3. The math of uncertainty

The model's `.predict_proba()` returns a probability distribution over the classes. For a 3-class problem like `target_population`:

```
user @example: predict_proba → [0.40, 0.35, 0.25]
                                  ↑     ↑     ↑
                              non_   target  unknown
                              target
```

The model is saying: "I think this user has a 40% chance of being non_target, 35% target, 25% unknown."

The **confidence_level** is the highest of those three: `0.40`. The model's best guess.

The **uncertainty_score** is `1 - confidence_level = 0.60`. Higher = more uncertain.

Compare to a confident prediction:

```
user @clearcut: predict_proba → [0.95, 0.03, 0.02]
                confidence_level = 0.95
                uncertainty_score = 0.05   ← low, model is sure
```

The PDF says: sort all unlabeled users by `uncertainty_score` descending, and **label the top 100**.

---

## 4. Which Step 5 model to use

PDF page 14 (literal Hebrew):

> *"בחירת המודל הטוב ביותר — בחרו את המודל בעל הביצועים הטובים ביותר מהשלב הקודם (לדוגמה, המודל עם ה-AUC הגבוה ביותר). הפעילו אותו על הדאטאסט המלא שטרם תויג."*

Translation: **choose THE model (singular) with the best performance. Apply IT to the unlabeled dataset.**

So you pick ONE model — your Step 5 winner. From `iteration_1_winners_summary.csv`:

| Task | Algorithm | Features | AUC |
|---|---|---|---|
| **target_population** | **LogReg** | **numeric** | **0.747** ⭐ |
| locals_vs_diaspora | (degenerate) | — | 0.61 |
| person_vs_organization | AdaBoost | desc_fullname | 0.62 |

**Use target_population's winner.** Why?
- Highest AUC (real signal).
- It's the **primary task** anyway — predicting whether someone belongs to the target population.
- Uses only numeric features → **no translation needed for unlabeled users** (huge time-saver, see §5).
- The PDF wants ONE model — this is the natural choice.

The manual labelers in Phase B will fill in **all three** target columns (`target_population`, `locals_vs_diaspora`, `person_vs_organization`) regardless of which model picked the user. So the manual labels still grow all three tasks' training sets.

---

## 5. ⚠ The translation problem (and how to avoid it)

You have 100 labeled users with translated descriptions (`description_en`). The 846 unlabeled users do **not** have translations. The naive fix is to translate them all with Google Translate — but that takes **3+ hours** with free-tier rate limits.

### The smart workaround: skip translation entirely

The Step 5 winner for `target_population` uses **only numeric features** — including `bio_mentions_iran`, which was computed on translated text. If you compute `bio_mentions_iran` using a **multilingual keyword list directly on raw text**, you can avoid translating anything:

```python
iran_keywords_multilingual = [
    # English
    'iran', 'iranian', 'persian', 'persia', 'tehran', 'shiraz',
    'esfahan', 'isfahan', 'mashhad', 'tabriz', 'farsi',
    # Persian / Farsi
    'ایران', 'ایرانی', 'تهران', 'شیراز', 'اصفهان', 'فارسی',
    # Arabic
    'إيران', 'إيراني', 'طهران',
]
```

For consistency, **re-compute `bio_mentions_iran` on labeled users too** using this same multilingual rule (on raw `description`, not `description_en`). Then re-train the model. Now train and predict use exactly the same feature definition.

The whole Step 6 prediction pipeline becomes a **<10 second** notebook run instead of 3 hours.

Document this trade-off in your report: "We use multilingual keyword matching instead of translation for Step 6 to avoid the translation API rate limit. Feature definitions are consistent between labeled and unlabeled sets."

---

## 6. Phase A: predict + pick (the notebook code)

### 6.1 Data sources

| File | Rows | Purpose |
|---|---|---|
| `Iteration_1/Step5_Analysis/iteration_1_consensus_translated.csv` | 100 | The labeled users from Step 3 |
| `../data/Candidates_user_data_MERGED.csv` | 946 | The full candidate pool (this is the source your labeled 100 came from!) |

Important: the unlabeled pool is `Candidates_user_data_MERGED`, **NOT** `POI_twitter_users_data` (which is a different file). The 100 labeled users are a subset of the 946 candidates. So unlabeled = 946 − 100 = **846 users**.

### 6.2 Pipeline overview

```
1. Load labeled (100) and candidates pool (946).
2. Filter to unlabeled (846) by removing rows whose username is in labeled.
3. Compute 11 numeric features for BOTH labeled and unlabeled.
   - bio_mentions_iran uses multilingual keywords on RAW description (not translated).
4. Scale numeric features (StandardScaler).
5. Re-train LogReg + numeric (Step 5 winner) on the full 100 labeled.
6. Predict on 846 unlabeled → predicted_class, predict_proba.
7. Compute confidence_level and uncertainty_score.
8. Save iteration_2_unlabeled_users_predictions.csv with PDF page 14 columns.
9. Sort by uncertainty, take top 100, save iteration_2_users_to_label.csv (with empty label columns ready for manual filling).
```

### 6.3 Required CSV columns (PDF page 14)

```
username, display_name, description, location, followers_count, following_count,
statuses_count, created_at,            ← user info
predicted_class,                       ← 0, 1, or 2
confidence_level,                      ← max(prob_0, prob_1, prob_2)
prob_0, prob_1, prob_2,                ← probabilities per class
uncertainty_score                      ← 1 - confidence_level
```

Sort the rows by `uncertainty_score` descending (so the most uncertain user is row 0).

---

## 7. Phase B: manual labeling (the slog)

**This is the same workflow as Step 3.** You + your partner:

1. Open `iteration_2_users_to_label.csv` in Excel or Google Sheets.
2. For each of the 100 users, open `https://x.com/<username>` in your browser.
3. Inspect bio, location, posts, profile pic, language.
4. Fill in the three label columns:
   - `target_population` → 0 (non_target) / 1 (target) / 2 (unknown)
   - `locals_vs_diaspora` → 0 (diaspora) / 1 (local) / 2 (unknown). Only matters if target_population=1.
   - `person_vs_organization` → 0 (organization) / 1 (person) / 2 (unknown)
5. Add a `comments` note when you want to explain a non-obvious decision.
6. **Both annotators label the same 100 users independently.** Then merge — `auto_agree` for matching rows, discussion for disagreements (same Double Annotation process as Step 3).
7. Compute Cohen's Kappa + percent agreement for iteration 2, like you did in Step 3.

The PDF (page 15) wants the result saved as **3 separate files**:

```
Iteration_2/iteration_2_manual_labels_target_population.csv
Iteration_2/iteration_2_manual_labels_locals_vs_diaspora.csv
Iteration_2/iteration_2_manual_labels_person_vs_organization.csv
```

Time estimate: **3–4 hours of two-person work.** Don't underestimate.

---

## 8. Phase C: combine + re-train + compare

After manual labeling:

### 8.1 Combine the data
```
labeled_iteration_2 = labeled_iteration_1 ∪ new 100 manual labels
                    = 200 labeled users
```

### 8.2 Re-run the Step 5 experiment loop
With `iteration = 2` in every row. Save to `experiments_results_iteration_2.csv`. You're literally calling the same `run_one_experiment` function from your Step 5 Cell 6, just on the new 200-user dataset.

**Heads-up on speed:** LOOCV scales with N. With 200 users, LOOCV does 200 fits per row instead of 100. So the loop runs roughly 2× slower — **expect ~60 minutes** instead of 30.

### 8.3 Compare iterations
PDF page 16 requires a graph:
- X-axis: iteration number (1, 2, ...)
- Y-axis: mean Accuracy across all classifiers and feature sets, per iteration

```
Iteration 1 → mean accuracy from experiments_results_iteration_1.csv
Iteration 2 → mean accuracy from experiments_results_iteration_2.csv
```

The graph should slope upward (more labels → better models). If it doesn't, that's still a valid finding worth discussing.

---

## 9. Files you produce in Step 6

```
Classification/
├── part6.ipynb                                              ← your code
├── iteration_1_winners_summary.csv                          ← from Step 5
├── experiments_results_iteration_1.csv                      ← from Step 5
├── experiments_results_iteration_2.csv                      ← NEW: 1,296 rows, iteration=2
├── iteration_2_winners_summary.csv                          ← NEW: optional helper
├── plot_iteration_comparison.png                            ← NEW: PDF mandatory
│
└── Iteration_2/
    ├── iteration_2_unlabeled_users_predictions.csv          ← PDF page 14 deliverable
    ├── iteration_2_users_to_label.csv                       ← top 100 for manual labeling
    ├── iteration_2_manual_labels_target_population.csv      ← PDF page 15 deliverable
    ├── iteration_2_manual_labels_locals_vs_diaspora.csv     ← PDF page 15 deliverable
    └── iteration_2_manual_labels_person_vs_organization.csv ← PDF page 15 deliverable
```

---

## 10. Self-check quiz

Try to answer each in your own words. If you can't, re-read the relevant section.

1. What's the difference between "label 100 random users" and "Active Learning"?
2. If `predict_proba` returns `[0.4, 0.35, 0.25]`, what's the uncertainty_score?
3. Why does the PDF say "THE model" (singular) instead of "the models"?
4. Why doesn't Step 6 need to translate the 846 unlabeled users?
5. The pool of 946 users comes from which file? Why not `POI_twitter_users_data.csv`?
6. What does Phase B require from YOU specifically (not the computer)?
7. After Phase C, why is the experiment loop expected to be ~2× slower than Step 5?
8. What axes does the PDF-required comparison graph use?
9. After Step 6 you'll have how many labeled users in total?
10. If iteration 2's accuracy doesn't improve over iteration 1, what should you do in your report?

---

## 11. Suggested notebook structure (`part6.ipynb`)

| Cell | Purpose |
|---|---|
| 1 | Imports + load labeled + load candidates pool + filter unlabeled (846 rows) |
| 2 | Compute 11 numeric features for BOTH labeled and unlabeled (multilingual Iran keywords on raw text) |
| 3 | Scale numeric features; re-train Step 5 winner (LogReg + numeric) on the 100 labeled |
| 4 | Predict on unlabeled → predicted_class, predict_proba → compute confidence + uncertainty |
| 5 | Save `iteration_2_unlabeled_users_predictions.csv` and `iteration_2_users_to_label.csv` (top 100) |
| 6 | **[after manual labeling]** Load the 3 new label CSVs and merge with iteration 1 labeled set → 200 rows |
| 7 | Re-run the Step 5 experiment loop on the 200-row set, save `experiments_results_iteration_2.csv` |
| 8 | Build the iteration comparison plot, save `plot_iteration_comparison.png` |

Cells 1–5 are short — together maybe 80 lines of code total. Cells 6–8 are short too. Cell 7 is the long-running one (~60 min).

---

## 12. Common pitfalls

1. **Loading the wrong pool file.** Use `Candidates_user_data_MERGED.csv` (946 rows), not `POI_twitter_users_data.csv` (343 unrelated rows). Verify by checking that your 100 labeled usernames are a subset.
2. **Forgetting to re-train the model.** You can't reuse the pickled Step 5 model directly because (a) the multilingual keyword feature has a slightly different definition than the English-only Step 5 version, (b) for predictions you should train on the **full** 100 labeled rows, not the 80 from a CV fold.
3. **`predict_proba` column order.** `model.classes_` might be `[0, 1, 2]` or it might be `[0, 2]` if a class is missing from training. Always pad to 3 columns based on `model.classes_`.
4. **Sorting by the wrong column.** Sort by `uncertainty_score` descending. Top of file = most uncertain = first to label.
5. **Manual label format mismatch.** Your manual labels CSV must have the same column types as the iteration 1 consensus file (integers 0/1/2 in the label columns).

---

## 13. When you're ready

1. Re-read this guide once.
2. Answer the §10 quiz out loud.
3. Open `part6.ipynb` and start building Cell 1.

Same rules as Step 5 — build incrementally. Don't paste the whole thing at once. Cell-by-cell, with sanity checks (print shape, count rows) after each.

When you finish Phase A, **don't proceed to Phase C until manual labeling is complete.** Phase B requires the manual work; Phase C cannot start without those 100 new labels.

Good luck.
