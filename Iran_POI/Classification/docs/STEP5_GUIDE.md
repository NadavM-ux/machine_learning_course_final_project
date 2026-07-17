# Step 5 — A Learning Guide

> **⚠ CURRENT STATE.** All **three** tasks (`target_population`, `locals_vs_diaspora`,
> `person_vs_organization`) are trained, evaluated (K-Fold + LOOCV) and logged every
> iteration, in **both** balanced and unbalanced modes — so the sweep is **1,296 rows**
> per iteration (3 × 2 classes × 9 feature sets × 6 algos × 2 validation × 2 balance).
> Note `locals_vs_diaspora` is a small class (most users are unknown=2), so read its
> metrics with caution — but it is fully included.

> **Don't write any code while reading this.** Read it through once, top to bottom, ~30 minutes. Then read it again. Only after you can answer the self-check questions in §8 should you open a notebook and start building.

---

## 1. The big picture: what Step 5 is asking

You have 100 hand-labeled Twitter users. The PDF wants you to build a machine that can predict the same labels automatically on users you haven't seen yet.

That's the whole goal.
v
The way the PDF asks you to do it is **scientific**: don't just pick one approach and hope. Try many combinations — different algorithms, different features, different evaluation methods, different ways of handling imbalanced classes — and **document every experiment in a CSV**. Then look at the CSV and pick the winner.

The PDF (page 10) breaks Step 5 into 6 sub-steps:

```
sub-step 1.  Feature extraction       → turn each user into a vector of numbers
sub-step 2.  Algorithm selection      → pick the ML model (6 required)
sub-step 3.  Validation strategy      → how do you measure performance fairly?
sub-step 4.  Train the model          → run model.fit()
sub-step 5.  Compute the metrics      → accuracy, precision, recall, F1, AUC
sub-step 6.  Save the row to a CSV    → one row per experiment
```

Sub-steps 1–5 are what you do for **one experiment**. Step 6 is the bookkeeping. You'll wrap sub-steps 1–5 in a loop and produce many many rows.

**The deliverable** that the grader will open: `experiments_results_iteration_1.csv` with every experiment as a row.

---

## 2. WHY we need machine learning at all

You already made flowcharts in Step 4. Couldn't you just code those flowcharts as if/else rules and skip all this ML?

You could. But:

- **Rules are brittle.** "If bio contains the word 'Iran', then target." Misses every user who wrote `ایرانی` instead. Misses every user who's Iranian but didn't say so explicitly.
- **Rules can't combine many weak signals.** A user with 1,000 followers, an Iranian display name, and an account 8 years old is *probably* Iranian. A rule for that is `if followers > 1000 AND name_iranian AND old`, but where do the thresholds come from? ML learns them from your labeled examples.
- **Rules don't tell you their confidence.** ML gives you a probability — "this user is 73% likely to be a target." That probability is what makes Active Learning possible in Step 6.

So: Step 5 is where the flowchart's logic becomes a learned, probabilistic, generalizable model.

---

## 3. Sub-step 1: Features — turning users into numbers

A machine-learning model can't read text. It can only do math on numbers. So your job is to **convert each user into a vector of numbers** that captures everything the model might need.

This conversion is called **feature engineering**.

There are two flavors of features.

### 3.1 Numeric features

These are already numbers (or trivial to compute):

```
followers_count
following_count
statuses_count
bio_length  (number of characters in the description)
followers / following ratio
account_age_years
has_description  (1 if the user has a bio, 0 if blank)
```

You can also compute domain-specific ones, like:

```
bio_mentions_iran    (1 if "Iran" or "Persian" or "Tehran" appears in bio)
location_mentions_iran
```

Each user becomes a row of numbers like `[542, 1200, 800, 87, 0.45, 7.2, 1, 1, 0]`.

### 3.2 Text features (TF-IDF)

Numbers like `bio_length` don't tell the model **what** the bio says. To capture the actual words, we use **TF-IDF**.

Here's TF-IDF in one paragraph: imagine a vocabulary of all words that appear across all 100 bios — maybe 500 words total. For each user, build a vector of length 500. Each position is a number that says "how prominent is THIS word in THIS user's bio, weighted by how rare the word is overall." Common words like "the" get small numbers. Distinctive words like "Tehran" get big numbers.

Concretely:

```
User 1 bio: "Iranian journalist based in Tehran"
User 2 bio: "Persian poet, lover of Hafez"
User 3 bio: "Hi I'm Sarah from California"

Vocabulary across all bios: [iranian, journalist, based, tehran, persian, poet, lover,
                              hafez, hi, sarah, california]

User 1's TF-IDF vector: [0.6, 0.4, 0.3, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
User 2's TF-IDF vector: [0.0, 0.0, 0.0, 0.0, 0.6, 0.5, 0.4, 0.7, 0.0, 0.0, 0.0]
User 3's TF-IDF vector: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.6, 0.5]
```

(The exact numbers come from the TF-IDF formula — you don't compute them by hand, `sklearn` does it.)

### 3.3 The PDF wants you to try multiple combinations

PDF page 12 suggests these 7 text combinations to try as **separate feature sets**:

1. TF-IDF on `description` only
2. TF-IDF on `username` only
3. TF-IDF on `display_name` only
4. TF-IDF on `description + username` (concatenated, then vectorized)
5. TF-IDF on `description + display_name`
6. TF-IDF on `username + display_name`
7. TF-IDF on `description + username + display_name`

Plus the numeric features as an 8th feature set. You can add a 9th — numeric stacked with TF-IDF on description — which is often the strongest.

**Why try so many?** Because you don't know in advance which features will best predict each task. The whole point of Step 5 is to find out empirically.

### 3.4 ⚠ Translation matters

If a user's bio is in Persian, TF-IDF will treat `ایران` and `Iran` as completely different words. So before TF-IDF you should **translate** the description and display name to English. The PDF (page 12) recommends this. The result of translation should be saved to a new column `description_en` so you don't re-translate every time you re-run.

You already have a copy of the consensus file — somewhere it would live in your Step 5 work, the column `description_en` should exist.

---

## 4. Sub-step 2: Six algorithms (PDF page 12)

The PDF requires you to try **at least these 6** algorithms. You don't need to deeply understand each — you need to be able to say "I tried them all, results are in the CSV." But here's a one-line summary so you can defend the choice:

| Algorithm | What it does |
|---|---|
| **Logistic Regression** | Draws a straight (well, linear-in-feature-space) boundary between classes. Simple, fast, often surprisingly strong. |
| **Decision Tree** | Recursive if/else splits. Easy to interpret. Tends to overfit on small data. |
| **Random Forest** | Average of many random Decision Trees. Reduces overfitting; almost always beats a single Tree. |
| **SVM** | Finds the boundary that maximizes the margin (gap) between classes. Great for text. |
| **AdaBoost** | Trains weak models sequentially, each one focusing on what the previous got wrong. |
| **XGBoost** | A more sophisticated gradient-boosting method. Often the top performer on tabular data. |

All 6 follow the same scikit-learn pattern:

```
model = SomeAlgorithm(...)    # create
model.fit(X_train, y_train)   # learn
model.predict(X_test)         # use
```

You can swap one for another without changing any other code.

**Heads-up:** XGBoost on macOS sometimes needs `brew install libomp`. If you get a "libomp.dylib not found" error, that's the fix.

---

## 5. Sub-step 3: Validation — how to measure performance fairly

After you train a model, you ask: how well does it do?

The naive answer: predict on the data you trained on, measure accuracy. **This is the most common mistake in ML.** The model has *seen* this data — of course it does well. You'd be measuring how well it memorized, not how well it generalizes.

The fix is **cross-validation**. The PDF demands two variants.

### 5.1 K-Fold (K=5)

Split your 100 users into 5 chunks of 20. For each chunk, train on the other 80 and predict on the 20 held out. Repeat 5 times so every user gets a turn as test. Average the metrics.

```
[Fold 1: train on chunks 2,3,4,5 → predict on chunk 1]
[Fold 2: train on chunks 1,3,4,5 → predict on chunk 2]
[Fold 3: train on chunks 1,2,4,5 → predict on chunk 3]
[Fold 4: train on chunks 1,2,3,5 → predict on chunk 4]
[Fold 5: train on chunks 1,2,3,4 → predict on chunk 5]

Concatenate the 5 sets of predictions → compute metrics on all 100.
```

Use **StratifiedKFold** (not plain KFold) so class proportions stay roughly equal in each fold. Important when your classes are imbalanced.

### 5.2 LOOCV (Leave-One-Out)

The extreme version: hold out exactly 1 user, train on 99, predict on 1. Repeat 100 times.

```
[Fold 1: train on rows 2..100 → predict on row 1]
[Fold 2: train on rows 1,3..100 → predict on row 2]
...
[Fold 100: train on rows 1..99 → predict on row 100]
```

Most thorough estimate but slow (100 training runs per experiment).

### 5.3 Why the PDF demands both

K-Fold and LOOCV are two ways to estimate the same thing (generalization performance). If they agree, you trust the result. If they disagree wildly, something's weird. Reporting both is a sanity check.

---

## 6. Sub-step 5 (intermediate): Class imbalance + class count

This is the trickiest concept in Step 5. Pay attention.

### 6.1 The imbalance problem

Your data has 50 non_target / 13 target / 37 unknown.

If you train a dumb model that just always predicts "non_target", it scores **50% accuracy** — without learning anything. If your data was 95 non_target / 5 target, that dumb model would score 95% accuracy. Looks fantastic, totally useless.

This is why **accuracy alone is a liar** for imbalanced data. You need other metrics (AUC, F1) AND you need to consider rebalancing.

### 6.2 Two ways to rebalance

**Way 1 — `class_weight='balanced'`.** Tells the model: "when you make a mistake on the rare class, penalize yourself harder." The math: each class contributes equally to the loss function, regardless of how many samples it has.

```python
LogisticRegression(class_weight='balanced')
```

Works for LogReg, DecisionTree, RandomForest, SVM. Doesn't work for AdaBoost or XGBoost — for those you pass `sample_weight` at fit time:

```python
sw = compute_sample_weight('balanced', y_train)
model.fit(X_train, y_train, sample_weight=sw)
```

**Way 2 — don't rebalance.** Pass `class_weight=None`. The model heavily favors the majority class.

The PDF demands you try **both versions** and record them in the CSV (the `balanced` column is True or False).

### 6.3 Why 2-class AND 3-class?

PDF page 12 says: train each model **on 3 classes** (including the "unknown" label) **AND on 2 classes** (drop the unknown rows).

Why both?

- **3-class** is more realistic (in production, you'll encounter users you can't tell about).
- **2-class** is cleaner (no ambiguous third option) and usually scores higher.

You record `#classes = 3` or `#classes = 2` in the CSV.

### 6.4 What "drop unknown" actually means

For `target_population`:
- 3-class: keep all 100 rows, predict 0/1/2.
- 2-class: keep only the 63 rows where label is 0 or 1, predict 0/1.

The model is trained and tested on the 63 rows in 2-class mode. The unknown rows are simply not part of the experiment.

---

## 7. Sub-step 5: The five metrics

The PDF wants 5 metrics per experiment. Plain-English meanings:

| Metric | What it answers | When it matters |
|---|---|---|
| **Accuracy** | "What % of predictions match the truth?" | When classes are balanced |
| **Precision** | "When the model says POSITIVE, how often is it right?" | When false alarms are costly |
| **Recall** | "Of the truly positive cases, what fraction did the model catch?" | When missing positives is costly |
| **F1** | Harmonic mean of precision and recall | When you want a balanced single number |
| **AUC** (Area Under ROC) | "How well does the model rank positives above negatives?" | Almost always — use it to catch lying-accuracy models |

A model with **accuracy = 0.95 but AUC = 0.50** is a model that just predicts the majority class. It learned nothing. The AUC reveals the lie.

For multi-class problems use `average='weighted'` for precision/recall/F1, and `multi_class='ovr', average='weighted'` for AUC.

---

## 8. Self-check quiz — can you answer these?

Before you write any code, try to answer each in your own words. If you can't, re-read the relevant section.

1. Why does the PDF want you to try 6 algorithms instead of just picking one?
2. What's the difference between numeric features and TF-IDF features?
3. Explain TF-IDF to a friend in two sentences.
4. What's wrong with measuring performance by predicting on the data you trained on?
5. In K-Fold with K=5, how many times does the model get trained per experiment?
6. Why do we test both `balanced=True` and `balanced=False`?
7. What does `#classes = 2` mean in practice — what changes about your data?
8. If a model has accuracy 0.92 and AUC 0.51, what's probably going on?
9. How is `experiments_results_iteration_1.csv` structured — what does each row represent?
10. Why is the file in §9's "winner" example sorted by F1 with `AUC > 0.6` as a filter?

---

## 9. The grand structure: how it all comes together

After all the sub-steps, **one experiment** looks like this:

```
1. Pick a task             (e.g. target_population)
2. Pick a #classes mode    (3 or 2)
3. Pick a feature set      (e.g. desc+numeric)
4. Pick an algorithm       (e.g. LogReg)
5. Pick a CV strategy      (e.g. K-Fold)
6. Pick balanced mode      (True or False)

7. Build the feature matrix X
8. Build the label vector y
9. Use the CV strategy to get cross-validated predictions
10. Compute the 5 metrics
11. Record one row in the CSV with everything: task, #classes, ..., metrics
```

That's it. To do all the experiments the PDF demands, you wrap this in 6 nested loops:

```
for task in [target_population, locals_vs_diaspora, person_vs_organization]:
    for n_classes in [3, 2]:
        for feature_set_name, X in feature_sets.items():
            for algorithm in 6_algorithms:
                for cv_strategy in [K-Fold, LOOCV]:
                    for balanced in [True, False]:
                        run_experiment(...)
                        append_row_to_results(...)
```

Multiply the lengths: **3 × 2 × 9 × 6 × 2 × 2 = 1,296 experiments** (if you have 9 feature sets). With 7 feature sets it's 1,008. Either is fine — the PDF doesn't mandate the exact number, only the exhaustive coverage.

Runtime: 30–90 minutes total. Save the CSV incrementally (every 50 rows) so a crash doesn't lose progress.

---

## 10. Reading the results

When the loop finishes, you have a CSV with ~1,200 rows. To find the winner per task, sort and filter:

```python
results = pd.read_csv('experiments_results_iteration_1.csv')

# Best model for target_population, 3-class
winners = (
    results
    .query("target_column == 'target_population' and `#classes` == 3 and AUC > 0.6")
    .sort_values(['F1', 'AUC'], ascending=False)
    .head(1)
)
print(winners)
```

The `AUC > 0.6` filter is critical. It excludes the "fake winners" — models that achieve high accuracy by predicting the majority class. Always look at AUC alongside accuracy.

---

## 11. The "high-accuracy lie" example you'll see

Your `locals_vs_diaspora` labels are 92 unknown / 6 local / 2 diaspora. Any model that predicts "unknown" for everyone scores 92% accuracy. The AUC for that model is about 0.5 (random).

When you scan your results CSV, you'll see rows with accuracy = 0.95 and F1 = 0.94 for locals_vs_diaspora — and AUC around 0.5. **Those are useless.** The model learned the majority class, not the task. Active Learning (Step 6) is where you fix this by adding more diaspora and local examples to your training set.

Your job in the report is to:
1. Identify the high-accuracy/low-AUC models in your CSV and call them out.
2. Pick the genuinely best (non-degenerate) model per task and use that as your "winner."

---

## 12. Suggested cell structure for your notebook

Here's a skeleton with no code — fill it in yourself:

| Cell | Purpose |
|---|---|
| 1 | Imports + load `iteration_1_labels_consensus.csv` (or the translated version if it exists) |
| 2 | Translate description + display_name to English if not done already (cache to file) |
| 3 | Build numeric features into new columns of `df` |
| 4 | Build the 7 TF-IDF feature sets — each gets a fresh `TfidfVectorizer` |
| 5 | Stack `desc TF-IDF + numeric` as a 9th feature set |
| 6 | Write a helper function `run_one_experiment(...)` that takes (X, y, algo, cv, balanced) and returns a dict with the metrics + the metadata for the CSV row |
| 7 | The 6 nested loops calling `run_one_experiment` for each combination, appending to a list. Save the CSV after every ~50 iterations so progress isn't lost. |
| 8 | Load the saved CSV, print top-5 by F1 per task, optionally filter for AUC > 0.6 |

**Build incrementally.** Don't write all of cell 7 at once. First do ONE experiment outside the loop. Verify the metrics look right. Then wrap it in `run_one_experiment`. Then test that function on 2–3 different inputs. Only then add the loop structure.

---

## 13. Common pitfalls

These will bite you if you don't watch out:

1. **Reusing the same TfidfVectorizer instance** across feature sets. The vectorizer remembers the vocabulary from its first `.fit_transform()` call. Make a fresh `TfidfVectorizer()` for each feature set.
2. **Plain `KFold` instead of `StratifiedKFold`** — folds can end up with only one class, and metrics break.
3. **Computing AUC on `.predict()` output** instead of `.predict_proba()` — AUC needs probabilities.
4. **Not setting `random_state=42`** — your numbers won't be reproducible.
5. **AdaBoost/XGBoost class balance** — they don't accept `class_weight`, you must pass `sample_weight` at fit time.
6. **`average='binary'` for F1 on multi-class** — silently picks the wrong metric. Always specify `average='weighted'`.
7. **Not handling the case where a CV fold has zero examples of a class** — metrics will throw or silently return nonsense. Either skip those folds or use `StratifiedKFold` which prevents the issue.

---

## 14. What you should produce

After Step 5 is done, `Classification/` will have:

```
Classification/
├── part5.ipynb                                  ← your code
├── iteration_1_target_population.csv            ← already there (Step 3 deliverable)
├── iteration_1_locals_vs_diaspora.csv           ← already there
├── iteration_1_person_vs_organization.csv       ← already there
└── experiments_results_iteration_1.csv          ← YOU PRODUCE THIS (Step 5 main deliverable)
```

That's the bare minimum the grader will look for. Plots, summary tables, and your written analysis paragraph are supplementary (for your report and presentation).

---

## 15. When you're ready

1. Re-read this guide once.
2. Answer the §8 quiz out loud.
3. If you can answer them all without re-reading: open a notebook and start §12 cell 1.
4. If you can't answer one: read the relevant section again before opening the notebook.

**The reason a lot of students struggle with Step 5 is that they jump to code before understanding the concepts.** You're going to do it backwards — concepts first, then code. That's why this works.

Take your time. When you have specific questions while building (`"why doesn't my AUC work?"`, `"is this the right shape?"`), ping me — but ask about ONE specific thing at a time, not "explain everything again." Small questions build understanding faster than big ones.

Good luck.
