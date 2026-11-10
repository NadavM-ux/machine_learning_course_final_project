# Handoff prompt — paste this whole thing into Claude Code

> Copy everything below the line into a fresh Claude Code session opened in the repo root.

---

## Context

I'm working on the **"Intro to Machine Learning" (SE 2026, Semester B)** final project at SCE.
The project identifies **Iranian points-of-interest (POIs)** among X/Twitter users using
**Active Learning**. My partner and I share this repo:

```
https://github.com/NadavM-ux/machine_learning_course_final_project
```

Everything lives under `Iran_POI/`. The work I need help with is in `Iran_POI/Classification/`.

**Read these two files first — they explain everything:**
1. `Iran_POI/Classification/ITERATIONS_HOWTO.md`  ← the authoritative guide, start here
2. `Iran_POI/Classification/active_learning.py`   ← the engine that runs everything

### Setup
```bash
git clone https://github.com/NadavM-ux/machine_learning_course_final_project
cd machine_learning_course_final_project
python3 -m venv .venv && .venv/bin/pip install pandas numpy scikit-learn xgboost matplotlib scipy deep-translator
```
All commands below run from `Iran_POI/Classification/` using `../../.venv/bin/python`.

---

## What the project does

We classify Twitter users on **3 tasks**. Every task uses the SAME 0/1/2 encoding:

| column | 1 | 0 | 2 |
|---|---|---|---|
| `target_population` | target (Iranian) | non-target | unknown |
| `locals_vs_diaspora` | local (lives in Iran) | diaspora (Iranian abroad) | unknown |
| `person_vs_organization` | person | organization | unknown |

Rule of thumb: **1 = the first word in the column name, 0 = the second word, 2 = unknown.**
`locals_vs_diaspora` only really applies when `target_population = 1`.

**Active Learning loop** (this is the core idea): train a model on the labeled users → have it
predict on the *unlabeled* pool → find the 100 users it is **least confident** about → a **human**
labels exactly those → add them to the training set → retrain. Repeat. Labeling the model's
*confusing* cases teaches it far more than labeling random users.

---

## Current state (as of this handoff)

| iteration | labeled users | status |
|---|---|---|
| 1 | 100 | done (random sample) |
| 2 | 200 | done |
| **3** | **300** | **IN PROGRESS — 34/100 labeled, 66 to go** |
| 4 | 400 | not started |
| 5 | 500 | not started |
| 6 | 600 | not started |

We need **at least 6 iterations** total. Unlabeled pool has ~746 users left, so this fits.

---

## The loop — this is all you need to run

Each iteration is exactly two commands, with **manual labeling in between**.

### Phase A — model picks the 100 users it's most confused about
```bash
cd Iran_POI/Classification
../../.venv/bin/python active_learning.py --iteration 3 --phase A
```
Produces:
- `Iteration_3/iteration_3_users_to_label.csv` ← **the 100 users to label** (label columns empty)
- `Iteration_3/iteration_3_unlabeled_users_predictions.csv` ← all 746 predictions (a required deliverable)

*(Phase A for iteration 3 has ALREADY been run — the file exists and is 34/100 labeled.)*

### Manual labeling — THE HUMAN PART
Open `Iteration_3/iteration_3_users_to_label.csv`. For each row:
1. Click the `profile_url` (opens `https://x.com/<username>`).
2. Look at the bio, location, language, tweets, profile picture.
3. Fill in `target_population`, `locals_vs_diaspora`, `person_vs_organization` with **0 / 1 / 2**.
4. Optionally add a note in `comments`.

### Phase C — merge labels, retrain everything, refresh the charts
```bash
../../.venv/bin/python active_learning.py --iteration 3 --phase C
```
It automatically: saves the 3 manual-label CSVs (required deliverables), builds
`iteration_3_combined_labeled.csv` (300 rows), translates new bios, runs the **648-experiment
sweep**, and rebuilds both comparison charts.

### Then repeat for the next iteration
```bash
../../.venv/bin/python active_learning.py --iteration 4 --phase A   # label the new 100...
../../.venv/bin/python active_learning.py --iteration 4 --phase C
# ...same for 5 and 6
```

### Bonus: rebuild the improvement chart any time (no retrain)
```bash
../../.venv/bin/python active_learning.py --phase H
```

---

## HARD RULES — do not violate these

1. **`balanced=True` ALWAYS.** The instructor explicitly said balanced is always better on this
   imbalanced data. `BALANCED_MODES = [True]` in `active_learning.py`. Never add `False` back.
2. **All 3 tasks stay.** `locals_vs_diaspora` was briefly removed and then restored — it IS
   required. Do not drop it.
3. **Both 2-class AND 3-class must run.** This is already automatic: the sweep runs each task
   twice — 3-class keeps 0/1/2, and 2-class drops the rows labeled `2`. Humans always label with
   all three values; the code decides when to ignore the 2s. Don't change this.
4. **STOP at step 6.** Do NOT start Step 7 (stopping criteria) or Step 8 (LLM comparison /
   confidence threshold). Finish all 6 iterations first.
5. **DO NOT label the users yourself (Claude).** You cannot see X profiles, and the only fields
   you'd have (`description`, `location`, `username`) are *exactly the features the model already
   trains on* — labeling from them is circular and teaches the model nothing. The labels must come
   from a human actually opening the profiles. This is also a graded double-annotation requirement.

---

## GOTCHAS — these have already bitten us

- **Excel silently deletes columns.** If the CSV is open in Excel while a script edits it, saving
  from Excel overwrites the script's changes. **Always close the CSV in Excel before running any
  command that touches it.** This already wiped the `locals_vs_diaspora` column once.
- **Too many `2` (unknown) labels flatten the improvement curve.** The model learns almost nothing
  from "unknown" — the 0s and 1s are what teach it. Only use 2 when the account is genuinely
  private, empty, or contradictory. Don't use it as a "can't be bothered" escape hatch.
- **`locals_vs_diaspora` is starved.** Out of 200 labeled users we only have ~3 diaspora examples.
  Make a real effort to decide local vs diaspora, or this task will never produce a usable model.
- **Phase C is slow** (LOOCV scales with N): ~1.5–2h at 300 rows, up to ~3–4h at 600. It **saves
  progress every 50 experiments and resumes** if interrupted. Safe to run in the background.

---

## How to read the results (important for the grade)

Three numbers exist; they are NOT equally trustworthy:

| metric | trust |
|---|---|
| `mean_accuracy` over all 648 experiments (`iteration_comparison_summary.csv`) | ⚠️ **blunt** — dragged down by dozens of weak configs. The PDF requires this plot, so keep it, but don't read model quality into it. It can go DOWN even when the model improves. |
| **Fixed hold-out F1 + target-recall** (`plot_holdout_improvement.png`) | ✅ **lead with this** — the real improvement proof |
| `best_AUC_*` (K-Fold winner per task) | ✅ trustworthy |
| LOOCV rows with AUC ≈ 0.95–1.0 | ❌ **ignore** — overfitting artifact on the tiny minority class |

**Why the CV metric can dip:** uncertainty sampling deliberately adds the *hardest, most ambiguous*
users each round, so the cross-validation test set gets harder over time. That's expected, not a bug.
(A friend's parallel project shows the same plateau.)

**The honest improvement proof** is `plot_holdout_improvement.png`: we freeze **30 users from
iteration 1** as a test set that is **never trained on**, and score every iteration's model on it.
Training data grows, test set stays fixed → a clean learning curve. It already shows
`target_population` F1 **0.85 → 0.90** and target-class recall **0.75 → 1.00** from iteration 1 → 2.

---

## What I want from you

1. Help me finish labeling iteration 3 efficiently (e.g. translate the Persian/Arabic bios to
   English, or build a fast labeling view) — **but do not invent the labels**.
2. Run Phase C for iteration 3.
3. Walk me through iterations 4, 5, 6 the same way.
4. Keep the improvement charts updated and help me interpret them for the report.

Start by reading `Iran_POI/Classification/ITERATIONS_HOWTO.md`, then tell me the current labeling
progress and what you recommend as the next concrete step.
