# Finding a target population on X/Twitter with active learning

Final project for *Introduction to Machine Learning* (B.Sc. Software Engineering, SCE, 2026), built with a partner.

**Goal.** Automatically identify X/Twitter accounts that belong to a chosen target population (Iran-related accounts)
in a pool of about 950 candidate profiles, when labels are expensive: a person has to open each profile and decide.
Every account is classified on three tasks (`target_population`, `locals_vs_diaspora`, `person_vs_organization`),
and the identified population is clustered at the end.

## Approach and results

1. **Statistical profiling** of the candidate pool: follower, following and status distributions.
2. **Double manual labelling** of a first random sample of 100 users. Inter-annotator agreement measured with
   Cohen's κ: 0.88 / 1.0 / 0.94 across the three tasks.
3. **Classifier bake-off:** 6 algorithms × feature sets × K-Fold and LOOCV × 2- and 3-class formulations ×
   balanced/unbalanced training, 1,296 experiments per iteration, with a degeneracy check so a "winner" that only
   predicts the majority class is rejected.
4. **Active learning:** the model scores the unlabelled pool, the 100 least-confident users are labelled by hand,
   the model is retrained. Seven iterations took the labelled set from 100 to 640 users.
5. **Stopping criterion:** going from 600 to 640 labels moved accuracy and AUC by less than 0.5%, so labelling stopped.
6. **Feature enrichment** with a domain lexicon over bios and tweets lifted target-population AUC from 0.76 to 0.89.
7. **Confidence threshold** of 0.8 for the final sweep: 173 users flagged, 85.7–90% hit rate on manual checks.
8. **LLM baseline:** the trained model scored 0.95 against 0.53 for a general-purpose LLM prompted on the same
   profiles (10 runs, majority vote).
9. **Clustering** of the identified population: k = 8, three super-groups, ARI 0.73 across runs.

**Stack:** Python, pandas, scikit-learn, XGBoost, matplotlib, Jupyter.

## Repository layout

Everything lives under `Iran_POI/`, ordered by the course's step numbers. The detailed file index (in Hebrew) is
[`Iran_POI/README.md`](Iran_POI/README.md); how an iteration is run is in
[`Iran_POI/Classification/docs/ITERATIONS_HOWTO.md`](Iran_POI/Classification/docs/ITERATIONS_HOWTO.md).

```
Iran_POI/
  Step1_Population_Choice/      why this population
  Step2_Statistical_Analysis/   notebook and histograms
  Step4_Decision_Flowcharts/    labelling decision trees
  Classification/               steps 3, 5, 6, 7, 8: labelling, training, active learning, stopping, thresholds
    active_learning.py          the engine: phase A picks users to label, phase C merges labels and retrains
    iran_features.py            domain lexicon features used in the enrichment step
    Iteration_1 … Iteration_7/  per-iteration labels, predictions and experiment results
    Feature_Enrichment/         before/after experiments and plots
    Step7_Stopping_Criteria/    stopping decision and plots
    Step8_Confidence_Threshold/ threshold sweep and hit rate
  LLM_Comparison/               trained model vs LLM on the same users
  Step9_Clustering/             clustering code and report
```

## Running an iteration

```bash
python3 -m venv .venv && .venv/bin/pip install pandas numpy scikit-learn xgboost matplotlib scipy deep-translator
cd Iran_POI/Classification
../../.venv/bin/python active_learning.py --iteration N --phase A   # pick the 100 least-confident users
# label them by hand in Iteration_N/iteration_N_users_to_label.csv
../../.venv/bin/python active_learning.py --iteration N --phase C   # merge labels, retrain, refresh plots
```

