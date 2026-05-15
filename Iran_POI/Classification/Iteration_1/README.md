# Iteration 1 — Files Index

Contents of this folder, organized by which PDF step they belong to.

## Step3_Manual_Labeling/

Annotator outputs and Step 3 deliverables (PDF pages 4–7).

| File | What it is | PDF requirement |
|---|---|---|
| `iteration_1_labels_NADAV.csv` | Annotator 1 labels (50 users) | Step 3 work product |
| `iteration_1_labels_NADAV copy.csv` | Annotator 1 — alternate labeling pass | Step 3 work product |
| `iteration_1_labels_NADAV_IL.csv` | Annotator 2 labels (50 users) | Step 3 work product |
| `iteration_1_labels_NADAV_IL copy.csv` | Annotator 2 — alternate labeling pass | Step 3 work product |
| `iteration_1_labels_merged_sidebyside.csv` | Both annotators side-by-side | Step 3 work product |
| `iteration_1_disagreements.csv` | Where annotators disagreed (pre-resolution) | Step 3 work product |
| **`iteration_1_labels_consensus.csv`** | Final consensus labels | ✅ Mandatory (page 6) |
| **`iteration_1_agreement_report.csv`** | percent_agreement + cohens_kappa per label type | ✅ Mandatory (page 6) |
| **`iteration_1_target_population_summary.csv`** | Class counts + percentages | ✅ Mandatory (page 7) |
| **`iteration_1_locals_vs_diaspora_summary.csv`** | Class counts + percentages | ✅ Mandatory (page 7) |
| **`iteration_1_person_vs_organization_summary.csv`** | Class counts + percentages | ✅ Mandatory (page 7) |
| `iteration_1_target_population.csv` | Per-task labels (working copy) | Original Step 3 output — same content as the copy at Classification/ root |
| `iteration_1_locals_vs_diaspora.csv` | Per-task labels (working copy) | Same |
| `iteration_1_person_vs_organization.csv` | Per-task labels (working copy) | Same |

## Step5_Analysis/

Step 5 supplementary materials (PDF pages 10–13). The mandatory deliverable for Step 5 — `experiments_results_iteration_1.csv` — lives one level up at `Classification/` root.

| File | What it is |
|---|---|
| `iteration_1_consensus_translated.csv` | Consensus + English translations of `description` and `display_name` (intermediate input to part5.ipynb) |
| `iteration_1_best_models_summary.csv` | Winner per (task, n_classes), with honesty-filtered AUC ≥ 0.60 |
| `iteration_1_degeneracy_check.csv` | Proof that the locals_vs_diaspora "winner" is degenerate (predicts majority class only) |
| `iteration_1_report_draft.md` | Word-ready Step 5 report draft |
| `step5_analysis.py` | Re-runnable analysis script that produced the summaries + plots |
| `plot_f1_by_algorithm.png` | Mean F1 per algorithm, faceted by task (report figure) |
| `plot_f1_by_feature_set.png` | Mean F1 per feature set, faceted by task (report figure) |
| `plot_kfold_vs_loocv.png` | K-Fold vs LOOCV agreement (report figure) |

## How to re-run

From the repository root or anywhere:
```bash
python Iran_POI/Classification/Iteration_1/Step5_Analysis/step5_analysis.py
```
The script writes its outputs back into `Step5_Analysis/`.
