# Iteration 1 — annotation process note

This note documents how the iteration-1 labels were produced, for transparency.

## Who labeled
Iteration 1 was labeled **collaboratively by both partners** (Nadav Iloz, Nadav Magen).
- Where both agreed, a single agreed label was recorded.
- The **7 cases of disagreement** were flagged and resolved by discussion — see
  `iteration_1_disagreements.csv` (each partner's differing value + the resolution)
  and the `consensus_source` column in `iteration_1_labels_consensus.csv`
  (`auto_agree` for the 93 agreed, `discussed` for the 7).
- **The two independent labelings are preserved on disk:** annotator 1 in
  `iteration_1_labels_NADAV.csv` / `iteration_1_labels_NADAV_IL.csv`, and annotator 2 in
  `iteration_1_labels_NADAV_annotator2.csv` / `iteration_1_labels_NADAV_IL_annotator2.csv`.
  They differ on exactly the 7 disagreement users — with a different label **and** a
  different reasoning comment (e.g. "terrorist" vs "bot") — which is precisely what
  Percent Agreement and Cohen's Kappa are computed from.

From **iteration 3 onward**, labeling was done by a **single annotator (Nadav Iloz)**,
because the partner (Nadav Magen) was on reserve duty (miluim). Active-learning
iterations do not require double annotation, so this is expected.

## Agreement metrics
`iteration_1_agreement_report.csv` reports Percent Agreement and Cohen's Kappa per
label column (93/100 agreement on `target_population`, with 7 discussed cases).

## Note on the comments column (known artifact)
In `iteration_1_labels_merged_sidebyside.csv`, a build script copied each user's
single real comment into **both** comment columns, so the two comment columns look
identical. This is a cosmetic script artifact and does **not** affect the labels,
the disagreements, or the agreement metrics.

**The original per-user comments are preserved in the source files:**
- `iteration_1_labels_NADAV.csv` — concise comment style (e.g. "Looks like a bot")
- `iteration_1_labels_NADAV_IL.csv` — detailed comment style (e.g. "Western-style
  username with no discernible connection to Iran.")

These two files are the authoritative record of the hand-written comments.
