"""
Honest, presentation-grade figure for the Active-Learning narrative.

Reads ONLY existing result CSVs (no model is retrained):
  - iteration_comparison_summary.csv      (best 5-fold CV AUC per task, per iteration)
  - Feature_Enrichment/enrichment_summary.csv   (base vs Iran-enriched AUC, same 5-fold CV basis)

Message (two panels, same evaluation basis so it is fully comparable):
  (a) Adding labeled data (100 -> 640) makes best-model AUC PLATEAU -> Stopping Criterion (Step 7) fires.
  (b) The real lever was FEATURES: Iran-specific lexicon + tweets break the ceiling (target 0.76 -> 0.89).
"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

HERE = "Iran_POI/Classification"
comp = pd.read_csv(f"{HERE}/iteration_comparison_summary.csv")
enr = pd.read_csv(f"{HERE}/Feature_Enrichment/enrichment_summary.csv")

TASKS = {
    "target_population": ("Target population", "#1f77b4"),
    "locals_vs_diaspora": ("Locals vs diaspora", "#2ca02c"),
    "person_vs_organization": ("Person vs org.", "#ff7f0e"),
}

# best base / best enriched per task (same 5-fold CV basis)
def best(task, col):
    return enr.loc[enr.task == task, col].max()

plt.rcParams.update({"font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})
fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 5.6))

# ---------- Panel A: iterations plateau ----------
it = comp["iteration"].values
n = comp["n_labeled_users"].values
for task, (label, color) in TASKS.items():
    axA.plot(it, comp[f"best_AUC_{task}"], "o-", color=color, lw=2, ms=6, label=label)

# feature-enrichment endpoint for the headline task (dashed = different lever)
tgt_last_x = it[-1]
tgt_base = comp["best_AUC_target_population"].values[-1]
tgt_enr = best("target_population", "enriched_bestAUC")
axA.plot([tgt_last_x, tgt_last_x + 1], [tgt_base, tgt_enr], "--", color="#1f77b4", lw=2)
axA.plot(tgt_last_x + 1, tgt_enr, "*", color="#d62728", ms=20, zorder=5)
axA.annotate("+ Iran-specific\nfeatures", xy=(tgt_last_x + 1, tgt_enr),
             xytext=(tgt_last_x + 1, tgt_enr + 0.005), ha="center", va="bottom",
             fontsize=9.5, fontweight="bold", color="#d62728")

axA.axhspan(0.74, 0.84, color="gray", alpha=0.10)
axA.text(3.7, 0.755, "plateau (base features)", color="gray", fontsize=9, style="italic")
axA.annotate("Stopping criterion\ntriggered (Step 7)",
             xy=(7, comp['best_AUC_target_population'].values[-1]),
             xytext=(4.6, 0.60), fontsize=9.5, ha="center",
             arrowprops=dict(arrowstyle="->", color="black", lw=1.2))
axA.axhline(0.5, ls=":", color="red", alpha=0.6)
axA.text(1.05, 0.51, "random (0.5)", color="red", fontsize=8)

xt = list(it) + [it[-1] + 1]
axA.set_xticks(xt)
axA.set_xticklabels([f"{i}\n({m})" for i, m in zip(it, n)] + ["+feat\n(640)"], fontsize=9)
axA.set_xlabel("Active-Learning iteration  (labeled users)")
axA.set_ylabel("Best-model AUC  (5-fold CV)")
axA.set_ylim(0.48, 0.95)
axA.set_title("(a) More labels → performance plateaus", fontweight="bold")
axA.legend(loc="lower left", fontsize=9, framealpha=0.9)

# ---------- Panel B: features break the ceiling ----------
labels = [v[0] for v in TASKS.values()]
base_v = [best(t, "baseline_bestAUC") for t in TASKS]
enr_v = [best(t, "enriched_bestAUC") for t in TASKS]
x = range(len(labels))
w = 0.36
b1 = axB.bar([i - w / 2 for i in x], base_v, w, label="Base features", color="#b0b0b0")
b2 = axB.bar([i + w / 2 for i in x], enr_v, w, label="+ Iran-enriched", color="#d62728")
for i, (bv, ev) in enumerate(zip(base_v, enr_v)):
    axB.text(i - w / 2, bv + 0.008, f"{bv:.2f}", ha="center", fontsize=9)
    axB.text(i + w / 2, ev + 0.008, f"{ev:.2f}", ha="center", fontsize=9, fontweight="bold")
    axB.annotate(f"+{ev - bv:.2f}", xy=(i, max(bv, ev) + 0.03), ha="center",
                 fontsize=10, fontweight="bold", color="#d62728")
axB.axhline(0.5, ls=":", color="red", alpha=0.6)
axB.set_xticks(list(x))
axB.set_xticklabels(labels, fontsize=10)
axB.set_ylabel("Best AUC  (5-fold CV)")
axB.set_ylim(0.48, 0.95)
axB.set_title("(b) Iran-specific features → breakthrough", fontweight="bold")
axB.legend(loc="lower right", fontsize=9)

fig.suptitle("Active Learning saturates on labels — the gain comes from better features",
             fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = f"{HERE}/plot_active_learning_story.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
