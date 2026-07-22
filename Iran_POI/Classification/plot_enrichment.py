"""Before/after figure: data-growth saturation (iterations 1-7) vs the jump from
feature enrichment (Iran-specific lexicon + tweets). Report deliverable."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
comp = pd.read_csv(HERE / 'iteration_comparison_summary.csv')
summ = pd.read_csv(HERE / 'Feature_Enrichment' / 'enrichment_summary.csv')

fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.5))

# --- Panel A: data growth saturated (best K-Fold AUC for target_population) ---
axL.plot(comp['iteration'], comp['best_AUC_target_population'],
         'o-', lw=2, ms=9, color='#888', label='data growth (iter 1-7)')
enr_target = float(summ[(summ['task'] == 'target_population') &
                        (summ['family'] == 'tweets+numeric+iran(full)')]['enriched_bestAUC'].iloc[0])
axL.scatter([7], [enr_target], s=220, color='#2ca02c', zorder=5, marker='*',
            label=f'+ feature enrichment = {enr_target:.3f}')
axL.annotate('', xy=(7, enr_target), xytext=(7, comp['best_AUC_target_population'].iloc[-1]),
             arrowprops=dict(arrowstyle='-|>', color='#2ca02c', lw=2.5))
axL.annotate(f"+{enr_target - comp['best_AUC_target_population'].iloc[-1]:.3f}",
             (7, (enr_target + comp['best_AUC_target_population'].iloc[-1]) / 2),
             xytext=(8, 0), textcoords='offset points', color='#2ca02c', fontweight='bold')
axL.axhline(0.85, ls='--', color='#d62728', alpha=0.5, label='~0.85 saturation ceiling')
axL.set_xticks(comp['iteration'])
axL.set_xlabel('Iteration'); axL.set_ylabel('Best AUC — target_population')
axL.set_title('Data growth saturated; features broke through')
axL.set_ylim(0.5, 1.0); axL.grid(alpha=0.3); axL.legend(fontsize=9, loc='lower left')

# --- Panel B: baseline vs enriched vs full, per task ---
tasks = ['target_population', 'locals_vs_diaspora', 'person_vs_organization']
labels = ['target\npopulation', 'locals vs\ndiaspora', 'person vs\norganization']
base, enr, full = [], [], []
for t in tasks:
    d = summ[summ['task'] == t]
    base.append(float(d[d['family'] == 'desc+numeric']['baseline_bestAUC'].iloc[0]))
    enr.append(float(d[d['family'] == 'desc+numeric']['enriched_bestAUC'].iloc[0]))
    f = d[d['family'] == 'tweets+numeric+iran(full)']['enriched_bestAUC']
    full.append(float(f.iloc[0]) if len(f) and pd.notna(f.iloc[0]) else np.nan)

x = np.arange(len(tasks)); w = 0.27
b1 = axR.bar(x - w, base, w, label='baseline (desc+numeric)', color='#b0b0b0')
b2 = axR.bar(x, enr, w, label='+ Iran features', color='#1f77b4')
b3 = axR.bar(x + w, full, w, label='+ Iran + tweets', color='#2ca02c')
for bars in (b1, b2, b3):
    for bar in bars:
        h = bar.get_height()
        if not np.isnan(h):
            axR.annotate(f'{h:.2f}', (bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8)
axR.set_xticks(x); axR.set_xticklabels(labels)
axR.set_ylabel('Best K-Fold AUC'); axR.set_ylim(0.5, 1.0)
axR.set_title('Feature enrichment lifts all three tasks')
axR.axhline(0.5, ls='--', color='grey', alpha=0.5)
axR.grid(alpha=0.3, axis='y'); axR.legend(fontsize=9, loc='lower right')

plt.tight_layout()
out = HERE / 'Feature_Enrichment' / 'plot_enrichment_before_after.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"saved: {out.relative_to(HERE)}")
