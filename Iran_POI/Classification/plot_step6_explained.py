"""Step 6 — honest explanation of the 'declining' Active-Learning curve.

The PDF expects mean accuracy to RISE across iterations. Ours DIPS — not because
the model degrades, but because uncertainty sampling feeds it the hardest users
each round, so the cross-validation yardstick hardens. Proof: on a FIXED hold-out
test set the model holds ~0.87 AUC (stable). This figure shows both signals so the
dip is explained, not hidden."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
comp = pd.read_csv(HERE / 'iteration_comparison_summary.csv')
hold = pd.read_csv(HERE / 'holdout_improvement_trend.csv')

fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.5))

# LEFT — the misleading CV curve (dips)
axL.plot(comp['iteration'], comp['mean_accuracy'], 'o-', lw=2, ms=9, color='#d62728')
axL.set_title('CV mean accuracy — DIPS (misleading)')
axL.set_xlabel('Iteration'); axL.set_ylabel('mean accuracy (all experiments, K-Fold)')
axL.set_xticks(comp['iteration']); axL.set_ylim(0.4, 1.0); axL.grid(alpha=0.3)
axL.annotate('each round adds the HARDEST users\n(uncertainty sampling) → the CV\ntest set hardens, so accuracy dips',
             xy=(6, comp['mean_accuracy'].iloc[5]), xytext=(2.6, 0.83), fontsize=9,
             color='#7a1416', arrowprops=dict(arrowstyle='->', color='#7a1416'))

# RIGHT — the honest fixed-test-set curve (stable)
axR.plot(hold['iteration'], hold['target_population_AUC'], 'o-', lw=2, ms=9, color='#2ca02c')
axR.axhline(hold['target_population_AUC'].mean(), ls='--', color='grey', alpha=0.6,
            label=f"mean = {hold['target_population_AUC'].mean():.2f}")
axR.set_title('Fixed hold-out AUC — STABLE (honest signal)')
axR.set_xlabel('Iteration'); axR.set_ylabel('target_population AUC on frozen 30-user test set')
axR.set_xticks(hold['iteration']); axR.set_ylim(0.4, 1.0); axR.grid(alpha=0.3); axR.legend()
axR.annotate('on a FIXED test set the model\nholds ~0.87 — it did NOT degrade,\nit reached its ceiling (plateau)',
             xy=(7, hold['target_population_AUC'].iloc[-1]), xytext=(2.6, 0.55), fontsize=9,
             color='#1a5e1a', arrowprops=dict(arrowstyle='->', color='#1a5e1a'))

fig.suptitle('Step 6: the model PLATEAUED, it did not degrade — the CV dip is a yardstick artifact',
             fontsize=13, fontweight='bold')
plt.tight_layout()
out = HERE / 'plot_step6_explained.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"saved: {out.name}")
