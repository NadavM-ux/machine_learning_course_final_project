"""Plot baseline vs Iran-enriched best-model performance across iterations (K-Fold)."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
t = pd.read_csv(HERE / 'Feature_Enrichment' / 'enriched_kfold_trend.csv')
x = t['n_labeled_users']

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for a, metric, label in [(axes[0], 'bestAcc', 'Accuracy'),
                         (axes[1], 'bestAUC', 'AUC')]:
    a.plot(x, t[f'target_population__baseline_{metric}'], 'o--', color='#888',
           label='baseline features')
    a.plot(x, t[f'target_population__enriched_{metric}'], 'o-', color='#1f77b4', lw=2.5,
           label='+ Iran enrichment')
    a.set_title(f'{label} — target_population (2-class, K-Fold)')
    a.set_xlabel('# labeled users (iteration)')
    a.set_ylabel(label)
    a.set_ylim(0.4, 1.0)
    a.axhline(0.5, ls=':', color='grey', lw=1)
    a.grid(alpha=0.3)
    a.legend()
fig.suptitle('Step 6 improvement: Iran-specific feature enrichment across all iterations',
             fontsize=13, fontweight='bold')
fig.tight_layout()
out = HERE / 'plot_enriched_accuracy_trend.png'
fig.savefig(out, dpi=120, bbox_inches='tight')
print("saved:", out.name)
print(t.to_string(index=False))
