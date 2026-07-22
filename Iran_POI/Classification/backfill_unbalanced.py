"""
Backfill the missing UNBALANCED (balanced=False) experiments for iterations 3-7.

Why: PDF Step 5 & Step 6 require every iteration's experiment log to contain BOTH
balance modes. Iterations 1-2 already have both; the engine's BALANCED_MODES was
later set to [True] only, so iters 3-7 hold balanced rows only. This script flips
that flag and re-runs the sweep. run_sweep() RESUMES from the existing CSV, so it
adds ONLY the balanced=False rows (the balanced=True ones are already 'done').

It does NOT retrain anything that already exists, and does NOT touch the comparison
plot (rebuild_comparison filters balanced=True). Fully additive, resumable, safe to
kill and restart (saves every 50 experiments).

Usage:
    python backfill_unbalanced.py 3            # one iteration
    python backfill_unbalanced.py 3 4 5 6 7    # all remaining
"""
import sys
import time
import active_learning as al

al.BALANCED_MODES = [True, False]          # <-- the fix: run both modes

iters = [int(x) for x in sys.argv[1:]] or [3, 4, 5, 6, 7]
print(f"Backfilling unbalanced runs for iterations: {iters}")

for n in iters:
    t0 = time.time()
    print(f"\n########## iteration {n} ##########")
    combined = al._combined_with_text(n)                 # labeled set + translation + numeric
    combined['account_age_years'] = combined['account_age_years'].fillna(
        combined['account_age_years'].median())
    feats = al.build_feature_sets(combined)
    out = al.HERE / f'Iteration_{n}' / f'experiments_results_iteration_{n}.csv'
    al.run_sweep(combined, feats, n, out)
    print(f"iteration {n} done in {time.time()-t0:.0f}s")

print("\nALL DONE. (Run rebuild_comparison / holdout_trend separately if desired;")
print(" they are unaffected by unbalanced rows — the trend uses balanced=True only.)")
