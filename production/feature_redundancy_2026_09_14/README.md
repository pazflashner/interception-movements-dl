# Feature-redundancy sensitivity, 14 September 2026

All four training rotations selected the same nine features. Straight-line
distance and endpoint y were redundant with path length under the recorded rule:
absolute pooled Spearman rho and median within-participant rho both at least .90,
with matching signs. The priority order favors path length as an extent measure.
This is a transparent design choice, not proof that the other measurements lack
scientific meaning. The protocol was written before computing these sensitivity
results; this was not a preregistered study.

Both VAE and CVAE still beat spline + PCA on reduced-feature KS, energy and MMD
at both n=3 and n=8 under BH and Holm. The largest Holm p in those 12 comparisons
is .003315. At n=8, VAE versus CVAE has BH q=.149102, .289293 and .427519,
respectively: none is significant. Thus reduction preserves the variational
advantage over spline but weakens a claim that VAE is uniquely superior.

The new correction family contains **72 tests together**: six compact-model
pairs × two capacities × three endpoints × two feature views. The historical
140-test family remains unchanged. Mean KS averages eleven (or nine) statistics,
not feature p-values. Dropping two features therefore does not automatically
remove two tests from the correction. Distances from different feature views
must not be interpreted as an absolute accuracy improvement.

No model was retrained. The 560 participant/model/seed generations reproduce
the original eleven-feature scores to at most 4.44e-16. The reduced view changes
only the scoring features and their training-derived distance geometry. The
primary eleven-feature evaluation and dashboard feature set remain unchanged.

## Reproduce

From the repository root, with the original data/cache and frozen checkpoints:

```powershell
python scripts/analyze_feature_redundancy.py
python production/verify_feature_redundancy.py
```

The runner caches generated feature arrays in the ignored
`production/assets/feature_redundancy_2026_09_14/`. Delete only this task's cache
if intentionally recomputing after changing its protocol or implementation.
The runner otherwise resumes completed model/fold scores. A changed protocol
must use a new analysis directory instead of silently reusing cached scores.

## Evidence

- `protocol.json`: selection rule, priority, endpoints and correction family.
- `selection.json`: train identifiers and retain/drop decisions per rotation.
- `training_correlations.csv`: all 55 pairs in each of four rotations.
- `per_seed_scores.csv`: 1,120 rows, including both feature views.
- `participant_scores.csv`: seed averages, one paired score per person/model/view.
- `summary.csv`: equal-participant means.
- `paired_comparisons.csv`: all 72 tests, BH and Holm values.
- `verification.json`: original-score reproduction and unchanged-weight check.
- `independent_verification.json`: all 72 p-values/corrections, 220 correlations,
  four training selections and 16 decoded-feature cases independently recomputed.

The independent check ranks raw training features directly and uses separate
empirical-CDF and pair-distance calculations. It verifies computation, not
independent replication of a scientific finding.
