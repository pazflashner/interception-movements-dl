# Post-review correction protocol

## Numerical and reporting amendment, 5 September 2026

The 56 main paired model comparisons now round participant differences to 12
decimal places in each metric's reporting units before ranking (two-sided
Wilcoxon, Pratt zeros, SciPy method='auto'). Means and effect sizes are unchanged.
This resolves machine-precision ties in discrete KS summaries; it is not a
precision selected to obtain significance. All 56 significance decisions agree
with the previous table. Independent aggregation agrees with the amended tests.
Execution used Python 3.13, NumPy 2.3.4, pandas 2.3.3 and SciPy 1.16.3; the output
records the SciPy version and test settings. Other sensitivity-test families
retain their existing numerical conventions. Original and amended tables are
preserved under review_fresh_2026_09_05/checks/numerical_fix.

Minimum-jerk legacy mj_fit_success denotes execution completion, not optimizer
convergence. New fits record completion, selected/BIC convergence and candidate
optimizer diagnostics separately. Historical fits retain their original diagnostics. New matched-procedure fits are
reported separately below. The legacy min_onset_spacing_s config name means an indexed
absolute onset lower-bound step, not pairwise separation. Effective constraints
and selection rules are unchanged. The report now discloses unequal recorded
and generated fitting budgets and filtering/sampling histories.

Jason's supplied emails and project slides were reviewed (TAD PDF pages 7-9
only). The research objective is distribution generation across movement
features and people. Motion before inferred target-motion onset remains a
descriptive audit finding; these trials are retained. The same-model fingerprint control and matched component refitting are completed
post-review evaluations, reported below.

This is an explicitly post-review evaluation of the frozen
`strategy-confirmatory-v1` checkpoints, not a new preregistered experiment.
Original runs and reports are retained. No selection is based on improved test
scores. Dataset exclusions, timing labels, trained CVAE weights, and participant
folds remain unchanged unless a separate, documented decision is made.

## Decisions fixed before corrected evaluation

- Keep target MOTION onset as time zero. Marker 5 identifies target appearance,
  not the go signal. Pre-go samples are retained for an audit, not added to the
  encoder input. Do not delete zero-onset trials by assumption.
- Audit all 4,732 retained trials for raw/filtered pre-go motion and the
  CSV-end versus MAT-arrival offset. Report threshold evidence, not a diagnosis
  of anticipation. The last tracker sample remains the arrival proxy pending
  event-synchronization confirmation.
- Use all four fixed participant folds and all saved seeds in the five-family
  model matrix. Neural checkpoints are reused. Deterministic classical models
  are reconstructed from their recorded train/validation settings.
- Multivariate feature distances use one reference per fold fitted only to
  training trials. Continuous-feature scales are training standard deviations;
  count scales are at least one count. A constant training feature uses scale
  one, so a generated discrepancy is not silently removed. The RBF bandwidth
  is fixed from at most 512 training examples, seed 2026, and is shared across
  models in that fold. This changes the metric definition; recompute all
  families rather than comparing corrected and old values.
- Univariate KS/Wasserstein use the explicit 11-feature x-y list. Keep speed
  peak count as a reported feature; it is not a minimum-jerk component count.
- Include all available behavioural probes, including negative results.
  Present per-fold R-squared separately from pooled out-of-fold R-squared and
  explicit train-mean baseline errors; never average scores and call them
  pooled R-squared.
- Timing sensitivity: retain original predictions as the primary historical
  benchmark. Add validation-fitted multiplicative MAE calibration (not a test
  oracle) and common supervised heads on frozen n=3/8 CVAE and spline codes.
  Common-head comparisons do not erase differences in representation training.
- Minimum-jerk fitting remains secondary. Calibrate finite-sample distances
  with empirical resampling at the actual sample sizes and retain discrete
  counts. Report uncertainty rather than interpreting a generic KS value as a
  hard floor. Additional generated fits, if run, are identified by sample count
  and fitting settings rather than silently mixed with old fits.

## Completed extension, 5 September 2026

- All 24 n=3/8 CVAE fold/seed checkpoints underwent own/population/wrong-person
  fingerprint controls; 1,344 donor-level rows. Own scores reproduce saved
  benchmarks within 1e-6. All 12 BH-adjusted contrasts favour own fingerprints.
- Direct-context and 21-development-participant constant baselines cover all 14
  targets. Direct context has lower MAE on all 14 at each dimension; this is
  descriptive and does not equalize compression budgets.
- 2,376 recorded query and 1,680 generated model-window trajectories underwent
  matched downstream preprocessing and two-restart/400-evaluation decomposition.
  Count TV is 0.213891/0.232263 at n=3/8. All 4,056 selected fits converged;
  505 fits contain a nonconverged alternative order. No count was excluded.
- Four-restart/700-evaluation sensitivity: selected counts agree for 56/56
  recorded and 111/112 generated trajectories. Nine dimension comparisons:
  none survives BH. The matched 500-replicate sampling reference is descriptive.
- The main paper has eight pages, plus a separate appendix and student guide.
  The release bundles summary evidence and four illustrative checkpoints.

The controls' frozen settings, per-checkpoint hashes and full result tables are
under results/review_controls. Compact copies are tracked in review_evidence;
full CSVs are included in the release ZIP. The PAZ_REVIEW_HANDOFF.md file is the
current entry point. Its restore script supports independent clone testing.

## Verification

The full 124-evaluation coverage, common distance references and unchanged
reconstruction/timing predictions were checked by verify_post_review_outputs.py.
Its VERIFICATION.json records PDF hashes and the 96 training histories (150-epoch
cap, five reaching it). New control software and restoration checks join the
existing synthetic/reference tests. Final delivery check results are recorded
in review_evidence/DELIVERY_CHECKS.json after packaging.
