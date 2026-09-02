# Post-review correction protocol

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

## Status

- [x] Implement and test metric corrections.
- [x] Complete pre-go/event audit without relabelling trials.
- [x] Re-evaluate the full saved model matrix (124 evaluations).
- [x] Summarize behavioural probes and timing sensitivity (48 common-head fits).
- [x] Calibrate minimum-jerk sampling uncertainty (500 empirical repetitions).
- [x] Rebuild and visually verify the scientific report and student guide (12/5 pages).
- [x] Refresh and test dashboard assets/package; commit code and provenance.

## Verification completed

- 61 automated tests passed, zero skipped.
- 124 corrected evaluations share the same training reference within each fold.
- Original timing/reconstruction predictions reproduce within numerical tolerance.
- All six dashboard sections run at n=2,3,4,8, both in the checkout and a fresh
  ZIP extraction; no app exceptions in those 48 section/dimension checks.
- Both PDFs were rendered and all 17 pages visually checked; text bounds checked.
- Windows AppTest emitted a temporary-directory cleanup permission warning at
  interpreter exit; application checks passed with process exit code zero.
