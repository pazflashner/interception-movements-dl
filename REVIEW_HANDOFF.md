# Independent code and results review

Prepared 2026-09-02. Scientific snapshot: commit `a1fea96` before the
navigation-only cleanup. This file is a map, not a certification of correctness.
Please challenge the code, saved results, and report independently; do not try
to defend a preferred model or inherit earlier assistants' conclusions.

## Task

Review the current study before Simaan and Paz present it to Prof. Jason
Friedman (motor-control researcher) and Moni Shahar (deep-learning advisor).
Check implementation, evaluation design, numerical results, figures, wording,
references, and reproducibility. Read the actual functions and saved predictions;
passing the existing tests or repeating the PDF's claims is not sufficient.

Start read-only. Do not edit the model, data, report, dashboard, or existing
outputs, and do not launch a training sweep. Small independent calculations,
tests, and checkpoint inference are appropriate. Put audit outputs in a new
`review/` directory. If an expensive rerun is necessary, state exactly why and
ask before starting it. Never modify the Dropbox source files.

## Start here

Repository on this computer:

```text
D:\oneDrive_Main\OneDrive\Desktop_onedrive\uni2024\Third Year\semester B\Workshop in AI\interception-movements-dl
```

Paths below are relative to this repository unless written in full.

| Read | Purpose |
|---|---|
| `output/pdf/Interception_Movements_Final_Scientific_Report.pdf` | Current scientific report to audit, including figures and claims |
| `output/pdf/Interception_Movements_Results_Guide.pdf` | Student explanation; check consistency with the scientific report |
| `studies/final_strategy_evaluation/PROTOCOL.md` and `protocol/*.json` | Declared design, participant folds, and model settings |
| `scripts/README.md` | Current entry points and shared dependencies |
| `src/README.md` | Source-code map |
| `reports/build_final_reports.py` | Exact tables, figure calculations, captions, and report text |

The current report is not any earlier preliminary PDF in the parent workshop
directory or archive. The submitted proposal describes intentions, not evidence
that a method was implemented. If needed, locate proposal versions in the parent
directory/downloads and identify which version you used rather than guessing.

## Data and lineage

- Raw trajectories and MAT metadata: `D:\DropBox\Dropbox\results`.
- Target trajectories/instructions: `D:\DropBox\Dropbox\stimuli`.
- Shared processed dataset: `studies/strategy_window_comparison/data/canonical_trials.pkl`.
- Companion metadata and preprocessing record: `canonical_trial_metadata.csv`
  and `canonical_dataset_protocol.json` in that same data directory.
- Completion audit: `studies/strategy_window_comparison/data_audit/`.
- Shared real minimum-jerk fits: `studies/strategy_window_comparison/results/submovements_real.csv`.
- K-means evidence used in the report:
  `studies/strategy_window_comparison/results/go_to_arrival/baselines/kmeans_selection_corrected.csv`.

The old study directory is still a deliberate data dependency. Do not assume
everything in it is obsolete, and do not delete or relocate it during review.
`config.py` retains that directory as its default root. Trace effective runtime
settings through the confirmatory entry points and saved configurations; comments
and defaults from earlier work may not describe the final runs.

CSV names encode condition, start/speed category, side, and repetition. Verify
the CSV/MAT join and event definitions against the real files. In particular,
appearance marker 5, target motion onset, finger movement onset, and arrival are
distinct events. Position units require verification; do not silently call raw
tracker coordinates millimetres. The stated finger/target rates are 240/60 Hz.

## Current study and saved evidence

The declared analysis uses condition 2 and the target-motion-onset to
finger-arrival interval, retaining waiting. It uses 100 phase samples of x-y
position with 10 Hz filtering. Physical initiation and movement time are
separate decoder targets, not encoder inputs. Verify these claims in execution,
including how resampling, feature extraction, and timing inversion interact.

Local inventory on 2026-09-02 (presence only, not a successful audit):

| Family under `studies/final_strategy_evaluation/runs/` | Dimensions | Saved `result.json` files | Neural checkpoints |
|---|---|---:|---:|
| `cvae` | 2, 3, 4, 8 | 48 | 48 |
| `conditional_ae` | 3, 8 | 24 | 24 |
| `unconditional_vae` | 3, 8 | 24 | 24 |
| `spline_pca` | 2, 3, 4, 8 | 16 | not applicable |
| `condition_ridge` | no trajectory latent | 12 | not applicable |

Expected design: four fixed outer participant folds, 17/4/7 train/validation/test
participants in each; each of the 28 participants appears in test once. Neural
seeds are 42, 43, and 44 within each fold. This is not all possible participant
rotations. Ridge seeds govern stochastic generation, not neural initialization.
Check the complete fold/dimension/seed matrix, not just the total file count.

Example run: `runs/cvae/fold0/cvae_z3_seed42/` under the final study.
Inspect its checkpoint, configuration, split, history, `result.json`,
`timing_predictions.csv`, `reconstruction_predictions.csv`, and
`context_query_fidelity.csv`, then verify analogous files across all runs.
File existence alone does not prove a run was complete or produced by current code.

Under `studies/final_strategy_evaluation/results/`:

| Evidence | What to trace |
|---|---|
| `*_all_runs.csv` | Model-run inventory and run-level metrics |
| `analysis/oof_metrics_summary.csv` | Reported out-of-fold comparison |
| `analysis/participant_metrics_raw.csv`, `participant_metrics_seed_averaged.csv` | Participant weighting and seed aggregation |
| `analysis/paired_model_comparisons.csv` | Paired contrasts, uncertainty, and multiplicity corrections |
| `analysis/feature_fidelity_raw.csv`, `feature_fidelity_summary.csv` | Individual distribution-feature results |
| `analysis/fold_vs_seed_variability.csv` | Fold versus optimization variability |
| `analysis/timing_outlier_audit.csv` | Unclipped timing predictions and tail errors |
| `condition_effects/` | Conditioning/ablation diagnostics |
| `minimum_jerk/` | Generated fits, participant fidelity, and assumption sensitivity |
| `dashboard/` | Compact dashboard assets, not the complete analysis |

## Review priorities

1. **Preprocessing and selection.** Independently trace representative raw trials
   through frame timing, missing frames, filtering, onset, arrival, resampling,
   and unit scaling. Audit exclusions and all relevant outcome labels. Verify
   that early arrival is not confused with movement before the go signal and
   that the late cutoff is defined relative to the target window. Assess the
   retained fixation-warning trials and sensitivity to arbitrary thresholds.
2. **Leakage and fairness.** Check participant separation, train-only fitting of
   preprocessing/PCA/scalers, validation-only selection, and context/query
   separation. Distinguish reconstruction of an observed trial from generation
   using context trials. Compare information access, code dimensions, timing
   heads, conditioning, residual sampling, and tuning budgets across baselines.
3. **Actual model.** Trace encoder/decoder inputs, KL and reconstruction losses,
   timing weighting, latent sampling/aggregation, and fingerprint-conditioned
   generation. Does this really implement the stated low-dimensional subject
   model? Check whether generation requires more than a two/three-number mean.
   A continuous VAE latent space does not by itself establish interpretable axes,
   meaningful distances, valid averaging, or a stable personal fingerprint.
4. **Independent arithmetic.** Recompute reported MSE, MAE, R-squared, KS, MMD,
   JSD/TV, enrollment accuracy, and paired comparisons from saved predictions
   where applicable. State each formula, units, aggregation level, denominator,
   and direction of improvement. Check outliers, NaNs, zero-variance features,
   confidence intervals, corrections, and the unit of independent replication.
   Explain any mismatch with a concrete table row or report page.
5. **Scientific claims.** Separate reconstruction, timing, generation, and
   enrollment; no global "CVAE is better" claim from one metric. Check whether
   any conditioning benefit, strategy interpretation, low-data explanation,
   clustering claim, or dimension choice exceeds the tests actually performed.
   A failure to reject distributional difference is not proof of equivalence.
   Closed-set enrollment after context is not zero-shot subject identification.
6. **Submovements and heatmaps.** Compare the adaptation with Jason's source
   at `external/jason-submovements/` (upstream:
   `https://github.com/JasonFriedman/submovements`). Check minimum-jerk basis,
   time/space units, component selection, constraints, fitting failures, and
   sensitivity. Distinguish empirical latent associations from correlations
   computed on decoder-generated samples. Neither is a causal strategy label.
7. **Reports and dashboard.** Trace every main numerical conclusion and figure
   to source data; inspect labels, plotted examples, selection, and references.
   Verify the ZIP against current source and assets, not only the working-tree
   dashboard. Check whether limitations and assumptions are disclosed clearly.

## Dashboard and delivery

Current app: `src/confirmatory_dashboard.py`.
Package: `output/release/Interception_Movements_Advisor_Package.zip`.
Builder: `scripts/package_final_release.py`; usage notes: `release/README.txt`.

The manifest declares live generation at fold 0 / seed 42 for n=2,3,4,8.
Stored generated validation examples cover n=3,8 and all test participants at
seed 42; benchmark tables aggregate the repeated runs. Verify this distinction
in the interface and report. Compact assets are not substitutes for full-run
evidence. The ZIP contains pseudonymous derived measurements; do not upload it
or raw data to external services as part of the review.

Useful commands, run from the repository:

```powershell
git status --short
python -m pytest -q -p no:cacheprovider tests/test_confirmatory_protocol.py tests/test_confirmatory_controls.py tests/test_fidelity_feature_dimensions.py tests/test_model_ablation_controls.py
python -m streamlit run src/confirmatory_dashboard.py --server.port 8504
```

Use a different free port if needed. The complete test suite includes slow toy
training tests and legacy asset tests. Passing it does not establish scientific
validity. Report failures and skips honestly, including missing local artifacts.
The current environment uses Python 3.13.5; see `requirements.txt`. Report building
also imports `reportlab`; check the environment instead of assuming dependencies
are fully specified. Do not rebuild reports just to review them.

Run and result directories are largely Git-ignored. A GitHub clone alone is
not the full local evidence set; the ZIP is only a compact subset. Work here or
explicitly report which missing artifacts prevent independent verification.
Archived files are historical context, not current results. Some shared code
and legacy tests remain in place to avoid breaking imports; see the source map.

## Deliverables

Write `review/REVIEW_FINDINGS.md`: findings ordered by severity, each with a
file/line or PDF page, supporting output/reproduction, scientific impact, and
proposed fix. Separate confirmed errors, methodological limitations, unsupported
claims, and questions for Jason/Moni. Include what you actually ran, what you
checked manually, what passed, and what remains unverified. Do not promise that
any review proves the complete absence of mistakes.

Add `review/CLAIM_EVIDENCE_MAP.md`: report claim/figure -> code -> saved evidence
-> independent check -> verdict. Keep calculations or regression tests alongside
it without overwriting existing results. Finish with a clear ready/not-ready
assessment and the minimum fixes needed before sending. Do not fix scientific
issues silently; first return the findings for discussion.

## Cleanup verification, not scientific validation

The 2026-09-02 cleanup changed navigation and moved historical files, not active
model code or scientific outputs. All 15 archived files were SHA-256 checked
against their original bytes. Source compilation and import checks for 12 current
entry points passed. The four targeted test files listed above plus
`tests/test_confirmatory_dashboard.py` passed: 17 tests in 24.03 seconds. The
full suite and model sweeps were not rerun during cleanup.

Frozen output SHA-256 values for detecting accidental changes:

```text
Scientific report  1275d667fc77f6efa8ffd4a27553b0487c37700e97227a829739da38407032b2
Student guide      fdddd8fe0ef192b818331d97747c8d4b3c75dd5f4621f0d9a7c48f9359f8849a
Advisor ZIP        7e5be17dabd97370a92ba4694986642a8604a621c0a7030d0a69be5205c4af96
```
