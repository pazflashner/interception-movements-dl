# Script navigation

Run commands from the repository root. This index distinguishes the current
four-fold study from earlier fixed-split work. Read `../production/README.md` before
reviewing results. This is a code map, not an instruction to rerun the pipeline.

## Current study entry points

Post-review outputs are under `studies/review_corrected_evaluation/`. The report
and dashboard builders read this version and reuse original neural checkpoints.
The commands below train the original matrix; do not rerun training just to
rebuild a PDF. See `../production/FIXES_AND_CLEANUP.md` for current fixes and
`../archive/review_cleanup_2026-09-08/REVIEW_CORRECTIONS.md` for the earlier sequence.

| Stage | Script |
|---|---|
| Audit raw CSV/MAT completion | `audit_trial_completion.py` |
| Build canonical strategy dataset | `build_strategy_dataset.py` |
| CVAE, four folds / n=2,3,4,8 / three seeds | `run_confirmatory_cvae.py` |
| Matched spline+PCA baseline | `run_confirmatory_spline.py` |
| Conditional AE and unconditional VAE controls | `run_confirmatory_neural_ablation.py` |
| Condition-only Ridge baseline | `run_confirmatory_task_baseline.py` |
| Participant-balanced comparison and timing audit | `analyze_confirmatory_results.py` |
| Condition diagnostics | `analyze_condition_effects.py` |
| Fit real trajectories with minimum-jerk components | `extract_submovements.py` |
| Generate and fit confirmatory submovements | `run_confirmatory_submovements.py` |
| Aggregate submovement results | `analyze_confirmatory_submovements.py` |
| Submovement assumption sensitivity | `audit_submovement_assumptions.py` |
| Shared dashboard assets and historical diagnostics | `build_confirmatory_dashboard_assets.py` |
| Multi-model references and verified generation exports | `build_multimodel_dashboard_assets.py` |
| Current standalone multi-model package | `package_multimodel_dashboard.py` |
| Historical standalone package builder | `package_final_release.py` |
| Frozen-checkpoint reevaluation | `reevaluate_review_corrections.py` |
| Full pre-go/event audit | `audit_pre_go_motion.py` |
| All saved behavioural probes | `analyze_behavioral_probes.py` |
| Validation calibration and common timing heads | `analyze_timing_fairness.py` |
| Minimum-jerk sample-size reference | `calibrate_submovement_sampling.py` |
| Output coverage, invariance, PDF rendering | `verify_post_review_outputs.py` |
| All dashboard sections and dimensions | `check_dashboard_delivery.py` |
| Same-decoder fingerprint controls / matched component refits | `run_review_controls.py` |
| Direct-context baseline / matched-component summaries and sampling reference | `analyze_review_controls.py` |
| Restore bundled evidence/checkpoints in a fresh clone | `prepare_review_workspace.py` |
| Verify delivered summary arithmetic and PDF hashes | `verify_review_handoff.py` |

The current manuscript builder is `../production/build_methods_reconstruction.py`.
`../reports/build_final_reports.py` rebuilds the earlier complete-report layout;
`verify_review_handoff.py` verifies the September 5 delivery, not the new draft.
The current app is `../src/confirmatory_dashboard.py`. Inspect CLI arguments before running anything;
generation, decomposition, training, and report commands can overwrite outputs.

## Shared and historical code retained in place

- `run_corrected_study.py`: despite its historical name, imported by current
  model runners, evaluators, report, and dashboard-asset builder. Its CLI is not
  the final four-fold orchestration.
- `generate_final_samples.py`: generation/decomposition helpers imported by
  `run_confirmatory_submovements.py`.
- `run_final_baselines.py`: provenance for the earlier K-means diagnostic still
  referenced by the report; not the matched four-fold spline runner.
- `analyze_latent_submovements.py`, `evaluate_final_fingerprints.py`,
  `audit_submovement_stability.py`, `baseline_report.py`,
  `compare_representations.py`, `create_latent_traversals.py`,
  `evaluate_report.py`, `make_dataset.py`, and `augment_stimulus_conditions.py`:
  earlier analyses/utilities. Some defaults refer to old directories. Do not
  treat their outputs as final-study results without tracing their provenance.

Superseded standalone training/report/dashboard packaging entry points were
moved unchanged to `../archive/pre_confirmatory_2026-09-02/`. Their historical
root-relative paths are not maintained as runnable commands in the archive.
