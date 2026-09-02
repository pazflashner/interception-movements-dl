# Script navigation

Run commands from the repository root. This index distinguishes the current
four-fold study from earlier fixed-split work. Read `REVIEW_HANDOFF.md` before
reviewing results. This is a code map, not an instruction to rerun the pipeline.

## Current study entry points

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
| Current dashboard assets | `build_confirmatory_dashboard_assets.py` |
| Current standalone package | `package_final_release.py` |

The report builder is `../reports/build_final_reports.py` and the app is
`../src/confirmatory_dashboard.py`. Inspect CLI arguments before running anything;
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
