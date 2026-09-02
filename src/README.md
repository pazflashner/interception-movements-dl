# Source map

| Area | Files |
|---|---|
| Raw CSV/MAT linkage and event timing | `data_loading.py` |
| Frame grid, filtering, segmentation, resampling | `preprocessing.py`, `trajectory_view.py` |
| Kinematic feature definitions | `features.py` |
| CVAE, ablations, conditions, timing transform, normalization | `vae_model.py` |
| Training, early stopping, run configuration | `train.py`, `run_config.py` |
| Frozen participant folds and context/query partition | `confirmatory_protocol.py` |
| Reconstruction/timing prediction tables and aggregation | `evaluate.py`, `confirmatory_evaluation.py` |
| Context fingerprints, distribution distances, enrollment, tests | `context_query.py` |
| Spline+PCA representation and matched timing/generation | `baseline_spline.py`, `confirmatory_spline.py` |
| Condition-only baseline | `confirmatory_controls.py` |
| Condition perturbation diagnostics | `condition_effects.py` |
| Minimum-jerk decomposition | `submovements.py` |
| Current interactive dashboard | `confirmatory_dashboard.py` |
| Earlier K-means diagnostic | `baseline_kmeans.py`, `report_kmeans.py` |

Some evaluation and generation helpers also live in
`scripts/run_corrected_study.py` and `scripts/generate_final_samples.py`.
Follow imports rather than assuming all shared code lives under `src/`.

## Not the current model or interface

- `hierarchical_vae.py`: prototype with tests; its presence does not mean an HVAE
  was trained or benchmarked in the final study.
- `dashboard.py`, `strategy_dashboard.py`: earlier interfaces. Kept for history
  and existing tests; launch `confirmatory_dashboard.py` for the current study.
- `dummy_data.py`: synthetic toy data for plumbing tests, not real-data evidence.

Defaults and comments can outlive an experiment. Confirm effective settings
using the current runner, its frozen protocol, and the saved run configuration.
