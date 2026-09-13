# Interception Movement Strategy Fingerprints

This repository studies whether fast human interception movements admit compact
participant-specific latent representations that reproduce held-out trajectory,
timing, and kinematic-feature distributions. The confirmatory analysis uses the
strategy-inclusive interval from MAT target motion onset to the last tracker
sample (arrival proxy). Marker 5 is target appearance, not motion onset.

The frozen `strategy-confirmatory-v1` protocol contains 4,732 condition-2 trials
from 28 participants. Table-plane x-y position is filtered at 10 Hz and resampled
to 100 phase points; physical initiation and movement time are withheld from the
encoder and decoded separately. Four deterministic 17/4/7 participant folds give
every participant one out-of-fold test appearance. Neural models are repeated at
seeds 42, 43, and 44.

The primary CVAE sweep (`n=2,3,4,8`) is benchmarked against spline+PCA, a
conditional deterministic autoencoder, an unconditional VAE, and a
condition-only stochastic Ridge baseline. Minimum-jerk decomposition is a
secondary, model-order-sensitive kinematic analysis.

## Navigation

- **September 13 trajectory follow-up:** [results and methods](production/trajectory_distribution_2026_09_13/README.md).
  Complete normalized-path generation, evaluated separately from the existing
  feature scores. The laboratory PDF and Paz's drafts are unchanged.
- **Paz's course drafts:** `production/8_pages_draft.pdf` and
  `production/10_min_presentation.pdf`; see the separate
  [progress review](production/PAZ_DRAFT_REVIEW_2026-09-13.md).
- **New readers:** use the setup instructions below, then follow the
  [source map](src/README.md) and [script map](scripts/README.md).
- **Current manuscript review: [production/README.md](production/README.md).**
  Start with the detailed methods/results PDF and its page guide. The eight-page
  main paper, appendix selection, introduction and discussion will follow review.
- **For Paz: [PAZ_REVIEW_HANDOFF.md](PAZ_REVIEW_HANDOFF.md).**
- **Meeting context: [SIMAAN_MONI_REPORT_HANDOFF_2026-09-08.md](SIMAAN_MONI_REPORT_HANDOFF_2026-09-08.md).**

- **Audit follow-up fixes: [production/FIXES_AND_CLEANUP.md](production/FIXES_AND_CLEANUP.md).**
  [The earlier correction record](archive/review_cleanup_2026-09-08/REVIEW_CORRECTIONS.md)
  remains archived for provenance.
  The dashboard and September 5 report builders consume
  `studies/review_corrected_evaluation/`; the current manuscript also uses the
  independent audit tables in `production/audit_2026_09_08/`.
  Original checkpoints and historical submovement/condition analyses remain in
  the frozen training study; neither raw data nor main neural weights changed.
- **Historical reviews:** [archive/review_cleanup_2026-09-08/](archive/review_cleanup_2026-09-08/).
- `studies/final_strategy_evaluation/`: frozen confirmatory protocols, runs, and
  original result tables (not the current multivariate distance evaluation).
- `studies/review_corrected_evaluation/`: corrected evaluations, additional
  timing-head tests, event audit, behavioural probes, and verification manifest.
- `src/`: shared code; [source map](src/README.md).
- `scripts/`: [current entry points and dependency notes](scripts/README.md).
- `reports/`: scientific-report and interpretation-guide builders.
- `production/`: current manuscript draft, reproducible builder and new audit evidence.
- `output/pdf/`: frozen September 5 paper, supplementary appendix and results guide.
- `output/release/`: current multi-model dashboard ZIP and frozen September 5 advisor ZIP.
- `tests/`: protocol and model checks.
- `archive/`: [historical snapshots and superseded entry points](archive/README.md).

The canonical data and a few shared diagnostics remain under
`studies/strategy_window_comparison/`; the final study still reads them. Do not
archive that entire directory. Most run/checkpoint/result files are local and
Git-ignored, so a GitHub clone is not the full evidence set. The September 5 PDFs
and compact advisor ZIP are tracked; the ZIP is not a complete run archive.

## Set up and explore

Use Python 3.11 or newer. Local verification used Python 3.13 on Windows.
Run these commands from the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python scripts\prepare_review_workspace.py
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python -m streamlit run src\confirmatory_dashboard.py
```

Preparation restores the newest available bundled dashboard evidence: eight
neural checkpoints and four spline references from the multi-model ZIP. It requires no private
recordings and refuses to overwrite differing existing artifacts. These commands
are intended for a fresh clone; an existing research workspace may already have
newer results. The dashboard supports CVAE and spline + PCA at n=2,3,4,8,
and VAE and CAE at n=3,8. VAE n=8 defaults to the lowest cohort-average
generation distances; reconstruction has a different ranking. If the new ZIP
is absent, preparation falls back to the historical CVAE-only bundle.

Reading the manuscript PDF needs no Python setup. Rebuilding the new manuscript
figures additionally requires the canonical processed cache, audit tables and
the additional saved checkpoints listed in [production/README.md](production/README.md).
The bundled dashboard assets alone are insufficient. Training from raw recordings
requires access to the private CSV/MAT data and local path configuration in
`config.py` (`DATA_RAW_DIR` and `STIMULI_DIR`).

To follow the implementation, read loading and preprocessing first, then the
participant split, model and training code, evaluation, and dashboard. The
[source map](src/README.md) identifies the files at each stage; the
[script map](scripts/README.md) identifies the runnable entry points.

## Build and verify

The commands below assume the required local research artifacts are present and
`python` refers to the configured environment. In PowerShell, you can substitute
`.\.venv\Scripts\python` for `python`.

Build the current manuscript section:

```powershell
python production\build_methods_reconstruction.py
python production\verify_detailed_report.py
```

Rebuild the earlier complete-paper layout (not the new manuscript section):

```powershell
python reports\build_final_reports.py
```

Build dashboard assets and launch the confirmatory dashboard:

```powershell
python scripts\build_confirmatory_dashboard_assets.py
python scripts\build_multimodel_dashboard_assets.py
python -m streamlit run src\confirmatory_dashboard.py
```

Build the current standalone dashboard package (preserves the historical ZIP):

```powershell
python scripts\package_multimodel_dashboard.py
```

Run verification:

```powershell
python -m pytest -q
```

Raw data remains read-only at `D:\DropBox\Dropbox\results` and stimuli at
`D:\DropBox\Dropbox\stimuli`. The project never writes to Dropbox.
