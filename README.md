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

- **Current correction record: [REVIEW_CORRECTIONS.md](REVIEW_CORRECTIONS.md)**.
  The PDFs and dashboard now consume `studies/review_corrected_evaluation/`.
  Original checkpoints and unaffected submovement/condition analyses remain in
  the frozen training study; neither raw data nor main neural weights changed.
- **Independent review: [REVIEW_HANDOFF.md](REVIEW_HANDOFF.md)** lists the exact
  current reports, source modules, saved predictions, and checks to perform.
- `studies/final_strategy_evaluation/`: frozen confirmatory protocols, runs, and
  original result tables (not the current multivariate distance evaluation).
- `studies/review_corrected_evaluation/`: corrected evaluations, additional
  timing-head tests, event audit, behavioural probes, and verification manifest.
- `src/`: shared code; [source map](src/README.md).
- `scripts/`: [current entry points and dependency notes](scripts/README.md).
- `reports/`: scientific-report and interpretation-guide builders.
- `output/pdf/`: final scientific report and companion results guide.
- `output/release/`: standalone dashboard bundle and email-ready ZIP.
- `tests/`: protocol and model checks.
- `archive/`: [historical snapshots and superseded entry points](archive/README.md).

The canonical data and a few shared diagnostics remain under
`studies/strategy_window_comparison/`; the final study still reads them. Do not
archive that entire directory. Most run/checkpoint/result files are local and
Git-ignored, so a GitHub clone is not the full evidence set. The current PDFs and
compact advisor ZIP are tracked; do not mistake the ZIP for a complete run archive.

## Final outputs

Build the report PDFs:

```powershell
python reports\build_final_reports.py
```

Build dashboard assets and launch the confirmatory dashboard:

```powershell
python scripts\build_confirmatory_dashboard_assets.py
python -m streamlit run src\confirmatory_dashboard.py
```

Build the standalone advisor package:

```powershell
python scripts\package_final_release.py
```

Run verification:

```powershell
python -m pytest -q
```

Raw data remains read-only at `D:\DropBox\Dropbox\results` and stimuli at
`D:\DropBox\Dropbox\stimuli`. The project never writes to Dropbox.
