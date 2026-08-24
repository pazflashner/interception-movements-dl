# Interception Movement Strategy Fingerprints

This repository studies whether fast human interception movements admit compact
participant-specific latent representations that reproduce held-out trajectory,
timing, and kinematic-feature distributions. The confirmatory analysis uses the
strategy-inclusive interval from target motion onset to finger arrival.

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

- `studies/final_strategy_evaluation/`: frozen confirmatory protocols, runs, and
  result tables.
- `src/`: shared loading, preprocessing, CVAE, evaluation, and submovement code.
- `scripts/`: reproducible data, training, evaluation, and reporting commands.
- `reports/`: scientific-report and interpretation-guide builders.
- `output/pdf/`: final scientific report and companion results guide.
- `output/release/`: standalone dashboard bundle and email-ready ZIP.
- `tests/`: protocol and model checks.
- `archive/movement_only_2026-08-10/`: local, ignored, explicitly outdated
  snapshot kept only for context.

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
