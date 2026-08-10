# Experiments

Scripts that reproduce the results in `Interception_Preliminary_Results.pdf`.
Run from the repo root after `python scripts/make_dataset.py` has built
`data/processed/trials.pkl`.

| Script | Produces |
|---|---|
| `scan_speed.py` | `results/object_speed.csv` (per-trial target speed, screen px/s) |
| `sweep_run_10.py` | `results/sweep_summary_10seed.csv` (latent sweep, baseline MLP) |
| `experiment_full.py` | `results/experiment_full.csv` (MLP vs CNN vs +speed, all n) |
| `loss_experiment.py` | `results/loss_experiment.csv` (ELBO vs beta-VAE vs discriminative) |
| `make_figure.py` / `make_comparison_fig.py` / `make_loss_fig.py` / `latent_interp.py` | figures/*.png |
| `build_pdf.py` | the results PDF |

The CNN is also selectable in the main pipeline via `config.ARCHITECTURE = "cnn"`.
