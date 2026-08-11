# Interception Movement Fingerprint Dashboard

The dashboard uses the final 2-D trajectory-only conditional VAE checkpoints. It does not retrain the models and does not require the raw Dropbox dataset.

## Run

From the repository root:

```powershell
python -m pip install -r requirements_dashboard.txt
python -m streamlit run src/dashboard.py
```

The packaged version also includes `setup_and_launch.bat` and `launch_dashboard.bat`.

## Views

- **Generate:** choose `n=2`, `n=3`, or `n=8`; select a population or held-out participant fingerprint; change latent controls and task condition; inspect trajectory, velocity, timing, and a minimum-jerk decomposition; sample and export a generated distribution.
- **Distribution check:** compare a held-out participant's recorded query trials with generated samples for any reported output. Continuous outputs use KS and Wasserstein metrics; component count uses JSD and total variation.
- **Model comparison:** compare reconstruction, timing, enrolled-participant identification, and distribution metrics across latent dimensions.
- **Protocol and downloads:** review the fixed protocol, assumptions for Prof. Friedman, and download compact result tables.

## Interpretation

The dashboard is an analysis instrument, not evidence that every latent axis is a universal human trait. VAE axes may rotate, reflect, or permute across initialization seeds. Minimum-jerk component counts are kinematic descriptions and should not be labeled as cognitive strategies without independent validation.

