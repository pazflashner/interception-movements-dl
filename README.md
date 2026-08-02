# Interception Movements – Deep Learning Pipeline

**From Raw Trajectories to Individual Behavioral Signatures**

A machine learning pipeline for analyzing 3D interception movement data using a Conditional Variational Autoencoder (CVAE). The model learns individual movement "fingerprints" from kinematic trajectory data captured during a ballistic interception task.

## Project Overview

This project develops a deep learning model that maps complex interception movement trajectories into a low-dimensional latent space, creating compact, interpretable behavioral signatures for each individual. The pipeline progresses through three phases:

1. **K-Means Clustering Baseline** – Validates that individual trajectories are inherently separable
2. **Polynomial Spline Baseline** – Establishes a non-ML reconstruction reference
3. **Conditional VAE** – Learns a structured latent space encoding individual movement style

## Data

### Experiment Data
- **Source**: [Dropbox – Experiment Data](https://www.dropbox.com/scl/fo/h7zb2xesuvz7xvqy1u8r3/ABK9KpEatdyf_MxkrkyObm0?rlkey=2m5xvbomsfkpn8hj2zwyfn5yq&dl=0)
- **Format**: CSV files with 9 columns (frame, x, y, z, rot1, rot2, rot3, time, marker)
- **Recording rate**: 240 Hz
- **Structure**: Each subdirectory = one subject; filenames encode trial metadata

### Stimulus Trajectories
- **Source**: [Dropbox – Stimuli](https://www.dropbox.com/scl/fo/yv5oydibhmisudu80fn8b/ACZ30rNkccF5id6UGvr2JVA?rlkey=tuiwuwsu56x9ytcr8j8tgx4ri&dl=0)
- **Rate**: 60 Hz

### Filename Convention
Files follow `li_{condition}_{sp}_{side}_{rep}.csv`:
| Field | Values | Description |
|-------|--------|-------------|
| condition | 1, 2 | 1 = fixed gaze, **2 = free eye movements** (used) |
| sp | 1, 2, 3 | Starting position (120/140/160 mm) & speed range |
| side | 1, 2 | Starting side: 1 = left, 2 = right |
| rep | 1–N | Trial repetition number |

### Setup
Download data from Dropbox and place subject folders under `data/raw/`:
```
data/raw/
  subject01/
    li_2_1_1_1.csv
    li_2_2_2_3.csv
    ...
  subject02/
    ...
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Build the dataset
```bash
python scripts/make_dataset.py                    # reuses the trial cache
python scripts/make_dataset.py --force            # re-run preprocessing
python scripts/make_dataset.py --seed 7           # splits.json for another seed
python scripts/make_dataset.py --drop-implausible # exclude QC-flagged trials
```

Writes three artefacts into `data/processed/`:

| File | Contents |
|------|----------|
| `dataset.npz` | `trajectories` (N, 100, 3), `timing` (N, 2), `conditions` (N, 4), `subjects`, `trial_ids`, `sp`/`side`/`rep`, `timing_plausible` |
| `metadata.csv` | one row per trial: identifiers + kinematic features (seconds, mm/s) |
| `splits.json` | leave-N-subjects-out assignment — subject ids **and** trial indices |

Splits are produced by `src.train.split_subject_ids`, the same function training
calls, so `splits.json` is a record of the split rather than a second
implementation that can drift.

Current build: **4,763 trials, 28 subjects**, split 17 / 4 / 7.

### Smoke test (no data required)
```bash
python tests/test_vae_smoke.py     # standalone
pytest tests/test_vae_smoke.py -v  # or under pytest
```

Runs the model, loss and full training loop on synthetic trials from
`src/dummy_data.py`, built to the same schema as preprocessed real trials. It
checks that the loss decreases, gradients reach every parameter, the timing head
learns, splits stay disjoint, and checkpoints round-trip. Run it first: if it
passes, anything that then fails on real data is a data problem, not a model one.

### Full Pipeline
```bash
python main.py --data-dir path/to/raw/data
```

### Individual Phases
```bash
python main.py --phase 1              # K-Means baseline
python main.py --phase 2              # Spline baseline
python main.py --phase 3              # VAE training
python main.py --phase eval           # Evaluation
python main.py --sweep                # Latent-dim hyperparameter sweep
```

### Reproducibility & repeated runs

Every training run is described by a single `RunConfig` and gets its own
directory under `results/runs/`:

```
results/runs/z3_seed7_20260725-143012/
├── config.yaml       # seed, split sizes, model, optimiser, git commit, versions
├── checkpoint.pt     # best-val weights, with the same config embedded
└── history.json      # per-epoch train/val losses
```

`config.yaml` is written before training starts, so an interrupted run is still
identifiable, and the checkpoint carries its own copy of the config — a
checkpoint can never drift from the settings that produced it.

```bash
python main.py --phase 3 --seed 7            # one seeded run
python main.py --phase 3 --seeds 0 1 2 3 4   # repeat, one run dir per seed
python main.py --phase 3 --config results/runs/z3_seed7_.../config.yaml
python main.py --phase eval --run results/runs/z3_seed7_.../
python main.py --summarise                   # table of all runs + mean ± sd
```

The seed drives the subject split, weight initialisation, DataLoader shuffling,
the VAE's sampling, and the K-Means baseline.

**Read results as distributions, not point estimates.** With 28 subjects
(17 train / 4 val / 7 test), a single run is noisy: changing only the seed
changes which 7 subjects are held out. Repeat each configuration over several
seeds and compare mean ± sd via `--summarise` before concluding that one
setting beats another.

### Interactive Dashboard
```bash
streamlit run src/dashboard.py
```

The dashboard provides:
- **Inference Mode**: Upload raw CSV → extract latent fingerprint
- **Exploration Mode**: Manipulate latent sliders → generate trajectories

## Pipeline Details

### Preprocessing
1. Filter for condition 2 (free eye movements)
2. Low-pass Butterworth filter (10 Hz cutoff, 4th order)
3. Segment movement using stimulus onset marker (value=5) and velocity threshold
4. Temporal normalization to T=100 frames via cubic spline interpolation
5. Spatial normalization (subtract initial position)

### K-Means baseline (Phase 1)

Two representations are clustered and scored against true subject labels:
the full normalised trajectory (300 dims), and extracted kinematic features.

The feature matrix is built from an **explicit allowlist**
(`config.KMEANS_FEATURE_COLUMNS`), not from every numeric column. Taking all
numeric columns sweeps in the trial counter (`rep`, 1–180), the task labels
(`sp`, `side`), a deterministic duplicate of `sp` (`starting_position_mm`), and
a constant (`condition`). Those five carry no subject identity — clustering on
them alone scores ARI = −0.001 — but `StandardScaler` weights every column
equally, so they acted as 5 of 16 dimensions of pure noise that **diluted** the
result rather than inflating it.

Peak ARI per seed, aggregated over 10 seeds (`--seeds 0 1 2 3 4 5 6 7 8 9`):

| representation | ARI | NMI |
|---|---|---|
| trajectories (300 dims) | 0.0800 ± 0.0015 | 0.2506 |
| **features — allowlist, 11 dims** | **0.0963 ± 0.0017** | **0.2828** |
| features — all numeric, 16 dims | 0.0341 ± 0.0012 | 0.1285 |

**This flips the Phase 1 conclusion.** With the nuisance columns included,
features lost to raw trajectories by 0.046; with the allowlist they win by
0.0163, consistently on 10 of 10 seeds (Wilcoxon W = 0.0, p = 0.002). That
matches what the proposal predicted — that raw trajectories would cluster poorly
because of their dimensionality, and extracted parameters would do better.
Trajectory clustering is bit-identical before and after, which confirms nothing
else moved.

Agreement with subject identity is still only **weak** in absolute terms
(ARI is chance-corrected; 0.096 ≪ 1), so this remains a useful negative result
setting the bar for Phase 3. ARI plateaus at ~0.10 across k = 25–60, so
`KMEANS_N_CLUSTERS_RANGE = range(5, 35)` is wide enough.

Caveat carried from the report: the seed varies only K-Means initialisation, so
the ±0.0017 spread is algorithmic variance, **not** subject-sampling variance —
a floor on the real noise, not an estimate of it.

```bash
python main.py --phase 1 --seeds 0 1 2 3 4 5 6 7 8 9   # sweep -> kmeans_seed_sweep.csv
python -m src.report_kmeans                            # -> kmeans_phase1_report.docx + figures
```

### VAE Architecture
- **Encoder**: trajectory (300-dim) + timing (2-dim) + condition (4-dim) → hidden (256) → (μ, log σ²)
- **Decoder**: z + condition → hidden (256) → trajectory (300-dim) **and** timing (2-dim)
- **Condition vector**: one-hot starting position (3) + binary side (1)
- **Timing channels**: `movement_time_s`, `initiation_time_s`
- **Loss**: trajectory MSE + λ·timing MSE + β·KL divergence

#### Why the model predicts movement time

Resampling every trial to T=100 frames is what makes the input dimension
uniform, but it also removes the time axis. Without it the model only ever sees
trajectory *shape*: two movements along the same path at half the speed become
identical inputs, velocity is recoverable only up to an unknown scale factor,
and a sampled latent produces a path with no duration to play it back over.

Movement time and initiation time are therefore carried as explicit channels —
appended to the encoder input so the latent must encode them, and reconstructed
by a second decoder head. Generating from `z` now yields a shape *and* a
duration, which together determine a physical velocity profile:

```
dt      = movement_time_s / (T - 1)
v(t)    = d(trajectory)/d(phase) / dt        # mm/s
```

Set `predict_timing: false` in a run config to recover the shape-only model for
comparison.

All kinematic features are reported in **physical units** (seconds, mm/s) for
the same reason: a gradient taken on the resampled trajectory is a shape
derivative whose scale depends on the discarded duration.

#### Timing quality control

The task requires ballistic movements under one second, but `find_movement_window`
extends to the last frame above the velocity threshold, which can run past the
interception into the return movement. About **3.5% of trials** land outside
plausible bounds (movement time up to 9.7 s against a 0.63 s median). These are
flagged in `dataset.npz['timing_plausible']` and `metadata.csv`, not dropped —
whether to exclude them is an analysis decision, but with a timing head they can
no longer be ignored, since a handful of 9 s outliers dominate the timing loss.
Bounds live in `config.py` (`MIN/MAX_MOVEMENT_TIME_S`, `MAX_INITIATION_TIME_S`).

### Evaluation
- Reconstruction MSE (vs spline baseline)
- Timing reconstruction: MAE / RMSE / R² for movement and initiation time, in ms
- Latent-kinematics Spearman correlations
- Behavioral probing R² (Linear Regression + SVR)
- Generative fidelity (KS test on path length **and** on movement time)

## Project Structure
```
├── config.py              # All configuration parameters
├── main.py                # Main pipeline entry point
├── requirements.txt       # Python dependencies
├── scripts/
│   └── make_dataset.py    # Raw CSVs → dataset.npz + metadata.csv + splits.json
├── tests/
│   └── test_vae_smoke.py  # Model/loss/training-loop tests on dummy data
├── src/
│   ├── data_loading.py    # Raw data loading & filename parsing
│   ├── preprocessing.py   # Filtering, segmentation, normalization
│   ├── features.py        # Kinematic feature extraction (seconds, mm/s)
│   ├── dummy_data.py      # Synthetic trials for data-free testing
│   ├── baseline_kmeans.py # Phase 1: K-Means baseline
│   ├── baseline_spline.py # Phase 2: Spline baseline
│   ├── vae_model.py       # Phase 3: CVAE model, timing head & dataset
│   ├── run_config.py      # Per-run config (config.yaml) & seeding
│   ├── train.py           # Training loop & data splitting
│   ├── evaluate.py        # Full evaluation suite
│   └── dashboard.py       # Streamlit interactive dashboard
├── data/
│   ├── raw/               # Raw subject data (from Dropbox)
│   ├── processed/         # trials.pkl, dataset.npz, metadata.csv, splits.json
│   └── stimuli/           # Stimulus trajectory files
├── models/                # Legacy checkpoint location
└── results/
    ├── runs/              # One directory per run: config.yaml + checkpoint.pt
    └── runs_summary.csv   # Aggregated across runs (--summarise)
```

## Authors
- Seman Libbiss (semanlibbiss@mail.tau.ac.il)
- Paz Flashner (pazflashner@mail.tau.ac.il)

In collaboration with Prof. Jason Friedman, Dept. Physical Therapy & Sagol School of Neuroscience, Tel Aviv University.
