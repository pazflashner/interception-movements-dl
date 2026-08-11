# Interception Movements – Deep Learning Pipeline

> **Audited results notice:** The original sections below document the legacy
> pipeline and may contain superseded labels, units, and numerical claims. Use
> `CORRECTED_STUDY_README.md` and
> `output/pdf/Interception_Corrected_Results.pdf` for the corrected-v3 protocol
> and results.

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
python scripts/build_dashboard_assets.py
python -m streamlit run src/dashboard.py
```

The final-study dashboard provides:
- **Generate**: switch between `n=2`, `n=3`, and `n=8`; select a population or held-out participant fingerprint; change latent and task controls; inspect generated 2-D trajectories, velocity, timing, and minimum-jerk components.
- **Distribution check**: compare disjoint held-out query trials with generated samples for each reported output using KS/Wasserstein or JSD/total variation.
- **Model comparison**: inspect repeated-seed reconstruction, timing, identification, and generation metrics.
- **Protocol and downloads**: review assumptions for Prof. Friedman and export the report and compact result tables.

`python scripts/package_dashboard.py` creates the email-sized standalone bundle at
`output/share/Interception_Movement_Dashboard.zip`. The bundle contains the trained
models and compact held-out results, but no raw Dropbox trajectories.

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

#### Reconstruction vs the spline baselines — current status

```bash
python scripts/baseline_report.py --seeds 0 1 2 3 4 5 6 7 8 9 --latent-dim 5 --pca-components 3
```

Three references, which measure different things:

| baseline | capacity | sees the test trial? | generalises to new subjects? |
|---|---|---|---|
| Spline, per-trial fit | 27 params/trial | **yes** | n/a |
| Spline+PCA | `n_components`/trial | no | yes (basis fitted on train) |
| CVAE | `latent_dim`/trial | no | yes |

The per-trial spline (MSE ≈ 0.012 mm²) is an **interpolation ceiling**, not a
competing representation — it gets 9× the capacity *and* is fitted to the trial
it reconstructs. Spline+PCA is the like-for-like comparison.

**Held-out reconstruction MSE (mm²), 5 seeds, paired within seed:**

| CVAE config | CVAE | Spline+PCA (z=3) | wins | mean Δ |
|---|---|---|---|---|
| z=3, with timing | 0.416 ± 0.061 | 0.317 | 0/5 | −34.6% |
| z=3, shape-only | 0.305 ± 0.158 | 0.317 | 3/5 | +5.9% |
| z=5, with timing | 0.262 ± 0.106 | 0.317 | 4/5 | +19.2% |

The timing head costs roughly **two latent dimensions** (R² 0.88/0.94 has to be
stored somewhere). At z=3 with timing only ~1 dim is left for shape against
PCA's 3, which is why that row loses outright. The z=5 row is the matched-*shape*
comparison: 3 shape dims + 2 for timing.

The timing head costs roughly **two latent dimensions** (R² 0.88/0.94 has to be
stored somewhere). At z=3 with timing only ~1 dim is left for shape against
PCA's 3, which is why that row loses outright. The z=5 row is the matched-*shape*
comparison: 3 shape dims + 2 for timing. Reconstruction MSE alone is
inconclusive at 10 seeds (9/10 seeds, p = 0.065) — see the full battery below.

### §4 evaluation — CVAE vs the spline representation

```bash
python scripts/evaluate_report.py                       # §4 plan on one run
python scripts/compare_representations.py --seeds 0 1 2 3 4 5 6 7 8 9 --latent-dim 5
```

The proposal evaluates the CVAE alone, which answers "is it any good?" but not
"is it worth it?". `compare_representations.py` runs the **same** battery on
both representations, on the same held-out subjects, at the same code width,
both encoding shape *and* timing — so no metric difference is a difference
between two implementations of the same test.

| | code/trial | shared params | form |
|---|---|---|---|
| CVAE | 5 | **292,920** | non-linear, stochastic, KL-regularised |
| Spline+PCA | 5 | **174** | linear, deterministic, no prior |
| Random codes | 5 | 0 | N(0, I) — the floor |

The random control is not decoration: R² is unbounded below and leave-one-
subject-out over 7 subjects is volatile, so a figure like "R² = −4.6" is
uninterpretable without knowing what noise scores on the same subjects.

**10 seeds, paired within seed** (n=10 allows p as low as 0.002):

| Metric | CVAE | Spline+PCA | Random | wins | p |
|---|---|---|---|---|---|
| Movement time R² ↑ | **0.883 ± 0.104** | 0.597 ± 0.195 | — | 10/10 | **0.002** |
| Initiation time R² ↑ | **0.940 ± 0.065** | 0.751 ± 0.086 | — | 10/10 | **0.002** |
| Max \|Spearman\| ↑ | **0.873** | 0.814 | 0.068 | 10/10 | **0.002** |
| Energy distance ↓ | **3.98 ± 1.37** | 31.5 ± 37.9 | — | 9/10 | **0.004** |
| R² curvature (unseen subj) ↑ | **+0.382 ± 0.581** | −0.088 ± 0.555 | −0.530 | 8/10 | **0.049** |
| KS features rejected ↓ | **9.51** | 9.96 | — | 7/10 | 0.043 |
| Recon MSE ↓ | 0.503 ± 0.439 | 0.704 ± 0.537 | — | 9/10 | 0.065 |
| Targets with R² > 0 (of 11) ↑ | 5.5 ± 1.7 | 4.3 ± 2.8 | 0.8 | 7/10 | 0.25 |
| MMD ↓ | 0.262 | 0.321 | — | 7/10 | 0.13 |
| Fingerprint between/within ↑ | 0.485 ± 0.071 | **0.530 ± 0.089** | 0.078 | 3/10 | 0.049 |
| Subjects MMD-matched (of 7) ↑ | 0/7 | 0/7 | — | — | — |

**The CVAE is a better model of movement, and not a better model of individuals.**

- **Wins decisively on timing** (R² 0.88 vs 0.60, 0.94 vs 0.75) and on energy
  distance (8× closer generated distributions). Non-linearity buys most where
  timing relates non-linearly to shape — which PCA cannot represent.
- **Wins on curvature for unseen subjects** (+0.38 vs −0.09, above the −0.53
  floor), the metric proposal §4.3 asks for. Note the ±0.58: with 7 test
  subjects this is a real effect on a badly underpowered estimate.
- **Ties on overall behavioural probing.** 5.5 vs 4.3 targets above chance,
  p = 0.25. But *both* beat the random floor of 0.8/11 (p = 0.004 and 0.006),
  so both representations carry genuine behavioural signal — they just do not
  differ from each other.
- **Loses on fingerprint separation** (0.485 vs 0.530, p = 0.049). Both sit far
  above random (0.078) yet **below 1**, meaning within-subject scatter still
  exceeds between-subject differences. 292,920 parameters do not separate
  individuals better than 174.
- **Neither achieves generative fidelity**: 0/7 subjects MMD-indistinguishable
  and ~9.5 of 11 KS features rejected, every seed, both representations.

That the individual-signature failure is identical across two representations
differing 1,700× in parameter count points at **the data**, not the
architecture — 28 subjects with within-subject variability dominating. It
matches Phase 1's near-chance ARI (0.098) independently.

Probing targets are the three named in §4.3 (initiation time, movement time,
curvature) plus peak speed, time-to-peak, path length and lateral deviation,
and four **within-subject SD** targets — the only way to test whether the
variance half of the §3.3 fingerprint carries information.

Not covered: the §4.4 benchmark against Prof. Friedman's submovement
decomposition pipeline, an external dependency not present in this repository.
It is reported as missing rather than approximated.

#### Conditioning

The condition vector is concatenated into **both** the encoder input and the
decoder input. That is what lets the latent model movement *style*: the decoder
is told the task, so `z` does not need to spend capacity encoding it.

`sp` indexes the starting position (120/140/160 mm) and the target speed range
(255–300 / 298–350 / 340–400 mm/s) **jointly** — the experiment confounds them
behind one filename index, so the one-hot over `sp` *is* the (start config,
target speed) encoding. Separate columns would be exactly collinear with it. The
randomised within-range speed is not recoverable from filenames and
`data/stimuli/` is empty in this checkout, so the range index is the finest
speed information available.

#### KL annealing

β ramps from 0 to `KL_WEIGHT` over `KL_ANNEAL_EPOCHS` (`linear`, the default;
`cyclical` and `none` also available). Without it the cheapest way to cut the KL
term early is to ignore `z` altogether, and a decoder that has learned to work
without the latent gets no gradient pulling it back.

Two consequences are handled explicitly in `train.py`:

- **Model selection cannot use the annealed loss.** While β ramps, val loss
  rises for reasons unrelated to model quality, so the "best" epoch would always
  be epoch 1 at β=0. Selection and early stopping run on `val_objective` — the
  loss recomputed at the *target* β — which is comparable across all epochs.
- **Early stopping is suppressed until the ramp finishes**, so a run cannot end
  before it has ever trained at the objective it is judged on.

#### Loss reduction (β actually means β)

`vae_loss` sums each term over its own dimensions and averages over the batch —
the standard ELBO. Averaging the reconstruction over 300 trajectory dims while
averaging the KL over `latent_dim` dims inflates the effective β by
`input_dim / latent_dim` (**100× at z=3**), which crushes the latent regardless
of training time. Fixing this took held-out reconstruction MSE from 1.81 to
0.70 and was the single change that made the model competitive.

`TIMING_WEIGHT` defaults to `300 / TIMING_DIM` so a timing channel and a
trajectory channel carry equal per-dimension weight; unweighted, timing would be
0.7% of the reconstruction signal.

Gradient clipping (`GRAD_CLIP_NORM = 5.0`) is on by default: a 9.7 s
segmentation artefact standardises to a z-score of ~25 and produced a 3×10⁵ loss
spike before clipping.

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
Implements the proposal's §4 plan; run with `scripts/evaluate_report.py`.
- **§4.1** Reconstruction MSE vs both spline references
- **§4.1** Timing reconstruction: MAE / RMSE / R², in ms
- **§4.2** Latent traversal (figures + per-dimension range, flagging collapse)
- **§4.2** Latent–kinematics Spearman correlations
- **§4.3** Behavioural probing R², **leave-one-subject-out** over 11 targets
  (7 means + 4 within-subject SDs), Linear + SVR
- **§4.4** Generative fidelity: KS **per feature**, plus MMD and energy distance
  with permutation p-values
- **§3.3** Per-subject fingerprints (latent mean *and* spread), with the
  between/within separation ratio

Probing is scored by LOSO because §4 asks for R² "for unseen subjects". Fitting
and scoring on the same 7 subjects — the previous behaviour — reported R² ≈ 0.96
for probes carrying no predictive content.

## Project Structure
```
├── config.py              # All configuration parameters
├── main.py                # Main pipeline entry point
├── requirements.txt       # Python dependencies
├── scripts/
│   ├── make_dataset.py            # Raw CSVs → dataset.npz + metadata.csv + splits.json
│   ├── baseline_report.py         # CVAE vs spline reconstruction, paired across seeds
│   ├── evaluate_report.py         # The proposal's §4 plan on a trained run
│   └── compare_representations.py # §4 battery on CVAE vs spline vs random codes
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
