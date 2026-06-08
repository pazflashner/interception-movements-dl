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

### VAE Architecture
- **Encoder**: trajectory (300-dim) + condition (4-dim) → hidden (256) → (μ, log σ²)
- **Decoder**: z + condition → hidden (256) → reconstructed trajectory (300-dim)
- **Condition vector**: one-hot starting position (3) + binary side (1)
- **Loss**: MSE reconstruction + β·KL divergence

### Evaluation
- Reconstruction MSE (vs spline baseline)
- Latent-kinematics Spearman correlations
- Behavioral probing R² (Linear Regression + SVR)
- Generative fidelity (KS test)

## Project Structure
```
├── config.py              # All configuration parameters
├── main.py                # Main pipeline entry point
├── requirements.txt       # Python dependencies
├── src/
│   ├── data_loading.py    # Raw data loading & filename parsing
│   ├── preprocessing.py   # Filtering, segmentation, normalization
│   ├── features.py        # Kinematic feature extraction
│   ├── baseline_kmeans.py # Phase 1: K-Means baseline
│   ├── baseline_spline.py # Phase 2: Spline baseline
│   ├── vae_model.py       # Phase 3: CVAE model & dataset
│   ├── train.py           # Training loop & data splitting
│   ├── evaluate.py        # Full evaluation suite
│   └── dashboard.py       # Streamlit interactive dashboard
├── data/
│   ├── raw/               # Raw subject data (from Dropbox)
│   ├── processed/         # Preprocessed trial cache
│   └── stimuli/           # Stimulus trajectory files
├── models/                # Saved model checkpoints
└── results/               # Evaluation outputs
```

## Authors
- Seman Libbiss (semanlibbiss@mail.tau.ac.il)
- Paz Flashner (pazflashner@mail.tau.ac.il)

In collaboration with Prof. Jason Friedman, Dept. Physical Therapy & Sagol School of Neuroscience, Tel Aviv University.
