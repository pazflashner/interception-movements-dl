"""
Configuration for the Interception Movements Deep Learning Pipeline.
"""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
STIMULI_DIR = PROJECT_ROOT / "data" / "stimuli"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# ── Recording parameters ──────────────────────────────────────────────────────
RECORDING_HZ = 240          # Raw data sampling rate
STIMULUS_HZ = 60            # Stimulus trajectory sampling rate
NORMALISED_LENGTH = 100     # Fixed trajectory length after resampling

# ── Filtering ─────────────────────────────────────────────────────────────────
LOWPASS_CUTOFF_HZ = 10      # Butterworth low-pass filter cutoff
LOWPASS_ORDER = 4            # Filter order (4th-order → 24 dB/octave)

# ── Filename encoding ─────────────────────────────────────────────────────────
STARTING_POSITIONS = {1: 120, 2: 140, 3: 160}  # mm
SPEED_RANGES = {
    1: (255, 300),
    2: (298, 350),
    3: (340, 400),
}  # mm/s – speed randomly selected within range

# ── Condition filter ──────────────────────────────────────────────────────────
CONDITION_FREE_EYE = 2  # Only analyse condition 2 (free eye movements)

# ── Marker value ──────────────────────────────────────────────────────────────
STIMULUS_ONSET_MARKER = 5

# ── Data splits (leave-N-subjects-out) ────────────────────────────────────────
N_TRAIN = 17
N_VAL = 4
N_TEST = 7

# ── VAE hyperparameters ──────────────────────────────────────────────────────
LATENT_DIMS_SWEEP = [2, 3, 4, 8, 16]
DEFAULT_LATENT_DIM = 3
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 200
KL_WEIGHT = 1.0  # β for β-VAE; 1.0 = standard VAE

# ── Spline baseline ──────────────────────────────────────────────────────────
SPLINE_DEGREE = 3
SPLINE_N_KNOTS = 5

# ── K-Means baseline ─────────────────────────────────────────────────────────
KMEANS_N_CLUSTERS_RANGE = range(5, 35)

# ── CSV column names ──────────────────────────────────────────────────────────
CSV_COLUMNS = [
    "frame", "x", "y", "z",
    "rot1", "rot2", "rot3",
    "time", "marker",
]
