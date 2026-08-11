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
RUNS_DIR = RESULTS_DIR / "runs"   # one subdirectory per run: config.yaml + checkpoint

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42                   # default seed; overridden per run via --seed/--seeds
# Only 28 subjects, so single-run metrics are noisy — repeat across these seeds
# and report the spread rather than one number.
REPEAT_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

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
STIMULUS_ONSET_MARKER = 5   # marks the frame the object *appears* (always frame 1)

# ── Trial segmentation (event-based, from the .mat metadata) ──────────────────
# The object appears (marker=5), holds still for a randomised foreperiod
# (0.18-0.48 s), then starts moving — the "go-signal". The participant may only
# move once the object moves. We take the go-signal as the behavioural zero-time
# (reaction/wait time is measured from it), which removes the randomised
# foreperiod from the trajectories. The recording ends at finger arrival
# (pressedTime), i.e. at interception. Confirming with Prof. Friedman — see
# jason_clarifications.md, Q2.
#
# Movement ONSET is the first frame after the go-signal whose finger speed
# exceeds this threshold (position units per second). The movement END is the
# recorded arrival, NOT a speed threshold, so late sensor jitter can no longer
# extend the window (the old find_movement_window bug).
ONSET_SPEED_THRESHOLD = 5.0
# Require the threshold crossing to persist. A single derivative spike can be a
# tracker artefact and should not define reaction time.
ONSET_SUSTAIN_FRAMES = 3

# The CSV documentation does not state the physical unit of x/y/z. A typical
# reach spans about 13 raw units, which is consistent with centimetres, but we
# do not silently convert or label these values as millimetres without
# confirmation from the experiment owner.
POSITION_UNIT = "tracker_unit"

# ── Trial filtering (outcome-based, from the .mat responseText/timing) ────────
# Arrival more than this many seconds past the object's in-centre window is read
# as disengagement ("gave up / skip to next"), not a real interception attempt,
# and the trial is dropped. There is no clean gap in the lateness distribution;
# this is a judgement call being confirmed with Prof. Friedman (Q3). Single knob.
LATE_ARRIVAL_CUTOFF_S = 1.0
# A completed trial ends at arrival (< ~2 s). A recording near the 10 s cap means
# the finger never intercepted (pressedTime empty) — a timeout — and is dropped.
MAX_TRIAL_DURATION_S = 3.0
# "Too early" = finger left the start box before the go-signal. Invalid by the
# task definition (and breaks the "starts at rest at the go-signal" assumption).
DROP_TOO_EARLY = True
# The "Not fixating on the dot enough!!!" flag is RETAINED: the hand movement is
# normal and the flag is suspected spurious in condition 2 (Q1). Set False to
# drop those trials if Prof. Friedman says the flag is real.
KEEP_NOT_FIXATING = True

# ── Timing plausibility (quality control) ─────────────────────────────────────
# The task requires fast ballistic movements under one second, so a segmented
# window far above that means find_movement_window ran past the interception
# (it extends to the last frame above the velocity threshold, which can pick up
# the return movement). Trials outside these bounds are flagged in the dataset,
# not dropped: whether to exclude them is a decision for the analysis, but with
# a timing head they can no longer be ignored — a 9 s outlier next to a 0.6 s
# median dominates the timing loss.
MAX_MOVEMENT_TIME_S = 1.0
MIN_MOVEMENT_TIME_S = 0.1
MAX_INITIATION_TIME_S = 1.0

# ── Data splits (leave-N-subjects-out) ────────────────────────────────────────
N_TRAIN = 17
N_VAL = 4
N_TEST = 7   # 17 + 4 + 7 = 28 subjects

# ── VAE hyperparameters ──────────────────────────────────────────────────────
LATENT_DIMS_SWEEP = [2, 3, 4, 8, 16]
DEFAULT_LATENT_DIM = 3
HIDDEN_DIM = 256
# Encoder/decoder architecture: "mlp" (default) or "cnn" (1-D convolutional ConvCVAE).
ARCHITECTURE = "mlp"
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 200
KL_WEIGHT = 1.0  # target β for β-VAE; 1.0 = standard VAE (see vae_loss reduction)
EARLY_STOPPING_PATIENCE = 30
# One trial with a 9.7 s segmentation artefact standardises to a z-score of ~25
# and produced a 3e5 loss spike; clipping keeps a single bad trial from wrecking
# the run. 0 disables.
GRAD_CLIP_NORM = 5.0

# ── KL annealing ──────────────────────────────────────────────────────────────
# Applying the full β from step 0 invites posterior collapse: the KL term is
# cheapest to minimise by ignoring z entirely (q(z|x) -> prior), and once the
# decoder has learned to work without the latent it has no gradient pulling it
# back. Annealing β from 0 lets the model first learn to reconstruct, then pays
# the regularisation cost — Bowman et al. (2016).
#   "linear"   : 0 -> KL_WEIGHT over KL_ANNEAL_EPOCHS, constant thereafter
#   "cyclical" : repeated 0 -> KL_WEIGHT ramps (Fu et al. 2019), which keeps
#                re-opening the latent instead of collapsing once
#   "none"     : constant KL_WEIGHT (the previous behaviour)
KL_ANNEAL = "linear"
KL_ANNEAL_EPOCHS = 50   # ramp length (linear) or cycle length (cyclical)
KL_ANNEAL_CYCLES = 4    # cyclical only
# Fraction of each cycle spent ramping; the rest holds at full β so the model
# trains at the true objective before the next cycle.
KL_ANNEAL_RATIO = 0.5

# ── Timing head ───────────────────────────────────────────────────────────────
# Resampling every trial to NORMALISED_LENGTH frames discards how long the
# movement actually took, so the network only ever sees trajectory *shape*.
# The timing channels restore the temporal axis: they are encoded alongside the
# trajectory and reconstructed by a dedicated decoder head, so a sampled latent
# yields a shape *and* the duration to play it back over.
PREDICT_TIMING = True
TIMING_FEATURES = ["movement_time_s", "initiation_time_s"]  # seconds
TIMING_DIM = len(TIMING_FEATURES)
# With a summed reconstruction loss, weight=1 gives each scalar equal weight
# and timing contributes 2 of 302 terms. Weight=150 instead gives the timing
# modality the same aggregate weight as the trajectory, making each timing
# scalar 150x as influential as one coordinate. The corrected study uses 20 as
# a documented compromise and reports timing/shape separately.
TIMING_WEIGHT = 20.0

# Corrected evaluation protocol. A fingerprint is inferred from CONTEXT trials
# only and evaluated against disjoint QUERY trials from the same held-out
# subject. The split is stratified by task condition where possible.
CONTEXT_FRACTION = 0.5
CONTEXT_QUERY_SEED = 2026

# ── Spline baseline ──────────────────────────────────────────────────────────
SPLINE_DEGREE = 3
SPLINE_N_KNOTS = 5

# ── K-Means baseline ─────────────────────────────────────────────────────────
KMEANS_N_CLUSTERS_RANGE = range(5, 35)

# Explicit allowlist of the columns the feature-based baseline clusters on.
# Selecting every numeric column instead would sweep in:
#   rep                   - trial counter (1..180); pure nuisance
#   sp, side              - task labels, which the CVAE *conditions on* precisely
#                           to keep them out of the latent
#   starting_position_mm  - deterministic function of sp, so double-weights it
#   condition             - constant (2) after filtering; contributes nothing
#   index, timing_plausible - bookkeeping added by scripts/make_dataset.py
#
# Measured: those columns carry no subject identity at all (clustering on them
# alone gives ARI = -0.001). Because StandardScaler weights every column
# equally, they acted as 5 of 16 dimensions of pure noise and *diluted* the
# result rather than inflating it - restricting to kinematics raises ARI from
# 0.036 to 0.098 and NMI from 0.149 to 0.288 at seed 42.
#
# Phase 1 asks whether *individual* signatures are recoverable, so the matrix
# holds kinematics only. Note straight_line_dist is the norm of (end_x, end_y,
# end_z): related but not linearly redundant, and kept deliberately.
KMEANS_FEATURE_COLUMNS = [
    "initiation_time_s",
    "movement_time_s",
    "peak_speed_tracker_units_s",
    "time_to_peak_speed",
    "path_length",
    "straight_line_dist",
    "curvature_index",
    "max_lateral_deviation",
    "end_x",
    "end_y",
    "end_z",
]

# ── CSV column names ──────────────────────────────────────────────────────────
CSV_COLUMNS = [
    "frame", "x", "y", "z",
    "rot1", "rot2", "rot3",
    "time", "marker",
]
