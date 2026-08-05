# Preliminary Results (draft)

*Single-condition (free eye movement, condition 2). CVAE with a timing head,
evaluated leave-N-subjects-out on 7 strictly held-out test subjects, aggregated
over 3 seeds unless noted. Figure: `figures/latent_sweep.png`.*

## 1. Data preparation and trial segmentation

We segment each trial by task events rather than by a velocity threshold. Each
recording begins when the target **appears** (`marker = 5`), holds still for a
randomised foreperiod (0.18–0.48 s), and then **starts moving** — the go-signal,
recovered per trial from the target trajectory. The participant may only move
once the target moves, so we take the **go-signal as the behavioural zero-time**
(reaction time is measured from it), which removes the randomised foreperiod from
the trajectories. The window **ends at finger arrival** (interception), read from
the recorded press time. Each movement is low-pass filtered (10 Hz), resampled to
100 frames, and origin-aligned.

From 4,763 condition-2 trials we retain **4,684 (98.3 %)**. We drop **48**
"too early" trials (finger moved before the go-signal), **27** trials that arrive
> 1 s after the target's in-centre window (disengagement), and **4** timeouts
(the finger never intercepts). Trials carrying the eye-fixation flag are retained
(see §4). The event-based end also removes a segmentation artefact in which a late
sensor-jitter blip extended a 0.5 s reach to as long as **9.7 s**; after the fix
the maximum movement time is **1.36 s** and only **0.3 %** of trials fall outside
plausible movement-timing bounds.

## 2. Latent-dimension sweep

We sweep the latent dimension n ∈ {2, 3, 4, 8, 16} (Figure, mean ± sd over 3
seeds).

**Reconstruction** improves monotonically with capacity, from MSE 0.43 (n=2) to
0.06 (n=16). A capacity-matched classical baseline (cubic-spline coefficients
reduced to the same number of dimensions by PCA) reconstructs the raw geometry
somewhat better than the CVAE at low n (e.g. 0.20 vs 0.28 at n=3) — as expected,
since a spline is an interpolation optimised for point-wise fit. The CVAE trades
that point-wise accuracy for an organised, continuous latent space, which is what
the behavioural analysis relies on.

**Timing reconstruction is strong and saturates.** Predicting movement time and
reaction time for unseen subjects reaches R² ≈ 0.89 / 0.93 at n=3 and plateaus at
**R² ≈ 0.99** by **n=8** — a clear elbow. The timing head therefore recovers the
temporal axis that resampling removes, addressing the concern that normalising to
a fixed length discards time and velocity.

**Latent interpretability.** At n=3, 20 of 21 latent-dimension × kinematic-feature
Spearman correlations are significant; individual latent axes align with reaction
time, movement time, and peak speed, so the compact code is interpretable rather
than opaque.

## 3. Subject fingerprints and behavioural probing

A subject **fingerprint** is the aggregate of that subject's trial latent codes
(mean and spread per dimension). Fitting simple probes from the fingerprint to a
subject's macro-level behaviour, scored leave-one-subject-out, the number of
behavioural features predicted above chance rises with capacity, from ~2/11 at
n=3 to **~5/11 at n=16**. Per-feature R² remains **noisy and weak**, and subject
separation stays well below a clean-fingerprint regime at every n.

## 4. Choice of latent dimension

We report the full sweep and take **n = 3 as the headline fingerprint**: it is the
smallest dimensionality that is directly visualizable, already captures timing at
R² ≈ 0.9, and yields an interpretable latent space. **n = 8** is the accuracy
elbow (reconstruction and timing saturate), and **n = 16** captures the most
behavioural structure at the cost of interpretability.

## 5. Interpretation and limitation

The model is an excellent **kinematic / timing** model (timing R² ≈ 0.99,
interpretable low-dimensional code) but does not produce a **sharply separable
individual fingerprint**. This weakness is **structural to the data, not the
architecture**: it persists across a 3.5×-range of latent capacity and is not
relieved by more parameters, and an earlier independent comparison found a
1,700×-larger CVAE and a small PCA model plateauing at the same separation. With
28 subjects and high within-subject trial-to-trial variability — larger than the
between-subject differences — the individual signature is only weakly recoverable.
We therefore present the fingerprint as a **validated data-limited finding** and
lead with the kinematic and timing results, which are strong.
