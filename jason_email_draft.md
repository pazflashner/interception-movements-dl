DRAFT — for Seman/Paz to review and send. Attach figures/latent_sweep.png.

---

**Subject:** Interception project — preliminary results and a few questions

Hi Jason,

An update on the interception movement modelling, plus a few points we'd value
your input on.

**What we built.** An end-to-end pipeline with a Conditional VAE that maps each
movement into a low-dimensional latent code, alongside K-Means and polynomial-
spline baselines. Following your note about resampling removing time, the model
also predicts **movement time and reaction time** as explicit outputs, so a
generated movement carries a shape *and* its durations.

**How we segment a trial.** We define the movement window by task events rather
than by a speed threshold:

- Start = the moment the **target starts moving** (the go-signal), recovered per
  trial from the target trajectory. We measure reaction time from here, so the
  randomised foreperiod (≈0.18–0.48 s) does not leak into the movements.
- End = **finger arrival / interception** (from the recorded press time).

This also fixed a segmentation artefact where late sensor jitter was stretching a
~0.5 s reach to as much as 9.7 s; the maximum movement time is now 1.36 s.

**Preliminary findings** (evaluated on 7 strictly held-out subjects, 3 seeds; see
attached figure):

- Predicting movement time and reaction time for unseen subjects reaches
  **R² ≈ 0.99** (saturating around latent dimension n = 8).
- Reconstruction improves steadily with n; the latent space is interpretable
  (individual axes track reaction time, movement time, peak speed).
- We take **n = 3** as the headline fingerprint (visualizable, timing R² ≈ 0.9).

**The main limitation — and we'd welcome your view.** The model is a strong
*kinematic/timing* model but does **not** yield a sharply separable individual
"fingerprint". This persists across the whole range of latent sizes and is not
relieved by more capacity — and a larger model and a small PCA model plateau at
the same separation — so it appears **structural to the data** (28 subjects, with
within-subject trial-to-trial variability larger than the between-subject
differences) rather than a modelling failure. We plan to present it that way.

**A few clarifications** (full detail in a shared notes file):

1. **The "Not fixating on the dot enough!!!" flag.** It appears on ~29 % of
   condition-2 (free-eye) trials — more than in condition 1 — and those trials'
   hand movements look completely normal. Was there an eye tracker, and is that
   check meaningful in condition 2? We are currently **keeping** these trials.
2. **Segmentation zero-time.** Do you agree the correct reference is the target's
   motion onset (the go-signal), and that predicting movement + reaction time
   addresses the resampling concern?
3. **Late-arrival cutoff.** We currently drop trials that arrive **> 1 s** after
   the target's in-centre window as disengagement (there is no clean gap in the
   data). Is 1 s reasonable, or would you set it differently?
4. **"Too early" trials** (finger moved before the go-signal) — we exclude these
   as invalid; please confirm.

Happy to walk through any of this. Thanks!

Best,
Seman & Paz
