DRAFT — review with Paz first, then send to Moni & Prof. Friedman.
Attach: Interception_Preliminary_Results.pdf

---

**Subject:** Interception project — preliminary results, model comparison, and a few questions

Hi Jason,

An update on the interception-movement modelling. Full detail (with figures) is in
the attached PDF; the short version is below.

**What we built.** An end-to-end pipeline with a Conditional VAE that maps each
movement into a low-dimensional latent code, plus K-Means and polynomial-spline
baselines. Following your note about resampling removing time, the model also
predicts **movement time and reaction time** as explicit outputs.

**How we segment a trial.** By task events, not a speed threshold: the window
starts at the **target's motion onset** (the go-signal, so reaction time is
measured from it and the randomised foreperiod is removed) and ends at **finger
arrival**. This also fixed an artefact where late sensor jitter was stretching a
~0.5 s reach to as much as 9.7 s.

**Key findings** (7 held-out subjects, 10 seeds):

- Predicting movement and reaction time for unseen subjects reaches
  **R² ≈ 0.99** (saturating around latent size n = 8).
- Reconstruction improves steadily with n; the latent space is interpretable
  (its axes track reaction time, movement time, peak speed).
- We take **n = 3** as the headline fingerprint (visualizable, timing R² ≈ 0.9),
  with **n = 8** as the accuracy sweet spot.

**Model / feature comparison.** We tested the three natural extensions against
the baseline (all latent sizes, 10 seeds):

- A **1-D convolutional** network (respects the trajectory's time-order):
  **improves timing prediction** (movement-time R² up 0.03–0.08; reaches ~0.99 by
  n = 4–8), but does **not** reconstruct better or improve the fingerprint. We
  adopt it as an optional architecture for the kinematic side.
- **Exact target speed** as a condition: **no measurable effect** (it duplicates
  the speed range already encoded).
- **Sub-movement count** (hesitation strategy): **no variant's fingerprint
  predicts it** for unseen subjects.

**The main limitation.** The model is a strong *kinematic / timing* model but does
**not** produce a sharply separable individual "fingerprint." This holds across
every latent size *and* every architecture/feature above, and a much larger model
and a small PCA model plateau at the same separation — so it appears **structural
to the data** (28 subjects, with within-subject variability larger than the
between-subject differences), not a modelling failure. More subjects is the one
lever likely to help. We plan to present it this way.

**A few clarifications we'd value your input on:**

1. **The "not fixating" flag** appears on ~29 % of condition-2 (free-eye) trials —
   more than in condition 1 — yet those trials' hand movements look normal. Was
   there an eye tracker, and is that check meaningful in condition 2? We are
   currently keeping these trials.
2. **Segmentation zero-time:** do you agree the correct reference is the target's
   motion onset, and that predicting movement + reaction time addresses the
   resampling concern?
3. **Late-arrival cutoff:** we drop trials arriving > 1 s after the target's
   in-centre window as disengagement (no clean gap in the data). Is 1 s reasonable?
4. **"Too early" trials** (moved before the go-signal) — we exclude these as
   invalid; please confirm.
5. **Sub-movement benchmark:** could we get access to your sub-movement
   decomposition pipeline, to benchmark generative fidelity against it?

Happy to walk through any of it. Thanks!

Best,
Seman &amp; Paz
