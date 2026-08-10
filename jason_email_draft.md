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

- The model **reconstructs** movement and reaction time at **R² ≈ 0.99**
  (saturating around n = 8). Note: the encoder is *given* the timing, so this is
  efficient compression, not prediction of timing from trajectory shape.
- Trajectory reconstruction improves steadily with n; the latent is interpretable
  (its axes track timing/speed, though partly because timing is fed in).
- We take **n = 3** as the headline fingerprint (visualizable), **n = 8** as the
  accuracy sweet spot.

**Model / feature comparison.** We tested the three natural extensions against
the baseline (all latent sizes, 10 seeds):

- A **1-D convolutional** network (respects the trajectory's time-order):
  **improves timing reconstruction** (movement-time R² up 0.03–0.08; reaches ~0.99
  by n = 4–8), but does **not** reconstruct trajectories better or improve the
  fingerprint. We adopt it as an optional architecture for the kinematic side.
- **Exact target speed** as a condition: **no measurable effect** (it duplicates
  the speed range already encoded).
- **Sub-movement count** (hesitation strategy): **no variant's fingerprint
  predicts it** for unseen subjects.

**Loss-function test (Paz's suggestion).** We also checked whether a different
training loss yields better-separated fingerprints — a β-VAE and a discriminative
(subject-separating) term. Neither helped on held-out subjects: the discriminative
loss raised separation on the *training* subjects but did not transfer (held-out
identification actually dropped) and badly degraded reconstruction and timing. One
useful by-product: by nearest-fingerprint matching, the **baseline already
identifies unseen subjects at ~64 % (vs 14 % chance)** — a stronger reading of the
fingerprint than the regression probing gave.

**The main limitation.** The model is a strong *kinematic / timing* model but does
**not** yet produce a sharply separable individual "fingerprint." No approach we
tried — latent size, architecture (MLP/CNN), conditioning, or loss — lifted it. We
are careful **not** to claim this is proven to be a pure data limitation: the
per-trial objective with post-hoc averaging, timing dominating the latent,
unmodelled trial-order effects, and an **untested hierarchical (subject-level) VAE**
are all candidate causes alongside the 28-subject count. A hierarchical VAE — the
design that directly targets your goal — is the next step we'd propose.

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
