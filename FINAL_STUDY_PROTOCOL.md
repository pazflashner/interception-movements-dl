# Final Study Protocol

## Research question

Can a conditional variational autoencoder learn a low-dimensional subject
fingerprint from interception trajectories and use that fingerprint to
reproduce held-out distributions of movement shape, timing, and interpretable
minimum-jerk submovement features?

## Primary deep-learning model

- Input: movement-onset-to-arrival x-y table-plane trajectory, spatially
  translated and resampled to 100 phase points, plus the task condition (`sp`
  and side), plus the exact target speed derived from the matched 60 Hz
  stimulus trajectory. The z coordinate is retained in the audit data but excluded from
  the final baseline and neural-model inputs.
- Encoder: trajectory and condition only. True initiation time and movement
  duration are withheld.
- Latent dimensions: `n=2` and `n=3` are the primary low-dimensional models;
  `n=8` is a pre-specified capacity comparator.
- Decoder: trajectory, log movement duration, and log initiation time.
- Split: 17 train, 4 validation, and 7 test subjects. No subject appears in
  more than one split.
- Repeats: three initialization seeds on the same pre-specified subject split.

Temporal resampling removes duration but not trajectory shape. Duration and
initiation time are therefore generated as separate outputs. This prevents a
long stationary waiting period from dominating trajectory reconstruction while
retaining the timing information needed to study movement strategy.

## Subject fingerprint protocol

For every held-out subject, trials are stratified by `sp` and side and divided
into disjoint context and query halves. The subject fingerprint is the mean
latent posterior location over context trials. Query trajectories and outcomes
are never used to infer the fingerprint.

Generated within-subject variation uses one latent-noise covariance estimated
from residual trial codes of training subjects. The covariance is fixed for
every participant, so the only participant-specific controls are the `n` values
in the context-mean fingerprint; no hidden subject-specific variance parameters
are introduced.

The fingerprint is evaluated by:

1. closed-set identification of query trials after context enrollment;
2. prediction of query-only subject distribution summaries;
3. generation of new trials under the query condition mixture;
4. comparison of generated and empirical query distributions.

## Minimum-jerk decomposition

The behavioral interpretation layer follows Prof. Jason Friedman's
`submovements` repository at commit
`9c2f40ccc922d542242329c46cfd524c21188b4a`.

- Primary plane: x-y table plane. The forward y axis dominates; lateral x is
  retained after 10 Hz filtering to detect meaningful corrections.
- Candidate count: 1-4 minimum-jerk components.
- Parameters per component: onset, duration, lateral displacement, and forward
  displacement.
- Minimum component duration: 100 ms.
- Minimum onset spacing: 50 ms. Components may overlap.
- Primary order rule: choose the smallest count with normalized error <= 0.05;
  if none, use the smallest count below 0.10; otherwise use the minimum-error
  count.
- Full extraction: two deterministic optimizer restarts. A stratified subset is
  refitted with eight restarts as a stability audit.

The decomposition produces kinematic patterns, not direct observations of a
cognitive strategy. Final language is therefore limited to single-component,
overlapping-component, and sequential-component patterns.

## Distribution metrics

- Continuous variables: KS statistic and Wasserstein distance.
- Component count: total-variation distance and Jensen-Shannon divergence.
- Recorded success: rate error, Brier score, and calibration/AUC where
  applicable.
- Multivariate behavior: energy distance and MMD.
- Subject-level uncertainty: model-seed spread, with all final distribution
  metrics shown over the seven held-out participants.

## Assumptions to validate with Prof. Friedman

1. Confirm the 100 ms minimum duration and 50 ms onset-spacing adaptation for
   these short interception reaches; the repository default is 167 ms for both.
2. Confirm the smallest-adequate-error component-order rule and 0.05/0.10
   thresholds.
3. Confirm that x-y is the intended task plane and that z should be treated as
   off-plane noise.
4. Confirm whether the condition-2 `Not fixating on the dot enough!!!` outcome
   should be interpreted as failure or as a task-irrelevant eye-tracking flag.
5. Confirm the physical unit of tracker x/y/z coordinates.
6. Confirm the retained late-arrival cutoff and the absence of condition-2 data
   for the twenty-ninth participant.
7. Confirm that the executed `thistrial.dotArray` should take precedence when
   an external stimulus CSV with the same recorded filename differs in timing
   or speed, and confirm the 1920-pixel screen-width normalization used for the
   continuous target-speed condition.

Every assumption is centralized so a requested change can be followed by a
reproducible rerun rather than manual edits to reported numbers.
