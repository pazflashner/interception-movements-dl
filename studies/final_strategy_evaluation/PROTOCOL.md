# Final strategy-window evaluation protocol

## Scope

This study uses only the target-motion-onset to finger-arrival window. The
window preserves waiting and initiation behavior. Movement-onset to arrival is
an earlier execution-only control and is not a second primary dataset.

## Confirmatory CVAE

- Input: 100 phase samples of x lateral and y forward position after the 10 Hz
  low-pass filter, plus task condition.
- Task condition: start/speed category, starting side, and exact executed target
  speed.
- Timing is not supplied to the encoder.
- Output: reconstructed trajectory, movement time, and initiation time.
- Latent dimensions: 2, 3, 4, and 8. Dimension 3 is the pre-specified primary
  low-dimensional model; dimension 8 is the capacity comparator.
- Evaluation: four disjoint outer participant folds. Each fold uses 17 train,
  four validation, and seven test participants. Every participant is tested
  once.
- Optimisation: seeds 42, 43, and 44 within every fixed participant fold.
- Held-out participant context and query trials are disjoint and stratified by
  task condition where possible.
- Raw held-out timing predictions and per-trial reconstruction errors are saved
  for every run. The final analysis separates trial-pooled metrics,
  participant-balanced errors, and prediction of participant means.

## Claim policy

Every final factual claim must point to a reported analysis. Untested causal
explanations are labeled as hypotheses. Explanations that are neither tested
nor required are omitted. Reconstruction, timing prediction, participant
enrollment, and fingerprint-conditioned generation are reported as separate
tasks.
