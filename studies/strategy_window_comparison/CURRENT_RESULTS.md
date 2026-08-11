# Current strategy-window study

## Release artifacts

- PDF: `output/pdf/Interception_Strategy_Window_Comparison.pdf`
- Interactive dashboard bundle: `output/share/Interception_Strategy_Dashboard.zip`
- Email draft: `output/share/EMAIL_DRAFT.txt`
- Aggregate model table: `results/summary/model_comparison_by_seed.csv`
- Latent association table: `results/latent_associations/latent_submovement_associations.csv`

Generated outputs are ignored by Git because they are reproducible from the
tracked code and the local data. The study logic and report/dashboard builders
are tracked.

## Canonical cohort

- 4,763 condition-2 CSV/MAT pairs were audited across 28 participants.
- No trial ended before target motion began.
- All 48 `Too early` trials reached the target after it began moving and are
  retained.
- The final cohort has 4,732 trials: 27 arrivals more than one second after the
  target window and four no-arrival timeouts are excluded.

## Fair comparison

The same trials, x-y table plane, 10 Hz filter, task conditions, participant
split, dimensions, and seeds are used for both windows:

1. `movement_only`: detected finger movement onset to arrival.
2. `go_to_arrival`: target motion onset to arrival, including the waiting
   interval.

All trajectories are resampled to 100 phase points. Physical movement and
initiation time are withheld from the encoder and decoded separately.

## Main findings

- K-Means is above the permutation null but weak in absolute terms (trajectory
  ARI about 0.05-0.06). Subject information exists without clean natural
  28-cluster separation.
- `n=2` is too restrictive and gives unstable initiation-time prediction.
- `n=3` is the smallest stable strategy-inclusive model: held-out initiation
  R2 is 0.45 and movement-time R2 is 0.65 across three seeds.
- `n=8` is the strongest capacity model, not an eight-variable interpretable
  fingerprint.
- The strategy window is better for initiation timing and generally for the
  basic generated-distribution KS statistic.
- Movement-only is better for execution reconstruction, participant enrollment,
  and generated minimum-jerk component-count fidelity.
- The best held-out participant-enrollment result is movement-only `n=8` at
  57.4% balanced accuracy versus 14.3% chance.
- For the seed-42 `n=3` strategy model, trial latents have strong partial
  associations with initiation and movement time after controlling for
  participant and task conditions. These are model-specific associations, not
  causal or universal axis meanings.

## Scientific interpretation

The two representations are complementary. The strategy window is required if
waiting and movement initiation are part of the target behavior. The
movement-only view remains useful for detailed execution and correction
structure. The current evidence supports a low-dimensional generative model,
but it does not establish that two universal numbers reproduce every person's
full behavioral distribution.
