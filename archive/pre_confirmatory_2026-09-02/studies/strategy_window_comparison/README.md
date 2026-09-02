# Strategy-window comparison study

## Scientific question

Does a conditional VAE recover a more useful participant fingerprint when its
trajectory contains the entire strategy interval from target motion onset to
arrival, rather than only the detected finger movement?

## Fair comparison

The two protocols differ only in temporal window:

| Protocol | Resampled trajectory | What it can represent |
|---|---|---|
| `movement_only` | finger movement onset -> arrival | execution shape and corrections |
| `go_to_arrival` | target motion onset -> arrival | waiting, initiation, execution, corrections |

Shared controls:

- condition 2 only;
- completed arrivals only;
- `Too early` arrival-feedback trials retained;
- arrivals more than 1 s after the target window currently excluded and flagged
  as a Prof. Friedman validation question;
- x-y table plane, fourth-order 10 Hz Butterworth filter;
- 100 phase samples plus separately decoded physical time;
- task condition: `sp`, side, and exact target speed;
- 17/4/7 participant train/validation/test split;
- latent widths 2, 3, 4, and 8; seeds 42, 43, and 44.

Latent widths 2 and 3 are the primary interpretable fingerprints. Width 4 is a
transition point; width 8 is a capacity comparator, not claimed to be directly
interpretable. Width 16 is omitted because it adds capacity without serving the
stated low-dimensional research goal and previously showed stronger overfit.

## Trial-completion finding

The raw audit covers all 4,763 condition-2 CSV/MAT pairs from 28 participants:

- no recording ended before the target began moving;
- all 48 `Too early` labels have a recorded arrival after target motion onset;
- four trials have no arrival and run to the approximately 10 s timeout;
- therefore MAT arrival (`pressedTime`) is the primary collision-completion
  criterion, while `Too early` is retained as valid arrival-timing information.

See `data_audit/TRIAL_COMPLETION_AUDIT.md` for exact counts and distributions.
