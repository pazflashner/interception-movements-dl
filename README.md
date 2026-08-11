# Interception Movement Strategy Fingerprints

This repository compares two conditional-VAE representations of fast human
interception movements:

1. **Execution-only:** detected finger movement onset to recorded arrival.
2. **Strategy-inclusive:** target motion onset (the go signal) to recorded
   arrival, preserving each participant's waiting period and early corrections.

Both paths use the same condition-2 trials, x-y table-plane coordinates, 10 Hz
low-pass filtering, participant-held-out evaluation, and task conditioning. The
comparison asks whether retaining pre-movement waiting improves low-dimensional
fingerprints and the reproduction of behavioral distributions.

## Navigation

- `studies/strategy_window_comparison/`: current protocol, audit, results, and
  advisor-facing outputs.
- `src/`: shared loading, preprocessing, CVAE, evaluation, and submovement code.
- `scripts/`: reproducible data, training, evaluation, and reporting commands.
- `tests/`: protocol and model checks.
- `archive/movement_only_2026-08-10/`: local, ignored, explicitly outdated
  snapshot kept only for context.

Raw data remains read-only at `D:\DropBox\Dropbox\results` and stimuli at
`D:\DropBox\Dropbox\stimuli`. The project never writes to Dropbox.
