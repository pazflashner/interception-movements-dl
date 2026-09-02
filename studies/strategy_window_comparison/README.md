# Shared inputs and earlier window comparison

The current study is `../final_strategy_evaluation/`. This directory is retained
because it still supplies canonical data and diagnostics to that study.

## Active dependencies: do not move or delete

- `data/canonical_trials.pkl`, `canonical_trial_metadata.csv`, and
  `canonical_dataset_protocol.json`: shared preprocessing output.
- `data_audit/`: original condition-2 completion audit.
- `results/submovements_real.csv`: real-trajectory minimum-jerk fits.
- `results/go_to_arrival/baselines/kmeans_selection_corrected.csv`: diagnostic
  still referenced by the final report.
- `protocols/ASSUMPTIONS_FOR_JASON.md`: earlier record of task assumptions;
  check current code/report before treating it as current policy.

`config.py` still points to this directory for shared data. Current model runs
and comparative analyses belong under `../final_strategy_evaluation/` instead.
Other model runs, dashboard assets, and outputs here are from the earlier
fixed-split comparison and should not be mixed with final results.

The original README and `CURRENT_RESULTS.md` were preserved unchanged in
`../../archive/pre_confirmatory_2026-09-02/studies/strategy_window_comparison/`.
See `../../REVIEW_HANDOFF.md` for the current review and evidence map.
