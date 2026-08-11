# Final Interception-Movement Study

This is the advisor-facing analysis path. The main experiment is a conditional
variational autoencoder (CVAE); clustering, spline/PCA, and minimum-jerk
decomposition are reference and interpretation layers.

## Scientific design

- Retained data: condition-2 trials passing the event and arrival checks.
- Neural input: movement-onset-to-arrival x-y table-plane trajectory, translated
  to its start and resampled to 100 phase points.
- Task condition: `sp`, starting side, and exact executed target speed from the
  paired MAT `thistrial.dotArray`.
- Neural outputs: x-y trajectory, movement duration, and initiation time.
- Latent sizes: n=2 and n=3 are the main low-dimensional fingerprints; n=8 is a
  capacity comparison.
- Split: 17 training, 4 validation, and 7 held-out test participants.
- Repeats: initialization seeds 42, 43, and 44 on the same participant split.
- Fingerprint: context-trial mean latent only. Generated variation uses one
  shared covariance estimated from training participants, so no extra personal
  parameters are hidden in the generator.

See `FINAL_STUDY_PROTOCOL.md` for the complete assumptions and questions for
Prof. Friedman.

## Reproduce

Run these commands from the repository root:

```powershell
python scripts\augment_stimulus_conditions.py --jobs 8
python scripts\extract_submovements.py --out results\final_study\submovements_real.csv --jobs 8 --restarts 2 --max-nfev 400 --backend process
python scripts\audit_submovement_stability.py --jobs 8
python scripts\run_final_baselines.py --permutations 200 --dims 2 3 8
python scripts\run_final_models.py --dims 2 3 8 --seeds 42 43 44 --epochs 150
python scripts\evaluate_final_fingerprints.py
python scripts\generate_final_samples.py --jobs 8 --samples-per-subject 60
python scripts\create_latent_traversals.py
python scripts\summarize_final_results.py
python reports\build_final_study_pdf.py
```

Every long-running extraction is resumable from saved CSV artifacts.

## Output layout

- `data/final_study/`: augmented retained-trial cache and stimulus-link audit.
- `results/final_study/baselines/`: K-Means and spline/PCA references.
- `results/final_study/core_models/`: checkpoints and repeated CVAE results.
- `results/final_study/fingerprint_evaluation/`: held-out subject and trial probes.
- `results/final_study/generation/`: generated samples and distribution distances.
- `results/final_study/latent_traversals/`: controlled n=2/n=3 decoder traversals.
- `results/final_study/summary/`: machine-readable final summaries.
- `output/final_report/`: final PDF and rendered QA pages.
