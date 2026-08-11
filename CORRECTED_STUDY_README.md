# Corrected Study (v3)

This is the authoritative result set after the methods audit. Legacy results remain under `results/` and in the old preliminary PDF, but must not be mixed with v3.

## Protocol

- 4684 retained condition-2 trials, 28 subjects.
- Fixed 17/4/7 subject split.
- Context/query split within every subject; query trials are not used to infer the fingerprint.
- Frame-counter regularization, 10 Hz filtering, movement-onset to arrival trajectory, log timing outputs.
- Primary low-dimensional fingerprint = subject context mean in n dimensions.

## Main findings

- Fixed-k K-Means: trajectory ARI 0.093; feature ARI 0.098. Both exceed the 200-permutation null, but clustering is weak.
- Trajectory-only n=8 movement-time prediction: R2 0.729. Initiation-time R2 0.128.
- Best trajectory-only distribution model: n=16, mean KS 0.197, mean FDR-rejected features 4.86/12.
- n=2/3 subject fingerprints do not generalize to subject timing or curvature distributions.
- Hierarchical subject/trial CVAE did not improve generative fidelity.

## Run

```powershell
python scripts/run_corrected_study.py --epochs 150 --dims 2 3 4 8 16 --hier-dims 2 3 4 --timing-weight 20 --out results\corrected_v3
python reports/build_corrected_pdf.py
```
