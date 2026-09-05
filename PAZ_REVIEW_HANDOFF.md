# Start here: Paz's review

Prepared 5 September 2026. This is a review candidate for Jason and Moni, with the
remaining scientific assumptions stated explicitly. It is not a claim of publication readiness.

## Read these first

1. `output/pdf/Interception_Movements_Final_Scientific_Report.pdf`: eight-page scientific report, including references and the important positive and negative results.
2. `output/pdf/Interception_Movements_Supplementary_Appendix.pdf`: complete comparisons, methods, diagnostic examples and numerical evidence.
3. `output/pdf/Interception_Movements_Results_Guide.pdf`: explanations for us; read alongside the report, then inspect the code and ask questions. It cannot replace understanding the analysis.

The ready-to-share bundle is `output/release/Interception_Movements_Advisor_Package.zip`.
Private emails, raw recordings and scratch audits are kept locally and excluded from Git.

## What changed, and what the results support

- The saved model matrix covers four participant folds and all declared seeds: 124 evaluations of five families. Spline reconstructs better; CVAE has advantages on some timing and distribution endpoints. There is no single best model across all endpoints.
- New same-decoder controls use all 24 CVAE n=3/8 fold/seed checkpoints. Correct-person fingerprints improve all three distances over population and wrong-person controls (all 12 BH-adjusted p-values below 0.0004). This supports personal information in this generator, not a discovered cognitive strategy.
- The direct-context baseline has lower MAE than latent-code probes on all 14 behavioural summaries at both dimensions. It stores target-specific measurements rather than an equally compact shared code. This negative comparison is now visible in the main paper.
- Minimum-jerk fitting was repeated on 2,376 recorded query trajectories and 1,680 generated trajectories with the same downstream representation and fitting procedure. Count total variation fell from historical 0.353/0.355 to 0.214/0.232 (n=3/8); this is a changed evaluation, not improved model training. All seven reported discrepancies remain above the empirical reference's 95th percentile, which is descriptive and excludes fitting/training uncertainty.
- All 4,056 selected component fits converged. Some alternative component orders did not. Higher-budget counts agree in 56/56 recorded and 111/112 generated subset cases; convergence does not prove unique biological components.
- Numerical tie handling was corrected in the main paired tests: means and all 56 significance decisions remain unchanged.

## What still needs a scientific decision

Ask Jason about motion/event alignment, the arrival proxy, tracker units, onset threshold,
fixation-warning inclusion and the late-arrival rule. Pre-target-motion movement can be
part of the behaviour of interest: it is retained in raw data, but the current encoder
starts at target motion. If Jason requires the earlier appearance-to-motion period in
the model, that changes the representation and requires rebuilding data and retraining.
No full neural retraining was needed for the completed evaluation corrections.

## Test independently from a fresh clone

Use Python 3.11 or newer; this snapshot was checked on Windows with Python 3.13.
Final delivery checks: 70 tests passed from a clean copy of tracked files after
restoration; all 24 dashboard section/dimension combinations passed there and
in a separate ZIP extraction. These checks used the installed environment in
`review_evidence/environment.json`, not a newly installed virtual environment.
From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python scripts/prepare_review_workspace.py
.\.venv\Scripts\python scripts/verify_review_handoff.py
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python -m streamlit run src/confirmatory_dashboard.py
```

The restore step unpacks the committed review snapshot into ignored artifact folders.
It refuses to overwrite different existing files unless `--replace-artifacts` is explicit.
The unit tests check software behaviour and reference formulas; they do not certify the
scientific interpretation. `verify_review_handoff.py` checks delivered arithmetic and PDF hashes.
The dashboard shows full-matrix tables, but live generation uses only four illustrative
fold0/seed42 checkpoints. It can be tested without Dropbox access.

Full reproduction from recordings is separate: it needs the private raw data, canonical
cache and the other saved checkpoints or model training. The release is not a complete
archive of every trained model. See `scripts/README.md` and
`studies/review_corrected_evaluation/PROTOCOL.md` for analysis entry points.
The paired-test execution used SciPy 1.16.3; installing another version may alter
method='auto' p-values. Recorded CSV outputs preserve the actual environment/settings.

## Where the code flows

Use `README.md` and `scripts/README.md` to locate loading, preprocessing, model training,
evaluation and reporting. The new controls are `scripts/run_review_controls.py` and
`scripts/analyze_review_controls.py`. The current paper is authored in
`reports/concise_article.py`; figures and rebuilding are in `reports/build_final_reports.py`.
Restored evidence is under `studies/review_corrected_evaluation/results/`.
Small review summaries are also browseable in `review_evidence/` on GitHub.

## Suggested message accompanying the handoff

Paz, the updated review package is ready in GitHub. Please start with the eight-page
scientific report and use the appendix for details. We completed the matched
minimum-jerk rerun and fingerprint controls, and added a simple baseline that exposes
a limitation of the latent probes. The paper includes these positive and negative
results and the assumptions to confirm with Jason. The handoff file has commands
for testing the code/dashboard independently. Please review the wording before we
decide together whether to send it to Jason and Moni.
