# Start here: Paz's review

Updated 10 September 2026 following the code audit and multi-model dashboard update.

## Current draft

Read `production/Interception_Movements_Methods_Reconstruction_Review.pdf`.
This thirteen-page working draft explains the experimental data, input/output pipeline, neural
models, spline + PCA, reconstruction and generated-feature results with recorded/decoded examples.
It includes task/data figures, equations and red questions for Jason. It will be
condensed later and is not the finished eight-page submission. Generation starts
on page 9; side-by-side trajectories and feature distributions are on pages 12-13.
Review these sections first; timing, fingerprint controls, enrollment/probes, minimum-jerk results and the
complete abstract, introduction and discussion are the next writing stages.

Use `production/FIXES_AND_CLEANUP.md` for the confirmed corrections and
`production/audit_2026_09_08/AUDIT_REPORT.md` for the earlier audit snapshot.
The current meeting context is in `SIMAAN_MONI_REPORT_HANDOFF_2026-09-08.md`.

## What is established

- The audit found no misrouting of conditions in the saved MLP models. This does
  not establish that task conditions are irrelevant to movement.
- Spline + PCA has the lowest mean reconstruction error at n=3 and n=8. Among
  neural models, VAE is best at n=3 and conditional AE at n=8. This is not a
  universal ranking for generation or timing; the draft states correction scope
  and significance limitations explicitly.
- Confirmed fixes cover Holm adjustment, saved model flags in dashboard loading,
  CNN ablations/checkpoint loading and a short-filter boundary. All 90 tests pass,
  including checks of all twelve supported model/capacity references.
  No neural retraining or changes to frozen neural results were made.
- The dashboard supports CVAE and spline + PCA at n=2,3,4,8, and VAE and CAE at
  n=3,8. Its default is VAE n=8 for the lowest cohort-average generation distances.
  This is a numerical preference, not superiority on every endpoint or a uniformly
  significant advantage. VAE ignores conditions; spline uses them for timing only.

Please check whether you can explain Figures 3-4's inputs and outputs and whether
Figures 5-10 support the precise reconstruction and generation claims. Units, event semantics,
onset threshold and the appearance-to-motion window still need domain judgment.
Do not infer a biological strategy solely from a latent code or fitted components.

## Independent code and dashboard check

From a fresh clone, use Python 3.11 or newer (local verification used 3.13):

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python scripts/prepare_review_workspace.py
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python -m streamlit run src/confirmatory_dashboard.py
```

The preparation script restores the newest available multi-model review bundle into
ignored artifact directories. It refuses to overwrite differing existing files
without an explicit replacement option. The multi-model ZIP includes eight
illustrative fold-0/seed-42 neural checkpoints and four spline references,
plus all-fold seed-42 validation exports. It supports a local dashboard review
without Dropbox access. The older CVAE-only ZIP is a fallback if the new one is absent.

The new PDF can be read directly. Rebuilding its figures with
`python production/build_methods_reconstruction.py` needs the locally retained
canonical cache, additional frozen checkpoints and audit tables; the old ZIP alone
is insufficient. Full training reproduction additionally needs private recordings.
The installed local test environment was used. Bundle restoration and standalone
dashboard checks are recorded in production/VERIFICATION.json; dependency installation
in a newly created virtual environment was not tested.

## Historical material

The complete September 5 PDFs and release bundle remain in `output/pdf/` and
`output/release/` as frozen prior versions. `scripts/verify_review_handoff.py`
verifies that earlier bundle's arithmetic and PDF hashes, not the current draft.
The prior handoff and correction notes are preserved in
`archive/review_cleanup_2026-09-08/`. They must not be mistaken for current status.
Current source entry points are in `README.md` and `scripts/README.md`.
