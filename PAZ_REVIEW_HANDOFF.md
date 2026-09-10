# Start here: Paz's review

Updated 10 September 2026 following the code audit and multi-model dashboard update.

## Current draft

Read `production/Interception_Movements_Methods_Reconstruction_Review.pdf`.
This 31-page detailed draft covers the data, model pipelines and all major completed
analyses, with 18 figures, 19 tables and 15 red questions for Jason. It is the
laboratory-review version; selecting the eight-page main paper and appendix is
still a joint decision. The abstract, introduction and discussion remain to be written.

Generation starts on page 9, personal controls on page 14, timing on page 16,
enrollment/probes on page 18, conditioning on page 23, minimum jerk on page 24,
the early-movement audit on page 28 and the dashboard on page 29.
The complete page guide and reproducible builders are in production/README.md.

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
- The detailed continuation checks the existing own/population/other-person controls
  and shows the limitation of behavioural-summary prediction. Direct context summaries
  have lower mean error on all 14 targets for every matched model/capacity; this
  descriptive extension uses saved probe predictions and trains no new model.
- The matched minimum-jerk comparison reduced the earlier mismatch but did not
  eliminate it. VAE n=8 did not demonstrate better component fidelity than CVAE n=8.

Please check the connection between the objective, each experiment and its result.
Use the red questions to resolve domain assumptions with Jason. Units, event semantics,
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
