# Current manuscript work

**Read `Interception_Movements_Methods_Reconstruction_Review.pdf` first.**
This is the detailed laboratory-review draft, not the condensed submission.
It covers data, model pipelines, reconstruction, generation, personal fingerprint
controls, timing, enrollment, behavioural probes, conditioning, minimum-jerk
analysis, event timing and the dashboard deliverable.

Revision 4 has 31 working pages, 18 figures, 19 tables, 19 typeset equations and
15 red bracketed questions for Jason. The main-paper/appendix selection will be
agreed with Paz and the advisors. No results have been discarded to fit a page limit.

The abstract, introduction and discussion will follow the agreed scientific story.
The final target remains an eight-page main paper plus supplementary material.

## Page guide

| Pages | Content |
|---|---|
| 1-8 | Data, preprocessing, models and reconstruction |
| 9-13 | Generation and eleven-feature distributions |
| 14-15 | Same-decoder personal fingerprint controls |
| 16-17 | Timing prediction, tail errors and head sensitivities |
| 18-22 | Enrollment, clustering, all behavioural probes and direct-context comparisons |
| 23 | Condition routing, shuffling and stratum fidelity |
| 24-27 | Matched minimum-jerk evaluation and assumption sensitivity |
| 28 | Event audit and early movement |
| 29-30 | Dashboard, remaining capacities and evidence boundaries |
| 31 | References |

## Files

- `build_methods_reconstruction.py`: reproducible PDF/figure builder; requires
  the locally retained canonical cache, frozen checkpoints, audit tables, raw
  recording filenames for the cohort inventory and `instructions2.jpg` from the
  stimulus directory. Raw/stimulus roots are configured in `config.py`.
- `manuscript_sections.py`: narrative text, equations, captions and red questions.
- `generation_section.py`: context/query generation, feature definitions, distances and figures.
- `remaining_results.py`: the detailed continuation narrative and tables.
- `remaining_figures.py`: plots from frozen auxiliary results, including three-dimensional codes.
- `remaining_evidence.py`: source-table loading and descriptive all-model direct-context comparison.
- `verify_detailed_report.py`: numerical checks, unchanged checkpoint checks and final PDF inventory.
- `DETAILED_REVIEW_VERIFICATION.json`: verification results and a page map extracted from the PDF.
- `remaining_results_provenance.json`: hashes of source tables and the implementation reviewed.
- `all_models_direct_context_summary.csv`: saved-probe comparisons with direct context measurements.
- `report_equations.py`: properly typeset mathematical equations.
- `MULTIMODEL_UPDATE.md`: current dashboard scope, default selection and verification.
- `cohort_trial_counts.csv`: available, retained and excluded counts for all 28 participants.
- `data_figure_provenance.json`: data-example selection and original task-image source.
- `FIXES_AND_CLEANUP.md`: source changes, verification and repository organization.
- `VERIFICATION.json`: test count, unchanged checkpoint hashes and PDF verification.
- `reconstruction_example_manifest.csv`: selected trials and checked reconstruction errors.
- `audit_2026_09_08/AUDIT_REPORT.md`: frozen-model audit and additional comparisons.
  This is the audit snapshot before the follow-up fixes; read the fix record too.
- `audit_2026_09_08/all_pairs_140.csv`: exploratory, matched-capacity comparisons.
- `assets/`: regenerated figures, example-selection manifest and rendered QA,
  excluded from Git. Selected examples are deterministic, not chosen by appearance.
- `archive/2026-09-05/`: local copies of the earlier complete PDFs. The tracked
  September 5 delivery remains in `../output/pdf/` and `../output/release/`.
- `archive/2026-09-09-review1/`: the previous four-page manuscript and builder.

Build from the repository root:

```powershell
python production/build_methods_reconstruction.py
python production/verify_detailed_report.py
```

The dashboard supports CVAE and spline + PCA at n=2,3,4,8, and VAE and CAE at
n=3,8. VAE n=8 is the default because its cohort-average generation distances
are lowest in the tested matrix. This is an endpoint-specific numerical choice.
VAE condition controls are inactive; spline conditions affect timing only.
Domain questions in the meeting handoff remain open.
