## Presentation handoff after the Paz meeting

The course deck borrows Paz's conceptual opening and feature overview as well
as her design. Slides 1-7 are Paz's section; slides 8-10 cover full-path
generation, personal-center validation and the dashboard. Enrollment and the
standalone empirical-reference slide are omitted from the talk. The report
retains its empirical-reference comparison. Original cumulative feature curves
are restored in both paper and slides.

Charts show the existing participant-balanced means. Markers compare each model
with spline at the same dimension: `*` passes BH and Holm, `†` passes BH only,
and `ns` passes neither at .05. The personal-center chart instead compares each
control with own context using its original BH family. No error bars or new
confidence intervals are included. The markers use the saved corrected tests.

The build uses the existing artifact-tool pipeline and Microsoft PowerPoint's
Windows file renderer (`render_course_powerpoint.ps1`) to check the final PPTX.
It opens the file without a presentation window and preserves other open decks.
For teaching, paste [LEARNING_CHAT_PROMPT.md](LEARNING_CHAT_PROMPT.md) into a new
Codex chat with this project accessible. The prompt requires code-backed,
step-by-step explanations and honest handling of undocumented fixed choices.

## September 14 course update

Use `Interception_Movements_Course_Presentation.pptx` for the ten-slide talk and
`8_pages_draft.pdf` for the eight-page paper. The course deck uses Paz's navy,
blue and white design with the course narrative and figures. Her styled source
files remain separate and unchanged. The course figures use cm and cm/s;
reconstruction MSE is cm². The earlier density revision has been superseded by restored cumulative
curves; cohort scores and tests are unchanged.
No model retraining, preprocessing change or revision of the frozen 31-page
laboratory PDF is included. Methods explanations remain a separate PDF.

# Current manuscript work

The **31-page laboratory PDF remains frozen; substantive feedback will be addressed after the course presentation**. The current
course delivery, revised September 14 from Paz's drafts, is:

- `8_pages_draft.pdf`: exactly eight pages, including abstract and references.
- `Course_Report_Appendix.pdf`: 14 pages of methods, statistics and supporting results.
- `Interception_Movements_Course_Presentation.pptx`: editable slides with speaker notes.
- `10_min_presentation.pdf`: matching ten-slide PDF, containing only the main talk.
- `Course_Methods_Explanations.pdf`: separate 16-topic preparation and Q&A companion.
- `Course_Presentation_Speaker_Notes.md`: the notes in a readable text format.

Read [COURSE_DELIVERY.md](COURSE_DELIVERY.md) for scope, rebuild commands and sources.
The [September 13 progress review](PAZ_DRAFT_REVIEW_2026-09-13.md) remains historical.

The new [full-trajectory distribution analysis](trajectory_distribution_2026_09_13/README.md)
has CSVs, figures, protocol and checks. Both dimensions (3 and 8) are included in
the course paper and slides. The dashboard already defaults to VAE n=8.

The [training-only feature-redundancy sensitivity](feature_redundancy_2026_09_14/README.md)
retains nine features and preserves VAE/CVAE versus spline, without establishing
a unique VAE advantage. Main-report trajectory overlays and feature curves are
retained in `course_assets/` for portable rebuilds.

**For the extended laboratory investigation, read
`Interception_Movements_Methods_Reconstruction_Review.pdf`.**
This is the frozen detailed laboratory-review draft.
It covers data, model pipelines, reconstruction, generation, personal fingerprint
controls, timing, enrollment, behavioural probes, conditioning, minimum-jerk
analysis, event timing and the dashboard deliverable.

Revision 4 has 31 working pages, 18 figures, 19 tables, 19 typeset equations and
15 red bracketed questions for Jason. The main-paper/appendix selection will be
reviewed with Paz and the advisors. All extended evidence remains in that report.

The separate course version now has an abstract, introduction, focused methods
and results, discussion and references. Confirmed laboratory facts from Jason's September 14 comments are incorporated
in the course version. The frozen laboratory PDF retains the historical questions.

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
