# Current manuscript work

**Read `Interception_Movements_Methods_Reconstruction_Review.pdf` first.**
This is a section-review draft, not the final submission: data and evaluation
design, detailed neural and spline pipelines, reconstruction comparisons, and
recorded-versus-reconstructed figures.

Revision 3 expands these sections to thirteen working pages, before final
condensation. It includes the original task display, one participant's 180 trials,
a median-duration example, resampling, model equations and red bracketed questions
for Jason. It is not the complete eight-page main paper.

Generation begins on page 9, with all four model families compared numerically
and visually. Review these sections before expanding the paper to timing,
fingerprint controls, enrollment/probes and minimum-jerk analysis. The complete
abstract, introduction and discussion will follow the assembled evidence. The
final target remains an eight-page main paper plus a separate appendix.

## Files

- `build_methods_reconstruction.py`: reproducible PDF/figure builder; requires
  the locally retained canonical cache, frozen checkpoints, audit tables, raw
  recording filenames for the cohort inventory and `instructions2.jpg` from the
  stimulus directory. Raw/stimulus roots are configured in `config.py`.
- `manuscript_sections.py`: narrative text, equations, captions and red questions.
- `generation_section.py`: context/query generation, feature definitions, distances and figures.
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
```

The dashboard supports CVAE and spline + PCA at n=2,3,4,8, and VAE and CAE at
n=3,8. VAE n=8 is the default because its cohort-average generation distances
are lowest in the tested matrix. This is an endpoint-specific numerical choice.
VAE condition controls are inactive; spline conditions affect timing only.
Domain questions in the meeting handoff remain open.
