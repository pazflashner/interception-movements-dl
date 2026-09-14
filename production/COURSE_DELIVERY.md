September 14 editorial update: the course presentation now follows Paz's navy
cover, blue/teal/violet rule and white content-slide design, retaining our ten
main slides. Confirmed cm units, response-time terminology, data-collection
credit and spline/timing explanations were incorporated. The main feature
figure uses shared-bin density histograms from the same saved samples. No
training, preprocessing, inference protocol or historical score table changed.
The 31-page laboratory PDF and Paz's styled source deck remain unchanged.

# Course report and presentation, 14 September 2026

Start with **8_pages_draft.pdf**. Its eight pages cover motivation, data,
preprocessing, model pipelines, context-based generation, evaluation,
reconstruction, eleven features, the new full-path result, personal-center
controls, dashboard and discussion. **Course_Report_Appendix.pdf** supplies the
matrix dimensions, loss, sampling covariance, feature definitions, statistical
estimators, corrected comparisons, secondary findings and red questions for Jason.

**Interception_Movements_Course_Presentation.pptx** is editable. Tables, text and five charts
are native PowerPoint objects. Each chart includes an editable data workbook;
chart display data are rounded to six decimals while research CSVs retain full
precision. Scientific trajectories and feature curves are measured-data images. Speaker notes explain the code and inference in plain language.
**10_min_presentation.pdf** contains the exact rendered slide views.

Slides 1–10 have a planned nine-minute duration, including an 80-second dashboard
demonstration. The remaining minute allows transitions. There are no backup slides in the PPTX.
**Course_Methods_Explanations.pdf** holds all 15 former backup topics plus the
feature-redundancy explanation, separately from the ten-slide presentation.
**Course_Presentation_Speaker_Notes.md** contains the same notes without requiring
PowerPoint. Practice is still needed to check the actual timing.

## Scientific interpretation

- VAE n=8 is the practical default, with the lowest mean feature discrepancies.
  CVAE remains statistically comparable on the new full-path endpoints.
- Spline + PCA has the lowest mean reconstruction MSE. All six comparisons
  against neural models pass BH-140; spline versus CAE n=8 does not pass Holm.
- VAE/CVAE n=8 beat spline on full-path energy and MMD under BH and Holm across
  all 80 comparisons, including the axis-balanced sensitivity. At n=3, the
  conclusion depends on metric and scaling.
- Resampling stored context paths gives lower mean path discrepancies, without
  a compact latent representation. Those reference differences are descriptive.
- Personal-center replacement supports useful same-session information. Failed
  behavioral-summary probes and the stronger empirical reference remain visible.

The 31-page laboratory PDF is unchanged (SHA-256
`a1461d66e8034810342be4b89d9ed8e759e79db31c9f2a65d385f51c54b3ba2a`).
No training code, model checkpoints or historical analysis score tables changed.
A new training-only feature-redundancy sensitivity is saved separately. Paz's prior drafts remain in Git at d03beb2, and a local copy is
in the ignored `production/archive/course_before_2026_09_14/` directory.

## Rebuild

From the repository root:

```powershell
python production/build_submission_report.py
python production/build_presentation.py
```

The first command builds the paper and appendix using Python, reportlab,
matplotlib, pandas, numpy, Pillow and pypdf. Scientific figures are retained in
`course_assets/`, so a fresh checkout does not need the private decoded arrays
merely to rebuild the paper. To regenerate those path figures from actual
decoded arrays, run `python -m production.course_figures` for the original
figures or `python -m production.course_visuals` for the visual revision. The
latter requires original trials, frozen checkpoints, and both local generation
caches under `assets/trajectory_distribution_2026_09_13/` and
`assets/feature_redundancy_2026_09_14/`.

The presentation builder uses the Codex bundled Node artifact-tool runtime and
its validators, plus Python for the slide PDF. Defaults locate the installed
runtime under the user's profile. Set `COURSE_RUNTIME_ROOT` and
`COURSE_SLIDE_SKILL` for a different installation. Without this runtime, open
and edit the delivered PPTX in PowerPoint. Do not substitute the older slide
builder, which describes the previous report scope.

Authoring files:

- `course_main_visual.py`: eight-page main report prose.
- `course_manuscript.py`: detailed appendix A1-A10.
- `course_visual_appendix.py`: redundancy analysis and additional visual examples.
- `course_evidence.py`: authoritative CSV reads and matched comparison lookup.
- `course_figures.py`: original scientific figures from saved evidence.
- `course_visuals.py`: reconstruction overlays, generated paths and feature ECDFs.
- `course_slides.py`: base text, sourced tables, notes and timing.
- `course_visual_slides.py`: visual main talk and separate explanation PDF.
- `course_explanations.json`: detailed Q&A excluded from the actual deck.
- `course_slides.json`: generated portable slide content.
- `build_course_report.py`: paper layout and eight-page assertion.
- `build_course_slides.py` / `build_course_presentation.mjs`: PPTX and PDF export.
- `verify_course_delivery.py`: checks headline claims, native table values,
  notes, page counts, all 80 trajectory tests and sample decoded-array metrics.
- `verify_feature_redundancy.py`: independent checks of the 72-test sensitivity,
  all training correlations and selected decoded-feature metrics.
  Its full audit requires local checkpoints and decoded arrays.

`COURSE_DELIVERY_VERIFICATION.json` records the final numerical/artifact checks.
`COURSE_REPORT_VERIFICATION.json` records report pagination. Rendered QA files,
runtime links and presentation validator receipts stay in ignored `assets/`.

## Evidence sources

Original model means and 140 corrected tests:
`audit_2026_09_08/all_model_means.csv`, `all_pairs_140.csv`.
New complete-path scores, 80 tests, protocol and original independent verifier:
`trajectory_distribution_2026_09_13/`.
Personal-center controls: `audit_2026_09_08/uvae8_fingerprint_*.csv` and
`../studies/review_corrected_evaluation/results/review_controls/fingerprint_*.csv`.

The equations describe this code's implementation. Method references include
[Kingma and Welling](https://arxiv.org/abs/1312.6114),
[Sohn, Lee and Yan](https://papers.nips.cc/paper_files/paper/2015/hash/8d55a249e6baa5c06772297520da2051-Abstract.html),
[Gretton et al.](https://www.jmlr.org/papers/v13/gretton12a.html),
Szekely and Rizzo (2013), and Benjamini and Hochberg (1995).
These references support the general methods; they do not establish the validity
of our chosen personal-center heuristic or our experimental findings.

## Visual revision

The main paper remains eight pages with five figures. Detailed equations remain
in the 14-page appendix, alongside all eleven feature curves at n=3 and n=8.
The main methods retain the N-by-18 spline coefficient matrix, inverse decoding,
neural objectives, context generation and participant-level statistical logic.
Laboratory implementation details, including MATLAB event offsets and code
paths, are excluded from the main submission narrative. The frozen laboratory
report has not changed.

The nine-feature sensitivity preserves VAE/CVAE over spline but no reduced
metric significantly favors VAE over CVAE at n=8. See
[the complete sensitivity](feature_redundancy_2026_09_14/README.md).
