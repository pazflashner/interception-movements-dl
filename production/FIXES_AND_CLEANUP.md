# Follow-up fixes and cleanup — 9 September 2026

The frozen audit is preserved in `audit_2026_09_08/`; its references to unfixed
defects describe that earlier snapshot. This record describes the follow-up edits.

For the subsequent September 10 multi-model dashboard and generation manuscript
update, read [MULTIMODEL_UPDATE.md](MULTIMODEL_UPDATE.md). The interface status
below describes the September 9 stage.

## Confirmed defects corrected

- Implemented Holm's step-down adjustment and used it in the condition-trajectory
  summary. The n=3 adjusted value is 0.551874, not 1.000; n=8 stays 0.132480.
  The two significance decisions remain unchanged. The old values are backed up.
- The earlier full-report builder now reads these adjusted values from its table
  instead of hardcoding them. The historical September 5 PDFs are not rewritten
  during this section-review pass.
- Dashboard model loading reads saved variational/condition flags. The current
  CVAE interface is unchanged; this does not implement a multi-model selector.
- ConvCVAE accepts the trainer's ablation flags and implements their expected
  conditioning/sampling behavior. Frozen-checkpoint loading also honors the CNN
  architecture. Unsupported input lengths fail explicitly. No CNN results were
  added to the scientific comparison.
- The low-pass filter handles exactly 15 samples without triggering filtfilt's
  padding error. All current retained recordings have at least 112 samples.
- Corrected the spline representation docstring and a preprocessing units comment.
  The final spline runner remains shape-only, with 18 coefficients and a separate
  timing head; the historical timing-inclusive API is retained for old callers.

Focused regression tests cover Holm order/step-down behavior, the filter boundary,
CNN ablation forward/backward/checkpoint behavior and dashboard flag loading.
The complete test suite passed: **76 tests**, in 21.39 seconds. The local log is
`verification_tests.log`. Tests verify software behavior, not scientific validity.

The first four-page PDF was rendered and all pages visually inspected. Twelve
plotted reconstruction errors match saved trial results within 4.02e-8 tracker
units squared. All 96 neural checkpoint hashes and all three historical PDF
hashes remain unchanged. See `VERIFICATION.json` for the verification record.

Revision 2 expands the working manuscript with the task display, recorded examples,
resampling and equations. The cohort inventory confirms that subject41 had 29
available recordings and 28 retained trials, and subject02 had 60 of each. No
participant was newly excluded; missing-session causes are marked for Jason.
The previous manuscript and verification record are archived under
`archive/2026-09-09-review1/`. Current PDF verification is in `VERIFICATION.json`.

## Numerical and scientific choices retained

No neural weights, recorded trajectories, inclusion rules, onset definitions,
condition encodings or primary model comparisons were changed. The audited Ridge
thread sensitivity changes no paired-test decisions; its reproduction settings
and diagnostic remain in the audit. Numerical uncertainty is disclosed rather
than changing a frozen modeling protocol to improve results.

Sampling-seed collisions, early selected checkpoints, the onset/window domain
questions and statistical dependence limits are documented in the audit. These
are not resolved by silently changing the pipeline during manuscript preparation.

## Organization

- Earlier REVIEW_HANDOFF and REVIEW_CORRECTIONS notes moved to
  `../archive/review_cleanup_2026-09-08/`.
- The earlier PAZ handoff is archived there; the root PAZ handoff points to the
  current draft and keeps the fresh-clone preparation commands available.
- Local `review/`, `review_second_pass/`, and `review_fresh_2026_09_05/` folders
  moved intact to the same archive. Their scripts are historical snapshots.
- Root temporary directories and leftover pytest caches moved to that archive's
  ignored `runtime_scratch/` directory.
- Previous production PDF copies moved to `archive/2026-09-05/`.
- Large audit intermediates and rendered QA images are excluded from Git; compact
  audit evidence, manuscript code and the review PDF remain visible for versioning.
- Canonical data, trained-study folders, current tests, and shared historical
  helpers remain in place because current code still reads them.

No raw research evidence was deleted. Existing release archives remain frozen;
they should not be presented as the newly revised submission.
