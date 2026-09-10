# Multi-model dashboard and generation draft

10 September 2026. Revision 3 extends the working manuscript to thirteen pages.
It is still a section-review draft before condensation to the final main paper.

## Dashboard

- CVAE and spline + PCA: n=2,3,4,8. CAE and unconditional VAE: n=3,8.
- Default VAE n=8: lowest cohort-average KS, energy discrepancy and MMD squared
  in the tested matrix. This uses the evaluated cohort; there is no separate
  post-selection validation. It does not imply uniformly significant superiority.
- Reconstruction has different winners: spline + PCA overall at n=3/8, VAE
  among neural models at n=3 and CAE among neural models at n=8.
- VAE ignores condition controls. Spline conditions affect its timing regression,
  while its spatial decoder depends on the latent scores only.
- No n=16 reference is exposed because none was evaluated in the matched study.
- Live references use fold 0, neural seed 42. Validation distributions use all
  four folds at seed 42, with 120 generated samples per participant. Score cards
  use the study's available seeds. These scopes are labelled in the interface.

## Scientific and delivery checks

The export builder reused the frozen neural checkpoints and recreated the fixed
spline training/validation procedure. All 336 participant generation evaluations
matched saved distances; the largest absolute discrepancy was 4.44e-16. No neural
retraining was performed. The full code suite passed 90 tests, and the repository
dashboard passed all 72 model/dimension/section combinations.

The current PDF contains properly typeset equations, the context/query generation
method, all eleven feature definitions, KS/energy/MMD equations, matched-capacity
results and selected trajectory/ECDF figures. Generation starts on page 9.
Example selection is explicit and independent of generation scores.

`VERIFICATION.json` records final PDF, checkpoint and standalone-bundle checks.
The new archive is `../output/release/Interception_Movements_Multimodel_Dashboard.zip`.
It is separate from the frozen September 5 delivery. Rebuild it with
`python scripts/package_multimodel_dashboard.py` from the repository root.

Fingerprint replacement controls, the remaining experiments, abstract,
introduction and discussion still need their agreed section-by-section rewrite.
Red bracketed questions for Jason remain open.
