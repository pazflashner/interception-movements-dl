# Review of Paz's course drafts against the agreed plan

Initially reviewed commit `3e33443`, then incorporated Paz's concurrent presentation
revision `ac8374b`, without editing either PDF or its builder. This is a
focused progress review, not a completed sentence-by-sentence scientific audit.
The 31-page laboratory report remains frozen pending Jason's feedback.

## Assessment

The nine-slide presentation largely follows the agreed sequence: question,
models/protocol, reconstruction, feature evaluation, generation, personal
fingerprint control, limitations and dashboard. This is a useful presentation
base. Its short explanations and selected comparisons are more focused than the
paper. Preserve that sequence, then correct scientific wording and projection layout.

Paz's subsequent `ac8374b` revision improves the talk: reconstruction now shows
the cohort comparison, the fingerprint slide focuses on VAE n=8, and the summary
has shorter bullets and a practical recommendation. The previous summary-column
overlap is resolved. The cropped fingerprint figure, however, loses its numerical
y-axis labels and the explanation of what a positive difference measures; restore
a concise metric/direction label. The new recommendation line extends past its
intended right margin and should be wrapped. The scientific corrections below
still apply. The eight-page PDF did not change in that commit.

The eight-page paper satisfies the page count, but keeps 17 numbered sections
and most of the laboratory draft's secondary analyses. Our agreed plan was to
select a smaller main story and move supporting investigations to an appendix.
It currently has no abstract and no cohesive discussion section. Its extensive
timing, enrollment, probes, clustering, conditioning and minimum-jerk sections
leave little room to explain the central data/model/generation pipeline.

Keep in the main paper: motivation, data/task image, preprocessing and participant
splits, shared model pipeline with the spline comparison explained, reconstruction,
generation, own-fingerprint control, dashboard and a discussion with limitations.
The newly requested whole-trajectory distribution analysis is a candidate for
the generation subsection once reviewed; it is separate exploratory evidence.
Move detailed secondary analyses to an appendix, retaining brief statements of
material limitations in the discussion. Completed negative findings are results,
not future work. Interpretability and cross-session validation are future work.

## Corrections needed before sharing as a finished submission

1. **Paper pages 6-7: unfinished tables.** Remove the visible `[EDIT]` instructions
   and unresolved cells, select a complete main-text table and reference the
   remaining results in an appendix. Table numbering skips Table 8.
2. **Paper section 11: reconstruction versus generation.** The decoder-shuffle
   table compares reconstruction MSE with fixed observed-trial latent codes.
   Calling the result improved "generation accuracy" changes the endpoint.
3. **Paper section 15: temporal order.** Flattening ordered 100 x 2 coordinates
   preserves their order in fixed input positions. The MLP lacks an explicit
   temporal convolution/locality bias; it does not discard temporal ordering.
4. **Paper section 4 and slide 4: PCA explanation.** PCA's optimal linear
   least-squares projection property concerns the matrix and criterion it is
   fitted to. Here it is fitted to spline coefficients. That property alone
   neither guarantees the lowest held-out trajectory MSE nor establishes why
   it beat these neural models. Report the observed advantage; describe loss
   tradeoffs as possible explanations rather than demonstrated causes.
5. **Paper section 9: provenance and inference.** The nested leave-one-person-out
   residual test, positive control and mean delta-R-squared of -0.017 were not
   located in the current tracked analysis implementations/results during this
   review. Request the exact source before retaining those assertions. Regardless
   of provenance, nonsignificant improvement does not establish "no additional
   information". The verified direct-context MAE comparison supports the narrower
   statement that direct summaries performed better on the tested targets.
6. **Paper tables: preserve source precision.** For example, VAE n=8 peak-speed
   mean probe MAE is approximately 8.691 in the verified all-model summary, not
   8.760. Several generation entries use inconsistent rounding padded to six
   decimals. Generate tables from verified CSVs instead of hand-transcribing them.
7. **Paper section 14: distinguish the two controls.** Timing-head architecture
   and minimum-jerk input/optimizer asymmetry are different issues. Neither is
   correctly summarized as both having arisen from "classifier capacity".
   The matched checks also did not eliminate every observed difference.
8. **Slides 2 and 8: scope of the claims.** The experiment concerns interception
   movements; "ballistic" should not silently rule out corrective movement.
   VAE ignores explicit condition inputs, not the task or its training data.
   Neural generation advantages refer to the reported feature metrics and
   tested capacities, not every possible aspect of movement fidelity.
9. **Slide 7: label the control accurately.** The 27/28 count refers to mean KS
   for each replacement control. Use the adjusted-p label and a correct bound:
   the largest of six VAE n=8 BH values is 1.54227018e-6, so `p < 1.5e-6` is
   slightly too small. `BH-adjusted p < 1.6e-6` is supported.
10. **Presentation usability.** The original slide-8 overlap is fixed in `ac8374b`;
    its new long recommendation needs wrapping and the cropped slide-7 graph
    needs a metric/scale label. Slide 3 still renders the parameter-count exponent
    as a missing glyph, and the declared slide budgets
    total 605 seconds (7:05 slides plus 3:00 demo), leaving no transition buffer.
    Plan approximately 9 minutes of content, with backup screenshots for the demo.

## Sources checked

- `production/8_pages_draft.pdf` (8 pages) and `build_submission_report.py`.
- `production/10_min_presentation.pdf` (9 slides) and `build_presentation.py`.
- Frozen `production/Interception_Movements_Methods_Reconstruction_Review.pdf`.
- `production/audit_2026_09_08/all_pairs_140.csv`.
- `production/audit_2026_09_08/uvae8_fingerprint_paired.csv`.
- `production/all_models_direct_context_summary.csv`.
- `src/vae_model.py`, `src/baseline_spline.py` and current generation helpers.

Private figures rendered for layout inspection are under the ignored
`production/assets/paz_progress_2026_09_13/` directory. No claim of a full
literature audit or exhaustive revalidation of Paz's newly written prose is made.
