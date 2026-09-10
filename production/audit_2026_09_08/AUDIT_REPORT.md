# Model integrity and model-selection review — 8 September 2026

**Decision:** no condition-routing, timing-input leakage, participant-split leakage, or primary loss/metric calculation defect was found in the frozen confirmatory neural runs. Their saved predictions reproduce. This supports using the existing results as evidence about the tested implementations; it does not establish optimal training, physiological validity, or one universally best model.

All new work is in this audit directory. Existing source, trained models, results and PDFs were left unchanged. The production PDFs remain the earlier report copies, not a newly revised submission. The updated meeting handoff was read and its hash is recorded in `source_manifest.json`.

## 1. What was checked

| Area | Evidence from this audit |
|---|---|
| Data and conditions | All 4,732 retained trial IDs and five-value conditions checked; raw MAT stimulus filenames, executed target speed and target-motion onset independently matched the cache for every retained trial. |
| Preprocessing | Independently recomputed filtering, onset detection, and both 100-point windows for all 4,732 cached recordings; also independently matched 112 prespecified random raw CSVs to cached frame regularization and marker indices. No discrepancy. This is **not** a fresh audit of every excluded raw trial. |
| Neural models | All 96 checkpoints: 48 CVAE, 24 conditional AE and 24 unconditional VAE; correct architecture/flags, 200 shape inputs, five conditions, two timing outputs, and physical timing withheld from encoders. |
| Training/evaluation separation | Four disjoint 17/4/7 participant partitions verified; every participant held out once; all context/query partitions nonempty, disjoint and complete. Train-only normalization independently reproduced for every checkpoint. Validation-objective arithmetic, KL schedule and selected epoch checked against every history. |
| Neural predictions | Independently implemented the MLP forward arithmetic using checkpoint tensors, bypassing the model's encode/decode methods; reconstructed saved per-trial MSE and timing predictions for every neural run. |
| Neural generation | Regenerated 80,640 samples across all 96 runs; manually checked empirical CDF distances and independently evaluated energy/MMD using pairwise-distance matrices. Maximum discrepancy from saved scores: 4.44e-16. No missing/nonfinite generated feature rows, no movement-duration floor or 98% crop cap triggered. 1,828 generated initiation values were zero after the declared nonnegative inverse transform. |
| Classical models | Refit all 16 spline/PCA configurations, including validation choice of coefficient scaling and Timing Ridge; refit Condition Ridge in all four folds. All saved reconstruction/timing predictions reproduced within declared tolerances. Also regenerated all 28 classical evaluations: 23,520 samples. One numerical-thread sensitivity is explained below. |
| Primary statistics | Independently rebuilt all 868 participant-run rows from individual prediction CSVs; recomputed all 56 primary paired tests and their BH adjustment. Reproduced. |
| Auxiliary statistics | 83 additional numerical checks passed: fingerprint controls, matched components, common-head/calibrated timing, condition diagnostics, and 560 pooled behavioral-probe scores. One correction-name error identified below. |
| Tests and loss | Existing suite: **70 passed**. Independently verified trajectory/timing loss reductions and Gaussian KL arithmetic. Two dormant source defects were reproduced, neither used by the reported study. |

The main executable audit recorded 2,457 passing checks. These are checks of specified properties, **not 2,457 independent scientific replications**. The audit scripts, tolerances, raw comparison tables and summaries are retained beside this report.

The first test attempt failed because Windows sandbox permissions blocked three temporary-directory fixtures. The normal-permission run passed all 70 tests. The first audit-script attempt also needed a correction to its own gradient test: an intentionally unused unconditional input has no gradient tensor. Neither issue was a project-model defect.

## 2. Answer to the conditioning concern

The current implementations really do use the following condition vector:

`[one_hot_sp_1, one_hot_sp_2, one_hot_sp_3, side_is_right, executed_target_speed]`.

CVAE and conditional AE concatenate it to **both** the encoder and decoder. Every saved conditional checkpoint had nonzero encoder and decoder sensitivity to every condition coordinate on the tested inputs. Valid category/side swaps changed their outputs. Unconditional VAE replaces the entire condition input with zeros; all 24 saved unconditional networks were exactly invariant to changed condition values. Conditional AE returns its deterministic latent mean, whereas both variational families sample Gaussian latent noise as specified.

As an additional constructive check, every unconditional checkpoint was copied in memory, conditioning was enabled, and just the two condition-weight blocks were zeroed. The resulting conditional network produced **exactly the same** latent means, trajectories and timings. Thus the conditional model class contains the unconditional solution. This is an implementation fact, not a claim that optimization reached the best solution.

Adding inputs therefore need not improve the result obtained by finite training, validation selection and held-out testing. Also, the minimized objective combines standardized trajectory error, timing error and KL; it is not the reported raw trajectory MSE alone. A worse MSE does not imply a worse optimized combined objective. Possible explanations include optimization/generalization behavior and redundant information already present in trajectory shape. This audit does not establish which explanation caused each observed difference.

### Frozen condition-shuffle diagnostic

Five fixed permutations shuffled whole condition tuples **within each test participant**, preserving valid category/speed combinations. No model was retrained. The following results hold when only decoder conditions are shuffled and the original latent codes are retained:

| Model | Correct-condition MSE | Shuffled-condition MSE | Participants better with correct conditions |
|---|---:|---:|---:|
| CVAE, n=3 | 0.176981 | 0.180578 | 25/28 |
| CVAE, n=8 | 0.032481 | 0.035105 | 28/28 |
| Conditional AE, n=3 | 0.169661 | 0.171582 | 26/28 |
| Conditional AE, n=8 | 0.025968 | 0.026693 | 27/28 |

Changing conditions at both encoder and decoder produces much smaller, mixed changes; for CAE8 the aggregate difference slightly favors shuffled conditions. Both arms are preserved in `condition_shuffle_summary.csv`. These are descriptive perturbation results, not causal tests or proof of reliable task-controlled generation. The decoder-only experiment deliberately breaks the usual association between code and condition; that limitation matters.

**Defensible wording:** “Explicit conditioning was implemented and affected the learned mappings, but did not consistently improve the evaluated outcomes relative to the unconditional model.”

**Unsupported wording:** “Conditions do not explain movement differences,” “conditions were ignored,” or “conditioning should always improve these held-out metrics.”

## 3. Findings that require a change or disclosure

| Finding | Importance and action | Full neural rerun? |
|---|---|---|
| A condition-trajectory table calls Bonferroni adjustment “Holm.” | Correct `scripts/analyze_condition_effects.py:198` and the corresponding report/dashboard table before final release. Proper Holm gives n=3 **0.551874**, not 1.000; n=8 remains **0.132480**. Neither is significant. Corrected table is already in this directory. | No; recompute the two adjusted values and update wording/assets. |
| One Condition Ridge generative score depends slightly on numerical thread count. | At fold1/seed44/subject08, energy is 4.055111697 with one thread versus saved 4.055317064; MMD differs by 0.00005397. Saved values reproduce exactly at 2, 4, 8 and 16 threads. Record numerical-library/thread settings in future exports. The model uses alpha=0 with redundant one-hot predictors/intercept, so coefficient solutions also vary numerically. | No. Replacing all classical joint scores by the one-thread reproductions changes **none** of the 140 raw p-values or BH decisions. |
| Dashboard loader assumes CVAE. | `src/confirmatory_dashboard.py:118` hardcodes both `variational=True` and `use_condition=True`. Correct for its current CVAE files; incorrect if another family is substituted. Supporting VAE/CAE requires reading saved flags and matching model-specific fingerprints, covariance and normalization. | No; dashboard code/assets need work if model support is expanded. |
| CNN training option rejects arguments supplied by the trainer. | `src/train.py` supplies `variational` and `use_condition`; `ConvCVAE` does not accept them. Fix the interface or disable this unsupported option before advertising CNN training. All reported checkpoints are MLPs. | No impact on current results. |
| Filter short-input boundary is off by one. | `src/preprocessing.py:38` skips lengths below 15 but length 15 also fails `filtfilt`. Correct the guard if maintaining this API. Shortest retained recording is 112 samples; modeled windows have 100 points. | No impact on current results. |
| Some comments describe old settings/units. | The spline class docstring still describes timing-inclusive PCA, and some comments use mm despite unconfirmed tracker units. The effective final runner is shape-only; use its settings when explaining the work. | No; documentation correction. |

Additional limits that should remain explicit:

* Validation selection is correctly implemented but is not proof of optimization convergence. One CVAE n=3 checkpoint (fold3, seed43) was selected at epoch 2; training continued to epoch 51 and its later full-objective validation scores were worse. Five runs reached the 150-epoch cap. Keep these runs; do not replace an unfavorable seed after looking at test performance. A future selection/training sensitivity should use a common prespecified protocol for all compared families.
* Source histories identify the training checkout as dirty; this audit reproduces saved checkpoint behavior and current evaluation, not every historical training update bit for bit. Source hashes and current checkpoint hashes are archived.
* Participant errors share trained models and folds have overlapping development participants. In addition, the existing character-sum seed scheme maps 28 subject IDs to only 10 sampling seeds. This is not data leakage, but shared simulation randomness adds another dependence concern. Future experiments should use stable collision-resistant subject seeds while preserving common randomness across model arms where intended.
* Wilcoxon calculations reproduce, but their usual independence/symmetry assumptions are not established merely by correct arithmetic. BH and the supplementary Holm adjustment do not remove cross-validation dependence or selection on these same benchmark participants.
* SciPy's standard two-sample KS p-values assume continuous distributions. Quantized timing and heuristic peak counts contain ties/discrete values. The **KS statistics used to compare models remain usable discrepancies**; per-feature KS rejection counts should not be presented as fully calibrated biological significance tests. A separate calibrated test would require an appropriate sampling/dependence design.
* Generated latent samples use a context centroid and shared training-derived covariance. This empirical generation procedure is different from simply drawing the fitted CVAE prior, and does not guarantee disentangled person/task factors.
* Fitted minimum-jerk counts, heuristic speed-peak counts, and biological submovements are distinct. The heuristic count is floored at one. Zero-phase filtering, timing-based cropping, fixed minimum-jerk bounds and model-order selection remain analysis assumptions.

## 4. New all-model comparisons

These are **post-review exploratory** comparisons, prespecified in this directory before calculation: all ten family pairs at n=3 and n=8, seven endpoints each, for **140 paired tests**. Three training/generation seeds were averaged within each participant; spline has one deterministic fit per fold/dimension. BH uses the complete family of 140. Holm across the same 140 is a conservative sensitivity, not a correction for cross-validation dependence. The original 56-test family is preserved unchanged.

| Question | Supported conclusion |
|---|---|
| Best neural reconstruction at n=3? | Unconditional VAE: MSE 0.148299 versus CVAE 0.176981 (BH q=0.00163) and CAE 0.169661 (q=0.03695). These two contrasts do not survive Holm across 140. |
| Best neural reconstruction at n=8? | CAE: 0.025968 versus CVAE 0.032481 and VAE 0.033487; both survive BH and Holm. |
| Best neural feature-distribution performance at n=8? | VAE has the lowest means: KS 0.215121, energy 0.411014 and MMD² 0.063881. Both VAEs beat CAE on these three measures after BH and Holm. |
| Does VAE8 reliably beat CVAE8 on everything? | No. Energy favors VAE (BH q=0.02575), with the same direction in all four fold means; KS q=0.13686 and MMD² q=0.05182 do not pass 0.05. The energy contrast does not survive Holm across 140. Reconstruction also shows no significant difference. |
| Best classical model? | Spline+PCA is the strong reconstruction/timing baseline. There is no universal classical winner: at n=3 Condition Ridge beats spline on all three main distribution distances after BH, despite much worse reconstruction and timing. At n=8 their main distribution contrasts do not survive BH. |

Do not count metric “wins” to invent an overall score. Reconstruction sees each complete test trajectory; generation receives context information and sampled query conditions but not query shapes. The tasks are different.

## 5. New VAE8 controls

### Does its personal fingerprint contribute?

All 12 VAE8 checkpoints were evaluated with the same own/population/wrong-person protocol used for CVAE. Each comparison holds decoder, latent noise, condition draws, covariance and distance reference fixed. Wrong-person results average six donor distances, not pooled donor samples.

| Fingerprint | Mean KS | Energy | MMD² |
|---|---:|---:|---:|
| Own | 0.215121 | 0.411014 | 0.063881 |
| Training-population mean | 0.284126 | 0.878681 | 0.141604 |
| Other held-out person | 0.314405 | 1.168337 | 0.194781 |

Own improves every distance for 27/28 participants against each control; all six BH-adjusted p-values are at most 1.55e-6. This supports useful **within-session personal information**, not cross-session identity, cognitive-strategy discovery or zero-shot personalization.

### Does it repair the minimum-jerk limitation?

No demonstrated repair. All 2,376 recorded input arrays and durations were checked before reusing their matched fits. VAE8 generated 840 new trajectories (10 per person per training seed) and received the same two-restart/400-evaluation fit procedure.

* All 840 fits completed; 839 selected candidates converged and 734 fits had all candidate counts converged. Convergence was not used to exclude results.
* The fixed 56-sample higher-budget subset retained the same counts in 56/56 cases, with all selected fits converged.
* The one other nonconverged selected fit was separately rerun with four restarts/700 evaluations. It converged, retained two components, and changed normalized error from 0.03781723 to 0.03781690. Original fixed-budget results remain the reported comparison.
* Count total variation is **0.235518 for VAE8 versus 0.232263 for CVAE8**. None of nine paired component-distribution comparisons survives BH (all q=0.86429).
* All seven principal VAE component discrepancies remain above the existing empirical plugin reference's 95th percentile. This reference is a finite-sample diagnostic, not a population confidence interval or formal null test.

The result supports “VAE also captures useful personal distributional information,” not “VAE solves submovement generation.”

## 6. Effect of the transcript-updated handoff

The update changes emphasis and the dashboard recommendation, not the verified tensor routing:

1. Lead Results with recorded-input/decoded-reconstruction examples and participant-balanced MSE, then separately explain timing, generation, personal-context controls, enrollment/probes and components. A generated trajectory must not be labeled a reconstruction.
2. Retain **n=3 as the compact scientific setting** if Jason's two/three-number aim is the primary requirement; n=8 is the observed capacity comparison. Better n=8 error alone does not satisfy a two/three-number constraint. n=15/20 is proposed future work, not an existing result.
3. Show an actual legible three-dimensional n=3 latent plot, ideally trial points and centroids, within a single fitted model/fold or separate panels. Independently trained latent axes cannot simply be pooled as a shared coordinate system.
4. The suggested linear discrimination analysis still needs an explicit question and leakage-resistant split before choosing a test. The present nearest-centroid enrollment result is not already that analysis.
5. Limited data, weak conditioning, and a task favoring deterministic compression remain discussion hypotheses. The audit does not turn them into measured causes.

### Dashboard recommendation after these checks

**Provisional compact, task-conditional main interface: CVAE n=3.** This is justified by the compact research requirement, explicit condition support, useful own-person controls, and competitive distribution results; it is **not** a claim of lowest reconstruction error or universal superiority. For n=3, CVAE's mean KS is lower than VAE's, while VAE reconstructs better.

**Additional marginal-distribution option: unconditional VAE n=8.** Its numerical generation results and personal-context control justify offering it. Its condition controls must be disabled or explicitly inapplicable: it cannot change output in response to requested side/speed/category. It should not be marketed as the winning task-conditional generator.

Spline+PCA remains the principal classical comparator, and CAE8 is useful as the reconstruction-focused comparison. A model selector is reasonable after implementing family-aware loading/assets and explaining what each mode supports. Existing condition sliders remain exploratory because condition-stratified fidelity has not established reliable task control.

Before claiming physiological strategies or extrapolation, Jason still needs to confirm units, events, onset/endpoint choices, pre-target-motion coverage, inclusion rules and the adapted minimum-jerk assumptions. These are distinct from a neural code bug. If Jason requires appearance-to-motion behavior inside the modeled window, changing that scientific scope would require preprocessing and matched model reruns.

## 7. Files and reproduction

Start with `audit_summary.json`, `all_pairs_140.csv`, `condition_shuffle_summary.csv`, `uvae8_fingerprint_paired.csv`, and `uvae8_vs_cvae8_components.csv`. Corrected auxiliary values are in `condition_trajectory_corrected_holm.csv`. Detailed source/checkpoint provenance is in `source_manifest.json` and `checkpoint_audit.csv`.

Run the scripts in this order from the repository with Python and its existing dependencies:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python production/audit_2026_09_08/audit_integrity.py
python production/audit_2026_09_08/audit_generation.py
python production/audit_2026_09_08/audit_classical_generation.py
python production/audit_2026_09_08/diagnose_ridge_precision.py
python production/audit_2026_09_08/audit_statistics.py
python production/audit_2026_09_08/audit_preprocessing_loss.py
python production/audit_2026_09_08/condition_sensitivity.py
python production/audit_2026_09_08/compare_and_control.py
python production/audit_2026_09_08/finish_audit.py
```

The candidate component runner resumes saved jobs and needs normal Windows worker-process permissions. The one targeted convergence recheck is separately preserved in `uvae8_nonconverged_targeted_recheck.csv`. No script above retrains the neural models or overwrites the old study/report outputs.

Methodological references: SciPy documents the continuous-sample assumption for [two-sample KS](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html). Overlap/dependence in cross-validation estimates is discussed by [Bengio and Grandvalet (2004)](https://www.jmlr.org/papers/v5/grandvalet04a.html). The numerical findings in this report come from the local artifacts and executed checks, not those external sources.
