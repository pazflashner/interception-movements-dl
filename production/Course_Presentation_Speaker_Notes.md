# Presentation notes

Slides 1–10 form a nine-minute planned talk including an 80-second demo. One minute remains for transitions. Slides 11 onward are question backups.

## 1. Modeling interception movement distributions

Planned time: 25 seconds.

Our goal is to model the variability of interception movements using a compact representation of a participant. We compare neural models with spline plus PCA and implement them in a dashboard. The principal result concerns generation at eight dimensions. Three numbers were our original compact target, so we keep that comparison visible. Say distribution, rather than promising an exact prediction of the next trial. Sources: course report Abstract and Introduction.

## 2. The task and the evaluation split

Planned time: 50 seconds.

The participant moves a finger from the lower starting region to the upper interception region as the target arrives. There are three starting categories on each side, with a related speed range. We excluded four trials without an arrival and 27 very late arrivals. The two low-count participants mostly had fewer files available, not extensive exclusions. Population training excludes the seven test people. We then use some recordings from each test person as context and reserve the other recordings as query. Source: course report section 2; src/data_loading.py, src/preprocessing.py; production/cohort_trial_counts.csv. Task image: production/presentation_assets/task_schematic.png, inherited from Paz’s supplied presentation.

## 3. The compression pipelines

Planned time: 65 seconds.

First we resample a filtered go-to-end recording onto 100 equally spaced phases. This keeps relative waiting but removes seconds from the coordinate input. In spline plus PCA, one trial gives nine x and nine y spline coefficients. We stack training trials as rows, giving N by 18, and fit PCA on that matrix. The neural model receives the 200 coordinate values directly after training-derived standardization. CVAE and VAE have mean and variance heads and a KL penalty. CAE is deterministic and has no KL penalty. CVAE/CAE receive conditions at encoder and decoder; VAE zeros them. The timing targets are never encoder inputs. Backup slides explain the actual objective and inverse matrices. Sources: src/baseline_spline.py, src/confirmatory_spline.py, src/vae_model.py, src/train.py.

## 4. Reconstruction favors spline + PCA

Planned time: 45 seconds.

Reconstruction means compressing and decoding the same observed trajectory. We compare the resulting 200 coordinates with the original and average squared errors. Thus it measures information retained by the representation; it is not a prediction without seeing the trial. Scores first average within person and across neural seeds, then equally across 28 people. Spline has the lowest mean at both capacities. The six spline-versus-neural comparisons pass BH, but the spline-versus-CAE eight-dimensional difference has Holm p=0.106. The other five survive Holm. VAE wins the neural reconstruction comparison at three dimensions under BH; CAE is best at eight. Source: production/audit_2026_09_08/all_model_means.csv and all_pairs_140.csv.

## 5. Generation uses separate context trials

Planned time: 60 seconds.

A fingerprint is the average of the encoded context means. We then add training-derived latent noise to that center and decode 120 movements. All participants share the covariance, but their centers differ. VAE generation here is not simply drawing from the standard-normal training prior. For CVAE the query-condition mixture supplies conditions, but the decoder never receives the actual query paths or their durations. The comparison is between two sets of trials, so there is no correct one-to-one pairing of a generated path with an observed trial. Source: src/context_query.py, src/confirmatory_spline.py, scripts/run_review_controls.py; appendix A4.

## 6. Generated kinematic-feature distributions

Planned time: 55 seconds.

KS measures the largest gap between empirical cumulative curves for one feature. We average eleven KS statistics, not eleven p-values. Energy and MMD instead compare the joint eleven-feature vectors after training standardization. VAE eight has mean KS 0.2151, energy 0.4110 and MMD squared 0.0639. CVAE is close. VAE versus CVAE eight has BH q=.0257 for energy, .137 for KS and .0518 for MMD; none survives Holm. We can select VAE as a practical default but cannot claim it significantly wins every endpoint. Features depend on predicted durations as well as shape. Sources: course report Table 2; audited means and 140-test CSV.

## 7. Full-trajectory generation supports VAE and CVAE

Planned time: 70 seconds.

This is our new central analysis. We flatten each whole phase-normalized path and divide by square root of 200. Euclidean distance then means RMS coordinate difference. Energy and MMD compare the generated set with the held-out set and also account for within-set variation. At eight dimensions both variational models improve on spline, and this survives both multiple-testing corrections and equalizing the training variation of the two axes. Their difference from each other is not significant. At three dimensions, raw MMD favors them over spline, but raw energy does not, and balanced-axis MMD no longer passes BH. This is why the eight-dimensional conclusion is stronger. These are phase-path results, not a validation of physical timing. Source: production/trajectory_distribution_2026_09_13/summary.csv, paired_comparisons.csv and protocol.json.

## 8. The participant’s own center improves generation

Planned time: 45 seconds.

This is the direct check that the personal center helps the generator. Replacing it by a population center or another person’s center makes the feature distributions worse. We keep random draws and other inputs fixed, so the center is what changes. This supports useful information about this participant’s session. It does not give a name to each latent axis or show the same fingerprint would remain stable on another day. The control uses feature endpoints; we have not repeated it on the new full-trajectory endpoints. Source: frozen laboratory report section on personal fingerprints; course report Figure 3 and appendix A4.

## 9. What the compact model achieves

Planned time: 45 seconds.

Our main conclusion is useful compact generation, not that deep learning beats every possible baseline. Keeping the entire context paths and resampling them yields lower mean discrepancies. That uses more participant memory and cannot be summarized as an eight-number fingerprint. Repeating only the context mean is much worse as a distribution despite having a good mean-path error. Direct context behavioral summaries also beat our latent linear probes. These observations describe what compression loses and motivate a richer model. Default selection used this cohort and needs independent confirmation. Sources: full-path summary CSV; production/all_models_direct_context_summary.csv; course Discussion.

## 10. Dashboard demonstration

Planned time: 80 seconds.

Open the dashboard already running in a browser. Explain that generated curves are samples, not predictions paired with particular recorded trials. Show the VAE eight default, then three dimensions and an alternative model. Do not interpret a single chosen plot as the significance test. VAE does not respond to condition controls because it ignores conditions. Spline condition changes affect only timing; CVAE/CAE use conditions for shape and timing. Finish with Benchmarks. If live launch fails, the slide contains recorded and generated examples from a fixed subject, chosen by identifier. Launch from the repository: python -m streamlit run src/confirmatory_dashboard.py --server.port 8510 --server.address 127.0.0.1 --browser.gatherUsageStats false. Open http://127.0.0.1:8510. Ctrl+C stops the terminal server. Source: production/MULTIMODEL_UPDATE.md; src/confirmatory_dashboard.py; generated-array figure provenance in course_figures.py.

## 11. Backup: resampling and the two time windows

Backup only.

We start with a regularized 240 Hz grid, smooth using a fourth-order 10 Hz forward-backward Butterworth filter and select the modeled window. We fit a cubic interpolant through the existing filtered points and evaluate it at j/99. This interpolation is different from fitting the later five-knot compression model. Subtracting the first position aligns starts, but does not scale amplitude or rotate paths. We retain initiation and movement seconds as separate targets. Early movement after target motion remains; the earlier appearance interval is outside the modeled window. Source: src/preprocessing.py, src/trajectory_view.py; appendix A1.

## 12. Backup: what becomes the PCA matrix?

Backup only.

Moni asked what we feed into PCA. It is a matrix where each row is one trial’s eighteen spline coefficients. We first fit a spline separately to x and y of each trial. Nine x and nine y coefficients give eighteen numbers. We stack training trials and PCA learns n directions in that eighteen-dimensional space. Neither test trials nor validation trials train those PCA directions. The number of knots and the number of latent coordinates are different controls. Source: src/baseline_spline.py, CompactSplineRepresentation; appendix A2.

## 13. Backup: fitting and decoding spline + PCA

Backup only.

There are two compression steps: sampled path into smooth coefficients, then coefficients into a smaller PCA code. Fit B c_x to x and B c_y to y. The code z is obtained by centering/scaling the coefficient vector and projecting onto retained axes. To decode, apply inverse PCA and inverse scaling, split the eighteen coefficients into two blocks and multiply both by the basis B. Raw PCA still centers columns. PCA optimizes variance in coefficient space and does not guarantee the best trajectory MSE against all methods. Sources: src/baseline_spline.py; appendix A2. The on-slide equation matches the effective forward and inverse transformations.

## 14. Backup: predicting physical timing

Backup only.

The coordinate path is on normalized phase, so the model must predict seconds separately. Initiation is time from target motion to finger onset, movement time is onset to end. We take log(t+.001) to accommodate zeros, standardize training targets, and reverse these transformations after prediction. Spline Ridge receives the code and five conditions, with training input standardization and alpha from .01, .1, 1, 10, 100 selected using validation error on standardized log targets. Neural models have a two-value timing head. Huge positive timing errors were not upper-clipped out of the main scores. Source: src/confirmatory_spline.py, TimingRidge; src/vae_model.py; appendix A3.

## 15. Backup: the neural training objective

Backup only.

The actual code uses reduction=none, then sum across output coordinates and mean across trials. We must not simply write MSE plus 20 MSE because one output has two hundred coordinates and the other has two. KL measures how far the encoder’s Gaussian posterior is from the standard normal. Beta warms from zero to one over fifty epochs, while checkpoint validation uses the final beta-one objective. Adam learning rate .001, batch size 64, gradient norm clipping at five, patience 25, maximum 150 epochs, three seeds. Source: src/vae_model.py vae_loss; src/train.py; appendix A3.

## 16. Backup: VAE, CVAE and CAE

Backup only.

Our VAE is an unconditional ablation in the same code framework. It does not receive meaningful conditions because we zero them at both ends. CVAE uses actual conditions. CAE uses actual conditions but suppresses probabilistic sampling and KL. The condition vector is three one-hot category entries, a side value (left zero, right one), and normalized executed speed. We checked input routes and shuffled conditions. The routes are active, but these results cannot settle whether another architecture or training regime would use conditions better. Source: src/vae_model.py encode_condition/encode_trial_condition; src/train.py; audit records.

## 17. Backup: where the generated variation comes from

Backup only.

For each training person, subtract that person’s mean code from each trial code. Pool these residuals and estimate their covariance. Add the mean encoder posterior variance on the diagonal for VAE/CVAE and add a tiny identity regularizer for numerical stability. CAE and spline have no posterior variance term. We do not fit a separate covariance to a new participant’s context trials. Generation draws a multivariate normal around their context center and uses the shared decoder. A nonlinear decoder can still produce different visible spread at different centers. Source: src/context_query.py and src/confirmatory_spline.py; appendix A4.

## 18. Backup: leakage and held-out evaluation

Backup only.

A test person is new to model fitting, but provides context recordings. Query recordings only provide the targets for generation scores, with task metadata allowed for a matched-condition mixture. Reconstruction deliberately reads the query path because that task is compression. None of the test data changes fitted PCA axes, learned neural parameters, training covariance or normalization. This prevents a common misunderstanding: test context is not retraining. Model selection for the final dashboard still used the evaluated cohort, so independent post-selection validation remains future work. Source: participant_folds.json, context_query.py; appendices A1/A4.

## 19. Backup: what the eleven features measure

Backup only.

For observed data we use the onset-to-end movement segment. For generated data we use the predicted initiation fraction to crop the go-to-end path, capped at .98, and floor movement duration at one millisecond. Both movement segments become one hundred phase points and are filtered with cutoff min(10 Hz, .45 times sample rate). Sample rate is 99 over movement seconds. Gradients give speed. Peak count uses prominence ten percent of maximum and roughly fifty milliseconds separation, with at least one reported peak. This is distinct from minimum-jerk component fitting. Source: src/features.py; appendix A5.

## 20. Backup: KS and joint distribution distances

Backup only.

If a recorded feature has seventy percent of trials below a value and the generated feature has forty percent, the empirical CDF gap there is .3. KS takes the largest gap over all thresholds. Its output is a discrepancy. We average discrepancies across features, not significance tests. Energy and MMD use complete feature vectors and can notice dependence between features that separate marginal KS statistics may miss. Standardization comes from training. Correlated features do not mean we should remove whichever feature makes a p-value improve; removal defines a new endpoint. Sources: src/context_query.py; appendix A5/A6; Gretton et al. JMLR 2012.

## 21. Backup: energy discrepancy and mean collapse

Backup only.

Average error against all recorded paths can favor a collapsed output near the center. Energy subtracts within-generated and within-recorded distances from twice the cross average, addressing the distributions rather than only their centers. Our energy estimator includes within-sample diagonal zeros and is not square-rooted afterward. The mean-only reference is actual analysis output, not a hypothetical numerical illustration. Source: src/trajectory_distribution.py; trajectory_distribution summary CSV; Szekely and Rizzo 2013; appendix A6/A8.

## 22. Backup: Gaussian-kernel MMD

Backup only.

The kernel gives high similarity to close vectors and low similarity to distant vectors. MMD compares average within-recorded and within-generated similarity with cross similarity. The within terms omit identical-element pairs and divide by m(m−1) or r(r−1); cross divides by mr. We estimate gamma once from at most 512 deterministic training trials in each representation and use it for all models in the rotation. Whole paths and feature vectors have different units and scales, so their absolute MMD values are not interchangeable. Source: src/trajectory_distribution.py; src/context_query.py; Gretton et al., A Kernel Two-Sample Test, JMLR 13 (2012).

## 23. Backup: 28 paired scores, not thousands of independent trials

Backup only.

Each participant supplies one score per model for the paired comparison. Three seeds are repeated model fits, not three independent people. Wilcoxon ranks the absolute paired differences and uses their signs. Pratt keeps zeros in ranking but omits their signed contribution. The usual location interpretation assumes a symmetric difference distribution. BH adjusts the complete declared family, not only the comparisons we liked. We additionally show Holm. Training folds overlap, so these remain exploratory comparisons, and a fresh cohort is still needed for replication. No formal three-versus-eight or resampling-reference test was added. Source: src/statistical_tests.py, all_pairs_140.csv, paired_comparisons.csv; appendix A6.

## 24. Backup: the boundaries of the conclusion

Backup only.

We should volunteer the relevant limitation instead of defending every result as a success. Direct context means and standard deviations were more accurate than our latent linear probes, and many probe R-squared values were negative. Matched minimum-jerk component comparisons did not establish corrected model differences. Timing and event assumptions remain important, including the tracker endpoint arriving on average 26 ms before the MATLAB arrival marker. More data might help, but we did not run a learning curve that proves sample count caused the failures. Source: frozen report pp.16–28, production/all_models_direct_context_summary.csv; course Discussion and appendix A10.

## 25. Backup: launch and source navigation

Backup only.

Full PowerShell path: cd "D:\oneDrive_Main\OneDrive\Desktop_onedrive\uni2024\Third Year\semester B\Workshop in AI\interception-movements-dl". Then use the command shown, adding --server.address 127.0.0.1 --browser.gatherUsageStats false if desired. On the authoring computer Python is C:\Python313\python.exe. Source navigation: src/vae_model.py and src/train.py for neural models; src/baseline_spline.py and src/confirmatory_spline.py for spline/PCA/Ridge; src/context_query.py and src/features.py for generation/metrics; src/trajectory_distribution.py for full paths; production/audit_2026_09_08 for original result CSVs; production/trajectory_distribution_2026_09_13 for new results. Keep the terminal running during the demo.
