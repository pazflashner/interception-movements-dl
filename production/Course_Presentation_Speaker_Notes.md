# Presentation notes

Slides 1–10 form a nine-minute planned talk including an 80-second demo. One minute remains for transitions. Detailed question explanations are in Course_Methods_Explanations.pdf, separate from the PPTX.

## 1. Modeling interception movement distributions

Presenter: Paz. Planned time: 20 seconds.

Our goal is to model the variability of interception movements using a compact representation of a participant. We compare neural models with spline plus PCA and implement them in a dashboard. The principal result concerns generation at eight dimensions. Three numbers were our original compact target, so we keep that comparison visible. Say distribution, rather than promising an exact prediction of the next trial. Sources: course report Abstract and Introduction.

## 2. One project, two modeling tasks

Presenter: Paz. Planned time: 40 seconds.

Our project models movement variability using compact personal representations and exposes the models in a dashboard. Reconstruction sees the particular trial; generation never sees the query path. Personal-center replacement asks whether the correct participant context improves the generator. This is evidence of within-session personalization, not a complete stable fingerprint. Sources: course report sections 1, 3 and 6.

## 3. The task and the evaluation split

Presenter: Paz. Planned time: 50 seconds.

The participant moves a finger from the lower starting region to the upper interception region as the target arrives. There are three starting categories on each side, with a related speed range. We excluded four trials without an arrival and 27 very late arrivals. The two low-count participants mostly had fewer files available, not extensive exclusions. Population training excludes the seven test people. We then use some recordings from each test person as context and reserve the other recordings as query. Source: course report section 2; src/data_loading.py, src/preprocessing.py; production/cohort_trial_counts.csv. Task image: production/presentation_assets/task_schematic.png, inherited from Paz’s supplied presentation. Data were collected by Beatris Giterman in Jason Friedman’s laboratory. Spatial coordinates are cm; nominal starting positions are mm from screen center.

## 4. The compression pipelines

Presenter: Paz. Planned time: 65 seconds.

First we resample a filtered go-to-end recording onto 100 equally spaced phases. This keeps relative waiting but removes seconds from the coordinate input. In spline plus PCA, one trial gives nine x and nine y spline coefficients. We stack training trials as rows, giving N by 18, and fit PCA on that matrix. The neural model receives the 200 coordinate values directly after training-derived standardization. CVAE and VAE have mean and variance heads and a KL penalty. CAE is deterministic and has no KL penalty. CVAE/CAE receive conditions at encoder and decoder; VAE zeros them. The timing targets are never encoder inputs. The separate methods explanation PDF explains the actual objective and inverse matrices. Sources: src/baseline_spline.py, src/confirmatory_spline.py, src/vae_model.py, src/train.py. Five interior knots plus cubic degree three plus one give nine basis coefficients per axis. Timing uses log(t + 0.001) to accommodate zero and compress its scale, then training standardization.

## 5. Reconstruction favors spline + PCA

Presenter: Paz. Planned time: 45 seconds.

Reconstruction means compressing and decoding the same observed trajectory. We compare the resulting 200 coordinates with the original and average squared errors. Thus it measures information retained by the representation; it is not a prediction without seeing the trial. Scores first average within person and across neural seeds, then equally across 28 people. Spline has the lowest mean at both capacities. The six spline-versus-neural comparisons pass BH, but the spline-versus-CAE eight-dimensional difference has Holm p=0.106. The other five survive Holm. VAE wins the neural reconstruction comparison at three dimensions under BH; CAE is best at eight. Source: production/audit_2026_09_08/all_model_means.csv and all_pairs_140.csv. Bars show participant-balanced means after seed averaging. Markers compare each bar with the reference named above the chart, using the saved adjusted paired tests. In model-comparison charts, * means BH and Holm both pass .05; dagger means BH only; ns means neither. The personal-center chart uses its original BH-corrected control family. There are no error bars, and no p-value has been converted into an uncertainty interval.

## 6. Generation uses separate context trials

Presenter: Paz. Planned time: 40 seconds.

A fingerprint is the average of the encoded context means. We then add training-derived latent noise to that center and decode 120 movements. All participants share the covariance, but their centers differ. VAE generation here is not simply drawing from the standard-normal training prior. For CVAE the query-condition mixture supplies conditions, but the decoder never receives the actual query paths or their durations. The comparison is between two sets of trials, so there is no correct one-to-one pairing of a generated path with an observed trial. Source: src/context_query.py, src/confirmatory_spline.py, scripts/run_review_controls.py; appendix A4.

## 7. Generated kinematic-feature distributions

Presenter: Paz. Planned time: 50 seconds.

KS measures the largest gap between empirical cumulative curves for one feature. We average eleven KS statistics, not eleven p-values. Energy and MMD instead compare the joint eleven-feature vectors after training standardization. VAE eight has mean KS 0.2151, energy 0.4110 and MMD squared 0.0639. CVAE is close. VAE versus CVAE eight has BH q=.0257 for energy, .137 for KS and .0518 for MMD; none survives Holm. We can select VAE as a practical default but cannot claim it significantly wins every endpoint. Features depend on predicted durations as well as shape. Sources: course report Table 2; audited means and 140-test CSV. The restored figure uses empirical cumulative distributions, without histogram bins. KS is the largest vertical gap between two such curves. Bars show participant-balanced means after seed averaging. Markers compare each bar with the reference named above the chart, using the saved adjusted paired tests. In model-comparison charts, * means BH and Holm both pass .05; dagger means BH only; ns means neither. The personal-center chart uses its original BH-corrected control family. There are no error bars, and no p-value has been converted into an uncertainty interval.

## 8. Full-trajectory generation supports VAE and CVAE

Presenter: Seman. Planned time: 80 seconds.

This is our new central analysis. We flatten each whole phase-normalized path and divide by square root of 200. Euclidean distance then means RMS coordinate difference. Energy and MMD compare the generated set with the held-out set and also account for within-set variation. At eight dimensions both variational models improve on spline, and this survives both multiple-testing corrections and equalizing the training variation of the two axes. Their difference from each other is not significant. At three dimensions, raw MMD favors them over spline, but raw energy does not, and balanced-axis MMD no longer passes BH. This is why the eight-dimensional conclusion is stronger. These are phase-path results, not a validation of physical timing. Source: production/trajectory_distribution_2026_09_13/summary.csv, paired_comparisons.csv and protocol.json. Bars show participant-balanced means after seed averaging. Markers compare each bar with the reference named above the chart, using the saved adjusted paired tests. In model-comparison charts, * means BH and Holm both pass .05; dagger means BH only; ns means neither. The personal-center chart uses its original BH-corrected control family. There are no error bars, and no p-value has been converted into an uncertainty interval.

## 9. The participant’s own center improves generation

Presenter: Seman. Planned time: 70 seconds.

This is the direct check that the personal center helps the generator. Replacing it by a population center or another person’s center makes the feature distributions worse. We keep random draws and other inputs fixed, so the center is what changes. This supports useful information about this participant’s session. It does not give a name to each latent axis or show the same fingerprint would remain stable on another day. The control uses feature endpoints; we have not repeated it on the new full-trajectory endpoints. Source: frozen laboratory report section on personal fingerprints; course report Figure 3 and appendix A4. Bars show participant-balanced means after seed averaging. Markers compare each bar with the reference named above the chart, using the saved adjusted paired tests. In model-comparison charts, * means BH and Holm both pass .05; dagger means BH only; ns means neither. The personal-center chart uses its original BH-corrected control family. There are no error bars, and no p-value has been converted into an uncertainty interval.

## 10. Dashboard demonstration

Presenter: Seman. Planned time: 80 seconds.

Open the dashboard already running in a browser. Explain that generated curves are samples, not predictions paired with particular recorded trials. Show the VAE eight default, then three dimensions and an alternative model. Do not interpret a single chosen plot as the significance test. VAE does not respond to condition controls because it ignores conditions. Spline condition changes affect only timing; CVAE/CAE use conditions for shape and timing. Finish with Benchmarks. If live launch fails, the slide contains recorded and generated examples from a fixed subject, chosen by identifier. Launch from the repository: python -m streamlit run src/confirmatory_dashboard.py --server.port 8510 --server.address 127.0.0.1 --browser.gatherUsageStats false. Open http://127.0.0.1:8510. Ctrl+C stops the terminal server. Source: production/MULTIMODEL_UPDATE.md; src/confirmatory_dashboard.py; generated-array figure provenance in course_figures.py.
