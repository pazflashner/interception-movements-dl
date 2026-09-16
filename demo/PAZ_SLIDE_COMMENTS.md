# Comments on Paz's preserved slides

These comments concern the pushed deck at commit 1429c07. Slides 2–8 in the new
presentation remain unchanged as requested. These are wording/data-label fixes;
they do not require rerunning models.

| Slide | Comment / proposed replacement |
| --- | --- |
| 4 | The selected spline runs used **unstandardized coefficients, centered by PCA**. They did not divide the 18 coefficients by their standard deviations. All eight matched spline fits at n=3,8 have `standardize_spline_coefficients=False`. |
| 4 | “VAE trained on the trajectory alone” is better expressed as **“The VAE encoder receives the trajectory without conditions.”** Training also includes predicted timing targets. |
| 5 | PCA minimizes reconstruction error in the **coefficient representation**, not necessarily decoded trajectory MSE. Replace the broad optimality justification with **“Spline + PCA achieved the lowest observed trajectory MSE.”** |
| 5 | The n=3 spline MSE rounds to **0.0973** at four decimals, rather than 0.0972. “Condition-only Ridge reference” is clearer than “Floor,” which could suggest a mathematical bound. |
| 6 | Context/query trials were split randomly with stratification; use **“separate context trials”** instead of “earlier.” Query **paths and timing labels** are withheld, while conditional generation uses query-condition metadata. |
| 6 | The thin curves show 30 paths; the thick line averages **all paths**, not just those 30. |
| 7 | The feature list should include **speed-peak count** and **endpoint x and y**. There is no endpoint-z feature in the evaluated eleven. |
| 7 | **“Calculate a KS distance per feature, then average the eleven distances.”** We do not average KS p-values. |
| 8 | The blanket **“p < .001 under BH and Holm”** is too strong: some Holm-adjusted values are up to about .00362. **“Significant under both BH and Holm at .05”** is supported for VAE/CVAE versus spline on the three feature metrics at both dimensions. |
| 8 | Distinguish **“VAE n=8 is the selected dashboard default”** from a claim that VAE significantly beats every model. CVAE is competitive; at n=3 its mean KS is numerically lower. |

Evidence: `src/confirmatory_spline.py`, the eight saved spline `result.json` files,
`src/context_query.py`, `src/features.py`, `production/course_visuals.py`,
`production/audit_2026_09_08/all_model_means.csv`, and `all_pairs_140.csv`.

The new personal-context slide's **27/28** means that 27 participants had lower
mean feature KS with their own center than with the population center (VAE n=8).
It is not identification accuracy or a count of individually significant tests.
