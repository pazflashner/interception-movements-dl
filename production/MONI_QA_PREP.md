# Questions Moni is likely to ask, and how to answer them

Built from three records: the 8 September feedback meeting, the 10 September
meeting with Jason, and the workshop presentation round where Moni questioned
the other two groups. In that round he is transcribed as "שמעון שחר" — the
speaker labelling is wrong, the questions are his.

His pattern across all three: he goes at the **metric** (is it the right one,
in what units, are all errors equally bad, where do they concentrate), at
**whether you understand your own pipeline**, and at **what a number is
relative to**. He accepts "we used AI to write the code" without complaint, but
not "we don't know exactly what it does".

Answers marked **GAP** are ones we cannot currently defend. Better to know that
now than to discover it in the room.

---

## A. The metric and its units

### A1. "What are the units of the MSE?"

He asked this explicitly on 8 September and it is written in his action items.

Squared centimetres. The deck says so on the slide 5 table heading, `MSE, cm²`.

We confirmed the unit from the data rather than assuming it: raw tracker
coordinates span about 25 units across the workspace, a reach measures about
13.5 and a path length about 14.2, while the apparatus records its own target
start positions as 120–160 mm. Millimetres would make the workspace 2.5 cm
across, which is impossible for a hand movement.

Note Jason believed in his meeting that it was "almost certainly not
centimetres (values too large)" and took the question away to check. The data
says centimetres. If he has since confirmed millimetres, our figure axes and
the cm² label all need revisiting.

### A2. "How exactly is the MSE computed for one trial?"

Per trial, the mean of 200 squared differences: the trajectory is 100 points on
normalised time in two dimensions, flattened, and compared with the
reconstruction coordinate by coordinate.

Two things to volunteer before he asks:

- Because x and y are each counted as a separate output, the number is **half**
  the mean squared point-to-point distance. It is a mean per coordinate, not
  per time point.
- It is therefore not a distance. Converted, `sqrt(2 × MSE)` gives an RMS miss
  of about **2.1 mm at n = 8** on a 130 mm reach, against **15.7 mm** for the
  condition-only floor.

The factor is identical for every model, so no comparison or p-value changes.

### A3. "Are all the errors equally important? Where are they concentrated?"

This is the question he pressed Nitzan on, and it transfers directly: our MSE
weights every time point and both axes equally, so a deviation at the start of
the movement counts the same as one at the interception point, and lateral
error counts the same as error along the direction of travel.

First, be able to say what the axes are. `src/trajectory_view.py` sets
`MODEL_AXES = (0, 1)` with names `x_lateral` and `y_forward`: the tracker
records x/y/z, the movement happens in the table plane, and the model keeps
lateral x and forward y while dropping height z. A trial is a forward reach of
about 13.5 cm with roughly 1 cm of side-to-side wander, so **y carries 98.5% of
the positional variance** (variance 20.58 against 0.32).

Honest answer on the error itself: **GAP**. We have not decomposed the error by
phase of the movement or by axis, and it must not be assumed from the variance
split — a quick unofficial check suggested the lateral axis may actually carry
the *larger* share of the reconstruction error at n = 8, the opposite of what
the variance ratio would lead you to guess. That check did not reproduce the
study's headline MSE, so no number from it is quotable.

What to say: the metric weights both axes and every time point equally, which
is a real limitation given that the two axes differ by a factor of forty in
variance; it is the same metric for every model, so the ranking stands; and the
decomposition is a short analysis we can run properly — error against normalised
time, and lateral versus forward separately.

### A4. "What is a good MSE? What is it relative to?"

He made this point himself on 8 September: nobody — not him, not Yoni, not
Jason — can say in advance what MSE counts as good; the comparison is always
relative.

We have an explicit reference: the **condition-only Ridge floor**, a model that
sees the task condition and nothing of the trajectory. It scores 1.2338 cm²,
against 0.0228 for Spline + PCA at n = 8 — a factor of 54 in squared error, or
7.4 in distance. That is the anchor that makes the other numbers mean
something, and it is a row in the slide 5 table.

---

## B. Do you understand your own pipeline?

### B1. "Is the B-spline basis fixed? Is the PCA basis the same one throughout?"

He asked exactly this on 8 September, was not satisfied with the answer, and
wrote it down as an action item: *"צריך לרדת לזה עד הסוף."* Expect it again.

Both are fixed, and it can be shown in the code:

- **The spline basis is fixed by construction.** Five interior knots at evenly
  spaced positions, cubic degree. It is not learned and has no free parameters:
  `fit_spline` in `src/baseline_spline.py` places knots with `np.linspace` and
  calls `splrep`. Every trial is projected onto the same basis, so one trial
  becomes 9 coefficients per axis, 18 in total.
- **The PCA basis is fitted on training participants only and then frozen.**
  `SplinePCARepresentation.fit` calls `PCA(n_components).fit(X)` on the training
  coefficient matrix and stores it as `self.pca_`; `encode` afterwards only
  calls `transform`. Test trials are projected through a basis fitted without
  them, which is what makes the comparison against the neural models fair.
- **The standardisation constants are frozen the same way** — `coef_mean_` and
  `coef_std_` come from training trials only.

So at inference a new trajectory becomes 18 coefficients and is then projected
through a fixed linear map to n numbers. Nothing is re-fitted per trial.

### B2. "You used Claude for the code. What did you actually verify yourself?"

He said explicitly that using AI is fine and that he does not expect us to write
every line — but that we must understand what was done, and not at high level
only.

Concrete things to name:

- The units question above, settled from the raw coordinates.
- The fingerprint formula: we caught that it was written as a mean of encoder
  posterior means, which only the neural families have. Spline + PCA has PCA
  scores and no posterior. The slide now uses a neutral per-trial code.
- The pipeline slide claimed the decoder had "2 timing branches". It has one
  trajectory head and one timing head (`src/vae_model.py:162`), which matches
  the laboratory draft, p. 6.
- The same slide credited all three neural families with 5 task conditions. The
  unconditional VAE — the model we recommend — has them zeroed
  (`src/vae_model.py:201`).
- A misattributed citation in the report: Rohrer & Hogan was cited as
  supporting the minimum-jerk method when the source cites it as what the
  implementation is **not**.

That list is the honest answer to "what did you check", and it is stronger than
any general claim of understanding.

---

## C. How the results are organised

### C1. "Are the prediction problems still mixed together?"

His single most repeated comment on 8 September.

They are separated. Slide 5 is reconstruction with its own metric (MSE in cm²).
Slide 8 is generation with its own metrics (mean KS, energy, MMD²). Slide 9 is
the fingerprint control. The combined table across all tasks lives in the
report's appendix, which is where he said it belongs.

The eight-page report follows the same split, one section and one table per
task.

---

## D. Interpreting the results

### D1. "Why are CVAE and the unconditional VAE almost identical?"

He offered two explanations himself and we should engage with both rather than
pick one.

The numbers: on generation at n = 8, unconditional VAE reaches mean KS 0.2151
and CVAE 0.2194 — the unconditional model is very slightly **ahead**. On
reconstruction CVAE is slightly ahead (0.0325 against 0.0335). The differences
are small either way, and VAE-versus-CVAE at n = 8 does not survive Holm on any
of the three generation metrics.

Against his hypothesis (a), that conditioning was not really implemented: we can
rule this out mechanically. The condition vector is five values — three one-hot
start categories, a binary side, and continuous target speed — concatenated to
both encoder and decoder inputs, and zeroed only when `use_condition=False`.
It reaches the network.

That leaves his hypothesis (b), a weak or uninformative condition, as the live
explanation, and it fits the wider picture: the generation protocol already
conditions on the participant through the fingerprint centre, so the task
condition adds little on top of knowing who is moving.

### D2. "So the classical method wins. Is that a problem?"

He answered this himself: it is common with small data, not surprising and not a
problem. Jason added a mechanistic reason — human movement is smooth, which is
exactly what a cubic spline represents efficiently.

The more interesting answer is that it does **not** win everywhere, and the
reversal is the finding: Spline + PCA has the lowest reconstruction error at
both dimensions, and the **highest** mean KS at n = 8 — worst of the four at
reproducing a person's distribution. Pointwise accuracy and distributional
realism are different objectives, and the linear projection is optimal by
construction for the first.

### D3. "Autoencoder better than VAE — what does that tell you?"

His own reading on 8 September: it means the task is compression, not sampling
around the representation, and that sits well with the PCA result.

Our numbers support it for reconstruction — CAE beats VAE at n = 8, 0.0260
against 0.0335 — and reverse it for generation, where VAE beats CAE, 0.2151
against 0.2587. That is consistent: the variational latent costs pointwise
accuracy and buys a latent you can sample from.

---

## E. The fingerprint and separation in latent space

### E1. "Is there really separation between people, and how do you quantify it?"

He asked for two specific things: a 3-D latent scatter coloured by participant,
and a linear classifier — logistic regression or Fisher LDA — to put a number on
the separation.

What exists: a logistic-regression attribution classifier
(`scripts/evaluate_final_fingerprints.py`), and latent scatter plots in the
dashboards. Jason saw the coloured scatter and said separation was visible but
the figure was hard to read on screen.

What we can state firmly is the **control**, which is stronger evidence than
separation alone: with the decoder, the noise and the conditions all frozen, we
replaced only the fingerprint centre. The participant's own centre gives mean KS
0.2151; the training-population mean gives 0.2841; another test participant's
centre gives 0.3144. 27 of 28 participants improve with their own fingerprint,
adjusted p < 1.5 × 10⁻⁶. So the representation is carrying person-specific
information, not a generic average movement.

**GAP**: a clean, presentable 3-D scatter is not in the deck, and we do not
report an LDA separation score.

### E2. "Did you check a naive baseline — simple hand-made parameters?"

His control suggestion: take a few invented movement parameters, such as mean
speed, and show that within-person variance is large, i.e. that these spaces do
not separate people. His point was that we often assume a person has a
fingerprint when it is not actually very distinctive.

Partial answer: we ran a related test — whether the latent adds anything beyond
direct summaries of the same context trials. Across 28 participants it did not:
0 of 28 cells significant, mean ΔR² of −0.017, with a positive control detected.
So direct measurement from a person's own trials beats latent regression for
predicting pointwise summaries. That is in the report's caveat, and it is an
honest partial answer to his question rather than a full one.

**GAP**: the specific naive-parameter control he described has not been run.

---

## F. Dimensionality

### F1. "Why 3 and 8? Why not 15 or 20?"

This is Jason's central methodological push and Moni made the same request
independently — try more dimensions to show what each extra number is worth.

Honest answer: **GAP**. The dimensions were chosen for comparability across
methods, not from a criterion. `config.LATENT_DIMS_SWEEP` covers 2, 3, 4 and 8.
No variance-explained cutoff or elbow analysis was performed, and nothing beyond
8 was tried.

What we can say: the improvement from 3 to 8 is large and consistent — spline
MSE 0.0973 → 0.0228, mean KS 0.3251 → 0.2548 — which is expected up to some
point, and Jason's own constraint for a usable model is 3. Jason's framing is
worth borrowing: if one method needs 3 dimensions where another needs 8 for the
same quality, that gap is itself the result.

The `explained_variance_ratio_` is already computed and returned by
`evaluate_spline_pca_baseline`, so a scree/elbow figure is a short job, not a
new experiment.

### F2. "Are you sure it is really compressed to 8 numbers?"

His action item: verify that what the model compresses genuinely passes through
n numbers.

Yes, and it is checkable in the architecture. The neural encoder produces an
n-dimensional mean and log-variance, and the decoder receives only z plus the
condition — there is no skip connection. For Spline + PCA the code is literally
n PCA scores, and reconstruction runs inverse PCA then multiplies by the spline
basis. The condition-only floor at 1.2338 shows the conditions alone carry
almost nothing, so the reconstruction quality is coming through the n numbers.

---

## G. Presentation-level questions

### G1. "Which method should Jason actually use?"

Both Jason and Moni want a plain recommendation. Slide 10 gives it: Spline + PCA
for compact reconstruction of recorded trajectories, VAE at n = 8 for generating
a participant's movement distribution, with the caveat that direct measurement
from a person's own context trials still beats latent regression for predicting
single summary numbers.

### G2. "You are showing corrected p-values — corrected against what family?"

140 tests: 5 models, all 10 pairs, at 2 latent dimensions, across 7 metrics.
Both BH and Holm are reported. Worth volunteering that this is a conservative
choice — the slides show 6 of the 140, and correcting against the full family is
what turns one comparison (Spline versus CAE at n = 8) from significant under BH
into non-significant under Holm.

---

## Summary of the gaps, in the order he is most likely to hit them

1. No principled criterion for the latent dimension, and nothing tried beyond 8.
2. No error decomposition — where in the movement, and which axis.
3. No presentable latent scatter or LDA separation score in the deck.
4. The naive hand-made-parameter control was not run.
5. No per-trial confidence for subject attribution (Jason's ask, not Moni's).

Each of these is a short analysis rather than a new experiment. If there is time
before the talk, 1 and 2 are the two that are most likely to be asked and the
cheapest to close.
