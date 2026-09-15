"""Slide content for the styled deck: the Hebrew reference deck, translated.

Structure and wording follow the reference presentation one-for-one, so the two
decks can be compared side by side. Numbers were checked against the same
evidence the PowerPoint uses (production/course_evidence.py) and the laboratory
draft; where the reference showed a LaTeX artefact such as "$\\rightarrow$" the
intended symbol is written instead.

Each slide is a dict. Recognised blocks, rendered in the order listed here:
    body       str                      lead paragraph under the rule
    columns    {"left": [...], "right": [...]}   two-column layout
    table      {"rows": [...], "mark": row index highlighted green}
    cards      [{"eyebrow","title","text"}]      2-3 equal cards
    statcard   {"value","title","text"}          big green number panel
    equation   str                               boxed formula
    terminal   {"label","badge","command"}       dark command strip
    callout    {"kind": info|warn, "title", "text"}
    figure     path relative to the repository root
"""

TITLE = {
    "title": "Compact Representations of Human Interception Movements",
    "subtitle": "Low-dimensional models for representing and generating human interception movement",
    "authors": "Seman Libbiss  &  Paz Flashner",
    "affiliation": [
        "Research supervision: Prof. Jason Friedman   |   Course advisor: Moni Shahar",
        "Workshop on Deep Learning, Tel Aviv University",
    ],
    "badge": "Planned talk: 10 minutes  (11 core slides + backup slides)",
}

SLIDES = [
    {
        "title": "The experimental task and the data structure",
        "seconds": 50,
        "columns": {
            "right": [
                ("heading", "The experimental task"),
                ("bullet", "Participant's task:",
                 "move the finger from the lower start zone into the upper interception "
                 "square exactly as the ball passes through it."),
                ("bullet", "Conditions:", "the free eye-movement condition is analysed. Six target "
                                          "start locations (3 right, 3 left) over a continuous speed range."),
                ("bullet", "Sample:", "4,732 valid trials from 28 participants, measured in 3D at "
                                      "240 Hz with a Polhemus system."),
            ],
            "left": [("figure", "production/presentation_assets/task_schematic.png")],
        },
        "callout": {
            "kind": "info",
            "title": "A strict split that prevents information leakage",
            "text": "Four fixed folds: 17 training participants, 4 validation and 7 test. "
                    "Every participant appears in a test fold exactly once.",
        },
    },
    {
        "title": "What we set out to answer",
        "seconds": 40,
        "columns": {
            "ratio": 0.56,
            "left": [
                ("heading", "The core question"),
                ("text", "Human movement is stochastic: perception and motor control "
                         "vary from trial to trial. One participant, one target - and "
                         "every trial takes a different path."),
                ("heading-alt", "Can a compact fingerprint capture that spread?"),
                ("text", "Compress 200 numbers - the 100 × 2 path - into a 3-to-8 "
                         "dimensional fingerprint of an individual's movement "
                         "distribution."),
                ("equation", "100 × 2 coordinates   ———>   "
                             "n = 3 or 8 latent numbers   (durations predicted)"),
            ],
            "right": [
                ("figure", "production/presentation_assets/real_trajectories.png", 2.6),
            ],
        },
        "cards_heading": "What would count as success, evaluated on participants the "
                         "model never trained on",
        "cards": [
            {"eyebrow": "SUCCESS TEST 1", "title": "Reconstruct",
             "text": "Rebuild specific recorded trajectories from the compressed code."},
            {"eyebrow": "SUCCESS TEST 2", "title": "Generate",
             "accent": "green",
             "text": "Sample new trajectories matching that person's stochastic "
                     "distribution."},
            {"eyebrow": "SUCCESS TEST 3", "title": "Stay personal",
             "text": "The fit must come from that person's own fingerprint, not a "
                     "generic one."},
        ],
    },
    {
        "title": "Trajectory processing and the compression architectures",
        "seconds": 65,
        "table": {
            "rows": [
                ["Stage in the model", "Spline + PCA  (linear baseline)",
                 "Neural models  (VAE / CVAE / CAE)"],
                ["Primary input", "The same 100 × 2 normalised trajectory",
                 "The same 100 × 2 trajectory; CVAE and CAE add 5 task conditions"],
                ["Where standardisation happens", "On the 18 spline coefficients, after fitting",
                 "On the 200 coordinates, with train-set constants"],
                ["Encoder / compression", "Cubic B-spline fit, 9 coefficients per axis -> PCA to n latents",
                 "Two fully connected 256-unit ReLU layers -> latent n"],
                ["Decoder / reconstruction", "Inverse PCA -> multiply by the spline basis for the full path",
                 "Two 256-unit ReLU layers -> separate heads for 200 coordinates and 2 durations"],
                ["Timing prediction", "Separate Ridge regression on the PCA scores + conditions",
                 "Dedicated head predicting initiation and movement time"],
            ],
        },
        "callout": {
            "kind": "info",
            "title": "Three neural variants, one architecture",
            "text": [
                ("VAE - variational autoencoder, trained on the trajectory alone. "
                 "CVAE - conditional variational autoencoder, which also receives the "
                 "task conditions. CAE - conditional autoencoder, the same network "
                 "without the variational latent.", False),
                ("All are compared at identical latent dimensions, n = 3 (extreme "
                 "compression) and n = 8 (extended capacity).", False),
            ],
        },
    },
    {
        "title": "Task 1: Trajectory Reconstruction",
        "seconds": 45,
        "columns": {
            "ratio": 0.62,
            "left": [
                ("figure", "production/course_assets/reconstruction_slide.png", 3.6),
                ("text", "One median-duration trial. Black: recorded; dashed: "
                         "reconstructed. Both axes in cm at equal scale."),
            ],
            "right": [
                ("heading", "Trajectory reconstruction error (MSE, cm²)"),
                ("table", {
                    "rows": [
                        ["Model", "MSE  n=3", "MSE  n=8"],
                        ["Spline + PCA", "0.0972", "0.0228"],
                        ["VAE (Unconditional)", "0.1483", "0.0335"],
                        ["Conditional AE (CAE)", "0.1697", "0.0260"],
                        ["CVAE", "0.1770", "0.0325"],
                        ["Condition-only Ridge (Floor)", "1.2338", "1.2338"],
                    ],
                    "mark": 1,
                }),
                ("callout-info", "Key insight on pointwise reconstruction", [
                    ("PCA wins at reconstructing a single trajectory.", True),
                    ("As a linear projection, PCA is analytically optimal for minimising squared error among all linear methods. VAEs trade off pointwise accuracy to obtain a continuous and samplable latent space.", False),
                ]),
            ],
        },
    },
    {
        "title": "Task 2: generating a personal movement distribution",
        "seconds": 60,
        "columns": {
            "left": [
                ("heading", "How do we generate movements from a fingerprint?"),
                ("bullet", "Enrollment:", "earlier context trials of the participant, "
                                          "whose mean defines the personal fingerprint centre."),
                ("bullet", "Stochastic sampling:", "draw 120 latent vectors around that "
                                                   "centre using a shared covariance learned "
                                                   "from training data."),
                ("bullet", "Unseen movements:", "query trials are held aside to evaluate "
                                                "the distribution and are never exposed to "
                                                "the model."),
                ("text", "Below, n = 8. Thin: first 30 draws. Thick: their mean. "
                         "Equal x/y scale, cm."),
            ],
            "right": [
                ("equation", "fingerprint centre:   θ_s  =  (1 / |C_s|) Σ c_i"),
                ("equation", "one generated movement:   z  ∼  "
                             "N ( θ_s ,  Σ_train )"),
                ("text", "c_i is the trial's code: the encoder mean μ_i for the "
                         "neural families, the PCA scores for spline + PCA."),
            ],
        },
        "figure": ("production/course_assets/generation_slide_n8.png", 2.6),
    },
    {
        "title": "How we score a generated distribution",
        "seconds": 30,
        "cards": [
            {"eyebrow": "WHAT WE MEASURE", "title": "11 kinematic features",
             "text": "Initiation and movement time, peak speed, time to peak, path "
                     "length, straight-line distance, curvature, maximum lateral "
                     "deviation and the three endpoint coordinates."},
            {"eyebrow": "HOW WE COMPARE", "title": "Two views of the distribution",
             "text": "One KS test per feature, averaged over the 11 into Mean KS. "
                     "Energy discrepancy and MMD squared judge the joint structure of "
                     "all 11 at once. Lower is better throughout."},
        ],
        "callout": {
            "kind": "info",
            "title": "One fixed yardstick for every model",
            "text": "Feature scales and the RBF bandwidth are fitted on training "
                    "participants only and then frozen, so every model is measured "
                    "against the same geometry rather than one tuned to its own output.",
        },
    },
    {
        "title": "Generation results: VAE leads on movement distributions",
        "seconds": 55,
        "figure": ("production/presentation_assets/mean_ks_chart.png", 3.6),
        "callout": {
            "kind": "info",
            "title": "A reversal in the ranking: VAE wins at generation",
            "text": [
                ("Although Spline + PCA won pointwise reconstruction, VAE and CVAE "
                 "achieve much better distribution matching (corrected p < 0.001 under "
                 "BH and Holm). VAE at n = 8 was set as the dashboard default.", False),
                ("The two joint measures agree: at n = 8, VAE reaches energy 0.4110 and "
                 "MMD squared 0.0639, against 0.7347 and 0.1153 for Spline + PCA.", False),
            ],
        },
    },
    {
        "title": "Validating the personal fingerprint  (Control Test)",
        "seconds": 45,
        "columns": {
            "right": [
                ("heading", "Does the personal fingerprint really matter?"),
                ("text", "To verify that the model is not merely producing a generic average "
                         "movement, we replaced the fingerprint while the decoder, the noise and "
                         "the conditions stayed completely frozen:"),
                ("table", {
                    "rows": [
                        ["Injected fingerprint centre", "CVAE n=8 (Mean KS)", "VAE n=8 (Mean KS)"],
                        ["The participant's own centroid", "0.2194", "0.2151"],
                        ["Training-population mean", "0.2896", "0.2841"],
                        ["Another test participant", "0.3172", "0.3144"],
                    ],
                    "mark": 1,
                }),
            ],
            "left": [
                ("stat", "27 / 28", "participants improved markedly",
                 "Using the personal fingerprint improved distribution matching against generic "
                 "attribution (p < 1.5 x 10^-6)."),
            ],
        },
    },
    {
        "title": "Recommended summary: the trade-off between reconstruction and generation",
        "seconds": 45,
        "cards": [
            {"eyebrow": "To reconstruct known trajectories", "title": "Spline + PCA",
             "text": "Recommended when the goal is compact compression and the most accurate "
                     "pointwise reconstruction of a measured trajectory (MSE = 0.0228)."},
            {"eyebrow": "For generation and simulation", "title": "VAE  (n = 8)", "accent": "green",
             "text": "Recommended for simulating a participant's future movements. Produces the "
                     "kinematic distributions closest to reality (MMD squared = 0.0639)."},
        ],
        "callout": {
            "kind": "warn",
            "title": "Key caveat",
            "text": "Direct measurement from the participant's own context data still beats latent "
                    "regressions at predicting pointwise summary measures. The compact "
                    "representation is meant to generate whole trajectories, not a single number.",
        },
    },
    {
        "title": "The applied product: an interactive dashboard",
        "seconds": 80,
        "cards": [
            {"eyebrow": "1", "title": "Model and dimension",
             "text": "Supports CAE, CVAE, VAE and Spline + PCA at dimensions 2, 3, 4 and 8."},
            {"eyebrow": "2", "title": "Fingerprint selection",
             "text": "Slider control over the latent coordinates and real-time simulation of the "
                     "resulting trajectory."},
            {"eyebrow": "3", "title": "Distribution sampling",
             "text": "Generate 25 to 500 trajectories and compare them against the participant's "
                     "real data."},
        ],
        "terminal": {
            "label": "Run locally in a terminal:",
            "badge": "Streamlit App",
            "command": "python -m streamlit run src/confirmatory_dashboard.py --server.port 8510",
        },
    },
]

BACKUP = [
    {
        "title": "Backup: resampling and the two time windows",
        "body": [
            "Model input: target motion to final tracker sample.",
            "Feature input: detected finger onset to end.",
            "A phase of 0.5 means half the selected time interval, not half "
            "the path length.",
        ],
        "figure": ("production/presentation_assets/equation_resample.png", 1.5),
        "callout": {
            "kind": "info",
            "title": "",
            "text": "100 samples are a common representation, not 100 "
                    "independent observations.",
        },
        "notes": "We start with a regularized 240 Hz grid, smooth using a "
                 "fourth-order 10 Hz forward-backward Butterworth filter "
                 "and select the modeled window. We fit a cubic interpolant "
                 "through the existing filtered points and evaluate it at "
                 "j/99. This interpolation is different from fitting the "
                 "later five-knot compression model. Subtracting the first "
                 "position aligns starts, but does not scale amplitude or "
                 "rotate paths. We retain initiation and movement seconds "
                 "as separate targets. Early movement after target motion "
                 "remains; the earlier appearance interval is outside the "
                 "modeled window. Source: src/preprocessing.py, "
                 "src/trajectory_view.py; appendix A1.",
    },
    {
        "title": "Backup: what becomes the PCA matrix?",
        "table": {
            "rows": [
                ["Object", "Dimensions", "One row / one column means"],
                ["Sampled trial", "100 × 2", "Time point / x or y"],
                ["Spline basis B", "100 x 9", "Time point / basis function"],
                ["Trial coefficient vector", "18", "Nine x coefficients followed by nine y"],
                ["Training coefficient matrix", "N training x 18", "Trial / coefficient"],
                ["PCA directions V", "18 x n", "Coefficient / retained component"],
                ["Training latent codes", "N training x n", "Trial / latent coordinate"],
            ],
        },
        "callout": {
            "kind": "info",
            "title": "",
            "text": "Five interior knots + degree 3 + 1 = nine coefficients "
                    "per axis.",
        },
        "notes": "Moni asked what we feed into PCA. It is a matrix where "
                 "each row is one trial's eighteen spline coefficients. We "
                 "first fit a spline separately to x and y of each trial. "
                 "Nine x and nine y coefficients give eighteen numbers. We "
                 "stack training trials and PCA learns n directions in that "
                 "eighteen-dimensional space. Neither test trials nor "
                 "validation trials train those PCA directions. The number "
                 "of knots and the number of latent coordinates are "
                 "different controls. Source: src/baseline_spline.py, "
                 "CompactSplineRepresentation; appendix A2.",
    },
    {
        "title": "Backup: fitting and decoding spline + PCA",
        "body": [
            "Fit the two coordinate coefficient vectors by least squares.",
            "Validation selects raw or standardized coefficients.",
            "Inverse PCA gives coefficients; multiplying by B restores the "
            "path.",
        ],
        "figure": ("production/presentation_assets/equation_pca.png", 1.5),
        "callout": {
            "kind": "info",
            "title": "",
            "text": "m and d combine centering/scaling; V contains retained "
                    "PCA directions. Raw PCA uses d = 1.",
        },
        "notes": "There are two compression steps: sampled path into smooth "
                 "coefficients, then coefficients into a smaller PCA code. "
                 "Fit B c_x to x and B c_y to y. The code z is obtained by "
                 "centering/scaling the coefficient vector and projecting "
                 "onto retained axes. To decode, apply inverse PCA and "
                 "inverse scaling, split the eighteen coefficients into two "
                 "blocks and multiply both by the basis B. Raw PCA still "
                 "centers columns. PCA optimizes variance in coefficient "
                 "space and does not guarantee the best trajectory MSE "
                 "against all methods. Sources: src/baseline_spline.py; "
                 "appendix A2. The on-slide equation matches the effective "
                 "forward and inverse transformations.",
    },
    {
        "title": "Backup: predicting physical timing",
        "table": {
            "rows": [
                ["Stage", "Spline + PCA", "Neural models"],
                ["Predictors", "n PCA scores + 5 conditions", "Latent code + conditions where enabled"],
                ["Targets", "Two standardized log(t + .001) values", "Same two standardized targets"],
                ["Predictor", "Ridge, alpha selected on validation", "Separate decoder output head"],
                ["Inverse", "Undo scaling, exponentiate, subtract .001", "Same inverse transform"],
            ],
        },
        "callout": {
            "kind": "info",
            "title": "",
            "text": "Observed timing never enters the trajectory encoder. "
                    "Reconstruction-time timing prediction sees the "
                    "observed shape.",
        },
        "notes": "The coordinate path is on normalized phase, so the model "
                 "must predict seconds separately. Initiation is time from "
                 "target motion to finger onset, movement time is onset to "
                 "end. We take log(t+.001) to accommodate zeros, "
                 "standardize training targets, and reverse these "
                 "transformations after prediction. Spline Ridge receives "
                 "the code and five conditions, with training input "
                 "standardization and alpha from .01, .1, 1, 10, 100 "
                 "selected using validation error on standardized log "
                 "targets. Neural models have a two-value timing head. Huge "
                 "positive timing errors were not upper-clipped out of the "
                 "main scores. Source: src/confirmatory_spline.py, "
                 "TimingRidge; src/vae_model.py; appendix A3.",
    },
    {
        "title": "Backup: the neural training objective",
        "body": [
            "Coordinate error sums 200 squared differences.",
            "Timing error sums two squared differences and receives weight "
            "20.",
            "KL sums across latent dimensions; the batch averages the "
            "total.",
        ],
        "figure": ("production/presentation_assets/equation_loss.png", 1.5),
        "callout": {
            "kind": "info",
            "title": "",
            "text": "Replacing both summed terms by separate MSE means "
                    "would change the effective timing weight.",
        },
        "notes": "The actual code uses reduction=none, then sum across "
                 "output coordinates and mean across trials. We must not "
                 "simply write MSE plus 20 MSE because one output has two "
                 "hundred coordinates and the other has two. KL measures "
                 "how far the encoder's Gaussian posterior is from the "
                 "standard normal. Beta warms from zero to one over fifty "
                 "epochs, while checkpoint validation uses the final "
                 "beta-one objective. Adam learning rate .001, batch size "
                 "64, gradient norm clipping at five, patience 25, maximum "
                 "150 epochs, three seeds. Source: src/vae_model.py "
                 "vae_loss; src/train.py; appendix A3.",
    },
    {
        "title": "Backup: VAE, CVAE and CAE",
        "table": {
            "rows": [
                ["Property", "VAE", "CVAE", "CAE"],
                ["Trajectory encoder", "Two hidden layers", "Two hidden layers", "Two hidden layers"],
                ["Task inputs", "Masked to zero", "Encoder and decoder", "Encoder and decoder"],
                ["Latent training code", "Gaussian sample", "Gaussian sample", "Deterministic"],
                ["KL penalty", "Yes", "Yes", "No"],
                ["Reconstruction code", "Posterior mean", "Posterior mean", "Deterministic code"],
            ],
        },
        "callout": {
            "kind": "info",
            "title": "",
            "text": "A nonsignificant conditional-model advantage does not "
                    "mean the experimental conditions are irrelevant.",
        },
        "notes": "Our VAE is an unconditional ablation in the same code "
                 "framework. It does not receive meaningful conditions "
                 "because we zero them at both ends. CVAE uses actual "
                 "conditions. CAE uses actual conditions but suppresses "
                 "probabilistic sampling and KL. The condition vector is "
                 "three one-hot category entries, a side value (left zero, "
                 "right one), and normalized executed speed. We checked "
                 "input routes and shuffled conditions. The routes are "
                 "active, but these results cannot settle whether another "
                 "architecture or training regime would use conditions "
                 "better. Source: src/vae_model.py "
                 "encode_condition/encode_trial_condition; src/train.py; "
                 "audit records.",
    },
    {
        "title": "Backup: where the generated variation comes from",
        "body": [
            "The personal center averages context codes.",
            "Training residuals estimate within-person latent variation.",
            "Variational models also add average posterior variance.",
        ],
        "figure": ("production/presentation_assets/equation_covariance.png", 1.5),
        "callout": {
            "kind": "info",
            "title": "",
            "text": "One shared covariance for every participant; 120 draws "
                    "around each personal center.",
        },
        "notes": "For each training person, subtract that person's mean "
                 "code from each trial code. Pool these residuals and "
                 "estimate their covariance. Add the mean encoder posterior "
                 "variance on the diagonal for VAE/CVAE and add a tiny "
                 "identity regularizer for numerical stability. CAE and "
                 "spline have no posterior variance term. We do not fit a "
                 "separate covariance to a new participant's context "
                 "trials. Generation draws a multivariate normal around "
                 "their context center and uses the shared decoder. A "
                 "nonlinear decoder can still produce different visible "
                 "spread at different centers. Source: src/context_query.py "
                 "and src/confirmatory_spline.py; appendix A4.",
    },
    {
        "title": "Backup: leakage and held-out evaluation",
        "table": {
            "rows": [
                ["Information", "Training", "Validation", "Test context", "Test query"],
                ["Fit network/PCA/scalers", "Yes", "No", "No", "No"],
                ["Select checkpoint/settings", "No", "Yes", "No", "No"],
                ["Personal center", "No", "No", "Yes", "No"],
                ["Reconstruction input", "No", "No", "No", "Observed path"],
                ["Generation task mixture", "No", "No", "No", "Metadata only"],
                ["Generation scoring", "No", "No", "No", "Paths and timings"],
            ],
        },
        "callout": {
            "kind": "info",
            "title": "",
            "text": "Population model fitting and participant enrollment "
                    "are separate stages.",
        },
        "notes": "A test person is new to model fitting, but provides "
                 "context recordings. Query recordings only provide the "
                 "targets for generation scores, with task metadata allowed "
                 "for a matched-condition mixture. Reconstruction "
                 "deliberately reads the query path because that task is "
                 "compression. None of the test data changes fitted PCA "
                 "axes, learned neural parameters, training covariance or "
                 "normalization. This prevents a common misunderstanding: "
                 "test context is not retraining. Model selection for the "
                 "final dashboard still used the evaluated cohort, so "
                 "independent post-selection validation remains future "
                 "work. Source: participant_folds.json, context_query.py; "
                 "appendices A1/A4.",
    },
    {
        "title": "Backup: what the eleven features measure",
        "table": {
            "rows": [
                ["Group", "Features"],
                ["Timing", "Initiation time; movement time; relative time to peak speed"],
                ["Speed", "Peak speed; heuristic speed-peak count"],
                ["Geometry", "Path length; straight-line distance; curvature ratio; maximum lateral deviation"],
                ["Endpoint", "Final x; final y"],
            ],
        },
        "callout": {
            "kind": "info",
            "title": "",
            "text": "Generated feature computation uses predicted timing "
                    "and an onset-based movement crop.",
        },
        "notes": "For observed data we use the onset-to-end movement "
                 "segment. For generated data we use the predicted "
                 "initiation fraction to crop the go-to-end path, capped at "
                 ".98, and floor movement duration at one millisecond. Both "
                 "movement segments become one hundred phase points and are "
                 "filtered with cutoff min(10 Hz, .45 times sample rate). "
                 "Sample rate is 99 over movement seconds. Gradients give "
                 "speed. Peak count uses prominence ten percent of maximum "
                 "and roughly fifty milliseconds separation, with at least "
                 "one reported peak. This is distinct from minimum-jerk "
                 "component fitting. Source: src/features.py; appendix A5.",
    },
    {
        "title": "Backup: KS and joint distribution distances",
        "body": [
            "KS: largest gap between two empirical cumulative curves.",
            "Mean KS: average of eleven feature statistics.",
            "Energy and MMD: compare whole vectors and account for joint "
            "structure.",
        ],
        "figure": ("production/presentation_assets/equation_ks.png", 1.5),
        "callout": {
            "kind": "info",
            "title": "",
            "text": "KS statistics are not p-values. Correlated features "
                    "can affect joint geometry.",
        },
        "notes": "If a recorded feature has seventy percent of trials below "
                 "a value and the generated feature has forty percent, the "
                 "empirical CDF gap there is .3. KS takes the largest gap "
                 "over all thresholds. Its output is a discrepancy. We "
                 "average discrepancies across features, not significance "
                 "tests. Energy and MMD use complete feature vectors and "
                 "can notice dependence between features that separate "
                 "marginal KS statistics may miss. Standardization comes "
                 "from training. Correlated features do not mean we should "
                 "remove whichever feature makes a p-value improve; removal "
                 "defines a new endpoint. Sources: src/context_query.py; "
                 "appendix A5/A6; Gretton et al. JMLR 2012.",
    },
    {
        "title": "Backup: energy discrepancy and mean collapse",
        "body": [
            "Energy compares cross-set distances with within-set distances.",
            "A generator needs plausible variation as well as a good "
            "center.",
        ],
        "figure": ("production/presentation_assets/equation_energy.png", 1.5),
        "table": {
            "rows": [
                ["Example reference", "Mean-path RMSE", "Path energy"],
                ["VAE n=8", "0.1604", "0.0548"],
                ["Repeated context mean", "0.1146", "0.4621"],
            ],
        },
        "callout": {
            "kind": "info",
            "title": "",
            "text": "Lower is better. The mean-only output has a closer "
                    "mean but a much worse distribution.",
        },
        "notes": "Average error against all recorded paths can favor a "
                 "collapsed output near the center. Energy subtracts "
                 "within-generated and within-recorded distances from twice "
                 "the cross average, addressing the distributions rather "
                 "than only their centers. Our energy estimator includes "
                 "within-sample diagonal zeros and is not square-rooted "
                 "afterward. The mean-only reference is actual analysis "
                 "output, not a hypothetical numerical illustration. "
                 "Source: src/trajectory_distribution.py; "
                 "trajectory_distribution summary CSV; Szekely and Rizzo "
                 "2013; appendix A6/A8.",
    },
    {
        "title": "Backup: Gaussian-kernel MMD",
        "body": [
            "Kernel: exp(-gamma x squared Euclidean distance).",
            "Gamma comes from the median positive training pair distance.",
            "Within-set diagonals are excluded in the unbiased estimator.",
        ],
        "figure": ("production/presentation_assets/equation_mmd.png", 1.5),
        "callout": {
            "kind": "info",
            "title": "",
            "text": "Finite-sample unbiased MMD squared may be slightly "
                    "negative. It is not an accuracy percentage.",
        },
        "notes": "The kernel gives high similarity to close vectors and low "
                 "similarity to distant vectors. MMD compares average "
                 "within-recorded and within-generated similarity with "
                 "cross similarity. The within terms omit identical-element "
                 "pairs and divide by m(m-1) or r(r-1); cross divides by "
                 "mr. We estimate gamma once from at most 512 deterministic "
                 "training trials in each representation and use it for all "
                 "models in the rotation. Whole paths and feature vectors "
                 "have different units and scales, so their absolute MMD "
                 "values are not interchangeable. Source: "
                 "src/trajectory_distribution.py; src/context_query.py; "
                 "Gretton et al., A Kernel Two-Sample Test, JMLR 13 (2012).",
    },
    {
        "title": "Backup: 28 paired scores, not thousands of independent "
                 "trials",
        "body": [
            "First average neural seeds within each participant.",
            "Then compare paired participant scores using two-sided Pratt "
            "Wilcoxon.",
            "Round differences to 12 decimals before ranking.",
        ],
        "table": {
            "rows": [
                ["Correction family", "Tests"],
                ["Original seven endpoints x ten pairs x two dimensions", "140"],
                ["New two path endpoints x ten pairs x two dimensions x two geometries", "80"],
                ["VAE replacement controls", "6"],
                ["CVAE replacement controls", "12"],
            ],
        },
        "callout": {
            "kind": "info",
            "title": "",
            "text": "BH controls false discoveries under its assumptions. "
                    "Holm is a familywise-error sensitivity. "
                    "Nonsignificance is not equivalence.",
        },
        "notes": "Each participant supplies one score per model for the "
                 "paired comparison. Three seeds are repeated model fits, "
                 "not three independent people. Wilcoxon ranks the absolute "
                 "paired differences and uses their signs. Pratt keeps "
                 "zeros in ranking but omits their signed contribution. The "
                 "usual location interpretation assumes a symmetric "
                 "difference distribution. BH adjusts the complete declared "
                 "family, not only the comparisons we liked. We "
                 "additionally show Holm. Training folds overlap, so these "
                 "remain exploratory comparisons, and a fresh cohort is "
                 "still needed for replication. No formal "
                 "three-versus-eight or resampling-reference test was "
                 "added. Source: src/statistical_tests.py, "
                 "all_pairs_140.csv, paired_comparisons.csv; appendix A6.",
    },
    {
        "title": "Backup: the boundaries of the conclusion",
        "body": [
            "Useful same-session personal centers do not establish stable "
            "traits.",
            "Direct context summaries beat latent linear probes on all 14 "
            "tested targets.",
            "Condition routes work, but their value depends on the task and "
            "model.",
            "Minimum-jerk decomposition remains a separate investigation.",
        ],
        "callout": {
            "kind": "info",
            "title": "",
            "text": "The 31-page laboratory report retains the extended "
                    "results and Jason's open questions.",
        },
        "notes": "We should volunteer the relevant limitation instead of "
                 "defending every result as a success. Direct context means "
                 "and standard deviations were more accurate than our "
                 "latent linear probes, and many probe R-squared values "
                 "were negative. Matched minimum-jerk component comparisons "
                 "did not establish corrected model differences. Timing and "
                 "event assumptions remain important, including the tracker "
                 "endpoint arriving on average 26 ms before the MATLAB "
                 "arrival marker. More data might help, but we did not run "
                 "a learning curve that proves sample count caused the "
                 "failures. Source: frozen report pp.16-28, "
                 "production/all_models_direct_context_summary.csv; course "
                 "Discussion and appendix A10.",
    },
    {
        "title": "Backup: launch and source navigation",
        "body": [
            "Repository terminal:",
            "python -m streamlit run src/confirmatory_dashboard.py "
            "--server.port 8510",
            "Browser: http://127.0.0.1:8510",
            "Stop: Ctrl+C in that terminal",
        ],
        "callout": {
            "kind": "info",
            "title": "",
            "text": "The standalone dashboard archive contains launch "
                    "instructions for use without the full research "
                    "checkout.",
        },
        "notes": "Full PowerShell path: cd "
                 "\"D:\\oneDrive_Main\\OneDrive\\Desktop_onedrive\\uni2024\\Third "
                 "Year\\semester B\\Workshop in "
                 "AI\\interception-movements-dl\". Then use the command "
                 "shown, adding --server.address 127.0.0.1 "
                 "--browser.gatherUsageStats false if desired. On the "
                 "authoring computer Python is C:\\Python313\\python.exe. "
                 "Source navigation: src/vae_model.py and src/train.py for "
                 "neural models; src/baseline_spline.py and "
                 "src/confirmatory_spline.py for spline/PCA/Ridge; "
                 "src/context_query.py and src/features.py for "
                 "generation/metrics; src/trajectory_distribution.py for "
                 "full paths; production/audit_2026_09_08 for original "
                 "result CSVs; "
                 "production/trajectory_distribution_2026_09_13 for new "
                 "results. Keep the terminal running during the demo.",
    },
]
