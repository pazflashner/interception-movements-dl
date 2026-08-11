INTERCEPTION MOVEMENT FINGERPRINT DASHBOARD

QUICK START (WINDOWS)
1. Extract this ZIP file.
2. Double-click setup_and_launch.bat the first time.
3. On later runs, double-click launch_dashboard.bat.
4. Streamlit opens the dashboard in the default browser.

The first setup installs Python packages and can take several minutes. Python 3.11 or newer is recommended.

DASHBOARD CONTENTS
- Generate: n=2, n=3, or n=8; latent sliders; task-condition controls; trajectory, velocity, timing, and minimum-jerk outputs.
- Distribution check: held-out recorded versus generated distributions with KS/Wasserstein or JSD/total variation.
- Model comparison: the repeated-seed study results.
- Protocol and downloads: assumptions, questions, PDF, and compact tables.

SCIENTIFIC SCOPE
- The models use the x-y table plane and condition 2 only.
- True timing is withheld from the encoder and predicted by the decoder.
- Participant presets use context trials; comparisons use disjoint query trials.
- n=8 gives the strongest overall fidelity. n=2 and n=3 provide lower-dimensional controls but do not reproduce every distribution.
- Minimum-jerk components are kinematic patterns, not direct cognitive-strategy labels.

DATA INCLUDED
This bundle contains trained model checkpoints, generated validation samples, held-out query feature summaries, and aggregate results. It contains no raw Dropbox trajectory CSVs or identifying participant information.
