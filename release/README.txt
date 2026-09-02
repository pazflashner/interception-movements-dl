INTERCEPTION MOVEMENT STRATEGY EXPLORER
=======================================

This bundle contains the frozen strategy-window dashboard and its reference
outputs. It does not contain the raw participant data.

Requirements
------------
- Windows, macOS, or Linux
- Python 3.11 or newer
- Approximately 1 GB free for the Python packages

Setup
-----
Open a terminal in this folder and run:

    python -m pip install -r requirements_dashboard.txt
    python -m streamlit run src/confirmatory_dashboard.py

Streamlit will print a local URL, normally http://localhost:8501.

Dashboard scope
---------------
- Strategy window only: MAT target motion onset to the last tracker sample
  (arrival proxy). Marker 5 indicates appearance, not motion.
- Live reference CVAE: outer fold 0, seed 42, n=2,3,4,8.
- Held-out generated validation: n=3 and n=8 across all four participant
  folds; each of the 28 participants appears in test once.
- Benchmark values aggregate the frozen four-fold, three-seed study using
  corrected training-reference feature distances. Main checkpoints are unchanged.
- Diagnostics include behavioural probes (including negative results), common
  timing-head comparisons, the pre-go audit and a finite-sample submovement reference.
- Task-condition sliders are exploratory: the confirmatory diagnostic did not
  establish participant-specific conditioning benefit.
- Minimum-jerk component count is model-order sensitive and must not be read
  as a validated cognitive-strategy label.

Included reports
----------------
- Interception_Movements_Final_Scientific_Report.pdf: concise scientific report.
- Interception_Movements_Results_Guide.pdf: matched interpretation guide for
  Simaan and Paz.

The dashboard's Protocol & downloads tab can also download both reports and
the confirmatory model-comparison table.

The evaluation/ folder contains full corrected comparison tables and additional
audit/sensitivity results. The original training matrix and raw recordings are
not included. The guide is intended for the students; the scientific report is
the advisor-facing document. Source snapshots of the upstream licence are in
licenses/; see THIRD_PARTY_NOTICES.txt.
