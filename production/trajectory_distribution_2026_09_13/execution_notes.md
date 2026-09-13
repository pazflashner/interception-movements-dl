# Numerical reproduction note

The initial execution stopped when Condition Ridge fold 1 / seed 44 / subject08
failed the feature-reproduction gate with a maximum discrepancy of
0.00020536714364016717. This exactly reproduced the already documented one-thread
BLAS sensitivity in `production/audit_2026_09_08/AUDIT_REPORT.md` and
`diagnose_ridge_precision.py`: a rank-deficient alpha=0 condition-Ridge fit.

The runner now fits and samples every Condition Ridge fold with two numerical
threads, the earlier audit's setting that reproduced the original exported
scores. It keeps the same tolerance, historical model settings and random draws.
All one-thread Ridge caches from the interrupted attempt are excluded. Other
models use one numerical thread. This changes no distance definitions, model
selection rules, correction families or original evidence.
