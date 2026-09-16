# Presentation demo

Double-click **START_DEMO.cmd**. It starts the demo locally and opens **Google Chrome** at
<http://127.0.0.1:8511>. The first model load can take a few seconds.
Double-click **STOP_DEMO.cmd** when finished; closing Chrome alone does not stop the server.

The launcher uses the project's `.venv`, `C:\Python313`, or a Python on PATH with
the existing dependencies installed. It opens a Chrome window and keeps the server
window hidden. Launching it again reuses the demo server.

## Files to present

- **Presentation_Demo.pptx**: 11 editable slides.
- **Presentation_Demo.pdf**: the same 11 slides, exported by PowerPoint.
- **PAZ_SLIDE_COMMENTS.md**: remaining factual comments on the preserved slides;
  these comments have not been applied to Paz's section.

The source is Paz's pushed presentation at commit **1429c07**. Slides 2–8 are
unchanged, including their layout and text. Slide 1 adds the requested course
credits and number. New slides 9–11 cover full-trajectory generation, personal
context, and the dashboard. There are no backup slides in this copy. Neither
Paz's original presentation nor `src/confirmatory_dashboard.py` was edited.

## A short live demonstration

1. Begin with **Unconditional VAE, n=8**.
2. Move a latent slider to show the trajectory and timing change together.
3. Optionally select a saved participant fingerprint or switch to n=3.

The sliders are offsets from the selected center, measured in training-latent
standard deviations. This screen decodes **one latent vector**; it does not draw
120 movements or perform the distribution evaluation during the talk. Decoding
the center does not necessarily produce the mean of the generated paths.

The interface reuses the original model-loading and decoding functions. It
loads the fold-0 reference models (seed 42 for neural models), with VAE/CAE at
n=3,8 and CVAE/spline + PCA at n=2,3,4,8. VAE ignores task conditions; spline
uses them for timing only. Conditional model controls are in the collapsed
**Task conditions** section. The main view contains no research tabs, feature
histograms, or minimum-jerk fitting.

Spatial axes are in cm and use equal scale. The onset marker is positioned using
the predicted initiation time on the normalized path. Model output order is
movement time followed by initiation time; labels display the corresponding
values in milliseconds.

## Another computer / fresh clone

Run these commands from the repository root before double-clicking the launcher:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements_dashboard.txt
.\.venv\Scripts\python scripts\prepare_review_workspace.py
```

The existing multi-model evidence package must be available. The demo depends
on the repository's `src`, configuration and saved assets; copying only this
folder to another computer is insufficient.

If a terminal launch is preferred, from the repository root:

```powershell
python -m streamlit run demo\app.py --server.address=127.0.0.1 --server.port=8511
```

Then open the local URL in Chrome. Stop that terminal-launched server with Ctrl+C.

## Rebuilding and checking

`build_slides.mjs` creates only slides 9–11 using the installed Codex presentation
runtime; `assemble_presentation.ps1` combines them with a separate source copy.
The default source-copy path is `.tmp/demo_build/Paz_latest.pptx`, extracted from
the commit above, never the open original. Override it with `-SourcePath` if
needed. `dashboard_preview.png` is a screenshot of the running demo. The source
result tables remain under `production/trajectory_distribution_2026_09_13` and
`production/audit_2026_09_08`; the new charts reuse the audited chart data in
`production/course_slides.json`. Detailed methods and evidence references are
included in the new slides' speaker notes.

Run `python demo/verify_demo.py` to exercise the 12 live configurations and
compare displayed timing against the shared decoder. The private `.build` and
`.runtime` directories contain generated checks and server logs and are ignored
by Git. No training or new statistical analysis is needed to run this demo.
