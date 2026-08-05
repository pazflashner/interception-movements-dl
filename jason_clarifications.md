# Clarifications for Prof. Jason Friedman

A running list of questions to confirm with Jason as the project progresses.
Each entry records the context, what we observed in the data, our tentative
decision, and the open question.

---

## Q1 — The "Not fixating on the dot enough!!!" flag in condition 2

**Context (Jason's original email):**
> "1 = look the whole time at the center of the screen, 2 = free eye movements.
> We said for now, just analyze condition=2."

So condition 2 is the **free eye-movement** condition.

**What we observed (all subjects, from the `.mat` `responseText` field):**
- `"Not fixating on the dot enough!!!"` appears on **2,176 trials** (`successful = -1`).
- Split by condition: **805 in condition 1** vs **1,371 in condition 2**.
- This is backwards from a real fixation check — we'd expect *more* failures in
  condition 1 (fixation required) and *fewer* in condition 2 (eyes free). We see
  the opposite.
- The **hand trajectories look completely normal** — indistinguishable from a
  Success trial. Example: `results/subject05/li_2_1_1_10.csv`.

**Our tentative decision:** keep these trials (do not filter), on the hypothesis
that the flag is a vestigial / spurious eye-tracking message in condition 2.

**Questions:**
1. Was there an eye tracker, and is the fixation check active in condition 2?
2. What does the flag mean for a condition-2 trial — is the movement data valid?
3. Can we safely retain these trials for a model of *hand* kinematics?

---

## Q2 — How we segment and normalize each trial

**Context (Jason's note):**
> "My primary concern regards resampling the different trajectories to the same
> number of data points. In this way, the aspect of time / velocity is removed
> ... One way around this might be for the model to also predict movement time."

**What we found in the data that shaped our approach:**
- Each recording starts when the **object appears** (`marker = 5`, first frame).
- The object then **holds still for a randomised foreperiod** and only *starts
  moving* after **0.18–0.48 s** (varies trial to trial). The participant is only
  allowed to move once the object starts moving.
- The finger starts at rest; the recording **ends at finger arrival**
  (`pressedTime` ≈ end of the CSV), i.e. at the interception moment.

**Our segmentation (fully event-based, no fragile velocity thresholds):**
- **START** = the moment the **object starts moving** (from the object trajectory
  `dotArray`, synced to `marker = 5`). We align to this — not to object
  appearance — so the randomised foreperiod does not leak into the trajectories.
- **END** = **finger arrival** (`pressedTime` / CSV end).
- We resample this window to **T = 100 frames** (cubic spline) and subtract the
  start position. Multi-submovement ("hold–go–hold–go") structure is preserved.
- To restore the time axis, the model also predicts two scalars:
  **wait time** (object-starts-moving → finger movement onset) and
  **movement time** (movement onset → arrival). So a generated sample carries a
  shape *and* its real durations — directly following your suggestion.

**Questions:**
1. Do you agree the correct zero-time is **object-motion onset** (the go-signal),
   not object appearance — so the randomised foreperiod is removed?
2. Does predicting **wait time + movement time** address the resampling concern?
3. Should we predict any other temporal quantity (e.g. number of submovements)?

---

## Q3 — Threshold for discarding "gave-up / skip" trials  ⟶ needs your call

We treat as invalid the trials where the finger reaches the centre **long after**
the object was there — our read is that the only reason to arrive that late is to
end the trial and move on, not to intercept.

**We are currently using a cutoff of `LATE_ARRIVAL_CUTOFF_S = 1.0 s`** (arrival
more than 1 s past the success window ⟶ discard), plus trials that never arrive
(10 s timeout). This is a single variable we will set to whatever you decide.

**Distribution of lateness across all 4,763 condition-2 trials** (lateness =
finger arrival − end of the object's in-centre window; ≤ 0 means on-time/early):

| arrival vs. window            | trials |
|-------------------------------|--------|
| on time or early (≤ 0)        | 81%    |
| late by 0.0–0.5 s             | 843    |
| late by 0.5–1.0 s             | 31     |
| late by 1.0–1.5 s             | 15     |
| late by 1.5–2.0 s             | 4      |
| late by 2.0–3.0 s             | 5      |
| late by > 3 s                 | 3      |

There is **no clean gap** — a smooth heavy-tailed decay. The mass of real
near-miss attempts is over by ~0.5 s; beyond ~1 s it is a thin tail (≈ 27
trials > 1 s, ≈ 8 > 2 s). We can't tell "genuinely slow" from "gave up" in the
1–2 s band from data alone — but it's a handful of trials either way.

**Question:** what lateness cutoff do you want us to use (we default to 1 s)?
And should we drop the never-arrived timeouts (we plan to)?

---

## Q4 — "Too early" trials (moved before the object started moving)

**48 condition-2 trials** are labelled `"Too early"` — the finger left the start
box **before the object started moving** (before the go-signal). By the task's
own definition these are invalid responses, and they break our assumption that
the movement begins from rest at the go-signal.

**Our tentative decision:** discard them.

**Question:** agreed that "Too early" trials should be excluded?

---

## Q5 — (add future questions here)
