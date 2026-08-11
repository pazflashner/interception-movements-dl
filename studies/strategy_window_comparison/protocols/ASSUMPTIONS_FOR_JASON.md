# Assumptions to validate with Prof. Jason Friedman

1. A non-empty `pressedTime` is the correct indication that the finger reached
   the interception/collision site.
2. `Too early` describes arrival before the target window, not departure before
   target motion; these trials are therefore retained.
3. The current late-arrival exclusion remains 1 s after the target window ends.
4. The target motion onset derived from the executed MAT `dotArray` is the go
   signal and should define time zero for the strategy-inclusive model.
5. The x-y tracker plane is the task plane; z is off-plane noise.
6. The adapted minimum-jerk settings (100 ms duration, 50 ms onset spacing,
   smallest adequate 0.05/0.10 error order) are suitable for these short reaches.
7. The tracker position unit and external-stimulus/executed-`dotArray` priority
   should be confirmed before physical-unit claims are made.
