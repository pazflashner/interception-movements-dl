import os, sys
REPO = __import__("pathlib").Path(__file__).resolve().parents[1].as_posix()
sys.path.insert(0, REPO); sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import scipy.io as sio, numpy as np, pandas as pd
import config
ROOT = r"D:\DropBox\Dropbox\results"
rows = []
for s in sorted(os.listdir(ROOT)):
    if not s.startswith("subject"): continue
    sdir = os.path.join(ROOT, s)
    for f in os.listdir(sdir):
        if f.startswith("trialinfo_2_") and f.endswith(".mat"):
            tag = f[len("trialinfo_"):-4]
            try:
                m = sio.loadmat(os.path.join(sdir, f), squeeze_me=True, struct_as_record=False)["thistrial"]
                da = np.array(m.dotArray)
                step = np.linalg.norm(np.diff(da, axis=0), axis=1)
                moving = step[step > 1e-6]
                speed = float(moving.mean() * config.STIMULUS_HZ) if moving.size else np.nan
            except Exception:
                speed = np.nan
            rows.append((f"{s}_li_{tag}", speed))
df = pd.DataFrame(rows, columns=["trial_id", "object_speed"])
os.makedirs(os.path.join(REPO, "results"), exist_ok=True)
df.to_csv(os.path.join(REPO, "results", "object_speed.csv"), index=False)
print(f"saved {len(df)} object speeds; mean={df.object_speed.mean():.1f}, sd={df.object_speed.std():.1f}")
