"""Full comparison: 4 variants x all latent dims x 10 seeds.
Variants: baseline MLP, CNN, MLP+exact-speed, CNN+exact-speed.
Metrics on 7 held-out subjects: reconstruction MSE, timing R2 (move/react),
overall fingerprint (features > chance, of 12), sub-movement probing R2.
Writes results/experiment_full.csv incrementally (crash-safe)."""
import sys, os, pickle, time, csv
REPO = __import__("pathlib").Path(__file__).resolve().parents[1].as_posix()
sys.path.insert(0, REPO); sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd, torch
from sklearn.metrics import r2_score
import config
from src.vae_model import ConditionalVAE, ConvCVAE, encode_condition, encode_timing, vae_loss, kl_weight_at
from src.run_config import set_seed
from src.train import split_subject_ids
from src.evaluate import behavioural_probing

DIMS = [2, 3, 4, 8, 16]
SEEDS = list(range(10))
EPOCHS, HIDDEN = 100, 256
TW = config.TIMING_WEIGHT

with open(config.DATA_PROCESSED_DIR / "trials.pkl", "rb") as f:
    trials = pickle.load(f)
spd = pd.read_csv(config.RESULTS_DIR / "object_speed.csv").set_index("trial_id")["object_speed"].to_dict()

traj = np.stack([t["pos_norm"].flatten() for t in trials]).astype(np.float32)
timing = np.stack([encode_timing(t) for t in trials]).astype(np.float32)
subjects = np.array([t["metadata"]["subject"] for t in trials])
cond4 = np.stack([encode_condition(t["metadata"].get("sp", 1), t["metadata"].get("side", 1)) for t in trials]).astype(np.float32)
sp_raw = np.array([spd.get(t["metadata"].get("trial_id", ""), np.nan) for t in trials], dtype=np.float32)
sp_z = np.nan_to_num((sp_raw - np.nanmean(sp_raw)) / (np.nanstd(sp_raw) + 1e-8)).astype(np.float32)
cond5 = np.concatenate([cond4, sp_z[:, None]], axis=1)

def train_one(arch, cond, tr_ids, va_ids, seed, latent):
    set_seed(seed)
    trm, vam = np.isin(subjects, tr_ids), np.isin(subjects, va_ids)
    tm = torch.tensor(traj[trm].mean(0)); ts = torch.tensor(traj[trm].std(0) + 1e-8)
    im = torch.tensor(timing[trm].mean(0)); is_ = torch.tensor(timing[trm].std(0) + 1e-8)
    Model = ConvCVAE if arch == "cnn" else ConditionalVAE
    model = Model(condition_dim=cond.shape[1], hidden_dim=HIDDEN, latent_dim=latent, timing_dim=2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xtr, Ttr, Ctr = torch.tensor(traj[trm]), torch.tensor(timing[trm]), torch.tensor(cond[trm])
    Xva, Tva, Cva = torch.tensor(traj[vam]), torch.tensor(timing[vam]), torch.tensor(cond[vam])
    g = torch.Generator().manual_seed(seed)
    best, best_state, patience = 1e9, None, 0
    for epoch in range(1, EPOCHS + 1):
        beta = kl_weight_at(epoch)
        model.train()
        perm = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(Xtr), 64):
            idx = perm[i:i + 64]
            xz = (Xtr[idx] - tm) / ts; tz = (Ttr[idx] - im) / is_
            recon, rt, mu, lv, _ = model(xz, Ctr[idx], tz)
            loss, *_ = vae_loss(recon, xz, mu, lv, beta, recon_timing=rt, target_timing=tz, timing_weight=TW)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        model.eval()
        with torch.no_grad():
            xz = (Xva - tm) / ts; tz = (Tva - im) / is_
            recon, rt, mu, lv, _ = model(xz, Cva, tz)
            _, rl, kl, tl = vae_loss(recon, xz, mu, lv, config.KL_WEIGHT, recon_timing=rt, target_timing=tz, timing_weight=TW)
            obj = rl.item() + TW * tl.item() + config.KL_WEIGHT * kl.item()
        if obj < best:
            best, best_state, patience = obj, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 25 and beta >= config.KL_WEIGHT - 1e-9:
                break
    model.load_state_dict(best_state)
    return model, (tm, ts, im, is_)

def eval_one(model, stats, cond, te_ids):
    tm, ts, im, is_ = stats
    tem = np.isin(subjects, te_ids); idx = np.flatnonzero(tem)
    Xte, Tte, Cte = torch.tensor(traj[tem]), torch.tensor(timing[tem]), torch.tensor(cond[tem])
    with torch.no_grad():
        mu, _ = model.encode((Xte - tm) / ts, Cte, (Tte - im) / is_)
        recon_z, rt_z = model.decode(mu, Cte)
        recon = (recon_z * ts + tm).numpy(); rt = (rt_z * is_ + im).numpy()
    mse = float(np.mean((recon - traj[tem]) ** 2))
    r2m = float(r2_score(timing[tem][:, 0], rt[:, 0])); r2i = float(r2_score(timing[tem][:, 1], rt[:, 1]))
    probe = behavioural_probing(mu.numpy(), [trials[i] for i in idx], list(subjects[tem]))
    fp = int((probe["r2_linear_loso"] > 0).sum())
    sub = probe.loc[probe.target == "n_submovements", "r2_linear_loso"]
    return mse, r2m, r2i, fp, len(probe), (float(sub.iloc[0]) if len(sub) else float("nan"))

VARIANTS = [("baseline_mlp", "mlp", cond4), ("cnn", "cnn", cond4),
            ("mlp+speed", "mlp", cond5), ("cnn+speed", "cnn", cond5)]
out = config.RESULTS_DIR / "experiment_full.csv"
COLS = ["variant", "latent", "seed", "recon_mse", "r2_move", "r2_init", "feat_pos", "n_feat", "submov_r2"]
with open(out, "w", newline="") as f:
    csv.writer(f).writerow(COLS)

t0 = time.time(); done = 0; total = len(DIMS) * len(VARIANTS) * len(SEEDS)
for latent in DIMS:
    for name, arch, cond in VARIANTS:
        for seed in SEEDS:
            try:
                tr, va, te = split_subject_ids(list(subjects), seed=seed)
                model, stats = train_one(arch, cond, tr, va, seed, latent)
                mse, r2m, r2i, fp, nf, sub = eval_one(model, stats, cond, te)
            except Exception as e:
                mse = r2m = r2i = sub = float("nan"); fp = nf = 0
                print(f"  ERR {name} z{latent} seed{seed}: {e}", flush=True)
            row = [name, latent, seed, mse, r2m, r2i, fp, nf, sub]
            with open(out, "a", newline="") as f:
                csv.writer(f).writerow(row)
            done += 1
            print(f"[{time.time()-t0:6.0f}s {done:3d}/{total}] {name:13s} z{latent:2d} s{seed}: "
                  f"mse={mse:.3f} R2m={r2m:.3f} R2i={r2i:.3f} feat={fp}/{nf} submov={sub:.2f}", flush=True)

# ── summary ──
df = pd.read_csv(out)
print("\n" + "=" * 90)
print("FULL COMPARISON (mean over 10 seeds)")
print("=" * 90)
for latent in DIMS:
    print(f"\n--- latent n = {latent} ---")
    sub = df[df.latent == latent].groupby("variant").agg(
        recon_mse=("recon_mse", "mean"), R2_move=("r2_move", "mean"), R2_init=("r2_init", "mean"),
        feat_pos=("feat_pos", "mean"), submov_R2=("submov_r2", "mean")).round(3)
    print(sub.reindex([v[0] for v in VARIANTS]).to_string())
print(f"\nSaved {out}   ({time.time()-t0:.0f}s total)")
