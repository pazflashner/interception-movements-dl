"""Does changing the loss improve fingerprints? Compare baseline ELBO vs high-beta
vs a subject-discriminative term, at n=3 and n=8, 10 seeds. Key metric: held-out
fingerprint identification accuracy (separation on UNSEEN subjects, chance=1/7)."""
import sys, os, pickle, time, csv
REPO = __import__("pathlib").Path(__file__).resolve().parents[1].as_posix()
sys.path.insert(0, REPO); sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd, torch
from sklearn.metrics import r2_score
import config
from src.vae_model import ConditionalVAE, encode_condition, encode_timing, vae_loss, kl_weight_at
from src.run_config import set_seed
from src.train import split_subject_ids
from src.evaluate import behavioural_probing

DIMS = [3, 8]; SEEDS = list(range(10)); EPOCHS, HIDDEN = 100, 256
TW = config.TIMING_WEIGHT

with open(config.DATA_PROCESSED_DIR / "trials.pkl", "rb") as f:
    trials = pickle.load(f)
traj = np.stack([t["pos_norm"].flatten() for t in trials]).astype(np.float32)
timing = np.stack([encode_timing(t) for t in trials]).astype(np.float32)
subjects = np.array([t["metadata"]["subject"] for t in trials])
cond = np.stack([encode_condition(t["metadata"].get("sp", 1), t["metadata"].get("side", 1)) for t in trials]).astype(np.float32)
subj_int_all = pd.factorize(subjects)[0]

# ── loss variants: (beta_target, disc_lambda) ──
VARIANTS = [("baseline", 1.0, 0.0), ("beta4", 4.0, 0.0), ("discriminative", 1.0, 1.0)]

def disc_loss(mu, sid):
    """Fisher-style: minimise within-subject spread, maximise between-subject spread."""
    uniq = torch.unique(sid); within = 0.0; centers = []; k = 0
    for s in uniq:
        m = sid == s
        if m.sum() < 2:
            continue
        zs = mu[m]; c = zs.mean(0)
        within = within + ((zs - c) ** 2).sum(1).mean()
        centers.append(c); k += 1
    if k < 2:
        return mu.sum() * 0.0
    within = within / k
    C = torch.stack(centers)
    between = ((C - C.mean(0)) ** 2).sum(1).mean()
    return within - between

def train_one(latent, beta_t, disc_l, tr_ids, va_ids, seed):
    set_seed(seed)
    trm, vam = np.isin(subjects, tr_ids), np.isin(subjects, va_ids)
    tm = torch.tensor(traj[trm].mean(0)); ts = torch.tensor(traj[trm].std(0) + 1e-8)
    im = torch.tensor(timing[trm].mean(0)); is_ = torch.tensor(timing[trm].std(0) + 1e-8)
    model = ConditionalVAE(condition_dim=cond.shape[1], hidden_dim=HIDDEN, latent_dim=latent, timing_dim=2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xtr, Ttr, Ctr = torch.tensor(traj[trm]), torch.tensor(timing[trm]), torch.tensor(cond[trm])
    Str = torch.tensor(subj_int_all[trm])
    Xva, Tva, Cva = torch.tensor(traj[vam]), torch.tensor(timing[vam]), torch.tensor(cond[vam])
    g = torch.Generator().manual_seed(seed)
    best, best_state, patience = 1e9, None, 0
    for epoch in range(1, EPOCHS + 1):
        beta = kl_weight_at(epoch, target=beta_t)
        model.train()
        perm = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(Xtr), 64):
            idx = perm[i:i + 64]
            xz = (Xtr[idx] - tm) / ts; tz = (Ttr[idx] - im) / is_
            recon, rt, mu, lv, _ = model(xz, Ctr[idx], tz)
            loss, *_ = vae_loss(recon, xz, mu, lv, beta, recon_timing=rt, target_timing=tz, timing_weight=TW)
            if disc_l > 0:
                loss = loss + disc_l * disc_loss(mu, Str[idx])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        model.eval()
        with torch.no_grad():
            xz = (Xva - tm) / ts; tz = (Tva - im) / is_
            recon, rt, mu, lv, _ = model(xz, Cva, tz)
            _, rl, kl, tl = vae_loss(recon, xz, mu, lv, beta_t, recon_timing=rt, target_timing=tz, timing_weight=TW)
            obj = rl.item() + TW * tl.item() + beta_t * kl.item()
        if obj < best:
            best, best_state, patience = obj, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 25 and beta >= beta_t - 1e-9:
                break
    model.load_state_dict(best_state)
    return model, (tm, ts, im, is_)

def fp_id_acc(mu, subj, seed=0):
    rng = np.random.RandomState(seed); subs = np.unique(subj)
    centers, labels, testB = [], [], []
    for s in subs:
        idx = np.where(subj == s)[0]; rng.shuffle(idx); half = len(idx) // 2
        if half < 1: continue
        centers.append(mu[idx[:half]].mean(0)); labels.append(s)
        testB += [(i, s) for i in idx[half:]]
    C = np.stack(centers)
    correct = sum(labels[int(((mu[i] - C) ** 2).sum(1).argmin())] == s for i, s in testB)
    return correct / len(testB), 1.0 / len(labels)

def sep_ratio(mu, subj):
    centers, within = [], []
    for s in np.unique(subj):
        z = mu[subj == s]; c = z.mean(0); centers.append(c); within.append(((z - c) ** 2).sum(1).mean())
    C = np.stack(centers); between = ((C - C.mean(0)) ** 2).sum(1).mean()
    return float(between / (np.mean(within) + 1e-9))

def eval_one(model, stats, te_ids):
    tm, ts, im, is_ = stats
    tem = np.isin(subjects, te_ids); idx = np.flatnonzero(tem)
    Xte, Tte, Cte = torch.tensor(traj[tem]), torch.tensor(timing[tem]), torch.tensor(cond[tem])
    with torch.no_grad():
        mu, _ = model.encode((Xte - tm) / ts, Cte, (Tte - im) / is_)
        recon_z, rt_z = model.decode(mu, Cte)
        recon = (recon_z * ts + tm).numpy(); rt = (rt_z * is_ + im).numpy()
    mu = mu.numpy()
    mse = float(np.mean((recon - traj[tem]) ** 2)); r2m = float(r2_score(timing[tem][:, 0], rt[:, 0]))
    acc, chance = fp_id_acc(mu, subjects[tem])
    probe = behavioural_probing(mu, [trials[i] for i in idx], list(subjects[tem]))
    fp = int((probe["r2_linear_loso"] > 0).sum())
    return mse, r2m, fp, acc, chance, sep_ratio(mu, subjects[tem])

out = config.RESULTS_DIR / "loss_experiment.csv"
COLS = ["variant", "latent", "seed", "recon_mse", "r2_move", "feat_pos", "fp_id_acc", "chance", "sep_ratio"]
with open(out, "w", newline="") as f: csv.writer(f).writerow(COLS)
t0 = time.time(); done = 0; total = len(DIMS) * len(VARIANTS) * len(SEEDS)
for latent in DIMS:
    for name, beta_t, disc_l in VARIANTS:
        for seed in SEEDS:
            try:
                tr, va, te = split_subject_ids(list(subjects), seed=seed)
                model, stats = train_one(latent, beta_t, disc_l, tr, va, seed)
                mse, r2m, fp, acc, ch, sr = eval_one(model, stats, te)
            except Exception as e:
                mse = r2m = acc = ch = sr = float("nan"); fp = 0
                print(f"  ERR {name} z{latent} s{seed}: {e}", flush=True)
            with open(out, "a", newline="") as f:
                csv.writer(f).writerow([name, latent, seed, mse, r2m, fp, acc, ch, sr])
            done += 1
            print(f"[{time.time()-t0:6.0f}s {done:2d}/{total}] {name:14s} z{latent} s{seed}: "
                  f"mse={mse:.3f} R2m={r2m:.3f} feat={fp} fp_id_acc={acc:.3f}(ch {ch:.2f}) sep={sr:.2f}", flush=True)

df = pd.read_csv(out)
print("\n" + "=" * 84 + "\nLOSS COMPARISON (mean over 10 seeds) — fp_id_acc chance ~ 0.14\n" + "=" * 84)
for latent in DIMS:
    print(f"\n--- n = {latent} ---")
    print(df[df.latent == latent].groupby("variant").agg(
        recon_mse=("recon_mse", "mean"), R2_move=("r2_move", "mean"), feat_pos=("feat_pos", "mean"),
        fp_id_acc=("fp_id_acc", "mean"), sep_ratio=("sep_ratio", "mean")
    ).round(3).reindex([v[0] for v in VARIANTS]).to_string())
print(f"\nSaved {out}  ({time.time()-t0:.0f}s)")
