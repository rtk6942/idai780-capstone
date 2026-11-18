"""
Ryan Kenward
IDAI 780
Capstone - Sequence-Aware Collision Risk Forecasting 
Checkpoint 3

Code Purpose: Train the GRU and compare it with the baseline methods.
Quick run: py -3.13 train_gru.py --splits 42 --seeds 0
Baselines only: py -3.13 train_gru.py --baselines-only --splits 42
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from prepare_data import build
from eval_utils import (CLIP, HIGH_RISK, lightgbm_baseline, censored_loss, cost_threshold,
                        couple_heads, evaluate, threshold_sweep)

SEQ_FULL = [
    "risk", "time_to_tca", "gap", "max_risk_estimate", "max_risk_scaling",
    "miss_distance", "relative_speed", "mahalanobis_distance",
    "t_sigma_r", "t_sigma_t", "t_sigma_n", "c_sigma_r", "c_sigma_t", "c_sigma_n",
    "t_position_covariance_det", "c_position_covariance_det",
    "t_obs_used", "c_obs_used", "t_weighted_rms", "c_weighted_rms",
    "t_recommended_od_span", "t_actual_od_span", "c_recommended_od_span", "c_actual_od_span",
    "t_time_lastob_end", "c_time_lastob_end",
]
CTX_FULL = [
    "latest_input_tta", "c_object_type", "t_h_apo", "t_h_per", "c_h_apo", "c_h_per",
    "t_j2k_inc", "c_j2k_inc", "t_j2k_ecc", "c_j2k_ecc", "c_rcs_estimate",
    "F10", "F3M", "SSN", "AP",
]
SEQ_RISK, CTX_RISK = ["risk", "time_to_tca", "gap"], ["latest_input_tta"]
LOG_FIELDS = [
    "max_risk_scaling", "miss_distance", "mahalanobis_distance",
    "t_sigma_r", "t_sigma_t", "t_sigma_n", "c_sigma_r", "c_sigma_t", "c_sigma_n",
    "t_position_covariance_det", "c_position_covariance_det",
    "t_weighted_rms", "c_weighted_rms", "c_rcs_estimate",
]

def featurize(cdm, risk_only):
    cdm = cdm.sort_values(["parent_event_id", "time_to_tca"], ascending=[True, False]).reset_index(drop=True).copy()
    cdm["gap"] = cdm.groupby("parent_event_id").time_to_tca.diff().abs().fillna(0.0)
    cdm["risk_raw"] = cdm["risk"]
    for c in LOG_FIELDS:
        if c in cdm:
            cdm[c] = np.sign(cdm[c]) * np.log1p(np.abs(cdm[c]))
    seq = SEQ_RISK if risk_only else [c for c in SEQ_FULL if c in cdm]
    ctx = CTX_RISK if risk_only else [c for c in CTX_FULL if c in cdm]
    return cdm, seq, ctx

def to_samples(cdm, seq, ctx, stats=None):
    cols = seq + ctx
    if stats is None:
        med = cdm[cols].median()
        filled = cdm[cols].fillna(med)
        stats = (med, filled.mean(), filled.std().replace(0, 1).fillna(1))
    med, mu, sd = stats
    Z = ((cdm[cols].fillna(med) - mu) / sd).clip(-10, 10).astype(np.float32)
    Zs, Zc = Z[seq].to_numpy(), Z[ctx].to_numpy()
    risk = cdm.risk_raw.to_numpy()
    D = {k: [] for k in ("seq", "ctx", "latest", "recent3", "parent")}
    for pid, idx in cdm.groupby("parent_event_id", sort=False).indices.items():
        D["seq"].append(Zs[idx])
        D["ctx"].append(Zc[idx[-1]])
        D["latest"].append(risk[idx[-1]])
        D["recent3"].append(risk[idx[-3:]].mean())
        D["parent"].append(pid)
    for k in ("ctx", "latest", "recent3", "parent"):
        D[k] = np.asarray(D[k])
    return D, stats

def collate(D, idx, device, y=None):
    x = pad_sequence([torch.from_numpy(D["seq"][i]) for i in idx], batch_first=True).to(device)
    lengths = torch.tensor([len(D["seq"][i]) for i in idx])
    c = torch.from_numpy(D["ctx"][idx]).float().to(device)
    base = torch.from_numpy(D["base"][idx]).float().to(device)
    out = [x, lengths, c, base]
    if y is not None:
        out.append(torch.from_numpy(y[idx]).float().to(device))
    return out

class DualHeadGRU(nn.Module):
    def __init__(self, n_seq, n_ctx, hidden=32, dropout=0.2):
        super().__init__()
        self.init_h = nn.Sequential(nn.Linear(n_ctx, hidden), nn.Tanh())
        self.gru = nn.GRU(n_seq, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.cls = nn.Linear(hidden + n_ctx, 1)
        self.reg = nn.Linear(hidden + n_ctx, 1)
        nn.init.zeros_(self.reg.weight)
        nn.init.zeros_(self.reg.bias)
        self.log_sigma = nn.Parameter(torch.zeros(()))

    def forward(self, x, lengths, c):
        h0 = self.init_h(c).unsqueeze(0)
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed, h0)
        z = self.drop(torch.cat([h[-1], c], dim=1))
        return self.cls(z).squeeze(1), self.reg(z).squeeze(1)

def train_model(D, idx, y, args, device, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = DualHeadGRU(D["seq"][0].shape[1], D["ctx"].shape[1], args.hidden).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.pos_weight, device=device))
    for _ in range(args.epochs):
        m.train()
        order = rng.permutation(idx)
        for s in range(0, len(order), args.batch):
            b = order[s:s + args.batch]
            x, L, c, latest, yb = collate(D, b, device, y)
            logit, resid = m(x, L, c)
            loss = (bce(logit, (yb >= HIGH_RISK).float()) +
                    args.reg_weight * censored_loss(resid, latest, yb, m.log_sigma, hr_weight=args.hr_weight))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
    return m

@torch.no_grad()
def predict(m, D, idx, device, batch=1024):
    m.eval()
    logits, resids = [], []
    for s in range(0, len(idx), batch):
        x, L, c, _ = collate(D, idx[s:s + batch], device)
        lg, rs = m(x, L, c)
        logits.append(lg.cpu().numpy())
        resids.append(rs.cpu().numpy())
    return np.concatenate(logits), np.concatenate(resids)

def corrected_predict(p, resid, base, t_veto, t_add, alpha):
    base_flag = base >= HIGH_RISK
    flag = np.where(base_flag, p >= t_veto, p >= t_add)
    risk = base + alpha * resid
    return np.where(flag, np.maximum(risk, HIGH_RISK), CLIP)

def choose_correction(y, p, resid, base):
    base_rec = evaluate("b", y, base)["recall"]
    best = (np.inf, 0.0, 1.01, 0.0)
    best_score = evaluate("b", y, base)["ESA_score"]
    for tv in (0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2):
        for ta in (1.01, 0.9, 0.7, 0.5, 0.3):
            for a in (0.0, 0.25, 0.5, 0.75, 1.0):
                r = evaluate("c", y, corrected_predict(p, resid, base, tv, ta, a))
                if r["recall"] + 1e-9 >= base_rec and r["ESA_score"] < best_score:
                    best_score = r["ESA_score"]
                    best = (r["ESA_score"], tv, ta, a)
    return best[1], best[2], best[3], best_score

def run_split(split_seed, args, device):
    out = Path(args.work) / f"split{split_seed}"
    build(split_seed, raw=Path(args.raw), out=out)
    tr_lab = pd.read_csv(out / "train_labels.csv", index_col="parent_event_id")
    te_lab = pd.read_csv(out / "test_labels.csv", index_col="parent_event_id")
    tr_raw = pd.read_csv(out / "train_cdms.csv")
    te_raw = pd.read_csv(out / "test_cdms.csv")
    tr_cdm, seq, ctx = featurize(tr_raw, args.risk_only)
    te_cdm, _, _ = featurize(te_raw, args.risk_only)
    Dtr, stats = to_samples(tr_cdm, seq, ctx)
    Dte, _ = to_samples(te_cdm, seq, ctx, stats)
    for D in (Dtr, Dte):
        D["base"] = D["recent3"] if args.anchor == "recent3" else D["latest"]
    ytr = tr_lab.risk_final.loc[Dtr["parent"]].to_numpy()
    yte = te_lab.risk_final.loc[Dte["parent"]].to_numpy()
    all_tr = np.arange(len(ytr))
    all_te = np.arange(len(yte))

    rows, sweeps = [], []
    tag = {"split": split_seed}
    rows.append({**evaluate("Persistence", yte, Dte["latest"]), **tag, "seed": -1})
    rows.append({**evaluate("Recent avg (3)", yte, Dte["recent3"]), **tag, "seed": -1})
    if not args.no_lgbm:
        lgb_pred, lgb_s = lightgbm_baseline(tr_raw, tr_lab, te_raw, Dte["parent"], seed=split_seed)
        rows.append({**evaluate("LightGBM", yte, lgb_pred), **tag, "seed": -1, "final_fit_s": lgb_s})
        print(f"  split {split_seed}: LightGBM ESA {rows[-1]['ESA_score']:.3f}, fit {lgb_s:.1f}s", flush=True)
    if args.baselines_only:
        return rows, []

    parents = tr_lab.index.to_numpy()
    folds = list(StratifiedKFold(args.folds, shuffle=True, random_state=split_seed).split(
        parents, tr_lab.high_risk.to_numpy()))

    def fit_seed(seed):
        oof_l = np.full(len(ytr), np.nan)
        oof_r = np.full(len(ytr), np.nan)
        for a, b in folds:
            ti = np.where(np.isin(Dtr["parent"], parents[a]))[0]
            vi = np.where(np.isin(Dtr["parent"], parents[b]))[0]
            oof_l[vi], oof_r[vi] = predict(train_model(Dtr, ti, ytr, args, device, seed), Dtr, vi, device)
        t1 = time.time()
        m = train_model(Dtr, all_tr, ytr, args, device, seed)
        fit_s = time.time() - t1
        t2 = time.time()
        tl, tr_ = predict(m, Dte, all_te, device)
        infer_ms = 1000 * (time.time() - t2) / len(all_te)
        return oof_l, oof_r, tl, tr_, fit_s, infer_ms, sum(p.numel() for p in m.parameters())

    groups = [args.seeds] if args.ensemble else [[s] for s in args.seeds]
    for grp in groups:
        t0 = time.time()
        outs = [fit_seed(s) for s in grp]
        total_s = time.time() - t0
        oof_l = np.mean([x[0] for x in outs], 0)
        oof_r = np.mean([x[1] for x in outs], 0)
        te_l = np.mean([x[2] for x in outs], 0)
        te_r = np.mean([x[3] for x in outs], 0)
        fit_s = sum(x[4] for x in outs)
        infer_ms = sum(x[5] for x in outs)
        params = sum(x[6] for x in outs)

        platt = LogisticRegression(C=1e4).fit(oof_l.reshape(-1, 1), (ytr >= HIGH_RISK).astype(int))
        p_oof = platt.predict_proba(oof_l.reshape(-1, 1))[:, 1]
        p = platt.predict_proba(te_l.reshape(-1, 1))[:, 1]
        seed_tag = "ens" + "+".join(map(str, grp)) if args.ensemble else grp[0]
        base_name = "GRU"
        if args.anchor == "recent3":
            base_name += " anchor-recent3"
        if args.risk_only:
            base_name += " risk-only"
        if args.ensemble:
            base_name += " ensemble"
        extra = {"final_fit_s": fit_s, "total_s": total_s, "infer_ms_per_event": infer_ms, "params": params}

        if args.mode in ("coupled", "both"):
            risk = Dte["base"] + te_r
            pred = couple_heads(p, risk, cost_threshold(args.cost_ratio))
            rows.append({**evaluate(base_name + " coupled", yte, pred, p), **tag, "seed": seed_tag, **extra})
            sw = threshold_sweep(base_name + " coupled", yte, p, risk)
            sw["split"] = split_seed
            sw["seed"] = seed_tag
            sweeps.append(sw)

        if args.mode in ("correction", "both"):
            tv, ta, a, cv_score = choose_correction(ytr, p_oof, oof_r, Dtr["recent3"])
            base_te = Dte["recent3"]
            resid_te = te_r + (Dte["base"] - base_te)
            pred = corrected_predict(p, resid_te, base_te, tv, ta, a)
            rows.append({**evaluate(base_name + " correction", yte, pred, p), **tag, "seed": seed_tag,
                         **extra, "t_veto": tv, "t_add": ta, "alpha": a, "cv_ESA": cv_score})

        last = rows[-1]
        msg = (f"  split {split_seed} seed {seed_tag}: {last['method']} ESA {last['ESA_score']:.3f} "
               f"(recent avg {rows[1]['ESA_score']:.3f}), fit {fit_s:.1f}s")
        if "t_veto" in last:
            msg += f", veto<{last['t_veto']} add>={last['t_add']} alpha={last['alpha']}"
        print(msg, flush=True)
    return rows, sweeps


def main():
    print("Training models...")
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/train_data.csv")
    ap.add_argument("--work", default="data/splits")
    ap.add_argument("--out", default="results")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--splits", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--pos-weight", type=float, default=4.0)
    ap.add_argument("--hr-weight", type=float, default=4.0)
    ap.add_argument("--reg-weight", type=float, default=1.0)
    ap.add_argument("--cost-ratio", type=float, default=10.0)
    ap.add_argument("--anchor", choices=["latest", "recent3"], default="latest")
    ap.add_argument("--mode", choices=["coupled", "correction", "both"], default="both")
    ap.add_argument("--ensemble", action="store_true")
    ap.add_argument("--no-lgbm", action="store_true")
    ap.add_argument("--baselines-only", action="store_true")
    ap.add_argument("--risk-only", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    res_f = out_dir / "cp3_results.csv"
    sw_f = out_dir / "cp3_cost_sweep.csv"
    rows, sweeps = [], []
    done = set()
    if args.resume and res_f.exists():
        prev = pd.read_csv(res_f)
        rows = prev.to_dict("records")
        done = set(prev.split.unique())
        if sw_f.exists():
            sweeps = [pd.read_csv(sw_f)]
        print(f"resuming: splits already done {sorted(done)}")

    for s in args.splits:
        if s in done:
            continue
        r, sw = run_split(s, args, device)
        rows += r
        sweeps += sw
        pd.DataFrame(rows).to_csv(res_f, index=False)
        if any(len(x) for x in sweeps):
            pd.concat(sweeps).to_csv(sw_f, index=False)
        print(f"  saved split {s} to {res_f}", flush=True)

    res = pd.DataFrame(rows)
    cols = ["ESA_score", "MSE_high_risk", "F2", "precision", "recall", "PR_AUC", "TP", "FP", "FN"]
    with pd.option_context("display.width", 220, "display.precision", 3):
        if res.groupby("method").size().max() == 1:     # one run per method: no spread to show
            print("\n=== Results ===")
            print(res.set_index("method")[cols].to_string())
        else:
            print("\n=== Mean (std) over splits x seeds ===")
            print(res.groupby("method")[cols].agg(["mean", "std"]).to_string(na_rep="-"))

    base = res[res.method == "Recent avg (3)"].set_index("split").ESA_score
    for meth, gru in res[res.method.str.startswith("GRU")].groupby("method"):
        b = base.loc[gru.split].to_numpy()
        wins = (gru.ESA_score.to_numpy() < b - 1e-9).sum()
        ties = np.isclose(gru.ESA_score.to_numpy(), b).sum()
        print(f"\n{meth}: beats recent avg on {wins} of {len(gru)} paired runs "
              f"({ties} ties) | fit {gru.final_fit_s.mean():.1f}s | "
              f"params {int(gru.params.iloc[0]):,} | inference {gru.infer_ms_per_event.mean():.3f} ms/event")
    print("Complete.")


if __name__ == "__main__":
    main()
