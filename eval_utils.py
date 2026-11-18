"""
eval_utils.py - Checkpoint 3 scoring + censored loss

evaluate()       : unchanged ESA score at -6 (leaderboard-comparable), plus the
                   coupled two-head prediction and a cost-ratio threshold sweep.
couple_heads()   : flag -> risk >= -6, no flag -> risk clipped to -6.001.
censored_loss()  : Tobit-style loss for the residual head; floor (-30) labels
                   count as "at or below", not as an exact target.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, fbeta_score, precision_score, recall_score

HIGH_RISK, CLIP, FLOOR = -6.0, -6.001, -30.0

def evaluate(name, y_true, y_pred, p_high=None):
    """Same ESA metric as baseline.py. p_high (optional) = calibrated P(high risk),
    used only for PR-AUC so ranking is threshold-free."""
    y_true = np.asarray(y_true, float)
    y_pred = np.maximum(np.asarray(y_pred, float), CLIP)
    true_hr, pred_hr = y_true >= HIGH_RISK, y_pred >= HIGH_RISK
    mse_hr = float(np.mean((y_true[true_hr] - y_pred[true_hr]) ** 2))
    f2 = fbeta_score(true_hr, pred_hr, beta=2, zero_division=0)
    score = y_pred if p_high is None else p_high
    return {
        "method": name,
        "ESA_score": mse_hr / f2 if f2 > 0 else np.inf,
        "MSE_high_risk": mse_hr,
        # Diagnostic split: regressor error on caught events only.
        "MSE_HR_caught": float(np.mean((y_true[true_hr & pred_hr] - y_pred[true_hr & pred_hr]) ** 2))
                         if (true_hr & pred_hr).any() else np.nan,
        "F2": f2,
        "precision": precision_score(true_hr, pred_hr, zero_division=0),
        "recall": recall_score(true_hr, pred_hr, zero_division=0),
        "PR_AUC": average_precision_score(true_hr, score),
        "TP": int((true_hr & pred_hr).sum()),
        "FP": int((~true_hr & pred_hr).sum()),
        "FN": int((true_hr & ~pred_hr).sum()),
    }

def couple_heads(p_high, risk_pred, threshold):
    """Make the two heads consistent so the classifier alone decides the flag."""
    risk_pred = np.asarray(risk_pred, float)
    flag = np.asarray(p_high) >= threshold
    return np.where(flag, np.maximum(risk_pred, HIGH_RISK), CLIP)

def cost_threshold(cost_ratio):
    """Bayes-optimal cut on a calibrated probability: flag if p > 1/(1+R),
    R = cost(missed conjunction) / cost(false alarm). R=10 -> ~0.091."""
    return 1.0 / (1.0 + cost_ratio)

def threshold_sweep(name, y_true, p_high, risk_pred, ratios=(1, 2, 5, 10, 20, 50, 100)):
    """ESA metrics and expected cost per event across cost ratios."""
    y_true = np.asarray(y_true, float)
    rows = []
    for R in ratios:
        t = cost_threshold(R)
        r = evaluate(name, y_true, couple_heads(p_high, risk_pred, t), p_high)
        r.update(cost_ratio=R, threshold=t,
                 expected_cost=(R * r["FN"] + r["FP"]) / len(y_true))
        rows.append(r)
    return pd.DataFrame(rows)

# ---------------------------------------------------------------- loss (torch)
def censored_loss(resid_pred, latest_risk, y_final, log_sigma, hr_weight=4.0, delta=1.0):
    """Tobit-style censored loss on the residual head.
    pred_risk = latest_risk + resid_pred.
    Uncensored labels (y > -30): Huber on (y - pred_risk), weighted by hr_weight if y >= -6.
    Censored labels (y == -30): -log P(risk <= -30) = -log Phi((-30 - pred_risk)/sigma).
    log_sigma: learnable scalar (or per-event head) for the censoring noise scale."""
    import torch
    import torch.nn.functional as F
    pred = latest_risk + resid_pred
    sigma = log_sigma.exp().clamp_min(1e-3)
    cens = y_final <= FLOOR + 1e-6
    w = torch.where(y_final >= HIGH_RISK, torch.full_like(y_final, hr_weight), torch.ones_like(y_final))
    unc = F.huber_loss(pred, y_final, delta=delta, reduction="none") * w
    z = (FLOOR - pred) / sigma
    cen = -torch.special.log_ndtr(z)          # numerically stable log Phi
    loss = torch.where(cens, cen, unc)
    return loss.mean()

# ---------------------------------------------------------------- LightGBM baseline
# Moved from Checkpoint 2 baseline.py, unchanged in features and settings.
RECENT_K = 3
LAST_CDM_FIELDS = [
    "time_to_tca", "risk", "max_risk_estimate", "max_risk_scaling", "miss_distance",
    "relative_speed", "mahalanobis_distance", "c_object_type",
    "t_sigma_r", "t_sigma_t", "t_sigma_n", "c_sigma_r", "c_sigma_t", "c_sigma_n",
    "t_position_covariance_det", "c_position_covariance_det", "c_rcs_estimate",
    "t_span", "c_span", "F10", "AP",
]

def make_features(cdms, key="parent_event_id"):
    """One row per event: the latest CDM's fields plus simple sequence summaries
    (31 features, same as Checkpoint 2). Expects raw, untransformed CDM rows."""
    cdms = cdms.sort_values([key, "time_to_tca"], ascending=[True, False])
    g = cdms.groupby(key)
    last = g.tail(1).set_index(key)[LAST_CDM_FIELDS].add_prefix("last_")
    risk = g.risk
    summary = pd.DataFrame({
        "n_cdms": g.size(),
        "risk_mean": risk.mean(),
        "risk_std": risk.std().fillna(0),
        "risk_min": risk.min(),
        "risk_max": risk.max(),
        f"risk_mean_last{RECENT_K}": g.tail(RECENT_K).groupby(key).risk.mean(),
        "risk_change": risk.last() - risk.first(),
        "miss_distance_change": g.miss_distance.last() - g.miss_distance.first(),
        "span_days": g.time_to_tca.max() - g.time_to_tca.min(),
    })
    span = summary.span_days.replace(0, np.nan)
    summary["risk_slope_per_day"] = (summary.risk_change / span).fillna(0)
    return last.join(summary)

def lightgbm_baseline(train_cdms, train_labels, test_cdms, test_event_ids, seed=42):
    """Checkpoint 2 LightGBM: 300 trees, lr 0.03, 31 leaves, target clipped at -6.001,
    high-risk events weighted 20x.
    Returns (predictions aligned to test_event_ids, fit seconds)."""
    import time
    import warnings
    import lightgbm as lgb
    warnings.filterwarnings("ignore", message=".*eval_set.*")
    # Same row order as Checkpoint 2 (label order); LightGBM's row sampling depends on it.
    X_tr = make_features(train_cdms).loc[train_labels.index]
    y_tr = train_labels.risk_final.to_numpy()
    target = np.maximum(y_tr, CLIP)
    weights = np.where(y_tr >= HIGH_RISK, 20.0, 1.0)
    model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=31,
                              min_child_samples=20, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, random_state=seed, verbose=-1)
    t0 = time.time()
    model.fit(X_tr, target, sample_weight=weights)
    fit_s = time.time() - t0
    X_te = make_features(test_cdms).loc[list(test_event_ids)]
    return model.predict(X_te), fit_s
