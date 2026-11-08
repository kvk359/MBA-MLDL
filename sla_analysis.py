"""
Dataset 14: Managed Application Incident SLA Breach
Early-escalation classifier: predict next_4h_sla_breach at incident creation (opened_at).

Usage:  python sla_analysis.py <path_to_workbook.xlsx> <output_dir>
Outputs: results.json, CSV tables and PNG figures in <output_dir>.
"""
import sys, os, json, warnings, platform
import numpy as np, pandas as pd
import sklearn, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             precision_recall_curve, roc_curve, auc, confusion_matrix)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.base import clone

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)
WB, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
R = {"versions": {"python": platform.python_version(), "numpy": np.__version__,
                  "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
                  "matplotlib": matplotlib.__version__}, "seed": SEED}

# ---------------------------------------------------------------- 1. Load and audit
df = pd.read_excel(WB, sheet_name="Data").sort_values("opened_at", kind="stable").reset_index(drop=True)
TARGET = "next_4h_sla_breach"
POST = ["next_4h_sla_breach", "resolution_minutes", "credit_penalty_usd", "escalation_required"]
IDS = ["incident_id", "customer_account_id", "incident_team_id"]
CAT = ["declared_severity", "service_id", "customer_tier", "incident_description"]
NUM_RAW = ["affected_users_estimate", "prior_week_incident_count", "observed_error_count_5m",
           "apdex_last_15m", "rolling_latency_p95_ms", "recent_deployment", "dependency_failure_flag",
           "oncall_experience_years", "available_oncall_engineers", "hourly_service_revenue_usd",
           "historical_service_sla_rate"]
y = df[TARGET].astype(int)

audit = {"rows": len(df), "cols": df.shape[1], "dup_rows": int(df.duplicated().sum()),
         "dup_ids": int(df.incident_id.duplicated().sum()),
         "date_min": str(df.opened_at.min()), "date_max": str(df.opened_at.max()),
         "missing": {c: [int(df[c].isna().sum()), round(df[c].isna().mean() * 100, 2)]
                     for c in df.columns if df[c].isna().any()},
         "rows_any_missing": int(df.isna().any(axis=1).sum()),
         "breach_n": int(y.sum()), "breach_rate": round(y.mean(), 4),
         "escalation_rate": round(df.escalation_required.mean(), 4),
         "breach_escalated": int(((y == 1) & (df.escalation_required == 1)).sum()),
         "nonbreach_escalated": int(((y == 0) & (df.escalation_required == 1)).sum()),
         "penalty_pos_equals_breach": bool(((df.credit_penalty_usd > 0) == (y == 1)).all()),
         "breach_resolved_under_240": int(((y == 1) & (df.resolution_minutes < 240)).sum()),
         "accounts": int(df.customer_account_id.nunique()), "teams": int(df.incident_team_id.nunique()),
         "inc_per_account": [int(df.customer_account_id.value_counts().min()), int(df.customer_account_id.value_counts().max())],
         "inc_per_team": [int(df.incident_team_id.value_counts().min()), int(df.incident_team_id.value_counts().max())],
         "accounts_with_multiple_tiers": int((df.groupby("customer_account_id").customer_tier.nunique() > 1).sum()),
         "breach_rate_by_year": df.groupby(df.opened_at.dt.year)[TARGET].mean().round(4).to_dict(),
         "missing_vs_breach": {c: df.groupby(df[c].isna())[TARGET].mean().round(4).rename({False: "present", True: "missing"}).to_dict()
                               for c in ["apdex_last_15m", "rolling_latency_p95_ms", "historical_service_sla_rate"]}}
desc = []
for c in NUM_RAW + ["resolution_minutes", "credit_penalty_usd"]:
    s = df[c].dropna(); q1, q3 = s.quantile([.25, .75]); iqr = q3 - q1
    desc.append({"variable": c, "n": int(s.size), "mean": s.mean(), "sd": s.std(), "min": s.min(),
                 "median": s.median(), "max": s.max(), "skew": s.skew(),
                 "iqr_outliers": int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())})
pd.DataFrame(desc).round(3).to_csv(f"{OUT}/descriptives.csv", index=False)
cat_rates = {}
for c in CAT + ["recent_deployment", "dependency_failure_flag"]:
    g = df.groupby(c)[TARGET].agg(["size", "mean"])
    cat_rates[c] = {str(k): [int(v["size"]), round(v["mean"], 4)] for k, v in g.iterrows()}
audit["cat_rates"] = cat_rates
audit["sev_x_deploy"] = {f"{s}|{d}": [int(n), round(m, 4)] for (s, d), (n, m) in
                         df.groupby(["declared_severity", "recent_deployment"])[TARGET].agg(["size", "mean"]).iterrows()}
audit["exp_quintile_rate"] = [round(v, 4) for v in df.groupby(pd.qcut(df.oncall_experience_years, 5))[TARGET].mean()]
audit["breach_by_hour_range"] = [round(v, 4) for v in df.groupby(df.opened_at.dt.hour)[TARGET].mean().agg(["min", "max"])]
audit["breach_by_weekday_range"] = [round(v, 4) for v in df.groupby(df.opened_at.dt.dayofweek)[TARGET].mean().agg(["min", "max"])]
audit["exp_by_severity"] = df.groupby("declared_severity").oncall_experience_years.mean().round(2).to_dict()
R["audit"] = audit

# ---------------------------------------------------------------- 2. Features (decision-time only)
df["log_latency_p95"] = np.log1p(df.rolling_latency_p95_ms)
df["log_hourly_revenue"] = np.log1p(df.hourly_service_revenue_usd)
NUM = [c for c in NUM_RAW if c not in ("rolling_latency_p95_ms", "hourly_service_revenue_usd")] + ["log_latency_p95", "log_hourly_revenue"]
FEATS = CAT + NUM
assert not set(FEATS) & set(POST + IDS)
X = df[FEATS]

# ---------------------------------------------------------------- 3. Chronological split with 4-hour embargo
B1, B2 = pd.Timestamp("2025-01-01"), pd.Timestamp("2025-08-01")
EMB = pd.Timedelta(hours=4)  # outcome window length; earlier-period rows whose window crosses a boundary are dropped
tr = df.opened_at < B1 - EMB
va = (df.opened_at >= B1) & (df.opened_at < B2 - EMB)
te = df.opened_at >= B2
R["split"] = {k: {"n": int(m.sum()), "breach_rate": round(y[m].mean(), 4),
                  "from": str(df.opened_at[m].min()), "to": str(df.opened_at[m].max())}
              for k, m in [("train", tr), ("validation", va), ("test", te)]}
R["split"]["embargo_dropped"] = int(len(df) - tr.sum() - va.sum() - te.sum())
R["split"]["test_accounts_unseen_in_train"] = int(len(set(df.customer_account_id[te]) - set(df.customer_account_id[tr])))
Xtr, ytr, Xva, yva, Xte, yte = X[tr], y[tr], X[va], y[va], X[te], y[te]

def prep():
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                          ("sc", StandardScaler())]), NUM)])

def scores(yt, s):
    p, r, _ = precision_recall_curve(yt, s)
    return {"roc_auc": roc_auc_score(yt, s), "avg_precision": average_precision_score(yt, s),
            "pr_auc_trapezoid": auc(r, p), "brier": brier_score_loss(yt, s)}

def ece(yt, s, bins=10):
    edges = np.quantile(s, np.linspace(0, 1, bins + 1)); idx = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, bins - 1)
    return float(sum(abs(np.mean(yt[idx == b]) - np.mean(s[idx == b])) * np.mean(idx == b) for b in range(bins) if np.any(idx == b)))

# ---------------------------------------------------------------- 4. Baselines (fitted on training data only)
base = {}
prev = ytr.mean()
base["Prevalence (no skill)"] = (np.full(va.sum(), prev), np.full(te.sum(), prev))
sev_rate = df[tr].groupby("declared_severity")[TARGET].mean()
base["Severity rate"] = (df.loc[va, "declared_severity"].map(sev_rate).values, df.loc[te, "declared_severity"].map(sev_rate).values)
sd_rate = df[tr].groupby(["declared_severity", "recent_deployment"])[TARGET].mean()
key = lambda m: list(zip(df.loc[m, "declared_severity"], df.loc[m, "recent_deployment"]))
base["Severity x deployment rate"] = (np.array([sd_rate[k] for k in key(va)]), np.array([sd_rate[k] for k in key(te)]))
R["severity_rate_train"] = sev_rate.round(4).to_dict()
R["sev_deploy_rate_train"] = {f"{a}|{b}": round(v, 4) for (a, b), v in sd_rate.items()}

# ---------------------------------------------------------------- 5. Candidate models, tuned on validation AP
grids = {
    "Logistic regression": [LogisticRegression(C=C, l1_ratio=l1, solver="saga", max_iter=5000, random_state=SEED)
                            for C in [0.01, 0.03, 0.1, 0.3, 1, 3] for l1 in [0.0, 1.0]],
    "Random forest": [RandomForestClassifier(n_estimators=400, min_samples_leaf=leaf, max_features=mf, max_depth=md,
                                             n_jobs=-1, random_state=SEED)
                      for leaf in [10, 30, 60] for mf in ["sqrt", 0.5] for md in [None, 8]],
    "Gradient boosting": [HistGradientBoostingClassifier(learning_rate=lr, max_depth=d, max_iter=it, l2_regularization=l2,
                                                         random_state=SEED)
                          for lr in [0.03, 0.1] for d in [2, 3] for it in [100, 300] for l2 in [0.0, 1.0]],
    "Neural network (MLP)": [MLPClassifier(hidden_layer_sizes=h, alpha=a, max_iter=1000, early_stopping=True,
                                           validation_fraction=0.15, random_state=SEED)
                             for h in [(16,), (32, 16)] for a in [1e-3, 1e-2, 1e-1]],
}
tuning, best = {}, {}
for name, cands in grids.items():
    rows = []
    for m in cands:
        p = Pipeline([("prep", prep()), ("model", m)]).fit(Xtr, ytr)
        s = p.predict_proba(Xva)[:, 1]
        params = {k: v for k, v in m.get_params().items() if k in
                  ("C", "l1_ratio", "min_samples_leaf", "max_features", "max_depth", "learning_rate", "max_iter",
                   "l2_regularization", "hidden_layer_sizes", "alpha")}
        rows.append({"params": str(params), "val_ap": average_precision_score(yva, s), "val_auc": roc_auc_score(yva, s), "pipe": p})
    rows.sort(key=lambda r: -r["val_ap"])
    tuning[name] = [{"params": r["params"], "val_ap": round(r["val_ap"], 4), "val_auc": round(r["val_auc"], 4)} for r in rows]
    best[name] = rows[0]["pipe"]
R["tuning"] = tuning

# class-weighting check on the tuned logistic regression
lr_best = best["Logistic regression"]
lr_bal = Pipeline([("prep", prep()), ("model", clone(lr_best.named_steps["model"]).set_params(class_weight="balanced"))]).fit(Xtr, ytr)

allm = {}
for n, (sv, st) in base.items(): allm[n] = (sv, st)
for n, p in best.items(): allm[n] = (p.predict_proba(Xva)[:, 1], p.predict_proba(Xte)[:, 1])
allm["Logistic regression (balanced weights)"] = (lr_bal.predict_proba(Xva)[:, 1], lr_bal.predict_proba(Xte)[:, 1])
comp = []
for n, (sv, st) in allm.items():
    v, t = scores(yva.values, sv), scores(yte.values, st)
    comp.append({"model": n, **{f"val_{k}": x for k, x in v.items()}, **{f"test_{k}": x for k, x in t.items()},
                 "val_ece": ece(yva.values, sv) if np.std(sv) > 0 else np.nan, "test_ece": ece(yte.values, st) if np.std(st) > 0 else np.nan})
comp = pd.DataFrame(comp)
comp.round(4).to_csv(f"{OUT}/model_comparison.csv", index=False)
R["comparison"] = comp.round(4).to_dict(orient="records")

# Selection rule (stated before reading test results): highest validation AP among learned models;
# if a more interpretable model is within 0.01 AP of the best, prefer it. Order of interpretability: LR > GB > RF > MLP.
learned = comp[comp.model.isin(list(best))].set_index("model")
top_ap = learned.val_avg_precision.max()
for cand in ["Logistic regression", "Gradient boosting", "Random forest", "Neural network (MLP)"]:
    if learned.loc[cand, "val_avg_precision"] >= top_ap - 0.01:
        FINAL = cand; break
R["final_model"] = FINAL
R["final_params"] = tuning[FINAL][0]["params"]
fm = best[FINAL]
s_va, s_te = allm[FINAL]

# ---------------------------------------------------------------- 6. Bootstrap uncertainty on the test set
rng = np.random.default_rng(SEED)
yv = yte.values; sev_te = df.loc[te, "declared_severity"].values; dep_te = df.loc[te, "recent_deployment"].values
rule_te = np.isin(sev_te, ["P1", "P2"])
rule_va = np.isin(df.loc[va, "declared_severity"].values, ["P1", "P2"])
t_rule_eq = np.quantile(s_va, 1 - rule_va.mean())  # model threshold giving the rule's validation review volume
model_eq_te = s_te >= t_rule_eq
sev_s_te = allm["Severity rate"][1]
boots = {"auc": [], "ap": [], "auc_sev": [], "ap_sev": [], "recall_diff": [], "prec_diff": []}
for _ in range(1000):
    i = rng.integers(0, len(yv), len(yv)); yb = yv[i]
    if yb.min() == yb.max(): continue
    boots["auc"].append(roc_auc_score(yb, s_te[i])); boots["ap"].append(average_precision_score(yb, s_te[i]))
    boots["auc_sev"].append(roc_auc_score(yb, sev_s_te[i])); boots["ap_sev"].append(average_precision_score(yb, sev_s_te[i]))
    tp_m = (model_eq_te[i] & (yb == 1)).sum(); tp_r = (rule_te[i] & (yb == 1)).sum()
    boots["recall_diff"].append((tp_m - tp_r) / yb.sum())
    boots["prec_diff"].append(tp_m / max(model_eq_te[i].sum(), 1) - tp_r / max(rule_te[i].sum(), 1))
R["bootstrap_test_95ci"] = {k: [round(np.percentile(v, 2.5), 4), round(np.percentile(v, 97.5), 4)] for k, v in boots.items()}

# ---------------------------------------------------------------- 7. Threshold and workload analysis
def cm_stats(yt, flag, pen=None):
    tn, fp, fn, tp = confusion_matrix(yt, flag, labels=[0, 1]).ravel()
    d = {"flagged_pct": flag.mean() * 100, "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
         "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1),
         "f1": 2 * tp / max(2 * tp + fp + fn, 1)}
    if pen is not None: d["penalty_share"] = pen[flag].sum() / pen.sum()
    return d
pen_te = df.loc[te, "credit_penalty_usd"].values
policies = []
policies.append({"policy": "Escalate P1/P2 (current-practice proxy)", "rule": True, **cm_stats(yv, rule_te, pen_te)})
r2 = rule_te | (dep_te == 1)
policies.append({"policy": "Escalate P1/P2 or recent deployment", "rule": True, **cm_stats(yv, r2, pen_te)})
for cap in [0.10, 0.20, 0.30]:
    t = np.quantile(s_va, 1 - cap)
    policies.append({"policy": f"Model: top {int(cap*100)}% capacity", "threshold": float(t), **cm_stats(yv, s_te >= t, pen_te)})
policies.append({"policy": "Model: same volume as P1/P2 rule", "threshold": float(t_rule_eq), **cm_stats(yv, model_eq_te, pen_te)})
r2_va = rule_va | (df.loc[va, "recent_deployment"].values == 1)
t_r2 = np.quantile(s_va, 1 - r2_va.mean())
policies.append({"policy": "Model: same volume as P1/P2 or deployment rule", "threshold": float(t_r2), **cm_stats(yv, s_te >= t_r2, pen_te)})
# F1-maximising threshold chosen on validation
p_, r_, th_ = precision_recall_curve(yva, s_va); f1 = 2 * p_[:-1] * r_[:-1] / np.maximum(p_[:-1] + r_[:-1], 1e-9)
t_f1 = th_[np.argmax(f1)]
policies.append({"policy": "Model: F1-maximising threshold", "threshold": float(t_f1), **cm_stats(yv, s_te >= t_f1, pen_te)})
pol = pd.DataFrame(policies); pol.round(4).to_csv(f"{OUT}/policies_test.csv", index=False)
R["policies_test"] = pol.round(4).to_dict(orient="records")
# same rule numbers on validation for reference
R["rules_validation"] = {"P1/P2": cm_stats(yva.values, rule_va), "P1/P2 or deploy": cm_stats(yva.values, r2_va)}

# Assumed cost ratios (cost of a missed breach relative to one unnecessary escalation = 1). ASSUMPTIONS, not measured costs.
grid_t = np.unique(np.quantile(s_va, np.linspace(0, 1, 401)))
cost_rows = []
for ratio in [2, 3, 5, 10]:
    def cost(yt, flag): fp = ((flag) & (yt == 0)).sum(); fn = ((~flag) & (yt == 1)).sum(); return ratio * fn + fp
    vc = [cost(yva.values, s_va >= t) for t in grid_t]
    tstar = grid_t[int(np.argmin(vc))]
    n = len(yv)
    row = {"ratio": ratio, "threshold": float(tstar), "theory_threshold": 1 / (1 + ratio),
           "model": cost(yv, s_te >= tstar) / n, "escalate_none": cost(yv, np.zeros(n, bool)) / n,
           "escalate_all": cost(yv, np.ones(n, bool)) / n, "p1p2_rule": cost(yv, rule_te) / n,
           "p1p2_or_deploy": cost(yv, r2) / n, "model_flagged_pct": (s_te >= tstar).mean() * 100,
           "model_recall": ((s_te >= tstar) & (yv == 1)).sum() / yv.sum()}
    cost_rows.append(row)
costs = pd.DataFrame(cost_rows); costs.round(4).to_csv(f"{OUT}/cost_test.csv", index=False)
R["costs_test"] = costs.round(4).to_dict(orient="records")

# Gains curve data
def gains(yt, s):
    o = np.argsort(-s); c = np.cumsum(yt[o]) / yt.sum(); return np.arange(1, len(yt) + 1) / len(yt), c

# ---------------------------------------------------------------- 8. Calibration
def calib(yt, s, bins=10):
    q = pd.qcut(s, bins, duplicates="drop"); g = pd.DataFrame({"y": yt, "s": s, "q": q}).groupby("q", observed=True)
    return g.s.mean().values, g.y.mean().values
R["calibration_test"] = {n: {"ece": round(ece(yv, allm[n][1]), 4), "mean_pred": round(allm[n][1].mean(), 4)}
                         for n in [FINAL, "Logistic regression (balanced weights)", "Random forest", "Gradient boosting"]}
R["test_breach_rate"] = round(yv.mean(), 4)

# ---------------------------------------------------------------- 9. Explainability
names = fm.named_steps["prep"].get_feature_names_out()
coef = pd.Series(fm.named_steps["model"].coef_[0], index=names)
coef_df = pd.DataFrame({"feature": names, "coef": coef.values, "odds_ratio": np.exp(coef.values)}).sort_values("coef", key=abs, ascending=False)
coef_df.round(4).to_csv(f"{OUT}/coefficients.csv", index=False)
R["coefficients"] = coef_df.round(4).to_dict(orient="records")
R["intercept"] = float(fm.named_steps["model"].intercept_[0])
pi = permutation_importance(fm, Xva, yva, scoring="average_precision", n_repeats=30, random_state=SEED, n_jobs=-1)
pi_df = pd.DataFrame({"feature": FEATS, "mean": pi.importances_mean, "sd": pi.importances_std}).sort_values("mean", ascending=False)
pi_df.round(5).to_csv(f"{OUT}/permutation_importance.csv", index=False)
R["permutation_importance"] = pi_df.round(5).to_dict(orient="records")
pi_rf = permutation_importance(best["Random forest"], Xva, yva, scoring="average_precision", n_repeats=15, random_state=SEED, n_jobs=-1)
R["permutation_importance_rf_top5"] = pd.Series(pi_rf.importances_mean, FEATS).sort_values(ascending=False).head(5).round(5).to_dict()
# Model with only severity + deployment + prior incidents (feature-selection check)
small = ["declared_severity", "recent_deployment", "prior_week_incident_count"]
sm = Pipeline([("prep", ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), ["declared_severity"]),
                                           ("num", StandardScaler(), ["recent_deployment", "prior_week_incident_count"])])),
               ("model", clone(fm.named_steps["model"]))]).fit(Xtr[small], ytr)
R["reduced_model"] = {"features": small, **{f"val_{k}": round(v, 4) for k, v in scores(yva.values, sm.predict_proba(Xva[small])[:, 1]).items()},
                      **{f"test_{k}": round(v, 4) for k, v in scores(yv, sm.predict_proba(Xte[small])[:, 1]).items()}}

# Local explanations: contribution = coefficient x transformed value
Zte = fm.named_steps["prep"].transform(Xte); Zte = Zte.toarray() if hasattr(Zte, "toarray") else Zte
contrib = Zte * coef.values
te_idx = np.where(te)[0]
def local(i_pos):
    c = pd.Series(contrib[i_pos], names); c = c[c.abs() > 1e-9].sort_values(key=abs, ascending=False).head(4)
    r = df.iloc[te_idx[i_pos]]
    return {"incident_id": r.incident_id, "severity": r.declared_severity, "recent_deployment": int(r.recent_deployment),
            "prior_week_incident_count": int(r.prior_week_incident_count), "oncall_experience_years": float(r.oncall_experience_years),
            "tier": r.customer_tier, "score": round(float(s_te[i_pos]), 3), "actual": int(r[TARGET]),
            "top_contributions": {k: round(v, 3) for k, v in c.items()}}
p4_mask = (sev_te == "P4")
ex = [int(np.argmax(s_te)), int(np.where(p4_mask)[0][np.argmax(s_te[p4_mask])]), int(np.where(sev_te == "P1")[0][np.argmin(s_te[sev_te == "P1"])])]
R["local_examples"] = [local(i) for i in ex]

# ---------------------------------------------------------------- 10. Subgroups, within-severity ranking, temporal stability
t20 = np.quantile(s_va, 0.80)
flag20 = s_te >= t20
sub = []
for col in ["customer_tier", "service_id", "declared_severity"]:
    for lvl in sorted(df.loc[te, col].unique()):
        m = (df.loc[te, col].values == lvl)
        yy = yv[m]
        sub.append({"group": col, "level": lvl, "n": int(m.sum()), "breach_rate": yy.mean(), "flagged_pct": flag20[m].mean() * 100,
                    "recall": (flag20[m] & (yy == 1)).sum() / max(yy.sum(), 1),
                    "precision": (flag20[m] & (yy == 1)).sum() / max(flag20[m].sum(), 1),
                    "auc": roc_auc_score(yy, s_te[m]) if 0 < yy.sum() < len(yy) else np.nan})
sub = pd.DataFrame(sub); sub.round(4).to_csv(f"{OUT}/subgroups_test.csv", index=False)
R["subgroups_test"] = sub.round(4).to_dict(orient="records")
R["threshold_20pct"] = float(t20)
q_all = va | te; s_all = np.empty(len(df)); s_all[:] = np.nan
s_all[np.where(va)[0]] = s_va; s_all[np.where(te)[0]] = s_te
temp = []
edges = pd.to_datetime(["2025-01-01", "2025-04-01", "2025-08-01", "2025-11-01", "2026-02-01", "2026-04-01"])
labels = ["Jan-Mar 2025 (V)", "Apr-Jul 2025 (V)", "Aug-Oct 2025 (T)", "Nov 2025-Jan 2026 (T)", "Feb-Mar 2026 (T)"]
per = pd.cut(df.opened_at[q_all], edges, right=False, labels=labels)
for qtr, g in df[q_all].assign(s=s_all[q_all]).groupby(per, observed=True):
    temp.append({"quarter": str(qtr), "n": len(g), "breach_rate": g[TARGET].mean(), "auc": roc_auc_score(g[TARGET], g.s),
                 "ap": average_precision_score(g[TARGET], g.s)})
temp = pd.DataFrame(temp); temp.round(4).to_csv(f"{OUT}/temporal.csv", index=False)
R["temporal"] = temp.round(4).to_dict(orient="records")

# Entity-dependence sensitivity: grouped CV on train+validation (not used for selection)
dev = tr | va
for grp in ["customer_account_id", "incident_team_id"]:
    pr = cross_val_predict(clone(fm), X[dev], y[dev], cv=GroupKFold(5), groups=df.loc[dev, grp], method="predict_proba")[:, 1]
    R[f"groupcv_{grp}_auc"] = round(roc_auc_score(y[dev], pr), 4)
    R[f"groupcv_{grp}_ap"] = round(average_precision_score(y[dev], pr), 4)
team_rate = df[tr].groupby("incident_team_id")[TARGET].agg(["mean", "size"])
R["team_rate_train_range"] = [round(team_rate["mean"].min(), 4), round(team_rate["mean"].max(), 4)]
# expected spread of team rates under pure chance with ~65 incidents each
R["team_size_train_median"] = int(team_rate["size"].median())

# ---------------------------------------------------------------- 11. Figures
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8984", "axes.labelcolor": "#2b2b29", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.8, "axes.axisbelow": True,
                     "legend.frameon": False, "figure.dpi": 200})
BLUE, ORANGE, AQUA, YELLOW, GREY, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8984", "#6250d6"

# Fig 1: breach rate by severity and deployment
fig, ax = plt.subplots(figsize=(6.4, 3.2))
sevs = ["P1", "P2", "P3", "P4"]; w = 0.36; xs = np.arange(4)
for j, (d, col, lab) in enumerate([(0, BLUE, "No recent deployment"), (1, ORANGE, "Recent deployment")]):
    vals = [audit["sev_x_deploy"][f"{s}|{d}"][1] * 100 for s in sevs]
    bars = ax.bar(xs + (j - 0.5) * (w + 0.02), vals, w, color=col, label=lab)
    for b, v in zip(bars, vals): ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}%", ha="center", fontsize=8, color="#2b2b29")
ax.axhline(y.mean() * 100, color=GREY, ls="--", lw=1); ax.text(1.62, y.mean() * 100 + 1.2, f"Overall {y.mean()*100:.1f}%", ha="center", fontsize=8, color="#52514e")
ax.set_xticks(xs, sevs); ax.set_ylabel("SLA breach rate (%)"); ax.set_xlabel("Declared severity"); ax.set_ylim(0, 72)
ax.legend(loc="upper right", fontsize=8); ax.grid(axis="x", visible=False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_severity_deploy.png"); plt.close(fig)

# Fig 2: ROC and PR on validation
fig, axs = plt.subplots(1, 2, figsize=(6.8, 3.2))
for n, col in [(FINAL, BLUE), ("Gradient boosting", ORANGE), ("Random forest", AQUA), ("Severity rate", GREY)]:
    sv = allm[n][0]; fpr, tpr, _ = roc_curve(yva, sv); p, r, _ = precision_recall_curve(yva, sv)
    ls = "--" if n == "Severity rate" else "-"
    axs[0].plot(fpr, tpr, color=col, lw=1.6, ls=ls, label=f"{n} ({roc_auc_score(yva, sv):.3f})")
    axs[1].plot(r, p, color=col, lw=1.6, ls=ls, label=f"{n} ({average_precision_score(yva, sv):.3f})")
axs[0].plot([0, 1], [0, 1], color="#c3c2b7", lw=1); axs[1].axhline(yva.mean(), color="#c3c2b7", lw=1)
axs[0].set(xlabel="False positive rate", ylabel="True positive rate (recall)", title="ROC (AUC)")
axs[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall (AP)", ylim=(0, 1))
for a in axs: a.legend(fontsize=6.5, loc="lower right" if a is axs[0] else "upper right"); a.title.set_fontsize(10)
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_roc_pr_validation.png"); plt.close(fig)

# Fig 3: gains / workload curve on test
fig, ax = plt.subplots(figsize=(6.4, 3.4))
xg, cg = gains(yv, s_te); ax.plot(xg * 100, cg * 100, color=BLUE, lw=1.8, label=f"{FINAL} score")
xg2, cg2 = gains(yv, sev_s_te + rng.normal(0, 1e-9, len(sev_s_te))); ax.plot(xg2 * 100, cg2 * 100, color=GREY, lw=1.4, ls="--", label="Severity rate only")
ax.plot([0, 100], [0, 100], color="#c3c2b7", lw=1, label="Random review")
for ppol, mk, col in [(policies[0], "o", ORANGE), (policies[1], "s", YELLOW)]:
    ax.scatter(ppol["flagged_pct"], ppol["recall"] * 100, s=40, marker=mk, color=col, zorder=5, edgecolor="white", linewidth=1.5, label=ppol["policy"])
for cap in [10, 20, 30]: ax.axvline(cap, color="#e6e5e0", lw=1, zorder=0)
ax.set(xlabel="Share of new incidents escalated for early review (%)", ylabel="Share of breaches captured (%)", xlim=(0, 100), ylim=(0, 100))
ax.legend(fontsize=7.5, loc="lower right"); fig.tight_layout(); fig.savefig(f"{OUT}/fig3_gains_test.png"); plt.close(fig)

# Fig 4: calibration on test
fig, ax = plt.subplots(figsize=(4.6, 3.6))
for n, col, lab in [(FINAL, BLUE, FINAL), ("Logistic regression (balanced weights)", ORANGE, "LR with balanced class weights"), ("Gradient boosting", AQUA, "Gradient boosting")]:
    mp, fo = calib(yv, allm[n][1]); ax.plot(mp, fo, marker="o", ms=4, color=col, lw=1.4, label=lab)
ax.plot([0, 1], [0, 1], color="#c3c2b7", lw=1, label="Perfect calibration")
ax.set(xlabel="Mean predicted breach probability", ylabel="Observed breach rate", xlim=(0, 0.85), ylim=(0, 0.85))
ax.legend(fontsize=7.5, loc="upper left"); fig.tight_layout(); fig.savefig(f"{OUT}/fig4_calibration_test.png"); plt.close(fig)

# Fig 5: permutation importance
fig, ax = plt.subplots(figsize=(6.4, 3.6))
pp = pi_df.sort_values("mean")
ax.barh(pp.feature, pp["mean"], xerr=pp["sd"], color=BLUE, height=0.6, error_kw={"ecolor": "#52514e", "lw": 0.8, "capsize": 2})
ax.axvline(0, color="#8a8984", lw=0.8); ax.set_xlabel("Drop in validation average precision when feature is shuffled")
ax.tick_params(axis="y", labelsize=8); ax.grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig5_permutation_importance.png"); plt.close(fig)

# Fig 6: cost per incident under assumed ratios
fig, ax = plt.subplots(figsize=(6.4, 3.2))
xs = np.arange(len(costs)); w = 0.2
for j, (k, col, lab) in enumerate([("escalate_all", GREY, "Escalate all"), ("p1p2_rule", ORANGE, "P1/P2 rule"),
                                   ("p1p2_or_deploy", YELLOW, "P1/P2 or deployment"), ("model", BLUE, "Model, cost-based threshold")]):
    ax.bar(xs + (j - 1.5) * (w + 0.01), costs[k], w, color=col, label=lab)
ax.set_xticks(xs, [f"{r}:1" for r in costs.ratio]); ax.set_xlabel("Assumed cost of a missed breach : one unnecessary escalation")
ax.set_ylabel("Cost units per incident (test)"); ax.legend(fontsize=7.5, ncol=2, loc="upper left"); ax.grid(axis="x", visible=False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig6_cost_test.png"); plt.close(fig)

# Fig 7: temporal stability
fig, ax = plt.subplots(figsize=(6.4, 2.8))
ax.plot(temp.quarter, temp.auc, marker="o", color=BLUE, lw=1.8, label="ROC-AUC")
ax.plot(temp.quarter, temp.ap, marker="s", color=ORANGE, lw=1.8, label="Average precision")
ax.plot(temp.quarter, temp.breach_rate, marker="^", color=GREY, lw=1.2, ls="--", label="Breach rate (no-skill AP)")
ax.axvline(1.5, color="#c3c2b7", lw=1); ax.text(0.5, 0.72, "Validation", ha="center", fontsize=8, color="#52514e"); ax.text(3, 0.72, "Test", ha="center", fontsize=8, color="#52514e"); ax.tick_params(axis="x", labelsize=7.5)
ax.set(ylim=(0, 0.8), ylabel="Score"); ax.legend(fontsize=7.5, loc="lower right", ncol=3)
fig.tight_layout(); fig.savefig(f"{OUT}/fig7_temporal.png"); plt.close(fig)

# Fig 8: missingness and distribution of skewed variables
fig, axs = plt.subplots(1, 3, figsize=(6.8, 2.6))
for a, c, lab in [(axs[0], "rolling_latency_p95_ms", "p95 latency (ms)"), (axs[1], "hourly_service_revenue_usd", "Hourly service revenue (USD)"), (axs[2], "oncall_experience_years", "On-call experience (years)")]:
    a.hist(df.loc[y == 0, c].dropna(), bins=40, color=BLUE, alpha=0.75, density=True, label="No breach")
    a.hist(df.loc[y == 1, c].dropna(), bins=40, color=ORANGE, alpha=0.6, density=True, label="Breach")
    a.set_xlabel(lab, fontsize=8); a.set_yticks([]); a.tick_params(labelsize=7)
axs[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{OUT}/fig8_distributions.png"); plt.close(fig)

def conv(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    return str(o)
json.dump(R, open(f"{OUT}/results.json", "w"), indent=1, default=conv)
print("Final model:", FINAL, R["final_params"])
print(comp[["model", "val_roc_auc", "val_avg_precision", "test_roc_auc", "test_avg_precision", "test_brier"]].round(3).to_string())
