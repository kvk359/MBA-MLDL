"""
Phase 3 supporting analysis: turns the Phase 2 model into a decision framework.
Rebuilds the Phase 2 split and selected model exactly (same seed and settings), then evaluates
risk bands, impact bands, layered escalation policies and daily workload on the untouched test period.

Usage: python phase3_framework.py <workbook.xlsx> <output_dir>
"""
import sys, os, json, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")
SEED = 42
WB, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
R = {}

# ---- identical data preparation, split and model to Phase 2
df = pd.read_excel(WB, sheet_name="Data").sort_values("opened_at", kind="stable").reset_index(drop=True)
T = "next_4h_sla_breach"; y = df[T].astype(int)
CAT = ["declared_severity", "service_id", "customer_tier", "incident_description"]
df["log_latency_p95"] = np.log1p(df.rolling_latency_p95_ms); df["log_hourly_revenue"] = np.log1p(df.hourly_service_revenue_usd)
NUM = ["affected_users_estimate", "prior_week_incident_count", "observed_error_count_5m", "apdex_last_15m", "recent_deployment",
       "dependency_failure_flag", "oncall_experience_years", "available_oncall_engineers", "historical_service_sla_rate",
       "log_latency_p95", "log_hourly_revenue"]
X = df[CAT + NUM]
B1, B2, EMB = pd.Timestamp("2025-01-01"), pd.Timestamp("2025-08-01"), pd.Timedelta(hours=4)
tr = df.opened_at < B1 - EMB; va = (df.opened_at >= B1) & (df.opened_at < B2 - EMB); te = df.opened_at >= B2
prep = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
                          ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)), ("sc", StandardScaler())]), NUM)])
model = Pipeline([("prep", prep), ("model", LogisticRegression(C=0.1, l1_ratio=1.0, solver="saga", max_iter=5000, random_state=SEED))]).fit(X[tr], y[tr])
s_va, s_te = model.predict_proba(X[va])[:, 1], model.predict_proba(X[te])[:, 1]
R["check_test_auc"] = round(roc_auc_score(y[te], s_te), 4); R["check_test_ap"] = round(average_precision_score(y[te], s_te), 4)

# ---- risk bands (thresholds fixed on validation)
t_high, t_med = np.quantile(s_va, 0.90), np.quantile(s_va, 0.70)
R["thresholds"] = {"high": round(float(t_high), 4), "medium": round(float(t_med), 4)}
D = df[te].copy(); D["score"] = s_te; D["y"] = y[te].values
D["risk"] = np.where(D.score >= t_high, "High", np.where(D.score >= t_med, "Medium", "Low"))
# impact band: mission-critical tier OR hourly revenue in top quartile of the training period (business policy, not predicted)
rev_q75 = df.loc[tr, "hourly_service_revenue_usd"].quantile(0.75); R["revenue_q75_train"] = round(float(rev_q75), 2)
imp = lambda d: np.where((d.customer_tier == "mission-critical") | (d.hourly_service_revenue_usd >= rev_q75), "High impact", "Standard impact")
D["impact"] = imp(D); df["impact"] = imp(df)
days = (D.opened_at.max() - D.opened_at.min()).days + 1; R["test_days"] = int(days)
R["test_incidents_per_day"] = round(len(D) / days, 2)

def summ(g):
    return pd.Series({"n": len(g), "share_pct": 100 * len(g) / len(D), "per_day": len(g) / days, "breach_rate": g.y.mean(),
                      "share_of_breaches": g.y.sum() / D.y.sum(), "penalty_share": g.credit_penalty_usd.sum() / D.credit_penalty_usd.sum()})
risk_tab = D.groupby("risk").apply(summ).reindex(["High", "Medium", "Low"]); R["risk_bands"] = risk_tab.round(4).reset_index().to_dict(orient="records")
mat = D.groupby(["risk", "impact"]).apply(summ).reset_index(); R["matrix"] = mat.round(4).to_dict(orient="records")

# Does the impact band relate to the size of the (simulated) penalty when a breach happens? (all data, breaches only)
br = df[df[T] == 1]
R["penalty_per_breach_by_impact"] = br.groupby("impact").credit_penalty_usd.agg(["count", "mean", "median"]).round(1).reset_index().to_dict(orient="records")
R["penalty_per_breach_by_tier"] = br.groupby("customer_tier").credit_penalty_usd.agg(["count", "mean", "median"]).round(1).reset_index().to_dict(orient="records")
R["penalty_per_breach_by_severity"] = br.groupby("declared_severity").credit_penalty_usd.agg(["count", "mean", "median"]).round(1).reset_index().to_dict(orient="records")
R["breach_rate_by_impact_all"] = df.groupby("impact")[T].mean().round(4).to_dict()
R["impact_share_all"] = df.impact.value_counts(normalize=True).round(4).to_dict()

# ---- layered policies on test
sev, dep = D.declared_severity.values, D.recent_deployment.values
rule_sev = np.isin(sev, ["P1", "P2"]); rule_chg = rule_sev | (dep == 1)
high, med = (D.risk == "High").values, (D.risk == "Medium").values
hi_imp = (D.impact == "High impact").values
policies = {
    "A. Current proxy: escalate P1/P2": rule_sev,
    "B. Rule change: P1/P2 or recent deployment": rule_chg,
    "C. Model only: High band (top 10%)": high,
    "D. Model only: same volume as rule B": D.score.values >= np.quantile(s_va, 1 - (df.loc[va, "declared_severity"].isin(["P1", "P2"]) | (df.loc[va, "recent_deployment"] == 1)).mean()),
    "E. Escalate everything": np.ones(len(D), bool),
}
yv = D.y.values; pen = D.credit_penalty_usd.values
rows = []
for k, f in policies.items():
    tp = (f & (yv == 1)).sum(); fp = (f & (yv == 0)).sum()
    rows.append({"policy": k, "escalated_pct": 100 * f.mean(), "per_day": f.sum() / days, "caught": int(tp), "false_alarms": int(fp),
                 "missed": int(((~f) & (yv == 1)).sum()), "recall": tp / yv.sum(), "precision": tp / max(f.sum(), 1),
                 "penalty_share": pen[f].sum() / pen.sum()})
pol = pd.DataFrame(rows); R["policies"] = pol.round(4).to_dict(orient="records")
# incremental value of the model on top of the rule
R["high_band_mix"] = D.loc[high, "declared_severity"].value_counts().to_dict(); R["high_band_deploy"] = int(D.loc[high, "recent_deployment"].sum())
add = high & ~rule_chg
R["model_adds_beyond_rule"] = {"incidents": int(add.sum()), "per_day": round(add.sum() / days, 2), "breaches": int((add & (yv == 1)).sum()),
                               "precision": round((add & (yv == 1)).sum() / max(add.sum(), 1), 4)}
# breaches not caught by recommended policy D: what do they look like?
missD = (~(rule_chg | high)) & (yv == 1)
R["missed_by_D"] = {"n": int(missD.sum()), "share_of_breaches": round(missD.sum() / yv.sum(), 4),
                    "severity_mix": D.loc[missD, "declared_severity"].value_counts(normalize=True).round(3).to_dict()}
# bootstrap: recall gain of D over B
rng = np.random.default_rng(SEED); gains = []
fD, fB = rule_chg | high, rule_chg
for _ in range(1000):
    i = rng.integers(0, len(yv), len(yv)); yb = yv[i]
    gains.append(((fD[i] & (yb == 1)).sum() - (fB[i] & (yb == 1)).sum()) / yb.sum())
R["recall_gain_D_over_B_ci"] = [round(float(np.percentile(gains, 2.5)), 4), round(float(np.percentile(gains, 97.5)), 4)]

# ---- where the model adds value: ordering inside the escalation pool and inside the non-escalated pool
pool = rule_chg; rest = ~rule_chg
R["pool_auc"] = round(roc_auc_score(yv[pool], D.score.values[pool]), 4)
R["rest_auc"] = round(roc_auc_score(yv[rest], D.score.values[rest]), 4)
R["pool_size"] = int(pool.sum()); R["rest_size"] = int(rest.sum())
# inside the pool: priority tier = model High band (all High-band incidents are in the pool)
pr = pool & high; st = pool & ~high
R["pool_priority"] = {"n": int(pr.sum()), "per_day": round(pr.sum() / days, 2), "breach_rate": round(yv[pr].mean(), 4), "share_of_pool_breaches": round(yv[pr].sum() / yv[pool].sum(), 4)}
R["pool_standard"] = {"n": int(st.sum()), "per_day": round(st.sum() / days, 2), "breach_rate": round(yv[st].mean(), 4)}
# P1 split by model band
p1 = sev == "P1"
R["p1_by_band"] = {b: [int((p1 & (D.risk.values == b)).sum()), round(float(yv[p1 & (D.risk.values == b)].mean()), 4) if (p1 & (D.risk.values == b)).sum() else None] for b in ["High", "Medium", "Low"]}
# inside the non-escalated pool: breach rate by model score fifths (cut points from validation non-escalated incidents)
Vd = df[va]; vrest = ~(Vd.declared_severity.isin(["P1", "P2"]) | (Vd.recent_deployment == 1)).values
cuts = np.quantile(s_va[vrest], [0.2, 0.4, 0.6, 0.8])
q = np.searchsorted(cuts, D.score.values[rest], side="right")
R["rest_by_fifth"] = [{"fifth": int(k + 1), "n": int((q == k).sum()), "breach_rate": round(float(yv[rest][q == k].mean()), 4)} for k in range(5)]
R["rest_breach_rate"] = round(float(yv[rest].mean()), 4)
R["rest_top40_share_of_rest_breaches"] = round(float(yv[rest][q >= 3].sum() / yv[rest].sum()), 4)
# capacity-limited: only 20% of all incidents can get full escalation; choose from pool by model score vs by severity order
k = int(round(0.20 * len(D)))
idx_pool = np.where(pool)[0]
by_model = idx_pool[np.argsort(-D.score.values[idx_pool])][:k]
sev_rank = pd.Series(sev[idx_pool]).map({"P1": 0, "P2": 1, "P3": 2, "P4": 3}).values * 2 + (1 - dep[idx_pool])
rng2 = np.random.default_rng(SEED); by_sev = idx_pool[np.lexsort((rng2.random(len(idx_pool)), sev_rank))][:k]
R["capacity20"] = {"k": k, "per_day": round(k / days, 2), "recall_by_model": round(float(yv[by_model].sum() / yv.sum()), 4),
                   "recall_by_severity_order": round(float(yv[by_sev].sum() / yv.sum()), 4),
                   "precision_by_model": round(float(yv[by_model].mean()), 4), "precision_by_severity_order": round(float(yv[by_sev].mean()), 4)}
gains2 = []
for _ in range(1000):
    i = rng.integers(0, len(idx_pool), len(idx_pool))
    yy = yv[idx_pool][i]; sc = D.score.values[idx_pool][i]; sr = sev_rank[i]
    kk = int(round(k * len(i) / len(idx_pool)))
    a = yy[np.argsort(-sc)][:kk].sum(); b = yy[np.lexsort((rng.random(len(i)), sr))][:kk].sum()
    gains2.append((a - b) / max(yv.sum(), 1))
R["capacity20_gain_ci"] = [round(float(np.percentile(gains2, 2.5)), 4), round(float(np.percentile(gains2, 97.5)), 4)]

# ---- decision routes (mutually exclusive, P1 first)
route = np.where(sev == "P1", "1 Major incident route", np.where(rule_chg & high, "2 Priority escalation", np.where(rule_chg, "3 Standard escalation", "4 Normal queue + SLA-clock alerts")))
rt = []
for r_ in sorted(set(route)):
    m = route == r_
    rt.append({"route": r_, "n": int(m.sum()), "share_pct": round(100 * m.mean(), 2), "per_day": round(m.sum() / days, 2),
               "breach_rate": round(float(yv[m].mean()), 4), "share_of_breaches": round(float(yv[m].sum() / yv.sum()), 4),
               "high_impact_share": round(float(hi_imp[m].mean()), 4)})
R["routes"] = rt

# ---- monitoring baselines: monthly metrics on validation + test for control limits
D2 = df[va | te].copy(); D2["score"] = np.r_[s_va, s_te] if False else np.nan
D2.loc[va[va | te].values if False else D2.index.isin(df.index[va]), "score"] = s_va
D2.loc[D2.index.isin(df.index[te]), "score"] = s_te
D2["month"] = D2.opened_at.dt.to_period("M")
mon = []
for m, g in D2.groupby("month"):
    f = (g.score >= t_high) | g.declared_severity.isin(["P1", "P2"]) | (g.recent_deployment == 1)
    mon.append({"month": str(m), "n": len(g), "breach_rate": g[T].mean(), "mean_score": g.score.mean(),
                "auc": roc_auc_score(g[T], g.score), "recall_D": (f & (g[T] == 1)).sum() / g[T].sum(), "escalated_pct_D": 100 * f.mean()})
mon = pd.DataFrame(mon); R["monthly"] = mon.round(4).to_dict(orient="records")
R["monthly_summary"] = {"auc_mean": round(mon.auc.mean(), 4), "auc_sd": round(mon.auc.std(), 4), "auc_min": round(mon.auc.min(), 4),
                        "recall_D_mean": round(mon.recall_D.mean(), 4), "recall_D_sd": round(mon.recall_D.std(), 4), "recall_D_min": round(mon.recall_D.min(), 4),
                        "calib_gap_max": round(float((mon.mean_score - mon.breach_rate).abs().max()), 4)}

# ---- figures
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e5e0", "axes.axisbelow": True, "legend.frameon": False, "figure.dpi": 200})
BLUE, ORANGE, AQUA, YELLOW, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8984"
fig, ax = plt.subplots(figsize=(6.6, 3.3))
lbl = ["A. Escalate P1/P2", "B. P1/P2 or recent change", "C. Model top 10%", "D. Model, same volume as B", "E. Escalate everything"]
cols = [GREY, ORANGE, YELLOW, BLUE, GREY]; offs = [(8, -4), (-60, 10), (8, -4), (8, -14), (-110, -14)]
for (_, r), c, l, o in zip(pol.iterrows(), cols, lbl, offs):
    ax.scatter(r.escalated_pct, r.recall * 100, s=70, color=c, edgecolor="white", linewidth=1.5, zorder=5)
    ax.annotate(l, (r.escalated_pct, r.recall * 100), textcoords="offset points", xytext=o, fontsize=7.5, color="#2b2b29")
ax.plot([0, 100], [0, 100], color="#c3c2b7", lw=1); ax.text(70, 62, "Random selection", fontsize=7, color="#8a8984", rotation=27)
ax.set(xlabel="Incidents escalated at logging (%)", ylabel="Breaches caught (%)", xlim=(0, 105), ylim=(0, 105))
fig.tight_layout(); fig.savefig(f"{OUT}/p3_fig1_policies.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(6.6, 2.8))
ax.plot(mon.month, mon.auc, marker="o", color=BLUE, lw=1.6, label="ROC-AUC (model)")
ax.plot(mon.month, mon.recall_D, marker="s", color=ORANGE, lw=1.6, label="Breaches caught by recommended rule")
ax.plot(mon.month, mon.breach_rate, marker="^", color=GREY, lw=1.2, ls="--", label="Breach rate")
ax.axvline(6.5, color="#c3c2b7", lw=1); ax.text(3, 0.9, "Validation", ha="center", fontsize=7.5, color="#52514e"); ax.text(10.5, 0.9, "Test", ha="center", fontsize=7.5, color="#52514e")
ax.set(ylim=(0, 1), ylabel="Value"); ax.tick_params(axis="x", labelsize=7, rotation=45)
ax.legend(fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
fig.tight_layout(); fig.savefig(f"{OUT}/p3_fig2_monthly.png"); plt.close(fig)

# Decision flow diagram (values from this run)
from matplotlib.patches import FancyBboxPatch
fig, ax = plt.subplots(figsize=(6.8, 4.3)); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
def box(x, y, w, h, text, fc, ec="#52514e", bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5", fc=fc, ec=ec, lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.2, fontweight="bold" if bold else "normal", color="#1f1f1d", wrap=True)
def arrow(x1, y1, x2, y2, t=""):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color="#52514e", lw=1))
    if t: ax.text((x1 + x2) / 2 + 1, (y1 + y2) / 2 + 1, t, fontsize=6.8, color="#52514e")
box(30, 89, 40, 8, "Incident logged\n(severity, change status, model score)", "#e8eef7", bold=True)
box(2, 66, 28, 12, "Severity P1?", "#f3f2ee")
box(36, 66, 28, 12, "P2, or recent change /\ndeployment on service?", "#f3f2ee")
box(70, 66, 28, 12, "Otherwise\n(P3/P4, no recent change)", "#f3f2ee")
arrow(50, 89, 16, 78.5); arrow(50, 89, 50, 78.5); arrow(50, 89, 84, 78.5)
pp, ps = R["pool_priority"], R["pool_standard"]
box(2, 38, 28, 18, f"MAJOR INCIDENT ROUTE\nFunctional + hierarchical\nescalation, bridge call,\ncustomer notification", "#fbe3d9")
box(36, 46, 28, 10, f"Model score in High band?", "#e8eef7")
box(70, 38, 28, 18, f"NORMAL QUEUE\nSLA-clock alerts at 50%\nand 75% of target;\nescalate on alert", "#e6f4ec")
arrow(16, 66, 16, 56.5); arrow(50, 66, 50, 56.5); arrow(84, 66, 84, 56.5)
box(30, 14, 20, 22, f"PRIORITY\nESCALATION\nSenior resolver\nnow; check change\nfor back-out", "#fbe3d9")
box(54, 14, 20, 22, f"STANDARD\nESCALATION\nSpecialist resolver\ngroup; change owner\ninformed", "#fdf1d6")
arrow(44, 46, 40, 36.5, "yes"); arrow(56, 46, 64, 36.5, "no")
ax.text(50, 4, "Human decision at every step: the incident manager can override any route and records the reason.", ha="center", fontsize=7, style="italic", color="#52514e")
fig.tight_layout(); fig.savefig(f"{OUT}/p3_fig3_flow.png"); plt.close(fig)

# Roadmap chart (proposed timeline, months from approval)
tasks = [("Approve rule change and RACI", 0, 1, ORANGE), ("Link incidents to change records", 0, 2, ORANGE), ("SLA-clock alerts for normal queue", 0.5, 2, ORANGE),
         ("Baseline KPIs and dashboard", 1, 3, BLUE), ("Retrain model on real records", 2, 4, BLUE), ("Controlled pilot of model tiering", 3, 6, BLUE),
         ("Go / no-go review of model", 6, 6.5, YELLOW), ("Capture escalation actions and SLA targets", 3, 9, AQUA), ("In-flight re-scoring model (if justified)", 7, 12, AQUA)]
fig, ax = plt.subplots(figsize=(6.8, 3.2))
for i, (t, a, b, c) in enumerate(tasks[::-1]):
    ax.barh(i, b - a, left=a, color=c, height=0.55); ax.text(-0.15, i, t, ha="right", va="center", fontsize=7.5)
ax.set_yticks([]); ax.set_xlim(0, 12.2); ax.set_xticks(range(0, 13, 1)); ax.set_xlabel("Months from approval (proposed)")
ax.grid(axis="y", visible=False); ax.spines["left"].set_visible(False)
fig.subplots_adjust(left=0.38, right=0.98, bottom=0.17, top=0.97); fig.savefig(f"{OUT}/p3_fig4_roadmap.png"); plt.close(fig)

json.dump(R, open(f"{OUT}/phase3_results.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print(json.dumps({k: R[k] for k in ["routes", "pool_auc", "rest_auc", "pool_size", "rest_size", "pool_priority", "pool_standard", "p1_by_band", "rest_by_fifth", "rest_breach_rate", "rest_top40_share_of_rest_breaches", "capacity20", "capacity20_gain_ci", "check_test_auc", "check_test_ap", "thresholds", "test_days", "test_incidents_per_day", "model_adds_beyond_rule", "missed_by_D", "recall_gain_D_over_B_ci", "monthly_summary", "breach_rate_by_impact_all", "impact_share_all"]}, indent=1))
print(risk_tab.round(3).to_string()); print(mat.round(3).to_string()); print(pol.round(3).to_string())
for k in ["penalty_per_breach_by_impact", "penalty_per_breach_by_tier", "penalty_per_breach_by_severity"]: print(k, R[k])
print(mon.round(3).to_string())
