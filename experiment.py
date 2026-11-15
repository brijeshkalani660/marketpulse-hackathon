"""Honest test of 12-month stock estimates. Run: python experiment.py
Data: monthly_closes_sample.csv (400 randomly chosen US stocks with 9+ years of history). Past -> future split by DATE, never at random.
Train: predictions made up to Dec 2019 (outcome known by Dec 2020).  Test: predictions made Jan 2021 - Nov 2025 (outcome known by Nov 2026)."""
import math, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss

df = pd.read_csv("monthly_closes_sample.csv")
df["t"] = df.month.str[:4].astype(int) * 12 + df.month.str[5:7].astype(int) - 1
S = {k: dict(zip(g.t, g.close)) for k, g in df.groupby("ticker")}
Z = 1.2816

def model_a(P, K=240, win=120):
    """Our model: average monthly drift and swings from the stock's own past, pulled toward a modest typical value
    (more history = less pull), drift softly capped. Returns (mu per month, sigma per sqrt-month)."""
    pts = sorted(P)
    T = pts[-1]; pts = [t for t in pts if t >= T - win]
    if len(pts) < 3: return None
    R, G = [], []
    for a, b in zip(pts, pts[1:]):
        g = b - a
        if 1 <= g <= 12: R.append(math.log(P[b] / P[a])); G.append(g)
    n = sum(G)
    if n < 6: return None
    mh = sum(R) / n; s2 = sum((r - mh * g) ** 2 for r, g in zip(R, G)) / n
    w = n / (n + K); x = w * mh + (1 - w) * 0.005
    mu = 0.002 + 0.006 * math.tanh((x - 0.002) / 0.006)
    ws = n / (n + 36); sg = min(0.4, max(0.015, ws * math.sqrt(s2) + (1 - ws) * 0.09))
    return mu, sg

rows = []
for tk, P in S.items():
    ts = sorted(P)
    for t in ts:
        if t % 3: continue                       # every 3rd month, fewer overlapping rows
        if (t + 12) not in P or (t - 12) not in P: continue
        past = {u: P[u] for u in ts if u <= t}   # ONLY data up to the prediction date
        m = model_a(past)
        if not m: continue
        r3 = P[t] / P[t - 3] - 1 if (t - 3) in P else np.nan
        prev = [P[u] for u in range(t - 10, t) if u in P]
        d10 = P[t] / (sum(prev) / len(prev)) - 1 if len(prev) >= 2 else np.nan
        lr = [math.log(P[b] / P[a]) for a, b in zip(sorted(past)[-37:], sorted(past)[-36:]) if b - a == 1 and a in P and b in P]
        vol = float(np.std(lr)) if len(lr) >= 8 else np.nan
        rows.append(dict(tk=tk, t=t, mu=m[0], sg=m[1], r12=P[t] / P[t - 12] - 1, r3=r3, d10=d10, vol=vol,
                         y12=P[t + 12] / P[t] - 1))
D = pd.DataFrame(rows).dropna()
D["up"] = (D.y12 > 0).astype(int)
cut = 2019 * 12 + 11
tr = D[D.t <= cut]
te = D[(D.t > 2020 * 12 + 11) & (D.t <= 2025 * 12 + 10)]
print(f"rows: train {len(tr):,}  test {len(te):,}  ({te.tk.nunique()} stocks in test)")

out = {}
base_up = te.up.mean()
out["Test periods where price rose 12 months later"] = f"{base_up:.1%}"
out["Baseline 1: always predict up"] = f"{base_up:.1%}"
out["Baseline 2: same direction as last 12 months"] = f"{((te.r12 > 0).astype(int) == te.up).mean():.1%}"
out["Our model (drift estimate) direction"] = f"{(((te.mu > 0.0035).astype(int)) == te.up).mean():.1%}"
feat = ["r12", "r3", "d10", "vol"]
gb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, random_state=0).fit(tr[feat], tr.up)
p = gb.predict_proba(te[feat])[:, 1]
out["Machine-learning model (gradient boosting) direction"] = f"{((p > 0.5).astype(int) == te.up).mean():.1%}"
out["Gradient boosting AUC (0.5 = coin flip)"] = f"{roc_auc_score(te.up, p):.3f}"
out["Brier score: always-up / gradient boosting (lower is better)"] = f"{brier_score_loss(te.up, np.full(len(te), base_up)):.3f} / {brier_score_loss(te.up, p):.3f}"

# 12-month likely range: does the real result land inside the 80% range?
lo = np.exp(te.mu * 12 - Z * te.sg * math.sqrt(12)) - 1; hi = np.exp(te.mu * 12 + Z * te.sg * math.sqrt(12)) - 1
inside = ((te.y12 >= lo) & (te.y12 <= hi))
out["Our 80% range: real result inside it (target 80%)"] = f"{inside.mean():.1%}"
out["Our 80% range: average width"] = f"{(hi - lo).mean():.0%} of price"
# naive range: one fixed width for every stock, taken from the training period
q = tr.y12.quantile([0.1, 0.9]).values
naive = ((te.y12 >= q[0]) & (te.y12 <= q[1])).mean()
out["Naive fixed range (same for every stock): inside"] = f"{naive:.1%} (width {(q[1]-q[0]):.0%})"
# error of the middle estimate vs 'no change'
mid = np.exp(te.mu * 12) - 1
out["Middle estimate error (mean absolute): ours / 'no change'"] = f"{(te.y12 - mid).abs().mean():.1%} / {te.y12.abs().mean():.1%}"
# where it fails
te = te.assign(inside=inside.values, year=(te.t // 12))
out["Range coverage by year"] = ", ".join(f"{y}: {g.inside.mean():.0%}" for y, g in te.groupby("year"))
hv = te.vol > te.vol.quantile(0.8)
out["Range coverage, most volatile 20% of stocks"] = f"{te.inside[hv].mean():.1%}"
for k, v in out.items(): print(f"{k}: {v}")
pd.Series(out).to_csv("results.csv", header=["value"])
