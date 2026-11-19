"""Analysis of revision_results_raw.csv  ->  Tables for the revised manuscript.
Unit of analysis = language (n=11); seeds (and k) are averaged within language and resampled in a
hierarchical bootstrap (languages, then seeds/k within language). Exact Wilcoxon + Holm correction.
Usage: python analyze_revision.py revision_results_raw.csv [outdir]"""
import sys, itertools, numpy as np, pandas as pd
from scipy import stats
src = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else "."
df = pd.read_csv(src); rng = np.random.default_rng(0)
LANGS = sorted(df.lang.unique()); MODELS = [m for m in ["mBERT", "XLM-R", "MuRIL", "IndicBERT"] if m in set(df.model)]
ft = df[df.regime == "finetune"].copy(); zs = df[df.regime == "zero_shot"].copy()
for c in ("lalf", "wce"): ft[c] = ft[c].astype(int)

def per_lang(frame, col):                      # list over languages of arrays of F1 (over seed[,k])
    return [frame[frame.lang == l].sort_values(["seed", "k"])[col].values for l in LANGS]
def hboot(diffs, n=10000):
    L = len(diffs); r = np.empty(n)
    for i in range(n):
        li = rng.integers(0, L, L)
        r[i] = np.mean([diffs[j][rng.integers(0, len(diffs[j]), len(diffs[j]))].mean() for j in li])
    return np.percentile(r, [2.5, 97.5])
def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); a = np.empty(m); run = 0
    for r, i in enumerate(o): run = max(run, (m - r) * p[i]); a[i] = min(1, run)
    return a
def contrast(a_arrs, b_arrs):
    d = [x - y for x, y in zip(a_arrs, b_arrs)]; lm = np.array([v.mean() for v in d]) * 100
    ci = hboot(d) * 100
    try: w = stats.wilcoxon(lm).pvalue
    except ValueError: w = 1.0
    return dict(mean_diff=lm.mean(), ci_lo=ci[0], ci_hi=ci[1], langs_A_better=int((lm > 0).sum()), n_langs=len(lm), wilcoxon_p=w)

def cell(frame, model, init, lalf, wce, col):
    s = frame[(frame.model == model) & (frame.init == init) & (frame.lalf == lalf) & (frame.wce == wce)]
    return per_lang(s, col)

res = {}
# ---------- 1. factorial table (mean F1, collapse count) on each test set ----------
for tset in ("naam500", "wikiann"):
    col = f"{tset}_f1"; rows = []
    for m in MODELS:
        for init, lalf, wce in itertools.product(["base", "zs"], [0, 4], [0, 1]):
            s = ft[(ft.model == m) & (ft.init == init) & (ft.lalf == lalf) & (ft.wce == wce)]
            if s.empty: continue
            rows.append(dict(model=m, init=init, LALF=lalf, WCE=wce, mean_f1=100 * s[col].mean(),
                             collapsed_runs=int((s[col] < 0.01).sum()), runs=len(s), retries=int(s.retries.sum())))
    res[f"factorial_{tset}"] = pd.DataFrame(rows)

# ---------- 2. main effects (paired, other factors fixed at every combination, then pooled over languages) ----------
rows = []
for tset in ("naam500", "wikiann"):
    col = f"{tset}_f1"
    for m in MODELS:
        for fac, (lo, hi) in dict(warm_start=("base", "zs"), LALF=(0, 4), WCE=(0, 1)).items():
            others = [c for c in itertools.product(["base", "zs"], [0, 4], [0, 1])]
            def sel(level):
                key = lambda c: {"warm_start": c[0], "LALF": c[1], "WCE": c[2]}[fac]
                return [c for c in others if key(c) == level]
            A, B = [], []
            for ca, cb in zip(sel(hi), sel(lo)):          # matched cells: only `fac` differs
                A.append(cell(ft, m, *ca, col)); B.append(cell(ft, m, *cb, col))
            # average matched cells (equal weights) -> one array per language
            a = [np.mean([x[i] for x in A], axis=0) for i in range(len(LANGS))]
            b = [np.mean([x[i] for x in B], axis=0) for i in range(len(LANGS))]
            rows.append(dict(test_set=tset, model=m, factor=fac, **contrast(a, b)))
me = pd.DataFrame(rows); me["wilcoxon_holm"] = me.groupby("test_set").wilcoxon_p.transform(lambda p: holm(p.values)); res["main_effects"] = me

# ---------- 3. COMMON-EVALUATION regime table: zero-shot vs direct vs sequential on the SAME test sets ----------
rows = []
for tset in ("naam500", "naamfull", "wikiann"):
    col = f"{tset}_f1"
    if col not in df: continue
    for m in MODELS:
        z = [zs[(zs.model == m) & (zs.lang == l)].sort_values("seed")[col].values for l in LANGS]
        d = cell(ft, m, "base", 4, 1, col); s = cell(ft, m, "zs", 4, 1, col)
        rows.append(dict(test_set=tset, model=m, zero_shot=100 * np.mean([x.mean() for x in z]),
                         direct=100 * np.mean([x.mean() for x in d]), sequential=100 * np.mean([x.mean() for x in s]),
                         **{"seq_minus_zs_" + k: v for k, v in
                            contrast([np.repeat(x.mean(), 1) for x in [np.concatenate([v]) for v in s]],
                                     [np.repeat(x.mean(), 1) for x in z]).items()}))
res["regimes_common_eval"] = pd.DataFrame(rows)

pd.set_option("display.width", 250); pd.set_option("display.float_format", lambda v: f"{v:.3f}")
for k, v in res.items():
    v.to_csv(f"{out}/{k}.csv", index=False); print(f"\n== {k} =="); print(v.to_string(index=False))
