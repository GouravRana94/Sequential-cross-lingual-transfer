"""Revised significance analysis. Unit of analysis = language (n=11), seeds/k averaged within language.
Outputs: mean diff, 95% hierarchical-bootstrap CI (languages, then seeds within language), exact
Wilcoxon signed-rank p, exact sign-flip permutation p, #languages favouring A, Holm-adjusted p."""
import pandas as pd, numpy as np, itertools
from scipy import stats
d=pd.read_csv('original_runs_parsed.csv')
LANGS=['as','bn','gu','hi','kn','ml','mr','or','pa','ta','te']
MODELS=['mBERT','XLM-R','MuRIL','IndicBERT']
rng=np.random.default_rng(0)

def tidy(exp):
    x=d[d.exp==exp].copy()
    if exp=='zs': x['lang']=x.eval_on
    x=x[x.lang.isin(LANGS)]
    return x

def cell_arrays(x, a, b):
    # array [lang, unit(seed or seed×k)] for paired comparison, pairing on seed (and k when present)
    ka=['seed'] if x.k.isna().all() else ['seed','k']
    out=[]
    for l in LANGS:
        xa=x[(x.model==a)&(x.lang==l)].set_index(ka).f1.sort_index()
        xb=x[(x.model==b)&(x.lang==l)].set_index(ka).f1.sort_index()
        out.append((xa.values-xb.values))
    return out  # list of per-language paired differences

def hier_boot(diffs, n=10000):
    L=len(diffs); res=np.empty(n)
    for i in range(n):
        li=rng.integers(0,L,L)
        res[i]=np.mean([ diffs[j][rng.integers(0,len(diffs[j]),len(diffs[j]))].mean() for j in li])
    return np.percentile(res,[2.5,97.5])

def signflip_p(v):
    # exact two-sided sign-flip permutation test on language-level mean differences
    v=np.asarray(v); n=len(v); obs=abs(v.mean()); cnt=0
    for signs in itertools.product([1,-1],repeat=n):
        if abs((v*np.array(signs)).mean())>=obs-1e-12: cnt+=1
    return cnt/2**n

def holm(ps):
    ps=np.array(ps); order=np.argsort(ps); m=len(ps); adj=np.empty(m); run=0
    for r,i in enumerate(order):
        run=max(run,(m-r)*ps[i]); adj[i]=min(1,run)
    return adj

def compare(x, pairs, label):
    rows=[]
    for a,b in pairs:
        diffs=cell_arrays(x,a,b); lm=np.array([v.mean() for v in diffs])*100
        ci=hier_boot(diffs)*100
        try: w=stats.wilcoxon(lm,alternative='two-sided').pvalue
        except ValueError: w=1.0
        rows.append(dict(set=label,A=a,B=b,mean_diff=lm.mean(),ci_lo=ci[0],ci_hi=ci[1],
                         n_lang_A_wins=int((lm>0).sum()),n_lang_ties=int((lm==0).sum()),
                         wilcoxon_p=w,signflip_p=signflip_p(lm)))
    r=pd.DataFrame(rows); r['wilcoxon_holm']=holm(r.wilcoxon_p.values); r['signflip_holm']=holm(r.signflip_p.values)
    return r

pairs=list(itertools.combinations(['MuRIL','IndicBERT','XLM-R','mBERT'],2))
out=[]
for exp,lab in [('zs','Zero-shot'),('fs','Direct few-shot'),('seq','Sequential')]:
    out.append(compare(tidy(exp),pairs,lab))
# regime effect within model: sequential minus direct (pair on seed,k, lang)
fs=tidy('fs'); sq=tidy('seq'); rows=[]
for m in MODELS:
    diffs=[]
    for l in LANGS:
        a=sq[(sq.model==m)&(sq.lang==l)].set_index(['seed','k']).f1.sort_index()
        b=fs[(fs.model==m)&(fs.lang==l)].set_index(['seed','k']).f1.sort_index()
        diffs.append(a.values-b.values)
    lm=np.array([v.mean() for v in diffs])*100; ci=hier_boot(diffs)*100
    rows.append(dict(set='Seq - Direct',A=m,B='(same model)',mean_diff=lm.mean(),ci_lo=ci[0],ci_hi=ci[1],
                     n_lang_A_wins=int((lm>0).sum()),n_lang_ties=int((lm==0).sum()),
                     wilcoxon_p=stats.wilcoxon(lm).pvalue,signflip_p=signflip_p(lm)))
r=pd.DataFrame(rows); r['wilcoxon_holm']=holm(r.wilcoxon_p.values); r['signflip_holm']=holm(r.signflip_p.values); out.append(r)
res=pd.concat(out); res.to_csv('revised_significance.csv',index=False)
pd.set_option('display.width',250); pd.set_option('display.float_format',lambda v:f'{v:.4f}')
print(res.to_string(index=False))

# per-model mean F1 with 95% CI (bootstrap over languages, then seeds)
def model_ci(x,m,n=10000):
    ka=['seed'] if x.k.isna().all() else ['seed','k']
    per=[x[(x.model==m)&(x.lang==l)].f1.values for l in LANGS]
    res=np.empty(n)
    for i in range(n):
        li=rng.integers(0,len(per),len(per))
        res[i]=np.mean([per[j][rng.integers(0,len(per[j]),len(per[j]))].mean() for j in li])
    return np.mean([p.mean() for p in per])*100, *np.percentile(res,[2.5,97.5])*100
rows=[]
for exp,lab in [('zs','Zero-shot'),('fs','Direct few-shot'),('seq','Sequential')]:
    x=tidy(exp)
    for m in MODELS:
        mu,lo,hi=model_ci(x,m); rows.append(dict(regime=lab,model=m,mean=mu,ci_lo=lo,ci_hi=hi))
mc=pd.DataFrame(rows); mc.to_csv('model_mean_ci.csv',index=False); print(); print(mc.to_string(index=False))
