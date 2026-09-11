#!/usr/bin/env python3
"""Independent scientific invariants and artifact/split checks."""
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel

from extract_features import NAMES, DAYS, features
from analyze import decompose, mmd_unbiased

root=Path(__file__).parent
out=root/"results"

# Padding must not create negative IATs/bursts, and a final reversed timestamp
# must use the maximum observation as endpoint without reordering directions.
v,a=features(np.array([.1,.2,-.3,-.4,.5,0.,0.]))
f=dict(zip(NAMES,v))
assert f["n_packets"]==5 and f["burst_count"]==3
assert np.isclose(f["duration"],.4) and np.isclose(f["iat_mean"],.1)
assert f["switch_rate"]==.5 and np.isclose(f["positive_fraction"],.6)
v,a=features(np.array([.1,-.3,.2,0.]))
f=dict(zip(NAMES,v))
assert a["negative_iat_count"]==1 and f["iat_zero_fraction"]==.5
assert np.isclose(f["duration"],.2)

# A shared translation has exactly zero class residual and known energy.
rng=np.random.default_rng(731)
source=[rng.normal(size=(100,3))+j for j in range(5)]
shift=np.array([.2,-.5,.3])
target=[x+shift for x in source]
d,delta,local,_=decompose(source,target,rng,False)
assert np.allclose(delta,shift) and np.allclose(local,0,atol=1e-12)
assert np.isclose(d["global_raw"],shift@shift)
assert np.isclose(d["total_raw"],d["global_raw"]+d["local_raw"])

# Independently calculate the fixed-quota MMD using explicit class pairs.
xx=rng.normal(size=(12,2)); yy=rng.normal(size=(12,2))+.3
C,m=3,4
def class_kernel_mean(x,y,same_domain):
    terms=[]
    for c in range(C):
        for j in range(C):
            k=rbf_kernel(x[c*m:(c+1)*m],y[j*m:(j+1)*m],gamma=.4)
            terms.append((k.sum()-np.trace(k))/(m*(m-1)) if same_domain and c==j else k.mean())
    return np.mean(terms)
expected=class_kernel_mean(xx,xx,True)+class_kernel_mean(yy,yy,True)-2*class_kernel_mean(xx,yy,False)
assert np.isclose(mmd_unbiased(xx,yy,.4,classes=C),expected,atol=1e-12)

manifest=json.loads((out/"manifest.json").read_text())
assert sum(a["shape"][0] for a in manifest["audits"])==118068
assert manifest["duplicate_rows"]==0
for day in DAYS:
    f=pd.read_csv(out/f"features_day{day}.csv")
    a=json.loads((out/f"audit_day{day}.json").read_text())
    assert len(f)==a["shape"][0] and f.label.nunique()==102
    assert np.isfinite(f[NAMES].to_numpy()).all()
    assert ((f["iat_mean"]*(f.n_packets-1)-f.duration).abs()<1e-6).all()
for d in json.loads((out/"decomposition.json").read_text()):
    assert np.isclose(d["total_corrected"],d["global_corrected"]+d["local_corrected"])

splits=np.load(out/"baselines_predictions_and_splits.npz")
for day in DAYS[1:]:
    f=pd.read_csv(out/f"features_day{day}.csv").set_index("row_id")
    for seed in [11,29,47]:
        p=f"day{day}_seed{seed}_"
        adapt,test=splits[p+"adapt_row_id"],splits[p+"test_row_id"]
        assert not np.intersect1d(adapt,test).size
        for k in [1,5]:
            support=splits[p+f"{k}shot_support_row_id"]
            assert len(support)==102*k and np.isin(support,adapt).all()
            assert (f.loc[support].label.value_counts()==k).all()

report=(root/"REPORT.md").read_text()
for link in re.findall(r"\]\(([^)]+)\)",report):
    if not link.startswith("http"):
        assert (root/link).exists(),link
print("PASS: feature invariants, stratified MMD, decomposition, full-row counts, 12 target splits, support budgets, report links")
