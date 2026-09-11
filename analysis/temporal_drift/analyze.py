#!/usr/bin/env python3
"""Exploratory full-data statistics; target labels are used ONLY for diagnosis.

All p-values assume independent traces within class/day; missing session metadata
prevents clustered inference. Gaussian multiplier bootstrap preserves the estimated
covariance of each class/day mean. No causal interpretation is made.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

from extract_features import CORE, DAYS, NAMES, TIME, transform

ROOT = Path(__file__).parent
OUT = ROOT / "results"
FIG = ROOT / "figures"
SEED = 20260910
B = 1999


def bh(p):
    p = np.asarray(p)
    order = np.argsort(p)
    q = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p)+1))[::-1])[::-1]
    result = np.empty_like(q)
    result[order] = np.minimum(q, 1)
    return result


def holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    q = np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1))
    result = np.empty_like(q)
    result[order] = np.minimum(q, 1)
    return result


def class_arrays(z, labels):
    return [z[labels == c] for c in sorted(np.unique(labels))]


def decompose(a, b, rng, bootstrap=True):
    C, d = len(a), a[0].shape[1]
    ma, mb = np.array([x.mean(0) for x in a]), np.array([x.mean(0) for x in b])
    delta = mb-ma
    g = delta.mean(0)
    local = delta-g
    V = np.array([np.atleast_2d(np.cov(x, rowvar=False, ddof=1))/len(x) +
                  np.atleast_2d(np.cov(y, rowvar=False, ddof=1))/len(y) for x,y in zip(a,b)])
    noise_t = np.trace(V, axis1=1, axis2=2).mean()
    noise_g = noise_t/C
    T, G, L = np.square(delta).sum(1).mean(), g@g, np.square(local).sum(1).mean()
    out = dict(total_raw=T, global_raw=G, local_raw=L,
               total_noise=noise_t, global_noise=noise_g, local_noise=noise_t-noise_g,
               total_corrected=T-noise_t, global_corrected=G-noise_g,
               local_corrected=L-noise_t+noise_g,
               global_fraction=(G-noise_g)/(T-noise_t),
               same_direction_fraction=float(np.mean(delta@g > 0)),
               delta_singular_energy_top1=float(np.linalg.svd(delta, compute_uv=False)[0]**2/np.square(delta).sum()))
    if not bootstrap:
        return out, delta, local, None
    # Equivalent to normal multiplier draws of centered cell observations, with
    # finite-sample n/(n-1) correction. Covariance uncertainty is not resampled.
    eig, vec = np.linalg.eigh(V)
    roots = vec * np.sqrt(np.maximum(eig, 0))[:,None,:]
    eps = np.einsum("bck,cdk->bcd", rng.normal(size=(B,C,d)), roots)
    eg = eps.mean(1)
    el = eps-eg[:,None,:]
    ng = np.square(eg).sum(1)
    nl = np.square(el).sum(2).mean(1)
    out["p_global"] = (1+np.count_nonzero(ng >= G))/(B+1)
    out["p_local"] = (1+np.count_nonzero(nl >= L))/(B+1)
    class_p = (1+(np.square(el).sum(2) >= np.square(local).sum(1)[None,:]).sum(0))/(B+1)
    # Centered bootstrap of noise-corrected energy. Intervals are approximate.
    gs = np.square(g+eg).sum(1)-2*noise_g
    ls = np.square(local[None,:,:]+el).sum(2).mean(1)-2*(noise_t-noise_g)
    out["global_ci95"] = np.quantile(gs, [.025,.975]).tolist()
    out["local_ci95"] = np.quantile(ls, [.025,.975]).tolist()
    out["global_fraction_ci95"] = np.quantile(gs/(gs+ls), [.025,.975]).tolist()
    out["g_coordinate_ci95"] = np.stack([g-1.96*np.sqrt(V.sum(0).diagonal()/C**2),
                                            g+1.96*np.sqrt(V.sum(0).diagonal()/C**2)],axis=1).tolist()
    return out, delta, local, class_p


def balanced_sample(frame, n, rng):
    return np.concatenate([rng.choice(idx, n, replace=False) for idx in frame.groupby("label").indices.values()])


def mmd_unbiased(x, y, gamma, classes=102):
    """Stratified U-statistic for the equal-class mixture.

    balanced_sample concatenates equally sized class blocks. Exclude diagonals
    within each class block; do not apply pooled iid pair weights to fixed quotas.
    """
    xx, yy, xy = rbf_kernel(x, gamma=gamma), rbf_kernel(y, gamma=gamma), rbf_kernel(x,y,gamma=gamma)
    def within(kernel):
        n = len(kernel)
        assert n % classes == 0
        m = n // classes
        value = kernel.mean()
        for c in range(classes):
            block = kernel[c*m:(c+1)*m,c*m:(c+1)*m]
            value += ((block.sum()-np.trace(block))/(m*(m-1))-block.mean())/classes**2
        return value
    return within(xx) + within(yy) - 2*xy.mean()


def summary_and_distances(frames, zs, rng):
    summaries, distances, matrix = [], [], np.zeros((5,5))
    weights = {d: 1/frames[d].groupby("label")["label"].transform("size").to_numpy()/102 for d in DAYS}
    for day, frame in frames.items():
        w = weights[day]
        for name in NAMES:
            v = frame[name].to_numpy()
            mean = w@v
            summaries.append(dict(day=day, feature=name, mean=mean, variance=w@np.square(v-mean),
                                  std=np.sqrt(w@np.square(v-mean)), raw_median=np.median(v),
                                  mean_within_trace_variance=mean if name in ["iat_variance","direction_variance"] else np.nan))
        if day != 14:
            for j,name in enumerate(CORE):
                distances.append(dict(day=day, feature=name,
                    w1_standardized=wasserstein_distance(zs[14][:,j],zs[day][:,j],weights[14],w),
                    w1_raw=wasserstein_distance(frames[14][name],frame[name],weights[14],w)))
    for i,da in enumerate(DAYS):
        for j,db in enumerate(DAYS[:i]):
            matrix[i,j] = matrix[j,i] = np.mean([wasserstein_distance(zs[da][:,k], zs[db][:,k],weights[da],weights[db]) for k in range(len(CORE))])
    pd.DataFrame(summaries).to_csv(OUT/"feature_summary.csv", index=False)
    pd.DataFrame(distances).to_csv(OUT/"wasserstein.csv", index=False)
    pd.DataFrame(matrix,index=DAYS,columns=DAYS).to_csv(OUT/"pairwise_wasserstein.csv")
    ix = balanced_sample(frames[14],20,rng)
    gamma = 1/(2*np.median(pdist(zs[14][ix],metric="sqeuclidean")))
    mmd=[]
    for day in DAYS[1:]:
        estimates, null_estimates = [], []
        for repeat in range(10):
            ia=balanced_sample(frames[14],10,rng)
            ib=balanced_sample(frames[day],10,rng)
            estimates.append(mmd_unbiased(zs[14][ia],zs[day][ib],gamma))
            # Two disjoint equal-class source subsets: sampling variability reference.
            pairs=[rng.choice(idx,20,replace=False) for idx in frames[14].groupby("label").indices.values()]
            sa=np.concatenate([p[:10] for p in pairs]); sb=np.concatenate([p[10:] for p in pairs])
            null_estimates.append(mmd_unbiased(zs[14][sa],zs[14][sb],gamma))
        mmd.append(dict(day=day,mmd2_mean=np.mean(estimates),mmd2_sampling_sd=np.std(estimates,ddof=1),
                        source_split_mmd2_mean=np.mean(null_estimates),source_split_mmd2_sd=np.std(null_estimates,ddof=1),gamma=gamma,n_per_class=10,repeats=10))
    pd.DataFrame(mmd).to_csv(OUT/"mmd.csv",index=False)
    return pd.DataFrame(summaries), pd.DataFrame(distances), matrix


def geometry(frames,zs):
    data=[]
    for day in DAYS:
        a=class_arrays(zs[day],frames[day].label.to_numpy())
        means=np.array([x.mean(0) for x in a])
        within=np.mean([np.square(x-x.mean(0)).sum(1).mean() for x in a])
        between=np.square(means-means.mean(0)).sum(1).mean()
        data.append(dict(day=day,within=within,between=between,between_within=between/within))
    pd.DataFrame(data).to_csv(OUT/"geometry.csv",index=False)


def affine_cv(a,b):
    ma=np.array([x.mean(0) for x in a]); mb=np.array([x.mean(0) for x in b])
    errors={k:[] for k in ["none","translation","diagonal_affine","ridge_affine"]}
    for tr,te in KFold(n_splits=5,shuffle=True,random_state=SEED).split(ma):
        g=(mb[tr]-ma[tr]).mean(0)
        diag=np.empty_like(mb[te])
        for j in range(ma.shape[1]):
            model=Ridge(alpha=10).fit(ma[tr,j,None],mb[tr,j]-ma[tr,j])
            diag[:,j]=ma[te,j]+model.predict(ma[te,j,None])
        model=Ridge(alpha=10).fit(ma[tr],mb[tr]-ma[tr])
        preds={"none":ma[te],"translation":ma[te]+g,"diagonal_affine":diag,
               "ridge_affine":ma[te]+model.predict(ma[te])}
        for name,pred in preds.items(): errors[name].extend(np.square(pred-mb[te]).sum(1))
    return {k:float(np.mean(v)) for k,v in errors.items()}


def make_plots(frames,zs,summary,distances,matrix,decomp,classdf,scaler):
    plt.rcParams.update({"font.size":10,"figure.dpi":140,"savefig.bbox":"tight"})
    fig,axs=plt.subplots(2,3,figsize=(13,7))
    for ax,name,label,factor in zip(axs.flat,["n_packets","duration","positive_fraction","switch_rate","iat_mean","burst_p90"],
                     ["Records per trace","Observed duration (s)","Positive fraction","Direction switch rate","Mean IAT (ms)","Within-trace burst p90"],[1,1,1,1,1000,1]):
        sel=summary[summary.feature==name]
        ax.plot(sel.day,sel["mean"]*factor,"o-"); ax.set(xlabel="Day",ylabel=label,xticks=DAYS);ax.grid(alpha=.25)
    fig.suptitle("Equal website weighting; all stored traces");fig.tight_layout();fig.savefig(FIG/"01_trends.png");plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(13,6))
    arr=distances.pivot(index="feature",columns="day",values="w1_standardized").reindex(CORE)
    im=axs[0].imshow(arr,aspect="auto",cmap="viridis");axs[0].set(xticks=range(4),xticklabels=DAYS[1:],yticks=range(len(CORE)),yticklabels=CORE,title="W1 vs Day 14 (source SD units)");fig.colorbar(im,ax=axs[0])
    im=axs[1].imshow(matrix,cmap="viridis");axs[1].set(xticks=range(5),xticklabels=DAYS,yticks=range(5),yticklabels=DAYS,title="Mean marginal W1, 16 features");fig.colorbar(im,ax=axs[1])
    for i in range(5):
        for j in range(5): axs[1].text(j,i,f"{matrix[i,j]:.3f}",ha="center",color="white" if matrix[i,j]<matrix.max()*.6 else "black")
    fig.tight_layout();fig.savefig(FIG/"02_distances.png");plt.close(fig)
    pca=PCA(n_components=2).fit(zs[14]); rng=np.random.default_rng(SEED)
    fig,axs=plt.subplots(1,2,figsize=(12,5))
    for day,col in [(14,"#2477b4"),(270,"#d95b31")]:
        idx=balanced_sample(frames[day],15,rng); p=pca.transform(zs[day][idx])
        axs[0].scatter(p[:,0],p[:,1],s=5,alpha=.22,color=col,label=f"Day {day}")
    axs[0].legend();axs[0].set(title="Source-fit PCA: 15 traces / website / day",xlabel="PC1",ylabel="PC2")
    ma=np.array([x.mean(0) for x in class_arrays(zs[14],frames[14].label.to_numpy())]);mb=np.array([x.mean(0) for x in class_arrays(zs[270],frames[270].label.to_numpy())])
    pa,pb=pca.transform(ma),pca.transform(mb)
    axs[1].quiver(pa[:,0],pa[:,1],pb[:,0]-pa[:,0],pb[:,1]-pa[:,1],angles="xy",scale_units="xy",scale=1,alpha=.5,width=.003)
    axs[1].scatter(pa[:,0],pa[:,1],s=12,label="Day 14 centroids");axs[1].scatter(pb[:,0],pb[:,1],s=12,label="Day 270 centroids");axs[1].legend();axs[1].set(title="Website centroid movement",xlabel="PC1",ylabel="PC2")
    fig.suptitle(f"Explained source variance: {pca.explained_variance_ratio_.sum():.1%}");fig.tight_layout();fig.savefig(FIG/"03_pca.png");plt.close(fig)
    pd.DataFrame(pca.components_,columns=CORE).to_csv(OUT/"pca_loadings.csv",index=False)
    (OUT/"pca.json").write_text(json.dumps({"explained_variance_ratio":pca.explained_variance_ratio_.tolist(),"fit_day":14},indent=2))
    fig,axs=plt.subplots(1,2,figsize=(12,5))
    gs=np.array([x["global_corrected"] for x in decomp]);ls=np.array([x["local_corrected"] for x in decomp])
    axs[0].bar(range(4),gs,label="Common translation");axs[0].bar(range(4),ls,bottom=gs,label="Website heterogeneity");axs[0].set(xticks=range(4),xticklabels=DAYS[1:],ylabel="Noise-corrected squared shift",title="Equal website weights; 16 source-scaled features");axs[0].legend()
    for day in DAYS[1:]:
        vals=np.sort(classdf[classdf.day==day].local_norm)
        axs[1].plot(vals,np.arange(1,len(vals)+1)/len(vals),label=f"Day {day}")
    axs[1].set(xlabel="Website residual shift norm",ylabel="Website empirical CDF",title="Local shift is heterogeneous");axs[1].legend();fig.tight_layout();fig.savefig(FIG/"04_decomposition.png");plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    rng=np.random.default_rng(SEED)
    frames={d:pd.read_csv(OUT/f"features_day{d}.csv") for d in DAYS}
    assert all(len(frames[d]) == json.loads((OUT/f"audit_day{d}.json").read_text())["shape"][0] for d in DAYS)
    assert not pd.concat(list(frames.values())).trace_hash.duplicated().any(), "Deduplicate before inferential analysis"
    scaler=StandardScaler().fit(transform(frames[14]))
    zs={d:scaler.transform(transform(f)) for d,f in frames.items()}
    pd.DataFrame({"feature":CORE,"source_mean":scaler.mean_,"source_scale":scaler.scale_}).to_csv(OUT/"analysis_scaler.csv",index=False)
    summary,distances,matrix=summary_and_distances(frames,zs,rng)
    geometry(frames,zs)
    decs,classes,affine=[],[],[]
    a=class_arrays(zs[14],frames[14].label.to_numpy())
    for day in DAYS[1:]:
        b=class_arrays(zs[day],frames[day].label.to_numpy())
        dec,delta,local,p=decompose(a,b,rng)
        dec["day"]=day;decs.append(dec)
        for c in range(len(a)):
            classes.append(dict(day=day,label=c,total_norm=np.linalg.norm(delta[c]),local_norm=np.linalg.norm(local[c]),
                                projection_global=float(delta[c]@delta.mean(0)),p_local=p[c]))
        pd.DataFrame(delta,columns=CORE).assign(label=np.arange(102)).to_csv(OUT/f"class_shifts_day{day}.csv",index=False)
        affine.append(dict(day=day,**affine_cv(a,b)))
        print("decomposition",day,{k:v for k,v in dec.items() if "coordinate" not in k},flush=True)
    ph=holm([r[k] for r in decs for k in ["p_global","p_local"]])
    for i,r in enumerate(decs):r.update(p_global_holm=ph[2*i],p_local_holm=ph[2*i+1])
    classdf=pd.DataFrame(classes);classdf["q_local_all408"]=bh(classdf.p_local.to_numpy())
    classdf.to_csv(OUT/"class_drift.csv",index=False)
    (OUT/"decomposition.json").write_text(json.dumps(decs,indent=2))
    pd.DataFrame(affine).to_csv(OUT/"affine_sensitivity.csv",index=False)
    # Confirm decomposition across alternate measurements and censoring choices.
    sensitivity=[]
    specs=[("direction_burst",frames,[n for n in CORE if n not in TIME and n not in ["iat_variance","iat_zero_fraction"]]),
           ("time_only",frames,[n for n in CORE if n in TIME or n in ["iat_variance","iat_zero_fraction"]]),
           ("uncapped",{d:f[f.at_cap==0] for d,f in frames.items()},CORE),
           ("prefix1000",{d:pd.read_csv(OUT/f"prefix1000_day{d}.csv") for d in DAYS},CORE)]
    for name,fs,cols in specs:
        ss=StandardScaler().fit(transform(fs[14],cols))
        zz={d:ss.transform(transform(f,cols)) for d,f in fs.items()}
        aa=class_arrays(zz[14],fs[14].label.to_numpy())
        for day in DAYS[1:]:
            bb=class_arrays(zz[day],fs[day].label.to_numpy())
            de,_,_,_=decompose(aa,bb,rng,False);sensitivity.append(dict(setting=name,day=day,features=len(cols),**de))
    pd.DataFrame(sensitivity).to_csv(OUT/"sensitivity.csv",index=False)
    # Domain discrimination is diagnostic, not a website-label adaptation model.
    ia=balanced_sample(frames[14],50,rng);ib=balanced_sample(frames[270],50,rng)
    x=np.r_[zs[14][ia],zs[270][ib]];y=np.r_[np.zeros(len(ia)),np.ones(len(ib))]
    tr,te=train_test_split(np.arange(len(y)),test_size=.3,stratify=y,random_state=SEED)
    model=ExtraTreesClassifier(n_estimators=200,min_samples_leaf=5,n_jobs=4,random_state=SEED).fit(x[tr],y[tr])
    (OUT/"domain_diagnostic.json").write_text(json.dumps({"day14_vs_day270_auc":roc_auc_score(y[te],model.predict_proba(x[te])[:,1]),"n_train":len(tr),"n_test":len(te),"caveat":"exploratory domain discrimination, all-source-fitted feature scale"},indent=2))
    make_plots(frames,zs,summary,distances,matrix,decs,classdf,scaler)
    print("Analysis complete",flush=True)


if __name__ == "__main__":
    main()
