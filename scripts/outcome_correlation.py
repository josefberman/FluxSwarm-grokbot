#!/usr/bin/env python3
"""Cross-env outcome correlation within an update (same policy), near vs far in time."""
from pathlib import Path
import json,sys,numpy as np,pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent)); from analyze_stability import load

for r in sys.argv[1:]:
    m,e=load(Path(r)); e=e[e.source=='train']; e=e[e['update']>=len(m)//2]
    s=e.success.astype(float).to_numpy(); u=e['update'].to_numpy(); t=e.t.to_numpy(); env=e.env.to_numpy(); p=s.mean()
    near=[];far=[]
    for uu in np.unique(u):
        idx=np.where(u==uu)[0]
        for i in range(len(idx)):
            for j in range(i+1,len(idx)):
                a,b=idx[i],idx[j]
                if env[a]==env[b]: continue
                prod=(s[a]-p)*(s[b]-p)
                (near if abs(t[a]-t[b])<=20 else far if abs(t[a]-t[b])>=100 else []).append(prod)
    g=e.groupby('update').success.agg(['mean','count']); g=g[g['count']>=4]
    disp=g['mean'].var()/np.mean(p*(1-p)/g['count'])
    print(Path(r).name[20:], 'p=%.2f'%p,'dispersion(var/binom)=%.1f'%disp,'corr_near=%.2f corr_far=%.2f'%(np.mean(near)/(p*(1-p)),np.mean(far)/(p*(1-p))), 'n_pairs',len(near),len(far))
