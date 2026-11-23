"""Independent audit of WITNESS multi-seed/multi-graph evaluation.
Paired statistics preserve matching by model initialization and graph draw.
Two-way cluster resampling is exploratory, not a license to claim IID samples.
"""
from __future__ import annotations
import json,itertools
from pathlib import Path
from collections import defaultdict
import numpy as np
DATA=Path(__file__).resolve().parent/'results'


def paired_bootstrap(mat,seed=92038,reps=16000):
    rng=np.random.default_rng(seed);s,d=mat.shape
    sid=rng.integers(s,size=(reps,s));did=rng.integers(d,size=(reps,d))
    boot=mat[sid[:,:,None],did[:,None,:]].mean((1,2))
    return [float(v) for v in np.quantile(boot,[.025,.975])]


def sign_paired_pvalue(seed_diffs):
    vals=np.asarray(seed_diffs,dtype=float);obs=abs(vals.mean());s=len(vals)
    if s>20:return None
    totals=[]
    for bits in itertools.product([-1.,1.],repeat=s):totals.append(abs(np.mean(vals*np.array(bits))))
    return float(np.mean(np.array(totals)>=obs-1e-12))


def main():
    data=json.loads((DATA/'extension.json').read_text());rows=data['rows'];seeds=data['protocol']['seeds'];draws=data['protocol']['graph_seeds']
    assert len(rows)==len(seeds)*len(draws)*12,(len(rows),seeds,draws)
    lookup={(r['regime'],r['seed'],r['arm'],r['graph_draw_seed']):r for r in rows}
    groups={};paired={};runtimes={}
    for reg,arms in data['protocol']['arms'].items():
        groups[reg]={};runtimes[reg]={}
        for arm in arms:
            mat=np.array([[lookup[(reg,s,arm,d)]['cqa'] for d in draws] for s in seeds])
            groups[reg][arm]={'mean':float(mat.mean()),'median_over_seeds':float(np.median(mat.mean(1))),
                               'std_over_seed_means':float(mat.mean(1).std(ddof=1)),
                               'seed_means':{str(s):float(mat[i].mean()) for i,s in enumerate(seeds)},
                               'graph_draw_means':{str(d):float(mat[:,j].mean()) for j,d in enumerate(draws)},
                               'mean_with_two_way_bootstrap_ci':paired_bootstrap(mat,seed=3471)}
            runtimes[reg][arm]=float(np.mean([lookup[(reg,s,arm,draws[0])]['metadata']['training_seconds'] for s in seeds]))
    for reg,left,right in [('heavy','weighted_soft','joint_independent'),('correlated','robust_soft','joint_robust')]:
        delta=np.array([[lookup[(reg,s,right,d)]['cqa']-lookup[(reg,s,left,d)]['cqa'] for d in draws] for s in seeds])
        dmeans=delta.mean(1)
        paired[reg]={'left':left,'right':right,'mean_diff':float(delta.mean()),'two_way_bootstrap_ci':paired_bootstrap(delta),
                     'exact_two_sided_sign_randomization_p':sign_paired_pvalue(dmeans),
                     'seed_difference_means':{str(s):float(dmeans[i]) for i,s in enumerate(seeds)},
                     'graph_draw_diffs':{str(d):float(delta[:,j].mean()) for j,d in enumerate(draws)},
                     'wins_ties_losses':{'wins':int((dmeans>1e-10).sum()),'ties':int((np.abs(dmeans)<=1e-10).sum()),'losses':int((dmeans< -1e-10).sum())},
                     'leave_one_seed_out_mean_range':[float(min(np.delete(dmeans,i).mean() for i in range(len(seeds)))),float(max(np.delete(dmeans,i).mean() for i in range(len(seeds))))]}
    out={'protocol':data['protocol'],'gauge':data['gauge'],'groups':groups,'paired':paired,'runtime_mean_training_seconds':runtimes,
         'limitations':['Only synthetic related graph generators; test draws are shared across model seeds',
                        'Model seed also indexes weak-report observations, so seed variance combines neural initialization and training-report changes',
                        'Two-way bootstrap treats graph-draw clusters as exchangeable; it does not estimate generalization to natural networks',
                        'No external published model implementation is reproduced; Dawid-Skene-style EM is an adaptation',
                        'Model comparison is exploratory after prior development and paper should not claim preregistration']}
    (DATA/'extension_summary.json').write_text(json.dumps(out,indent=2))
    print(json.dumps({'groups':{a:{k:round(v['mean'],3) for k,v in b.items()} for a,b in groups.items()},'paired':paired,'gauge':data['gauge']},indent=2))
if __name__=='__main__':main()
