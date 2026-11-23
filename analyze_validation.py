"""Paired, cluster-aware exploratory summaries for WITNESS validation."""
import json,collections
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;R=HERE/'results'

def boot_two_way(mat,rep=4000):
    rng=np.random.default_rng(20261121)
    a=np.asarray(mat,float);s,d=a.shape
    means=[]
    for _ in range(rep):
        rows=rng.integers(s,size=s);cols=rng.integers(d,size=d)
        means.append(float(a[np.ix_(rows,cols)].mean()))
    return [round(float(x),5) for x in np.percentile(means,[2.5,97.5])]

label=json.loads((R/'label_validation.json').read_text())
summary={'label':{},'neural':{}}
for regime,t in label.items():
    methods={v['model']:v for v in t['summary']}
    pairrows=collections.defaultdict(lambda:collections.defaultdict(list))
    for r in t['rows']:pairrows[r['model']][r['pair_index']].append(r)
    contrasts={}
    for a,b in [('spectral_weighted','weighted_map'),('dawid_skene','weighted_soft'),('mixture_soft','weighted_soft')]:
        diff=np.array([np.mean([r['all_correct'] for r in pairrows[a][idx]])-
                       np.mean([r['all_correct'] for r in pairrows[b][idx]])
                       for idx in range(len(pairrows[a]))])
        rng=np.random.default_rng(4242)
        boot=np.array([diff[rng.integers(len(diff),size=len(diff))].mean() for _ in range(4000)])
        contrasts[f'{a}_minus_{b}']={'mean_pp':round(100*float(diff.mean()),3),
                                       'pair_cluster_boot_pp':[round(100*float(x),3) for x in np.percentile(boot,[2.5,97.5])]}
    summary['label'][regime]={'summary':{k:{f:round(float(v),4) if isinstance(v,(int,float,np.number)) else v for f,v in item.items()} for k,item in methods.items()},'contrasts':contrasts}

if (R/'neural_validation.json').exists():
    ne=json.loads((R/'neural_validation.json').read_text())
    rows=ne['rows'];sumrows=[]
    for regime in ('heavy','correlated'):
        sumrows.append({})
        for kind in ('old_generator','arborescence'):
            selected=[r for r in rows if r['regime']==regime and r['draw_kind']==kind]
            if not selected:continue
            arms=sorted(set(r['arm'] for r in selected));byarm={}
            for arm in arms:
                sub=[r for r in selected if r['arm']==arm]
                byarm[arm]={'mean_cqa':round(float(np.mean([r['cqa'] for r in sub])),5),
                            'n_rows':len(sub),'n_seeds':len(set(r['seed'] for r in sub)),
                            'n_draws':len(set(r['draw_seed'] for r in sub))}
            pairs={}
            controls=[('joint_independent','weighted_soft'),('joint_robust','robust_soft'),('spectral_weighted','weighted_soft' if regime=='heavy' else 'robust_soft'),('dawid_skene_soft','weighted_soft' if regime=='heavy' else 'robust_soft')]
            for a,b in controls:
                if a not in byarm or b not in byarm:continue
                keyed={k:{(v['seed'],v['draw_seed']):v['cqa'] for v in selected if v['arm']==k} for k in (a,b)}
                common=sorted(set(keyed[a])&set(keyed[b]))
                ss=sorted(set(x[0] for x in common));ds=sorted(set(x[1] for x in common))
                if len(common)!=len(ss)*len(ds):continue
                mat=np.array([[keyed[a][s,d]-keyed[b][s,d] for d in ds] for s in ss])
                pairs[f'{a}_minus_{b}']={'mean_pp':round(100*float(mat.mean()),3),
                    'interval_pp':[round(100*x,3) for x in boot_two_way(mat)],
                    'per_seed_mean_pp':{str(s):round(100*float(row.mean()),3) for s,row in zip(ss,mat)}}
            summary['neural'][f'{regime}/{kind}']={'by_arm':byarm,'paired':pairs}
    summary['crosscheck_max_diff']=ne.get('crosscheck_max_diff')
(R/'summary.json').write_text(json.dumps(summary,indent=2))
for regime,v in summary['label'].items():print('LABEL',regime,v['contrasts'])
for regime,v in summary['neural'].items():print('NEURAL',regime,{k:val['mean_cqa'] for k,val in v['by_arm'].items()},v['paired'])
