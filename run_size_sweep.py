"""Exploratory WITNESS training-data scaling at fixed optimizer/graph-forward budget.

This is additional post-hoc validation, not a preregistered confirmatory experiment.
The 24/48/96/192 quadruple training collections are nested and stratum balanced.
Only the weak anchor plus noisy parity reports are used for either training arm.
"""
from __future__ import annotations
import argparse, json, random, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from witness_core.robust_study import split,observe,infer
from witness_core.gated_baseline import GatedNet
from witness_core.legacy_directed import pack
from WITNESS_TMLR_Anonymized_Supplement.experiments.method import constraint_nll
from WITNESS_TMLR_Anonymized_Supplement.experiments.extended_study import GRAPH_SEEDS,make_test_draws,score_prepacked
from independent_graphs import test_draw

SIZES=(24,48,96,192)
STRATA=8
ARMS=('weighted_soft','joint_independent')
EVAL_SEEDS=(20261,20262,20263)

def nested_training_sets(data_seed:int):
    """Eight balanced strata: 2 sizes x 2 families x 2 change conditions."""
    pool=split(data_seed,[10,12],['chain_scc','layered'],24)
    bins=defaultdict(list)
    for p in pool:
        bins[(p.n,p.family,p.effect)].append(p)
    assert len(bins)==STRATA and all(len(v)==24 for v in bins.values())
    for v in bins.values(): v.sort(key=lambda p:p.seed)
    out={}
    for n in SIZES:
        subset=[p for key in sorted(bins) for p in bins[key][:n//STRATA]]
        assert len(subset)==n
        assert len(set(p.seed for p in subset))==n
        out[n]=subset
    for a,b in zip(SIZES,SIZES[1:]):
        assert set(p.seed for p in out[a]).issubset(set(p.seed for p in out[b]))
    return out

def build_units(pairs,report_seed,regime):
    """Stable per-graph noise seeds prevent size from changing overlapping examples."""
    return [observe(p,report_seed*100003+p.seed*31,regime) for p in pairs]

def train_fixed_budget(units,seed,arm,updates):
    assert arm in ARMS
    torch.manual_seed(seed)
    net=GatedNet()
    opt=torch.optim.AdamW(net.parameters(),lr=.003)
    inputs=[[pack(g,q,13) for g,q,_ in u.examples()] for u in units]
    targets=[]
    if arm=='weighted_soft':
        for u in units:
            y,m=infer(u,arm)
            targets.append((torch.as_tensor(y,dtype=torch.float32),torch.as_tensor(m,dtype=torch.float32)))
    rng=random.Random(seed*61+1337)
    indices=[]
    while len(indices)<updates:
        order=list(range(len(units)));rng.shuffle(order)
        indices.extend(order)
    t0=time.perf_counter()
    for i in indices[:updates]:
        out=torch.stack([net(inp) for inp in inputs[i]]).clamp(1e-5,1-1e-5)
        if arm=='joint_independent':loss=constraint_nll(out,units[i],'independent')
        else:
            y,m=targets[i]
            loss=(F.binary_cross_entropy(out,y,reduction='none')*m).sum()/m.sum().clamp_min(1)
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
    return net,{'seconds':round(time.perf_counter()-t0,3),'updates':updates,'graph_forwards':4*updates,
                'examples':len(units),'total_examples_presented':updates}

def prepare_draws():
    original=make_test_draws()
    external=[]
    for s in EVAL_SEEDS:
        pairs=test_draw(s)
        external.append([([pack(g,q,p.n+2) for g,q,_ in p.examples()],
                          np.asarray([y for _,_,y in p.examples()],dtype=np.int8)) for p in pairs])
    return [('original',seed,draw) for seed,draw in zip(GRAPH_SEEDS,original)]+[
       ('arborescence',seed,draw) for seed,draw in zip(EVAL_SEEDS,external)]

def main(args):
    torch.set_num_threads(1)
    data_seeds=tuple(map(int,args.data_seeds.split(',')))
    model_seeds=tuple(map(int,args.model_seeds.split(',')))
    sizes=tuple(map(int,args.sizes.split(',')))
    assert set(sizes).issubset(SIZES)
    draws=prepare_draws()
    proto={'status':'exploratory post-hoc training-size study','regime':args.regime,
      'size_strata':{'sizes':[10,12],'families':['chain_scc','layered'],'effects':[0,1]},
      'sizes':sizes,'model_seeds':model_seeds,'training_data_seeds':data_seeds,
      'report_seed_note':'deterministically derived from model seed; not an independent report-noise replication',
      'fixed_updates':args.updates,'fixed_graph_forwards':args.updates*4,
      'arms':ARMS,'test_draw_seeds':{'original':GRAPH_SEEDS,'arborescence':EVAL_SEEDS},
      'no_test_labels_in_training':True,'oracle_is_only_evaluation':True,
      'compute_warning':'additional marginalization arithmetic not matched by graph forwards'}
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    rows=[];started=time.perf_counter()
    for data_seed in data_seeds:
        groups=nested_training_sets(data_seed)
        for size in sizes:
            for seed in model_seeds:
                units=build_units(groups[size],seed,args.regime)
                for arm in ARMS:
                    net,meta=train_fixed_budget(units,seed,arm,args.updates)
                    for fam,draw_seed,draw in draws:
                        acc,correct=score_prepacked(net,draw)
                        rows.append({'data_seed':data_seed,'size':size,'model_report_seed':seed,
                           'arm':arm,'draw_family':fam,'draw_seed':draw_seed,'cqa':acc,
                           'n_quadruples':len(correct),'correct':correct,'training':meta})
                    out.write_text(json.dumps({'protocol':proto,'rows':rows},indent=2))
                    old=[r['cqa'] for r in rows if r['data_seed']==data_seed and r['size']==size and r['model_report_seed']==seed and r['arm']==arm and r['draw_family']=='original']
                    new=[r['cqa'] for r in rows if r['data_seed']==data_seed and r['size']==size and r['model_report_seed']==seed and r['arm']==arm and r['draw_family']=='arborescence']
                    print(f'DONE data={data_seed} size={size} model_seed={seed} arm={arm} old={np.mean(old):.3f} alternate={np.mean(new):.3f} train_seconds={meta["seconds"]:.1f} elapsed={time.perf_counter()-started:.1f}',flush=True)
    print('FINISHED',len(rows),'evaluation draw rows',flush=True)

if __name__=='__main__':
    pa=argparse.ArgumentParser()
    pa.add_argument('--model-seeds',default='71,73,79')
    pa.add_argument('--data-seeds',default='314159')
    pa.add_argument('--sizes',default='24,48,96,192')
    pa.add_argument('--updates',type=int,default=576)
    pa.add_argument('--regime',choices=['heavy','correlated'],default='heavy')
    pa.add_argument('--output',default='results/size_sweep_exploratory.json')
    main(pa.parse_args())
