"""Out-of-development replication for WITNESS.

Three separate uncertainties are tracked: training initialization/weak-report draws,
independent graph draws, and inference noise-model specification.

The EM parity label model is an explicitly labeled adaptation of classic
latent-label EM; it is not a reproduction of an external codebase.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, random, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from witness_core.robust_study import (RELATIONS, ASSIGNMENTS, XOR_TABLE,
                                       split, observe, check_pair, infer, posterior)
from witness_core.gated_baseline import GatedNet
from witness_core.legacy_directed import pack
from method import constraint_nll,robust_soft_targets,syndrome,starflip

SEEDS=[71,73,79,83,89,97,101,103,107,109,113,127]
GRAPH_SEEDS=[1618033,1414213,1732051,2236067,2449489]
N_VALUES=[22,34,48]
FAMILIES=['branch','dense_scc']
ARMS={'heavy':['anchor','unweighted_map','em_soft','weighted_soft','joint_independent','oracle'],
      'correlated':['unweighted_map','em_soft','weighted_soft','joint_independent','robust_soft','joint_robust']}


def em_parity(units,iterations=40):
    """Adapted Dawid--Skene-style EM for binary pairwise relation reports.
    Each of the six relation channels shares a learned confusion probability
    across units; soft truth assignments are latent with observed anchors.
    Reliability metadata are *not* used. Beta(2,8) smoothing is declared.
    """
    reports=np.stack([u.observed_bits for u in units]);n=len(units)
    eps=np.full(6,.20,dtype=float); beta_a,beta_b=2.,8.
    for _ in range(iterations):
        match=(XOR_TABLE[None,:,:]==reports[:,None,:])
        ll=np.where(match,np.log1p(-eps)[None,None,:],np.log(eps)[None,None,:]).sum(2)
        ll-=ll.max(1,keepdims=True)
        probs=np.exp(ll);probs/=probs.sum(1,keepdims=True)
        mistakes=(probs[:,:,None]*(~match)).sum((0,1))
        next_eps=np.clip((mistakes+beta_a-1)/(n+beta_a+beta_b-2),.015,.48)
        if np.max(np.abs(next_eps-eps))<1e-8:
            eps=next_eps;break
        eps=next_eps
    vals=[]
    for i,u in enumerate(units):
        match=(XOR_TABLE==reports[i][None,:]);ll=np.where(match,np.log1p(-eps)[None,:],np.log(eps)[None,:]).sum(1)
        probs=np.exp(ll-ll.max());probs/=probs.sum()
        vals.append(probs @ (ASSIGNMENTS ^ u.anchor_label).astype(float))
    return np.clip(np.stack(vals),0.,1.),eps


def score_prepacked(net,items):
    """No test labels touch training. Evaluator is exact oracle from generator."""
    net.eval();out=[]
    with torch.no_grad():
        for packs,ys in items:
            pred=[]
            for off in (0,2):
                adj,source,target,steps=packs[off]
                h=torch.tanh(net.initial(source[:,None]))
                for _ in range(steps):
                    nbr=torch.where(adj[:,:,None],h[None,:,:],torch.full_like(h[None,:,:],-1e5)).amax(1)
                    nbr=torch.where(adj.any(1,keepdim=True),nbr,0.)
                    h=net.gate(torch.cat((nbr,source[:,None]),-1),h)
                probs=torch.sigmoid(net.readout(h).squeeze(-1))
                pred.extend([int(float(probs[packs[off+j][2]])>=.5) for j in (0,1)])
            out.append(int(np.array_equal(np.array(pred,dtype=np.int8),ys)))
    return float(np.mean(out)),out


def run_audit(path:Path):
    """Gauge invariance tested on four independent structural generator settings.
    Counterexample also changes the inferred nonanchor labels while keeping anchor fixed.
    """
    rows=[]
    for fam in ['chain_scc','branch','layered','dense_scc']:
        for n in [10,14,22,34]:
            for effect in [False,True]:
                for trial in range(8):
                    seed=987341+733*n+5901*trial+13*effect+sum(map(ord,fam))*31
                    pairs=split(seed,[n],[fam],1)
                    p=next(x for x in pairs if x.effect==int(effect))
                    check_pair(p,deep=(trial==0))
                    u=observe(p,seed*13+7,'correlated')
                    bits=tuple(int(b) for b in u.observed_bits)
                    altered=starflip(bits)
                    assert syndrome(bits)==syndrome(altered)
                    true=np.array([y for _,_,y in u.examples()],dtype=int)
                    fake=true.copy();fake[1:]^=1
                    assert fake[0]==true[0] and np.any(fake!=true)
                    # Under a simultaneous fan transformation, *every observed relation*
                    # can be reinterpreted relative to fake truths, independently of graph shape.
                    original_relation=np.array([true[a]^true[b] for a,b in RELATIONS])
                    alternate_relation=np.array([fake[a]^fake[b] for a,b in RELATIONS])
                    assert np.array_equal(original_relation ^ np.array([int(a==0 or b==0) for a,b in RELATIONS]),alternate_relation)
                    rows.append({'family':fam,'n':n,'effect':int(effect),'draw':trial,'graph_seed':seed,
                                 'syndrome_equal':True,'anchor_fixed':True,'different_nonanchor_assignment':True,
                                 'graph_nodes':p.g0.number_of_nodes(), 'graph_edges':p.g0.number_of_edges()})
    path.write_text(json.dumps({'n':len(rows),'rows':rows,'claim':'algebraic invariance of relation-only cycle syndrome; graph itself may disambiguate'},indent=2))
    return {'cases':len(rows),'families':sorted(set(r['family'] for r in rows)),'sizes':sorted(set(r['n'] for r in rows))}


def make_test_draws():
    draws=[]
    for dseed in GRAPH_SEEDS:
        pairs=split(dseed,N_VALUES,FAMILIES,2)
        assert len(pairs)==24
        for p in pairs:check_pair(p,deep=False)
        draw=[]
        for pair in pairs:
            xs=[pack(g,q,p.n+2) for g,q,_ in pair.examples()]
            y=np.array([y for g,q,y in pair.examples()],dtype=np.int8)
            draw.append((xs,y))
        draws.append(draw)
    return draws


def train(seed,arm,regime,units,prepared,epochs=6):
    torch.manual_seed(seed)
    net=GatedNet()
    opt=torch.optim.AdamW(net.parameters(),lr=.003)
    targets=[]; masks=[]; details={}
    if arm=='em_soft':
        em_targets,eps=em_parity(units)
        details['learned_eps']=[round(float(a),6) for a in eps]
    for i,u in enumerate(units):
        if arm=='em_soft':y=em_targets[i];m=np.ones(4)
        elif arm=='robust_soft':y=robust_soft_targets(u);m=np.ones(4)
        elif arm in ['joint_independent','joint_robust']:
            y,m=infer(u,'anchor')
        else:y,m=infer(u,arm)
        targets.append(torch.tensor(y,dtype=torch.float32))
        masks.append(torch.tensor(m,dtype=torch.float32))
    t0=time.perf_counter()
    for ep in range(epochs):
        order=list(range(len(units)));random.Random(seed*61+ep).shuffle(order)
        for i in order:
            pred=torch.stack([net(x) for x in prepared[i]]).clamp(1e-5,1-1e-5)
            if arm.startswith('joint_'):
                loss=constraint_nll(pred,units[i],'correlation_mixture' if arm=='joint_robust' else 'independent')
            else:
                loss=(F.binary_cross_entropy(pred,targets[i],reduction='none')*masks[i]).sum()/masks[i].sum().clamp_min(1)
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
    details.update({'training_seconds':round(time.perf_counter()-t0,3),'updates':len(units)*epochs,'graph_forwards':4*len(units)*epochs})
    return net,details


def main(args):
    torch.set_num_threads(1)
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    train_graphs=split(314159,[10,12],['chain_scc','layered'],3)
    test_draws=make_test_draws()
    audit=run_audit(out.parent/'gauge_audit.json')
    protocol={'seeds':SEEDS,'graph_seeds':GRAPH_SEEDS,'train_graph_seed':314159,
              'train_quadruples':len(train_graphs),'test_quadruples_per_draw':len(test_draws[0]),
              'train_component_sizes':[10,12],'test_component_sizes':N_VALUES,
              'train_families':['chain_scc','layered'],'test_families':FAMILIES,
              'oracle_on_training':False,'epochs':6,'updates':144,'graph_forwards':576,
              'arms':ARMS,'correlated_event_actual':.28,'correlated_event_model':.20,
              'fixed_noise_probs':{'heavy':[.16,.46],'correlated':[.06,.30]},
              'validation':'exploratory extension after earlier pilot/replication; no claim of external preregistration',
              'published_method_status':'EM adapted from latent-label method; Bayesian posterior and BFS are analytic controls; no external software method faithfully reproduced'}
    (out.parent/'protocol.json').write_text(json.dumps(protocol,indent=2))
    rows=[];t0=time.perf_counter()
    seeds=SEEDS[:args.seeds]
    regimes=['heavy','correlated'] if args.regime=='all' else [args.regime]
    for regime in regimes:
        for seed in seeds:
            units=[observe(p,seed*100003+i*31,regime) for i,p in enumerate(train_graphs)]
            prepared=[[pack(g,q,13) for g,q,_ in u.examples()] for u in units]
            for arm in ARMS[regime]:
                if args.arm and arm!=args.arm:continue
                net,meta=train(seed,arm,regime,units,prepared)
                for test_i,test_draw in enumerate(test_draws):
                    cqa,correct=score_prepacked(net,test_draw)
                    rows.append({'regime':regime,'seed':seed,'arm':arm,'graph_draw_seed':GRAPH_SEEDS[test_i],
                        'graph_draw_index':test_i,'cqa':cqa,'n_test_quadruples':len(test_draw),'metadata':meta,
                        'per_quadruple_correct':correct})
                out.write_text(json.dumps({'protocol':protocol,'gauge':audit,'rows':rows},indent=2))
                print('FINISHED',regime,seed,arm,'meanCQA',round(float(np.mean([r['cqa'] for r in rows if r['seed']==seed and r['regime']==regime and r['arm']==arm])),4),'train_seconds',meta['training_seconds'],'elapsed',round(time.perf_counter()-t0,1),flush=True)
    print('COMPLETE',len(rows),'elapsed',round(time.perf_counter()-t0,2),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='results/extension.json');ap.add_argument('--seeds',type=int,default=len(SEEDS));ap.add_argument('--regime',choices=['all','heavy','correlated'],default='all');ap.add_argument('--arm',default=None)
    main(ap.parse_args())
