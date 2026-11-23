"""Noisy reported intervention relations: robust inference vs learned directed reachability.
All graph truths verified via exact oracle. Noisy reports are observations, NOT certified facts.
Training uses one true anchor and noisy parity reports; no intermediate process traces.
"""
from __future__ import annotations
import dataclasses, random, math, json, time, itertools, pathlib, argparse
import numpy as np
import networkx as nx
import torch
from torch import nn
from torch.nn import functional as F
from .legacy_directed import Pair, Net, pack, evaluate

RELATIONS=((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))
ASSIGNMENTS=np.asarray([[0,*v] for v in itertools.product([0,1],repeat=3)],dtype=np.int8)
XOR_TABLE=np.asarray([[int(v[i]^v[j]) for i,j in RELATIONS] for v in ASSIGNMENTS],dtype=np.int8)

@dataclasses.dataclass
class NoisyUnit:
    pair:Pair
    anchor_index:int
    anchor_label:int
    observed_bits:np.ndarray
    error_probs:np.ndarray
    regime:str
    is_correlated:bool
    order:tuple[int,...]

    def examples(self):
        items=self.pair.examples()
        return [items[i] for i in self.order]

def make_diverse_pair(n:int,seed:int,family:str,change:bool)->Pair:
    """Two copies of a directed network with cycles, shortcut routes, and multi-bridge cuts.
    The suffix permutation is an explicit directed isomorphism after rewiring.
    """
    rng=random.Random(seed)
    if n<10:raise ValueError('n >=10 needed')
    target=rng.randint(4,n-4)
    cut=rng.randint(2,target-1) if change else rng.randint(target,n-3)
    g=nx.DiGraph();g.add_nodes_from(range(n));g.add_edges_from((i,i+1) for i in range(n-1))
    rates={'chain_scc':(.65,.22), 'branch':(.75,.37), 'dense_scc':(.9,.50), 'layered':(.5,.29)}
    backward_rate, forward_rate=rates[family]
    # Dense local SCC structure, only backward and intra-side forward shortcuts.
    for j in range(2,n):
        for gap in (1,2,3,4):
            i=j-gap
            if i>=0 and rng.random()<backward_rate/(1+.17*(gap-1)) and ((j<=cut and i<=cut) or (j>cut and i>cut)):
                g.add_edge(j,i)
    for i in range(n):
        for d in (2,3,4,5):
            j=i+d
            if j<n and ((j<=cut and i<=cut) or (j>cut and i>cut)) and rng.random()<forward_rate:
                g.add_edge(i,j)
    # Multiple directed bridges cross the cut. All are switched as an intervention group.
    bridge_count=1 if family=='chain_scc' else (2 if family in ('branch','layered') else 3)
    bridges={(cut,cut+1)}
    candidates=[(a,b) for a in range(max(0,cut-3),cut+1)
                      for b in range(cut+1,min(n,cut+5)) if (a,b)!=(cut,cut+1)]
    rng.shuffle(candidates)
    bridges.update(candidates[:bridge_count-1])
    g.add_edges_from(bridges)
    g0=nx.disjoint_union(g,g);g1=g0.copy()
    for a,b in bridges:
        g1.remove_edge(a,b);g1.remove_edge(a+n,b+n)
        g1.add_edge(a,b+n);g1.add_edge(a+n,b)
    qa=(0,target);qb=(0,n+target)
    perm=list(g0.nodes());rng.shuffle(perm)
    mapping={v:perm[v] for v in range(2*n)}
    p=Pair(nx.relabel_nodes(g0,mapping),nx.relabel_nodes(g1,mapping),
           tuple(mapping[v] for v in qa),tuple(mapping[v] for v in qb),int(change),family,n,seed)
    check_pair(p,deep=False)
    return p

def check_pair(p:Pair,deep:bool=True):
    assert p.g0.number_of_nodes()==p.g1.number_of_nodes()
    assert p.g0.number_of_edges()==p.g1.number_of_edges()
    assert all(p.g0.in_degree(v)==p.g1.in_degree(v) and p.g0.out_degree(v)==p.g1.out_degree(v) for v in p.g0)
    assert [int(nx.has_path(g,q[0],q[1])) for g,q,y in p.examples()]==[y for g,q,y in p.examples()]
    if deep:
        assert nx.is_isomorphic(p.g0,p.g1)
        assert not nx.is_directed_acyclic_graph(p.g0)

def split(seed:int,n_values, families, per_cell:int):
    pairs=[]
    for n in n_values:
        for fam_idx,family in enumerate(families):
            for eff in [False,True]:
                for i in range(per_cell):
                    pairs.append(make_diverse_pair(n,seed+1009*n+9011*fam_idx+997*int(eff)+73*i,family,eff))
    random.Random(seed).shuffle(pairs)
    return pairs

def observe(pair:Pair,seed:int,regime:str,anchor_idx:int=0)->NoisyUnit:
    rng=random.Random(seed)
    order=list(range(4));rng.shuffle(order)
    truth=np.asarray([pair.examples()[i][2] for i in order],dtype=np.int8)
    correct=np.asarray([truth[i]^truth[j] for i,j in RELATIONS],dtype=np.int8)
    probs=[]
    for k in range(len(RELATIONS)):
        high_quality=rng.random()<.5
        if regime=='clean':p=0.
        elif regime=='moderate':p=.06 if high_quality else .30
        elif regime=='heavy':p=.16 if high_quality else .46
        elif regime=='correlated':p=.06 if high_quality else .30
        else:raise ValueError(regime)
        probs.append(p)
    reported=correct.copy()
    for k,p in enumerate(probs):
        if rng.random()<p:reported[k]^=1
    correlated=False
    # Failure mode: systematic corruption of all anchor-adjacent reports; reliability tags are misspecified.
    if regime=='correlated' and rng.random()<.28:
        correlated=True
        for k,(i,j) in enumerate(RELATIONS):
            if anchor_idx in (i,j):reported[k]^=1
    return NoisyUnit(pair,anchor_idx,int(truth[anchor_idx]),reported,np.array(probs),regime,correlated,tuple(order))

def relation_violation_rate(unit:NoisyUnit):
    # A 3-cycle is inconsistent iff XOR of its three observed edge labels is 1.
    edge={e:int(unit.observed_bits[k]) for k,e in enumerate(RELATIONS)}
    triangles=list(itertools.combinations(range(4),3))
    def eb(i,j):return edge[(min(i,j),max(i,j))]
    return sum(eb(a,b)^eb(a,c)^eb(b,c) for a,b,c in triangles)/len(triangles)

def posterior(unit:NoisyUnit,weighted:bool=True,soft=True):
    bits=unit.observed_bits
    p=np.clip(unit.error_probs if weighted else np.full(6,.22),.001,.499)
    likelihood=np.where(XOR_TABLE==bits[None,:],1-p[None,:],p[None,:])
    logprob=np.log(likelihood).sum(1)
    # canonical assignments have y0=0; transform through observed anchor.
    logprob=logprob-logprob.max()
    pr=np.exp(logprob);pr/=pr.sum()
    vals=ASSIGNMENTS^unit.anchor_label
    marginal=np.matmul(pr,vals.astype(float))
    if soft:return marginal,pr
    # Tie-break lexicographically; deliberately no use of hidden labels.
    return vals[int(np.argmax(pr))].astype(float),pr

def brittle(unit:NoisyUnit):
    # Tree closure only along 0->1, 0->2, 2->3; uses three relations and anchor.
    index={v:i for i,v in enumerate(RELATIONS)}
    bits=unit.observed_bits
    x=np.zeros(4,dtype=float);x[0]=unit.anchor_label
    x[1]=int(x[0])^int(bits[index[(0,1)]])
    x[2]=int(x[0])^int(bits[index[(0,2)]])
    x[3]=int(x[2])^int(bits[index[(2,3)]])
    return x

def infer(unit:NoisyUnit,arm:str):
    if arm=='anchor':
        return np.array([float(unit.anchor_label)]*4),np.array([1.0,0,0,0])
    if arm=='tree':return brittle(unit),np.ones(4)
    if arm=='unweighted_map':return posterior(unit,weighted=False,soft=False)[0],np.ones(4)
    if arm=='weighted_map':return posterior(unit,weighted=True,soft=False)[0],np.ones(4)
    if arm=='weighted_soft':return posterior(unit,weighted=True,soft=True)[0],np.ones(4)
    if arm=='oracle':return np.array([y for _,_,y in unit.examples()],dtype=float),np.ones(4)
    raise ValueError(arm)

def decode_metrics(regime,replications=2500):
    out={}
    for arm in ('tree','unweighted_map','weighted_map','weighted_soft'):
        err=0;quad=0;brier=0;entropy=0;contr=0;n=0;corr=0
        for i in range(replications):
            # Generate truth pattern with both flip/stay; no need for a graph oracle here.
            effect=bool(i%2);base=np.array([1,0,0,1] if effect else [1,0,1,0],dtype=np.int8)
            perm=list(range(4));random.Random(99937+i).shuffle(perm)
            labels=base[perm]
            # lightweight mimic NoisyUnit: use constructed reports based on mathematical ground truth.
            rng=random.Random(1223245+i*1103)
            probs=np.array([0 if regime=='clean' else ((.06 if rng.random()<.5 else .3) if regime in ('moderate','correlated') else (.16 if rng.random()<.5 else .46)) for _ in RELATIONS])
            true_bits=np.array([labels[a]^labels[b] for a,b in RELATIONS],dtype=np.int8)
            reported=true_bits.copy()
            for k,p in enumerate(probs):
                if rng.random()<p:reported[k]^=1
            bad=False
            if regime=='correlated' and rng.random()<.28:
                bad=True
                for k,(a,b) in enumerate(RELATIONS):
                    if a==0 or b==0:reported[k]^=1
            class Fake:pass
            u=Fake();u.observed_bits=reported;u.error_probs=probs;u.anchor_label=int(labels[0])
            val=infer(u,arm)[0]
            est=(val>=.5).astype(int)
            err+=sum(est!=labels);quad+=int(np.array_equal(est,labels))
            brier+=float(np.mean((val-labels)**2));n+=1;corr+=int(bad)
            if arm=='tree':
                edge={e:int(reported[k]) for k,e in enumerate(RELATIONS)}
                contr+=sum(edge[(a,b)]^edge[(a,c)]^edge[(b,c)] for a,b,c in itertools.combinations(range(4),3))>0
        out[arm]={'query_accuracy':1-err/(4*n),'quad_accuracy':quad/n,'brier':brier/n,'num':n}
        if arm=='tree':out['diagnostics']={'contradictory_fraction':contr/n,'correlated_fraction':corr/n}
    return out

def train(seed:int,arm:str,units:list[NoisyUnit],epochs=5,steps=13):
    torch.manual_seed(seed);torch.set_num_threads(1)
    model=Net(width=24); opt=torch.optim.AdamW(model.parameters(),lr=.005,weight_decay=0.)
    packs=[[pack(g,q,steps) for g,q,_ in u.examples()] for u in units]
    ytargets=[];weights=[]
    for u in units:
        ys,mask=infer(u,arm); ytargets.append(torch.tensor(ys,dtype=torch.float32));weights.append(torch.tensor(mask,dtype=torch.float32))
    t0=time.perf_counter();last=[]
    for epoch in range(epochs):
        order=list(range(len(units)));random.Random(seed*61+epoch).shuffle(order)
        for i in order:
            p=torch.stack([model(ex) for ex in packs[i]]).clamp(1e-5,1-1e-5)
            loss=(F.binary_cross_entropy(p,ytargets[i],reduction='none')*weights[i]).sum()/weights[i].sum().clamp_min(1)
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            last.append(float(loss.detach()))
    return model,{'updates':epochs*len(units),'seconds':round(time.perf_counter()-t0,3),'final_loss':float(np.mean(last[-min(50,len(last)):]))}

def run(args):
    torch.set_num_threads(1)
    train_pairs=split(314159,[10,12],['chain_scc','branch','layered'],args.percell)
    test_pairs=split(271828,[18,28,40],['dense_scc','layered'],args.testcell)
    for p in train_pairs[:8]+test_pairs[:8]:check_pair(p,True)
    out={'method':'Noisy intervention reports with one true endpoint anchor. All six parity relations observed.',
         'hypothesis':'Does uncertainty-aware parity reconstruction outperform common constraint-inference baselines for subsequent directed neural reachability learning?',
         'test_locked':True,'train_count':len(train_pairs),'test_count':len(test_pairs),'test_sizes':[18,28,40],
         'train_families':['chain_scc','branch','layered'],'test_families':['dense_scc','layered'],'unseen_family':['dense_scc'],'shared_family':['layered'],
         'training_notes':'No intermediate trace labels. Reports are corrupted, not certified. Exact BFS and true-label training are upper controls, not annotation-matched.',
         'oracle_BFS':{'query_accuracy':1.,'cqa':1.},'inference':{},'rows':[]}
    for regime in ('clean','moderate','heavy','correlated'):
        out['inference'][regime]=decode_metrics(regime,replications=args.decode_samples)
        print('DECODE',regime,{a:round(v['quad_accuracy'],3) for a,v in out['inference'][regime].items() if isinstance(v,dict) and 'quad_accuracy' in v},flush=True)
    target=pathlib.Path(args.out);target.parent.mkdir(parents=True,exist_ok=True)
    for regime in ('moderate','heavy','correlated'):
        for seed in args.seeds:
            units=[observe(p,seed*100003+i*31,regime) for i,p in enumerate(train_pairs)]
            for arm in ('anchor','tree','unweighted_map','weighted_map','weighted_soft','oracle'):
                model,info=train(seed,arm,units,epochs=args.epochs)
                score=evaluate(model,test_pairs)
                out['rows'].append({'regime':regime,'seed':seed,'arm':arm,'train':info,'eval':{k:v for k,v in score.items() if k!='slices'}})
                target.write_text(json.dumps(out,indent=2))
                print('RUN',regime,seed,arm, 'CQA',round(score['cqa'],3),'Q',round(score['accuracy'],3),'sec',info['seconds'],flush=True)
    target.write_text(json.dumps(out,indent=2))
    print('SAVED',target)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--percell',type=int,default=3);parser.add_argument('--testcell',type=int,default=4)
    parser.add_argument('--decode-samples',type=int,default=1200);parser.add_argument('--epochs',type=int,default=5)
    parser.add_argument('--seeds',type=int,nargs='+',default=[13,23,37]);parser.add_argument('--out',default='results/robust_study.json')
    run(parser.parse_args())
