"""Held-out stochastic *arborescence* graph families with a certified degree-preserving switch.

These are new synthetic topology families, NOT real-world graph benchmarks.
A rooted random tree and within-cut random cyclic edges replace the chain backbone.
"""
from __future__ import annotations
import random, networkx as nx, numpy as np, sys, pathlib
sys.path.insert(0,str(pathlib.Path(__file__).parent/'WITNESS_TMLR_Anonymized_Supplement'))
from witness_core.legacy_directed import Pair,pack
from witness_core.robust_study import check_pair

def one_pair(n,seed,family,change):
    r=random.Random(seed)
    parent={0:None};popularity=[1.0];depth={0:0}
    for i in range(1,n):
        if family=='random_recursive': p=r.randrange(i)
        elif family=='preferential':p=r.choices(range(i),weights=popularity,k=1)[0]
        elif family=='balanced_branch':p=(i-1)//2
        elif family=='layered_random':
            level=max(0,max(depth.values())-2)
            eligible=[j for j in range(i) if depth[j]>=level]
            p=r.choice(eligible)
        else: raise ValueError(family)
        parent[i]=p;depth[i]=depth[p]+1;popularity[p]+=1;popularity.append(1.0)
    tree=nx.DiGraph();tree.add_nodes_from(range(n));tree.add_edges_from((p,i) for i,p in parent.items() if p is not None)
    possible=[]
    for v in range(1,n):
        desc=nx.descendants(tree,v)|{v}
        if 2<=len(desc)<=n-3:
            possible.append((parent[v],v,desc))
    if not possible:raise RuntimeError('no valid cut')
    u,v,desc=r.choice(possible)
    outside=set(range(n))-desc
    target=r.choice(sorted(desc if change else (outside-{0})))
    base=tree.copy()
    # Sample extra arcs only inside the two sides; no bypass across the cut.
    density={'random_recursive':.08,'preferential':.10,'balanced_branch':.13,'layered_random':.09}[family]
    for part in (sorted(desc),sorted(outside)):
        for a in part:
            for b in part:
                if a!=b and not base.has_edge(a,b) and r.random()<density:
                    base.add_edge(a,b)
    # Guaranteed directed cycle on the outside of the cut if >=2 non-root nodes.
    cyc=[x for x in outside if x!=0 and parent[x] in outside]
    if cyc:base.add_edge(cyc[0],parent[cyc[0]])
    g0=nx.disjoint_union(base,base);g1=g0.copy()
    assert g1.has_edge(u,v) and g1.has_edge(u+n,v+n)
    g1.remove_edge(u,v);g1.remove_edge(u+n,v+n)
    g1.add_edge(u,v+n);g1.add_edge(u+n,v)
    qa=(0,target);qb=(0,target+n)
    perm=list(range(2*n));r.shuffle(perm);mapping=dict(enumerate(perm))
    p=Pair(nx.relabel_nodes(g0,mapping),nx.relabel_nodes(g1,mapping),
           tuple(mapping[k] for k in qa),tuple(mapping[k] for k in qb),int(change),family,n,seed)
    check_pair(p,deep=True)
    return p

def test_draw(draw_seed,n_values=(22,34,48),families=('random_recursive','preferential','balanced_branch','layered_random')):
    pairs=[]
    for n in n_values:
        for fam in families:
            for flip in (False,True):
                i=0; successes=0
                while successes<2 and i<25:
                    seed=draw_seed + n*173+sum(map(ord,fam))*277 + 93971*int(flip)+10007*i
                    i+=1
                    try:
                        pairs.append(one_pair(n,seed,fam,flip));successes+=1
                    except RuntimeError:pass
                if successes!=2:raise RuntimeError(('generation exhausted',n,fam,flip))
    random.Random(draw_seed).shuffle(pairs)
    return pairs

if __name__=='__main__':
    for sd in (20261,20262):
        p=test_draw(sd)
        print(sd,len(p),'nodes',set(v.g0.number_of_nodes() for v in p),'edges',min(v.g0.number_of_edges() for v in p),max(v.g0.number_of_edges() for v in p))
