"""Independent, *single-network* directed interventions, never mirrored copies.

Three unequal structural zones: upstream A, downstream B, inaccessible decoy C.
A two-edge directed swap either redirects A->B to A->C (causal effect) or
rewires two C-only edges (no effect). Every node retains in/out degree.
The zones are independently randomized arborescences with distinct chords;
no two-copy scaffolds, cross-copy edge exchanges or isomorphism requirement.
Outcome-balanced evaluation is BY DESIGN and disclosed; do not confuse this
with natural unconditioned two-edge swaps on SNAP neighborhoods.
"""
from __future__ import annotations
import random, networkx as nx
from witness_core.legacy_directed import Pair,pack

def connected_arborescence(r,nodes,root,p_extra=.10,cycles=False):
    nodes=list(nodes);g=nx.DiGraph();g.add_nodes_from(nodes)
    for k in range(1,len(nodes)):
        par=r.choice(nodes[:k]);g.add_edge(par,nodes[k])
    for i,a in enumerate(nodes):
        for j,b in enumerate(nodes):
            if i>=j or g.has_edge(a,b):continue
            if r.random()<p_extra:g.add_edge(a,b)
    if cycles:
        g.add_edge(nodes[-1],root)
    return g

def _swap(g,ab,cd):
    a,b=ab;c,d=cd
    assert len({a,b,c,d})==4
    assert g.has_edge(a,b) and g.has_edge(c,d)
    assert not g.has_edge(a,d) and not g.has_edge(c,b)
    other=g.copy();other.remove_edge(a,b);other.remove_edge(c,d)
    other.add_edge(a,d);other.add_edge(c,b)
    assert all(g.in_degree(v)==other.in_degree(v) and g.out_degree(v)==other.out_degree(v) for v in g)
    return other

def make_pair(n,seed,change=True):
    if n<18:raise ValueError('component-size parameter n must be >= 18')
    r=random.Random(seed);N=2*n
    na=int(N*.39);nb=int(N*.34);nc=N-na-nb
    A=list(range(na));B=list(range(na,na+nb));C=list(range(na+nb,N))
    g=nx.DiGraph();g.add_nodes_from(range(N))
    for group,p,cycles in ((A,.07,True),(B,.11,True),(C,.09,False)):
        g.add_edges_from(connected_arborescence(r,group,group[0],p,cycles).edges())
    root=A[0];broot=B[0];croot=C[0]
    u=r.choice(A[1:]);g.add_edge(u,broot)
    assert nx.has_path(g,root,broot)
    # Since C is a rooted DAG, redirecting to a child cannot reach its root.
    cchildren=list(g.successors(croot));assert cchildren
    d=r.choice(cchildren)
    if change:
        out=_swap(g,(u,broot),(croot,d))
    else:
        # A neutral intervention confined to the inaccessible decoy module.
        eligible=list(g.subgraph(C).edges());r.shuffle(eligible);out=None
        for ab in eligible:
            for cd in eligible:
                if len(set(ab+cd))!=4:continue
                if g.has_edge(ab[0],cd[1]) or g.has_edge(cd[0],ab[1]):continue
                out=_swap(g,ab,cd);break
            if out is not None:break
        if out is None:raise RuntimeError('no decoy-only neutral switch found')
    btarget=r.choice(B); q_a=(root,btarget);q_b=(root,d)
    # Random node permutation hides the structural zone IDs from the GNN.
    perm=list(range(N));r.shuffle(perm);mapping=dict(zip(range(N),perm))
    p=Pair(nx.relabel_nodes(g,mapping),nx.relabel_nodes(out,mapping),
           tuple(mapping[k] for k in q_a),tuple(mapping[k] for k in q_b),
           int(bool(change)),'nonmirrored_three_zone',n,seed)
    expected=[1,0,0,1] if change else [1,0,1,0]
    actual=[int(nx.has_path(G,*q)) for G,q,_ in p.examples()]
    assert actual==expected,(expected,actual)
    assert p.g0.number_of_edges()==p.g1.number_of_edges()
    assert all(p.g0.in_degree(v)==p.g1.in_degree(v) and p.g0.out_degree(v)==p.g1.out_degree(v) for v in p.g0)
    assert not nx.is_directed_acyclic_graph(p.g0)
    return p

def draw(seed,sizes=(22,34,48),per_stratum=4):
    out=[]
    for n in sizes:
        for eff in (False,True):
            for i in range(per_stratum):
                k=0
                while True:
                    trial_seed=seed+n*17011+int(eff)*900001+i*1701+k*1597
                    try:p=make_pair(n,trial_seed,eff);break
                    except RuntimeError:
                        k+=1
                        if k>=30:raise
                out.append(p)
    random.Random(seed).shuffle(out)
    return out

if __name__=='__main__':
    import json
    for seed in (6101,6103,6109,6113,6119,6121):
        items=draw(seed);print(seed,len(items),sum(p.effect for p in items),
          [p.g0.number_of_nodes() for p in items[:3]],flush=True)
