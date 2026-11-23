"""External-network reachability feasibility protocol, NOT a neural benchmark result.

Source: SNAP p2p-Gnutella08 (directed network) when provided by the user.
No mirrored graph copies or hard-coded truth labels. The output reports selection
and failed-intervention rates, including cases with no reachability flip.

Download the file from https://snap.stanford.edu/data/p2p-Gnutella08.html
and pass --input /path/to/p2p-Gnutella08.txt.gz; we do not redistribute it.
"""
from __future__ import annotations
import argparse,gzip,hashlib,json,random
from pathlib import Path
import networkx as nx
import numpy as np

OFFICIAL_URL='https://snap.stanford.edu/data/p2p-Gnutella08.html'

def load_snap(path:Path)->tuple[nx.DiGraph,str]:
    if not path.is_file():raise FileNotFoundError(f'{path} not found; fetch the official directed edge list from {OFFICIAL_URL}')
    data=path.read_bytes()
    checksum=hashlib.sha256(data).hexdigest()
    with (gzip.open(path,'rt') if str(path).endswith('.gz') else path.open('rt')) as stream:
        g=nx.parse_edgelist((ln for ln in stream if ln.strip() and not ln.startswith('#')),
                           nodetype=int,create_using=nx.DiGraph())
    if len(g)<32 or g.number_of_edges()<32:raise ValueError('External graph too small for requested protocol')
    return g,checksum

def sample_nodes(g:nx.DiGraph,rng:random.Random,size:int):
    und=g.to_undirected(as_view=True)
    # Randomized breadth expansion samples one connected undirected neighborhood;
    # source and target selection remains independent of intervention outcome.
    seeds=list(g.nodes)
    for _ in range(100):
        first=rng.choice(seeds)
        selected={first};front=[first]
        while front and len(selected)<size:
            node=front.pop(0)
            neigh=list(und.neighbors(node));rng.shuffle(neigh)
            for v in neigh:
                if v not in selected:
                    selected.add(v);front.append(v)
                    if len(selected)>=size:break
        if len(selected)==size:return selected
    raise RuntimeError('Could not sample enough nodes from a connected neighborhood')

def directed_swap(g:nx.DiGraph,rng:random.Random,attempts:int=1500):
    edges=list(g.edges)
    if len(edges)<2:return None,attempts
    for t in range(1,attempts+1):
        (a,b),(c,d)=rng.sample(edges,2)
        if len({a,b,c,d})!=4:continue
        if g.has_edge(a,d) or g.has_edge(c,b):continue
        altered=g.copy();altered.remove_edge(a,b);altered.remove_edge(c,d)
        altered.add_edge(a,d);altered.add_edge(c,b)
        if altered.number_of_edges()!=g.number_of_edges():raise AssertionError('Edge count changed')
        if any(g.in_degree(v)!=altered.in_degree(v) or g.out_degree(v)!=altered.out_degree(v) for v in g):
            raise AssertionError('Named-vertex degree changed')
        return {'original_edges':[[a,b],[c,d]],'new_edges':[[a,d],[c,b]],'graph':altered},t
    return None,attempts

def draw_pair(g:nx.DiGraph,seed:int,size:int,swap_attempts:int=1500):
    rng=random.Random(seed)
    nodes=sample_nodes(g,rng,size)
    sub=g.subgraph(nodes).copy()
    # Relabel only to reduce downstream matrix sizes, not to insert mirrored copies.
    sub=nx.convert_node_labels_to_integers(sub,ordering='sorted')
    change,t=directed_swap(sub,rng,swap_attempts)
    if change is None:return {'seed':seed,'status':'no_valid_swap','swap_attempts':t}
    other=change.pop('graph')
    node_list=list(sub.nodes)
    src=rng.choice(node_list)
    targets=[v for v in node_list if v!=src]
    a,b=rng.sample(targets,2)
    qa=(src,a);qb=(src,b)
    labels=[int(nx.has_path(graph,*q)) for graph in (sub,other) for q in (qa,qb)]
    assert all(nx.has_path(graph,*q)==bool(labels[i*2+j]) for i,graph in enumerate((sub,other)) for j,q in enumerate((qa,qb)))
    assert all(sub.in_degree(v)==other.in_degree(v) and sub.out_degree(v)==other.out_degree(v) for v in sub)
    return {'seed':seed,'status':'valid','graph_nodes':size,'base_edges':sub.number_of_edges(),
            'base_edge_list':sorted([list(e) for e in sub.edges]),'change':change,
            'queries':[list(qa),list(qb)],'exact_oracle_labels':labels,
            'changed_answer_count':sum(labels[i]!=labels[i+2] for i in (0,1)),
            'has_positive_and_negative':len(set(labels))==2,'swap_attempts':t}

def generate(g:nx.DiGraph,hash_:str,seed:int,size:int,draws:int):
    rows=[draw_pair(g,seed+104729*i,size) for i in range(draws)]
    valid=[r for r in rows if r['status']=='valid']
    return {'source':OFFICIAL_URL,'source_sha256':hash_,'protocol':'one random, valid, degree-preserving two-edge swap per independent sampled natural-network neighborhood; no paired mirrors; randomized queries independent of outcome',
      'seed':seed,'sample_nodes':size,'requested_draws':draws,'original_graph_nodes':g.number_of_nodes(),
      'original_graph_edges':g.number_of_edges(),
      'valid_draws':len(valid),'failed_swaps':draws-len(valid),
      'with_query_reachability_change':sum(r['changed_answer_count']>0 for r in valid),
      'with_both_endpoint_classes':sum(r['has_positive_and_negative'] for r in valid),
      'important_caveat':'Random query sampling often yields no intervention effect. Do not train or claim external-domain generalization without confirming sufficient nontrivial class/flip balance and independent partitions.',
      'rows':rows}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--seed',type=int,default=34789);ap.add_argument('--size',type=int,default=64)
    ap.add_argument('--draws',type=int,default=64);ap.add_argument('--output',type=Path,default=Path('results/snap_external_feasibility.json'))
    args=ap.parse_args();g,checksum=load_snap(args.input)
    if args.size<32:raise ValueError('Minimum requested subgraph size is 32')
    result=generate(g,checksum,args.seed,args.size,args.draws)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))

if __name__=='__main__':main()
