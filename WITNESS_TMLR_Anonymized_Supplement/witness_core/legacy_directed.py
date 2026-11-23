"""WITNESS directed-cycle counterfactual pilot. Independent program, no hidden trace supervision.
Certificate bits: query a/b have opposite labels within each graph;
shared structural intervention flips both query labels iff the bridge precedes the targets.
"""
from __future__ import annotations
import argparse, json, random, copy, pathlib, time, math
from dataclasses import dataclass
import networkx as nx
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

@dataclass
class Pair:
    g0:nx.DiGraph
    g1:nx.DiGraph
    qa:tuple
    qb:tuple
    effect:int
    family:str
    n:int
    seed:int

    def examples(self):
        y=(1,0,0,1) if self.effect else (1,0,1,0)
        return [(self.g0,self.qa,y[0]),(self.g0,self.qb,y[1]),(self.g1,self.qa,y[2]),(self.g1,self.qb,y[3])]

def make_pair(n:int,seed:int,family:str,change:bool)->Pair:
    rng=random.Random(seed)
    if n<9:raise ValueError('n too short')
    target=rng.randint(3,n-4)
    cut=rng.randint(1,target-1) if change else rng.randint(target,n-2)
    base=nx.DiGraph();base.add_nodes_from(range(n));base.add_edges_from((i,i+1) for i in range(n-1))
    # Strictly backward chords *within* the cut, so they form directed cycles
    # without creating another route around the intervention.
    spans={'cycle2':[1], 'cycle3':[2], 'mixed':[1,2,3], 'cycle4':[3], 'dense':[1,2,3,4]}[family]
    for j in range(1,n):
        for gap in spans:
            i=j-gap
            if i>=0 and (j<=cut or i>cut) and (rng.random()<(.65 if family!='dense' else .8)):
                base.add_edge(j,i)
    g0=nx.disjoint_union(base,base)
    g1=g0.copy()
    g1.remove_edge(cut,cut+1);g1.remove_edge(cut+n,cut+1+n)
    g1.add_edge(cut,cut+1+n);g1.add_edge(cut+n,cut+1)
    qa=(0,target);qb=(0,n+target)
    perm=list(g0.nodes);rng.shuffle(perm)
    mapping=dict(zip(list(g0.nodes),perm))
    g0=nx.relabel_nodes(g0,mapping);g1=nx.relabel_nodes(g1,mapping)
    pair=Pair(g0,g1,tuple(mapping[x] for x in qa),tuple(mapping[x] for x in qb),int(change),family,n,seed)
    validate(pair,deep=False)
    return pair

def validate(p:Pair, deep:bool=True):
    assert sorted(p.g0.nodes)==sorted(p.g1.nodes)
    assert p.g0.number_of_edges()==p.g1.number_of_edges()
    assert all(p.g0.in_degree(v)==p.g1.in_degree(v) and p.g0.out_degree(v)==p.g1.out_degree(v) for v in p.g0)
    assert [int(nx.has_path(g,q[0],q[1])) for g,q,_ in p.examples()]==[y for _,_,y in p.examples()]
    if deep:
        assert nx.is_isomorphic(p.g0,p.g1)
        # Directed cycles explicitly present if any backward arcs generated
        assert not nx.is_directed_acyclic_graph(p.g0)

def data_split(name:str,seed:int, nvals:list[int],families:list[str],items_per_cell:int):
    out=[]
    for n in nvals:
        for family in families:
            for change in (False,True):
                for i in range(items_per_cell):
                    out.append(make_pair(n,seed+n*3149+families.index(family)*11881+int(change)*117 +i*31,family,change))
    random.Random(seed).shuffle(out)
    return out

def pack(g,q,steps):
    nodes=sorted(g.nodes()); idx={v:i for i,v in enumerate(nodes)}
    # A[dst,src] for directed message propagation
    mat=nx.to_numpy_array(g,nodelist=nodes,dtype=np.float32).T.copy()
    adj=torch.from_numpy(mat).bool()
    src=torch.zeros(len(nodes));src[idx[q[0]]]=1.
    return (adj,src,idx[q[1]],steps)

class Net(nn.Module):
    def __init__(self,width=24):
        super().__init__()
        self.f=nn.Sequential(nn.Linear(3,width),nn.Tanh(),nn.Linear(width,width),nn.Tanh(),nn.Linear(width,1))
    def forward(self,ex):
        adj,source,target,steps=ex
        h=source
        for _ in range(steps):
            nbr=torch.where(adj,h.unsqueeze(0),0.).amax(1)
            h=torch.sigmoid(self.f(torch.stack((h,nbr,source),-1)).squeeze(-1))
        return h[target]

def pack_pair(p,steps):
    return tuple(pack(g,q,steps) for g,q,_ in p.examples())

def cert_loss(pred, effect, mode):
    # No intervention direction labels; only whether paired decisions flip,
    # plus a within-graph complement relation. Both are graph certificates.
    eps=1e-5
    p=pred.clamp(eps,1-eps)
    # effect=1 -> cross-graph query labels opposite; otherwise same.
    q0=p[0]*(1-p[2])+(1-p[0])*p[2]
    q1=p[1]*(1-p[3])+(1-p[1])*p[3]
    # all graph states have one reachable and one unreachable query
    r0=p[0]*(1-p[1])+(1-p[0])*p[1]
    r1=p[2]*(1-p[3])+(1-p[2])*p[3]
    targets=pred.new_tensor([effect,effect,1.,1.])
    probs=torch.stack((q0,q1,r0,r1)).clamp(eps,1-eps)
    return F.binary_cross_entropy(probs,targets)

def completed_labels(anchor_index:int, anchor_label:int, effect:int):
    """Propagate one observed answer through certified pair relations."""
    graph={0:[(1,1),(2,effect)],1:[(0,1),(3,effect)],2:[(0,effect),(3,1)],3:[(1,effect),(2,1)]}
    known={anchor_index:int(anchor_label)}; todo=[anchor_index]
    while todo:
        i=todo.pop()
        for j,bit in graph[i]:
            value=known[i]^int(bit)
            if j in known: assert known[j]==value
            else: known[j]=value;todo.append(j)
    assert len(known)==4
    return [known[j] for j in range(4)]

def train(seed,arm,items,epochs=8,train_steps=13):
    torch.manual_seed(seed);random.seed(seed)
    net=Net(); opt=torch.optim.AdamW(net.parameters(),lr=.006)
    started=time.perf_counter(); loss_curve=[]
    packed=[pack_pair(p,train_steps) for p in items]
    for ep in range(epochs):
        order=list(range(len(items)));random.Random(seed*61+ep).shuffle(order)
        for i in order:
            preds=torch.stack([net(ex) for ex in packed[i]])
            oracle=preds.new_tensor([y for _,_,y in items[i].examples()])
            anchor=i%4  # one exact query-answer per quadruple in all arms
            loss=F.binary_cross_entropy(preds[anchor].clamp(1e-5,1-1e-5),oracle[anchor])
            if arm=='anchor_cert':
                loss=loss+cert_loss(preds,items[i].effect,arm)
            elif arm=='completed':
                # Same supervision is recoverable from one anchor plus the
                # certified within-graph complement and cross-graph parity.
                derived=preds.new_tensor(completed_labels(anchor,int(oracle[anchor].item()),items[i].effect))
                loss=F.binary_cross_entropy(preds.clamp(1e-5,1-1e-5),derived)
            elif arm=='anchor':pass
            else:raise ValueError(arm)
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
            loss_curve.append(float(loss.detach()))
    return net,{'seconds':round(time.perf_counter()-started,2),'updates':len(loss_curve),'last_loss':float(np.mean(loss_curve[-min(50,len(loss_curve)):]))}

@torch.no_grad()
def evaluate(net,items):
    net.eval();tot=0;good=0;quad_good=0;effect_good=0;pos_good=neg_good=0;pos_num=neg_num=0
    slices={}
    for p in items:
        outputs=[];ys=[]
        for g,q,y in p.examples():
            pred=float(net(pack(g,q,p.n+2)))>=.5;outputs.append(int(pred));ys.append(y)
            good+=int(pred==y);tot+=1
            if y:pos_num+=1;pos_good+=int(pred==y)
            else:neg_num+=1;neg_good+=int(pred==y)
        ok=int(outputs==ys)
        flip=int((outputs[0]!=outputs[2])==bool(p.effect) and (outputs[1]!=outputs[3])==bool(p.effect))
        quad_good+=ok;effect_good+=flip
        key=f'{p.family}/n={p.n}/effect={p.effect}'
        if key not in slices:slices[key]={'n':0,'quad_correct':0,'query_correct':0}
        slices[key]['n']+=1;slices[key]['quad_correct']+=ok;slices[key]['query_correct']+=sum(x==y for x,y in zip(outputs,ys))
    for s in slices.values():
        s['cqa']=s['quad_correct']/s['n'];s['query_accuracy']=s['query_correct']/(4*s['n'])
    return {'accuracy':good/tot,'cqa':quad_good/len(items),'effect_correct':effect_good/len(items),'positive_recall':pos_good/pos_num,'negative_recall':neg_good/neg_num,'n_pairs':len(items),'slices':slices}

def run(args):
    torch.set_num_threads(1)
    tr=data_split('train',727301,[9,11],['cycle2','cycle3','mixed'],args.percell)
    te=data_split('test',873091,[16,24,36],['cycle4','dense'],args.test_percell)
    print('datasets',len(tr),len(te),flush=True)
    for p in tr[:4]+te[:4]:validate(p,deep=True)
    out={'hypothesis':'Intervention parity certificates with one anchor vs anchor-only and explicit label completion',
         'data':{'train_pairs':len(tr),'test_pairs':len(te),'train_families':['cycle2','cycle3','mixed'],'test_families':['cycle4','dense'], 'test_n':[16,24,36]},
         'protocol':{'training_anchors_per_quadruple':1,'certificate_relations_per_quadruple':4,'no_intermediate_traces':True,'complete_labels_algebraically_derived_from_anchor_plus_certificates':True}, 'rows':[]}
    target=pathlib.Path(args.out);target.parent.mkdir(parents=True,exist_ok=True)
    for seed in (13,23,37):
        for arm in ('anchor','anchor_cert','completed'):
            net,meta=train(seed,arm,tr,epochs=args.epochs)
            res=evaluate(net,te)
            row={'seed':seed,'arm':arm,'train':meta,'eval':res}
            out['rows'].append(row)
            target.write_text(json.dumps(out,indent=2))
            print('RESULT',seed,arm,'acc',round(res['accuracy'],4),'cqa',round(res['cqa'],4),'effect',round(res['effect_correct'],4),'recall',round(res['positive_recall'],3),round(res['negative_recall'],3),'time',meta['seconds'],flush=True)
    target.write_text(json.dumps(out,indent=2));print('SAVED',target,flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='results/directed_pilot.json');ap.add_argument('--percell',type=int,default=4);ap.add_argument('--test-percell',type=int,default=6);ap.add_argument('--epochs',type=int,default=6)
    run(ap.parse_args())
