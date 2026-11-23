"""Pre-registered-in-package pilot: neural marginalization of noisy intervention reports."""
from __future__ import annotations
import argparse,random,json,time,hashlib,pathlib,sys
from collections import defaultdict
import numpy as np,torch
from torch.nn import functional as F
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from witness_core.robust_study import split,observe,infer,check_pair
from witness_core.gated_baseline import GatedNet
from witness_core.legacy_directed import pack,evaluate
from method import constraint_nll,robust_soft_targets

SEEDS=[13,23,37]
ARMS=['anchor','unweighted_map','weighted_soft','robust_soft','joint_independent','joint_robust','oracle']
REGIMES=['heavy','correlated']
DATA_TRAIN=split(314159,[10,12],['chain_scc','layered'],3)
DATA_TEST=split(112358,[18,28,40],['branch','dense_scc'],4)

def validate_data():
    for p in DATA_TRAIN+DATA_TEST:check_pair(p,deep=False)
    assert len(DATA_TRAIN)==24 and len(DATA_TEST)==48
    assert set(p.n for p in DATA_TRAIN).isdisjoint(set(p.n for p in DATA_TEST))
    assert set(p.family for p in DATA_TRAIN).isdisjoint(set(p.family for p in DATA_TEST))
    compact=[(p.n,p.family,p.seed,p.effect,len(p.g0.edges)) for p in DATA_TEST]
    return hashlib.sha256(json.dumps(compact).encode()).hexdigest()

def train(seed,arm,regime,units,epochs=6):
    torch.manual_seed(seed)
    net=GatedNet(); opt=torch.optim.AdamW(net.parameters(),lr=.003)
    exs=[[pack(g,q,13) for g,q,_ in u.examples()] for u in units]
    target=[];mask=[]
    for u in units:
        ys,ms=(robust_soft_targets(u),np.ones(4)) if arm=='robust_soft' else infer(u,arm if arm not in ('joint_independent','joint_robust') else 'anchor')
        target.append(torch.tensor(ys,dtype=torch.float32))
        mask.append(torch.tensor(ms,dtype=torch.float32))
    t0=time.perf_counter();losses=[]
    for ep in range(epochs):
        order=list(range(len(units)));random.Random(seed*61+ep).shuffle(order)
        for i in order:
            preds=torch.stack([net(ex) for ex in exs[i]]).clamp(1e-5,1-1e-5)
            if arm in ('joint_independent','joint_robust'):
                loss=constraint_nll(preds,units[i], 'independent' if arm=='joint_independent' else 'correlation_mixture')
            else:
                loss=(F.binary_cross_entropy(preds,target[i],reduction='none')*mask[i]).sum()/mask[i].sum().clamp_min(1)
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
            losses.append(float(loss.detach()))
    runtime=time.perf_counter()-t0
    return net,{'updates':epochs*len(units),'graph_forwards':4*epochs*len(units),'training_seconds':round(runtime,2),'last_loss':round(float(np.mean(losses[-24:])),4)}

def main(args):
    torch.set_num_threads(1)
    p=pathlib.Path(args.out);p.parent.mkdir(parents=True,exist_ok=True)
    h=validate_data()
    out={'protocol': 'PROTOCOL.md','test_fingerprint':h,'data':{'train_quadruples':len(DATA_TRAIN),'test_quadruples':len(DATA_TEST),'n_train':[10,12],'n_test':[18,28,40],'train_families':['chain_scc','layered'],'test_families':['branch','dense_scc']},'rows':[]}
    regimes=REGIMES if args.regime=='all' else [args.regime]
    seeds=SEEDS if args.seed==0 else [args.seed]
    arms=ARMS if args.arm=='all' else [args.arm]
    for regime in regimes:
        for seed in seeds:
            units=[observe(a,seed*100003+i*31,regime) for i,a in enumerate(DATA_TRAIN)]
            for arm in arms:
                net,details=train(seed,arm,regime,units,epochs=args.epochs)
                score=evaluate(net,DATA_TEST)
                row={'regime':regime,'seed':seed,'arm':arm,'metadata':details,'cqa':score['cqa'],'accuracy':score['accuracy'],'positive_recall':score['positive_recall'],'negative_recall':score['negative_recall']}
                out['rows'].append(row)
                p.write_text(json.dumps(out,indent=2))
                print('RESULT',regime,seed,arm,'CQA',round(row['cqa'],4),'ACC',round(row['accuracy'],4),'SECS',details['training_seconds'],flush=True)
    p.write_text(json.dumps(out,indent=2)); print('DONE',p,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='results/pilot.json');ap.add_argument('--regime',choices=['all','heavy','correlated'],default='all');ap.add_argument('--arm',default='all');ap.add_argument('--seed',type=int,default=0);ap.add_argument('--epochs',type=int,default=6)
    main(ap.parse_args())
