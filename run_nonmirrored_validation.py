"""Previously unseen single-network intervention evaluation; never train on this family.

This is NOT real-world graph generalization, and training uses only the old mirrored
family. Reports are independently resampled for every model seed and regime.
Outcome patterns on this constructed stress test are deliberately balanced.
"""
from __future__ import annotations
import json,time,argparse,random,sys
from pathlib import Path
import numpy as np, torch
from nonmirrored_graphs import draw
from witness_core.robust_study import split,observe
from witness_core.legacy_directed import pack
from WITNESS_TMLR_Anonymized_Supplement.experiments.extended_study import train as old_train,SEEDS

@torch.no_grad()
def score(net,items):
    net.eval();correct=[];queries=[];positives=[0,0];negatives=[0,0]
    for packs,ys in items:
        pred=[]
        for off in (0,2):
            adj,source,target,steps=packs[off]
            h=torch.tanh(net.initial(source[:,None]))
            for _ in range(steps):
                nbr=torch.where(adj[:,:,None],h[None,:,:],torch.full_like(h[None,:,:],-1e5)).amax(1)
                nbr=torch.where(adj.any(1,keepdim=True),nbr,0.)
                h=net.gate(torch.cat((nbr,source[:,None]),-1),h)
            z=torch.sigmoid(net.readout(h).squeeze(-1))
            pred.extend([int(float(z[packs[off+j][2]])>=.5) for j in (0,1)])
        correct.append(int(np.array_equal(np.array(pred,dtype=np.int8),ys)))
        for a,b in zip(pred,ys):
            queries.append(int(a==b))
            tally=positives if b else negatives
            tally[0]+=int(a==b);tally[1]+=1
    return {'cqa':sum(correct)/len(correct),'query_accuracy':sum(queries)/len(queries),
            'positive_recall':positives[0]/positives[1],
            'negative_recall':negatives[0]/negatives[1],
            'correct_count':sum(correct),'quadruples':len(correct),'correct':correct}

def main(args):
    torch.set_num_threads(1)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    train_graphs=split(314159,[10,12],['chain_scc','layered'],3)
    seeds=SEEDS[:args.seeds]
    draws_seeds=[6101,6103,6109,6113]
    external=[]
    for sd in draws_seeds:
        pairs=draw(sd)
        external.append([([pack(g,q,p.n+2) for g,q,_ in p.examples()],
                           np.asarray([y for _,_,y in p.examples()],dtype=np.int8)) for p in pairs])
    protocol={'status':'exploratory non-mirrored topology shift','training_graph_seed':314159,
     'training_quadruples':24,'seeds':seeds,'noise_regimes':['heavy','correlated'],
     'draw_seeds':draws_seeds,'draw_quadruples':24,'graph_size_nodes':[44,68,96],
     'new_generator':'single asymmetric 3-zone network with two-edge in/out-degree-preserving switch',
     'training_on_new_generator':False,
     'caveats':'new graphs still synthetically constructed, intentionally balanced and structurally distinct from but not equivalent to natural graphs'}
    if output.exists():
        obj=json.loads(output.read_text());assert obj['protocol']==protocol
        rows=obj['rows'];done={(z['regime'],z['seed'],z['arm']) for z in rows}
    else:rows=[];done=set()
    started=time.monotonic()
    for regime in ('heavy','correlated'):
        for seed in seeds:
            units=[observe(p,seed*100003+i*31,regime) for i,p in enumerate(train_graphs)]
            prepared=[[pack(g,q,13) for g,q,_ in u.examples()] for u in units]
            arms=['weighted_soft','joint_independent'] if regime=='heavy' else ['robust_soft','joint_robust']
            for arm in arms:
                if (regime,seed,arm) in done:continue
                net,meta=old_train(seed,arm,regime,units,prepared,6)
                evaluations=[{'draw_seed':sd,**score(net,evals)} for sd,evals in zip(draws_seeds,external)]
                rows.append({'regime':regime,'seed':seed,'arm':arm,'training':meta,'draws':evaluations})
                output.write_text(json.dumps({'protocol':protocol,'rows':rows},indent=2))
                print('COMPLETE',regime,seed,arm,
                      'meanCQA',round(float(np.mean([x['cqa'] for x in evaluations])),4),
                      'mean_query',round(float(np.mean([x['query_accuracy'] for x in evaluations])),4),
                      'elapsed',round(time.monotonic()-started,1),flush=True)
    print('END',len(rows),'elapsed',round(time.monotonic()-started,1),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,default=12)
    p.add_argument('--output',default='results/nonmirrored_neural.json');main(p.parse_args())
