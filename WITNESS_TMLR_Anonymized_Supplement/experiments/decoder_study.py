"""Model-free label-recovery stress test on distinct graph pairs and report draws.
Runs independently of neural training; oracle truth is used only for metrics.
"""
from __future__ import annotations
import sys,json, numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).resolve().parent))
from witness_core.robust_study import split,observe,infer
from method import robust_soft_targets


def main():
    graphs=split(4159265,[10,14,22,34],['chain_scc','branch','layered','dense_scc'],4)
    assert len(graphs)==128
    records=[]
    for regime in ['heavy','correlated']:
        for j,p in enumerate(graphs):
            for k in range(32):
                u=observe(p,99001+131*j+3001*k,regime)
                true=np.array([z[2] for z in u.examples()],dtype=int)
                for arm in ['tree','unweighted_map','weighted_map','weighted_soft','robust_soft']:
                    est=robust_soft_targets(u) if arm=='robust_soft' else infer(u,arm)[0]
                    pred=np.array(est>=.5,dtype=int)
                    records.append({'regime':regime,'family':p.family,'n':p.n,'pair_id':j,'report_draw':k,'arm':arm,
                                    'all_correct':int(np.array_equal(pred,true)), 'query_correct':float(np.mean(pred==true)),
                                    'brier':float(np.mean((est-true)**2)), 'correlated_event':int(u.is_correlated)})
    results={}
    for regime in ['heavy','correlated']:
        results[regime]={}
        for arm in ['tree','unweighted_map','weighted_map','weighted_soft','robust_soft']:
            r=[x for x in records if x['regime']==regime and x['arm']==arm]
            results[regime][arm]={'cqa':float(np.mean([x['all_correct'] for x in r])),
                                   'query_accuracy':float(np.mean([x['query_correct'] for x in r])),
                                   'brier':float(np.mean([x['brier'] for x in r]))}
    out=Path(__file__).resolve().parent/'results'/'decoder_extension.json'
    out.write_text(json.dumps({'n_graphs':len(graphs),'draws_per_graph':32,'regimes':results,'raw_rows':records},indent=2))
    print(json.dumps({'n_graphs':len(graphs),'draws_per_graph':32,'regimes':results},indent=2))
if __name__=='__main__':main()
