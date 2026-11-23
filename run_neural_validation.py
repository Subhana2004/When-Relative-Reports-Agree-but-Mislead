"""Matched neural end-model validation of published-algorithm adaptations.
Published math is reproduced; upstream original research code is NOT executed.
"""
import argparse,json,sys,time,random
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE/'WITNESS_TMLR_Anonymized_Supplement'),str(HERE/'WITNESS_TMLR_Anonymized_Supplement'/'experiments')]
from witness_core.robust_study import split,observe
from witness_core.gated_baseline import GatedNet
from witness_core.legacy_directed import pack
from extended_study import SEEDS,GRAPH_SEEDS,make_test_draws,score_prepacked,train as original_train
from published_baselines import spectral_synchronization,train_ds,seed_parity_truth_tests
from independent_graphs import test_draw
OUT=HERE/'results';OUT.mkdir(exist_ok=True)
torch.set_num_threads(1)

def packed_draw(pairs):
    return [([pack(g,q,p.n+2) for g,q,_ in p.examples()],
             np.asarray([y for _,_,y in p.examples()],dtype=np.int8)) for p in pairs]

def train(seed,arm,regime,units,prepared):
    if arm in ('weighted_soft','robust_soft','joint_independent','joint_robust'):
        return original_train(seed,arm,regime,units,prepared,6)
    torch.manual_seed(seed);net=GatedNet();opt=torch.optim.AdamW(net.parameters(),lr=.003)
    ds=train_ds(units) if arm=='dawid_skene_soft' else None
    targets=[]
    for u in units:
        v=spectral_synchronization(u,True) if arm=='spectral_weighted' else ds.predict_unit(u)
        targets.append(torch.tensor(v,dtype=torch.float32))
    t0=time.perf_counter()
    for ep in range(6):
        order=list(range(len(units)));random.Random(seed*61+ep).shuffle(order)
        for i in order:
            pred=torch.stack([net(x) for x in prepared[i]]).clamp(1e-5,1-1e-5)
            loss=F.binary_cross_entropy(pred,targets[i])
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
    meta={'training_seconds':round(time.perf_counter()-t0,3),'updates':len(units)*6,'graph_forwards':4*len(units)*6}
    if ds is not None: meta.update({'dawid_skene_confusion':ds.confusion_.tolist(),'dawid_skene_class_prior':ds.priors_.tolist(),'dawid_skene_em_iterations':len(ds.history_)})
    return net,meta

def main(n):
    seed_parity_truth_tests()
    training=split(314159,[10,12],['chain_scc','layered'],3)
    old_draws=make_test_draws()
    other_seeds=[20261,20262,20263]
    new_draws=[packed_draw(test_draw(s)) for s in other_seeds]
    protocol={'train_graph_seed':314159,'seeds':SEEDS[:n],
       'original_draw_seeds':GRAPH_SEEDS,'independent_draw_seeds':other_seeds,
       'train_quadruples':24,'epochs':6,'updates':144,'graph_forwards':576,
       'new_test_family':'random arborescence with within-cut stochastic arcs and certified degree-preserving intervention',
       'new_families':['random_recursive','preferential','balanced_branch','layered_random'],
       'new_quadruples_per_draw':48,'published_code_ran':False,
       'faithfulness':'published mathematical estimators, task-adapted input; not original implementation or published benchmark'}
    out=OUT/'neural_validation.json';rows=[];t0=time.perf_counter()
    old=json.loads((HERE/'WITNESS_TMLR_Anonymized_Supplement'/'experiments'/'results'/'extension.json').read_text())
    reference={(r['regime'],r['seed'],r['arm'],r['graph_draw_index']):float(r['cqa']) for r in old['rows']}
    checks=[]
    for regime in ('heavy','correlated'):
        arms=['weighted_soft','joint_independent','spectral_weighted','dawid_skene_soft'] if regime=='heavy' else ['robust_soft','joint_robust','spectral_weighted','dawid_skene_soft']
        for seed in SEEDS[:n]:
            units=[observe(p,seed*100003+i*31,regime) for i,p in enumerate(training)]
            prepared=[[pack(g,q,13) for g,q,_ in u.examples()] for u in units]
            for arm in arms:
                net,meta=train(seed,arm,regime,units,prepared)
                for s,draw in zip(GRAPH_SEEDS,old_draws):
                    cqa,correct=score_prepacked(net,draw)
                    rows.append({'regime':regime,'seed':seed,'arm':arm,'draw_kind':'old_generator','draw_seed':s,
                       'cqa':cqa,'n_quadruples':len(correct),'correct':correct,'train':meta})
                    key=(regime,seed,arm,GRAPH_SEEDS.index(s))
                    if key in reference:checks.append(abs(cqa-reference[key]))
                for s,draw in zip(other_seeds,new_draws):
                    cqa,correct=score_prepacked(net,draw)
                    rows.append({'regime':regime,'seed':seed,'arm':arm,'draw_kind':'arborescence','draw_seed':s,
                       'cqa':cqa,'n_quadruples':len(correct),'correct':correct,'train':meta})
                out.write_text(json.dumps({'protocol':protocol,'crosscheck_max_diff':max(checks or [0]),'rows':rows},indent=2))
                a=[r['cqa'] for r in rows if r['regime']==regime and r['seed']==seed and r['arm']==arm and r['draw_kind']=='arborescence']
                b=[r['cqa'] for r in rows if r['regime']==regime and r['seed']==seed and r['arm']==arm and r['draw_kind']=='old_generator']
                print(f'COMPLETE {regime} {seed} {arm} old={np.mean(b):.3f} new={np.mean(a):.3f} train={meta["training_seconds"]:.1f}s elapsed={time.perf_counter()-t0:.1f}s',flush=True)
    assert not checks or max(checks)<1e-9,('prior reported baseline mismatch',max(checks))
    print('ALL DONE',len(rows),'crosscheck max',max(checks or [0]),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seeds',type=int,default=12)
    main(ap.parse_args().seeds)
