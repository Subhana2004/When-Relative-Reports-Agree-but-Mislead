"""Post-hoc factorial training-size replication; separates three random factors.

Not a preregistered superiority test. All WITNESS weak-label arms receive
identical graph examples, weak reports and neural initial weights within cell.
Only the seed indices listed in protocol are considered independent sources;
repeated held-out graph draws are clustered by model.
"""
from __future__ import annotations
import argparse, json, pathlib, random, time
import numpy as np
import torch
from run_size_sweep import nested_training_sets, build_units, train_fixed_budget, prepare_draws
from WITNESS_TMLR_Anonymized_Supplement.experiments.extended_study import score_prepacked

SIZES=(24,48,96,192)
POOLS=(314159,271828,161803,577215)
REPORT_SEEDS=(9001,9007)
MODEL_SEEDS=(71,83)
ARMS=('weighted_soft','joint_independent')

def run(output, sizes=SIZES, pools=POOLS, report_seeds=REPORT_SEEDS,
        model_seeds=MODEL_SEEDS, regime='heavy', updates=192):
    torch.set_num_threads(1)
    all_draws=prepare_draws();draws=[all_draws[0],all_draws[5]]
    protocol={'status':'post-hoc factorial design, not preregistered','regime':regime,
       'training_quadruple_sizes':list(sizes),'training_graph_seeds':list(pools),
       'weak_report_seeds':list(report_seeds),'model_initialization_seeds':list(model_seeds),
       'fixed_optimizer_updates':updates,'graph_forwards_per_model':updates*4,
       'input_metadata':'both arms use the same six observed parities and per-report noise rates',
       'test_families':{'original':1,'arborescence':1},
       'test_graphs_are_reused_across_every_experimental_cell':True,
       'factorial_units':'training graph pool × weak report draw × neural initialization',
       'caution':'repeated graph draws from the same trained model are NOT independent training replications; dataset size and sample exposure are confounded at fixed updates',
       'arms':ARMS}
    output=pathlib.Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():
        state=json.loads(output.read_text());assert state['protocol']==protocol
        rows=state['rows'];done={tuple(z[k] for k in ['train_pool_seed','size','report_seed','model_seed','arm']) for z in rows}
    else:rows=[];done=set()
    started=time.monotonic()
    for pool_seed in pools:
        groups=nested_training_sets(pool_seed)
        for size in sizes:
            for report_seed in report_seeds:
                units=build_units(groups[size],report_seed,regime)
                for model_seed in model_seeds:
                    for arm in ARMS:
                        key=(pool_seed,size,report_seed,model_seed,arm)
                        if key in done:continue
                        net,trainmeta=train_fixed_budget(units,model_seed,arm,updates)
                        accuracies=[];per_draw=[]
                        for family,seed,draw in draws:
                            acc,correct=score_prepacked(net,draw)
                            accuracies.append(acc)
                            per_draw.append({'family':family,'graph_seed':seed,
                                             'correct_quadruples':int(sum(correct)),
                                             'n_quadruples':len(correct),'cqa':acc,
                                             'per_quadruple_correct':list(map(int,correct))})
                        rows.append({'train_pool_seed':pool_seed,'size':size,
                           'report_seed':report_seed,'model_seed':model_seed,
                           'arm':arm,'mean_unweighted_draw_cqa':float(np.mean(accuracies)),
                           'draws':per_draw,'training':trainmeta})
                        output.write_text(json.dumps({'protocol':protocol,'rows':rows},indent=2))
                        done.add(key)
                        print('FINISHED',pool_seed,size,report_seed,model_seed,arm,
                           'original',round(float(np.mean(accuracies[:1])),3),
                           'arborescence',round(float(np.mean(accuracies[1:])),3),
                           'elapsed',round(time.monotonic()-started,1),flush=True)
    return {'cells':len(rows),'elapsed_seconds':round(time.monotonic()-started,1)}

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='results/factorial_targeted.json')
    p.add_argument('--sizes',default='24,192')
    p.add_argument('--pools',default='314159,271828,161803,577215')
    p.add_argument('--reports',default='9001,9007')
    p.add_argument('--models',default='71,83')
    p.add_argument('--regime',choices=['heavy','correlated'],default='heavy')
    p.add_argument('--updates',type=int,default=192)
    a=p.parse_args()
    print(run(a.output,tuple(map(int,a.sizes.split(','))),tuple(map(int,a.pools.split(','))),
              tuple(map(int,a.reports.split(','))),tuple(map(int,a.models.split(','))),a.regime,a.updates),flush=True)
