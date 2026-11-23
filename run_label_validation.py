"""Held-out graph-family/noise validation of published algorithm adaptations.
No labels from evaluation examples are used by the unsupervised label models.
"""
import json,sys,random,collections
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE/'WITNESS_TMLR_Anonymized_Supplement')]
from witness_core.robust_study import split,observe,infer
from experiments.method import robust_soft_targets
from published_baselines import train_ds,spectral_synchronization,seed_parity_truth_tests
from independent_graphs import test_draw
seed_parity_truth_tests()
OUT=HERE/'results';OUT.mkdir(exist_ok=True)

methods=['tree','unweighted_map','weighted_map','weighted_soft','spectral_unweighted','spectral_weighted','dawid_skene','mixture_soft']
all_results={}
for regime in ('heavy','correlated'):
    tr_pairs=split(99007,[10,14,22,34],['chain_scc','layered','branch','dense_scc'],3)
    va_pairs=split(323089,[10,14,22,34],['chain_scc','layered','branch','dense_scc'],4)
    train=[observe(pair,2001103+157*i+1109*k,regime) for i,pair in enumerate(tr_pairs) for k in range(8)]
    validation=[(i,observe(pair,3100103+983*i+173*k,regime)) for i,pair in enumerate(va_pairs) for k in range(8)]
    ds=train_ds(train)
    assert len(set(p.seed for p in tr_pairs)&set(p.seed for p in va_pairs))==0
    rows=[]
    for pair_idx,u in validation:
        y=np.array([v for _,_,v in u.examples()],dtype=float)
        for method in methods:
            if method in ('tree','unweighted_map','weighted_map','weighted_soft'):
                v=infer(u,method)[0]
            elif method=='spectral_unweighted':v=spectral_synchronization(u,False)
            elif method=='spectral_weighted':v=spectral_synchronization(u,True)
            elif method=='dawid_skene':v=ds.predict_unit(u)
            elif method=='mixture_soft':v=robust_soft_targets(u)
            pred=(np.asarray(v)>=.5).astype(int)
            rows.append({'pair_index':pair_idx,'family':u.pair.family,'n':u.pair.n,'effect':u.pair.effect,
                'model':method,'all_correct':int(np.array_equal(pred,y)),'query_correct':int(np.sum(pred==y)),
                'brier':round(float(np.mean((np.asarray(v)-y)**2)),8)})
    summ=[]
    for method in methods:
        R=[r for r in rows if r['model']==method]
        summ.append({'model':method,'n_examples':len(R),'complete_label_accuracy':np.mean([r['all_correct'] for r in R]),
                     'query_accuracy':np.mean([r['query_correct'] for r in R])/4,
                     'brier':np.mean([r['brier'] for r in R])})
    all_results[regime]={'protocol':{'n_train_graphs':len(tr_pairs),'n_eval_graphs':len(va_pairs),'training_report_draws_per_graph':8,
                                       'eval_report_draws_per_graph':8,'n_train_units':len(train),'n_eval_units':len(validation),
                                       'no_eval_truth_to_fit_Dawid_Skene':True,
                                       'DS_three_path_votes_are_correlated':True,
                                       'weighted_uses_per_example_reported_error_probabilities':True},
                         'dawid_skene':{'confusion':ds.confusion_.tolist(),'priors':ds.priors_.tolist(),'em_iterations':len(ds.history_)},
                         'summary':summ,'rows':rows}
    print(regime,[(v['model'],round(v['complete_label_accuracy'],3),round(v['brier'],3)) for v in summ],flush=True)
(OUT/'label_validation.json').write_text(json.dumps(all_results,indent=2))
