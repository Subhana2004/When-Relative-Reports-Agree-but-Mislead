"""WRENCH DawidSkene E/M update-equation reproduction on WITNESS path votes."""
import json,sys,random,time
from pathlib import Path
import numpy as np
H=Path(__file__).parent;sys.path[:0]=[str(H),str(H/'WITNESS_TMLR_Anonymized_Supplement')]
from witness_core.robust_study import split,observe
from published_baselines import train_wrench_style
base=json.loads((H/'results'/'label_validation.json').read_text());out={}
for regime in ('heavy','correlated'):
    tr_pairs=split(99007,[10,14,22,34],['chain_scc','layered','branch','dense_scc'],3)
    va_pairs=split(323089,[10,14,22,34],['chain_scc','layered','branch','dense_scc'],4)
    tr=[observe(p,2001103+157*i+1109*k,regime) for i,p in enumerate(tr_pairs) for k in range(8)]
    val=[(i,observe(p,3100103+983*i+173*k,regime)) for i,p in enumerate(va_pairs) for k in range(8)]
    t=time.perf_counter();m=train_wrench_style(tr)
    rows=[]
    for i,u in val:
        target=np.asarray([y for _,_,y in u.examples()]);p=m.predict_unit(u)
        rows.append({'pair_index':i,'all_correct':int(np.array_equal((p>=.5).astype(int),target)),
                     'query_correct':int(((p>=.5).astype(int)==target).sum()),'brier':float(np.mean((p-target)**2))})
    out[regime]={'source_commit':'7c59cf863f2630c6d9cc89939acb0c5be049e2f9','n_train_units':len(tr),'n_test_units':len(val),
       'method':'WRENCH DawidSkene E/M equation-level reimplementation with task-adapted correlated 3-path votes; not original package execution',
       'fit_seconds':round(time.perf_counter()-t,3),'em_iterations':m.iterations_,
       'complete_label_accuracy':float(np.mean([v['all_correct'] for v in rows])),
       'query_accuracy':float(np.mean([v['query_correct'] for v in rows])/4),
       'brier':float(np.mean([v['brier'] for v in rows])),
       'confusion':m.confusion_.tolist(),'priors':m.priors_.tolist(),'rows':rows}
    print(regime,'iterations',m.iterations_,'cqa',out[regime]['complete_label_accuracy'],'brier',out[regime]['brier'],'time',round(time.perf_counter()-t,2),flush=True)
(H/'results'/'wrench_style_validation.json').write_text(json.dumps(out,indent=2))
