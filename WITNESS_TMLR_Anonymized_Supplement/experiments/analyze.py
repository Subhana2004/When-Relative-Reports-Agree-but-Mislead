import json,pathlib,csv,statistics
from collections import defaultdict
import numpy as np
BASE=pathlib.Path(__file__).resolve().parent
RESULT=BASE/'results'

def get_rows(filename):return json.loads((RESULT/filename).read_text())['rows']
pilot=get_rows('pilot.json')+get_rows('robust_soft.json')+get_rows('robust_soft_last.json')
rep=get_rows('replication.json')

def aggregate(rows):
    d=defaultdict(list)
    for r in rows:d[(r['regime'],r['arm'])].append(r)
    return {f'{k[0]}::{k[1]}':{'n_seeds':len(v),'mean_cqa':float(np.mean([x['cqa'] for x in v])),'cqa_by_seed':{str(x['seed']):x['cqa'] for x in v},'mean_train_seconds':float(np.mean([x['metadata']['training_seconds'] for x in v]))} for k,v in d.items()}

def paired(rows,reg,a,b):
    v={r['seed']:r['cqa'] for r in rows if r['regime']==reg and r['arm']==a}
    w={r['seed']:r['cqa'] for r in rows if r['regime']==reg and r['arm']==b}
    seeds=sorted(set(v)&set(w)); diff=np.array([v[s]-w[s] for s in seeds]);
    rng=np.random.default_rng(20261118)
    draws=rng.choice(diff,size=(20000,len(diff)),replace=True).mean(axis=1)
    return {'a':a,'b':b,'regime':reg,'seeds':seeds,'differences':diff.tolist(),'mean_pp':round(100*diff.mean(),3),'bootstrap_seed_95pct_pp':[round(float(100*x),3) for x in np.quantile(draws,[.025,.975])], 'positive_seeds':int(sum(diff>0)),'negative_seeds':int(sum(diff<0)),'tied_seeds':int(sum(diff==0))}

summary={'scope':'exploratory three-seed pilot followed by seven-new-seed independent graph-sampling replication','pilot':aggregate(pilot),'replication':aggregate(rep),'paired_pilot':[paired(pilot,'heavy','joint_independent','weighted_soft'),paired(pilot,'correlated','joint_robust','robust_soft')], 'paired_replication':[paired(rep,'heavy','joint_independent','weighted_soft'),paired(rep,'correlated','joint_robust','robust_soft')]}
(RESULT/'summary.json').write_text(json.dumps(summary,indent=2))
with (RESULT/'all_scores.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['stage','regime','seed','arm','cqa','query_accuracy','pos_recall','neg_recall','training_seconds','updates','graph_forwards'])
    for stage,rows in [('pilot',pilot),('replication',rep)]:
        for r in rows:w.writerow([stage,r['regime'],r['seed'],r['arm'],r['cqa'],r['accuracy'],r['positive_recall'],r['negative_recall'],r['metadata']['training_seconds'],r['metadata']['updates'],r['metadata']['graph_forwards']])
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8.8,3.5))
    labels=['Heavy: soft labels','Heavy: joint','Correlated: robust soft','Correlated: robust joint']
    vals=[summary['replication']['heavy::weighted_soft']['mean_cqa'],summary['replication']['heavy::joint_independent']['mean_cqa'],summary['replication']['correlated::robust_soft']['mean_cqa'],summary['replication']['correlated::joint_robust']['mean_cqa']]
    bars=ax.bar(labels,[100*v for v in vals],width=.65)
    ax.set_ylabel('Complete quadruple accuracy (%)');ax.set_ylim(0,105)
    for bar,v in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,100*v+1,f'{100*v:.1f}%',ha='center',fontsize=8)
    ax.tick_params(axis='x',labelsize=8);fig.tight_layout()
    fig.savefig(BASE/'paper'/'replication.pdf');fig.savefig(BASE/'paper'/'replication.png',dpi=180);plt.close(fig)
except Exception as e: print('plot error:',e)
print(json.dumps({'pilot':summary['paired_pilot'],'replication':summary['paired_replication']},indent=2))
