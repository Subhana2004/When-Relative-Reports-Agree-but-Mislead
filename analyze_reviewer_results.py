"""Grouped, paired analysis of WITNESS reviewer-response experiments.

Never treat repeated held-out graph draws as separate training runs.
Bootstrap intervals resample training-pool, report and initialization factors
independently (multiway crossed factor bootstrap); with 4/2/2 factors, these
intervals are exploratory and cannot provide confirmatory significance.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

def factorial():
  x=json.loads(Path('results/factorial_targeted.json').read_text());r=x['rows']
  assert len(r)==64
  out={'protocol':x['protocol'],'per_size':{},'factor_dispersion':{}}
  pools=sorted(set(z['train_pool_seed'] for z in r));reports=sorted(set(z['report_seed'] for z in r));models=sorted(set(z['model_seed'] for z in r));
  assert len(pools)==4 and len(reports)==len(models)==2
  for n in (24,192):
    dat={}
    for kind in ('original','arborescence'):
      scores={}
      for arm in ('weighted_soft','joint_independent'):
        X=np.array([[[next(z for z in r if z['train_pool_seed']==p and z['report_seed']==q and z['model_seed']==m and z['size']==n and z['arm']==arm)['draws'][0 if kind=='original' else 1]['cqa'] for m in models] for q in reports] for p in pools])
        scores[arm]=X
      D=scores['joint_independent']-scores['weighted_soft']
      rng=np.random.default_rng(5017+n+(0 if kind=='original' else 1))
      boot=[]
      for _ in range(10000):
        a=rng.integers(0,4,4);b=rng.integers(0,2,2);c=rng.integers(0,2,2)
        boot.append(D[np.ix_(a,b,c)].mean())
      dat[kind]={'weighted_soft_cqa':float(scores['weighted_soft'].mean()),
                 'joint_cqa':float(scores['joint_independent'].mean()),
                 'paired_delta':float(D.mean()),
                 'exploratory_3factor_percentile_interval':[float(z) for z in np.quantile(boot,[.025,.975])],
                 'factorial_cells':int(D.size),
                 'wins':int(np.sum(D>0)),'ties':int(np.sum(D==0)),'losses':int(np.sum(D<0)),
                 'zero_CQA_soft':int(np.sum(scores['weighted_soft']==0)),
                 'zero_CQA_joint':int(np.sum(scores['joint_independent']==0)),
                 'difference_by_pool':[float(z) for z in D.mean(axis=(1,2))],
                 'difference_by_report_seed':[float(z) for z in D.mean(axis=(0,2))],
                 'difference_by_model_seed':[float(z) for z in D.mean(axis=(0,1))]}
    out['per_size'][str(n)]=dat
  return out

def nonmirrored():
  x=json.loads(Path('results/nonmirrored_neural.json').read_text());rs=x['rows']
  assert len(rs)==48
  out={'protocol':x['protocol'],'comparisons':{}}
  for regime,soft,joint in [('heavy','weighted_soft','joint_independent'),('correlated','robust_soft','joint_robust')]:
    ss=sorted(set(z['seed'] for z in rs if z['regime']==regime));assert len(ss)==12
    get=lambda arm:np.array([[d['cqa'] for d in next(z for z in rs if z['seed']==s and z['regime']==regime and z['arm']==arm)['draws']] for s in ss])
    A=get(soft);B=get(joint);D=B-A
    rng=np.random.default_rng(1729);bs=[]
    for _ in range(10000):bs.append(D[np.ix_(rng.integers(0,12,12),rng.integers(0,4,4))].mean())
    out['comparisons'][regime]={'soft_cqa':float(A.mean()),'joint_cqa':float(B.mean()),
       'delta':float(D.mean()),'exploratory_two_way_interval':[float(z) for z in np.quantile(bs,[.025,.975])],
       'wins':int(np.sum(D.mean(1)>0)),'ties':int(np.sum(D.mean(1)==0)),'losses':int(np.sum(D.mean(1)<0)),
       'mean_seed_differences':[float(z) for z in D.mean(1)],'n_neural_seeds':12,'n_graph_draws':4}
  return out

def main():
 o={'factorial':factorial(),'nonmirrored':nonmirrored(),
    'cautions':['Earlier exploratory size sweep uses coupled report/model seeds and 8 test draws; do not pool its estimates with the factorial design.','Gnutella network feasibility result does not include a neural model.','Bootstrap units are seed factors and graph draws, not independent sample quadruples.','Post-hoc studies lack preregistered hypothesis testing.']}
 Path('results/reviewer_resolution_summary.json').write_text(json.dumps(o,indent=2))
 for n,d in o['factorial']['per_size'].items():
  for k,v in d.items():print('factorial',n,k,'soft',round(v['weighted_soft_cqa']*100,2),'joint',round(v['joint_cqa']*100,2),'delta',round(v['paired_delta']*100,2),'CI',np.round(np.array(v['exploratory_3factor_percentile_interval'])*100,2),'cellw/t/l',v['wins'],v['ties'],v['losses'])
 for k,v in o['nonmirrored']['comparisons'].items():print('nonmirror',k,'soft',round(v['soft_cqa']*100,2),'joint',round(v['joint_cqa']*100,2),'delta',round(v['delta']*100,2),'CI',np.round(np.array(v['exploratory_two_way_interval'])*100,2))
if __name__=='__main__':main()
