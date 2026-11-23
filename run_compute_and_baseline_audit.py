"""Transparent information, computational-cost and published-estimator audit.

Timing measurements are machine-specific and non-equivalent operations: a 4-label
posterior inference call versus a spectral eigensolve versus a neural forward
or full EM fit. No FLOP-matched superiority is claimed.
"""
from __future__ import annotations
import json,time,platform,hashlib,sys
from pathlib import Path
import numpy as np,torch
from witness_core.robust_study import split, observe, posterior
from published_baselines import spectral_synchronization, train_ds, train_wrench_style, path_votes
from WITNESS_TMLR_Anonymized_Supplement.experiments.method import constraint_nll

PRIOR_ART={
 'spectral_synchronization': {'published_family':'Singer, Angular Synchronization by Eigenvectors and Semidefinite Programming, 2011', 'implementation':'custom Z2 spectral specialization','inputs':'six parity bits, six reliability tags, one anchor','relative_cost':'4x4 symmetric eigendecomposition; O(4^3) algebra','original_upstream_implementation_executed':False},
 'dawid_skene_regularized':{'published_family':'Dawid & Skene, JRSS-C 1979','implementation':'WITNESS-adapted confusion-matrix EM','inputs':'3 votes per nonanchor query, obtained from overlapping parity paths; no reliability tags','relative_cost':'iterative EM, approximately O(T * N * 3 * 2^2)','original_upstream_implementation_executed':False},
 'wrench_style_ds_unsmoothed':{'published_family':'WRENCH implementation of Dawid-Skene, Zhang et al., NeurIPS 2021','implementation':'local specialization of source, not original import','inputs':'same dependent path votes','relative_cost':'iterative EM, no per-report reliability metadata','original_upstream_implementation_executed':False,
 'upstream_blob_sha':'7c59cf863f2630c6d9cc89939acb0c5be049e2f9'},
 'posterior_soft': {'published_family':'exact Bayesian inference on binary constraints','implementation':'8-state exact enumeration under supplied independent noise model','inputs':'six bits, six reliability tags, one anchor','relative_cost':'2^(4-1)*6 likelihood evaluations plus normalization','original_upstream_implementation_executed':None},
 'joint':{'published_family':'marginal latent-label likelihood','implementation':'direct eight-state likelihood and autograd','inputs':'same reports, tags, anchor and graphs as matched posterior-soft','relative_cost':'above posterior arithmetic plus differentiable eight-state model loss','original_upstream_implementation_executed':None}
}

def microbenchmark(reps=500):
    training=split(314159,[10,12],['chain_scc','layered'],3)
    units=[observe(p,9001*100003+p.seed*31,'heavy') for p in training]
    results={};torch.set_num_threads(1)
    fake_p=torch.tensor([.33,.74,.55,.22],requires_grad=True)
    for name,call in (
      ('exact_posterior', lambda u: posterior(u,weighted=True,soft=True)),
      ('spectral_4x4', lambda u: spectral_synchronization(u,weighted=True)),
      ('joint_loss_and_autograd', lambda u: (lambda z:(z.backward(),z.item()))(constraint_nll(fake_p,u,'independent')))
    ):
        for _ in range(5):call(units[0])
        start=time.perf_counter();cpu=time.process_time()
        for i in range(reps):call(units[i%len(units)])
        results[name]={'calls':reps,'wall_seconds':time.perf_counter()-start,'process_cpu_seconds':time.process_time()-cpu}
        if name=='joint_loss_and_autograd':fake_p.grad=None
    for name,call in (('ds_regularized_fit',train_ds),('ds_wrench_style_fit',train_wrench_style)):
        t=time.perf_counter();p=time.process_time();model=call(units)
        results[name]={'calls':1,'wall_seconds':time.perf_counter()-t,'process_cpu_seconds':time.process_time()-p,
                       'estimated_class_prior':model.priors_.tolist()}
    return results

def main():
    out=Path('results/compute_baseline_audit.json');out.parent.mkdir(exist_ok=True)
    meta={'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,
          'numpy':np.__version__,'pytorch_threads':1,'n_training_quadruples':24,
          'caveats':'Timings of distinct operations NOT equal FLOP measurements, machine-dependent and not a model-end-to-end comparison. No natural-network benchmarks executed by this audit.',
          'methods':PRIOR_ART,'timings':microbenchmark()}
    out.write_text(json.dumps(meta,indent=2));print(json.dumps({k: {n:round(v,5) if isinstance(v,(int,float)) else v for n,v in x.items()} for k,x in meta['timings'].items()},indent=2))
if __name__=='__main__':main()
