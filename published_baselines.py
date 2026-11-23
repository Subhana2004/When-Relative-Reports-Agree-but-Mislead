"""Published-algorithm baselines on WITNESS four-query parity observations.

Two algorithms faithfully instantiate published mathematical estimators, but the
conversion of relative reports into annotator votes is our task adaptation:
  * Singer (2011): top eigenvector of the signed relative-measurement matrix.
  * Dawid and Skene (1979): latent class with independent annotator confusion
    matrices, fit via EM; pairwise reports are mapped to three path votes.

This is not a claim to run original author code or reproduce authors' benchmarks.
"""
from __future__ import annotations
import sys, pathlib, numpy as np
from scipy.special import logsumexp
ROOT=pathlib.Path(__file__).resolve().parent/'WITNESS_TMLR_Anonymized_Supplement'
sys.path.insert(0,str(ROOT))
from witness_core.robust_study import RELATIONS, ASSIGNMENTS, XOR_TABLE, posterior, infer

R2I={edge:i for i,edge in enumerate(RELATIONS)}

def spectral_synchronization(unit, weighted=True):
    """Singer eigenvector angular synchronization restricted to Z2 signs.

    Measurement matrix H_ij=w_ij * sign_ij, sign_ij=(-1)^r_ij.
    Leading algebraic eigenvector estimates vertex signs up to a global sign;
    the observed anchor fixes global sign. The 'weighted' version sets positive
    confidence weights to log ((1-eps)/eps) (a documented adaptation).
    """
    eps=np.clip(np.asarray(unit.error_probs,dtype=float),.001,.499)
    w=np.log((1-eps)/eps) if weighted else np.ones(6,dtype=float)
    H=np.zeros((4,4),dtype=float)
    for k,(i,j) in enumerate(RELATIONS):
        H[i,j]=H[j,i]=w[k]*(1-2*int(unit.observed_bits[k]))
    vals,vecs=np.linalg.eigh(H)
    v=vecs[:,-1]
    if abs(v[0])<1e-8: # deterministic gauge fallback
        v[0]=1e-8
    rel=np.where(v*v[0] >= 0,0,1)
    return np.bitwise_xor(rel,int(unit.anchor_label)).astype(np.int8)

def path_votes(unit):
    """Map six parity observations into three redundant LF votes per nonanchor.
    Votes (not independent): direct 0->j and two length-two paths 0->k->j.
    """
    r={e:int(unit.observed_bits[i]) for i,e in enumerate(RELATIONS)}
    def edge(a,b):return r[(min(a,b),max(a,b))]
    out=[]
    for j in (1,2,3):
        rivals=[i for i in (1,2,3) if i!=j]
        out.append([int(unit.anchor_label)^edge(0,j)]+[
            int(unit.anchor_label)^edge(0,k)^edge(k,j) for k in rivals])
    return np.asarray(out,dtype=np.int8)

class DawidSkeneEM:
    """Standard two-class Dawid--Skene confusion-matrix EM, 3 LFs.

    Explicitly assumes conditional independence across annotators, which is
    violated by path-vote construction. Thus a controlled misspecification test.
    """
    def __init__(self, maxiter=200, prior_strength=.01):
        self.maxiter=maxiter;self.prior_strength=prior_strength

    def fit(self, votes):
        V=np.asarray(votes,dtype=np.int64)
        assert V.ndim==2 and V.shape[1]==3
        n,m=V.shape; assert n>=3
        # Use symmetrical starts, no ground-truth leakage.
        self.confusion_=np.empty((m,2,2),float)
        for k in range(m):self.confusion_[k]=[[.70,.30],[.30,.70]]
        self.priors_=np.array([.5,.5])
        self.history_=[]
        for _ in range(self.maxiter):
            LL=np.broadcast_to(np.log(self.priors_)[None,:],(n,2)).copy()
            for k in range(m):LL+=np.log(self.confusion_[k][:,V[:,k]].T.clip(1e-12))
            z=np.exp(LL-logsumexp(LL,axis=1,keepdims=True))
            ll=float(logsumexp(LL,axis=1).sum());self.history_.append(ll)
            priors=z.mean(0)
            conf=np.zeros((m,2,2),float)
            for k in range(m):
                for truth in (0,1):
                    counts=np.bincount(V[:,k],weights=z[:,truth],minlength=2)
                    conf[k,truth,:]=(counts+self.prior_strength)/(counts.sum()+2*self.prior_strength)
            if np.max(np.abs(conf-self.confusion_))<1e-9 and np.max(np.abs(priors-self.priors_))<1e-9:
                self.confusion_=conf;self.priors_=priors;break
            self.confusion_=conf;self.priors_=priors
        return self

    def predict_proba(self,votes):
        V=np.asarray(votes,dtype=np.int64);n,m=V.shape
        LL=np.broadcast_to(np.log(self.priors_.clip(1e-10))[None,:],(n,2)).copy()
        for k in range(m):LL+=np.log(self.confusion_[k][:,V[:,k]].T.clip(1e-12))
        return np.exp(LL-logsumexp(LL,axis=1,keepdims=True))

    def predict_unit(self,unit):
        probs=self.predict_proba(path_votes(unit))[:,1]
        return np.concatenate(([float(unit.anchor_label)],probs))


def train_ds(units):
    all_votes=np.concatenate([path_votes(u) for u in units],axis=0)
    return DawidSkeneEM().fit(all_votes)


def seed_parity_truth_tests():
    class Fake:pass
    for labels in ((0,0,1,0),(1,0,1,1),(1,1,0,0),(0,1,1,1)):
        u=Fake();u.anchor_label=labels[0];u.error_probs=[.01]*6
        u.observed_bits=np.asarray([labels[i]^labels[j] for i,j in RELATIONS])
        assert np.array_equal(spectral_synchronization(u),labels)
        assert np.all(np.array(path_votes(u))==np.array(labels[1:])[:,None])
        assert np.array_equal(infer(u,'weighted_map')[0].astype(int),labels)
    rng=np.random.default_rng(12)
    V=rng.integers(0,2,(100,3))
    model=DawidSkeneEM().fit(V)
    P=model.predict_proba(V)
    assert np.allclose(P.sum(1),1,atol=1e-12)
    assert np.isfinite(P).all()
    assert np.all(np.diff(model.history_)>=-1e-4), model.history_
    return True

if __name__=='__main__':
    print('passed',seed_parity_truth_tests())

class WrenchStyleDawidSkene:
    """Reimplementation of WRENCH's DawidSkene EM updates (main branch).

    Reference source Git SHA 7c59cf863f2630c6d9cc89939acb0c5be049e2f9.
    No-abstain, 2-class, 3-labeler specialization of WRENCH's count-vote
    initialization, unsmoothed M step, E step, L1 tolerance=1e-5. It is NOT
    imported from the WRENCH package or a reproduction of its published tasks.
    """
    def __init__(self,maxiter=10000,tolerance=1e-5):
        self.maxiter=maxiter;self.tolerance=tolerance
    def fit(self,votes):
        V=np.asarray(votes,dtype=int);n,m=V.shape
        assert m==3 and V.min()>=0 and V.max()<=1
        self.V_train_shape=V.shape
        Y=np.stack([(V==0).sum(1),(V==1).sum(1)],axis=1).astype(float)/m
        prior_prev=None;conf_prev=None
        for iteration in range(self.maxiter):
            prior=Y.mean(0)
            conf=np.zeros((m,2,2))
            for k in range(m):
                for cls in (0,1):
                    num=np.bincount(V[:,k],weights=Y[:,cls],minlength=2)
                    conf[k,cls]=num/num.sum() if num.sum()>0 else [.5,.5]
            logp=np.log(prior.clip(1e-300))[None,:]+np.zeros((n,2))
            for k in range(m):logp+=np.log(conf[k][:,V[:,k]].T.clip(1e-300))
            Y=np.exp(logp-logsumexp(logp,axis=1,keepdims=True))
            if prior_prev is not None:
                if np.abs(prior-prior_prev).sum()<self.tolerance and np.abs(conf-conf_prev).sum()<self.tolerance:
                    break
            prior_prev=prior.copy();conf_prev=conf.copy()
        self.priors_=prior;self.confusion_=conf;self.iterations_=iteration+1
        return self
    def predict_proba(self,votes):
        V=np.asarray(votes,dtype=int);n,m=V.shape
        logp=np.log(self.priors_.clip(1e-300))[None,:]+np.zeros((n,2))
        for k in range(m):logp+=np.log(self.confusion_[k][:,V[:,k]].T.clip(1e-300))
        return np.exp(logp-logsumexp(logp,axis=1,keepdims=True))
    def predict_unit(self,unit):
        return np.concatenate(([float(unit.anchor_label)],self.predict_proba(path_votes(unit))[:,1]))


def train_wrench_style(units):
    return WrenchStyleDawidSkene().fit(np.concatenate([path_votes(u) for u in units]))
