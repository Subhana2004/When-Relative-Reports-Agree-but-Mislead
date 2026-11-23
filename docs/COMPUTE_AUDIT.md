# WITNESS computational-cost and information audit

This is a descriptive audit of **recorded neural optimization time** from the existing 96 validation training runs. It is **not** a complete cost-matched comparison of the training methods.

| Noise | Method | Training runs | Mean neural fit (s) | Median neural fit (s) | Updates / run | Graph forwards / run |
|---|---|---:|---:|---:|---:|---:|
| correlated | `dawid_skene_soft` | 12 | 2.34 | 2.27 | 144 | 576 |
| correlated | `joint_robust` | 12 | 2.35 | 2.33 | 144 | 576 |
| correlated | `robust_soft` | 12 | 2.37 | 2.32 | 144 | 576 |
| correlated | `spectral_weighted` | 12 | 2.35 | 2.27 | 144 | 576 |
| heavy | `dawid_skene_soft` | 12 | 2.33 | 2.32 | 144 | 576 |
| heavy | `joint_independent` | 12 | 2.43 | 2.31 | 144 | 576 |
| heavy | `spectral_weighted` | 12 | 2.40 | 2.36 | 144 | 576 |
| heavy | `weighted_soft` | 12 | 2.36 | 2.34 | 144 | 576 |


## What those times include

- Time inside the neural-network optimizer loop for a shared gated graph network, on the local benchmark environment used for the saved experiment.
- Six epochs over 24 graph quadruples; 144 optimizer updates and 576 graph forwards for each training run.
- Same training inputs and update counts across methods within one seed/noise condition. The cost of marginalizing eight latent assignments is additional arithmetic not captured by graph-forward counts.

## What they do **not** include

- Building the 24 training graph quadruples, calculating exact NetworkX reachability oracles, or generating noisy reports.
- Spectral synchronization/pseudo-target construction, Dawid–Skene EM fitting and soft-target generation, or Bayesian posterior-label preprocessing; the validated `run_neural_validation.py` starts its timing after most such work. Dawid–Skene's recorded EM iteration count is not equivalent to measured EM wall time.
- Held-out inference on five original and three alternate graph draws, bootstrapping, ZIP creation, installation, or transfer.
- FLOPs, carbon/energy accounting, end-to-end memory peak or cross-machine execution time.

**Interpretation:** neural fit times are similar on this tiny fixed collection. The data do not justify a claim that joint marginalization, spectral inference, or Dawid–Skene is end-to-end faster or equally expensive. A stronger cost study must include preprocessing and evaluation and provide peak RSS/memory and a machine/pinned-dependency specification.

## Information available to methods

| Arm | Exact endpoint supervision | Reported parity bits | Error probabilities | Correlated-event model |
|---|---|---|---|---|
| Bayesian soft (`weighted_soft`) | 1 anchor per quadruple | 6 | Yes | No |
| Joint (`joint_independent`) | 1 anchor per quadruple | 6 | Yes | No |
| Robust Bayesian soft (`robust_soft`) | 1 anchor per quadruple | 6 | Yes | Assumed 20% event mixture |
| Robust joint (`joint_robust`) | 1 anchor per quadruple | 6 | Yes | Same assumed 20% mixture |
| Spectral labels (`spectral_weighted`) | 1 anchor per quadruple | 6 | Yes | No explicit correlated-event model |
| Dawid–Skene labels (`dawid_skene_soft`) | 1 anchor per quadruple | Pair-derived votes | No per-report reliability tags | Conditional independence of votes (violated by shared reports) |

The source algorithms were **adapted** to WITNESS parity reports, not compared through their original authors' full benchmark implementations. The paper must not conflate these categories. Oracle endpoint experiments in earlier studies use more information and are only learnability controls.
