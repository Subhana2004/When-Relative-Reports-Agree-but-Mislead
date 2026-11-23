# Protocol for the next reviewer-critical experiments

The following is a **preregisterable proposal**, not additional measured evidence.

## 1. Training-size scaling

Compare 24 / 48 / 96 / 192 certified quadruples drawn independently for each training-data seed. Hold graph family, intervention balance, question balance, noise mechanism, architecture, and evaluation draws fixed. For each size report two budgets: (a) fixed epochs (more total updates for larger datasets), (b) fixed optimizer updates and graph forwards (controls compute). Use a repeated training example schedule for fixed-budget smaller sets; distinguish this from additional information. Use independently drawn noise realizations, and preserve per-example predictions and run-time.

## 2. Four separated sources of variability

- Training-data construction seed (new graph identities, not merely new noise).
- Noisy parity/report seed at a fixed training set.
- Neural initialization and ordering seed.
- Evaluation graph-draw seed, independent of all three above.

Report means and per-seed/draw performance, paired distributions, hierarchical/two-way bootstrap intervals with documented clusters, and model/generator failures. Do not treat multiple queries in one quadruple or test draw as independent replicates.

## 3. External topology and deployment relevance

Prioritize a license-compatible natural directed network with graph-level partitions. Verify query reachability with an independently implemented oracle; preserve nontrivial positive/negative balance and avoid overlap of isomorphic source/target neighborhoods. Apply a well-defined, valid degree-preserving intervention to the network **without** the paired-mirror construction. If interventions cannot change labels under that network's structure, label the benchmark inconclusive rather than manufacturing switches.

## 4. Baselines and costs

Differentiate exact reference algorithm, adapted mathematical estimator, and faithful author's implementation. Record package/commit, dependency versions, hyperparameters and defaults, calibration and access to error probabilities, whether report correlation was modeled, training examples, updates, wall-clock seconds, peak memory and preferably FLOPs. Do not assert identical compute merely because graph-forward counts match.

## 5. Decision gates

A general improvement claim needs consistency across multiple training and test graph draws and stronger published baselines, not a mean driven by one seed. A sound negative-results paper can proceed if the failure mode generalizes, assumptions are isolated, and all results and artifacts are verifiable. Any additional experiment must be run **before** adding a new result claim to the paper.
