# WITNESS: reproducible reviewer-response studies

The paper is a method/diagnostics study, **not** an accepted or submitted TMLR publication. Every result is exploratory and must remain explicitly identified as synthetic or natural graph data.

## External directed-network check


**Verified pinned mirror Git blob SHA:** `f6d6ea2941ef6c892abe5be37a6b48c403d8c0f3`.

The extraction preserves edge directions. Sampling uses uniform random source and two target vertices within each sampled 48-node neighborhood. One valid degree-preserving directed two-edge swap is applied, without outcome-conditioned filtering. A deterministic LCG produces exactly 512 accepted interventions in the inspected mirror; only 17 change either queried answer, and 411 result in four unreachable labels. These are feasibility/difficulty diagnostics, **not neural generalization results**. Source byte identity to the official SNAP archive was not established during this local audit. The downloadable dataset is intentionally not redistributed in the supplement; use the official SNAP link in the runbook. The GitHub Action preferentially downloads the official archive and records its SHA-256; it falls back to the pinned mirror only if unavailable.

If the executed workflow retrieves an upstream file with different edge ordering, the exact deterministic counts can differ even if nodes and edges are the same. To reproduce the numbers in the paper, use the named mirror at the pinned file blob revision.

Run:

```bash
node snap_external_audit.js /absolute/path/to/p2p-Gnutella08.txt > results/snap_external_result.json
```

## Additional *synthetic* topology test

`nonmirrored_graphs.py` generates single asymmetric three-zone networks (44, 68 and 96 nodes), and `run_nonmirrored_validation.py` trains on the ORIGINAL graph family, then evaluates without fine-tuning on the new intervention construction. It uses a single 2-switch on one network, **not mirrored copies**. Query locations are deliberately selected from structural roles to obtain balanced quadruples; this is distinct from the unconditioned real-network test. There is no evidence of neural generalization to natural graphs from this test.

## Factorial size experiment

`run_factorial_targeted.py` crosses four independent graph-pool seeds, two report noise seeds and two neural initialization seeds at 24 and 192 quadruples, and uses a matched fixed-update training budget. Report seeds are independent of neural initialization by design. Each model is scored on one original-generator draw and one arborescence draw; these are REUSED across all model runs and are not independent training replications.

The pre-existing four-size experiment (24, 48, 96, 192) is separate and uses fewer training-pool seeds with coupled model/report noise. Do not pool observations across the two protocols as if they were identically randomized.

## Theoretical diagnostic

`gauge_practical_audit.py` flips all anchor-incident reports within a matched observation while retaining the same independent report errors. Cycle syndromes stay unchanged, yet downstream posterior label reconstruction is severely degraded. Adding a second verified label is an additional-supervision upper control; it does not provide an equal-budget method improvement. Identical reconstruction rates across graph families reflect the parity observation model and fixed label patterns, **not independent proof of graph-dependent effects**.

## Baselines and cost

`run_compute_and_baseline_audit.py` records exact information supplied to all methods, upstream provenance, the fact that WRENCH/WEASEL packages themselves were not executed locally, CPU/elapsed times for microbenchmarks, and why equal graph forwards do not imply equal FLOPs. Claims about faithful published-system reproduction are explicitly limited.

## Reviewer anonymity

The public GitHub repository has an identifiable owner and commit history; **do not link it in the anonymous TMLR submission**. Instead, upload the anonymous ZIP supplied with the paper through OpenReview. The PDF may describe the supplementary ZIP generically without identifying any public repository. The anonymous ZIP intentionally excludes files mentioning account names, emails, absolute local paths, or personal identity.
