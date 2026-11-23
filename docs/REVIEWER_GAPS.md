# WITNESS reviewer-gap register — measured evidence versus open work

| Reviewer concern | Verified work | Remaining scientific gate |
|---|---|---|
| Small 24-quadruple training set | **Exploratory** fixed-192-update 24/48/96/192 scaling pilot, two independently generated training sets, three coupled model/report seeds, 48 training runs, eight old/alternate test draws per run; full raw per-quadruple correctness in repo | Expand to independently crossed graph/data/noise/initialization seeds and additional neural architectures, plus fixed-epoch budget; no reliable scaling law yet |
| Substantial statistical uncertainty | Initial 12-seed/8-draw validation preserved; follow-up has six matched trained model/data-seed cells per size with sharply non-monotonic failures | Several additional independent training-data, report-noise, neural, and test-graph draws; retain hierarchical pairing and report uncertainty without treating each quadruple as independent |
| Real-world applicability | Synthetic non-mirror, degree-preserving directed-swap **harness** and tests; SNAP Gnutella08 primary source identified | **Not run on real SNAP data**; obtain source graph and verify nontrivial label changes, class balance, leakage-free partitions and model generalization |
| Strong baselines and compute | Adapted Singer spectral and Dawid–Skene EM implementations; WRENCH-source equation check; separate audit of measured neural-only training times and information asymmetries | Faithfully run original author code, investigate mismatched output/inputs, measure preprocessing/evaluation time, peak memory, FLOPs where feasible |
| Reproducibility | Original archive and SHA-256 pinned; original 55 tests and browseable-source 61 tests pass locally; GitHub Actions verifies archive/summary; new size raw data + deterministic analyzer added | Confirm updated CI on newest commit; invite independent third-party reproduction and ensure anonymous TMLR supplement contains exact disclosed results |

## Limits governing claims

The follow-up scaling experiment was started **after** reviewer-style feedback, so it is post-hoc exploratory evidence. The noise and neural seeds are coupled in that pilot. The new external-network script has only been checked against synthetic fixtures, **not** executed on the named real dataset. The mirrored-copy graph families remain the source of all measured neural generalization figures.

There is no verified new state-of-the-art method, natural-network success, or statistically robust advantage of joint marginalization. The existing 11-page TMLR-layout manuscript is a research draft, not a cleared submission. Because this repository reveals account/commit identity, do not cite the public URL inside the anonymized review submission.
