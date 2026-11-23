# External directed-network validation protocol (not yet executed)

**Status: code implemented and synthetic fixture checks written; natural SNAP dataset has not been downloaded or evaluated in this environment.** This is a feasibility protocol, not an external generalization result.

## Source

- Stanford Network Analysis Project, **Gnutella peer-to-peer network, August 8, 2002**: <https://snap.stanford.edu/data/p2p-Gnutella08.html>.
- The SNAP catalogue describes the graph as **directed**, with 6,301 nodes and 20,777 edges. Use the source page to retrieve `p2p-Gnutella08.txt.gz`; retain its SHA-256 in outputs, and follow the dataset's use/citation terms. No edge list is redistributed in this repository.
- Example background source (as listed on SNAP): Leskovec, Kleinberg, and Faloutsos, *Graph Evolution: Densification and Shrinking Diameters*, ACM TKDD 2007.

## Execute

```bash
# Download the named file manually from the official SNAP source page.
PYTHONPATH=.:WITNESS_TMLR_Anonymized_Supplement \
  python run_external_snap_protocol.py \
    --input /absolute/path/to/p2p-Gnutella08.txt.gz \
    --size 64 --draws 128 --seed 34789 \
    --output results/snap_external_feasibility.json
```

The script samples connected undirected neighborhoods from the real directed network, retains the directed edge orientations, performs a **valid degree-preserving directed edge swap**, and obtains four exact labels via NetworkX reachability. The two graphs share a real-network subgraph, **not paired mirrored copies**. All accepted samples preserve each named node's in/out degree; attempts without a valid swap are counted. Query source/targets are selected independently of the reachability outcome, preventing a silent positive-label cherry-pick.

## Mandatory go/no-go diagnostics

- Report graph/subgraph size and edge distributions, the number and rate of valid swaps, and the frequency with which query labels actually flip.
- Report the positive/negative class proportions and complete-four-answer pattern frequencies. If all samples are trivial (e.g., all reachable or unchanged), declare the dataset **inconclusive** rather than citing it as a generalization test.
- If the feasibility gate passes, freeze the query sampler and graphs, then train/evaluate matched WITNESS models with source-level graph partitions, independent random seeds and credible class-balanced controls. Do not train on the same node neighborhoods used for model selection.
- Network interventions are *synthetic interventions on a natural graph*, not real observed changes or causal interventions in the historical network.
- The original paper's complete-quadruple correctness is not automatically comparable because those quadruples are balanced by construction; report query-level class balance and accuracy alongside CQA.

**No natural-network numerical results are claimed here.** Running the script on the official downloaded dataset is a necessary next step, not an implied accomplishment.
