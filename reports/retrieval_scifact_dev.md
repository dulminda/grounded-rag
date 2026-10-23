## Retrieval: scifact / dev (300 queries)

| System | ndcg@10 | mrr@10 | recall@10 | recall@100 | Δ ndcg@10 vs `bm25` | p50 / p95 latency |
|---|---|---|---|---|---|---|
| `bm25` | 0.662 <sub>[0.616, 0.707]</sub> | 0.626 <sub>[0.576, 0.675]</sub> | 0.789 <sub>[0.741, 0.833]</sub> | 0.906 <sub>[0.869, 0.938]</sub> | — | 0.22 / 0.36 ms |
| `bm25_tuned` | 0.668 <sub>[0.622, 0.713]</sub> | 0.631 <sub>[0.582, 0.680]</sub> | 0.796 <sub>[0.750, 0.839]</sub> | 0.903 <sub>[0.865, 0.934]</sub> | +0.006 (p=0.138) | 0.20 / 0.26 ms |
| `bm25_rm3` | 0.657 <sub>[0.610, 0.704]</sub> | 0.618 <sub>[0.567, 0.669]</sub> | 0.784 <sub>[0.734, 0.829]</sub> | 0.905 <sub>[0.869, 0.938]</sub> | -0.005 (p=0.561) | 0.68 / 0.87 ms |
| `bm25_chunk100` | 0.657 <sub>[0.611, 0.703]</sub> | 0.624 <sub>[0.574, 0.673]</sub> | 0.780 <sub>[0.733, 0.824]</sub> | 0.885 <sub>[0.844, 0.919]</sub> | -0.005 (p=0.511) | 0.24 / 0.37 ms |
| `dense_lsa` | 0.485 <sub>[0.437, 0.530]</sub> | 0.444 <sub>[0.394, 0.492]</sub> | 0.643 <sub>[0.589, 0.695]</sub> | 0.863 <sub>[0.820, 0.903]</sub> | -0.177 (p<0.001) ✱ | 3.27 / 3.52 ms |
| `hybrid_lsa_rrf` | 0.578 <sub>[0.532, 0.623]</sub> | 0.533 <sub>[0.487, 0.581]</sub> | 0.740 <sub>[0.689, 0.789]</sub> | 0.931 <sub>[0.899, 0.959]</sub> | -0.084 (p<0.001) ✱ | 4.48 / 5.63 ms |
| `hybrid_lsa_tuned` | 0.674 <sub>[0.628, 0.719]</sub> | 0.639 <sub>[0.588, 0.686]</sub> | 0.797 <sub>[0.749, 0.841]</sub> | 0.919 <sub>[0.886, 0.948]</sub> | +0.012 (p=0.023) ✱ | 3.99 / 5.61 ms |

Means over queries with 95% bootstrap CIs. Δ uses a paired bootstrap test over queries; ✱ marks p < 0.05. Latency is retrieval only, single-threaded CPU.
