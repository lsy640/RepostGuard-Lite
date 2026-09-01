# V3.2.1 fixed-1500 evaluation

AUROC and AP are threshold-independent. BA below uses the frozen V3.2.1 threshold.

| Scope | AUROC | AP | BA | AIGI recall | Real specificity |
|---|---:|---:|---:|---:|---:|
| overall | 0.654344 | 0.676611 | 0.578000 | 0.741333 | 0.414667 |
| sd14_exact_seen | 0.911288 | 0.717123 | 0.705333 | 0.996000 | 0.414667 |
| dfgan_hard | 0.639973 | 0.218085 | 0.591333 | 0.768000 | 0.414667 |
| galip_hard | 0.628123 | 0.213629 | 0.587333 | 0.760000 | 0.414667 |
| hourglass_hard | 0.417696 | 0.230428 | 0.439333 | 0.464000 | 0.414667 |

Generator slices reuse the same 750-image Real reference pool.
