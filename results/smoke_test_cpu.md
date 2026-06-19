# SASRec CE smoke test on MovieLens-1M

Date: 2026-06-19  
Hardware: CPU  
Epochs: 1  
Loss: full cross-entropy  
Evaluation: full-sort ranking over the item catalog  
Seed: 42

## Test metrics

| Metric |     @5 |    @10 |    @20 |
| ------ | -----: | -----: | -----: |
| Recall | 0.0944 | 0.1629 | 0.2575 |
| NDCG   | 0.0590 | 0.0809 | 0.1047 |
| MRR    | 0.0474 | 0.0563 | 0.0628 |

This run is a technical smoke test, not a final experiment.
