# Experiment 5: HaluEval Zero-Shot Transfer

| Metric | AUROC RAGTruth | AUROC HaluEval | Drop |
| --- | --- | --- | --- |
| Cosine drift | 0.5384 | 0.6822 | -0.1438 |
| Mahalanobis distance | 0.6916 | 0.6685 | 0.0231 |
| Logit lens divergence | 0.6536 | 0.3972 | 0.2563 |
| PCA deviation | 0.6910 | 0.6758 | 0.0152 |
| Full composite | 0.7001 | 0.6663 | 0.0338 |

No HaluEval statistics were refit. RAGTruth train/val provides feature orientation, standardization, and logistic-composite weights; RAGTruth train provides the fitted Mahalanobis/PCA statistics used by the metric artifacts.

Primary HaluEval evaluation uses the paired binary tasks (dialogue, qa, summarization). The general split is excluded by default because the held-out subset has no hallucinated positives, so it is not a valid AUROC subtask.

Most brittle metric: Logit lens divergence (drop 0.2563).
