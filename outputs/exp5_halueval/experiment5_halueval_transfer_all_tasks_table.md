# Experiment 5: HaluEval Zero-Shot Transfer

| Metric | AUROC RAGTruth | AUROC HaluEval | Drop |
| --- | --- | --- | --- |
| Cosine drift | 0.5384 | 0.5560 | -0.0176 |
| Mahalanobis distance | 0.6916 | 0.5743 | 0.1173 |
| Logit lens divergence | 0.6536 | 0.4556 | 0.1980 |
| PCA deviation | 0.6910 | 0.5759 | 0.1151 |
| Full composite | 0.7001 | 0.5835 | 0.1166 |

No HaluEval statistics were refit. RAGTruth train/val provides feature orientation, standardization, and logistic-composite weights; RAGTruth train provides the fitted Mahalanobis/PCA statistics used by the metric artifacts.

Primary HaluEval evaluation uses the paired binary tasks (dialogue, general, qa, summarization). The general split is excluded by default because the held-out subset has no hallucinated positives, so it is not a valid AUROC subtask.

Most brittle metric: Logit lens divergence (drop 0.1980).
