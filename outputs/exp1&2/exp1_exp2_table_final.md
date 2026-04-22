# Exp1/Exp2 Final Table

| Method | AUROC | F1 | Spearman | ECE |
| --- | --- | --- | --- | --- |
| Baseline: attention entropy | 0.666123 | 0.607042 | 0.284336 | 0.006601 |
| Baseline: logit confidence | 0.665016 | 0.608660 | 0.282441 | 0.013687 |
| Cosine drift | 0.538420 | 0.471874 | 0.065760 | 0.014677 |
| Mahalanobis score | 0.691631 | 0.615945 | 0.327996 | 0.036329 |
| PCA deviation | 0.690981 | 0.613517 | 0.326884 | 0.017991 |
| Logit lens divergence | 0.653551 | 0.601376 | 0.262819 | 0.017244 |
| CIE top-3 layers ([2, 1, 0]) | 0.538420 | 0.471874 | 0.065760 | 0.014677 |
| CIE full composite | 0.722404 | 0.608974 | 0.380667 | 0.046700 |

Notes:
- Baseline attention/logit-confidence are NA because compact Person1 artifacts in this run do not retain recoverable attention distributions / token-level logit confidences for all samples.
- If exact baseline JSON files exist (attention/logit confidence), they are used automatically.
- ECE for metric rows uses isotonic calibration fit on train split, evaluated on test split.
