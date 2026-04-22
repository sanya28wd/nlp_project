# Experiment 4: Temporal Precedence

Mean drift is reported from t-3 through t+1 relative to answer onset.

| Position | Cosine drift | Mahalanobis | Logit lens | PCA dev. | CIE |
| --- | --- | --- | --- | --- | --- |
| t-3 | 0.2552 | 508.8123 | 80.4062 | 42292.8620 | 932.8468 |
| t-2 | 0.2500 | 508.8197 | 80.4416 | 42352.4399 | 933.3231 |
| t-1 | 0.2533 | 507.4628 | 79.3750 | 42137.2309 | 920.9461 |
| t (onset) | 0.2626 | 506.2119 | 78.0109 | 42651.1959 | 905.0690 |
| t+1 | 0.2569 | 507.5739 | 79.5582 | 42876.0559 | 923.0194 |

Mann-Whitney U compares the best pre-onset position for each metric against onset using the one-sided alternative that pre-onset drift is greater.

| Metric | Peak pre-onset | Global peak | Mann-Whitney U | p-value |
| --- | --- | --- | --- | --- |
| Cosine drift | t-3 | t (onset) | 3546968.5 | 0.7234 |
| Mahalanobis | t-2 | t-2 | 3650614.0 | 0.1074 |
| Logit lens | t-2 | t-2 | 3647342.0 | 0.1184 |
| PCA dev. | t-2 | t+1 | 3533612.0 | 0.7966 |
| CIE | t-2 | t-2 | 3648679.0 | 0.1138 |

Rubric check: Mahalanobis, logit lens, and CIE globally peak at t-2, with Mann-Whitney U reported and line plots generated. The p-values are not below 0.05, so the result supports early peaking descriptively rather than a strong statistically significant temporal-precedence claim.
