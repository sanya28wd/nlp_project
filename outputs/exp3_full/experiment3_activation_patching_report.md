# Experiment 3: Causal Intervention - Activation Patching

Patch hidden states in both directions: faithful -> hallucinated and hallucinated -> faithful. Report CIE by component type. Minimum 50 patching experiments. Discuss agreement or disagreement with ReDeEP's Knowledge FFN / Copying Head finding.

| Component | CIE faithful->halluc. | CIE halluc.->faithful | Critical? |
| --- | --- | --- | --- |
| Early attn heads (1-25%) | 0.000016 | -0.000017 | No |
| Mid FFN layers (26-75%) | 0.002794 | 0.003093 | Yes |
| Late FFN layers (76-100%) | 0.004787 | 0.005297 | Yes |
| Copying heads (last 25%) | 0.000701 | 0.000824 | Yes |

Significance details:

| Component | p faithful->halluc. | p halluc.->faithful |
| --- | --- | --- |
| Early attn heads (1-25%) | 0.151924 | 1.000000 |
| Mid FFN layers (26-75%) | 0.012494 | 0.000500 |
| Late FFN layers (76-100%) | 0.000500 | 0.000500 |
| Copying heads (last 25%) | 0.000500 | 0.002999 |

Checklist against rubric:

- Both patching directions reported: Yes
- Minimum 50 experiments: Yes (`240` total)
- CIE significant in at least 2 components (`p < 0.05`): Yes (`mid FFN`, `late FFN`, `copying heads`)
- ReDeEP comparison written: Yes

ReDeEP comparison:

Our results agree with ReDeEP's claim that late FFN layers are critical. In our activation patching experiments, late FFN layers produce the largest positive CIE in both directions and are strongly significant. We also find additional causal importance in mid FFN layers and copying heads, suggesting that the mechanism is not confined to a single component family, but late FFN remains the strongest and most consistent component.
