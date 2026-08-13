# Mechanistic Hallucination Detection in Language Models

This repository contains an NLP research project focused on detecting and explaining hallucinations in language model outputs. Instead of only treating hallucination detection as a black-box classification task, the project studies internal model behavior through representation drift, logit-lens divergence, causal intervention effects, activation patching, and component-level analysis across attention and feed-forward layers.

The work is built around benchmark-style hallucination datasets such as RAGTruth and HaluEval. It includes a reproducible artifact-generation pipeline, metric computation scripts, experiment reports, visualizations, and final result tables.

## Project Goal

Large language models can produce fluent answers that are not supported by the provided context. This project investigates whether hallucinated generations leave measurable traces inside model activations before or during answer generation.

The core research questions are:

- Can internal representation metrics distinguish faithful and hallucinated answers?
- Which model components carry the strongest hallucination-related signal?
- Do hallucination signals appear before answer onset or only after the model starts generating unsupported text?
- How well do mechanistic metrics transfer across hallucination benchmarks?
- How close are lightweight internal-state metrics to stronger published hallucination detectors?

## What This Project Does

The repository implements an end-to-end experimental workflow:

1. Load and normalize raw hallucination datasets.
2. Split samples into train, validation, and test manifests.
3. Run language-model forward passes and save hidden-state/logit artifacts.
4. Compute internal hallucination metrics from saved artifacts.
5. Fit train-split statistics for representation-based scores.
6. Evaluate metrics with AUROC, F1, Spearman correlation, and calibration error.
7. Run mechanistic experiments such as activation patching and temporal analysis.
8. Generate result tables, plots, and written experiment reports.

## Main Techniques

- **Attention entropy baseline**: Measures uncertainty through attention distribution entropy.
- **Logit confidence baseline**: Uses top-token confidence as a simple uncertainty signal.
- **Cosine drift**: Compares context and answer representations.
- **Mahalanobis distance**: Scores answer representations against fitted faithful-answer statistics.
- **PCA deviation**: Measures how far answer activations move from a learned representation subspace.
- **Logit-lens divergence**: Tracks disagreement between intermediate-layer predictions and final logits.
- **Causal intervention effect**: Estimates how much selected internal states affect hallucination-related behavior.
- **Activation patching**: Tests whether replacing hidden states between faithful and hallucinated examples changes model behavior.
- **FFN vs attention decomposition**: Separates feed-forward and attention contributions by layer range.
- **Temporal precedence analysis**: Checks whether hallucination signals peak before answer onset.

## Key Results

The strongest RAGTruth result in the final table is a **CIE full composite AUROC of 0.7224**, outperforming the attention entropy baseline at **0.6661 AUROC**.

From the final Exp1/Exp2 table:

| Method | AUROC | F1 | Spearman | ECE |
| --- | ---: | ---: | ---: | ---: |
| Attention entropy baseline | 0.6661 | 0.6070 | 0.2843 | 0.0066 |
| Logit confidence baseline | 0.6650 | 0.6087 | 0.2824 | 0.0137 |
| Mahalanobis score | 0.6916 | 0.6159 | 0.3280 | 0.0363 |
| PCA deviation | 0.6910 | 0.6135 | 0.3269 | 0.0180 |
| Logit-lens divergence | 0.6536 | 0.6014 | 0.2628 | 0.0172 |
| CIE full composite | 0.7224 | 0.6090 | 0.3807 | 0.0467 |

Activation patching found that **mid FFN layers, late FFN layers, and copying heads** were significant in both patching directions. Late FFN layers had the strongest causal intervention effect, which supports the idea that hallucination-related behavior is strongly represented in feed-forward components rather than only attention heads.

The temporal analysis found descriptive early peaks for Mahalanobis, logit-lens, and CIE metrics around `t-2`, although these peaks were not statistically significant at `p < 0.05`. This suggests promising but limited evidence for pre-onset hallucination signals.

## Experiment Overview

### Experiments 1 and 2: Metric Evaluation

These experiments compare simple uncertainty baselines against representation-based and causal metrics. The final result table reports AUROC, F1, Spearman correlation, and expected calibration error on the held-out test split.

### Experiment 3: Activation Patching

This experiment patches hidden states in both directions:

- faithful sample to hallucinated sample
- hallucinated sample to faithful sample

It reports causal intervention effects by component family. The experiment includes 240 total patching runs and identifies mid FFN, late FFN, and copying-head components as significant.

### Experiment 4: Temporal Precedence

This experiment measures hallucination signals around answer onset from `t-3` through `t+1`. It tests whether internal drift appears before unsupported generation begins.

### Experiment 5: HaluEval Transfer

This experiment evaluates whether metrics trained or selected on one hallucination setting transfer to HaluEval tasks such as QA, dialogue, summarization, and general generation.

### Experiments 6-8: Component Localization, Failure Cases, and SOTA Gap

These experiments decompose attention and FFN drift, inspect qualitative failure cases, and compare the current approach against stronger published hallucination detectors such as ReDeEP and LUMINA.

## Repository Structure

```text
.
├── data/                         # Raw and converted benchmark data
├── outputs/                      # Experiment tables, plots, summaries, and reports
├── person2/                      # Saved metric artifacts and fitted statistics
├── scripts/                      # Experiment runners and analysis scripts
├── src/nlp_track_b/person1/      # Data loading, formatting, model forward pipeline
├── src/nlp_track_b/person2/      # Metric computation and artifact utilities
├── tests/                        # Existing pipeline tests
├── pyproject.toml                # Python project metadata and dependencies
└── requirements.txt              # Dependency list
```

## Example Commands

Generate model artifacts from a JSONL dataset:

```bash
python scripts/exp1_generate_artifacts.py \
  --dataset data/ragtruth/raw_subset.jsonl \
  --output-dir outputs/example_run \
  --provider mock \
  --save-format pt
```

Use a Hugging Face model backend:

```bash
python scripts/exp1_generate_artifacts.py \
  --dataset data/ragtruth/raw_subset.jsonl \
  --output-dir outputs/hf_run \
  --provider hf \
  --model-name distilgpt2 \
  --device auto \
  --compact-output \
  --save-format pt
```

Build the final Exp1/Exp2 table:

```bash
python scripts/build_exp12_table.py
```

Run activation patching:

```bash
python scripts/experiment3_activation_patching.py
```

Run temporal precedence analysis:

```bash
python scripts/run_experiment4_temporal_precedence.py
```

## Tech Stack

- Python 3.12+
- PyTorch
- NumPy
- pandas
- scikit-learn
- SciPy
- Matplotlib
- Hugging Face-compatible model execution path

## Why This Project Is Interesting

Most hallucination detectors focus on output text, retrieval overlap, or external verification. This project instead asks whether hallucinations are visible inside the model's own computation. By combining benchmark evaluation with mechanistic interpretability methods, it provides both detection scores and component-level explanations.

The result is a research-oriented codebase that connects practical hallucination detection with deeper analysis of where hallucination-related signals appear inside transformer models.
