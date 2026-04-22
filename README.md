NLP Track B: Pre-Generation Causal Drift Metric
This repository contains our Track B pipeline for hallucination detection using hidden-state representation metrics, causal patching, temporal precedence analysis, and zero-shot transfer evaluation.

What This Repo Contains
Exp 1 & 2: Composite AUROC on RAGTruth + layer localization
Exp 3: Activation patching (causal intervention)
Exp 4: Temporal precedence (t-3 to t+1)
Exp 5: Cross-domain transfer to HaluEval (zero-shot)
Exp 6–8: FFN vs attention decomposition, failure cases, SOTA gap analysis

Environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

Main Scripts
scripts/exp1_generate_artifacts.py
scripts/exp1_2_fit_train_stats.py
scripts/exp1_2_compute_metrics.py
scripts/build_exp12_table.py
scripts/experiment3_activation_patching.py
scripts/plot_experiment3_heatmap.py
scripts/run_experiment4_temporal_precedence.py
scripts/build_exp5_halueval_table.py
scripts/build_exp6_8_analysis.py

Reproduce Final Experiment Outputs

Exp 1 & 2
python scripts/build_exp12_table.py \
  --metrics-dir outputs/person2/metrics_full_gpt2medium_logitlens \
  --splits-dir artifacts/person1_ragtruth_full_gpt2medium_pt/splits \
  --out-csv "outputs/exp1&2/exp1_exp2_table_final.csv" \
  --out-md "outputs/exp1&2/exp1_exp2_table_final.md" \
  --out-layer-profile-png "outputs/exp1&2/E2_layer_profile.png" \
  --out-layer-profile-csv "outputs/exp1&2/E2_layer_profile.csv"
  
Exp 3
python scripts/experiment3_activation_patching.py \
  artifacts/person1_ragtruth_full_gpt2medium_pt/model_outputs \
  --output-dir outputs/exp3_full \
  --pairs-per-component 30 \
  --max-candidates-per-class 180 \
  --save-details

python scripts/plot_experiment3_heatmap.py \
  --summary-json outputs/exp3_full/experiment3_activation_patching_summary.json \
  --out-png outputs/exp3_full/E3_component_layer_heatmap.png \
  --out-csv outputs/exp3_full/E3_component_layer_heatmap.csv

  
Exp 4
python scripts/run_experiment4_temporal_precedence.py \
  --input-dir artifacts/person1_ragtruth_full_gpt2medium_pt/model_outputs/test \
  --stats-path outputs/person2/stats_full_gpt2medium_logitlens.pt \
  --exp2-results-json "outputs/exp1&2/E1_E2_results.json" \
  --model-name gpt2-medium \
  --device auto \
  --logit-lens-mode exact_full \
  --output-dir outputs/exp4_temporal_precedence

  
Exp 5 (Primary: dialogue + qa + summarization)
python scripts/build_exp5_halueval_table.py \
  --ragtruth-metrics-dir outputs/person2/metrics_full_gpt2medium_logitlens \
  --halueval-metrics-dir outputs/person2/metrics_halueval_gpt2medium_logitlens \
  --ragtruth-splits-dir artifacts/person1_ragtruth_full_gpt2medium_pt/splits \
  --halueval-split-path artifacts/person1_halueval_gpt2medium_pt/splits/test.jsonl \
  --halueval-tasks dialogue,qa,summarization \
  --out-csv outputs/exp5_halueval/experiment5_halueval_transfer_table.csv \
  --out-md outputs/exp5_halueval/experiment5_halueval_transfer_table.md \
  --out-summary outputs/exp5_halueval/experiment5_halueval_transfer_summary.json

  
Exp 6–8
python scripts/build_exp6_8_analysis.py \
  --metrics-dir outputs/person2/metrics_full_gpt2medium_logitlens \
  --splits-dir artifacts/person1_ragtruth_full_gpt2medium_pt/splits \
  --artifacts-dir artifacts/person1_ragtruth_full_gpt2medium_pt/model_outputs/test \
  --exp3-summary outputs/exp3_full/experiment3_activation_patching_summary.json \
  --exp2-json "outputs/exp1&2/E1_E2_results.json" \
  --output-dir outputs/exp6_8_analysis
Final Reported Numbers
RAGTruth headline composite AUROC: 0.7224
HaluEval transfer composite AUROC (3-task combined): 0.6663
Attention-entropy baseline AUROC: 0.6661

outputs:

outputs/final_results_summary.json
outputs/final_results_summary.md
Output Folders (Submission)
outputs/exp1&2/
outputs/exp3_full/
outputs/exp4_temporal_precedence/
outputs/exp5_halueval/
outputs/exp6_8_analysis/
outputs/final_results_summary.json
outputs/final_results_summary.md

Notes
artifacts/ is large and treated as intermediate generation data.
outputs/person2/ contains intermediate metric/stats artifacts used to build final experiment outputs.
For Exp 5 primary reporting, general is excluded; it is kept as optional stress-test analysis.
