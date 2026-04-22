from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path
import sys

# Ensure src-layout imports work when running this file directly.
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

config_mod = import_module("nlp_track_b.person1.config")
pipeline_mod = import_module("nlp_track_b.person1.pipeline")
ModelConfig = config_mod.ModelConfig
PipelineConfig = config_mod.PipelineConfig
SplitConfig = config_mod.SplitConfig
run_person1_pipeline = pipeline_mod.run_person1_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Person 1 data + inference pipeline")
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to raw JSONL dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for split manifests and model outputs",
    )
    parser.add_argument(
        "--provider",
        choices=["mock", "hf"],
        default="mock",
        help="Forward-pass backend. 'mock' is deterministic and dependency-free.",
    )
    parser.add_argument(
        "--model-name",
        default="distilgpt2",
        help="HF model name if provider='hf'",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
        help="Execution device for HF provider. 'auto' prefers mps on Mac, then cuda, else cpu.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=512,
        help="Maximum tokenized sequence length for HF forward pass.",
    )
    parser.add_argument(
        "--compact-output",
        action="store_true",
        help="Save smaller HF JSON artifacts (last-layer + top-k logits in metadata).",
    )
    parser.add_argument(
        "--logits-topk",
        type=int,
        default=5,
        help="Top-k logits to keep per token when --compact-output is enabled.",
    )
    parser.add_argument(
        "--limit-samples",
        type=int,
        default=0,
        help="Optional cap on number of normalized samples to process (0 = all).",
    )
    parser.add_argument(
        "--save-format",
        choices=["json", "pt"],
        default="pt",
        help="Artifact format for Person1 outputs. 'pt' is much smaller/faster.",
    )
    parser.add_argument(
        "--skip-existing-outputs",
        action="store_true",
        default=True,
        help="Skip samples whose output artifact already exists (recommended for resumable long runs).",
    )
    parser.add_argument(
        "--no-skip-existing-outputs",
        action="store_true",
        help="Force re-run even if output files already exist.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PipelineConfig(
        raw_dataset_path=args.dataset,
        output_dir=args.output_dir,
        limit_samples=args.limit_samples,
    skip_existing_outputs=(not args.no_skip_existing_outputs),
    output_format=args.save_format,
        split=SplitConfig(
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        ),
        model=ModelConfig(
            provider=args.provider,
            model_name=args.model_name,
            device=args.device,
            max_seq_len=args.max_seq_len,
            compact_output=args.compact_output,
            logits_topk=args.logits_topk,
        ),
    )
    summary = run_person1_pipeline(cfg)
    print(summary)


if __name__ == "__main__":
    main()



