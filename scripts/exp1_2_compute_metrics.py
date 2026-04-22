from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any
from tqdm import tqdm

import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nlp_track_b.person2.artifacts import (
    iter_artifact_paths,
    load_person1_artifact,
    save_metric_artifact,
)
from nlp_track_b.person2.metrics import (
    compute_attention_variance,
    compute_composite_score,
    compute_consistency_metric,
    compute_cosine_drift,
    compute_cross_layer_disagreement,
    compute_entropy_variance,
    compute_layer_confidence_degradation,
    compute_logit_lens_divergence,
    compute_mahalanobis,
    compute_pca_deviation,
    compute_uncertainty_ensemble,
    load_hf_model,
)


def _parse_layers(text: str) -> str | list[int]:
    if text in {"all", "last4"}:
        return text
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _metric_output_path(input_path: Path, output_dir: Path, save_format: str) -> Path:
    suffix = ".json" if save_format == "json" else ".pt"
    name = input_path.name
    for ext in (".json", ".pt", ".pth"):
        if name.endswith(ext):
            name = name[: -len(ext)]
            break
    return output_dir / f"{name}.person2_metrics{suffix}"


def _to_serializable_metric_artifact(
    record: dict[str, Any],
    metric_values: dict[str, torch.Tensor],
    per_layer_values: dict[str, torch.Tensor],
    layers_used: list[int],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    answer_start = int(record["answer_start_token_idx"])
    answer_end = int(record["answer_end_token_idx"])
    artifact: dict[str, Any] = {
        "sample_id": record.get("sample_id", record.get("id")),
        "split": record.get("split"),
        "answer_start_token_idx": answer_start,
        "answer_end_token_idx": answer_end,
        "answer_token_count": answer_end - answer_start,
        "layers_used": layers_used,
        "metadata": metadata,
    }
    artifact.update(metric_values)
    artifact.update(per_layer_values)
    return artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Person 2 token-level metrics")
    parser.add_argument("input", type=Path, help="Person 1 artifact file or directory")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/person2/metrics")
    )
    parser.add_argument(
        "--stats", type=Path, default=None, help="Optional fitted stats .pt path"
    )
    parser.add_argument(
        "--layers", default="last4", help="'last4', 'all', or comma-separated indices"
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Optional artifact limit for smoke tests"
    )
    parser.add_argument("--save-format", choices=["pt", "json"], default="pt")
    parser.add_argument(
        "--include-logit-lens",
        action="store_true",
        help="Also compute logit lens divergence by loading an HF model",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Execution device for HF logit-lens loading: auto, cpu, cuda, or cuda:N",
    )
    parser.add_argument(
        "--model-name", default=None, help="HF model name for logit lens"
    )
    return parser.parse_args()


def main() -> None:
    """Compute token-level metrics and save small metric artifacts."""
    args = parse_args()
    layers = _parse_layers(args.layers)
    paths = iter_artifact_paths(args.input)
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        raise ValueError(f"No artifacts found under: {args.input}")

    stats = (
        torch.load(args.stats, map_location="cpu", weights_only=False)
        if args.stats
        else {}
    )
    model = None
    if args.include_logit_lens:
        first = load_person1_artifact(paths[0], require_logits=True)
        model_name = args.model_name or first.get("metadata", {}).get("hf_model")
        if not model_name:
            raise ValueError(
                "--model-name is required when metadata.hf_model is missing."
            )

        model = load_hf_model(model_name, device=args.device)

    saved = 0
    for path in tqdm(paths, desc="Computing Person2 metrics", unit="artifact"):
        record = load_person1_artifact(path, require_logits=args.include_logit_lens)
        answer_start = int(record["answer_start_token_idx"])
        answer_end = int(record["answer_end_token_idx"])
        metric_values: dict[str, torch.Tensor] = {}
        per_layer_values: dict[str, torch.Tensor] = {}

        cosine = compute_cosine_drift(
            record["hidden_states"], answer_start, answer_end, layers=layers
        )
        layers_used = cosine["layers_used"]
        metric_values["cosine_drift"] = cosine["cosine_drift"]
        per_layer_values["cosine_drift_per_layer"] = cosine["cosine_drift_per_layer"]

        if "mahalanobis" in stats:
            mahalanobis = compute_mahalanobis(
                record["hidden_states"],
                answer_start,
                answer_end,
                stats["mahalanobis"],
            )
            metric_values["mahalanobis_distance"] = mahalanobis["mahalanobis_distance"]
            per_layer_values["mahalanobis_per_layer"] = mahalanobis[
                "mahalanobis_per_layer"
            ]
        if "pca" in stats:
            pca = compute_pca_deviation(
                record["hidden_states"], answer_start, answer_end, stats["pca"]
            )
            metric_values["pca_deviation"] = pca["pca_deviation"]
            per_layer_values["pca_deviation_per_layer"] = pca["pca_deviation_per_layer"]
        if model is not None:
            logit_lens = compute_logit_lens_divergence(
                record["hidden_states"],
                record["logits"],
                answer_start,
                answer_end,
                model=model,
                layers=layers,
            )
            metric_values["logit_lens_divergence"] = logit_lens["logit_lens_divergence"]
            per_layer_values["logit_lens_divergence_per_layer"] = logit_lens[
                "logit_lens_divergence_per_layer"
            ]

        # NEW: Advanced metrics for improved hallucination detection
        cross_layer = compute_cross_layer_disagreement(
            record["hidden_states"], answer_start, answer_end
        )
        metric_values["cross_layer_disagreement"] = cross_layer["cross_layer_disagreement"]
        per_layer_values["cross_layer_disagreement_per_pair"] = cross_layer["cross_layer_disagreement_per_pair"]

        uncertainty = compute_uncertainty_ensemble(
            record["hidden_states"], answer_start, answer_end
        )
        metric_values["uncertainty_ensemble"] = uncertainty["uncertainty_ensemble"]

        consistency = compute_consistency_metric(
            record["hidden_states"], answer_start, answer_end
        )
        metric_values["consistency_metric"] = consistency["consistency_metric"]
        per_layer_values["consistency_per_layer"] = consistency["consistency_per_layer"]

        attention = compute_attention_variance(
            record["hidden_states"], answer_start, answer_end
        )
        metric_values["attention_variance"] = attention["attention_variance"]
        per_layer_values["attention_variance_per_layer"] = attention["attention_variance_per_layer"]

        degradation = compute_layer_confidence_degradation(
            record["hidden_states"], answer_start, answer_end
        )
        metric_values["layer_confidence_degradation"] = degradation["layer_confidence_degradation"]
        per_layer_values["layer_magnitudes"] = degradation["layer_magnitudes"]

        entropy = compute_entropy_variance(
            record["hidden_states"], answer_start, answer_end
        )
        metric_values["entropy_variance"] = entropy["entropy_variance"]
        per_layer_values["entropy_per_layer"] = entropy["entropy_per_layer"]

        normalizers = stats.get("normalizers")
        if normalizers and all(name in metric_values for name in normalizers):
            metric_values["composite_score"] = compute_composite_score(
                metric_values, normalizers
            )

        artifact = _to_serializable_metric_artifact(
            record=record,
            metric_values=metric_values,
            per_layer_values=per_layer_values,
            layers_used=layers_used,
            metadata={
                "source_artifact": str(path),
                "stats_path": str(args.stats) if args.stats else None,
                "include_logit_lens": bool(args.include_logit_lens),
            },
        )
        out_path = _metric_output_path(path, args.output_dir, args.save_format)
        save_metric_artifact(out_path, artifact)
        saved += 1
        print(f"saved_metric_artifact={out_path}")

    print(f"artifact_count={saved}")


if __name__ == "__main__":
    main()
