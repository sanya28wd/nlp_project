from __future__ import annotations

import argparse
from pathlib import Path
import sys
from tqdm import tqdm

import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nlp_track_b.person2.artifacts import iter_artifact_paths, load_person1_artifact
from nlp_track_b.person2.metrics import (
    compute_cosine_drift,
    compute_logit_lens_divergence,
    compute_mahalanobis,
    compute_pca_deviation,
    fit_mahalanobis_stats,
    fit_normalizer_stats,
    fit_pca_stats,
    load_hf_model,
)


def _parse_layers(text: str) -> str | list[int]:
    if text in {"all", "last4"}:
        return text
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _limited_paths(path: Path, limit: int) -> list[Path]:
    paths = iter_artifact_paths(path)
    if limit:
        paths = paths[:limit]
    if not paths:
        raise ValueError(f"No artifacts found under: {path}")
    return paths


def _load_train_record(
    path: Path, *, require_logits: bool, expected_split: str
) -> dict:
    record = load_person1_artifact(path, require_logits=require_logits)
    if expected_split and record.get("split") != expected_split:
        raise ValueError(
            f"Stats must be fit on split={expected_split!r}; "
            f"{path} has split={record.get('split')!r}."
        )
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit Person 2 train-only metric stats")
    parser.add_argument(
        "input", type=Path, help="Train Person 1 artifact file or directory"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/person2/person2_stats.pt"),
        help="Output .pt stats path",
    )
    parser.add_argument(
        "--layers", default="last4", help="'last4', 'all', or comma-separated indices"
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Optional artifact limit for smoke tests"
    )
    parser.add_argument("--regularization", type=float, default=1e-4)
    parser.add_argument("--pca-components", type=int, default=16)
    parser.add_argument(
        "--expected-split",
        default="train",
        help="Required split for stats fitting; set empty string to disable",
    )
    parser.add_argument(
        "--include-logit-lens",
        action="store_true",
        help="Also fit normalizer stats for logit lens divergence",
    )
    parser.add_argument(
        "--model-name", default=None, help="HF model name for logit lens"
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Execution device for HF logit-lens loading: auto, cpu, cuda, or cuda:N",
    )
    return parser.parse_args()


def main() -> None:
    """Fit train-only Mahalanobis, PCA, and metric normalizer stats."""
    args = parse_args()
    layers = _parse_layers(args.layers)
    expected_split = args.expected_split or ""

    # If user passed model_outputs root and expects a split, auto-target that split folder.
    input_path = args.input
    if expected_split and args.input.is_dir():
        split_subdir = args.input / expected_split
        if split_subdir.exists() and split_subdir.is_dir():
            input_path = split_subdir

    paths = _limited_paths(input_path, args.limit)

    def records(require_logits: bool = False):
        for path in paths:
            yield _load_train_record(
                path, require_logits=require_logits, expected_split=expected_split
            )

    mahalanobis_stats = fit_mahalanobis_stats(
        records(),
        layers=layers,
        regularization=args.regularization,
    )
    pca_stats = fit_pca_stats(
        records(),
        layers=layers,
        n_components=args.pca_components,
    )

    metric_values: dict[str, list[torch.Tensor]] = {
        "cosine_drift": [],
        "mahalanobis_distance": [],
        "pca_deviation": [],
    }
    model = None
    if args.include_logit_lens:
        first_record = _load_train_record(
            paths[0], require_logits=True, expected_split=expected_split
        )
        model_name = args.model_name or first_record.get("metadata", {}).get("hf_model")
        if not model_name:
            raise ValueError(
                "--model-name is required when metadata.hf_model is missing."
            )
        model = load_hf_model(model_name, device=args.device)
        metric_values["logit_lens_divergence"] = []

    records_iter = records(require_logits=args.include_logit_lens)
    for record in tqdm(
        records_iter, total=len(paths), desc="Fitting Person2 stats", unit="artifact"
    ):
        answer_start = int(record["answer_start_token_idx"])
        answer_end = int(record["answer_end_token_idx"])
        cosine = compute_cosine_drift(
            record["hidden_states"], answer_start, answer_end, layers=layers
        )
        mahalanobis = compute_mahalanobis(
            record["hidden_states"],
            answer_start,
            answer_end,
            mahalanobis_stats,
        )
        pca = compute_pca_deviation(
            record["hidden_states"], answer_start, answer_end, pca_stats
        )
        metric_values["cosine_drift"].append(cosine["cosine_drift"])
        metric_values["mahalanobis_distance"].append(
            mahalanobis["mahalanobis_distance"]
        )
        metric_values["pca_deviation"].append(pca["pca_deviation"])

        if model is not None:
            logit_lens = compute_logit_lens_divergence(
                record["hidden_states"],
                record["logits"],
                answer_start,
                answer_end,
                model=model,
                layers=layers,
            )
            metric_values["logit_lens_divergence"].append(
                logit_lens["logit_lens_divergence"]
            )

    normalizers = fit_normalizer_stats(metric_values)
    artifact = {
        "mahalanobis": mahalanobis_stats,
        "pca": pca_stats,
        "normalizers": normalizers,
        "metadata": {
            "artifact_count": len(paths),
            "layers": args.layers,
            "expected_split": expected_split,
            "pca_components": args.pca_components,
            "regularization": args.regularization,
            "includes_logit_lens": bool(args.include_logit_lens),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.output)
    print(f"saved_stats={args.output}")
    print(f"artifact_count={len(paths)}")
    print(f"layers_used={mahalanobis_stats.get('layers_used', [])}")
    print(f"normalizers={sorted(normalizers)}")


if __name__ == "__main__":
    main()
