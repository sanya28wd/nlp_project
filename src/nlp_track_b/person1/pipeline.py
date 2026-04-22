from __future__ import annotations

from .config import PipelineConfig
from .data import load_jsonl_dataset, normalize_samples, save_split_manifests, split_samples
from .formatting import build_formatted_sample
from .io_utils import model_output_path, save_model_output, save_run_summary
from .model import ForwardRunner

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - fallback when tqdm missing
    tqdm = None


def run_person1_pipeline(cfg: PipelineConfig) -> dict[str, int]:
    cfg.validate()

    raw = load_jsonl_dataset(cfg.raw_dataset_path)
    cleaned = normalize_samples(raw, max_context_docs=cfg.model.max_context_docs)
    if cfg.limit_samples:
        cleaned = cleaned[: cfg.limit_samples]
    split_map = split_samples(cleaned, cfg.split)
    save_split_manifests(split_map, cfg.output_dir)

    runner = ForwardRunner(cfg.model)
    summary: dict[str, int] = {
        "train": 0,
        "val": 0,
        "test": 0,
        "processed_new": 0,
        "skipped_existing": 0,
    }

    total_samples = sum(len(rows) for rows in split_map.values())
    completed = 0

    pbar = None
    if tqdm is not None:
        pbar = tqdm(total=total_samples, desc="Person1 forward", unit="sample")

    for split_name, samples in split_map.items():
        for sample in samples:
            if cfg.skip_existing_outputs:
                out_path = model_output_path(
                    cfg.output_dir, split_name, sample.sample_id, fmt=cfg.output_format
                )
                if out_path.exists():
                    summary[split_name] += 1
                    summary["skipped_existing"] += 1
                    completed += 1
                    if pbar is not None:
                        remaining = max(total_samples - completed, 0)
                        pbar.update(1)
                        pbar.set_postfix(
                            done=completed,
                            remaining=remaining,
                            new=summary["processed_new"],
                            skipped=summary["skipped_existing"],
                        )
                    continue
            formatted = build_formatted_sample(sample, split=split_name)
            output = runner.run(formatted)
            save_model_output(cfg.output_dir, output, fmt=cfg.output_format)
            summary[split_name] += 1
            summary["processed_new"] += 1
            completed += 1
            if pbar is not None:
                remaining = max(total_samples - completed, 0)
                pbar.update(1)
                pbar.set_postfix(
                    done=completed,
                    remaining=remaining,
                    new=summary["processed_new"],
                    skipped=summary["skipped_existing"],
                )

    if pbar is not None:
        pbar.close()

    summary["total"] = summary["train"] + summary["val"] + summary["test"]
    save_run_summary(cfg.output_dir, summary)
    return summary

