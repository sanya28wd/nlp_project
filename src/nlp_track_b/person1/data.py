from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .config import SplitConfig
from .schemas import HallucinationSpan, RawSample


REQUIRED_FIELDS = {"sample_id", "question", "retrieved_context", "answer"}


def load_jsonl_dataset(path: Path) -> list[RawSample]:
    samples: list[RawSample] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            obj = json.loads(text)
            missing = REQUIRED_FIELDS.difference(obj)
            if missing:
                raise ValueError(f"Row {idx} missing required fields: {sorted(missing)}")
            contexts = obj["retrieved_context"]
            if not isinstance(contexts, list) or not all(isinstance(x, str) for x in contexts):
                raise ValueError(f"Row {idx} has invalid 'retrieved_context'; expected list[str]")

            spans = [
                HallucinationSpan(
                    start=int(s["start"]),
                    end=int(s["end"]),
                    label=str(s.get("label", "hallucinated")),
                )
                for s in obj.get("hallucination_spans", [])
            ]

            source_id = str(obj.get("source_id", obj["sample_id"]))
            metadata = dict(obj.get("metadata", {}))
            samples.append(
                RawSample(
                    sample_id=str(obj["sample_id"]),
                    question=str(obj["question"]),
                    retrieved_context=contexts,
                    answer=str(obj["answer"]),
                    hallucination_spans=spans,
                    source_id=source_id,
                    metadata=metadata,
                )
            )

    if not samples:
        raise ValueError(f"No records found in dataset file: {path}")
    return samples


def normalize_samples(samples: list[RawSample], max_context_docs: int) -> list[RawSample]:
    normalized: list[RawSample] = []
    for sample in samples:
        contexts = [doc.strip() for doc in sample.retrieved_context if doc and doc.strip()]
        normalized.append(
            RawSample(
                sample_id=sample.sample_id.strip(),
                question=" ".join(sample.question.split()),
                retrieved_context=contexts[:max_context_docs],
                answer=" ".join(sample.answer.split()),
                hallucination_spans=sample.hallucination_spans,
                source_id=sample.source_id.strip() or sample.sample_id.strip(),
                metadata=sample.metadata,
            )
        )
    return normalized


def split_samples(samples: list[RawSample], cfg: SplitConfig) -> dict[str, list[RawSample]]:
    cfg.validate()
    buckets = {"train": [], "val": [], "test": []}

    # Group by source to prevent near-duplicate leakage across splits.
    groups: dict[str, list[RawSample]] = {}
    for sample in samples:
        groups.setdefault(sample.source_id, []).append(sample)

    group_items = list(groups.items())
    group_items.sort(
        key=lambda x: hashlib.sha256(f"{x[0]}:{cfg.seed}".encode("utf-8")).hexdigest()
    )

    total_groups = len(group_items)
    train_groups = int(cfg.train_ratio * total_groups)
    val_groups = int(cfg.val_ratio * total_groups)
    test_groups = total_groups - train_groups - val_groups

    if total_groups >= 3:
        # Ensure each split gets at least one group when possible.
        train_groups = max(train_groups, 1)
        val_groups = max(val_groups, 1)
        test_groups = max(test_groups, 1)
        while train_groups + val_groups + test_groups > total_groups:
            if train_groups >= val_groups and train_groups >= test_groups and train_groups > 1:
                train_groups -= 1
            elif val_groups >= test_groups and val_groups > 1:
                val_groups -= 1
            elif test_groups > 1:
                test_groups -= 1

    train_cutoff = train_groups
    val_cutoff = train_groups + val_groups
    for idx, (_, group_samples) in enumerate(group_items):
        if idx < train_cutoff:
            target = "train"
        elif idx < val_cutoff:
            target = "val"
        else:
            target = "test"
        buckets[target].extend(group_samples)

    for split_name in buckets:
        buckets[split_name].sort(key=lambda x: x.sample_id)
    return buckets


def save_split_manifests(split_map: dict[str, list[RawSample]], out_dir: Path) -> None:
    split_dir = out_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    for split_name, rows in split_map.items():
        file_path = split_dir / f"{split_name}.jsonl"
        with file_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                payload = {
                    "sample_id": row.sample_id,
                    "source_id": row.source_id,
                    "question": row.question,
                    "retrieved_context": row.retrieved_context,
                    "answer": row.answer,
                    "hallucination_spans": [
                        {"start": s.start, "end": s.end, "label": s.label}
                        for s in row.hallucination_spans
                    ],
                    "metadata": row.metadata,
                }
                handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


