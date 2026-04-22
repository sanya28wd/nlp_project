from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert RAGTruth files to Person 1 pipeline JSONL schema"
    )
    parser.add_argument(
        "--response-jsonl", 
        type=Path,
        default=Path("RAGTruth-main/dataset/response.jsonl"),
        help="Path to RAGTruth response.jsonl",
    )
    parser.add_argument(
        "--source-info-jsonl",
        type=Path,
        default=Path("RAGTruth-main/dataset/source_info.jsonl"),
        help="Path to RAGTruth source_info.jsonl",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("data/ragtruth/raw.jsonl"),
        help="Output file path in Person 1 schema",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional row limit for smoke tests (0 = all rows)",
    )
    return parser.parse_args()


def load_source_info(path: Path) -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            source_id = str(row.get("source_id", "")).strip()
            if not source_id:
                continue
            index[source_id] = {
                "task_type": str(row.get("task_type", "")),
                "source": str(row.get("source", "")),
                "source_info": str(row.get("source_info", "")),
                "prompt": str(row.get("prompt", "")),
            }
    return index


def map_label_type(label_type: str) -> str:
    text = label_type.lower()
    if "conflict" in text:
        return "contradictory"
    if "baseless" in text or "introduction" in text:
        return "unsupported"
    if "fabric" in text:
        return "fabricated"
    return "hallucinated"


def choose_question(prompt: str, task_type: str) -> str:
    cleaned = prompt.strip()
    if cleaned:
        return cleaned
    if task_type:
        return f"Task: {task_type}"
    return ""


def main() -> None:
    args = parse_args()
    if not args.response_jsonl.exists():
        raise FileNotFoundError(f"Missing file: {args.response_jsonl}")
    if not args.source_info_jsonl.exists():
        raise FileNotFoundError(f"Missing file: {args.source_info_jsonl}")

    source_map = load_source_info(args.source_info_jsonl)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    missing_source = 0
    with args.response_jsonl.open("r", encoding="utf-8") as src, args.output_jsonl.open(
        "w", encoding="utf-8"
    ) as out:
        for line in src:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)

            source_id = str(row.get("source_id", "")).strip()
            source_info = source_map.get(source_id, {})
            if not source_info:
                missing_source += 1

            labels = row.get("labels", [])
            spans = []
            for label in labels:
                spans.append(
                    {
                        "start": int(label.get("start", 0)),
                        "end": int(label.get("end", 0)),
                        "label": map_label_type(str(label.get("label_type", ""))),
                    }
                )

            prompt = str(source_info.get("prompt", "")).strip()
            task_type = str(source_info.get("task_type", "")).strip()
            question = choose_question(prompt=prompt, task_type=task_type)
            context_text = str(source_info.get("source_info", "")).strip()

            out_row = {
                "sample_id": f"ragtruth_{row.get('id')}",
                "source_id": source_id or f"missing_source_{row.get('id')}",
                "question": question,
                "retrieved_context": [context_text] if context_text else [],
                "answer": str(row.get("response", "")),
                "hallucination_spans": spans,
                "metadata": {
                    "model": row.get("model"),
                    "temperature": row.get("temperature"),
                    "split_original": row.get("split"),
                    "quality": row.get("quality"),
                    "task_type": source_info.get("task_type"),
                    "source": source_info.get("source"),
                },
            }

            out.write(json.dumps(out_row, ensure_ascii=True) + "\n")
            written += 1

            if args.limit and written >= args.limit:
                break

    print(
        json.dumps(
            {
                "output": str(args.output_jsonl),
                "rows_written": written,
                "rows_missing_source_info": missing_source,
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()

