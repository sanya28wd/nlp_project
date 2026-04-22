from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


TASK_FILES = {
    "qa": "qa_data.json",
    "dialogue": "dialogue_data.json",
    "summarization": "summarization_data.json",
    "general": "general_data.json",
}


def _read_json_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
        if isinstance(payload, list):
            return [x for x in payload if isinstance(x, dict)]
        if isinstance(payload, dict):
            return [payload]
    except json.JSONDecodeError:
        pass

    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            obj = json.loads(line)
            if isinstance(obj, dict):
                records.append(obj)
    return records


def _full_span(answer: str, label: str = "hallucinated") -> list[dict[str, Any]]:
    answer = answer or ""
    if not answer.strip():
        return []
    return [{"start": 0, "end": len(answer), "label": label}]


def _binary_pair(
    *,
    sample_id: str,
    source_id: str,
    question: str,
    contexts: list[str],
    right_answer: str,
    hallucinated_answer: str,
    task: str,
) -> Iterable[dict[str, Any]]:
    base_meta = {"dataset": "HaluEval", "task_type": task}
    yield {
        "sample_id": f"{sample_id}_right",
        "source_id": source_id,
        "question": question,
        "retrieved_context": contexts,
        "answer": right_answer,
        "hallucination_spans": [],
        "metadata": {**base_meta, "label": "right"},
    }
    yield {
        "sample_id": f"{sample_id}_hallucinated",
        "source_id": source_id,
        "question": question,
        "retrieved_context": contexts,
        "answer": hallucinated_answer,
        "hallucination_spans": _full_span(hallucinated_answer),
        "metadata": {**base_meta, "label": "hallucinated"},
    }


def _convert_task_record(task: str, idx: int, row: dict[str, Any]) -> list[dict[str, Any]]:
    sid = str(row.get("id") or row.get("ID") or idx)
    source_id = f"halueval_{task}_{sid}"

    if task == "qa":
        question = str(row.get("question", ""))
        contexts = [str(row.get("knowledge", ""))]
        return list(
            _binary_pair(
                sample_id=source_id,
                source_id=source_id,
                question=question,
                contexts=contexts,
                right_answer=str(row.get("right_answer", "")),
                hallucinated_answer=str(row.get("hallucinated_answer", "")),
                task=task,
            )
        )

    if task == "dialogue":
        history = str(row.get("dialogue_history", ""))
        question = f"Continue the dialogue:\n{history}"
        contexts = [str(row.get("knowledge", ""))]
        return list(
            _binary_pair(
                sample_id=source_id,
                source_id=source_id,
                question=question,
                contexts=contexts,
                right_answer=str(row.get("right_response", "")),
                hallucinated_answer=str(row.get("hallucinated_response", "")),
                task=task,
            )
        )

    if task == "summarization":
        question = "Summarize the document faithfully."
        contexts = [str(row.get("document", ""))]
        return list(
            _binary_pair(
                sample_id=source_id,
                source_id=source_id,
                question=question,
                contexts=contexts,
                right_answer=str(row.get("right_summary", "")),
                hallucinated_answer=str(row.get("hallucinated_summary", "")),
                task=task,
            )
        )

    if task == "general":
        answer = str(row.get("chatgpt_response", ""))
        label_text = str(row.get("hallucination_label", "")).strip().lower()
        is_hallucinated = label_text in {"yes", "y", "true", "1", "hallucinated"}
        return [
            {
                "sample_id": source_id,
                "source_id": source_id,
                "question": str(row.get("user_query", "")),
                "retrieved_context": [],
                "answer": answer,
                "hallucination_spans": _full_span(answer) if is_hallucinated else [],
                "metadata": {
                    "dataset": "HaluEval",
                    "task_type": task,
                    "hallucination_label": row.get("hallucination_label"),
                },
            }
        ]

    raise ValueError(f"Unknown HaluEval task: {task}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert HaluEval data to Person1 JSONL")
    parser.add_argument("--input-dir", type=Path, default=Path("data/halueval/raw"))
    parser.add_argument("--output-jsonl", type=Path, default=Path("data/halueval/person1.jsonl"))
    parser.add_argument(
        "--tasks",
        default="qa,dialogue,summarization,general",
        help="Comma-separated subset of: qa, dialogue, summarization, general",
    )
    parser.add_argument(
        "--limit-per-task",
        type=int,
        default=0,
        help="Optional source-record limit per task before right/hallucinated expansion.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = [x.strip() for x in args.tasks.split(",") if x.strip()]
    unknown = sorted(set(tasks).difference(TASK_FILES))
    if unknown:
        raise ValueError(f"Unknown tasks: {unknown}")

    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for task in tasks:
        path = args.input_dir / TASK_FILES[task]
        if not path.exists():
            raise FileNotFoundError(f"Missing HaluEval file: {path}")
        records = _read_json_records(path)
        if args.limit_per_task:
            records = records[: args.limit_per_task]
        counts[task] = len(records)
        for idx, record in enumerate(records):
            rows.extend(_convert_task_record(task, idx, record))

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    print(
        json.dumps(
            {
                "output": str(args.output_jsonl),
                "tasks": tasks,
                "source_records_by_task": counts,
                "rows_written": len(rows),
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
