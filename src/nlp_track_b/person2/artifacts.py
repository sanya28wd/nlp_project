"""Load and save artifact files for Person 2 metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def iter_artifact_paths(input_path: Path) -> list[Path]:
    """Recursively find all .json/.pt artifact files from Person 1 output."""
    input_path = Path(input_path)
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        json_files = list(input_path.rglob("*.json"))
        pt_files = list(input_path.rglob("*.pt"))
        return sorted(json_files + pt_files)
    return []


def load_person1_artifact(
    path: Path, require_logits: bool = False
) -> dict[str, Any]:
    """Load a Person 1 forward-pass artifact and extract usable tensors."""
    if path.suffix == ".pt":
        data = torch.load(path, map_location="cpu", weights_only=False)
    else:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

    # Handle compact HF output (last N layers in metadata)
    metadata = data.get("metadata", {})
    is_compact = metadata.get("compact_output", False)

    if is_compact:
        hidden_states_list = metadata.get("hidden_states_last_n_layers", [])
        if isinstance(hidden_states_list, torch.Tensor):
            hidden_states = hidden_states_list.detach().to(dtype=torch.float32)
        else:
            hidden_states = torch.tensor(hidden_states_list, dtype=torch.float32)
        token_alignment = data.get("token_alignment", [])
    else:
        hidden_states_raw = data.get("hidden_states", [])
        if isinstance(hidden_states_raw, torch.Tensor):
            hidden_states = hidden_states_raw.detach().to(dtype=torch.float32)
        else:
            hidden_states = torch.tensor(hidden_states_raw, dtype=torch.float32)
        token_alignment = data.get("token_alignment", [])

    if require_logits and is_compact:
        logits_topk_indices = metadata.get("logits_topk_indices", [])
        logits_topk_values = metadata.get("logits_topk_values", [])
        logits_topk_k = metadata.get("logits_topk_k", 0)
        logits = (logits_topk_indices, logits_topk_values, logits_topk_k)
    else:
        logits_raw = data.get("logits", [])
        if isinstance(logits_raw, torch.Tensor):
            logits = logits_raw.detach().to(dtype=torch.float32)
        else:
            logits = torch.tensor(logits_raw, dtype=torch.float32)

    answer_span = _get_answer_span(token_alignment)

    return {
        "sample_id": data.get("sample_id"),
        "split": data.get("split"),
        "hidden_states": hidden_states,
        "logits": logits,
        "token_alignment": token_alignment,
        "prompt": data.get("prompt", ""),
        "token_outputs": data.get("token_outputs", []),
        "answer_start_token_idx": answer_span[0],
        "answer_end_token_idx": answer_span[1],
        "metadata": metadata,
    }


def _get_answer_span(token_alignment: list[dict]) -> tuple[int, int]:
    """Extract answer token span from token_alignment."""
    if not token_alignment:
        return (0, 0)
    return (0, len(token_alignment))


def save_metric_artifact(path: Path, artifact: dict[str, Any]) -> None:
    """Save computed metrics as JSON or PyTorch file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix == ".json":
        serialized = {
            k: (v.tolist() if isinstance(v, torch.Tensor) else v)
            for k, v in artifact.items()
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(serialized, f, ensure_ascii=True, indent=2)
    else:
        torch.save(artifact, path)
