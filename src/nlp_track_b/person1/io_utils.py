from __future__ import annotations

import json
from pathlib import Path
import torch

from .schemas import ModelOutput


def model_output_path(base_dir: Path, split: str, sample_id: str, fmt: str = "json") -> Path:
    target_dir = base_dir / "model_outputs" / split
    suffix = ".json" if fmt == "json" else ".pt"
    return target_dir / f"{sample_id}{suffix}"


def save_model_output(base_dir: Path, output: ModelOutput, fmt: str = "json") -> Path:
    target_dir = base_dir / "model_outputs" / output.split
    target_dir.mkdir(parents=True, exist_ok=True)
    out_file = model_output_path(base_dir, output.split, output.sample_id, fmt=fmt)

    payload = {
        "sample_id": output.sample_id,
        "split": output.split,
        "prompt": output.prompt,
        "token_outputs": output.token_outputs,
        "token_alignment": [
            {
                "token": x.token,
                "start": x.start,
                "end": x.end,
                "is_hallucinated": x.is_hallucinated,
                "hallucination_label": x.hallucination_label,
            }
            for x in output.token_alignment
        ],
        "hidden_states": output.hidden_states,
        "logits": output.logits,
        "metadata": output.metadata,
    }

    if fmt == "json":
        with out_file.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True)
    else:
        metadata_pt = dict(payload.get("metadata", {}))
        if metadata_pt.get("compact_output"):
            if "logits_topk_indices" in metadata_pt:
                metadata_pt["logits_topk_indices"] = torch.tensor(
                    metadata_pt["logits_topk_indices"], dtype=torch.int32
                )
            if "logits_topk_values" in metadata_pt:
                metadata_pt["logits_topk_values"] = torch.tensor(
                    metadata_pt["logits_topk_values"], dtype=torch.float16
                )
            if "hidden_states_last_n_layers" in metadata_pt:
                metadata_pt["hidden_states_last_n_layers"] = torch.tensor(
                    metadata_pt["hidden_states_last_n_layers"], dtype=torch.float16
                )

        payload_pt = {
            **payload,
            "hidden_states": torch.tensor(output.hidden_states, dtype=torch.float32)
            if output.hidden_states
            else torch.tensor([], dtype=torch.float32),
            "logits": torch.tensor(output.logits, dtype=torch.float32)
            if output.logits
            else torch.tensor([], dtype=torch.float32),
            "metadata": metadata_pt,
        }
        torch.save(payload_pt, out_file)

    return out_file


def save_run_summary(base_dir: Path, summary: dict[str, int]) -> Path:
    out = base_dir / "run_summary.json"
    with out.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=True, indent=2)
    return out

