from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any

from .config import ModelConfig
from .schemas import FormattedSample, ModelOutput


_HF_CACHE: dict[str, tuple[Any, Any, Any]] = {}


@dataclass(slots=True)
class ForwardRunner:
    cfg: ModelConfig

    def run(self, sample: FormattedSample) -> ModelOutput:
        if self.cfg.provider == "mock":
            return _run_mock_forward(sample, self.cfg)
        if self.cfg.provider == "hf":
            return _run_hf_forward(sample, self.cfg)
        raise ValueError(f"Unknown model provider: {self.cfg.provider}")


def _seed_for(sample: FormattedSample) -> int:
    digest = hashlib.sha256(sample.sample_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _run_mock_forward(sample: FormattedSample, cfg: ModelConfig) -> ModelOutput:
    seed = _seed_for(sample)
    rng = random.Random(seed)
    token_count = max(len(sample.answer_tokens), 1)

    hidden_states = [
        [
            [round(rng.uniform(-1.0, 1.0), 6) for _ in range(cfg.hidden_size)]
            for _ in range(token_count)
        ]
        for _ in range(cfg.num_layers)
    ]

    logits = [
        [round(rng.uniform(-7.0, 7.0), 6) for _ in range(cfg.vocab_size)]
        for _ in range(token_count)
    ]
    token_outputs = [sample.answer_tokens[idx] for idx in range(token_count)]

    return ModelOutput(
        sample_id=sample.sample_id,
        split=sample.split,
        hidden_states=hidden_states,
        logits=logits,
        token_outputs=token_outputs,
        token_alignment=sample.token_alignment,
        prompt=sample.prompt,
        metadata=sample.metadata,
    )


def _run_hf_forward(sample: FormattedSample, cfg: ModelConfig) -> ModelOutput:
    try:
        if cfg.model_name not in _HF_CACHE:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
            model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
            model.eval()
            _HF_CACHE[cfg.model_name] = (tokenizer, model, torch)

        tokenizer, model, torch = _HF_CACHE[cfg.model_name]
    except ImportError as exc:
        raise RuntimeError(
            "provider='hf' requires 'transformers' and 'torch'. "
            "Install them or switch to provider='mock'."
        ) from exc

    # Resolve device
    if cfg.device == "auto":
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(cfg.device)

    if next(model.parameters()).device != device:
        model.to(device)

    full_text = sample.prompt + " ".join(sample.answer_tokens)
    encoded = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=cfg.max_seq_len,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        result = model(**encoded, output_hidden_states=True)

    logits_tensor = result.logits[0].detach().cpu()
    hidden_tensor = tuple(layer.detach().cpu() for layer in result.hidden_states)

    token_count = logits_tensor.shape[0]
    logits = logits_tensor.tolist()
    hidden_states = [layer[0].tolist() for layer in hidden_tensor]
    token_ids = encoded["input_ids"][0].detach().cpu().tolist()
    token_outputs = tokenizer.convert_ids_to_tokens(token_ids)

    aligned = sample.token_alignment[:token_count]

    metadata = {
        **sample.metadata,
        "hf_model": cfg.model_name,
        "sequence_length": token_count,
    }

    if cfg.compact_output:
        k = max(1, min(cfg.logits_topk, logits_tensor.shape[-1]))
        topk_values, topk_indices = torch.topk(logits_tensor, k=k, dim=-1)
        last_n_layers = min(3, len(hidden_tensor))
        hidden_states_last_n = [
            hidden_tensor[-(i + 1)][0].tolist() for i in range(last_n_layers)
        ]
        metadata.update(
            {
                "compact_output": True,
                "device": str(device),
                "logits_topk_k": k,
                "logits_topk_indices": topk_indices.tolist(),
                "logits_topk_values": topk_values.tolist(),
                "hidden_states_last_n_layers": hidden_states_last_n,
                "num_hidden_layers_saved": last_n_layers,
            }
        )
        logits = []
        hidden_states = []

    metadata.setdefault("device", str(device))

    return ModelOutput(
        sample_id=sample.sample_id,
        split=sample.split,
        hidden_states=hidden_states,
        logits=logits,
        token_outputs=token_outputs,
        token_alignment=aligned,
        prompt=sample.prompt,
        metadata=metadata,
    )

