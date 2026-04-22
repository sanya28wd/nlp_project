from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class HallucinationSpan:
    start: int
    end: int
    label: str = "hallucinated"


@dataclass(slots=True)
class RawSample:
    sample_id: str
    question: str
    retrieved_context: list[str]
    answer: str
    hallucination_spans: list[HallucinationSpan] = field(default_factory=list)
    source_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TokenAlignment:
    token: str
    start: int
    end: int
    is_hallucinated: bool
    hallucination_label: str


@dataclass(slots=True)
class FormattedSample:
    sample_id: str
    split: str
    prompt: str
    answer_tokens: list[str]
    token_alignment: list[TokenAlignment]
    source_id: str
    metadata: dict[str, Any]


@dataclass(slots=True)
class ModelOutput:
    sample_id: str
    split: str
    hidden_states: list[list[list[float]]]
    logits: list[list[float]]
    token_outputs: list[str]
    token_alignment: list[TokenAlignment]
    prompt: str
    metadata: dict[str, Any]

