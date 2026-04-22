from __future__ import annotations

import re

from .schemas import FormattedSample, RawSample, TokenAlignment

_TOKEN_PATTERN = re.compile(r"\S+")


def _token_spans(text: str) -> list[tuple[str, int, int]]:
    spans: list[tuple[str, int, int]] = []
    for match in _TOKEN_PATTERN.finditer(text):
        spans.append((match.group(0), match.start(), match.end()))
    return spans


def _build_prompt(question: str, contexts: list[str]) -> str:
    context_text = "\n".join(f"[{idx + 1}] {doc}" for idx, doc in enumerate(contexts))
    return (
        "Question:\n"
        f"{question}\n\n"
        "Retrieved Context:\n"
        f"{context_text}\n\n"
        "Answer:\n"
    )


def build_formatted_sample(sample: RawSample, split: str) -> FormattedSample:
    prompt = _build_prompt(sample.question, sample.retrieved_context)

    answer_spans = _token_spans(sample.answer)
    token_alignment: list[TokenAlignment] = []
    for token, start, end in answer_spans:
        matched = [
            span
            for span in sample.hallucination_spans
            if start < span.end and end > span.start
        ]
        is_hallucinated = len(matched) > 0
        label = matched[0].label if matched else "none"
        token_alignment.append(
            TokenAlignment(
                token=token,
                start=start,
                end=end,
                is_hallucinated=is_hallucinated,
                hallucination_label=label,
            )
        )

    return FormattedSample(
        sample_id=sample.sample_id,
        split=split,
        prompt=prompt,
        answer_tokens=[x[0] for x in answer_spans],
        token_alignment=token_alignment,
        source_id=sample.source_id,
        metadata=sample.metadata,
    )

