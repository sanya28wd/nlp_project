from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class SplitConfig:
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42

    def validate(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")


@dataclass(slots=True)
class ModelConfig:
    provider: str = "mock"
    model_name: str = "distilgpt2"
    device: str = "auto"
    max_context_docs: int = 3
    max_seq_len: int = 512
    compact_output: bool = False
    compact_layers: int = 3
    logits_topk: int = 5
    hidden_size: int = 32
    num_layers: int = 4
    vocab_size: int = 128


@dataclass(slots=True)
class PipelineConfig:
    raw_dataset_path: Path
    output_dir: Path
    limit_samples: int = 0
    skip_existing_outputs: bool = True
    output_format: str = "json"
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    def validate(self) -> None:
        self.split.validate()
        if self.limit_samples < 0:
            raise ValueError(f"limit_samples must be >= 0, got {self.limit_samples}")
        if self.model.logits_topk < 1:
            raise ValueError(f"logits_topk must be >= 1, got {self.model.logits_topk}")
        if self.model.compact_layers < 1:
            raise ValueError(
                f"compact_layers must be >= 1, got {self.model.compact_layers}"
            )
        if self.model.device not in {"auto", "cpu", "mps", "cuda"}:
            raise ValueError(
                f"model.device must be one of auto/cpu/mps/cuda, got {self.model.device}"
            )
        if self.output_format not in {"json", "pt"}:
            raise ValueError(f"output_format must be json or pt, got {self.output_format}")
        if not self.raw_dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.raw_dataset_path}")
        self.output_dir.mkdir(parents=True, exist_ok=True)


