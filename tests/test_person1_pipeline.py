from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nlp_track_b.person1.config import PipelineConfig
from nlp_track_b.person1.pipeline import run_person1_pipeline


class Person1PipelineTests(unittest.TestCase):
    def test_pipeline_creates_artifacts_and_preserves_group_split(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        dataset = repo_root / "data" / "sample" / "raw_ragtruth_like.jsonl"

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            cfg = PipelineConfig(raw_dataset_path=dataset, output_dir=out)
            summary = run_person1_pipeline(cfg)

            self.assertEqual(summary["total"], 6)

            split_dir = out / "splits"
            self.assertTrue((split_dir / "train.jsonl").exists())
            self.assertTrue((split_dir / "val.jsonl").exists())
            self.assertTrue((split_dir / "test.jsonl").exists())

            # source_id=srcA appears twice and must stay in one split.
            src_splits = []
            for split in ("train", "val", "test"):
                with (split_dir / f"{split}.jsonl").open("r", encoding="utf-8") as handle:
                    for line in handle:
                        row = json.loads(line)
                        if row["source_id"] == "srcA":
                            src_splits.append(split)
            self.assertEqual(len(set(src_splits)), 1)

            for split in ("train", "val", "test"):
                outputs_dir = out / "model_outputs" / split
                if not outputs_dir.exists():
                    continue
                for file_path in outputs_dir.glob("*.json"):
                    payload = json.loads(file_path.read_text(encoding="utf-8"))
                    self.assertIn("hidden_states", payload)
                    self.assertIn("logits", payload)
                    self.assertIn("token_outputs", payload)
                    self.assertIn("token_alignment", payload)


if __name__ == "__main__":
    unittest.main()


