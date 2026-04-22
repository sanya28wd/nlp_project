#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score, roc_auc_score


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nlp_track_b.person1.formatting import build_formatted_sample
from nlp_track_b.person1.schemas import HallucinationSpan, RawSample


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute exact logit-confidence baseline in streaming mode (no artifact dump)."
    )
    p.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("artifacts/person1_ragtruth_full_gpt2medium_pt/splits"),
    )
    p.add_argument("--model-name", default="gpt2-medium")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument(
        "--limit-per-split",
        type=int,
        default=0,
        help="Optional sample cap per split for smoke tests (0 = all)",
    )
    p.add_argument(
        "--out-scores-csv",
        type=Path,
        default=Path("outputs/person3_full_logitlens/logit_confidence_scores.csv"),
    )
    p.add_argument(
        "--out-metrics-json",
        type=Path,
        default=Path("outputs/person3_full_logitlens/logit_confidence_metrics.json"),
    )
    return p.parse_args()


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device)


def _load_split(path: Path, limit: int) -> list[RawSample]:
    rows: list[RawSample] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            spans = [
                HallucinationSpan(
                    start=int(s["start"]),
                    end=int(s["end"]),
                    label=str(s.get("label", "hallucinated")),
                )
                for s in obj.get("hallucination_spans", [])
            ]
            rows.append(
                RawSample(
                    sample_id=str(obj["sample_id"]),
                    question=str(obj["question"]),
                    retrieved_context=[str(x) for x in obj.get("retrieved_context", [])],
                    answer=str(obj["answer"]),
                    hallucination_spans=spans,
                    source_id=str(obj.get("source_id", obj["sample_id"])),
                    metadata=dict(obj.get("metadata", {})),
                )
            )
            if limit and len(rows) >= limit:
                break
    rows.sort(key=lambda r: r.sample_id)
    return rows


def _compute_sample_logit_conf(
    tokenizer,
    model,
    sample: RawSample,
    split_name: str,
    max_seq_len: int,
    device: torch.device,
) -> tuple[float, int]:
    formatted = build_formatted_sample(sample, split=split_name)
    answer_text = " ".join(formatted.answer_tokens)
    full_text = formatted.prompt + answer_text

    tokenizer.truncation_side = "left"
    enc = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_seq_len,
    )
    answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]

    seq_len = int(enc["input_ids"].shape[1])
    ans_len = min(len(answer_ids), seq_len)
    ans_start = max(0, seq_len - ans_len)
    ans_end = seq_len
    if ans_end <= ans_start:
        return float("nan"), 0

    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc, return_dict=True)

    logits = out.logits[0, ans_start:ans_end, :].detach().float()
    if logits.numel() == 0:
        return float("nan"), ans_end - ans_start
    probs = torch.softmax(logits, dim=-1)
    max_conf = probs.max(dim=-1).values
    return float(max_conf.mean().item()), ans_end - ans_start


def _median_f1(scores: np.ndarray, y: np.ndarray) -> float:
    thr = np.nanmedian(scores)
    pred = (scores > thr).astype(int)
    return float(f1_score(y, pred))


def _ece_from_probs(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    probs = np.clip(np.asarray(probs, dtype=float), 0.0, 1.0)
    y = np.asarray(y, dtype=int)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs < hi if i < n_bins - 1 else probs <= hi)
        n = int(mask.sum())
        if n == 0:
            continue
        acc = float(y[mask].mean())
        conf = float(probs[mask].mean())
        ece += (n / len(y)) * abs(acc - conf)
    return float(ece)


def _metrics_from_scores(train_scores: np.ndarray, train_y: np.ndarray, test_scores: np.ndarray, test_y: np.ndarray) -> dict:
    tr = train_scores.copy()
    te = test_scores.copy()

    tr_mask = np.isfinite(tr)
    te_mask = np.isfinite(te)
    tr, tr_y = tr[tr_mask], train_y[tr_mask]
    te, te_y = te[te_mask], test_y[te_mask]

    if len(tr) < 2 or len(te) < 2 or len(np.unique(tr_y)) < 2 or len(np.unique(te_y)) < 2:
        return {
            "AUROC": float("nan"),
            "F1": float("nan"),
            "Spearman": float("nan"),
            "ECE": float("nan"),
            "train_corr": float("nan"),
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
        }

    corr = spearmanr(tr, tr_y, nan_policy="omit").correlation
    if np.isnan(corr):
        corr = 1.0
    if corr < 0:
        tr = -tr
        te = -te

    auroc = float(roc_auc_score(te_y, te))
    f1 = _median_f1(te, te_y)
    spear = float(spearmanr(te, te_y, nan_policy="omit").correlation)

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(tr, tr_y)
    probs = iso.predict(te)
    ece = _ece_from_probs(probs, te_y)

    return {
        "AUROC": auroc,
        "F1": f1,
        "Spearman": spear,
        "ECE": ece,
        "train_corr": float(corr),
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
    }


def main() -> None:
    args = parse_args()
    args.out_scores_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_metrics_json.parent.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = _resolve_device(args.device)
    print(f"Loading model={args.model_name} on device={device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    model.eval()

    all_rows: list[dict] = []
    split_scores: dict[str, list[float]] = {"train": [], "test": []}
    split_labels: dict[str, list[int]] = {"train": [], "test": []}

    for split_name in ["train", "test"]:
        split_file = args.splits_dir / f"{split_name}.jsonl"
        rows = _load_split(split_file, limit=args.limit_per_split)
        print(f"Processing {split_name}: {len(rows)} samples")

        for i, rs in enumerate(rows, start=1):
            score, used = _compute_sample_logit_conf(
                tokenizer=tokenizer,
                model=model,
                sample=rs,
                split_name=split_name,
                max_seq_len=args.max_seq_len,
                device=device,
            )
            y = 1 if len(rs.hallucination_spans) > 0 else 0
            all_rows.append(
                {
                    "sample_id": rs.sample_id,
                    "split": split_name,
                    "label": y,
                    "logit_confidence": score,
                    "answer_token_count_used": used,
                }
            )
            split_scores[split_name].append(score)
            split_labels[split_name].append(y)

            if i % 100 == 0:
                print(f"  {split_name}: {i}/{len(rows)}")

    df = pd.DataFrame(all_rows)
    df.to_csv(args.out_scores_csv, index=False)

    train_scores = np.asarray(split_scores["train"], dtype=float)
    test_scores = np.asarray(split_scores["test"], dtype=float)
    train_y = np.asarray(split_labels["train"], dtype=int)
    test_y = np.asarray(split_labels["test"], dtype=int)
    metrics = _metrics_from_scores(train_scores, train_y, test_scores, test_y)

    payload = {
        "metadata": {
            "model_name": args.model_name,
            "device": str(device),
            "max_seq_len": args.max_seq_len,
            "limit_per_split": args.limit_per_split,
            "scores_csv": str(args.out_scores_csv),
        },
        "logit_confidence_baseline": metrics,
    }
    with args.out_metrics_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved scores: {args.out_scores_csv}")
    print(f"Saved metrics: {args.out_metrics_json}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
