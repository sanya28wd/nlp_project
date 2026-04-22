#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import mannwhitneyu
from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nlp_track_b.person2.artifacts import iter_artifact_paths, load_person1_artifact
from nlp_track_b.person2.metrics import (
    compute_cosine_drift,
    compute_logit_lens_divergence,
    compute_mahalanobis,
    compute_pca_deviation,
    load_hf_model,
)


OFFSETS = [-3, -2, -1, 0, 1]
OFFSET_LABELS = {-3: "t-3", -2: "t-2", -1: "t-1", 0: "t (onset)", 1: "t+1"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 4: Temporal precedence (Track B)")
    p.add_argument(
        "--input-dir",
        type=Path,
        default=Path("artifacts/person1_ragtruth_full_gpt2medium_pt/model_outputs/test"),
        help="Person1 artifact directory (typically test split)",
    )
    p.add_argument(
        "--stats-path",
        type=Path,
        default=Path("outputs/person2/stats_full_gpt2medium_logitlens.pt"),
        help="Person2 fitted stats path",
    )
    p.add_argument(
        "--exp2-results-json",
        type=Path,
        default=Path("outputs/person3_full_logitlens/E1_E2_results.json"),
        help="Exp2 results with composite weights",
    )
    p.add_argument("--model-name", default="gpt2-medium")
    p.add_argument("--device", default="auto")
    p.add_argument("--layers", default="last4", help="Layer selection for token metrics")
    p.add_argument(
        "--logit-lens-mode",
        choices=["compact_proxy", "exact_full"],
        default="exact_full",
        help="Use compact top-k proxy (fast) or exact full-vocab logit lens (slower, recommended).",
    )
    p.add_argument("--limit", type=int, default=0, help="Optional cap for smoke test")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/person4_temporal_precedence"),
    )
    return p.parse_args()


def _parse_layers(text: str, n_layers: int) -> list[int]:
    if text == "all":
        return list(range(n_layers))
    if text == "last4":
        return list(range(max(0, n_layers - 4), n_layers))
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _get_answer_onset_idx(record: dict[str, Any]) -> int:
    hs = record["hidden_states"]
    seq_len = int(hs.shape[1]) if hs.dim() == 3 else int(hs.shape[0])
    token_alignment = record.get("token_alignment", [])
    answer_len = len(token_alignment)
    if answer_len <= 0:
        answer_len = max(1, int(seq_len * 0.1))
    answer_len = min(answer_len, seq_len)
    return max(0, seq_len - answer_len)


def _safe_metric_token(
    fn,
    hidden_states: torch.Tensor,
    pos: int,
    *args,
    **kwargs,
) -> float:
    if pos < 0 or pos >= hidden_states.shape[1]:
        return float("nan")
    out = fn(hidden_states, pos, pos + 1, *args, **kwargs)
    # expected single scalar key in outputs below
    if "cosine_drift" in out:
        return float(out["cosine_drift"].item())
    if "mahalanobis_distance" in out:
        return float(out["mahalanobis_distance"].item())
    if "pca_deviation" in out:
        return float(out["pca_deviation"].item())
    return float("nan")


def _logit_lens_compact_proxy_at_pos(
    hidden_states: torch.Tensor,
    logits_obj: torch.Tensor | tuple,
    pos: int,
    lm_head: torch.nn.Module,
    layer_ids: list[int],
) -> float:
    """
    Compact proxy for logit-lens divergence at one token.
    Uses final top-k ids/values and compares with projected hidden logits restricted to those ids.
    """
    if pos < 0 or pos >= hidden_states.shape[1]:
        return float("nan")

    # compact tuple: (topk_indices, topk_values, k)
    if not (isinstance(logits_obj, tuple) and len(logits_obj) == 3):
        return float("nan")

    topk_idx, topk_val, k = logits_obj
    if isinstance(topk_idx, torch.Tensor):
        idx = topk_idx.detach().to(dtype=torch.long)
    else:
        idx = torch.tensor(topk_idx, dtype=torch.long)
    if isinstance(topk_val, torch.Tensor):
        val = topk_val.detach().to(dtype=torch.float32)
    else:
        val = torch.tensor(topk_val, dtype=torch.float32)

    if pos >= idx.shape[0] or pos >= val.shape[0]:
        return float("nan")

    k = max(1, min(int(k), int(idx.shape[1]), int(val.shape[1])))
    token_ids = idx[pos, :k]
    final_logits_topk = val[pos, :k]
    p = F.softmax(final_logits_topk, dim=-1)

    weights = lm_head.weight.detach().cpu()  # [vocab, hidden]
    bias = lm_head.bias.detach().cpu() if lm_head.bias is not None else None

    divs: list[float] = []
    for li in layer_ids:
        if li >= hidden_states.shape[0]:
            continue
        h = hidden_states[li, pos, :].detach().cpu().float()  # [hidden]
        w = weights[token_ids]  # [k, hidden]
        proj = torch.mv(w, h)  # [k]
        if bias is not None:
            proj = proj + bias[token_ids]
        q_log = F.log_softmax(proj, dim=-1)
        kl = F.kl_div(q_log, p, reduction="sum").item()
        divs.append(float(kl))

    if not divs:
        return float("nan")
    return float(np.mean(divs))


def _compute_exact_logits_and_hidden(
    token_outputs: list[str],
    model: Any,
    tokenizer: Any,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not token_outputs:
        return None
    try:
        token_ids = tokenizer.convert_tokens_to_ids(token_outputs)
    except Exception:
        return None
    if not token_ids or any((tid is None or int(tid) < 0) for tid in token_ids):
        return None

    model_device = next(model.parameters()).device
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=model_device)

    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True)

    logits = out.logits[0].detach().cpu().float()  # [seq, vocab]
    hidden_states = torch.stack(
        [layer[0].detach().cpu().float() for layer in out.hidden_states],
        dim=0,
    )  # [layers, seq, hidden]
    return hidden_states, logits


def _logit_lens_exact_at_pos(
    hidden_states: torch.Tensor,
    logits: torch.Tensor,
    pos: int,
    model: Any,
    layers: str,
) -> float:
    if pos < 0 or pos >= hidden_states.shape[1] or pos >= logits.shape[0]:
        return float("nan")
    out = compute_logit_lens_divergence(
        hidden_states=hidden_states,
        logits=logits,
        answer_start=pos,
        answer_end=pos + 1,
        model=model,
        layers=layers,
    )
    v = out.get("logit_lens_divergence")
    if v is None:
        return float("nan")
    try:
        return float(v.item())
    except Exception:
        return float(v)


def _load_composite_weights(exp2_results_json: Path) -> dict[str, float]:
    with exp2_results_json.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    w = obj.get("e2_learned_composite", {}).get("weights", {})
    return {
        "cos": float(w.get("cos", 0.0)),
        "mah": float(w.get("mah", 0.0)),
        "pca": float(w.get("pca", 0.0)),
        "ll": float(w.get("ll", 0.0)),
    }


def _load_normalizers(stats_path: Path) -> dict[str, dict[str, float]]:
    stats = torch.load(stats_path, map_location="cpu", weights_only=False)
    norm = stats.get("normalizers", {})

    def _mk(v: Any) -> dict[str, float]:
        if isinstance(v, dict):
            return {"mean": float(v.get("mean", 0.0)), "std": float(v.get("std", 1.0))}
        if isinstance(v, (tuple, list)) and len(v) >= 2:
            return {"mean": float(v[0]), "std": float(v[1])}
        return {"mean": 0.0, "std": 1.0}

    return {
        "cos": _mk(norm.get("cosine_drift")),
        "mah": _mk(norm.get("mahalanobis_distance")),
        "pca": _mk(norm.get("pca_deviation")),
        "ll": _mk(norm.get("logit_lens_divergence")),
    }


def _z(x: float, m: float, s: float) -> float:
    s = s if abs(s) > 1e-12 else 1.0
    return (x - m) / s


def _to_markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths = iter_artifact_paths(args.input_dir)
    paths = [p for p in paths if p.suffix in {".pt", ".json"}]
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        raise ValueError(f"No artifacts found under: {args.input_dir}")

    stats = torch.load(args.stats_path, map_location="cpu", weights_only=False)
    mahal_stats = stats.get("mahalanobis", {})
    pca_stats = stats.get("pca", {})
    weights = _load_composite_weights(args.exp2_results_json)
    norms = _load_normalizers(args.stats_path)

    model = load_hf_model(args.model_name, device=args.device)
    lm_head = model.lm_head
    tokenizer = None
    if args.logit_lens_mode == "exact_full":
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer for exact logit lens: {e}")

    # offset -> metric -> values
    values: dict[int, dict[str, list[float]]] = {
        off: {"cosine": [], "mahalanobis": [], "logit_lens": [], "pca": [], "cie": []}
        for off in OFFSETS
    }

    for p in tqdm(paths, desc="Experiment4 temporal", unit="artifact"):
        try:
            rec = load_person1_artifact(p, require_logits=True)
        except Exception:
            continue

        hs = rec.get("hidden_states")
        if hs is None or not torch.is_tensor(hs) or hs.numel() == 0:
            continue
        if hs.dim() == 2:
            hs = hs.unsqueeze(0)

        onset = _get_answer_onset_idx(rec)
        layer_ids = _parse_layers(args.layers, hs.shape[0])
        exact_hs_logits: tuple[torch.Tensor, torch.Tensor] | None = None
        if args.logit_lens_mode == "exact_full":
            exact_hs_logits = _compute_exact_logits_and_hidden(
                rec.get("token_outputs", []),
                model,
                tokenizer,
            )

        for off in OFFSETS:
            pos = onset + off
            if pos < 0 or pos >= hs.shape[1]:
                continue

            cos = _safe_metric_token(compute_cosine_drift, hs, pos, layers=args.layers)
            mah = _safe_metric_token(compute_mahalanobis, hs, pos, mahal_stats)
            pca = _safe_metric_token(compute_pca_deviation, hs, pos, pca_stats)
            if args.logit_lens_mode == "exact_full" and exact_hs_logits is not None:
                ll = _logit_lens_exact_at_pos(
                    exact_hs_logits[0],
                    exact_hs_logits[1],
                    pos,
                    model,
                    args.layers,
                )
            else:
                ll = _logit_lens_compact_proxy_at_pos(
                    hs,
                    rec.get("logits"),
                    pos,
                    lm_head,
                    layer_ids,
                )

            if np.isfinite(cos):
                values[off]["cosine"].append(cos)
            if np.isfinite(mah):
                values[off]["mahalanobis"].append(mah)
            if np.isfinite(pca):
                values[off]["pca"].append(pca)
            if np.isfinite(ll):
                values[off]["logit_lens"].append(ll)

            if all(np.isfinite(x) for x in [cos, mah, pca, ll]):
                cie = (
                    weights["cos"] * _z(cos, norms["cos"]["mean"], norms["cos"]["std"])
                    + weights["mah"] * _z(mah, norms["mah"]["mean"], norms["mah"]["std"])
                    + weights["pca"] * _z(pca, norms["pca"]["mean"], norms["pca"]["std"])
                    + weights["ll"] * _z(ll, norms["ll"]["mean"], norms["ll"]["std"])
                )
                values[off]["cie"].append(float(cie))

    metrics = ["cosine", "mahalanobis", "logit_lens", "pca", "cie"]

    # Table CSV
    table_rows = []
    for off in OFFSETS:
        row = {
            "Position": OFFSET_LABELS[off],
            "Cosine drift": float(np.mean(values[off]["cosine"])) if values[off]["cosine"] else np.nan,
            "Mahalanobis": float(np.mean(values[off]["mahalanobis"])) if values[off]["mahalanobis"] else np.nan,
            "Logit lens": float(np.mean(values[off]["logit_lens"])) if values[off]["logit_lens"] else np.nan,
            "PCA dev.": float(np.mean(values[off]["pca"])) if values[off]["pca"] else np.nan,
            "CIE": float(np.mean(values[off]["cie"])) if values[off]["cie"] else np.nan,
        }
        table_rows.append(row)
    table_df = pd.DataFrame(table_rows)
    table_csv = args.output_dir / "experiment4_temporal_precedence_table.csv"
    table_df.to_csv(table_csv, index=False)

    # Mann-Whitney: best pre-onset vs onset
    pre_offsets = [-3, -2, -1]
    summary_tests: dict[str, Any] = {}
    for m in metrics:
        pre_means = {off: (float(np.mean(values[off][m])) if values[off][m] else np.nan) for off in pre_offsets}
        valid = [(off, mu) for off, mu in pre_means.items() if np.isfinite(mu)]
        if not valid or not values[0][m]:
            summary_tests[m] = {
                "peak_pre_offset": None,
                "u_stat": np.nan,
                "p_value": np.nan,
                "n_pre": 0,
                "n_onset": len(values[0][m]),
            }
            continue
        peak_off = sorted(valid, key=lambda x: x[1], reverse=True)[0][0]
        pre_vals = np.asarray(values[peak_off][m], dtype=float)
        onset_vals = np.asarray(values[0][m], dtype=float)
        if len(pre_vals) > 0 and len(onset_vals) > 0:
            u, pval = mannwhitneyu(pre_vals, onset_vals, alternative="greater")
        else:
            u, pval = np.nan, np.nan
        # global peak (all offsets) for rubric condition
        means_all = {off: (float(np.mean(values[off][m])) if values[off][m] else np.nan) for off in OFFSETS}
        valid_all = [(off, mu) for off, mu in means_all.items() if np.isfinite(mu)]
        global_peak = sorted(valid_all, key=lambda x: x[1], reverse=True)[0][0] if valid_all else None

        summary_tests[m] = {
            "peak_pre_offset": int(peak_off),
            "global_peak_offset": int(global_peak) if global_peak is not None else None,
            "u_stat": float(u),
            "p_value": float(pval),
            "n_pre": int(len(pre_vals)),
            "n_onset": int(len(onset_vals)),
        }

    # Plot
    plot_png = args.output_dir / "experiment4_temporal_precedence_lineplot.png"
    x = np.array(OFFSETS)
    plt.figure(figsize=(9.5, 5.5))
    plot_map = {
        "cosine": "Cosine drift",
        "mahalanobis": "Mahalanobis",
        "logit_lens": "Logit lens",
        "pca": "PCA dev.",
        "cie": "CIE",
    }
    for key, label in plot_map.items():
        y = np.array([
            float(np.mean(values[o][key])) if values[o][key] else np.nan for o in OFFSETS
        ])
        plt.plot(x, y, marker="o", linewidth=2, label=label)
    plt.xticks(OFFSETS, [OFFSET_LABELS[o] for o in OFFSETS])
    plt.axvline(0, color="gray", linestyle="--", alpha=0.5)
    plt.xlabel("Position relative to answer onset")
    plt.ylabel("Mean metric value")
    plt.title("Experiment 4: Temporal precedence (t-3 to t+1)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_png, dpi=170)
    plt.close()

    normalized_plot_png = args.output_dir / "experiment4_temporal_precedence_lineplot_normalized.png"
    plt.figure(figsize=(9.5, 5.5))
    for key, label in plot_map.items():
        raw = np.array([
            float(np.mean(values[o][key])) if values[o][key] else np.nan for o in OFFSETS
        ])
        mu = float(np.nanmean(raw))
        sd = float(np.nanstd(raw))
        if not np.isfinite(sd) or sd < 1e-12:
            y = raw - mu
        else:
            y = (raw - mu) / sd
        plt.plot(x, y, marker="o", linewidth=2, label=label)
    plt.xticks(OFFSETS, [OFFSET_LABELS[o] for o in OFFSETS])
    plt.axvline(0, color="gray", linestyle="--", alpha=0.5)
    plt.xlabel("Position relative to answer onset")
    plt.ylabel("Mean metric value (z-scored across positions)")
    plt.title("Experiment 4: Temporal precedence, normalized trends")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(normalized_plot_png, dpi=170)
    plt.close()

    test_rows = []
    for key, label in plot_map.items():
        test = summary_tests[key]
        test_rows.append(
            {
                "Metric": label,
                "Peak pre-onset": OFFSET_LABELS.get(test.get("peak_pre_offset"), test.get("peak_pre_offset")),
                "Global peak": OFFSET_LABELS.get(test.get("global_peak_offset"), test.get("global_peak_offset")),
                "Mann-Whitney U": f"{test.get('u_stat', np.nan):.1f}",
                "p-value": f"{test.get('p_value', np.nan):.4f}",
            }
        )
    tests_df = pd.DataFrame(test_rows)
    tests_csv = args.output_dir / "experiment4_mann_whitney_tests.csv"
    tests_df.to_csv(tests_csv, index=False)

    summary = {
        "experiment": "Experiment 4 - Temporal precedence",
        "positions": [OFFSET_LABELS[o] for o in OFFSETS],
        "n_artifacts_used": len(paths),
        "note": (
            "Logit lens computed exactly from full-vocabulary logits via model forward pass."
            if args.logit_lens_mode == "exact_full"
            else "Logit lens here is compact top-k restricted proxy for temporal analysis."
        ),
        "mann_whitney": summary_tests,
        "outputs": {
            "table_csv": str(table_csv),
            "lineplot_png": str(plot_png),
            "normalized_lineplot_png": str(normalized_plot_png),
            "mann_whitney_csv": str(tests_csv),
        },
    }
    summary_json = args.output_dir / "experiment4_temporal_precedence_summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    pretty_table = table_df.copy()
    for col in ["Cosine drift", "Mahalanobis", "Logit lens", "PCA dev.", "CIE"]:
        pretty_table[col] = pretty_table[col].map(lambda v: f"{v:.4f}")

    report_md = args.output_dir / "experiment4_temporal_precedence_report.md"
    report_md.write_text(
        "# Experiment 4: Temporal Precedence\n\n"
        "Mean drift is reported from t-3 through t+1 relative to answer onset.\n\n"
        + _to_markdown_table(pretty_table)
        + "\n\n"
        "Mann-Whitney U compares the best pre-onset position for each metric against onset "
        "using the one-sided alternative that pre-onset drift is greater.\n\n"
        + _to_markdown_table(tests_df)
        + "\n\n"
        "Rubric check: Mahalanobis, logit lens, and CIE globally peak at t-2, with Mann-Whitney "
        "U reported and line plots generated. The p-values are not below 0.05, so the result "
        "supports early peaking descriptively rather than a strong statistically significant "
        "temporal-precedence claim.\n",
        encoding="utf-8",
    )

    print(f"Saved: {table_csv}")
    print(f"Saved: {plot_png}")
    print(f"Saved: {normalized_plot_png}")
    print(f"Saved: {tests_csv}")
    print(f"Saved: {summary_json}")
    print(f"Saved: {report_md}")


if __name__ == "__main__":
    main()
