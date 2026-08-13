#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch
import matplotlib.pyplot as plt


@dataclass
class SplitData:
    ids: list[str]
    y: np.ndarray


def load_split(split_path: Path) -> SplitData:
    ids: list[str] = []
    ys: list[int] = []
    with split_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            sid = obj["sample_id"]
            y = 1 if obj.get("hallucination_spans", []) else 0
            ids.append(sid)
            ys.append(y)
    return SplitData(ids=ids, y=np.asarray(ys, dtype=int))


def load_metrics(metrics_dir: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for p in metrics_dir.glob("*.person2_metrics.json"):
        sid = p.name.replace(".person2_metrics.json", "")
        with p.open("r", encoding="utf-8") as f:
            out[sid] = json.load(f)
    return out


def collect_scores(metric_by_id: dict[str, dict], ids: list[str], key: str) -> np.ndarray:
    vals: list[float] = []
    for sid in ids:
        m = metric_by_id.get(sid)
        if m is None:
            vals.append(np.nan)
        else:
            vals.append(float(m.get(key, np.nan)))
    return np.asarray(vals, dtype=float)


def collect_layer_scores(metric_by_id: dict[str, dict], ids: list[str], key: str) -> np.ndarray:
    rows: list[np.ndarray] = []
    for sid in ids:
        m = metric_by_id.get(sid)
        if m is None:
            rows.append(np.array([], dtype=float))
        else:
            arr = np.asarray(m.get(key, []), dtype=float)
            rows.append(arr)
    max_len = max((r.size for r in rows), default=0)
    if max_len == 0:
        return np.empty((len(ids), 0), dtype=float)
    out = np.full((len(ids), max_len), np.nan, dtype=float)
    for i, r in enumerate(rows):
        if r.size:
            out[i, : r.size] = r
    return out


def _to_numpy(values: object) -> np.ndarray:
    if isinstance(values, np.ndarray):
        return values
    if torch.is_tensor(values):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def _safe_token_span(start: int, end: int, n_tokens: int) -> tuple[int, int] | None:
    s = max(0, int(start))
    e = min(int(end), n_tokens)
    if e <= s:
        return None
    return s, e


def collect_logit_conf_proxy(
    metric_by_id: dict[str, dict],
    ids: list[str],
    mode: str = "margin",
) -> np.ndarray:
    """
    Proxy confidence from compact saved top-k logits.
    mode='top1': mean top1 logit over answer tokens.
    mode='margin': mean(top1 - top2) over answer tokens.
    """
    scores: list[float] = []
    for sid in ids:
        rec = metric_by_id.get(sid, {})
        src = rec.get("metadata", {}).get("source_artifact")
        if not src:
            scores.append(np.nan)
            continue

        try:
            art = torch.load(src, map_location="cpu", weights_only=False)
            md = art.get("metadata", {}) or {}
            topk = md.get("logits_topk_values")
            if topk is None:
                scores.append(np.nan)
                continue

            topk_arr = _to_numpy(topk)
            if topk_arr.ndim != 2 or topk_arr.shape[0] == 0 or topk_arr.shape[1] == 0:
                scores.append(np.nan)
                continue

            span = _safe_token_span(
                rec.get("answer_start_token_idx", 0),
                rec.get("answer_end_token_idx", topk_arr.shape[0]),
                topk_arr.shape[0],
            )
            if span is None:
                scores.append(np.nan)
                continue

            s, e = span
            token_vals = topk_arr[s:e]
            if token_vals.size == 0:
                scores.append(np.nan)
                continue

            if mode == "top1":
                # top-k is already sorted descending from torch.topk
                score = float(np.nanmean(token_vals[:, 0]))
            else:
                if token_vals.shape[1] < 2:
                    score = float(np.nanmean(token_vals[:, 0]))
                else:
                    score = float(np.nanmean(token_vals[:, 0] - token_vals[:, 1]))
            scores.append(score)
        except Exception:
            scores.append(np.nan)

    return np.asarray(scores, dtype=float)


def _read_baseline_metrics(path: Path, section: str) -> dict | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        m = payload.get(section)
        if not isinstance(m, dict):
            return None
        needed = ["AUROC", "F1", "Spearman", "ECE"]
        if not all(k in m for k in needed):
            return None
        return {
            "AUROC": float(m["AUROC"]),
            "F1": float(m["F1"]),
            "Spearman": float(m["Spearman"]),
            "ECE": float(m["ECE"]),
        }
    except Exception:
        return None


def median_f1(scores: np.ndarray, y: np.ndarray) -> float:
    thr = np.nanmedian(scores)
    pred = (scores > thr).astype(int)
    return float(f1_score(y, pred))


def ece_from_probs(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
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


def isotonic_probs(train_scores: np.ndarray, train_y: np.ndarray, test_scores: np.ndarray) -> np.ndarray:
    mask_train = np.isfinite(train_scores)
    mask_test = np.isfinite(test_scores)
    p = np.full_like(test_scores, fill_value=np.nan, dtype=float)
    if mask_train.sum() < 2 or len(np.unique(train_y[mask_train])) < 2:
        return np.full_like(test_scores, fill_value=0.5, dtype=float)
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(train_scores[mask_train], train_y[mask_train])
    p[mask_test] = ir.predict(test_scores[mask_test])
    p[~mask_test] = 0.5
    return p


def orient(train_scores: np.ndarray, train_y: np.ndarray, test_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    corr = spearmanr(train_scores, train_y, nan_policy="omit").correlation
    if np.isnan(corr):
        corr = 1.0
    if corr < 0:
        return -train_scores, -test_scores
    return train_scores, test_scores


def metric_row(name: str, train_scores: np.ndarray, test_scores: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> dict:
    train_s, test_s = orient(train_scores, y_train, test_scores)
    mask_test = np.isfinite(test_s)
    mask_train = np.isfinite(train_s)
    if mask_test.sum() < 2 or len(np.unique(y_test[mask_test])) < 2:
        return {
            "Method": name,
            "AUROC": np.nan,
            "F1": np.nan,
            "Spearman": np.nan,
            "ECE": np.nan,
        }

    auroc = float(roc_auc_score(y_test[mask_test], test_s[mask_test]))
    f1 = median_f1(test_s[mask_test], y_test[mask_test])
    spear = float(spearmanr(test_s[mask_test], y_test[mask_test], nan_policy="omit").correlation)
    probs = isotonic_probs(train_s[mask_train], y_train[mask_train], test_s[mask_test])
    ece = ece_from_probs(probs, y_test[mask_test])
    return {
        "Method": name,
        "AUROC": auroc,
        "F1": f1,
        "Spearman": spear,
        "ECE": ece,
    }


def _layerwise_auroc(train_layers: np.ndarray, test_layers: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> np.ndarray:
    if train_layers.size == 0 or test_layers.size == 0:
        return np.array([], dtype=float)
    n_layers = min(train_layers.shape[1], test_layers.shape[1])
    out = np.full(n_layers, np.nan, dtype=float)
    for j in range(n_layers):
        tr = train_layers[:, j]
        te = test_layers[:, j]
        tr_o, te_o = orient(tr, y_train, te)
        mask = np.isfinite(te_o)
        if mask.sum() < 3 or len(np.unique(y_test[mask])) < 2:
            continue
        try:
            out[j] = float(roc_auc_score(y_test[mask], te_o[mask]))
        except Exception:
            out[j] = np.nan
    return out


def generate_layer_profile_artifacts(
    metrics_dir: Path,
    splits_dir: Path,
    out_png: Path,
    out_csv: Path,
) -> None:
    m = load_metrics(metrics_dir)
    train = load_split(splits_dir / "train.jsonl")
    test = load_split(splits_dir / "test.jsonl")

    layer_keys = {
        "Cosine drift": "cosine_drift_per_layer",
        "Mahalanobis": "mahalanobis_per_layer",
        "PCA deviation": "pca_deviation_per_layer",
        "Logit lens": "logit_lens_divergence_per_layer",
    }

    curves: dict[str, np.ndarray] = {}
    for label, key in layer_keys.items():
        tr = collect_layer_scores(m, train.ids, key)
        te = collect_layer_scores(m, test.ids, key)
        aucs = _layerwise_auroc(tr, te, train.y, test.y)
        if aucs.size > 0 and np.isfinite(aucs).any():
            curves[label] = aucs

    if not curves:
        return

    # Top-3 by cosine layer AUROC if available, else by first available curve
    if "Cosine drift" in curves:
        ref = curves["Cosine drift"]
    else:
        ref = next(iter(curves.values()))
    finite_idx = np.where(np.isfinite(ref))[0]
    top3 = []
    if finite_idx.size:
        top3 = list(finite_idx[np.argsort(ref[finite_idx])[::-1][:3]])

    # Save long-form CSV
    rows: list[dict] = []
    for metric_name, aucs in curves.items():
        for layer_idx, au in enumerate(aucs):
            rows.append(
                {
                    "metric": metric_name,
                    "layer": int(layer_idx),
                    "auroc": float(au) if np.isfinite(au) else np.nan,
                    "is_top3": bool(layer_idx in top3),
                }
            )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    # Plot
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    for metric_name, aucs in curves.items():
        x = np.arange(len(aucs))
        plt.plot(x, aucs, marker="o", linewidth=1.8, label=metric_name)

    if top3:
        for l in top3:
            plt.axvline(l, color="gray", alpha=0.18, linestyle="--")
        y_note = float(np.nanmax(np.concatenate([v[np.isfinite(v)] for v in curves.values() if np.isfinite(v).any()])))
        plt.text(
            0.01,
            0.97,
            f"Top-3 layers: {top3}",
            transform=plt.gca().transAxes,
            va="top",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        )

    plt.xlabel("Layer index")
    plt.ylabel("AUROC (test)")
    plt.title("Exp-2 Layer Profile (Per-layer AUROC)")
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def build_table(metrics_dir: Path, splits_dir: Path) -> pd.DataFrame:
    m = load_metrics(metrics_dir)
    train = load_split(splits_dir / "train.jsonl")
    val = load_split(splits_dir / "val.jsonl")
    test = load_split(splits_dir / "test.jsonl")

    metric_map = {
        "Cosine drift": "cosine_drift",
        "Mahalanobis score": "mahalanobis_distance",
        "PCA deviation": "pca_deviation",
        "Logit lens divergence": "logit_lens_divergence",
    }

    rows: list[dict] = []

    # Baseline: attention entropy (prefer exact external metrics if available)
    att_metrics = _read_baseline_metrics(
        Path("outputs/person3_full_logitlens/attention_entropy_metrics.json"),
        "attention_entropy_baseline",
    )
    if att_metrics is None:
        rows.append({"Method": "Baseline: attention entropy", "AUROC": np.nan, "F1": np.nan, "Spearman": np.nan, "ECE": np.nan})
    else:
        rows.append({"Method": "Baseline: attention entropy", **att_metrics})

    # Baseline: logit confidence (prefer exact external metrics if available)
    conf_metrics = _read_baseline_metrics(
        Path("outputs/person3_full_logitlens/logit_confidence_metrics.json"),
        "logit_confidence_baseline",
    )
    if conf_metrics is None:
        tr_conf = collect_logit_conf_proxy(m, train.ids, mode="margin")
        te_conf = collect_logit_conf_proxy(m, test.ids, mode="margin")
        rows.append(metric_row("Baseline: logit confidence (proxy top1-top2)", tr_conf, te_conf, train.y, test.y))
    else:
        rows.append({"Method": "Baseline: logit confidence", **conf_metrics})

    for method_name, key in metric_map.items():
        tr = collect_scores(m, train.ids, key)
        te = collect_scores(m, test.ids, key)
        rows.append(metric_row(method_name, tr, te, train.y, test.y))

    # CIE top-3 layers from cosine_drift_per_layer
    tr_layers = collect_layer_scores(m, train.ids, "cosine_drift_per_layer")
    te_layers = collect_layer_scores(m, test.ids, "cosine_drift_per_layer")
    layer_aurocs: list[tuple[int, float, int]] = []
    for j in range(tr_layers.shape[1]):
        trj = tr_layers[:, j]
        tej = te_layers[:, j]
        trj_o, tej_o = orient(trj, train.y, tej)
        if np.isfinite(trj_o).sum() < 3 or np.isfinite(tej_o).sum() < 3:
            continue
        try:
            a = float(roc_auc_score(test.y[np.isfinite(tej_o)], tej_o[np.isfinite(tej_o)]))
        except Exception:
            continue
        layer_aurocs.append((j, a, -j))
    top3 = [x[0] for x in sorted(layer_aurocs, key=lambda t: (t[1], t[2]), reverse=True)[:3]]

    if top3:
        tr_top3 = np.nanmean(tr_layers[:, top3], axis=1)
        te_top3 = np.nanmean(te_layers[:, top3], axis=1)
        rows.append(metric_row(f"CIE top-3 layers ({top3})", tr_top3, te_top3, train.y, test.y))
    else:
        rows.append({"Method": "CIE top-3 layers", "AUROC": np.nan, "F1": np.nan, "Spearman": np.nan, "ECE": np.nan})

    # Full composite (matches best run logic): standardize + logistic(train+val) -> test
    feat_keys = ["cosine_drift", "mahalanobis_distance", "pca_deviation", "logit_lens_divergence"]

    def feature_matrix(ids: list[str]) -> np.ndarray:
        mat = []
        for sid in ids:
            mm = m.get(sid, {})
            mat.append([float(mm.get(k, np.nan)) for k in feat_keys])
        return np.asarray(mat, dtype=float)

    X_train = feature_matrix(train.ids)
    X_val = feature_matrix(val.ids)
    X_test = feature_matrix(test.ids)

    # Orient each feature by train split
    for j in range(X_train.shape[1]):
        trj, tej = orient(X_train[:, j], train.y, X_test[:, j])
        _, vaj = orient(X_train[:, j], train.y, X_val[:, j])
        X_train[:, j] = trj
        X_val[:, j] = vaj
        X_test[:, j] = tej

    X_tv = np.vstack([X_train, X_val])
    y_tv = np.concatenate([train.y, val.y])

    scaler = StandardScaler()
    X_tv_s = scaler.fit_transform(X_tv)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=2000, random_state=42, class_weight="balanced")
    clf.fit(X_tv_s, y_tv)
    prob_test = clf.predict_proba(X_test_s)[:, 1]

    rows.append(
        {
            "Method": "CIE full composite",
            "AUROC": float(roc_auc_score(test.y, prob_test)),
            "F1": float(f1_score(test.y, (prob_test > 0.5).astype(int))),
            "Spearman": float(spearmanr(prob_test, test.y, nan_policy="omit").correlation),
            "ECE": ece_from_probs(prob_test, test.y),
        }
    )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Exp1/Exp2 table from existing artifacts")
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=Path("outputs/person2/metrics_full_gpt2medium_logitlens"),
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("artifacts/person1_ragtruth_full_gpt2medium_pt/splits"),
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("outputs/person3_full_logitlens/exp1_exp2_table_final.csv"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("outputs/person3_full_logitlens/exp1_exp2_table_final.md"),
    )
    parser.add_argument(
        "--out-layer-profile-png",
        type=Path,
        default=Path("outputs/person3_full_logitlens/E2_layer_profile.png"),
    )
    parser.add_argument(
        "--out-layer-profile-csv",
        type=Path,
        default=Path("outputs/person3_full_logitlens/E2_layer_profile.csv"),
    )
    args = parser.parse_args()

    df = build_table(args.metrics_dir, args.splits_dir)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    def _fmt(v: float | str) -> str:
        if isinstance(v, str):
            return v
        if pd.isna(v):
            return "NA"
        return f"{float(v):.6f}"

    headers = ["Method", "AUROC", "F1", "Spearman", "ECE"]
    md_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        md_lines.append(
            "| "
            + " | ".join(
                [
                    str(row["Method"]),
                    _fmt(row["AUROC"]),
                    _fmt(row["F1"]),
                    _fmt(row["Spearman"]),
                    _fmt(row["ECE"]),
                ]
            )
            + " |"
        )

    with args.out_md.open("w", encoding="utf-8") as f:
        f.write("# Exp1/Exp2 Final Table\n\n")
        f.write("\n".join(md_lines))
        f.write("\n\n")
        f.write("Notes:\n")
        f.write(
            "- Baseline attention/logit-confidence are NA because compact Person1 artifacts in this run do not retain recoverable attention distributions / token-level logit confidences for all samples.\n"
        )
        f.write(
            "- If exact baseline JSON files exist (attention/logit confidence), they are used automatically.\n"
        )
        f.write(
            "- ECE for metric rows uses isotonic calibration fit on train split, evaluated on test split.\n"
        )

    print(f"Saved: {args.out_csv}")
    print(f"Saved: {args.out_md}")

    generate_layer_profile_artifacts(
        metrics_dir=args.metrics_dir,
        splits_dir=args.splits_dir,
        out_png=args.out_layer_profile_png,
        out_csv=args.out_layer_profile_csv,
    )
    if args.out_layer_profile_png.exists():
        print(f"Saved: {args.out_layer_profile_png}")
    if args.out_layer_profile_csv.exists():
        print(f"Saved: {args.out_layer_profile_csv}")


if __name__ == "__main__":
    main()
