from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from build_exp12_table import collect_scores, load_metrics, load_split, orient


METRICS = {
    "Cosine drift": "cosine_drift",
    "Mahalanobis distance": "mahalanobis_distance",
    "Logit lens divergence": "logit_lens_divergence",
    "PCA deviation": "pca_deviation",
}

COMPOSITE_KEYS = [
    "cosine_drift",
    "mahalanobis_distance",
    "logit_lens_divergence",
    "pca_deviation",
]

DEFAULT_HALUEVAL_TASKS = ("qa", "dialogue", "summarization")


def _to_markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    return "\n".join(lines)


def _metric_auroc_from_source_orientation(
    *,
    metric_by_id: dict[str, dict],
    train_ids: list[str],
    train_y: np.ndarray,
    eval_ids: list[str],
    eval_y: np.ndarray,
    key: str,
) -> float:
    train_scores = collect_scores(metric_by_id, train_ids, key)
    eval_scores = collect_scores(metric_by_id, eval_ids, key)
    _, eval_oriented = orient(train_scores, train_y, eval_scores)
    mask = np.isfinite(eval_oriented)
    if mask.sum() < 2 or len(np.unique(eval_y[mask])) < 2:
        return float("nan")
    return float(roc_auc_score(eval_y[mask], eval_oriented[mask]))


def _feature_matrix(metric_by_id: dict[str, dict], ids: list[str]) -> np.ndarray:
    rows = []
    for sid in ids:
        row = metric_by_id.get(sid, {})
        rows.append([float(row.get(k, np.nan)) for k in COMPOSITE_KEYS])
    return np.asarray(rows, dtype=float)


def _impute_from_train(X_train: np.ndarray, *matrices: np.ndarray) -> list[np.ndarray]:
    fill = np.nanmean(X_train, axis=0)
    fill = np.where(np.isfinite(fill), fill, 0.0)
    out = []
    for X in matrices:
        Y = np.array(X, copy=True, dtype=float)
        bad = ~np.isfinite(Y)
        if bad.any():
            Y[bad] = np.take(fill, np.where(bad)[1])
        out.append(Y)
    return out


def _composite_aurocs(
    *,
    metric_by_id: dict[str, dict],
    source_train_ids: list[str],
    source_train_y: np.ndarray,
    source_val_ids: list[str],
    source_val_y: np.ndarray,
    source_test_ids: list[str],
    source_test_y: np.ndarray,
    halueval_ids: list[str],
    halueval_y: np.ndarray,
) -> tuple[float, float]:
    X_train = _feature_matrix(metric_by_id, source_train_ids)
    X_val = _feature_matrix(metric_by_id, source_val_ids)
    X_test = _feature_matrix(metric_by_id, source_test_ids)
    X_halu = _feature_matrix(metric_by_id, halueval_ids)

    for j in range(X_train.shape[1]):
        train_j, test_j = orient(X_train[:, j], source_train_y, X_test[:, j])
        _, val_j = orient(X_train[:, j], source_train_y, X_val[:, j])
        _, halu_j = orient(X_train[:, j], source_train_y, X_halu[:, j])
        X_train[:, j] = train_j
        X_val[:, j] = val_j
        X_test[:, j] = test_j
        X_halu[:, j] = halu_j

    X_tv = np.vstack([X_train, X_val])
    y_tv = np.concatenate([source_train_y, source_val_y])
    X_tv, X_test, X_halu = _impute_from_train(X_tv, X_tv, X_test, X_halu)

    scaler = StandardScaler()
    X_tv_s = scaler.fit_transform(X_tv)
    X_test_s = scaler.transform(X_test)
    X_halu_s = scaler.transform(X_halu)

    clf = LogisticRegression(max_iter=2000, random_state=42, class_weight="balanced")
    clf.fit(X_tv_s, y_tv)
    ragtruth_prob = clf.predict_proba(X_test_s)[:, 1]
    halueval_prob = clf.predict_proba(X_halu_s)[:, 1]

    return (
        float(roc_auc_score(source_test_y, ragtruth_prob)),
        float(roc_auc_score(halueval_y, halueval_prob)),
    )


def _load_halueval_split(
    path: Path,
    *,
    tasks: set[str] | None = None,
) -> tuple[list[str], np.ndarray, dict]:
    ids: list[str] = []
    ys: list[int] = []
    task_counts: dict[str, int] = {}
    skipped_task_counts: dict[str, int] = {}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            task = str(obj.get("metadata", {}).get("task_type", "unknown"))
            if tasks is not None and task not in tasks:
                skipped_task_counts[task] = skipped_task_counts.get(task, 0) + 1
                continue
            ids.append(obj["sample_id"])
            ys.append(1 if obj.get("hallucination_spans", []) else 0)
            task_counts[task] = task_counts.get(task, 0) + 1

    meta = {
        "included_tasks": sorted(task_counts),
        "task_counts": task_counts,
        "skipped_task_counts": skipped_task_counts,
    }
    return ids, np.asarray(ys, dtype=int), meta


def build_table(
    *,
    ragtruth_metrics_dir: Path,
    halueval_metrics_dir: Path,
    ragtruth_splits_dir: Path,
    halueval_split_path: Path,
    halueval_tasks: set[str] | None = set(DEFAULT_HALUEVAL_TASKS),
) -> tuple[pd.DataFrame, dict]:
    metric_by_id = {
        **load_metrics(ragtruth_metrics_dir),
        **load_metrics(halueval_metrics_dir),
    }
    source_train = load_split(ragtruth_splits_dir / "train.jsonl")
    source_val = load_split(ragtruth_splits_dir / "val.jsonl")
    source_test = load_split(ragtruth_splits_dir / "test.jsonl")
    halu_ids, halu_y, halu_meta = _load_halueval_split(
        halueval_split_path,
        tasks=halueval_tasks,
    )
    if len(halu_ids) == 0:
        raise ValueError("No HaluEval samples left after task filtering.")
    if len(np.unique(halu_y)) < 2:
        raise ValueError(
            "HaluEval evaluation needs both truthful and hallucinated samples after filtering."
        )

    rows: list[dict] = []
    for label, key in METRICS.items():
        ragtruth_auc = _metric_auroc_from_source_orientation(
            metric_by_id=metric_by_id,
            train_ids=source_train.ids,
            train_y=source_train.y,
            eval_ids=source_test.ids,
            eval_y=source_test.y,
            key=key,
        )
        halueval_auc = _metric_auroc_from_source_orientation(
            metric_by_id=metric_by_id,
            train_ids=source_train.ids,
            train_y=source_train.y,
            eval_ids=halu_ids,
            eval_y=halu_y,
            key=key,
        )
        rows.append(
            {
                "Metric": label,
                "AUROC RAGTruth": ragtruth_auc,
                "AUROC HaluEval": halueval_auc,
                "Drop": ragtruth_auc - halueval_auc,
            }
        )

    ragtruth_comp, halueval_comp = _composite_aurocs(
        metric_by_id=metric_by_id,
        source_train_ids=source_train.ids,
        source_train_y=source_train.y,
        source_val_ids=source_val.ids,
        source_val_y=source_val.y,
        source_test_ids=source_test.ids,
        source_test_y=source_test.y,
        halueval_ids=halu_ids,
        halueval_y=halu_y,
    )
    rows.append(
        {
            "Metric": "Full composite",
            "AUROC RAGTruth": ragtruth_comp,
            "AUROC HaluEval": halueval_comp,
            "Drop": ragtruth_comp - halueval_comp,
        }
    )

    df = pd.DataFrame(rows)
    metric_only = df[df["Metric"] != "Full composite"].copy()
    brittle = metric_only.sort_values("Drop", ascending=False).iloc[0].to_dict()
    meta = {
        "halueval_samples": int(len(halu_ids)),
        "halueval_positive_rate": float(np.mean(halu_y)),
        **halu_meta,
        "source_train_samples": int(len(source_train.ids)),
        "source_val_samples": int(len(source_val.ids)),
        "source_test_samples": int(len(source_test.ids)),
        "halueval_composite_auroc": float(halueval_comp),
        "most_brittle_metric": str(brittle["Metric"]),
        "most_brittle_drop": float(brittle["Drop"]),
        "spearman_halueval_composite": float(
            spearmanr(
                _feature_matrix(metric_by_id, halu_ids).mean(axis=1),
                halu_y,
                nan_policy="omit",
            ).correlation
        ),
    }
    return df, meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Experiment 5 HaluEval transfer table")
    parser.add_argument(
        "--ragtruth-metrics-dir",
        type=Path,
        default=Path("outputs/person2/metrics_full_gpt2medium_logitlens"),
    )
    parser.add_argument(
        "--halueval-metrics-dir",
        type=Path,
        default=Path("outputs/person2/metrics_halueval_gpt2medium_logitlens"),
    )
    parser.add_argument(
        "--ragtruth-splits-dir",
        type=Path,
        default=Path("artifacts/person1_ragtruth_full_gpt2medium_pt/splits"),
    )
    parser.add_argument(
        "--halueval-split-path",
        type=Path,
        default=Path("artifacts/person1_halueval_gpt2medium_pt/splits/test.jsonl"),
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("outputs/person5_halueval/experiment5_halueval_transfer_table.csv"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("outputs/person5_halueval/experiment5_halueval_transfer_table.md"),
    )
    parser.add_argument(
        "--out-summary",
        type=Path,
        default=Path("outputs/person5_halueval/experiment5_halueval_transfer_summary.json"),
    )
    parser.add_argument(
        "--halueval-tasks",
        default=",".join(DEFAULT_HALUEVAL_TASKS),
        help=(
            "Comma-separated HaluEval task types to evaluate, or 'all'. "
            "Default uses the paired binary hallucination tasks."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    halueval_tasks = None
    if args.halueval_tasks.strip().lower() != "all":
        halueval_tasks = {x.strip() for x in args.halueval_tasks.split(",") if x.strip()}
    df, meta = build_table(
        ragtruth_metrics_dir=args.ragtruth_metrics_dir,
        halueval_metrics_dir=args.halueval_metrics_dir,
        ragtruth_splits_dir=args.ragtruth_splits_dir,
        halueval_split_path=args.halueval_split_path,
        halueval_tasks=halueval_tasks,
    )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    pretty = df.copy()
    for col in ["AUROC RAGTruth", "AUROC HaluEval", "Drop"]:
        pretty[col] = pretty[col].map(lambda x: f"{x:.4f}")
    args.out_md.write_text(
        "# Experiment 5: HaluEval Zero-Shot Transfer\n\n"
        + _to_markdown_table(pretty)
        + "\n\n"
        + "No HaluEval statistics were refit. RAGTruth train/val provides feature orientation, "
        + "standardization, and logistic-composite weights; RAGTruth train provides the fitted "
        + "Mahalanobis/PCA statistics used by the metric artifacts.\n\n"
        + "Primary HaluEval evaluation uses the paired binary tasks "
        + f"({', '.join(meta['included_tasks'])}). "
        + "The general split is excluded by default because the held-out subset has no "
        + "hallucinated positives, so it is not a valid AUROC subtask.\n\n"
        + f"Most brittle metric: {meta['most_brittle_metric']} "
        + f"(drop {meta['most_brittle_drop']:.4f}).\n",
        encoding="utf-8",
    )
    args.out_summary.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved: {args.out_csv}")
    print(f"Saved: {args.out_md}")
    print(f"Saved: {args.out_summary}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
