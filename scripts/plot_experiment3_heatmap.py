#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot Exp-3 component-to-layer heatmap from activation patching summary."
    )
    p.add_argument(
        "--summary-json",
        type=Path,
        default=Path("outputs/person3_exp3_full/experiment3_activation_patching_summary.json"),
    )
    p.add_argument(
        "--details-jsonl",
        type=Path,
        default=Path("outputs/person3_exp3_full/experiment3_activation_patching_details.jsonl"),
    )
    p.add_argument(
        "--out-png",
        type=Path,
        default=Path("outputs/person3_exp3_full/E3_component_layer_heatmap.png"),
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=Path("outputs/person3_exp3_full/E3_component_layer_heatmap.csv"),
    )
    return p.parse_args()


def _infer_num_layers(details_jsonl: Path) -> int:
    with details_jsonl.open("r", encoding="utf-8") as f:
        first = json.loads(next(f))
    sample_path = Path(first["recipient_path"])
    obj = torch.load(sample_path, map_location="cpu", weights_only=False)
    hs = obj.get("hidden_states")
    if torch.is_tensor(hs):
        if hs.dim() == 3:
            return int(hs.shape[0])
        if hs.dim() == 4:
            return int(hs.shape[1])
        if hs.dim() == 2:
            return 1
    if isinstance(hs, list):
        if len(hs) == 0:
            return 0
        # Common: [layers][tokens][hidden]
        if isinstance(hs[0], (list, tuple)):
            return int(len(hs))
        return 1
    # fallback for compact artifacts that keep layers in metadata
    md = obj.get("metadata", {}) or {}
    if "hidden_states_last_n_layers" in md:
        v = md["hidden_states_last_n_layers"]
        if torch.is_tensor(v):
            if v.dim() >= 1:
                return int(v.shape[0])
        if isinstance(v, list):
            return int(len(v))
    raise ValueError("Could not infer number of layers from recipient artifact")


def _component_ranges(num_layers: int) -> dict[str, tuple[int, int]]:
    q = max(1, math.ceil(num_layers * 0.25))
    return {
        "early_attn_heads": (0, min(num_layers, q)),
        "mid_ffn_layers": (max(0, math.floor(num_layers * 0.25)), min(num_layers, math.ceil(num_layers * 0.75))),
        "late_ffn_layers": (max(0, math.floor(num_layers * 0.75)), num_layers),
        "copying_heads": (max(0, num_layers - q), num_layers),
    }


def main() -> None:
    args = parse_args()
    with args.summary_json.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    num_layers = _infer_num_layers(args.details_jsonl)
    ranges = _component_ranges(num_layers)

    component_order = [
        "early_attn_heads",
        "mid_ffn_layers",
        "late_ffn_layers",
        "copying_heads",
    ]

    label_by_key = {
        c["component_key"]: c["component"] for c in summary.get("component_results", [])
    }
    cie_by_key = {
        c["component_key"]: float(c.get("cie_faithful_to_hallucinated", np.nan))
        for c in summary.get("component_results", [])
    }

    mat = np.full((len(component_order), num_layers), np.nan, dtype=float)
    for r, key in enumerate(component_order):
        if key not in ranges:
            continue
        start, end = ranges[key]
        val = cie_by_key.get(key, np.nan)
        mat[r, start:end] = val

    # Save matrix CSV
    rows = []
    for r, key in enumerate(component_order):
        for l in range(num_layers):
            rows.append(
                {
                    "component_key": key,
                    "component": label_by_key.get(key, key),
                    "layer": l,
                    "cie_faithful_to_hallucinated": mat[r, l],
                }
            )
    df = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    # Plot
    plt.figure(figsize=(11, 4.8))
    cmap = plt.cm.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="#f0f0f0")
    vmax = np.nanmax(np.abs(mat)) if np.isfinite(mat).any() else 1.0
    im = plt.imshow(mat, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)

    ylabels = [label_by_key.get(k, k) for k in component_order]
    plt.yticks(np.arange(len(ylabels)), ylabels)

    xticks = list(range(0, num_layers, 4))
    if (num_layers - 1) not in xticks:
        xticks.append(num_layers - 1)
    plt.xticks(xticks, xticks)
    plt.xlabel("transformer layer")
    plt.ylabel("component")
    plt.title("Experiment 3 — mean Δ risk from faithful → hallucinated patching")
    cbar = plt.colorbar(im)
    cbar.set_label("Δ risk (CIE)")
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=180)
    plt.close()

    print(f"Saved: {args.out_png}")
    print(f"Saved: {args.out_csv}")


if __name__ == "__main__":
    main()
