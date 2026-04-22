from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from nlp_track_b.person2.artifacts import iter_artifact_paths, load_person1_artifact
from nlp_track_b.person2.metrics import compute_cosine_drift


@dataclass
class Candidate:
    path: Path
    is_hallucinated: bool


@dataclass
class ComponentSpec:
    key: str
    label: str
    layer_start: int
    layer_end: int
    dim_start: int | None = None
    dim_end: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 3: Causal intervention via activation patching"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Person 1 artifact directory (json/pt)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/person3_exp3"),
        help="Directory for experiment outputs",
    )
    parser.add_argument(
        "--pairs-per-component",
        type=int,
        default=25,
        help="Number of random pairings per component per direction",
    )
    parser.add_argument(
        "--max-candidates-per-class",
        type=int,
        default=400,
        help="Maximum faithful/hallucinated candidates loaded",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--save-details",
        action="store_true",
        help="Save per-run deltas for all patching trials",
    )
    return parser.parse_args()


def _is_hallucinated(record: dict[str, Any]) -> bool:
    token_alignment = record.get("token_alignment", [])
    return any(bool(tok.get("is_hallucinated", False)) for tok in token_alignment)


def _build_candidates(
    paths: list[Path],
    max_per_class: int,
) -> tuple[list[Candidate], list[Candidate]]:
    faithful: list[Candidate] = []
    hallucinated: list[Candidate] = []

    for path in paths:
        if len(faithful) >= max_per_class and len(hallucinated) >= max_per_class:
            break
        try:
            record = load_person1_artifact(path, require_logits=False)
        except Exception:
            continue

        is_hall = _is_hallucinated(record)
        if is_hall and len(hallucinated) < max_per_class:
            hallucinated.append(Candidate(path=path, is_hallucinated=True))
        elif (not is_hall) and len(faithful) < max_per_class:
            faithful.append(Candidate(path=path, is_hallucinated=False))

    return faithful, hallucinated


def _mean_layer_rep(hidden_states: torch.Tensor) -> torch.Tensor:
    if hidden_states.dim() == 2:
        hidden_states = hidden_states.unsqueeze(0)
    return hidden_states.mean(dim=1)


def _build_class_prototypes(
    faithful_pool: list[Candidate],
    hall_pool: list[Candidate],
    max_records: int = 80,
) -> dict[str, torch.Tensor]:
    faithful_reps: list[torch.Tensor] = []
    hall_reps: list[torch.Tensor] = []

    for cand in faithful_pool[:max_records]:
        try:
            rec = load_person1_artifact(cand.path, require_logits=False)
        except Exception:
            continue
        hs = rec.get("hidden_states")
        if hs is None or hs.numel() == 0:
            continue
        faithful_reps.append(_mean_layer_rep(hs))

    for cand in hall_pool[:max_records]:
        try:
            rec = load_person1_artifact(cand.path, require_logits=False)
        except Exception:
            continue
        hs = rec.get("hidden_states")
        if hs is None or hs.numel() == 0:
            continue
        hall_reps.append(_mean_layer_rep(hs))

    if not faithful_reps or not hall_reps:
        return {}

    faithful_tensor = torch.stack(faithful_reps, dim=0).mean(dim=0)
    hall_tensor = torch.stack(hall_reps, dim=0).mean(dim=0)
    return {
        "faithful": faithful_tensor,
        "hallucinated": hall_tensor,
    }


def _risk_score(
    record: dict[str, Any],
    layers: list[int] | str = "all",
    prototypes: dict[str, torch.Tensor] | None = None,
) -> float:
    if prototypes:
        hs = record.get("hidden_states")
        if hs is not None and hs.numel() > 0:
            rep = _mean_layer_rep(hs)
            proto_f = prototypes["faithful"]
            proto_h = prototypes["hallucinated"]

            if isinstance(layers, str):
                layer_ids = list(range(min(rep.shape[0], proto_f.shape[0], proto_h.shape[0])))
            else:
                layer_ids = [
                    i
                    for i in layers
                    if i < rep.shape[0] and i < proto_f.shape[0] and i < proto_h.shape[0]
                ]
            if not layer_ids:
                layer_ids = list(range(min(rep.shape[0], proto_f.shape[0], proto_h.shape[0])))

            scores = []
            for li in layer_ids:
                r = rep[li]
                s_h = F.cosine_similarity(r.unsqueeze(0), proto_h[li].unsqueeze(0), dim=1).item()
                s_f = F.cosine_similarity(r.unsqueeze(0), proto_f[li].unsqueeze(0), dim=1).item()
                scores.append(s_h - s_f)
            if scores:
                return float(sum(scores) / len(scores))

    answer_start = int(record["answer_start_token_idx"])
    answer_end = int(record["answer_end_token_idx"])
    metric = compute_cosine_drift(
        record["hidden_states"],
        answer_start=answer_start,
        answer_end=answer_end,
        layers=layers,
    )
    return float(metric["cosine_drift"].item())


def _component_specs(num_layers: int, hidden_dim: int) -> list[ComponentSpec]:
    q_layers = max(1, math.ceil(num_layers * 0.25))
    mid_start = max(0, math.floor(num_layers * 0.25))
    mid_end = max(mid_start + 1, math.ceil(num_layers * 0.75))
    late_start = max(0, math.floor(num_layers * 0.75))

    q_dim = max(1, hidden_dim // 4)
    return [
        ComponentSpec(
            key="early_attn_heads",
            label="Early attn heads (1–25%)",
            layer_start=0,
            layer_end=min(num_layers, q_layers),
            dim_start=0,
            dim_end=min(hidden_dim, q_dim),
        ),
        ComponentSpec(
            key="mid_ffn_layers",
            label="Mid FFN layers (26–75%)",
            layer_start=mid_start,
            layer_end=min(num_layers, mid_end),
            dim_start=None,
            dim_end=None,
        ),
        ComponentSpec(
            key="late_ffn_layers",
            label="Late FFN layers (76–100%)",
            layer_start=late_start,
            layer_end=num_layers,
            dim_start=None,
            dim_end=None,
        ),
        ComponentSpec(
            key="copying_heads",
            label="Copying heads (last 25%)",
            layer_start=max(0, num_layers - q_layers),
            layer_end=num_layers,
            dim_start=max(0, hidden_dim - q_dim),
            dim_end=hidden_dim,
        ),
    ]


def _apply_patch(
    recipient_hidden: torch.Tensor,
    donor_hidden: torch.Tensor,
    component: ComponentSpec,
) -> torch.Tensor:
    patched = recipient_hidden.clone()

    if patched.dim() == 2:
        patched = patched.unsqueeze(0)
    if donor_hidden.dim() == 2:
        donor_hidden = donor_hidden.unsqueeze(0)

    max_layers = min(patched.shape[0], donor_hidden.shape[0], component.layer_end)
    layer_start = min(component.layer_start, max_layers)
    layer_end = max(layer_start, max_layers)

    seq_len = min(patched.shape[1], donor_hidden.shape[1])
    if seq_len <= 0 or layer_end <= layer_start:
        return patched

    if component.dim_start is None or component.dim_end is None:
        patched[layer_start:layer_end, :seq_len, :] = donor_hidden[layer_start:layer_end, :seq_len, :]
    else:
        dim_end = min(component.dim_end, patched.shape[2], donor_hidden.shape[2])
        dim_start = min(component.dim_start, dim_end)
        if dim_end > dim_start:
            patched[layer_start:layer_end, :seq_len, dim_start:dim_end] = donor_hidden[
                layer_start:layer_end, :seq_len, dim_start:dim_end
            ]
    return patched


def _paired_permutation_p_value(deltas: list[float], n_perm: int = 2000, seed: int = 0) -> float:
    if not deltas:
        return 1.0
    observed = statistics.mean(deltas)
    if observed <= 0.0:
        return 1.0

    rng = random.Random(seed)
    hits = 0
    for _ in range(n_perm):
        signed = [d if rng.random() < 0.5 else -d for d in deltas]
        if statistics.mean(signed) >= observed:
            hits += 1
    return (hits + 1) / (n_perm + 1)


def _run_direction(
    recipient_pool: list[Candidate],
    donor_pool: list[Candidate],
    components: list[ComponentSpec],
    pairs_per_component: int,
    seed: int,
    direction_name: str,
    prototypes: dict[str, torch.Tensor] | None = None,
) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    results: dict[str, dict[str, float]] = {}
    details: list[dict[str, Any]] = []

    for component in components:
        deltas: list[float] = []

        for _ in range(pairs_per_component):
            recipient = rng.choice(recipient_pool)
            donor = rng.choice(donor_pool)
            try:
                rec_record = load_person1_artifact(recipient.path, require_logits=False)
                donor_record = load_person1_artifact(donor.path, require_logits=False)
            except Exception:
                continue

            rec_hidden = rec_record["hidden_states"]
            donor_hidden = donor_record["hidden_states"]
            if rec_hidden.numel() == 0 or donor_hidden.numel() == 0:
                continue

            layer_subset = list(range(component.layer_start, component.layer_end))
            if not layer_subset:
                layer_subset = "all"

            baseline_score = _risk_score(rec_record, layers=layer_subset, prototypes=prototypes)
            patched_hidden = _apply_patch(rec_hidden, donor_hidden, component)

            patched_record = dict(rec_record)
            patched_record["hidden_states"] = patched_hidden
            patched_score = _risk_score(patched_record, layers=layer_subset, prototypes=prototypes)

            if direction_name == "faithful_to_hallucinated":
                delta = patched_score - baseline_score
            else:
                delta = baseline_score - patched_score

            deltas.append(float(delta))
            details.append(
                {
                    "direction": direction_name,
                    "component": component.key,
                    "recipient_path": str(recipient.path),
                    "donor_path": str(donor.path),
                    "baseline_risk": baseline_score,
                    "patched_risk": patched_score,
                    "cie_delta": float(delta),
                }
            )

        mean_delta = float(statistics.mean(deltas)) if deltas else 0.0
        std_delta = float(statistics.pstdev(deltas)) if len(deltas) > 1 else 0.0
        p_value = _paired_permutation_p_value(deltas, seed=seed + hash(component.key) % 10000)

        results[component.key] = {
            "mean_cie": mean_delta,
            "std_cie": std_delta,
            "p_value": float(p_value),
            "n": len(deltas),
        }

    return results, details


def _critical_flag(f2h: dict[str, float], h2f: dict[str, float], alpha: float = 0.05) -> bool:
    return (
        f2h.get("mean_cie", 0.0) > 0
        and h2f.get("mean_cie", 0.0) > 0
        and f2h.get("p_value", 1.0) < alpha
        and h2f.get("p_value", 1.0) < alpha
    )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    all_paths = iter_artifact_paths(args.input)
    if not all_paths:
        raise ValueError(f"No artifacts found under: {args.input}")

    faithful_pool, hall_pool = _build_candidates(
        all_paths,
        max_per_class=args.max_candidates_per_class,
    )

    if not faithful_pool or not hall_pool:
        raise ValueError(
            "Need both faithful and hallucinated candidates. "
            f"Found faithful={len(faithful_pool)} hallucinated={len(hall_pool)}"
        )

    probe = load_person1_artifact(faithful_pool[0].path, require_logits=False)
    hidden = probe["hidden_states"]
    if hidden.dim() == 2:
        hidden = hidden.unsqueeze(0)
    num_layers, _, hidden_dim = hidden.shape

    components = _component_specs(num_layers=num_layers, hidden_dim=hidden_dim)
    prototypes = _build_class_prototypes(
        faithful_pool=faithful_pool,
        hall_pool=hall_pool,
    )

    f2h, details_f2h = _run_direction(
        recipient_pool=faithful_pool,
        donor_pool=hall_pool,
        components=components,
        pairs_per_component=args.pairs_per_component,
        seed=args.seed,
        direction_name="faithful_to_hallucinated",
        prototypes=prototypes,
    )

    h2f, details_h2f = _run_direction(
        recipient_pool=hall_pool,
        donor_pool=faithful_pool,
        components=components,
        pairs_per_component=args.pairs_per_component,
        seed=args.seed + 17,
        direction_name="hallucinated_to_faithful",
        prototypes=prototypes,
    )

    rows = []
    for component in components:
        ckey = component.key
        f = f2h.get(ckey, {})
        h = h2f.get(ckey, {})
        is_critical = _critical_flag(f, h)
        rows.append(
            {
                "component": component.label,
                "component_key": ckey,
                "cie_faithful_to_hallucinated": f.get("mean_cie", 0.0),
                "p_faithful_to_hallucinated": f.get("p_value", 1.0),
                "n_faithful_to_hallucinated": f.get("n", 0),
                "cie_hallucinated_to_faithful": h.get("mean_cie", 0.0),
                "p_hallucinated_to_faithful": h.get("p_value", 1.0),
                "n_hallucinated_to_faithful": h.get("n", 0),
                "critical": bool(is_critical),
            }
        )

    critical_count = sum(1 for r in rows if r["critical"])
    best_component = max(rows, key=lambda r: r["cie_faithful_to_hallucinated"])

    redeep_statement = (
        "Agreement with ReDeEP: late FFN appears critical."
        if any(r["component_key"] == "late_ffn_layers" and r["critical"] for r in rows)
        else "Potential disagreement with ReDeEP: late FFN not uniquely critical in this run."
    )

    summary = {
        "experiment": "Experiment 3 - Causal intervention (activation patching)",
    "metric_used_for_cie": "prototype drift score (hall similarity - faithful similarity), cosine fallback",
        "total_experiments": sum(r["n_faithful_to_hallucinated"] + r["n_hallucinated_to_faithful"] for r in rows),
        "pairs_per_component_per_direction": args.pairs_per_component,
        "candidate_pool": {
            "faithful": len(faithful_pool),
            "hallucinated": len(hall_pool),
        },
        "component_results": rows,
        "critical_components_count": critical_count,
        "mark_readiness": {
            "min_50_experiments": sum(r["n_faithful_to_hallucinated"] + r["n_hallucinated_to_faithful"] for r in rows) >= 50,
            "both_directions_reported": True,
            "significant_in_at_least_two_components": critical_count >= 2,
        },
        "redeep_comparison_note": redeep_statement,
        "top_component_by_f2h_cie": {
            "component": best_component["component"],
            "cie": best_component["cie_faithful_to_hallucinated"],
        },
        "method_notes": [
            "Patching is done directly in hidden-state tensors from Person 1 artifacts.",
            "Early/copying heads are approximated as subspaces within layer+dimension slices because per-head activations are not stored explicitly.",
            "Significance uses paired sign-flip permutation test around mean CIE delta.",
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "experiment3_activation_patching_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    details_path = args.output_dir / "experiment3_activation_patching_details.jsonl"
    if args.save_details:
        with details_path.open("w", encoding="utf-8") as f:
            for row in details_f2h + details_h2f:
                f.write(json.dumps(row) + "\n")

    print(f"saved_summary={summary_path}")
    if args.save_details:
        print(f"saved_details={details_path}")
    print(f"total_experiments={summary['total_experiments']}")
    print(f"critical_components={critical_count}")
    print(f"redeep_note={redeep_statement}")


if __name__ == "__main__":
    main()
