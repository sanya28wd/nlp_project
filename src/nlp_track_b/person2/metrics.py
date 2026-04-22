"""Core hallucination detection metrics """

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def compute_cosine_drift(
    hidden_states: torch.Tensor,
    answer_start: int,
    answer_end: int,
    layers: str | list[int] = "last4",
) -> dict[str, Any]:
    """
    Compute cosine drift: similarity between context and answer representations.
    
    Args:
        hidden_states: (num_layers, seq_len, hidden_dim) or (seq_len, hidden_dim)
        answer_start: token index where answer begins
        answer_end: token index where answer ends
        layers: "last4", "all", or list of layer indices
    
    Returns:
        Dictionary with cosine_drift and per-layer values.
    """
    # Handle 1D tensors (empty hidden_states) - return zero metric
    if hidden_states.numel() == 0 or len(hidden_states.shape) < 2:
        return {
            "cosine_drift": torch.tensor(0.0, dtype=torch.float32),
            "cosine_drift_per_layer": torch.tensor([], dtype=torch.float32),
            "layers_used": [],
        }
    
    # Ensure 3D tensor
    if len(hidden_states.shape) == 2:
        # (seq_len, hidden_dim) -> (1, seq_len, hidden_dim)
        hidden_states = hidden_states.unsqueeze(0)
    
    if isinstance(layers, str):
        if layers == "last4":
            layers_to_use = list(range(max(0, hidden_states.shape[0] - 4), hidden_states.shape[0]))
        elif layers == "all":
            layers_to_use = list(range(hidden_states.shape[0]))
        else:
            layers_to_use = []
    else:
        layers_to_use = layers

    if not layers_to_use:
        layers_to_use = list(range(hidden_states.shape[0]))

    # Edge case: if answer_start == 0, use first token as context proxy
    if answer_start == 0:
        context_reps = hidden_states[:, 0:1, :].mean(dim=1)
    else:
        context_reps = hidden_states[:, :answer_start, :].mean(dim=1)
    
    answer_reps = hidden_states[:, answer_start:answer_end, :].mean(dim=1)  # (num_layers, hidden_dim)

    cosine_drifts = []
    for layer_idx in layers_to_use:
        ctx = F.normalize(context_reps[layer_idx], p=2, dim=0)
        ans = F.normalize(answer_reps[layer_idx], p=2, dim=0)
        cos_sim = torch.dot(ctx, ans).item()
        # Clamp to avoid NaN from numerical errors
        cos_sim = max(-1.0, min(1.0, cos_sim))
        cosine_drifts.append(1.0 - cos_sim)

    return {
        "cosine_drift": torch.tensor(sum(cosine_drifts) / len(cosine_drifts), dtype=torch.float32),
        "cosine_drift_per_layer": torch.tensor(cosine_drifts, dtype=torch.float32),
        "layers_used": layers_to_use,
    }


def compute_mahalanobis(
    hidden_states: torch.Tensor,
    answer_start: int,
    answer_end: int,
    stats: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Compute Mahalanobis distance using fitted statistics."""
    mean = stats.get("mean")
    inv_cov = stats.get("inv_cov")

    if mean is None or inv_cov is None:
        return {
            "mahalanobis_distance": torch.tensor(0.0),
            "mahalanobis_per_layer": torch.tensor([]),
        }

    answer_reps = hidden_states[:, answer_start:answer_end, :].mean(dim=1)  # (num_layers, hidden_dim)
    
    diffs = answer_reps - mean
    distances = []
    for layer_idx in range(len(answer_reps)):
        diff = diffs[layer_idx]
        if inv_cov[layer_idx].numel() > 0:
            dist = torch.sqrt(torch.clamp(diff @ inv_cov[layer_idx] @ diff, min=0.0))
            distances.append(dist.item())
        else:
            distances.append(0.0)

    return {
        "mahalanobis_distance": torch.tensor(sum(distances) / len(distances) if distances else 0.0, dtype=torch.float32),
        "mahalanobis_per_layer": torch.tensor(distances, dtype=torch.float32),
    }


def compute_pca_deviation(
    hidden_states: torch.Tensor,
    answer_start: int,
    answer_end: int,
    stats: dict[str, Any],
) -> dict[str, Any]:
    """Compute PCA deviation using fitted PCA model."""
    pca_models = stats.get("pca_models", [])
    
    answer_reps = hidden_states[:, answer_start:answer_end, :].mean(dim=1)  # (num_layers, hidden_dim)
    
    deviations = []
    for layer_idx, pca_model in enumerate(pca_models):
        if layer_idx < len(answer_reps):
            rep = answer_reps[layer_idx].cpu().numpy() if hasattr(answer_reps[layer_idx], 'cpu') else answer_reps[layer_idx]
            projected = pca_model.transform([rep])[0]
            reconstructed = pca_model.inverse_transform([projected])[0]
            deviation = ((rep - reconstructed) ** 2).sum()
            deviations.append(deviation)

    return {
        "pca_deviation": torch.tensor(sum(deviations) / len(deviations) if deviations else 0.0, dtype=torch.float32),
        "pca_deviation_per_layer": torch.tensor(deviations, dtype=torch.float32),
    }


def compute_logit_lens_divergence(
    hidden_states: torch.Tensor,
    logits: torch.Tensor | tuple,
    answer_start: int,
    answer_end: int,
    model: Any,
    layers: str | list[int] = "last4",
) -> dict[str, Any]:
    """Compute logit lens divergence across layers.

    For full logits: KL divergence between projected layer logits and final logits.
    For compact top-k logits: 1 - top-k overlap between projected layer and final logits.
    """
    if hidden_states.numel() == 0 or len(hidden_states.shape) < 2:
        return {
            "logit_lens_divergence": torch.tensor(0.0, dtype=torch.float32),
            "logit_lens_divergence_per_layer": torch.tensor([], dtype=torch.float32),
        }

    if len(hidden_states.shape) == 2:
        hidden_states = hidden_states.unsqueeze(0)

    if isinstance(layers, str):
        if layers == "last4":
            layers_to_use = list(range(max(0, hidden_states.shape[0] - 4), hidden_states.shape[0]))
        elif layers == "all":
            layers_to_use = list(range(hidden_states.shape[0]))
        else:
            layers_to_use = []
    else:
        layers_to_use = layers

    if not layers_to_use:
        layers_to_use = list(range(hidden_states.shape[0]))

    if answer_end <= answer_start:
        answer_start = 0
        answer_end = hidden_states.shape[1]

    answer_hidden = hidden_states[:, answer_start:answer_end, :]  # (num_layers, ans_len, hidden_dim)
    if answer_hidden.shape[1] == 0:
        return {
            "logit_lens_divergence": torch.tensor(0.0, dtype=torch.float32),
            "logit_lens_divergence_per_layer": torch.tensor([], dtype=torch.float32),
        }

    model_device = next(model.parameters()).device
    divergences: list[float] = []

    # Case A: full logits tensor available
    full_logits = None
    if isinstance(logits, torch.Tensor) and logits.numel() > 0 and logits.dim() >= 2:
        full_logits = logits.detach().cpu().float()
        full_logits = full_logits[answer_start:answer_end]
        if full_logits.shape[0] == 0:
            full_logits = None

    # Case B: compact top-k tuple available
    topk_indices = topk_values = None
    topk_k = 0
    if isinstance(logits, tuple) and len(logits) == 3:
        topk_indices, topk_values, topk_k = logits
        if isinstance(topk_indices, torch.Tensor):
            topk_indices = topk_indices.detach().to(dtype=torch.long)
        else:
            topk_indices = torch.tensor(topk_indices, dtype=torch.long)
        if isinstance(topk_values, torch.Tensor):
            topk_values = topk_values.detach().to(dtype=torch.float32)
        else:
            topk_values = torch.tensor(topk_values, dtype=torch.float32)
        topk_indices = topk_indices[answer_start:answer_end]
        topk_values = topk_values[answer_start:answer_end]

    with torch.no_grad():
        for layer_idx in layers_to_use:
            if layer_idx >= answer_hidden.shape[0]:
                continue
            layer_h = answer_hidden[layer_idx].to(model_device)
            layer_logits = model.lm_head(layer_h).detach().cpu().float()  # (ans_len, vocab)

            if full_logits is not None:
                min_len = min(layer_logits.shape[0], full_logits.shape[0])
                if min_len == 0:
                    continue
                p = F.softmax(full_logits[:min_len], dim=-1)
                q_log = F.log_softmax(layer_logits[:min_len], dim=-1)
                div = F.kl_div(q_log, p, reduction="batchmean").item()
            elif topk_indices is not None and topk_values is not None and topk_indices.numel() > 0:
                min_len = min(layer_logits.shape[0], topk_indices.shape[0])
                if min_len == 0:
                    continue
                k = max(1, min(int(topk_k), layer_logits.shape[-1]))
                _, layer_topk_idx = torch.topk(layer_logits[:min_len], k=k, dim=-1)

                overlap_scores = []
                for t in range(min_len):
                    final_set = set(topk_indices[t].tolist()[:k])
                    layer_set = set(layer_topk_idx[t].tolist()[:k])
                    overlap = len(final_set.intersection(layer_set)) / float(k)
                    overlap_scores.append(overlap)
                div = 1.0 - (sum(overlap_scores) / len(overlap_scores) if overlap_scores else 0.0)
            else:
                # Fallback: normalized entropy as uncertainty proxy
                probs = F.softmax(layer_logits, dim=-1)
                ent = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
                div = ent.mean().item() / max(1.0, float(layer_logits.shape[-1]).bit_length())

            divergences.append(float(div))

    return {
        "logit_lens_divergence": torch.tensor(sum(divergences) / len(divergences) if divergences else 0.0, dtype=torch.float32),
        "logit_lens_divergence_per_layer": torch.tensor(divergences, dtype=torch.float32),
    }


def compute_cross_layer_disagreement(
    hidden_states: torch.Tensor,
    answer_start: int,
    answer_end: int,
) -> dict[str, Any]:
    """
    Compute cross-layer disagreement: pairwise cosine distance between adjacent layers.
    High disagreement indicates hallucinations (inconsistent representations across layers).
    
    Args:
        hidden_states: (num_layers, seq_len, hidden_dim) or (seq_len, hidden_dim)
        answer_start: token index where answer begins
        answer_end: token index where answer ends
    
    Returns:
        Dictionary with cross_layer_disagreement and per-pair values.
    """
    # Handle empty tensors
    if hidden_states.numel() == 0 or len(hidden_states.shape) < 2:
        return {
            "cross_layer_disagreement": torch.tensor(0.0, dtype=torch.float32),
            "cross_layer_disagreement_per_pair": torch.tensor([], dtype=torch.float32),
        }
    
    # Ensure 3D tensor
    if len(hidden_states.shape) == 2:
        hidden_states = hidden_states.unsqueeze(0)
    
    answer_reps = hidden_states[:, answer_start:answer_end, :].mean(dim=1)  # (num_layers, hidden_dim)
    
    disagreements = []
    for i in range(len(answer_reps) - 1):
        rep1 = F.normalize(answer_reps[i], p=2, dim=0)
        rep2 = F.normalize(answer_reps[i + 1], p=2, dim=0)
        cosine_sim = torch.dot(rep1, rep2).item()
        cosine_sim = max(-1.0, min(1.0, cosine_sim))
        disagreement = 1.0 - cosine_sim  # Higher = more disagreement
        disagreements.append(disagreement)
    
    return {
        "cross_layer_disagreement": torch.tensor(sum(disagreements) / len(disagreements) if disagreements else 0.0, dtype=torch.float32),
        "cross_layer_disagreement_per_pair": torch.tensor(disagreements, dtype=torch.float32),
    }


def compute_uncertainty_ensemble(
    hidden_states: torch.Tensor,
    answer_start: int,
    answer_end: int,
) -> dict[str, Any]:
    """
    Compute token-level uncertainty ensemble: variance of token representations across layers.
    High variance indicates model uncertainty about token identity.
    
    Args:
        hidden_states: (num_layers, seq_len, hidden_dim) or (seq_len, hidden_dim)
        answer_start: token index where answer begins
        answer_end: token index where answer ends
    
    Returns:
        Dictionary with uncertainty_ensemble metric.
    """
    # Handle empty tensors
    if hidden_states.numel() == 0 or len(hidden_states.shape) < 2:
        return {"uncertainty_ensemble": torch.tensor(0.0, dtype=torch.float32)}
    
    # Ensure 3D tensor
    if len(hidden_states.shape) == 2:
        hidden_states = hidden_states.unsqueeze(0)
    
    answer_hidden = hidden_states[:, answer_start:answer_end, :]  # (num_layers, answer_len, hidden_dim)
    
    # Compute per-token variance across layers
    token_variance = torch.var(answer_hidden, dim=0)  # (answer_len, hidden_dim)
    
    # Aggregate: average variance across tokens and dims
    uncertainty = token_variance.mean().item()
    
    return {
        "uncertainty_ensemble": torch.tensor(uncertainty, dtype=torch.float32),
    }


def compute_consistency_metric(
    hidden_states: torch.Tensor,
    answer_start: int,
    answer_end: int,
) -> dict[str, Any]:
    """
    Compute consistency: stability of answer representations across tokens.
    Low consistency (high variance) indicates hallucinations.
    
    Args:
        hidden_states: (num_layers, seq_len, hidden_dim) or (seq_len, hidden_dim)
        answer_start: token index where answer begins
        answer_end: token index where answer ends
    
    Returns:
        Dictionary with consistency_metric value.
    """
    # Handle empty tensors
    if hidden_states.numel() == 0 or len(hidden_states.shape) < 2:
        return {
            "consistency_metric": torch.tensor(0.0, dtype=torch.float32),
            "consistency_per_layer": torch.tensor([], dtype=torch.float32),
        }
    
    # Ensure 3D tensor
    if len(hidden_states.shape) == 2:
        hidden_states = hidden_states.unsqueeze(0)
    
    answer_hidden = hidden_states[:, answer_start:answer_end, :]  # (num_layers, answer_len, hidden_dim)
    
    # Compute variance within each layer
    layer_consistency = []
    for layer_idx in range(answer_hidden.shape[0]):
        tokens = answer_hidden[layer_idx]  # (answer_len, hidden_dim)
        # Compute pairwise cosine similarity between tokens
        normalized = F.normalize(tokens, p=2, dim=1)
        similarity_matrix = normalized @ normalized.T  # (answer_len, answer_len)
        
        # Average off-diagonal similarity (how consistent are tokens)
        if len(tokens) > 1:
            mask = ~torch.eye(len(tokens), dtype=torch.bool, device=similarity_matrix.device)
            consistency = similarity_matrix[mask].mean().item()
        else:
            consistency = 1.0
        
        layer_consistency.append(consistency)
    
    # Consistency: average across layers (higher = more consistent)
    consistency_score = sum(layer_consistency) / len(layer_consistency) if layer_consistency else 0.0
    
    return {
        "consistency_metric": torch.tensor(consistency_score, dtype=torch.float32),
        "consistency_per_layer": torch.tensor(layer_consistency, dtype=torch.float32),
    }


def compute_attention_variance(
    hidden_states: torch.Tensor,
    answer_start: int,
    answer_end: int,
) -> dict[str, Any]:
    """
    Compute attention pattern variance: consistency of hidden representations within answer.
    High variance (low consistency) suggests hallucination.
    
    Args:
        hidden_states: (num_layers, seq_len, hidden_dim) or (seq_len, hidden_dim)
        answer_start: token index where answer begins
        answer_end: token index where answer ends
    
    Returns:
        Dictionary with attention_variance metric.
    """
    # Handle empty tensors
    if hidden_states.numel() == 0 or len(hidden_states.shape) < 2:
        return {
            "attention_variance": torch.tensor(0.0, dtype=torch.float32),
            "attention_variance_per_layer": torch.tensor([], dtype=torch.float32),
        }
    
    # Ensure 3D tensor
    if len(hidden_states.shape) == 2:
        hidden_states = hidden_states.unsqueeze(0)
    
    answer_hidden = hidden_states[:, answer_start:answer_end, :]  # (num_layers, answer_len, hidden_dim)
    
    # Compute per-layer variance
    layer_variances = []
    for layer_idx in range(answer_hidden.shape[0]):
        tokens = answer_hidden[layer_idx]  # (answer_len, hidden_dim)
        # Variance across tokens in each dimension
        variance = tokens.var(dim=0).mean().item()
        layer_variances.append(variance)
    
    # Average variance across layers
    attention_variance = sum(layer_variances) / len(layer_variances) if layer_variances else 0.0
    
    return {
        "attention_variance": torch.tensor(attention_variance, dtype=torch.float32),
        "attention_variance_per_layer": torch.tensor(layer_variances, dtype=torch.float32),
    }


def compute_layer_confidence_degradation(
    hidden_states: torch.Tensor,
    answer_start: int,
    answer_end: int,
) -> dict[str, Any]:
    """
    Compute confidence degradation: trend in answer representation magnitude across layers.
    Hallucinations often show decreasing confidence (L2 norm) across layers.
    
    Args:
        hidden_states: (num_layers, seq_len, hidden_dim) or (seq_len, hidden_dim)
        answer_start: token index where answer begins
        answer_end: token index where answer ends
    
    Returns:
        Dictionary with layer_confidence_degradation metric.
    """
    # Handle empty tensors
    if hidden_states.numel() == 0 or len(hidden_states.shape) < 2:
        return {
            "layer_confidence_degradation": torch.tensor(0.0, dtype=torch.float32),
            "layer_magnitudes": torch.tensor([], dtype=torch.float32),
        }
    
    # Ensure 3D tensor
    if len(hidden_states.shape) == 2:
        hidden_states = hidden_states.unsqueeze(0)
    
    answer_reps = hidden_states[:, answer_start:answer_end, :].mean(dim=1)  # (num_layers, hidden_dim)
    
    # Compute L2 norm (magnitude) for each layer
    layer_magnitudes = []
    for layer_idx in range(len(answer_reps)):
        magnitude = torch.norm(answer_reps[layer_idx], p=2).item()
        layer_magnitudes.append(magnitude)
    
    # Compute degradation: trend from first to last layer
    if len(layer_magnitudes) > 1:
        # Linear regression slope: does magnitude decrease?
        first_half_avg = sum(layer_magnitudes[:len(layer_magnitudes)//2]) / max(1, len(layer_magnitudes)//2)
        second_half_avg = sum(layer_magnitudes[len(layer_magnitudes)//2:]) / max(1, len(layer_magnitudes) - len(layer_magnitudes)//2)
        degradation = first_half_avg - second_half_avg  # Positive = degradation
    else:
        degradation = 0.0
    
    return {
        "layer_confidence_degradation": torch.tensor(degradation, dtype=torch.float32),
        "layer_magnitudes": torch.tensor(layer_magnitudes, dtype=torch.float32),
    }


def compute_entropy_variance(
    hidden_states: torch.Tensor,
    answer_start: int,
    answer_end: int,
) -> dict[str, Any]:
    """
    Compute entropy variance: variance of token entropy across layers.
    High variance indicates uncertain, hallucinatory outputs.
    
    Args:
        hidden_states: (num_layers, seq_len, hidden_dim) or (seq_len, hidden_dim)
        answer_start: token index where answer begins
        answer_end: token index where answer ends
    
    Returns:
        Dictionary with entropy_variance metric.
    """
    # Handle empty tensors
    if hidden_states.numel() == 0 or len(hidden_states.shape) < 2:
        return {
            "entropy_variance": torch.tensor(0.0, dtype=torch.float32),
            "entropy_per_layer": torch.tensor([], dtype=torch.float32),
        }
    
    # Ensure 3D tensor
    if len(hidden_states.shape) == 2:
        hidden_states = hidden_states.unsqueeze(0)
    
    answer_hidden = hidden_states[:, answer_start:answer_end, :]  # (num_layers, answer_len, hidden_dim)
    
    # Compute entropy per layer (using softmax over dimension)
    layer_entropies = []
    for layer_idx in range(answer_hidden.shape[0]):
        tokens = answer_hidden[layer_idx]  # (answer_len, hidden_dim)
        # Convert to probability distribution (softmax over hidden_dim)
        probs = F.softmax(tokens, dim=1)
        # Compute entropy per token
        entropy_per_token = -(probs * torch.log(probs + 1e-8)).sum(dim=1)  # (answer_len,)
        # Average entropy for this layer
        avg_entropy = entropy_per_token.mean().item()
        layer_entropies.append(avg_entropy)
    
    # Variance of entropies across layers
    if len(layer_entropies) > 0:
        entropy_variance = torch.tensor(layer_entropies).var().item()
    else:
        entropy_variance = 0.0
    
    return {
        "entropy_variance": torch.tensor(entropy_variance, dtype=torch.float32),
        "entropy_per_layer": torch.tensor(layer_entropies, dtype=torch.float32),
    }


def compute_composite_score(
    metric_values: dict[str, torch.Tensor],
    normalizers: dict[str, tuple[float, float]],
) -> torch.Tensor:
    """Compute weighted composite score from normalized metrics."""
    scores = []
    for metric_name, (mean, std) in normalizers.items():
        if metric_name in metric_values:
            val = metric_values[metric_name].item() if hasattr(metric_values[metric_name], 'item') else metric_values[metric_name]
            normalized = (val - mean) / (std + 1e-8)
            scores.append(normalized)

    return torch.tensor(sum(scores) / len(scores) if scores else 0.0, dtype=torch.float32)


def load_hf_model(model_name: str, device: str = "auto") -> Any:
    """Load a Hugging Face model for logit lens computation."""
    try:
        from transformers import AutoModelForCausalLM
        if device == "auto":
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load HF model {model_name}: {e}")


def fit_mahalanobis_stats(
    records: list[dict] | Any,
    layers: str | list[int] = "last4",
    regularization: float = 1e-4,
) -> dict[str, torch.Tensor]:
    """Fit Mahalanobis statistics (mean, covariance) from training records.
    
    Args:
        records: Iterable of artifact dicts with 'hidden_states' key
        layers: "last4", "all", or list of layer indices
        regularization: Ridge regularization for covariance matrix
    
    Returns:
        Dict with 'mean' and 'inv_cov' tensors (per-layer).
    """
    if records is None:
        return {"mean": torch.tensor([]), "inv_cov": torch.tensor([]), "layers_used": []}
    
    def _resolve_layers(total_layers: int, layer_spec: str | list[int]) -> list[int]:
        if isinstance(layer_spec, str):
            if layer_spec == "last4":
                return list(range(max(0, total_layers - 4), total_layers))
            if layer_spec == "all":
                return list(range(total_layers))
            return list(range(total_layers))
        valid = [i for i in layer_spec if 0 <= i < total_layers]
        return valid if valid else list(range(total_layers))

    # Collect all representations
    all_reps = []
    layers_used: list[int] | None = None
    seen_any = False
    for record in records:
        seen_any = True
        hidden_states = record.get("hidden_states")
        if hidden_states is None or hidden_states.numel() == 0:
            continue
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)
        if hidden_states.dim() < 3:
            continue
        if layers_used is None:
            layers_used = _resolve_layers(hidden_states.shape[0], layers)
        hidden_states = hidden_states[layers_used]
        # Average across sequence and token dimension: (num_layers, seq_len, hidden_dim) -> (num_layers, hidden_dim)
        rep = hidden_states.mean(dim=1)
        all_reps.append(rep)
    
    if (not seen_any) or (not all_reps):
        return {"mean": torch.tensor([]), "inv_cov": torch.tensor([]), "layers_used": []}
    
    # Stack and compute per-layer statistics
    all_reps = torch.stack(all_reps, dim=1)  # (num_layers, num_samples, hidden_dim)
    mean = all_reps.mean(dim=1)  # (num_layers, hidden_dim)
    
    # Compute covariance and invert
    inv_covs = []
    for layer_idx in range(all_reps.shape[0]):
        reps = all_reps[layer_idx]  # (num_samples, hidden_dim)
        centered = reps - mean[layer_idx]
        cov = (centered.T @ centered) / max(1, len(reps) - 1)
        
        # Add regularization
        cov += regularization * torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
        
        try:
            inv_cov = torch.linalg.inv(cov)
            inv_covs.append(inv_cov)
        except:
            inv_covs.append(torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype))
    
    return {
        "mean": mean,
        "inv_cov": torch.stack(inv_covs, dim=0) if inv_covs else torch.tensor([]),
        "layers_used": layers_used if layers_used is not None else [],
    }


def fit_pca_stats(
    records: list[dict] | Any,
    layers: str | list[int] = "last4",
    n_components: int = 16,
) -> dict[str, Any]:
    """Fit PCA models per-layer from training records.
    
    Args:
        records: Iterable of artifact dicts with 'hidden_states' key
        layers: "last4", "all", or list of layer indices
        n_components: Number of PCA components to keep
    
    Returns:
        Dict with 'pca_models' list (one fitted PCA per layer).
    """
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        raise RuntimeError("scikit-learn required for PCA fitting")
    
    if records is None:
        return {"pca_models": [], "layers_used": []}
    
    def _resolve_layers(total_layers: int, layer_spec: str | list[int]) -> list[int]:
        if isinstance(layer_spec, str):
            if layer_spec == "last4":
                return list(range(max(0, total_layers - 4), total_layers))
            if layer_spec == "all":
                return list(range(total_layers))
            return list(range(total_layers))
        valid = [i for i in layer_spec if 0 <= i < total_layers]
        return valid if valid else list(range(total_layers))

    # Collect all representations
    all_reps = []
    layers_used: list[int] | None = None
    seen_any = False
    for record in records:
        seen_any = True
        hidden_states = record.get("hidden_states")
        if hidden_states is None or hidden_states.numel() == 0:
            continue
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)
        if hidden_states.dim() < 3:
            continue
        if layers_used is None:
            layers_used = _resolve_layers(hidden_states.shape[0], layers)
        hidden_states = hidden_states[layers_used]
        rep = hidden_states.mean(dim=1)
        all_reps.append(rep)
    
    if (not seen_any) or (not all_reps):
        return {"pca_models": [], "layers_used": []}
    
    all_reps = torch.stack(all_reps, dim=1)  # (num_layers, num_samples, hidden_dim)
    
    # Fit PCA per layer
    pca_models = []
    for layer_idx in range(all_reps.shape[0]):
        reps = all_reps[layer_idx].cpu().numpy()  # (num_samples, hidden_dim)
        n_comp = max(1, min(n_components, reps.shape[0], reps.shape[1]))
        pca = PCA(n_components=n_comp)
        pca.fit(reps)
        pca_models.append(pca)
    
    return {
        "pca_models": pca_models,
        "layers_used": layers_used if layers_used is not None else [],
    }


def fit_normalizer_stats(
    metric_values: dict[str, list[torch.Tensor]],
) -> dict[str, tuple[float, float]]:
    """Fit mean and std dev for each metric.
    
    Args:
        metric_values: Dict mapping metric names to lists of tensor values
    
    Returns:
        Dict mapping metric names to (mean, std_dev) tuples.
    """
    normalizers = {}
    for metric_name, values in metric_values.items():
        if not values:
            normalizers[metric_name] = (0.0, 1.0)
            continue
        
        # Convert to tensor and compute stats
        tensor_values = torch.stack([v if isinstance(v, torch.Tensor) else torch.tensor(v) for v in values])
        mean_val = tensor_values.mean().item()
        std_val = tensor_values.std().item()
        
        # Avoid zero std
        if std_val == 0:
            std_val = 1.0
        
        normalizers[metric_name] = (mean_val, std_val)
    
    return normalizers
