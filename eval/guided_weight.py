import torch
import numpy as np
from typing import List, Tuple



def ensembled_token(logits1: torch.Tensor,
                    logits2: torch.Tensor):
    """
    logits1, logits2: [1, V]
    sample_fn: (probs: Tensor[V]) -> token_id: int
    """
    conf1 = p1.max(dim=-1, keepdim=True).values  # [1,1]
    conf2 = p2.max(dim=-1, keepdim=True).values  # [1,1]

    lam = conf1 / (conf1 + conf2 + 1e-8)         # [1,1]

    p_ens = lam * p1 + (1 - lam) * p2            # [1, V]

    return p_ens



def calculate_attention_score(attention_matrix):
    """
    Compute a scalar score for a given attention vector.
    The score can be defined as the maximum value, entropy, or any other metric.

    Args:
        attention_matrix (torch.Tensor): Attention vector of shape [1, 1, L]

    Returns:
        float: Attention score
    """
    # Currently uses max value as score; can be replaced with entropy or other measures
    return attention_matrix.max().item()

def dynamic_ensemble_models(
    logits_list: List[torch.Tensor],
    attentions_list: List[List[torch.Tensor]],
    top_k_layer: int = 4,
    top_h_head: int = 4,
    lambda_step: float = 0.05
) -> Tuple[torch.Tensor, List[float]]:
    """
    Dynamically ensemble multiple models based on attention quality scores.

    Args:
        logits_list (List[Tensor]): Logits from each model, shape [B, V]
        attentions_list (List[List[Tensor]]): Attention tensors from each model (per-layer)
        top_k_layer (int): Number of top layers to consider per model
        top_h_head (int): Number of top heads to consider within each selected layer
        lambda_step (float): Step size when searching for optimal lambda during blending

    Returns:
        Tuple[Tensor, List[float]]:
            - Ensemble logits tensor [B, V]
            - List of fusion weights for each model
    """
    n_models = len(logits_list)
    if n_models == 0:
        raise ValueError("At least one model is required for ensembling.")
    elif n_models == 1:
        return logits_list[0], [1.0]

    # Step 1: Compute representative attention score for each model
    attention_scores = []
    for model_attentions in attentions_list:
        repr_attn = get_representative_attention(
            model_attentions,
            top_k=top_k_layer,
            top_h=top_h_head
        )
        score = calculate_attention_score(repr_attn)
        attention_scores.append(score)

    # Step 2: Rank models by attention score (descending)
    sorted_indices = sorted(range(n_models), key=lambda i: attention_scores[i], reverse=True)
    sorted_logits = [logits_list[i] for i in sorted_indices]
    sorted_scores = [attention_scores[i] for i in sorted_indices]

    # Step 3: Initialize with top-1 model
    current_ensemble = sorted_logits[0]
    current_score = sorted_scores[0]
    model_weights = [0.0] * n_models
    model_weights[sorted_indices[0]] = 1.0

    # Step 4: Iteratively add more models by weighted fusion
    for i in range(1, n_models):
        next_model_logits = sorted_logits[i]
        next_model_score = sorted_scores[i]

        best_lambda = 0.0
        best_score = current_score
        best_ensemble = current_ensemble

        # Try different blend weights between current and next model
        for lmbda in np.arange(0.0, 1.0 + lambda_step / 2, lambda_step):
            new_ensemble = (1 - lmbda) * current_ensemble + lmbda * next_model_logits
            new_score = (1 - lmbda) * current_score + lmbda * next_model_score

            if new_score > best_score:
                best_score = new_score
                best_lambda = lmbda
                best_ensemble = new_ensemble

        # Update ensemble if this model improves the score
        if best_lambda > 0:
            for j in range(i):
                model_weights[sorted_indices[j]] *= (1 - best_lambda)
            model_weights[sorted_indices[i]] = best_lambda

            current_ensemble = best_ensemble
            current_score = best_score

    # Step 5: Normalize final weights to sum to 1
    weight_sum = sum(model_weights)
    if weight_sum > 0:
        model_weights = [w / weight_sum for w in model_weights]

    return current_ensemble, model_weights

def calculate_entropy(logits: torch.Tensor) -> float:
    """
    Compute the entropy of a probability distribution as a measure of uncertainty.

    Args:
        logits (torch.Tensor): A probability distribution tensor (assumed already softmaxed).

    Returns:
        float: Entropy value (scalar)
    """
    probs = logits  # Assumes input is already a probability distribution
    log_probs = torch.log(probs + 1e-10)  # Add epsilon to avoid log(0)
    entropy = -torch.sum(probs * log_probs)
    return entropy.item()

'''
Dynamic ensemble based on greedy + grid search strategy
Fuse multiple model logits by minimizing uncertainty (entropy)

1. Compute the entropy of each model's logits
2. Sort models by entropy in ascending order (lower entropy = more confident)
3. Initialize with the model that has the lowest entropy
4. Iteratively consider adding the remaining models
5. For each candidate model, search over lambda values and choose the one minimizing entropy
6. Update the current ensemble with the best combination
7. Normalize the final weights to ensure they sum to 1
8. Return the fused logits and corresponding weights
9. Compute the weight assigned to each model
10. Return the final fused logits and fusion weights
'''
def dynamic_ensemble_with_uncertainty(
    logits_list: List[torch.Tensor],
    lambda_step: float = 0.05
) -> Tuple[torch.Tensor, List[float]]:
    """
    Dynamically ensemble multiple model logits by minimizing perplexity (i.e., entropy).

    Args:
        logits_list (List[Tensor]): List of logits from different models, shape [B, V].
        lambda_step (float): Step size for blending coefficient lambda (e.g., 0.05)

    Returns:
        Tuple[Tensor, List[float]]:
            - The ensembled logits (same shape as input logits)
            - The normalized fusion weights for each model
    """
    n_models = len(logits_list)
    if n_models == 0:
        raise ValueError("At least one model is required for ensembling.")
    elif n_models == 1:
        return logits_list[0], [1.0]

    # 1. Compute entropy for each model's logits
    entropy_scores = [calculate_entropy(logits) for logits in logits_list]

    # 2. Sort models in ascending order of entropy (lower = more confident)
    sorted_indices = sorted(range(n_models), key=lambda i: entropy_scores[i])
    sorted_logits = [logits_list[i] for i in sorted_indices]
    sorted_entropies = [entropy_scores[i] for i in sorted_indices]

    # 3. Initialize with the most confident model (lowest entropy)
    current_ensemble = sorted_logits[0]
    current_entropy = sorted_entropies[0]
    model_weights = [0.0] * n_models
    model_weights[sorted_indices[0]] = 1.0

    # 4. Iteratively add remaining models
    for i in range(1, n_models):
        next_model_logits = sorted_logits[i]

        best_lambda = 0.0
        best_entropy = current_entropy
        best_ensemble = current_ensemble

        # 5. Search for best lambda that minimizes entropy
        for lmbda in np.arange(0.0, 1.0 + lambda_step / 2, lambda_step):
            new_ensemble = (1 - lmbda) * current_ensemble + lmbda * next_model_logits
            new_entropy = calculate_entropy(new_ensemble)

            # Keep the best combination (lowest entropy)
            if new_entropy < best_entropy:
                best_entropy = new_entropy
                best_lambda = lmbda
                best_ensemble = new_ensemble

        # 6. Update ensemble if improvement was found
        if best_lambda > 0:
            # Update weights of previously added models
            for j in range(i):
                model_weights[sorted_indices[j]] *= (1 - best_lambda)
            model_weights[sorted_indices[i]] = best_lambda

            # Update current ensemble state
            current_ensemble = best_ensemble
            current_entropy = best_entropy
            print(current_entropy)

    # 7. Normalize fusion weights to sum to 1
    weight_sum = sum(model_weights)
    if weight_sum > 0:
        model_weights = [w / weight_sum for w in model_weights]

    # 8. Return final ensemble logits and weights
    return current_ensemble, model_weights

def dynamic_ensemble_with_perplexity2(
    logits_list: List[torch.Tensor],
    lambda_step: float = 0.05,
    entropy_threshold: float = 0.1
) -> Tuple[torch.Tensor, List[float]]:
    """
    Dynamically ensemble multiple model logits by minimizing perplexity (entropy),
    with optional early stopping based on an entropy threshold.

    Args:
        logits_list (List[Tensor]): List of logits from each model.
        lambda_step (float): Step size for searching blending coefficient λ.
        entropy_threshold (float): If the ensemble entropy drops below this value, stop early.

    Returns:
        Tuple[Tensor, List[float]]:
            - Fused logits tensor [B, V]
            - Normalized fusion weights for each model
    """
    n_models = len(logits_list)
    if n_models == 0:
        raise ValueError("At least one model is required for ensembling.")
    elif n_models == 1:
        return logits_list[0], [1.0]

    # 1. Compute entropy for each model
    entropy_scores = [calculate_entropy(logits) for logits in logits_list]
    print(entropy_scores)

    # 2. Sort models by ascending entropy (lower = more confident)
    sorted_indices = sorted(range(n_models), key=lambda i: entropy_scores[i])
    sorted_logits = [logits_list[i] for i in sorted_indices]
    sorted_entropies = [entropy_scores[i] for i in sorted_indices]

    # 3. Initialize with the lowest-entropy model
    current_ensemble = sorted_logits[0]
    current_entropy = sorted_entropies[0]
    model_weights = [0.0] * n_models
    model_weights[sorted_indices[0]] = 1.0

    # 4. Iteratively consider adding other models
    for i in range(1, n_models):
        # Stop early if entropy already below threshold
        if current_entropy < entropy_threshold:
            break

        next_model_logits = sorted_logits[i]
        best_lambda = 0.0
        best_entropy = current_entropy
        best_ensemble = current_ensemble

        # 5. Search for the best blending λ
        for lmbda in np.arange(0.0, 1.0 + lambda_step / 2, lambda_step):
            new_ensemble = (1 - lmbda) * current_ensemble + lmbda * next_model_logits
            new_entropy = calculate_entropy(new_ensemble)

            # If entropy drops below threshold, accept immediately
            if new_entropy < entropy_threshold:
                best_entropy = new_entropy
                best_lambda = lmbda
                best_ensemble = new_ensemble
                break

            # Otherwise, track best improvement
            if new_entropy < best_entropy:
                best_entropy = new_entropy
                best_lambda = lmbda
                best_ensemble = new_ensemble

        # 6. Update ensemble if any improvement was found
        if best_lambda > 0:
            for j in range(i):
                model_weights[sorted_indices[j]] *= (1 - best_lambda)
            model_weights[sorted_indices[i]] = best_lambda

            current_ensemble = best_ensemble
            current_entropy = best_entropy
            print(current_entropy)

    # 7. Normalize final weights to ensure they sum to 1
    weight_sum = sum(model_weights)
    if weight_sum > 0:
        model_weights = [w / weight_sum for w in model_weights]

    return current_ensemble, model_weights
