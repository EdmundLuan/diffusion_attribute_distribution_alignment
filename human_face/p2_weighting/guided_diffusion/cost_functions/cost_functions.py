import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Union, Optional, Dict, Callable
from copy import deepcopy

from guided_diffusion.sampling_util import (
    merge_probs,
    merge_probs_efficient,
    merge_logits,
    merge_logits_efficient,
)
from guided_diffusion.classifiers.fairface_classifiers import infer_fairface
from guided_diffusion.classifiers.latent_classifier_resnet_enc_multihead import (
    infer_pcd_classifier, 
    infer_pcd_latent_classifier,
)

def promote_target_class_loss(logits, k_tar, op_type=torch.float32) -> torch.Tensor:
    """
    Computes a loss that promotes the target class k_tar.
    Args:
        logits (torch.Tensor): Logits of shape (batch_size, K).
        k_tar (int): Index of the target class to promote.
    Returns:
        torch.Tensor: Scalar loss value.
    """
    # Negative log softmax probability of the target class
    loss = F.cross_entropy(
        logits.to(op_type), 
        torch.full((logits.size(0),), k_tar, dtype=torch.long, device=logits.device), 
        reduction='none',
    ).to(logits)
    return loss


# Helper: differentiable eps-smooth + renormalize to simplex
def smooth_norm(x: torch.Tensor, eps_t: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x = x + eps_t
    return x / (x.sum(dim=dim, keepdim=True) + eps_t * x.size(dim))

def kl_div_cost(probs:torch.Tensor, q_tar: torch.Tensor, minibatch: int = None, eps: float = 1e-8, op_type=torch.float32) -> torch.Tensor:
    assert probs.shape[-1] == q_tar.shape[-1], \
        f"Classifier output dim {probs.shape[-1]} does not match target distribution dim {q_tar.shape[-1]}."
    
    orig_type = probs.dtype
    probs = probs.to(op_type)
    q_tar = q_tar.to(op_type)
    eps_tensor = probs.new_tensor(eps, dtype=op_type)
    batch_size = probs.size(0)
    if minibatch is None or minibatch <= 0:
        p_emp = probs.mean(dim=0)
        ratio = (p_emp + eps_tensor) / (q_tar + eps_tensor)
        kl_val = (p_emp * ratio.log()).sum()
        return kl_val.expand(batch_size)

    if not isinstance(minibatch, int):
        raise TypeError("minibatch must be an integer or None.")
    # if minibatch <= 0:
    #     raise ValueError("minibatch must be positive.")

    group_size = min(minibatch, batch_size)
    num_groups = (batch_size + group_size - 1) // group_size
    group_ids = torch.arange(batch_size, device=probs.device) // group_size
    group_ids = torch.clamp(group_ids, max=num_groups - 1)

    membership = F.one_hot(group_ids, num_classes=num_groups).to(probs.dtype)
    group_counts = membership.sum(dim=0, keepdim=False).unsqueeze(1).clamp_min(1.0)
    group_probs = (membership.transpose(0, 1) @ probs) / group_counts

    ratio = (group_probs + eps_tensor) / (q_tar.unsqueeze(0) + eps_tensor)
    kl_group = (group_probs * ratio.log()).sum(dim=1)

    return kl_group.index_select(0, group_ids).to(orig_type)


def reverse_kl_div_cost(
    probs: torch.Tensor,
    q_tar: torch.Tensor,
    minibatch: int = None,
    eps: float = 1e-8,
    op_type=torch.float32
) -> torch.Tensor:
    """
    Reverse KL: KL[q_tar || p_emp] where p_emp is the empirical mean distribution.
    If minibatch is provided, compute KL[q_tar || p_group] per group and map back to samples.
    """
    assert probs.shape[-1] == q_tar.shape[-1], \
        f"Classifier output dim {probs.shape[-1]} does not match target distribution dim {q_tar.shape[-1]}."

    orig_type = probs.dtype
    eps_tensor = probs.new_tensor(eps, dtype=op_type)
    # probs = probs.to(op_type)
    # q_tar = q_tar.to(op_type)
    probs = smooth_norm(probs.to(op_type), eps_tensor, dim=-1)
    q_tar = smooth_norm(q_tar.to(op_type), eps_tensor, dim=-1)


    batch_size = probs.size(0)

    # q_tar is shared across batch; ensure it's 1D over classes
    # (kept consistent with your forward KL usage)
    # q_tar: [K]
    if minibatch is None or minibatch <= 0:
        # p_emp: [K]
        p_emp = probs.mean(dim=0)
        # ratio = (q_tar + eps_tensor) / (p_emp + eps_tensor)
        ratio = q_tar / p_emp 
        kl_val = (q_tar * ratio.log()).sum()
        return kl_val.expand(batch_size).to(orig_type)

    if not isinstance(minibatch, int):
        raise TypeError("minibatch must be an integer or None.")

    group_size = min(minibatch, batch_size)
    num_groups = (batch_size + group_size - 1) // group_size
    group_ids = torch.arange(batch_size, device=probs.device) // group_size
    group_ids = torch.clamp(group_ids, max=num_groups - 1)

    # membership: [B, G]
    membership = F.one_hot(group_ids, num_classes=num_groups).to(probs.dtype)

    # group_counts: [G, 1]
    group_counts = membership.sum(dim=0, keepdim=False).unsqueeze(1).clamp_min(1.0)

    # group_probs: [G, K]
    group_probs = (membership.transpose(0, 1) @ probs) / group_counts

    # Reverse KL per group: KL[q_tar || p_group]
    # ratio: [G, K]
    # ratio = (q_tar.unsqueeze(0) + eps_tensor) / (group_probs + eps_tensor)
    ratio = q_tar.unsqueeze(0) / group_probs 
    kl_group = (q_tar.unsqueeze(0) * ratio.log()).sum(dim=1)  # [G]

    # map group values back to each sample
    return kl_group.index_select(0, group_ids).to(orig_type)


def kl_div_cost_logits(
    logits: torch.Tensor,
    log_q_tar: torch.Tensor,
    minibatch: int = None,
    eps: float = 1e-8,
    op_type=torch.float32
) -> torch.Tensor:
    """
    Compute forward KL divergence KL[p_emp || q_tar] in log-probability space.

    This is numerically more stable than operating in probability space,
    especially when distributions are peaky or near boundaries.

    Mathematical formulation:
        KL[p || q] = sum_k p_k * (log p_k - log q_k)

    For empirical distribution from logits:
        log_p_emp = logsumexp(logits, dim=0) - log(B)

    Args:
        logits:      [B, K]  classifier output logits (unnormalized log-probs)
        log_q_tar:   [K]     log of target distribution (pre-computed)
        minibatch:   Optional group size for group-wise KL estimation
        eps:         numerical stability constant (for log of very small probs)
        op_type:     dtype for computation

    Returns:
        kl_cost:     [B]     per-sample KL divergence cost (same value for all in group)
    """
    assert logits.shape[-1] == log_q_tar.shape[-1], \
        f"Classifier output dim {logits.shape[-1]} does not match target dim {log_q_tar.shape[-1]}."

    orig_type = logits.dtype
    logits = logits.to(op_type)
    log_q_tar = log_q_tar.to(op_type)
    batch_size = logits.size(0)

    if minibatch is None or minibatch <= 0:
        # Compute log of empirical mean distribution:
        # log(p_emp) = logsumexp(log_softmax(logits), dim=0) - log(B)
        # = logsumexp(logits, dim=0) - logsumexp(logits) - log(B)
        # Simpler: use log_softmax then logsumexp
        log_probs = F.log_softmax(logits, dim=-1)  # [B, K]
        log_p_emp = torch.logsumexp(log_probs, dim=0) - torch.log(torch.tensor(batch_size, dtype=op_type, device=logits.device))  # [K]

        # KL = sum_k p_emp_k * (log_p_emp_k - log_q_tar_k)
        # = sum_k exp(log_p_emp_k) * (log_p_emp_k - log_q_tar_k)
        p_emp = torch.exp(log_p_emp)
        kl_val = (p_emp * (log_p_emp - log_q_tar)).sum()
        return kl_val.expand(batch_size).to(orig_type)

    # Group-wise computation
    if not isinstance(minibatch, int):
        raise TypeError("minibatch must be an integer or None.")

    group_size = min(minibatch, batch_size)
    num_groups = (batch_size + group_size - 1) // group_size
    group_ids = torch.arange(batch_size, device=logits.device) // group_size
    group_ids = torch.clamp(group_ids, max=num_groups - 1)

    # Convert to log-probs
    log_probs = F.log_softmax(logits, dim=-1)  # [B, K]

    # Compute log(p_emp) per group using logsumexp
    # For group g: log_p_emp_g = logsumexp(log_probs[group_g], dim=0) - log(group_count)
    kl_vals = torch.zeros(num_groups, device=logits.device, dtype=op_type)

    for g in range(num_groups):
        mask = (group_ids == g)
        group_log_probs = log_probs[mask]  # [G_size, K]
        group_count = group_log_probs.size(0)

        # log(empirical mean) for this group
        log_p_emp_g = torch.logsumexp(group_log_probs, dim=0) - torch.log(
            torch.tensor(group_count, dtype=op_type, device=logits.device)
        )  # [K]

        # KL for this group
        p_emp_g = torch.exp(log_p_emp_g)
        kl_vals[g] = (p_emp_g * (log_p_emp_g - log_q_tar)).sum()

    return kl_vals.index_select(0, group_ids).to(orig_type)


def reverse_kl_div_cost_logits(
    logits: torch.Tensor,
    log_q_tar: torch.Tensor,
    minibatch: int = None,
    eps: float = 1e-8,
    op_type=torch.float32
) -> torch.Tensor:
    """
    Compute reverse KL divergence KL[q_tar || p_emp] in log-probability space.

    Mathematical formulation:
        KL[q || p] = sum_k q_k * (log q_k - log p_k)

    Args:
        logits:      [B, K]  classifier output logits (unnormalized log-probs)
        log_q_tar:   [K]     log of target distribution (pre-computed)
        minibatch:   Optional group size for group-wise KL estimation
        eps:         numerical stability constant
        op_type:     dtype for computation

    Returns:
        kl_cost:     [B]     per-sample reverse KL divergence cost
    """
    assert logits.shape[-1] == log_q_tar.shape[-1], \
        f"Classifier output dim {logits.shape[-1]} does not match target dim {log_q_tar.shape[-1]}."

    orig_type = logits.dtype
    logits = logits.to(op_type)
    log_q_tar = log_q_tar.to(op_type)
    q_tar = torch.exp(log_q_tar)  # [K]
    batch_size = logits.size(0)

    if minibatch is None or minibatch <= 0:
        # Compute log of empirical mean distribution
        log_probs = F.log_softmax(logits, dim=-1)  # [B, K]
        log_p_emp = torch.logsumexp(log_probs, dim=0) - torch.log(
            torch.tensor(batch_size, dtype=op_type, device=logits.device)
        )  # [K]

        # Reverse KL = sum_k q_tar_k * (log_q_tar_k - log_p_emp_k)
        kl_val = (q_tar * (log_q_tar - log_p_emp)).sum()
        return kl_val.expand(batch_size).to(orig_type)

    # Group-wise computation
    if not isinstance(minibatch, int):
        raise TypeError("minibatch must be an integer or None.")

    group_size = min(minibatch, batch_size)
    num_groups = (batch_size + group_size - 1) // group_size
    group_ids = torch.arange(batch_size, device=logits.device) // group_size
    group_ids = torch.clamp(group_ids, max=num_groups - 1)

    log_probs = F.log_softmax(logits, dim=-1)  # [B, K]
    kl_vals = torch.zeros(num_groups, device=logits.device, dtype=op_type)

    for g in range(num_groups):
        mask = (group_ids == g)
        group_log_probs = log_probs[mask]  # [G_size, K]
        group_count = group_log_probs.size(0)

        # log(empirical mean) for this group
        log_p_emp_g = torch.logsumexp(group_log_probs, dim=0) - torch.log(
            torch.tensor(group_count, dtype=op_type, device=logits.device)
        )  # [K]

        # Reverse KL for this group
        kl_vals[g] = (q_tar * (log_q_tar - log_p_emp_g)).sum()

    return kl_vals.index_select(0, group_ids).to(orig_type)


def total_correlation(probs_dict: Dict[str, torch.Tensor], minibatch: int = None, eps: float = 1e-8) -> torch.Tensor:
    """
    Computes the Total Correlation (TC) of the empirical distribution efficiently.
    TC(P) = Sum_i(H(P_i)) - H(P_joint)
    
    This acts as a "correlation penalty." Minimizing this (along with marginal KL)
    forces the model to generate attributes that are statistically independent
    (e.g., Race and Gender become uncorrelated).

    Args:
        probs_dict: Dict of probabilities {head_name: [Batch, Num_Classes]}.
                    Values should be Softmaxed already.
        minibatch:  Size of the group to estimate empirical distribution. 
                    If None, uses the whole batch.
        eps:        Numerical stability constant.

    Returns:
        torch.Tensor: [Batch] Tensor containing the TC value for each sample's group.
    """
    # 1. Prepare Data
    # Convert dict values to a list for indexed access
    probs_list = list(probs_dict.values())
    device = probs_list[0].device
    dtype = probs_list[0].dtype
    batch_size = probs_list[0].size(0)
    eps_tensor = torch.tensor(eps, device=device, dtype=dtype)

    # 2. Construct Group Aggregation Matrix (Vectorized Grouping)
    # This matrix M allows us to convert Sample Probs [B, ...] -> Group Probs [G, ...]
    # via simple matrix multiplication: M @ Probs
    if minibatch is None or minibatch <= 0:
        # Single group containing all samples
        # Shape: [1, B]
        group_membership = torch.ones((1, batch_size), device=device, dtype=dtype)
        group_counts = torch.tensor([[batch_size]], device=device, dtype=dtype)
        group_ids = torch.zeros(batch_size, dtype=torch.long, device=device)
    else:
        # Multiple groups
        group_size = min(minibatch, batch_size)
        num_groups = (batch_size + group_size - 1) // group_size
        
        # Create group IDs: [0, 0, ..., 1, 1, ...]
        group_ids = torch.arange(batch_size, device=device) // group_size
        group_ids = torch.clamp(group_ids, max=num_groups - 1)
        
        # One-hot encoding transposed: [G, B]
        group_membership = F.one_hot(group_ids, num_classes=num_groups).to(dtype).T
        group_counts = group_membership.sum(dim=1, keepdim=True).clamp_min(1.0) # [G, 1]

    # Normalize membership to be an averaging operator
    # aggregator: [G, B] where each row sums to 1
    aggregator = group_membership / group_counts

    # 3. Compute Marginal Entropies (Sum_i H(P_i))
    # We calculate H(Mean(probs)), not Mean(H(probs))
    sum_marginal_entropy = 0
    for p in probs_list:
        # Aggregate samples into group empirical distribution
        # [G, B] @ [B, C] -> [G, C]
        p_emp = aggregator @ p
        
        # Entropy: -Sum(p * log(p))
        entropy = - (p_emp * (p_emp + eps_tensor).log()).sum(dim=1) # [G]
        sum_marginal_entropy = sum_marginal_entropy + entropy

    # 4. Compute Joint Entropy (H(P_joint))
    # We need the joint probability of every sample, then average it.
    # P_joint_sample[b, i, j, k...] = P1[b, i] * P2[b, j] * P3[b, k]...
    
    # We use Einstein Summation for efficiency.
    # We generate a dynamic equation string, e.g., "ba,bb,bc->babc"
    # The first char 'b' is the batch dim, shared across all inputs.
    # Subsequent chars 'a', 'b', 'c'... are class dims for each head.
    
    # Generate distinct char indices for each head (skip 'b' which is char 98)
    # starting from char 99 ('c')
    char_codes = [chr(99 + i) for i in range(len(probs_list))]
    
    # Equation: "bi,bj,bk->bijk" (using generated chars)
    # Inputs: "b" + char for each prob tensor
    inputs_str = ','.join(['b' + c for c in char_codes])
    # Output: "b" + all chars concatenated
    output_str = 'b' + ''.join(char_codes)
    einsum_eq = f"{inputs_str}->{output_str}"
    
    # Compute Sample-wise Joint Distribution
    # Result shape: [B, C1, C2, C3...]
    p_joint_samples = torch.einsum(einsum_eq, *probs_list)
    
    # Flatten all class dimensions into one "Joint Class" dimension
    # Shape: [B, Total_Combinations]
    p_joint_samples = p_joint_samples.reshape(batch_size, -1)
    
    # Aggregate into Group Empirical Joint Distribution
    # [G, B] @ [B, Total] -> [G, Total]
    p_joint_emp = aggregator @ p_joint_samples
    
    # Joint Entropy
    joint_entropy = - (p_joint_emp * (p_joint_emp + eps_tensor).log()).sum(dim=1) # [G]

    # 5. Compute Total Correlation
    # TC = Sum(Marginal H) - Joint H
    total_corr = sum_marginal_entropy - joint_entropy
    
    # # Numerical stability: TC should be >= 0, but float error can make it -1e-8
    # total_corr = torch.clamp(total_corr, min=-eps)

    # Expand back to batch size for compatibility with loss functions
    return total_corr.index_select(0, group_ids)


def total_correlation_logits(
    logits_dict: Dict[str, torch.Tensor],
    minibatch: int = None,
    eps: float = 1e-8,
    op_type=torch.float32
) -> torch.Tensor:
    """
    Computes the Total Correlation (TC) from logits in log-probability space.
    TC(P) = Sum_i(H(P_i)) - H(P_joint)

    This is numerically more stable than the probability-space version.

    Args:
        logits_dict: Dict of logits {head_name: [Batch, Num_Classes]}.
                     Values are raw classifier outputs (before softmax).
        minibatch:   Size of the group to estimate empirical distribution.
                     If None, uses the whole batch.
        eps:         Numerical stability constant.
        op_type:     dtype for computation.

    Returns:
        torch.Tensor: [Batch] Tensor containing the TC value for each sample's group.
    """
    # 1. Prepare Data
    logits_list = list(logits_dict.values())
    device = logits_list[0].device
    orig_dtype = logits_list[0].dtype
    batch_size = logits_list[0].size(0)

    # Convert to log-probs for each head
    log_probs_list = [F.log_softmax(logits.to(op_type), dim=-1) for logits in logits_list]

    # 2. Setup grouping
    if minibatch is None or minibatch <= 0:
        num_groups = 1
        group_ids = torch.zeros(batch_size, dtype=torch.long, device=device)
    else:
        group_size = min(minibatch, batch_size)
        num_groups = (batch_size + group_size - 1) // group_size
        group_ids = torch.arange(batch_size, device=device) // group_size
        group_ids = torch.clamp(group_ids, max=num_groups - 1)

    # 3. Compute Marginal Entropies per group
    # For each head, compute H(p_emp_g) where p_emp_g is the empirical mean for group g
    sum_marginal_entropy = torch.zeros(num_groups, device=device, dtype=op_type)

    for log_probs in log_probs_list:
        for g in range(num_groups):
            mask = (group_ids == g)
            group_log_probs = log_probs[mask]  # [G_size, K]
            group_count = group_log_probs.size(0)

            # log(p_emp) = logsumexp(log_probs, dim=0) - log(count)
            log_p_emp = torch.logsumexp(group_log_probs, dim=0) - torch.log(
                torch.tensor(group_count, dtype=op_type, device=device)
            )  # [K]

            # H(p_emp) = -sum_k p_emp_k * log(p_emp_k)
            # = -sum_k exp(log_p_emp_k) * log_p_emp_k
            p_emp = torch.exp(log_p_emp)
            entropy = -(p_emp * log_p_emp).sum()
            sum_marginal_entropy[g] = sum_marginal_entropy[g] + entropy

    # 4. Compute Joint Entropy per group
    # Joint log-prob: log(p_joint) = sum_i log(p_i) for each sample
    # Then aggregate to get empirical joint
    joint_entropy = torch.zeros(num_groups, device=device, dtype=op_type)

    # Stack log_probs to compute joint: [B, K1], [B, K2], ... -> joint log-prob
    # For each sample, joint_log_prob[b, i, j, ...] = log_p1[b, i] + log_p2[b, j] + ...
    # This can be done with einsum in log-space by using outer sum

    # Generate einsum equation for log-space (sum instead of product)
    # We need to broadcast and sum: result[b, i, j, k] = log_p1[b, i] + log_p2[b, j] + log_p3[b, k]
    # This is equivalent to: unsqueeze each log_prob appropriately and sum

    # Build joint log-probs with correct broadcasting
    # For head i, reshape to [B, 1, ..., 1, K_i, 1, ..., 1] where K_i is at position i+1
    num_heads = len(log_probs_list)
    joint_log_probs = log_probs_list[0]  # [B, K1]

    for i in range(1, num_heads):
        # Unsqueeze joint_log_probs to add new dimension at end:
        # [B, K1, ..., K_{i-1}] -> [B, K1, ..., K_{i-1}, 1]
        joint_log_probs = joint_log_probs.unsqueeze(-1)

        # Reshape log_probs_list[i] from [B, K_i] to [B, 1, 1, ..., 1, K_i]
        # (i ones after batch dim)
        lp = log_probs_list[i]
        for _ in range(i):
            lp = lp.unsqueeze(1)

        # Now broadcast and add:
        # [B, K1, ..., K_{i-1}, 1] + [B, 1, ..., 1, K_i] -> [B, K1, ..., K_i]
        joint_log_probs = joint_log_probs + lp

    # Result: [B, K1, K2, ..., K_n]

    # joint_log_probs: [B, K1, K2, ..., K_n]
    # Flatten to [B, total_combinations]
    joint_log_probs_flat = joint_log_probs.reshape(batch_size, -1)  # [B, prod(K_i)]

    # For each group, compute empirical joint distribution and its entropy
    for g in range(num_groups):
        mask = (group_ids == g)
        group_joint_log_probs = joint_log_probs_flat[mask]  # [G_size, total_comb]
        group_count = group_joint_log_probs.size(0)

        # log(p_joint_emp) = logsumexp(group_joint_log_probs, dim=0) - log(count)
        log_p_joint_emp = torch.logsumexp(group_joint_log_probs, dim=0) - torch.log(
            torch.tensor(group_count, dtype=op_type, device=device)
        )  # [total_comb]

        # H(p_joint_emp) = -sum exp(log_p) * log_p
        p_joint_emp = torch.exp(log_p_joint_emp)
        joint_entropy[g] = -(p_joint_emp * log_p_joint_emp).sum()

    # 5. TC = Sum(Marginal H) - Joint H
    total_corr = sum_marginal_entropy - joint_entropy

    # Expand back to batch size
    return total_corr.index_select(0, group_ids).to(orig_dtype)


class CostFunction(nn.Module): 
    def __init__(self, classifier:nn.Module, device):
        super(CostFunction, self).__init__()
        self.classifier = classifier.to(device)
        self.classifier.eval()
        self.device = device
        self.dtype = next(classifier.parameters()).dtype
        self.preprocess_transform = lambda x_: x_  # identity by default

    def set_transform(self, transform: Callable):
        self.preprocess_transform = transform

    def forward(self, x, *args, **kwargs):
        """
        Computes the cost for input x to promote target class target.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, C, H, W).
        Returns:
            torch.Tensor: Batch of cost values of shape (batch_size,).
        """
        raise NotImplementedError("Subclasses should implement this method.")


class CostFunctionDummy(CostFunction): 
    def __init__(self, classifier, device): 
        super(CostFunctionDummy, self).__init__(classifier, device)

    def forward(self, x, *args, **kwargs):
        """
        Computes the cost for input x to promote target class target.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, C, H, W).
            target (int): Index of the target class to promote.
        Returns:
            torch.Tensor: Batch of cost values of shape (batch_size,).
        """
        target = kwargs.get("target", 0)  # default target class 0
        logits = self.classifier(x)
        loss = promote_target_class_loss(logits, target)
        return loss


class CostFunctionFairFace(CostFunction): 
    def __init__(self, classifier, device, num_race_cls:int=4): 
        super().__init__(classifier, device)
        self.num_race_cls = num_race_cls

    def forward(self, x, target: int, head_key:str, *args, **kwargs):
        out = infer_fairface(self.classifier, x, self.num_race_cls)
        target_logits = out[head_key]  # [B, num_classes]
        return promote_target_class_loss(target_logits, target)


class CostFunctionPCD(CostFunction):
    def __init__(self,
        classifier:nn.Module,  
        device, 
        classifier_type: str = 'latent', 
    ):
        super().__init__(classifier, device)
        self.classifier_type = deepcopy(classifier_type)
        assert classifier_type in ['latent', 'image'], \
            f"Unsupported classifier_type {classifier_type}. Must be 'latent' or 'image'."
        self.classifier_infer_fn = \
            infer_pcd_classifier if self.classifier_type == 'image' else \
            infer_pcd_latent_classifier 
        self.target_cls = None

    def set_target_cls(self, target_cls: int):
        self.target_cls = target_cls
    
    def forward(self, x, head_key:str, target: Optional[int] = None,  timesteps: Optional[torch.Tensor] = None, *args, **kwargs):
        out = self.classifier_infer_fn(self.classifier, x, [head_key], timesteps=timesteps)
        assert target is not None or self.target_cls is not None, \
            "Either provide target class or set target class using `set_target_cls`."
        if target is None:
            target = self.target_cls
        pred_logits = out[head_key]  # [B, num_classes]

        # import pdb; pdb.set_trace()
        cost = promote_target_class_loss(pred_logits, target) 

        return cost



class CostFunctionKLFairFace(CostFunctionFairFace):
    def __init__(self, classifier:nn.Module, device, num_race_cls:int=4, use_logits_space: bool = True, temperature: float = 1.0):
        super().__init__(classifier, device, num_race_cls)
        self.q_tar = None
        self.log_q_tar = None  # Cached log(q_tar) for logits-space computation
        self.prob_merge_idx_map = None
        self.use_logits_space = use_logits_space
        self.temperature = temperature

    def set_target(self, q_tar: torch.Tensor, eps: float = 1e-8):
        """
        Set the target class distribution for KL divergence cost.

        Args:
            q_tar:  [num_classes]   target class distribution
            eps:    float           small constant for numerical stability in log computation
        """
        self.q_tar = q_tar.to(self.device, dtype=self.dtype)
        # Cache log(q_tar) for logits-space computation
        self.log_q_tar = torch.log(self.q_tar + eps)

    def set_prob_merge_mapping(self, idx_map: torch.Tensor):
        """
        Set the class merge mappings for each head to merge probabilities from more classes into fewer classes.

        Args:
            idx_map:  torch.Tensor   mapping from full classes to merged classes for each head
        """
        self.prob_merge_idx_map = idx_map.clone().detach().to(self.device, dtype=torch.long)

    def forward(self, xT: torch.Tensor, head_key:str, minibatch=None, eps=1e-8, **kwargs) -> torch.Tensor:
        """
        Compute the KL divergence cost between the empirical distribution of class labels
        predicted by the classifier on xT and the target distribution q_tar.

        Args:
            xT:         [B,C,H,W]         final states
            head_key:   str               which head to use from the classifier
            minibatch:  Optional[int]     if provided, use minibatch estimate of KL with given size
            eps:        float             small constant for numerical stability

        Returns:
            kl_cost:    [B]               per-sample KL divergence cost
        """
        assert self.q_tar is not None, "Target distribution `q_tar` must be set."
        out = infer_fairface(self.classifier, xT, self.num_race_cls)
        logits: torch.Tensor = out[head_key]  # [B, num_classes]

        logits = logits / self.temperature

        if self.use_logits_space:
            # Logits-space computation
            if self.prob_merge_idx_map is not None:
                logits = merge_logits(logits.float(), self.prob_merge_idx_map)  # [B, K_merged]
            cost = kl_div_cost_logits(logits.float(), self.log_q_tar, minibatch, eps)  # [B]
        else:
            # Probability-space computation (original behavior)
            probs = F.softmax(logits.float(), dim=-1)  # [B, num_classes]
            if self.prob_merge_idx_map is not None:
                probs = merge_probs(probs, self.prob_merge_idx_map)  # [B, K_merged]
            cost = kl_div_cost(probs, self.q_tar, minibatch, eps)  # [B]

        return cost.to(xT)


class CostFunctionKLMultiDimFairFace(CostFunctionKLFairFace):
    def __init__(self, classifier:nn.Module, device, num_race_cls:int=4, use_logits_space: bool = True):
        super().__init__(classifier, device, num_race_cls, use_logits_space)
        self.heads = []
        self.prob_merge_mappings = {}

    def set_target(self, q_tar: Dict[str, torch.Tensor], eps: float = 1e-8):
        """
        Set the target class distributions for each head for KL divergence cost.

        Args:
            q_tar:  Dict[str, torch.Tensor]   target class distributions for each head
            eps:    float                      small constant for numerical stability in log computation
        """
        self.heads = list(q_tar.keys())
        self.q_tar = {k: v.to(self.device, dtype=self.dtype) for k, v in q_tar.items()}
        # Cache log(q_tar) for each head for logits-space computation
        self.log_q_tar = {k: torch.log(v + eps) for k, v in self.q_tar.items()}

    def set_prob_merge_mapping(self, merge_mappings: Dict[str, torch.Tensor]):
        """
        Set the class merge mappings for each head to merge probabilities from more classes into fewer classes.

        Args:
            merge_mappings:  Dict[str, torch.Tensor]   mapping from full classes to merged classes for each head
        """
        self.prob_merge_mappings = {k: v.to(self.device, dtype=torch.long) for k, v in merge_mappings.items()}

    def forward(self, xT: torch.Tensor, minibatch=None, eps=1e-8, mode='soft', **kwargs) -> torch.Tensor:
        """
        Compute the total KL divergence cost across multiple heads between the empirical distribution of class labels
        predicted by the classifier on xT and the target distributions q_tar.

        Args:
            xT:         [B,C,H,W]         final states
            minibatch:  Optional[int]     if provided, use minibatch estimate of KL with given size
            eps:        float             small constant for numerical stability
            mode:       str               'soft' or 'straight_through' for probability computation

        Returns:
            total_kl_cost:    [B]               per-sample total KL divergence cost across all heads
        """
        assert self.q_tar is not None, "Target distributions `q_tar` must be set."
        assert mode in ['soft', 'straight_through']
        xT = self.preprocess_transform(xT)
        out = infer_fairface(self.classifier, xT, self.num_race_cls)
        unexpected_heads = [head for head in self.heads if head not in out]
        if unexpected_heads:
            raise ValueError(f"Heads {unexpected_heads} not found in classifier output.")

        total_cost = torch.zeros(xT.size(0), device=xT.device, dtype=torch.float32)  # FP32 for numerical stability

        if self.use_logits_space:
            # Logits-space computation
            logits_dict = {}
            for head in self.heads:
                logits: torch.Tensor = out[head].float()  # [B, num_classes]

                # 1) Merge first (if needed)
                if head in self.prob_merge_mappings:
                    # logits = merge_logits(logits, self.prob_merge_mappings[head])  # [B, K_merged]
                    logits = merge_logits_efficient(logits, self.prob_merge_mappings[head])  # [B, K_merged]

                # 2) Apply straight-through in log-space if requested
                if mode == 'soft':
                    logits_final = logits
                elif mode == 'straight_through':
                    # Straight-through: hard argmax with gradient bypass
                    # In log-space: set max logit to 0, others to -inf, then add back original for gradient
                    log_probs = F.log_softmax(logits, dim=-1)
                    hard_idx = log_probs.argmax(dim=1, keepdim=True)
                    log_probs_hard = torch.full_like(log_probs, float('-inf'))
                    log_probs_hard.scatter_(1, hard_idx, 0.0)  # log(1) = 0 for the max
                    logits_final = (log_probs_hard - log_probs).detach() + log_probs  # ST estimator in log-space
                else:
                    raise ValueError(f"Unknown mode: {mode}")

                logits_dict[head] = logits_final
                cost_head = kl_div_cost_logits(logits_final, self.log_q_tar[head], minibatch, eps)  # [B]
                total_cost = total_cost + cost_head

            total_cor = total_correlation_logits(logits_dict, minibatch, eps)  # [B]
        else:
            # Probability-space computation (original behavior)
            probs_dict = {}
            for head in self.heads:
                logits: torch.Tensor = out[head].float()  # [B, num_classes]
                probs_soft = F.softmax(logits, dim=-1)  # [B, num_classes]
                if mode == 'soft':
                    probs = probs_soft
                elif mode == 'straight_through':
                    probs_hard = torch.zeros_like(logits).scatter_(1, logits.argmax(dim=1, keepdim=True), 1.0)
                    probs = (probs_hard - probs_soft).detach() + probs_soft  # straight-through estimator
                else:
                    raise ValueError(f"Unknown mode: {mode}")
                if head in self.prob_merge_mappings:
                    probs = merge_probs(probs, self.prob_merge_mappings[head])  # [B, K_merged]
                probs_dict[head] = probs
                cost_head = kl_div_cost(probs, self.q_tar[head], minibatch, eps)  # [B]
                total_cost = total_cost + cost_head

            total_cor = total_correlation(probs_dict, minibatch, eps)  # [B]

        print(f"Marginal costs: {total_cost.detach().mean().item()}")
        print(f"Total Correlation cost: {total_cor.detach().mean().item()}")
        total_cost = total_cost + total_cor

        return total_cost.to(xT)



class CostFunctionKLMultiDimPCD(CostFunction):
    def __init__(self,
        classifier: nn.Module,
        device,
        classifier_type: str = 'image',
        use_logits_space: bool = True,
        temperature: float = 1.0,
        # use_logits_space: bool = False,
    ):
        super().__init__(classifier, device)
        self.heads = []
        self.prob_merge_mappings = {}
        self.q_tar = None
        self.log_q_tar = None  # Cached log(q_tar) for logits-space computation
        self.use_logits_space = use_logits_space
        self.temperature = temperature
        assert classifier_type in ['latent', 'image'], \
            f"Unsupported classifier_type {classifier_type}. Must be 'latent' or 'image'."
        self.classifier_type = deepcopy(classifier_type)
        self.classifier_infer_fn = \
            infer_pcd_classifier if self.classifier_type == 'image' else \
            infer_pcd_latent_classifier

    def set_target(self, q_tar: Dict[str, torch.Tensor], eps: float = 1e-8):
        """
        Set the target class distributions for each head for KL divergence cost.

        Args:
            q_tar:  Dict[str, torch.Tensor]   target class distributions for each head
            eps:    float                      small constant for numerical stability in log computation
        """
        self.heads = list(q_tar.keys())
        self.q_tar = {k: v.to(self.device, dtype=self.dtype) for k, v in q_tar.items()}
        # Cache log(q_tar) for each head for logits-space computation
        self.log_q_tar = {k: torch.log(v + eps) for k, v in self.q_tar.items()}

    def set_prob_merge_mapping(self, merge_mappings: Dict[str, torch.Tensor]):
        """
        Set the class merge mappings for each head to merge probabilities from more classes into fewer classes.

        Args:
            merge_mappings:  Dict[str, torch.Tensor]   mapping from full classes to merged classes for each head
        """
        self.prob_merge_mappings = {k: v.to(self.device, dtype=torch.long) for k, v in merge_mappings.items()}
        self.merged_cls_dims = {k: v.max().item() + 1 for k, v in self.prob_merge_mappings.items()}

    def forward(
        self,
        xT: torch.Tensor,
        minibatch=None,
        timesteps=None,
        eps=1e-8,
        mode='soft',
        reverse_kl: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute the total KL divergence cost across multiple heads between the empirical distribution of class labels
        predicted by the classifier on xT and the target distributions q_tar.

        Args:
            xT:         [B,C,H,W]         final states
            minibatch:  Optional[int]     if provided, use minibatch estimate of KL with given size
            timesteps:  Optional[Tensor]  timesteps for latent classifier
            eps:        float             small constant for numerical stability
            mode:       str               'soft' or 'straight_through' for probability computation
            reverse_kl: bool              if True, use reverse KL divergence

        Returns:
            total_kl_cost:    [B]               per-sample total KL divergence cost across all heads
        """
        assert self.q_tar is not None, "Target distributions `q_tar` must be set."
        assert mode in ['soft', 'straight_through']
        print_fn = kwargs.get('print_fn', print)
        out = self.classifier_infer_fn(self.classifier, xT, self.heads, timesteps=timesteps)
        unexpected_heads = [head for head in self.heads if head not in out]
        if unexpected_heads:
            raise ValueError(f"Heads {unexpected_heads} not found in classifier output.")

        total_cost = torch.zeros(xT.size(0), device=xT.device, dtype=torch.float32)  # FP32 for numerical stability

        if self.use_logits_space:
            # Logits-space computation
            logits_dict = {}
            for head in self.heads:
                logits: torch.Tensor = out[head].float()  # [B, num_classes]
                logits = logits / self.temperature

                # 1) Merge first (if needed)
                if head in self.prob_merge_mappings:
                    # logits = merge_logits(logits, self.prob_merge_mappings[head])  # [B, K_merged]
                    logits = merge_logits_efficient(logits, self.prob_merge_mappings[head], self.merged_cls_dims[head])  # [B, K_merged]

                # 2) Apply straight-through in log-space if requested
                if mode == 'soft':
                    logits_final = logits
                elif mode == 'straight_through':
                    # Straight-through: hard argmax with gradient bypass in log-space
                    log_probs = F.log_softmax(logits, dim=-1)
                    hard_idx = log_probs.argmax(dim=1, keepdim=True)
                    log_probs_hard = torch.full_like(log_probs, float('-inf'))
                    log_probs_hard.scatter_(1, hard_idx, 0.0)  # log(1) = 0 for the max
                    logits_final = (log_probs_hard - log_probs).detach() + log_probs  # ST estimator
                else:
                    raise ValueError(f"Unknown mode: {mode}")

                logits_dict[head] = logits_final
                if reverse_kl:
                    cost_head = reverse_kl_div_cost_logits(logits_final, self.log_q_tar[head], minibatch, eps)
                else:
                    cost_head = kl_div_cost_logits(logits_final, self.log_q_tar[head], minibatch, eps)
                total_cost = total_cost + cost_head
            
            if reverse_kl:
                print_fn("[WARNING] Reverse KL intractable. Assuming dimension independence. ")
                total_cor = torch.zeros_like(total_cost)
            else:
                total_cor = total_correlation_logits(logits_dict, minibatch, eps)  # [B]
        else:
            # Probability-space computation (original behavior)
            probs_dict = {}
            for head in self.heads:
                logits: torch.Tensor = out[head].float()  # [B, num_classes]
                logits = logits / self.temperature
                probs_soft_full = F.softmax(logits, dim=-1)  # [B, num_classes]

                # 1) Merge first (if needed)
                if head in self.prob_merge_mappings:
                    probs_soft = merge_probs(probs_soft_full, self.prob_merge_mappings[head])
                else:
                    probs_soft = probs_soft_full

                # 2) Apply straight-through if requested
                if mode == 'soft':
                    probs = probs_soft
                elif mode == 'straight_through':
                    hard_idx = probs_soft.argmax(dim=1, keepdim=True)
                    probs_hard = torch.zeros_like(probs_soft).scatter_(1, hard_idx, 1)
                    probs = (probs_hard - probs_soft).detach() + probs_soft
                else:
                    raise ValueError(f"Unknown mode: {mode}")

                probs_dict[head] = probs
                if reverse_kl:
                    cost_head = reverse_kl_div_cost(probs, self.q_tar[head], minibatch, eps)
                else:
                    cost_head = kl_div_cost(probs, self.q_tar[head], minibatch, eps)
                total_cost = total_cost + cost_head

            if reverse_kl:
                print_fn("[WARNING] Reverse KL intractable. Assuming dimension independence. ")
                total_cor = torch.zeros_like(total_cost)
            else:
                total_cor = total_correlation(probs_dict, minibatch, eps)  # [B]
        #end if use_logits_space
        print_fn(f"Marginal costs: {total_cost.detach().mean().item()}")
        print_fn(f"Total Correlation cost: {total_cor.detach().mean().item()}")
        total_cost = total_cost + total_cor

        return total_cost.to(xT)


class AnnealedCostFunction(nn.Module):
    """
    Wraps a cost function to apply temperature annealing and cost scaling.
    """
    def __init__(
        self,
        cost_fn: CostFunction,
        temp_schedule: str = 'constant',
        temp_start: float = 1.0,
        temp_end: float = 1.0,
        cost_scale: float = 1.0,
        compensation_strategy: str = 'none',  # 'none' or 'linear'
        total_iters: int = 1,
        verbose: bool = False,
    ):
        super().__init__()
        self.cost_fn = cost_fn
        self.temp_schedule = temp_schedule
        self.temp_start = temp_start
        self.temp_end = temp_end
        self.cost_scale = cost_scale
        self.compensation_strategy = compensation_strategy
        self.verbose = verbose
        self.reset_annealing_schedule(total_iters)

    def reset_annealing_schedule(self, total_iters: int):
        self.total_iters = total_iters
        self._precompute_schedule(total_iters)

    def _precompute_schedule(self, total_iters: int):
        if total_iters <= 1 or self.temp_schedule == 'constant':
            self.cached_temps = np.full(max(1, total_iters), self.temp_start, dtype=np.float32)
            return

        iterations = np.arange(total_iters)
        progress = iterations / (total_iters - 1)
        # Ensure numerical stability
        progress = np.clip(progress, 0.0, 1.0)

        if self.temp_schedule == 'linear':
            self.cached_temps = self.temp_start + progress * (self.temp_end - self.temp_start)
        
        elif self.temp_schedule == 'cosine':
            # Cosine decay: starts slow, accelerates, then slows down
            # T = end + (start - end) * 0.5 * (1 + cos(pi * progress))
            cosine_factor = 0.5 * (1 + np.cos(progress * np.pi)) # 1 -> 0
            self.cached_temps = self.temp_end + (self.temp_start - self.temp_end) * cosine_factor
        
        else:
            raise ValueError(f"Unknown temperature schedule: {self.temp_schedule}")

    def forward(self, x, **kwargs):
        iteration = kwargs.get('iteration', 0)
        print_fn = kwargs.get('print_fn', print)
        
        # 1. Retrieve Temperature
        idx = max(0, min(iteration, self.total_iters - 1))
        current_temp = float(self.cached_temps[idx])

        # 2. Update Cost Function Temperature (if it supports it)
        if hasattr(self.cost_fn, 'temperature'):
            self.cost_fn.temperature = current_temp
        
        # 3. Calculate Cost Scale
        effective_scale = self.cost_scale
        if self.compensation_strategy == 'linear':
            effective_scale *= current_temp
        
        # 4. Compute Base Cost
        cost = self.cost_fn(x, **kwargs)

        # 5. Apply Scaling
        if self.verbose: 
            # Print once per outer iter approx (imperfect check but useful for debug)
            print_fn(f"[Anneal] Iter {iteration}/{self.total_iters}: Temp={current_temp:.3f}, Scale={effective_scale:.3f}")

        return cost * effective_scale
