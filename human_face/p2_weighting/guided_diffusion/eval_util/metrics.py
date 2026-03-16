import numpy as np
import torch
from typing import Union


# ---------------------------------------------------------------------
# Basic metrics
def kl_div(p: Union[np.ndarray, torch.Tensor], q: Union[np.ndarray, torch.Tensor], eps: float = 1e-10) -> float:
    """Calculate KL divergence between two distributions.
    
    Args:
        p: First probability distribution (numpy array or torch tensor).
        q: Second probability distribution (numpy array or torch tensor).
        eps: Small epsilon value to avoid log(0). Defaults to 1e-10.
    
    Returns:
        KL divergence between p and q.
    
    Raises:
        AssertionError: If p and q are not of the same type.
    """
    assert type(p) == type(q), "p and q must be of the same type"
    
    if isinstance(p, torch.Tensor):
        p = torch.clamp(p, eps, 1)
        q = torch.clamp(q, eps, 1)
        return float(torch.abs(torch.sum(p * (torch.log(p) - torch.log(q)))).item())
    else:
        p = np.clip(p, eps, 1)
        q = np.clip(q, eps, 1)
        return float(np.abs(np.sum(p * (np.log(p) - np.log(q)))))

def tv_dist(p: Union[np.ndarray, torch.Tensor], q: Union[np.ndarray, torch.Tensor]) -> float:
    """Calculate total variation distance between two distributions.
    
    Args:
        p: First probability distribution (numpy array or torch tensor).
        q: Second probability distribution (numpy array or torch tensor).
    
    Returns:
        Total variation distance between p and q.
    
    Raises:
        AssertionError: If p and q are not of the same type.
    """
    assert type(p) == type(q), "p and q must be of the same type"
    
    if isinstance(p, torch.Tensor):
        return float((0.5 * torch.abs(p - q).sum()).item())
    else:
        return float(0.5 * np.abs(p - q).sum())

def l2_dist(p: Union[np.ndarray, torch.Tensor], q: Union[np.ndarray, torch.Tensor]) -> float:
    """Calculate L2 distance between two distributions.
    
    Args:
        p: First probability distribution (numpy array or torch tensor).
        q: Second probability distribution (numpy array or torch tensor).
    
    Returns:
        L2 distance between p and q.
    
    Raises:
        AssertionError: If p and q are not of the same type.
    """
    assert type(p) == type(q), "p and q must be of the same type"
    
    if isinstance(p, torch.Tensor):
        return float(torch.sqrt(torch.square(p - q).sum()).item())
    else:
        return float(np.sqrt(np.square(p - q).sum()))

def js_dist(p: Union[np.ndarray, torch.Tensor], q: Union[np.ndarray, torch.Tensor], eps: float = 1e-10) -> float:
    """Calculate Jensen-Shannon distance between two distributions.
    
    Args:
        p: First probability distribution (numpy array or torch tensor).
        q: Second probability distribution (numpy array or torch tensor).
        eps: Small epsilon value to avoid log(0). Defaults to 1e-10.
    
    Returns:
        Jensen-Shannon distance between p and q.
    
    Raises:
        AssertionError: If p and q are not of the same type.
    """
    assert type(p) == type(q), "p and q must be of the same type"
    
    if isinstance(p, torch.Tensor):
        p = torch.clamp(p, eps, 1)
        q = torch.clamp(q, eps, 1)
        m = 0.5 * (p + q)
    else:
        p = np.clip(p, eps, 1)
        q = np.clip(q, eps, 1)
        m = 0.5 * (p + q)
    
    kl_pm = kl_div(p, m)
    kl_qm = kl_div(q, m)
    js = 0.5 * (kl_pm + kl_qm)
    return float(np.sqrt(js))

def chi2_dist(p: Union[np.ndarray, torch.Tensor], q: Union[np.ndarray, torch.Tensor], eps: float = 1e-10) -> float:
    """Calculate chi-square distance between two distributions.
    
    Args:
        p: First probability distribution (numpy array or torch tensor).
        q: Second probability distribution (numpy array or torch tensor).
        eps: Small epsilon value to avoid division by zero. Defaults to 1e-10.
    
    Returns:
        Chi-square distance between p and q.
    
    Raises:
        AssertionError: If p and q are not of the same type.
    """
    assert type(p) == type(q), "p and q must be of the same type"
    
    if isinstance(p, torch.Tensor):
        p = torch.clamp(p, eps, 1)
        q = torch.clamp(q, eps, 1)
        return float((0.5 * torch.sum((p - q) ** 2 / (p + q + eps))).item())
    else:
        p = np.clip(p, eps, 1)
        q = np.clip(q, eps, 1)
        return float(0.5 * np.sum((p - q) ** 2 / (p + q + eps)))

def fairness_discrepancy(p_softmax: Union[np.ndarray, torch.Tensor], q: Union[np.ndarray, torch.Tensor]) -> float:
    """Calculate Fairness Discrepancy (FD) between two distributions.
    
    FD := || p_target - E[p_softmax] ||_2
    
    Args:
        p_softmax: The expected softmax output (numpy array or torch tensor).
        q: Target probability distribution (numpy array or torch tensor).
    
    Returns:
        Fairness Discrepancy between p_softmax and q.
    
    Raises:
        AssertionError: If p_softmax and q are not of the same type.
    """
    assert type(p_softmax) == type(q), "p_softmax and q must be of the same type"
    
    if isinstance(p_softmax, torch.Tensor):
        return float(torch.sqrt(torch.square(p_softmax - q).sum()).item())
    else:
        return float(np.sqrt(np.square(p_softmax - q).sum()))

# ---------------------------------------------------------------------
