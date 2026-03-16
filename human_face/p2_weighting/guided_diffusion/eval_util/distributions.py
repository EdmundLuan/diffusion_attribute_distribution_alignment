import numpy as np
import torch

def uniform(K, *args, **kwargs):
    dist = torch.ones(K)
    return dist / dist.sum()

def zigzag(K, ratio, *args, **kwargs):
    dist = torch.ones(K)
    for i in range(0, K, 2):
        dist[i] = ratio
    return dist / dist.sum()

def gaussian_peak(K, sigma, *args, **kwargs):
    c = (K-1)/2
    logits = -0.5 * ((torch.arange(K) - c) / sigma) ** 2
    dist = torch.exp(logits - torch.logsumexp(logits, -1))
    return dist / dist.sum()


dist_registry = {
    'uniform': uniform,
    'zigzag': zigzag,
    'gaussian': gaussian_peak,
}
