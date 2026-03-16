import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
import os
import yaml
import collections
import numpy as np
from typing import Dict, Callable, List, Tuple, Optional


class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        ## TODO: about the size for assertion. 
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        if len(size) == 1:
            return torch.stack([torch.randint(*args, size=(), generator=gen, **kwargs) for gen in self.generators])
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])



class FrozenBatchNorm2d(nn.Module):
    def __init__(self, bn: nn.BatchNorm2d):
        super().__init__()
        self.register_buffer("weight", bn.weight.detach().clone())
        self.register_buffer("bias",   bn.bias.detach().clone())
        self.register_buffer("running_mean", bn.running_mean.detach().clone())
        self.register_buffer("running_var",  bn.running_var.detach().clone())
        self.eps = bn.eps

    def forward(self, x):
        w = self.weight.view(1, -1, 1, 1)
        b = self.bias.view(1, -1, 1, 1)
        rm = self.running_mean.view(1, -1, 1, 1)
        rv = self.running_var.view(1, -1, 1, 1)
        return (x - rm) * (w / (rv + self.eps).sqrt()) + b

def convert_bn_to_frozen(model: nn.Module) -> nn.Module:
    for name, m in list(model.named_children()):
        if isinstance(m, nn.BatchNorm2d):
            setattr(model, name, FrozenBatchNorm2d(m))
        else:
            convert_bn_to_frozen(m)
    return model

def make_deterministic_classifier(classifier:torch.nn.Module):
    convert_bn_to_frozen(classifier)
    for mod in classifier.modules():
        if isinstance(mod, nn.ReLU):
            mod.inplace = False  # avoid in-place autograd surprises; deterministic either way
    return classifier




def merge_probs(probs: torch.Tensor, merge_mapping: torch.Tensor) -> torch.Tensor:
    """
    Merge probabilities from more classes into fewer classes if needed, preserving gradients.
    
    Args:
        probs:  [B, K_full]   classifier output probabilities with full classes
        merge_mapping:  [K_full]   mapping from full classes to merged classes

    Returns: 
        merged_probs:  [B, K_merged]   merged probabilities
    """
    K_merged = merge_mapping.max().item() + 1

    one_hot = F.one_hot(merge_mapping, num_classes=K_merged).to(probs)  # [K_full, K_merged]
    merged_probs = torch.einsum('bk,km->bm', probs, one_hot)
    return merged_probs


def merge_probs_efficient(probs: torch.Tensor, merge_mapping: torch.Tensor) -> torch.Tensor:
    K_merged = merge_mapping.max().item() + 1
    B = probs.size(0)

    # Initialize output tensor
    merged_probs = torch.zeros((B, K_merged), device=probs.device, dtype=probs.dtype)

    # Expand mapping to match batch size: [B, K_full]
    # This does not copy data, it just creates views.
    index = merge_mapping.view(1, -1).expand(B, -1)

    # Scatter the values by adding them to the target indices
    merged_probs.scatter_add_(1, index, probs)

    return merged_probs


def merge_logits(logits: torch.Tensor, merge_mapping: torch.Tensor) -> torch.Tensor:
    """
    Merge logits from more classes into fewer classes using log-sum-exp.

    This is the logits-space equivalent of merge_probs. Since:
        merged_prob[m] = sum_{k in group_m} prob[k]
        log(merged_prob[m]) = logsumexp(logits[k] for k in group_m)

    Args:
        logits:  [B, K_full]   classifier output logits (unnormalized log-probs)
        merge_mapping:  [K_full]   mapping from full classes to merged classes (0-indexed)

    Returns:
        merged_logits:  [B, K_merged]   merged logits (unnormalized log-probs)
    """
    K_merged = merge_mapping.max().item() + 1
    B = logits.size(0)

    # Build a mask for each merged class: [K_merged, K_full]
    # mask[m, k] = 1 if merge_mapping[k] == m
    mask = F.one_hot(merge_mapping, num_classes=K_merged).T.bool()  # [K_merged, K_full]

    # For each merged class, compute logsumexp over the corresponding original classes
    # Result: [B, K_merged]
    merged_logits = torch.zeros((B, K_merged), device=logits.device, dtype=logits.dtype)
    for m in range(K_merged):
        # Select logits belonging to group m: [B, num_in_group]
        group_logits = logits[:, mask[m]]
        # logsumexp over the group dimension
        merged_logits[:, m] = torch.logsumexp(group_logits, dim=1)

    return merged_logits


def merge_logits_efficient(logits: torch.Tensor, merge_mapping: torch.Tensor, reduced_dim_size:int=None) -> torch.Tensor:
    """
    Merge logits using vectorized scatter-based logsumexp (no Python loops).

    This is more efficient than merge_logits for large K_merged.

    Args:
        logits:  [B, K_full]   classifier output logits
        merge_mapping:  [K_full]   mapping from full classes to merged classes

    Returns:
        merged_logits:  [B, K_merged]   merged logits
    """
    if reduced_dim_size is None:
        K_merged = merge_mapping.max().item() + 1
    else:
        K_merged = reduced_dim_size
    B = logits.size(0)

    # Expand mapping: [B, K_full]
    index = merge_mapping.view(1, -1).expand(B, -1)

    # Step 1: Find max per merged group using scatter_reduce with 'amax'
    min_val = torch.finfo(logits.dtype).min
    max_vals:torch.Tensor = torch.full((B, K_merged), min_val, device=logits.device, dtype=logits.dtype)
    max_vals.scatter_reduce_(1, index, logits, reduce='amax', include_self=False)

    # Step 2: Compute exp(logits - max) for numerical stability
    # Expand max_vals back to K_full using gather
    max_expanded = max_vals.gather(1, index)  # [B, K_full]
    exp_shifted = torch.exp(logits - max_expanded)

    # Step 3: Sum exp values per merged group
    sum_exp:torch.Tensor = torch.zeros((B, K_merged), device=logits.device, dtype=logits.dtype)
    sum_exp.scatter_add_(1, index, exp_shifted)

    # Step 4: merged_logits = max + log(sum_exp)
    merged_logits = max_vals + torch.log(sum_exp)

    return merged_logits


def print_peak_memory(header="...", print_fn=print):
    max_mem = torch.cuda.max_memory_allocated() / 1024**3
    print_fn(f"[{header}] Peak Memory: {max_mem:.2f} GB")
    # Reset stats so we see the contribution of the NEXT step only
    torch.cuda.reset_peak_memory_stats()


def now_tag() -> str:
    # For run disambiguation; local time
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def save_configs(save_dir: str, args) -> None:
    os.makedirs(save_dir, exist_ok=True)
    args_dict = vars(args)
    with open(os.path.join(save_dir, "configs.yaml"), "w") as f:
        yaml.dump(args_dict, f)


def get_support_size(attribute_key: str) -> int:
    support_sizes = {
        # "age_group": 9,
        "age_group": 3,
        "race": 4, 
        "gender": 2, 
    }

    return support_sizes[attribute_key]

def get_target_dist(head:str) -> torch.Tensor:
    # K = 4 if head == "race" else \
    #     2 if head == "gender" else \
    #     9 if head == "age_group" else None
    # assert K is not None, f"Unknown head key: {head}"
    K = get_support_size(head)
    target_dist = torch.ones(K,) / K  # Uniform distribution
    return target_dist


def parse_target_distribution(
    target_dist_config: Optional[Dict[str, List[float]]],
    classifier_heads: List[str],
    warn_fn: Callable[[str], None] = print,
) -> Dict[str, torch.Tensor]:
    """
    Parse and validate target distributions from config.

    Args:
        target_dist_config: Dict mapping head names to probability lists.
                           Example: {"gender": [0.6, 0.4], "race": [0.3, 0.3, 0.2, 0.2]}
        classifier_heads: List of classifier head names being used.
        warn_fn: Function for warning messages.

    Returns:
        Dict mapping each head to a normalized torch.Tensor.

    Raises:
        ValueError: If distribution length doesn't match support size.
    """
    result = {}

    # Handle None or empty config
    if not target_dist_config:
        warn_fn("[WARNING] No 'target_distribution' in config. Using uniform for all heads.")
        return {head: get_target_dist(head) for head in classifier_heads}

    for head in classifier_heads:
        if head in target_dist_config:
            dist_list = target_dist_config[head]
            expected_size = get_support_size(head)

            # Validate size
            if len(dist_list) != expected_size:
                raise ValueError(
                    f"Target distribution for '{head}' has {len(dist_list)} elements, "
                    f"but expected {expected_size} (support size for {head})"
                )

            # Convert and normalize
            dist_tensor = torch.tensor(dist_list, dtype=torch.float32)
            dist_tensor = dist_tensor / dist_tensor.sum()
            result[head] = dist_tensor
        else:
            warn_fn(f"[WARNING] Head '{head}' not in target_distribution. Using uniform.")
            result[head] = get_target_dist(head)

    return result


def load_target_config(config_path: str) -> Dict[str, List[float]]:
    """
    Load target distribution config from JSON file.

    Expected JSON format (same as configs/target_distributions/*.json):
    {
        "name": "config_name",
        "support_sizes": {"age_group": 3, "gender": 2, "race": 4},
        "target_distribution": {
            "age_group": null,  // null means uniform
            "gender": [0.2, 0.8],
            "race": [0.4, 0.3, 0.2, 0.1]
        }
    }

    Args:
        config_path: Path to JSON config file.

    Returns:
        Dict mapping attribute names to probability lists.

    Raises:
        ValueError: If config is missing required keys.
    """
    with open(config_path, 'r') as f:
        config = json.load(f)

    if "target_distribution" not in config:
        raise ValueError(f"Config file missing 'target_distribution' key: {config_path}")

    target_dist = config["target_distribution"]
    support_sizes = config.get("support_sizes", {"gender": 2, "age_group": 3, "race": 4})

    result = {}
    for attr, value in target_dist.items():
        if value is None:
            # null → uniform distribution
            size = support_sizes.get(attr)
            if size is None:
                print(f"Warning: Attribute '{attr}' has null value but no support_size, skipping")
                continue
            result[attr] = [1.0 / size] * size
        else:
            # Array → normalize
            probs = np.array(value, dtype=np.float64)
            result[attr] = (probs / probs.sum()).tolist()

    return result


def get_merge_map(head:str, classifier_name:str) -> torch.Tensor:
    if classifier_name.startswith("fairface"):
        if head == "age_group":
            ## OLD threshold: 60
            # merge_map = torch.tensor([
            #     0,0,0, 
            #     1,1,1,1,
            #     2,2, 
            # ])
            ## OLD threshold: 50
            merge_map = torch.tensor([
                0,0,0, 
                1,1,1,
                2,2,2, 
            ])
        elif head == "race_7":
            merge_map = torch.tensor([
                0,  # white -> WMELH
                1,  # black -> black
                0,  # latino_hispanic -> wmelh
                2,  # east_asian -> asian
                2,  # southeast_asian -> asian
                3,  # indian -> indian
                0   # middle_eastern -> wmelh
            ])
        else:
            raise ValueError(f"Not supported head key: {head}")
    elif classifier_name.startswith("pcd"):
        if head == "age_group":
            merge_map = torch.tensor([
                0, 0, 0, 0, 0, 
                1, 1, 1, 
                2, 2,
            ])
        else:
            raise ValueError(f"Not supported head key: {head}")
    else: 
        raise ValueError(f"Not supported classifier name: {classifier_name}")
    return merge_map

#######################################################
######### PyTorch Utils ###############################
#######################################################

def dict_apply(
        x: Dict[str, torch.Tensor], 
        func: Callable[[torch.Tensor], torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result

def pad_remaining_dims(x, target):
    assert x.shape == target.shape[:len(x.shape)]
    return x.reshape(x.shape + (1,)*(len(target.shape) - len(x.shape)))

def dict_apply_split(
        x: Dict[str, torch.Tensor], 
        split_func: Callable[[torch.Tensor], Dict[str, torch.Tensor]]
        ) -> Dict[str, torch.Tensor]:
    results = collections.defaultdict(dict)
    for key, value in x.items():
        result = split_func(value)
        for k, v in result.items():
            results[k][key] = v
    return results

def dict_apply_reduce(
        x: List[Dict[str, torch.Tensor]],
        reduce_func: Callable[[List[torch.Tensor]], torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
    result = dict()
    for key in x[0].keys():
        result[key] = reduce_func([x_[key] for x_ in x])
    return result


def map_tensor_range(
    tensor: torch.Tensor,
    in_range: Tuple[float, float],
    out_range: Tuple[float, float]
) -> torch.Tensor:
    """
    Linearly map a tensor from in_range [a, b] to out_range [c, d].
    """
    a, b = in_range
    c, d = out_range
    a, b, c, d = float(a), float(b), float(c), float(d)
    
    ret = ((tensor.float() - a) / (b - a)) * (d - c) + c
    return ret.to(tensor)


def compute_joint_probs(
    probs_dict: Dict[str, torch.Tensor]
) -> Tuple[str, torch.Tensor]:
    """
    Compute joint probability distribution as the outer product of marginals.

    Args:
        probs_dict: Dict mapping attribute names to probability tensors [B, K_i]

    Returns:
        Tuple of (joint_key, joint_probs):
        - joint_key: Attribute names joined by '-' after sorting alphabetically
        - joint_probs: Tensor [B, prod(K_i)] with joint probabilities
    """
    # Sort attribute keys alphabetically
    sorted_keys = sorted(probs_dict.keys())
    joint_key = "-".join(sorted_keys)

    # Get batch size from first attribute
    B = probs_dict[sorted_keys[0]].shape[0]

    # Compute outer product iteratively
    joint = probs_dict[sorted_keys[0]]  # [B, K_0]
    for key in sorted_keys[1:]:
        marginal = probs_dict[key]  # [B, K_i]
        # Outer product: [B, K_joint, 1] * [B, 1, K_i] -> [B, K_joint, K_i]
        joint = joint.unsqueeze(-1) * marginal.unsqueeze(1)
        joint = joint.view(B, -1)  # Flatten to [B, K_joint * K_i]

    return joint_key, joint


def compute_joint_target_dist(
    target_dists: Dict[str, torch.Tensor]
) -> Tuple[str, torch.Tensor]:
    """
    Compute joint target distribution as the outer product of marginal targets.

    Args:
        target_dists: Dict mapping attribute names to target distribution tensors [K_i]

    Returns:
        Tuple of (joint_key, joint_target):
        - joint_key: Attribute names joined by '-' after sorting alphabetically
        - joint_target: Tensor [prod(K_i)] with joint target distribution
    """
    # Sort attribute keys alphabetically
    sorted_keys = sorted(target_dists.keys())
    joint_key = "-".join(sorted_keys)

    # Compute outer product iteratively
    joint = target_dists[sorted_keys[0]]  # [K_0]
    for key in sorted_keys[1:]:
        marginal = target_dists[key]  # [K_i]
        # Outer product: [K_joint, 1] * [1, K_i] -> [K_joint, K_i]
        joint = joint.unsqueeze(-1) * marginal.unsqueeze(0)
        joint = joint.view(-1)  # Flatten to [K_joint * K_i]

    return joint_key, joint

