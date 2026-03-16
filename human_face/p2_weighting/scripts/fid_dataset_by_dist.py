"""
Compute FID reference statistics for FFHQ-Aging dataset with custom joint distributions.

This script generates reference inception statistics for FFHQ-Aging images sampled
according to specified marginal distributions over gender, age_group, and race.
The joint distribution is computed as the product of marginals.

Usage:
    python scripts/fid_dataset_by_dist.py \
        --images_dir data/ffhq_aging/images/images512x512 \
        --labels_json data/ffhq_aging/labels/ffhq_aging_labels_with_race.json \
        --dest outputs/fid/ffhq_aging_uniform.npz \
        --num_samples 10000 \
        --gender_dist uniform \
        --age_dist uniform \
        --race_dist uniform \
        --seed 0 --batch 64

Distribution specifications:
    - "uniform": Uniform distribution
    - "gaussian": Gaussian peak (requires --{attr}_dist_param for sigma)
    - "zigzag": Zigzag pattern (requires --{attr}_dist_param for ratio)
    - Comma-separated: e.g., "0.5,0.5" for custom probabilities
"""

import argparse
import json
import os
from typing import Dict, Optional
import numpy as np
import torch as th
from guided_diffusion.eval_util.ffhq_aging_dataset import (
    FFHQAgingDataset,
    subset_by_joint_marginals,
    ATTRIBUTE_SIZES,
)
from guided_diffusion.eval_util.fid import calculate_inception_stats_from_dataset
from guided_diffusion.eval_util.distributions import dist_registry


def parse_distribution(dist_str: str, size: int, param: float = 1.0) -> np.ndarray:
    """
    Parse distribution specification into probability array.

    Args:
        dist_str: Distribution name ("uniform", "gaussian", "zigzag") or comma-separated probs
        size: Number of classes
        param: Distribution parameter (sigma for gaussian, ratio for zigzag)

    Returns:
        Probability array of length `size` that sums to 1.0
    """
    if dist_str in dist_registry:
        if dist_str == "uniform":
            probs = dist_registry[dist_str](size)
        elif dist_str == "gaussian":
            probs = dist_registry["gaussian"](size, param)
        elif dist_str == "zigzag":
            probs = dist_registry["zigzag"](size, param)
        else:
            raise ValueError(f"Unknown distribution: {dist_str}")
        return probs.numpy()
    else:
        # Parse comma-separated probabilities
        probs = np.array([float(x) for x in dist_str.split(",")])
        if len(probs) != size:
            raise ValueError(f"Expected {size} probabilities for distribution, got {len(probs)}")
        return probs / probs.sum()


def load_distribution_config(config_path: str) -> Dict[str, Optional[np.ndarray]]:
    """
    Load distribution config from JSON file.

    Args:
        config_path: Path to JSON config file.

    Returns:
        Dict mapping attribute names to either:
        - numpy array of probabilities, or
        - None (indicating uniform distribution should be used)

    Expected JSON format:
    {
        "name": "config_name",
        "support_sizes": {"age_group": 3, "gender": 2, "race": 4},
        "target_distribution": {
            "age_group": null,  // null means uniform
            "gender": [0.2, 0.8],
            "race": [0.4, 0.3, 0.2, 0.1]
        }
    }
    """
    with open(config_path, 'r') as f:
        config:Dict = json.load(f)

    if "target_distribution" not in config:
        raise ValueError(f"Config file missing 'target_distribution' key: {config_path}")

    target_dist = config["target_distribution"]
    support_sizes:Dict = config.get("support_sizes", ATTRIBUTE_SIZES)

    result = {}
    for attr in ["gender", "age_group", "race"]:
        if attr not in target_dist:
            # Attribute not specified in config - will use default
            result[attr] = None
        elif target_dist[attr] is None:
            # Explicitly null - use uniform based on support_sizes
            size = support_sizes.get(attr, ATTRIBUTE_SIZES.get(attr))
            result[attr] = np.ones(size) / size
        else:
            # Array specified - convert to numpy and normalize
            probs = np.array(target_dist[attr], dtype=np.float64)
            result[attr] = probs / probs.sum()

    return result


def was_cli_specified(args: argparse.Namespace, arg_name: str, default_value: str) -> bool:
    """
    Check if a CLI argument was explicitly provided (differs from default).

    Args:
        args: Parsed arguments namespace.
        arg_name: Name of the argument to check.
        default_value: The parser default for this argument.

    Returns:
        True if the argument value differs from default.
    """
    return getattr(args, arg_name) != default_value


def main():
    parser = argparse.ArgumentParser(
        description="Compute FID reference statistics for FFHQ-Aging with custom distributions"
    )
    parser.add_argument("--images_dir", required=True, help="Path to images directory")
    parser.add_argument("--labels_json", required=True, help="Path to labels JSON")
    parser.add_argument("--dest", required=True, help="Output .npz file path")
    parser.add_argument("--num_samples", type=int, required=True, help="Number of samples to use")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--batch", type=int, default=64, help="Batch size for inception model")
    parser.add_argument(
        "--target_config", type=str, default=None,
        help="Path to JSON config file with target distributions. CLI arguments override config values."
    )

    # Distribution specifications
    parser.add_argument("--gender_dist", default="uniform", help="Gender distribution")
    parser.add_argument("--age_dist", default="uniform", help="Age group distribution")
    parser.add_argument("--race_dist", default="uniform", help="Race distribution")
    parser.add_argument("--gender_dist_param", type=float, default=1.0, help="Gender dist param")
    parser.add_argument("--age_dist_param", type=float, default=2.5, help="Age dist param (sigma)")
    parser.add_argument("--race_dist_param", type=float, default=2.0, help="Race dist param")

    parser.add_argument("--allow_oversample", action="store_true", help="Allow sampling with replacement")
    parser.add_argument("--device", default="cuda", help="Device for inception model")

    args = parser.parse_args()

    # Parse marginal distributions with priority: CLI > JSON config > defaults
    marginals = {}

    # Load config if specified
    config_marginals = {}
    if args.target_config:
        print(f"Loading distribution config from {args.target_config}...")
        config_marginals = load_distribution_config(args.target_config)

    # Attribute mapping: (attr_key, cli_dist_arg, cli_param_arg, default_dist)
    attr_specs = [
        ("gender", "gender_dist", "gender_dist_param", "uniform"),
        ("age_group", "age_dist", "age_dist_param", "uniform"),
        ("race", "race_dist", "race_dist_param", "uniform"),
    ]

    for attr_key, dist_arg, param_arg, default_dist in attr_specs:
        size = ATTRIBUTE_SIZES[attr_key]
        cli_dist = getattr(args, dist_arg)
        cli_param = getattr(args, param_arg)

        # Priority 1: CLI explicitly specified (not default)
        if was_cli_specified(args, dist_arg, default_dist):
            marginals[attr_key] = parse_distribution(cli_dist, size, cli_param)
            print(f"  {attr_key}: using CLI value '{cli_dist}'")
        # Priority 2: JSON config has this attribute
        elif attr_key in config_marginals and config_marginals[attr_key] is not None:
            marginals[attr_key] = config_marginals[attr_key]
            print(f"  {attr_key}: using config file value")
        # Priority 3: Default (uniform)
        else:
            marginals[attr_key] = parse_distribution(default_dist, size, cli_param)
            print(f"  {attr_key}: using default 'uniform'")

    print("Marginal distributions:")
    for attr, dist in marginals.items():
        print(f"  {attr}: {dist}")
    print()

    # Load full dataset
    print(f"Loading dataset from {args.images_dir}...")
    full_dataset = FFHQAgingDataset(
        images_dir=args.images_dir,
        labels_json=args.labels_json, 
    )
    print(f"Loaded {len(full_dataset)} images from dataset\n")

    # Subset by joint distribution
    print(f"Sampling {args.num_samples} images to match joint distribution...")
    subset_dataset = subset_by_joint_marginals(
        full_dataset,
        marginals=marginals,
        total=args.num_samples,
        seed=args.seed,
        allow_oversample=args.allow_oversample,
    )
    print(f"Subsetted to {len(subset_dataset)} images\n")

    # Calculate inception stats
    device = th.device(args.device)
    mu, sigma = calculate_inception_stats_from_dataset(
        dataset=subset_dataset,
        max_batch_size=args.batch,
        device=device,
    )

    # Save
    os.makedirs(os.path.dirname(args.dest) or ".", exist_ok=True)
    np.savez(
        args.dest,
        mu=mu,
        sigma=sigma,
        num_samples=args.num_samples,
        seed=args.seed,
        marginals_gender=marginals["gender"],
        marginals_age_group=marginals["age_group"],
        marginals_race=marginals["race"],
    )
    print(f"\nSaved reference statistics to {args.dest}")


if __name__ == "__main__":
    main()
