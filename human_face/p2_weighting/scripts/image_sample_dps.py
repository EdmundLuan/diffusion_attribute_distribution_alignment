import argparse
import os
from pathlib import Path

# Third-party imports
import torch as th
import torch.distributed as dist
import yaml
import gc
from typing import List, Dict, Optional
from torchvision import utils

# Local imports - core utilities
from guided_diffusion import dist_util, logger
from guided_diffusion.launcher_util.utils import load_yaml_config

# Local imports - diffusion components
from guided_diffusion.gaussian_diffusion import get_named_beta_schedule, ModelMeanType, ModelVarType, LossType
from guided_diffusion.respace_dps import SpacedDiffusionDPS, space_timesteps
from guided_diffusion.sampling_util import StackedRandomGenerator, get_target_dist, get_merge_map, parse_target_distribution, load_target_config

# Local imports - script utilities
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

# Local imports - classifiers and cost functions
from guided_diffusion.classifiers import CLASSIFIER_CONFIGS
from guided_diffusion.cost_functions.cost_functions import CostFunctionPCD, CostFunctionKLMultiDimPCD


def create_dps_diffusion(args):
    """
    Create a SpacedDiffusionDPS instance with the given arguments.
    """
    betas = get_named_beta_schedule(args.noise_schedule, args.diffusion_steps)

    if args.use_kl:
        loss_type = LossType.RESCALED_KL
    elif args.rescale_learned_sigmas:
        loss_type = LossType.RESCALED_MSE
    else:
        loss_type = LossType.MSE

    if not args.timestep_respacing:
        timestep_respacing = [args.diffusion_steps]
    else:
        timestep_respacing = args.timestep_respacing

    return SpacedDiffusionDPS(
        use_timesteps=space_timesteps(args.diffusion_steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            ModelMeanType.EPSILON if not args.predict_xstart else ModelMeanType.START_X
        ),
        model_var_type=(
            (
                ModelVarType.FIXED_LARGE
                if not args.sigma_small
                else ModelVarType.FIXED_SMALL
            )
            if not args.learn_sigma
            else ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=args.rescale_timesteps,
    )


def create_white_image_loss_fn(target_value=1.0):
    """
    Create a loss function that guides towards all-white images.

    In the [-1, 1] range used by the diffusion model:
    - -1 corresponds to black
    - +1 corresponds to white

    :param target_value: target pixel value (default 1.0 for white)
    :return: loss function that takes pred_x0 and returns per-sample L2 loss
    """
    def loss_fn(pred_x0):
        # Compute MSE loss to target value for each sample
        # pred_x0 shape: [B, C, H, W]
        # Return shape: [B,] - one loss per sample
        target = th.full_like(pred_x0, target_value)
        return ((pred_x0 - target) ** 2).mean(dim=(1, 2, 3))

    return loss_fn


def create_classifier_loss_fn(
    classifier_name: str,
    classifier_head: str,
    target_class: int,
    classifier_type: str,
    device,
):
    """
    Create a classifier-based loss function for DPS guidance.

    :param classifier_name: name of the classifier in CLASSIFIER_CONFIGS
    :param classifier_head: which classifier head to use (e.g., "age_group", "gender")
    :param target_class: target class index to guide towards
    :param classifier_type: 'latent' or 'image' - whether classifier operates on
                           latent space or image space
    :param device: torch device
    :return: loss function that takes pred_x0 and returns per-sample loss [B,]
    """
    # Load classifier
    if classifier_name not in CLASSIFIER_CONFIGS:
        raise ValueError(
            f"Unknown classifier '{classifier_name}'. "
            f"Available: {list(CLASSIFIER_CONFIGS.keys())}"
        )

    config = CLASSIFIER_CONFIGS[classifier_name]
    classifier_path = config["path"]
    get_func = config["get_func"]

    logger.log(f"Loading classifier '{classifier_name}'...")
    classifier = get_func(
        config_pth=classifier_path["config"],
        weights_pth=classifier_path["weights"],
        device=device,
    )
    classifier.eval()

    # Create cost function
    cost_fn = CostFunctionPCD(
        classifier=classifier,
        device=device,
        classifier_type=classifier_type,
    )
    cost_fn.set_target_cls(target_class)

    logger.log(
        f"Classifier loaded: head='{classifier_head}', "
        f"target_class={target_class}, type='{classifier_type}'"
    )

    def loss_fn(pred_x0):
        """
        Compute classifier-based loss for DPS guidance.

        :param pred_x0: predicted clean image [B, C, H, W] in [-1, 1] range
        :return: per-sample loss [B,]
        """
        # CostFunctionPCD.forward returns per-sample loss [B,]
        # Note: timesteps=None since we don't have access to t in this interface
        # The classifier may or may not use timestep conditioning
        loss = cost_fn.forward(
            x=pred_x0,
            head_key=classifier_head,
            target=target_class,
            timesteps=None,
        )
        return loss

    return loss_fn

def create_multidim_classifier_loss_fn(
    classifier_name: str,
    classifier_heads: List[str],
    classifier_type: str,
    device,
    reverse_kl: bool = False,
    temperature: float = 1.0,
    minibatch: int = 4,
    target_distributions: Optional[Dict[str, List[float]]] = None,
):
    """
    Create a multi-head classifier-based loss function for DPS guidance
    that minimizes KL divergence to target distributions for each head.

    Unlike EMSA, this uses a fixed temperature (no annealing) throughout sampling.

    :param classifier_name: name of the classifier in CLASSIFIER_CONFIGS
    :param classifier_heads: list of classifier heads, e.g., ["gender", "race", "age_group"]
    :param classifier_type: 'latent' or 'image'
    :param device: torch device
    :param reverse_kl: if True, use reverse KL divergence
    :param temperature: softmax temperature (fixed, no annealing in DPS)
    :param minibatch: minibatch size for gradient accumulation in classifier
    :return: loss function that takes pred_x0 and returns per-sample loss [B,]
    """
    if classifier_name not in CLASSIFIER_CONFIGS:
        raise ValueError(
            f"Unknown classifier '{classifier_name}'. "
            f"Available: {list(CLASSIFIER_CONFIGS.keys())}"
        )

    config = CLASSIFIER_CONFIGS[classifier_name]
    classifier_path = config["path"]
    get_func = config["get_func"]

    logger.log(f"Loading classifier '{classifier_name}' for multi-head guidance...")
    classifier = get_func(
        config_pth=classifier_path["config"],
        weights_pth=classifier_path["weights"],
        device=device,
    )
    classifier.eval()

    base_cost_fn = CostFunctionKLMultiDimPCD(
        classifier=classifier,
        device=device,
        classifier_type=classifier_type,
        temperature=temperature,
    )

    # Set target distributions (custom or uniform fallback)
    q_tar = parse_target_distribution(
        target_dist_config=target_distributions,
        classifier_heads=classifier_heads,
        warn_fn=logger.log,
    )
    logger.log(f"Target distributions: {q_tar}")
    base_cost_fn.set_target(q_tar)

    # Set probability merge mappings (if applicable)
    merge_mappings = {}
    for head in classifier_heads:
        try:
            merge_map = get_merge_map(head, classifier_name)
            merge_mappings[head] = merge_map
        except ValueError:
            pass  # No merge map for this head
    if merge_mappings:
        base_cost_fn.set_prob_merge_mapping(merge_mappings)

    logger.log(
        f"Multi-head classifier loaded: heads={classifier_heads}, "
        f"type='{classifier_type}', reverse_kl={reverse_kl}, temperature={temperature}, minibatch={minibatch}"
    )

    def loss_fn(pred_x0):
        """
        Compute multi-head KL divergence loss for DPS guidance.

        :param pred_x0: predicted clean image [B, C, H, W] in [-1, 1] range
        :return: per-sample loss [B,]
        """
        # No iteration tracking needed for DPS (unlike EMSA)
        return base_cost_fn.forward(
            pred_x0,
            timesteps=None,
            reverse_kl=reverse_kl,
            minibatch=minibatch,
        )

    return loss_fn


def main():
    # Parse arguments with YAML config support
    args = parse_args_with_config()

    # Parse target distribution with priority: inline JSON > config file > None
    target_distribution_resolved = None

    # 1. Load from config file if specified
    if hasattr(args, 'target_config') and args.target_config:
        logger.log(f"Loading target distribution from config file: {args.target_config}")
        target_distribution_resolved = load_target_config(args.target_config)

    # 2. Inline JSON overrides/merges with config file
    if hasattr(args, 'target_distribution') and args.target_distribution:
        if isinstance(args.target_distribution, str):
            import json
            try:
                inline_dist = json.loads(args.target_distribution)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON in target_distribution: {args.target_distribution}")
        else:
            inline_dist = args.target_distribution

        if target_distribution_resolved is None:
            target_distribution_resolved = inline_dist
        else:
            # Merge: inline overrides config file on per-attribute basis
            logger.log(f"Merging inline JSON over config file (inline takes priority)")
            target_distribution_resolved.update(inline_dist)

    args.target_distribution = target_distribution_resolved
    # Setup device (before distributed init if using custom device selection)
    if args.devices is not None:
        dist_util.setup_device(args.devices, args.distributed)

    # Setup distributed training
    dist_util.setup_dist()
    logger.configure(dir=args.sample_dir)

    # Dump args as YAML for reproducibility
    args_path = os.path.join(args.sample_dir, "args.yaml")
    with open(args_path, "w") as f:
        yaml.dump(vars(args), f, default_flow_style=False)

    logger.log("creating model and diffusion...")

    # Create model using standard utility
    model, _ = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    # Create DPS diffusion
    diffusion = create_dps_diffusion(args)

    # Create loss function for DPS guidance
    if args.classifier_heads:
        # Multi-head mode: KL divergence to target distributions
        loss_fn = create_multidim_classifier_loss_fn(
            classifier_name=args.classifier_name,
            classifier_heads=args.classifier_heads,
            classifier_type=args.classifier_type,
            device=dist_util.dev(),
            reverse_kl=args.reverse_kl,
            temperature=args.temperature,
            minibatch=min(args.minibatch, args.batch_size), # filtering
            target_distributions=getattr(args, 'target_distribution', None),
        )
        if args.target_distribution is not None:
            logger.log(f"Using custom target distributions: {args.target_distribution}")
    elif args.classifier_name:
        # Single-head mode: promote target class (backward compatible)
        loss_fn = create_classifier_loss_fn(
            classifier_name=args.classifier_name,
            classifier_head=args.classifier_head,
            target_class=args.target_class,
            classifier_type=args.classifier_type,
            device=dist_util.dev(),
        )
    else:
        # Fallback to dummy white-image loss
        logger.log("No classifier specified, using dummy white-image loss")
        loss_fn = create_white_image_loss_fn(target_value=1.0)

    logger.log(f"sampling with DPS (scale={args.dps_scale})...")
    count = 0
    while count * args.batch_size < args.num_samples:
        # Create per-sample seeds for reproducibility
        batch_seeds = [args.seed + count * args.batch_size + i for i in range(args.batch_size)]
        generator = StackedRandomGenerator(device=dist_util.dev(), seeds=batch_seeds)

        model_kwargs = {}
        if args.class_cond:
            classes = generator.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes

        # Use DPS sampling
        sample = diffusion.ddim_sample_loop_dps(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            loss_fn=loss_fn,
            scale=args.dps_scale,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            device=dist_util.dev(),
            progress=args.progress,
            eta=args.eta,
            generator=generator,
        )

        # Save PNG images in subdirectories (100 images per subdirectory)
        for i in range(args.batch_size):
            img_idx = count * args.batch_size + i
            subdir_start = (img_idx // 100) * 100
            subdir = os.path.join(args.sample_dir, f"{subdir_start:04d}")
            os.makedirs(subdir, exist_ok=True)
            out_path = os.path.join(subdir, f"{img_idx:05d}.png")
            utils.save_image(
                sample[i].unsqueeze(0),
                out_path,
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
            )

        count += 1
        logger.log(f"created {count * args.batch_size} samples")

        # Explicitly free GPU memory after batch
        del sample  # Delete the full sample tensor
        gc.collect()
        th.cuda.empty_cache()

    if dist.is_initialized():
        dist.barrier()
    logger.log("sampling complete")
    dist_util.cleanup_dist()


def get_defaults():
    """
    Get default arguments for DPS sampling.

    Returns a dictionary containing:
    - Sampling parameters (clip_denoised, num_samples, batch_size, etc.)
    - DPS-specific parameters (dps_scale, eta)
    - Single-head classifier parameters (classifier_name, classifier_head, target_class)
    - Multi-head classifier parameters (reverse_kl, temperature)
    - Model/diffusion parameters from model_and_diffusion_defaults()
    """
    defaults = dict(
        # Sampling parameters
        clip_denoised=True,
        num_samples=32,
        batch_size=16,
        use_ddim=True,
        model_path="",
        sample_dir="",
        seed=0,
        progress=False,
        # DPS-specific parameters
        dps_scale=1.0,
        eta=0.0,  # DDIM eta parameter (0 = deterministic)
        # Single-head classifier guidance
        classifier_name="",  # e.g., "pcd_ldm", "pcd_pretrained"
        classifier_head="",  # e.g., "age_group", "gender"
        target_class=0,  # target class index
        classifier_type="image",  # "latent" or "image"
        # Multi-head classifier guidance (NO annealing parameters)
        reverse_kl=False,  # Use reverse KL divergence
        temperature=1.0,  # Softmax temperature (no annealing in DPS)
        minibatch=4,  # Minibatch size for gradient accumulation in classifier
        target_distribution=None,  # Dict[str, List[float]] for custom distributions
        target_config=None,  # Path to JSON config file for target distributions
        # Config file and distributed inference
        config=None,  # Path to YAML config file
        devices=None,  # GPU devices (e.g., "0,1,2,3")
        distributed=False,  # Enable distributed inference
    )
    # Add model and diffusion defaults
    defaults.update(model_and_diffusion_defaults())
    return defaults


def parse_args_with_config():
    """
    Parse command-line arguments with YAML config file support.

    Uses two-pass parsing:
    1. First pass: parse --config, --devices, --distributed
    2. Load YAML config if specified, merge with defaults
    3. Second pass: parse all arguments (CLI overrides config/defaults)

    Returns:
        Namespace: parsed arguments with config file merged
    """
    # First pass: parse config file path and distributed settings
    parser_first = argparse.ArgumentParser(add_help=False)
    parser_first.add_argument("--config", type=str, default=None)
    parser_first.add_argument("--devices", type=str, default=None)
    parser_first.add_argument("--distributed", action="store_true")
    first_args, _ = parser_first.parse_known_args()

    # Get default values
    defaults = get_defaults()

    # Load config file if specified
    if first_args.config is not None:
        config_path = Path(first_args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {first_args.config}")

        logger.log(f"Loading config from {first_args.config}")
        config_dict = load_yaml_config(first_args.config)

        # Merge config into defaults (config overrides defaults)
        defaults.update(config_dict)

    # Second pass: parse all arguments (CLI overrides config)
    parser = argparse.ArgumentParser()

    # Handle classifier_heads separately since it needs special nargs='*' handling
    classifier_heads_default = defaults.pop("classifier_heads", [])

    add_dict_to_argparser(parser, defaults)

    # Add classifier_heads as a special argument (list of strings)
    parser.add_argument(
        "--classifier_heads",
        nargs='*',
        default=classifier_heads_default,
        help="List of classifier heads for multi-dimensional guidance (e.g., gender race age_group)"
    )

    args = parser.parse_args()

    return args


# def create_argparser():
#     """
#     Create argument parser for backward compatibility.

#     This is maintained for scripts that import and use create_argparser() directly.
#     For new usage, prefer parse_args_with_config() which supports YAML configs.
#     """
#     defaults = get_defaults()
#     parser = argparse.ArgumentParser()
#     add_dict_to_argparser(parser, defaults)
    
#     return parser


if __name__ == "__main__":
    main()

# python scripts/image_sample_dps.py --attention_resolutions 16 --class_cond False --diffusion_steps 1000 --dropout 0.0 --learn_sigma True --noise_schedule linear --num_channels 128 --num_res_blocks 1 --num_head_channels 64 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --timestep_respacing ddim25 --use_ddim True  --image_size 256 --model_path models/ffhq_p2.pt --sample_dir outputs/samples_dps/  --dps_scale 2.0 --classifier_name  "pcd_pretrained" --classifier_head gender --target_class 0 --classifier_type image --num_samples 32 --batch_size 16 


"""
Generate image samples using Diffusion Posterior Sampling (DPS).

DPS guides the diffusion process towards samples consistent with measurements.
Supports both single-head and multi-head classifier guidance.

Example usage:

# 1. Basic usage with YAML config:
python scripts/image_sample_dps.py --config configs/image_dps.yaml

# 2. Single-head classifier guidance (backward compatible):
python scripts/image_sample_dps.py \
    --model_path models/ffhq_p2.pt \
    --classifier_name pcd_pretrained \
    --classifier_head gender \
    --target_class 0 \
    --num_samples 32 --batch_size 16 \
    --sample_dir outputs/dps_single/

# 3. Multi-head classifier guidance (KL divergence to uniform distributions):
python scripts/image_sample_dps.py \
    --model_path models/ffhq_p2.pt \
    --classifier_name pcd_all_heads \
    --classifier_heads gender race age_group \
    --temperature 1.0 \
    --reverse_kl False \
    --num_samples 32 --batch_size 16 \
    --sample_dir outputs/dps_multi/

# 4. CLI override of config file:
python scripts/image_sample_dps.py \
    --config configs/image_dps.yaml \
    --num_samples 64 \
    --dps_scale 2.0

# 5. Distributed inference (multi-GPU):
python scripts/image_sample_dps.py \
    --config configs/image_dps.yaml \
    --devices 0,1,2,3 \
    --distributed \
    --num_samples 128
"""