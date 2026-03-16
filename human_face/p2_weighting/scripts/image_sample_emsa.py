"""
Generate image samples using EMSA (Extended Method of Successive Approximations).

This script uses SpacedDiffusionODE with optimal control to guide sampling:
- State variable: z_t = x_t / sqrt(alpha_t)
- Time variable: h_t = sqrt((1-alpha_t)/alpha_t)
- Controlled ODE: dz/dh = epsilon_theta(z/sqrt(1+h^2)) + u

Supports classifier-based guidance and timestep respacing (e.g., --timestep_respacing ddim25).
"""

import argparse
import os
import sys
import gc
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import torch as th
import torch.distributed as dist
import yaml
from torchvision import utils

from guided_diffusion import dist_util, logger
from guided_diffusion.launcher_util.utils import load_yaml_config
from guided_diffusion.sampling_util import StackedRandomGenerator
from guided_diffusion.respace_ode import SpacedDiffusionODE, space_timesteps
from guided_diffusion.gaussian_diffusion import (
    get_named_beta_schedule,
    ModelMeanType,
    ModelVarType,
    LossType,
)
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from guided_diffusion.classifiers import CLASSIFIER_CONFIGS
from guided_diffusion.cost_functions.cost_functions import CostFunctionPCD, CostFunctionKLMultiDimPCD, AnnealedCostFunction
from guided_diffusion.sampling_util import get_target_dist, get_merge_map, parse_target_distribution
# from .image_sample_ode import create_ode_diffusion


def create_ode_diffusion(args):
    """
    Create SpacedDiffusionODE for ODE-based sampling with timestep respacing.
    """
    betas = get_named_beta_schedule(args.noise_schedule, args.diffusion_steps)

    if args.use_kl:
        loss_type = LossType.RESCALED_KL
    elif args.rescale_learned_sigmas:
        loss_type = LossType.RESCALED_MSE
    else:
        loss_type = LossType.MSE

    # Handle timestep respacing
    if not args.timestep_respacing:
        timestep_respacing = [args.diffusion_steps]
    else:
        timestep_respacing = args.timestep_respacing

    return SpacedDiffusionODE(
        use_timesteps=space_timesteps(args.diffusion_steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            ModelMeanType.EPSILON if not args.predict_xstart else ModelMeanType.START_X
        ),
        model_var_type=(
            (ModelVarType.FIXED_LARGE if not args.sigma_small else ModelVarType.FIXED_SMALL)
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
    def loss_fn(pred_x0, **kwargs):
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
    Create a classifier-based loss function for EMSA guidance.

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

    def loss_fn(pred_x0, **kwargs):
        """
        Compute classifier-based loss for EMSA guidance.

        :param pred_x0: predicted clean image [B, C, H, W] in [-1, 1] range
        :return: per-sample loss [B,]
        """
        # CostFunctionPCD.forward returns per-sample loss [B,]
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
    # temperature: float = 1.0, # Handled by annealing wrapper
    cost_scale: float = 1.0,
    temp_schedule: str = 'constant',
    temp_start: float = 1.0,
    temp_end: float = 1.0,
    compensation_strategy: str = 'none',
    total_iters: int = 1,
    target_distributions: Optional[Dict[str, List[float]]] = None,
):
    """
    Create a multi-head classifier-based loss function for EMSA guidance
    that minimizes KL divergence to target distributions for each head.

    :param classifier_name: name of the classifier in CLASSIFIER_CONFIGS
    :param classifier_heads: list of classifier heads, e.g., ["gender", "race", "age_group"]
    :param classifier_type: 'latent' or 'image'
    :param device: torch device
    :param reverse_kl: if True, use reverse KL divergence
    :param cost_scale: scaling factor for the cost
    :param temp_schedule: 'constant', 'linear', or 'cosine'
    :param temp_start: starting temperature
    :param temp_end: ending temperature
    :param compensation_strategy: 'none' or 'linear'
    :param total_iters: total number of optimization iterations for annealing
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
        temperature=temp_start, # Initialize with start temp
    )
    
    # Wrap with AnnealedCostFunction
    cost_fn = AnnealedCostFunction(
        cost_fn=base_cost_fn,
        temp_schedule=temp_schedule,
        temp_start=temp_start,
        temp_end=temp_end,
        cost_scale=cost_scale,
        compensation_strategy=compensation_strategy,
        total_iters=total_iters,
        verbose=True,
    )

    # Set target distributions (custom or uniform fallback)
    q_tar = parse_target_distribution(
        target_dist_config=target_distributions,
        classifier_heads=classifier_heads,
        warn_fn=logger.log,
    )
    logger.log(f"Target distributions for multi-head classifier: {q_tar}")
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
        f"type='{classifier_type}', reverse_kl={reverse_kl}, "
        f"schedule={temp_schedule}, start={temp_start}, end={temp_end}, "
        f"total_iters={total_iters}"
    )

    def loss_fn(x, **kwargs):
        """
        Compute multi-head KL divergence loss for EMSA guidance.

        :param x: predicted clean image [B, C, H, W] in [-1, 1] range
        :return: per-sample loss [B,]
        """
        # kwargs will contain iteration and total_iters passed from solver
        return cost_fn.forward(
            x, 
            timesteps=None,
            reverse_kl=reverse_kl,
            **kwargs 
        )

    return loss_fn

def main():
    # Parse arguments with config file support
    args = parse_args_with_config()

    # Parse target_distribution if it's a JSON string (from batch_exec.py)
    if hasattr(args, 'target_distribution') and isinstance(args.target_distribution, str):
        import json
        try:
            args.target_distribution = json.loads(args.target_distribution)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in target_distribution: {args.target_distribution}")

    args.emsa_lr = 1.0 * (1 - args.decay) / args.rho_u

    # Setup device before distributed initialization
    if args.devices is not None:
        dist_util.setup_device(args.devices, args.distributed)

    dist_util.setup_dist()
    logger.configure(dir=args.sample_dir)

    # Dump args as YAML for reproducibility
    args_path = os.path.join(args.sample_dir, "args.yaml")
    with open(args_path, "w") as f:
        yaml.dump(vars(args), f, default_flow_style=False)

    logger.log("creating model...")
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

    logger.log("creating ODE diffusion...")
    diffusion = create_ode_diffusion(args)
    logger.log(f"ODE diffusion created with {diffusion.num_timesteps} timesteps")

    # Create loss function for EMSA guidance
    if args.classifier_heads:
        # Multi-head mode: use KL divergence to target distributions
        phi_fn = create_multidim_classifier_loss_fn(
            classifier_name=args.classifier_name,
            classifier_heads=args.classifier_heads,
            classifier_type=args.classifier_type,
            device=dist_util.dev(),
            reverse_kl=args.reverse_kl,
            # temperature=args.temperature,
            cost_scale=args.cost_scale,
            temp_schedule=args.temp_schedule,
            temp_start=args.temp_start,
            temp_end=args.temp_end,
            compensation_strategy=args.compensation_strategy,
            total_iters=args.emsa_iters,
            target_distributions=getattr(args, 'target_distribution', None),
        )
    elif args.classifier_name:
        # Single-head mode: promote target class
        phi_fn = create_classifier_loss_fn(
            classifier_name=args.classifier_name,
            classifier_head=args.classifier_head,
            target_class=args.target_class,
            classifier_type=args.classifier_type,
            device=dist_util.dev(),
        )
    else:
        logger.log("No classifier specified, using dummy white-image loss")
        phi_fn = create_white_image_loss_fn(target_value=1.0)

    logger.log(f"Sampling using EMSA (rho_u={args.rho_u}, iters={args.emsa_iters})...")
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

        # EMSA-based sampling with optimal control
        result = diffusion.emsa_solve(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            phi_fn=phi_fn,
            rho_u=args.rho_u,
            iters=args.emsa_iters,
            eta=args.emsa_lr,
            xi=args.decay,
            minibatch=args.minibatch,
            chunk=args.chunk,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            device=dist_util.dev(),
            generator=generator,
            verbose=args.verbose,
            print_fn=logger.log,
            convergence_mode=args.convergence_mode,
            cost_tol_abs=args.cost_tol_abs,
            cost_tol_rel=args.cost_tol_rel,
        )
        sample = result["sample"]

        # Check memory before cleanup
        # mem_before_del = th.cuda.memory_allocated() / 1024**2
        # logger.log(f"Memory before del: {mem_before_del:.1f}MB")

        # saving png in subdirectories (100 images per subdirectory)
        first_img_idx = count * args.batch_size
        first_subdir_start = (first_img_idx // 100) * 100
        first_subdir = os.path.join(args.sample_dir, f"{first_subdir_start:04d}")

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

        # Save optimization results (cost_history, time_bench) for this batch
        os.makedirs(first_subdir, exist_ok=True)
        npz_path = os.path.join(first_subdir, f"emsa_results_{count}.npz")
        np.savez(
            npz_path,
            cost_history=result["cost_history"],
            time_bench=result["time_bench"],
        )

        count += 1
        logger.log(f"created {count * args.batch_size} samples")

        # Explicitly free GPU memory after batch
        del result  # Delete the full result dict
        gc.collect()
        th.cuda.empty_cache()

        # mem_after = th.cuda.memory_allocated() / 1024**2
        # logger.log(f"Memory after cleanup: {mem_after:.1f}MB")
        # logger.log(f"Freed: {mem_before_del - mem_after:.1f}MB")
    #end while

    if dist.is_initialized():
        dist.barrier()
    logger.log("sampling complete")
    dist_util.cleanup_dist()


def get_defaults():
    """Get default argument values."""
    defaults = dict(
        # clip_denoised=True,
        clip_denoised=False,
        num_samples=32,
        batch_size=16,
        model_path="",
        sample_dir="",
        seed=0,
        # EMSA-specific arguments
        rho_u=1.0,           # Weight for quadratic control cost
        emsa_iters=20,       # Number of EMSA iterations
        # emsa_lr=0.01,       # Step size for control update
        decay=0.9,       # Damping factor
        minibatch=0,         # Minibatch size for chunked execution
        chunk=False,         # Use chunked execution
        verbose=True,        # Print progress
        # Single-head classifier-based guidance arguments
        classifier_name="",  # e.g., "pcd_ldm", "pcd_pretrained"
        classifier_head="",  # e.g., "age_group", "gender"
        target_class=0,      # Target class index
        classifier_type="image",  # "latent" or "image"
        # Multi-head classifier arguments
        reverse_kl=False,    # Use reverse KL divergence
        temperature=1.0,     # Softmax temperature (deprecated/used as start if schedule is constant)

        # Annealing & Scaling
        cost_scale=1.0,
        temp_schedule='linear', # 'constant', 'linear', 'cosine'
        temp_start=1.0,
        temp_end=1.0,
        compensation_strategy='linear', # 'none', 'linear'

        # EMSA Convergence
        convergence_mode='both',  # 'abs', 'rel', or 'both'
        cost_tol_abs=1e-5,
        cost_tol_rel=1e-4,

        # Target distribution
        target_distribution=None,  # Dict[str, List[float]] for custom distributions 
    )
    defaults.update(model_and_diffusion_defaults())
    return defaults


def parse_args_with_config():
    """
    Parse arguments with config file support.

    Implements two-pass parsing:
    1. First pass: Parse --config, --devices, --distributed
    2. Load config file if specified
    3. Second pass: Parse all args with config as defaults, CLI taking precedence

    Returns:
        argparse.Namespace with merged arguments
    """
    # First pass: parse only config/device args
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None,
                            help="Path to YAML config file")
    pre_parser.add_argument("--devices", type=str, default=None,
                            help="Device specification: '0', '0,1,2', or 'cpu'")
    pre_parser.add_argument("--distributed", action="store_true",
                            help="Enable distributed multi-GPU mode")

    pre_args, _ = pre_parser.parse_known_args()

    # Get defaults
    defaults = get_defaults()

    # Load config file if specified and merge with defaults
    if pre_args.config:
        config_path = Path(pre_args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        config = load_yaml_config(config_path)
        # Config values override defaults
        defaults.update(config)

    # Second pass: full parsing with updated defaults
    parser = argparse.ArgumentParser(
        description="Generate image samples using EMSA (Extended Method of Successive Approximations)."
    )

    # Add config/device arguments (they'll be in the help)
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")
    parser.add_argument("--devices", type=str, default=None,
                        help="Device specification: '0', '0,1,2', or 'cpu'")
    parser.add_argument("--distributed", action="store_true",
                        help="Enable distributed multi-GPU mode")

    # Handle classifier_heads specially since it's a list with nargs='*'
    classifier_heads_default = defaults.pop("classifier_heads", [])

    # Add all other arguments from defaults
    add_dict_to_argparser(parser, defaults)

    # Multi-head argument with nargs='*' for multiple inputs
    parser.add_argument(
        "--classifier_heads", nargs='*', default=classifier_heads_default,
        help="Multiple classifier heads, e.g., --classifier_heads gender race age_group"
    )

    # Parse all arguments
    args = parser.parse_args()

    # Ensure config/device args from first pass are preserved
    args.config = pre_args.config
    args.devices = pre_args.devices
    args.distributed = pre_args.distributed

    return args


def create_argparser():
    """Create argument parser (backward compatible, without config file support)."""
    defaults = get_defaults()
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    # Multi-head argument with nargs='*' for multiple inputs
    parser.add_argument(
        "--classifier_heads", nargs='*', default=[],
        help="Multiple classifier heads, e.g., --classifier_heads gender race age_group"
    )
    return parser


if __name__ == "__main__":
    main()


# Example: EMSA sampling with white-image loss (dummy)
## python scripts/image_sample_emsa.py --attention_resolutions 16 --class_cond False --diffusion_steps 1000 --dropout 0.0 --learn_sigma True --noise_schedule linear --num_channels 128 --num_res_blocks 1 --num_head_channels 64 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --image_size 256 --timestep_respacing ddim25 --model_path models/ffhq_p2.pt --sample_dir outputs/emsa/ --num_samples 32 --batch_size 16 --rho_u 1.0 --emsa_iters 20

# Example: EMSA sampling with classifier-based guidance
## python scripts/image_sample_emsa.py --attention_resolutions 16 --class_cond False --diffusion_steps 1000 --dropout 0.0 --learn_sigma True --noise_schedule linear --num_channels 128 --num_res_blocks 1 --num_head_channels 64 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --image_size 256 --timestep_respacing ddim25 --model_path models/ffhq_p2.pt --sample_dir outputs/emsa_classifier/ --num_samples 32 --batch_size 16 --rho_u 1.0 --emsa_iters 20 --classifier_name pcd_pretrained --classifier_head gender --target_class 0 --classifier_type image

# Example: EMSA sampling with classifier-based guidance
## python scripts/image_sample_emsa.py --attention_resolutions 16 --class_cond False --diffusion_steps 1000 --dropout 0.0 --learn_sigma True --noise_schedule linear --num_channels 128 --num_res_blocks 1 --num_head_channels 64 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --image_size 256 --timestep_respacing ddim25 --model_path models/ffhq_p2.pt --sample_dir outputs/emsa_classifier/ --num_samples 32 --batch_size 16 --rho_u 1.0 --decay 0.8 --emsa_iters 10 --classifier_name pcd_pretrained --classifier_heads gender age_group  --classifier_type image 


# Example: EMSA sampling with classifier-based guidance
## python scripts/image_sample_emsa.py --attention_resolutions 16 --class_cond False --diffusion_steps 1000 --dropout 0.0 --learn_sigma True --noise_schedule linear --num_channels 128 --num_res_blocks 1 --num_head_channels 64 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --image_size 256 --timestep_respacing ddim25 --model_path models/ffhq_p2.pt --sample_dir outputs/emsa_classifier/ --num_samples 32 --batch_size 16 --rho_u 1.0 --decay 0.8 --emsa_iters 10 --classifier_name pcd_pretrained --classifier_heads gender age_group  --classifier_type image  --temp_schedule cosine --temp_start 10.0 --temp_end 1.0 
