"""
Generate image samples using ODE-based sampling (Euler method).

This script uses SpacedDiffusionODE which interprets DDIM as a neural ODE:
- State variable: z_t = x_t / sqrt(alpha_t)
- Time variable: h_t = sqrt((1-alpha_t)/alpha_t)
- ODE: dz/dh = epsilon_theta(z/sqrt(1+h^2))

Supports timestep respacing (e.g., --timestep_respacing ddim25).
"""

import argparse
import os

import torch.distributed as dist
import yaml

from guided_diffusion import dist_util, logger
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
from torchvision import utils


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


def main():
    args = create_argparser().parse_args()

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

    logger.log("sampling using ODE (Euler method)...")
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

        # Generate initial noise
        noise = generator.randn(
            (args.batch_size, 3, args.image_size, args.image_size),
            device=dist_util.dev(),
        )

        # ODE-based sampling using Euler method
        sample = diffusion.euler_solve_z(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            noise=noise,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            device=dist_util.dev(),
            progress=True,
        )

        # saving png in subdirectories (100 images per subdirectory)
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

    if dist.is_initialized():
        dist.barrier()
    logger.log("sampling complete")
    dist_util.cleanup_dist()


def create_argparser():
    defaults = dict(
        # clip_denoised=False,
        clip_denoised=True, ## Default to True
        num_samples=32,
        batch_size=32,
        model_path="",
        sample_dir="",
        seed=0,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()


## python scripts/image_sample_ode.py --attention_resolutions 16 --class_cond False --diffusion_steps 1000 --dropout 0.0 --learn_sigma True --noise_schedule linear --num_channels 128 --num_res_blocks 1 --num_head_channels 64 --resblock_updown True --use_fp16 True --use_scale_shift_norm True  --image_size 256 --timestep_respacing ddim25 --model_path models/ffhq_p2.pt --sample_dir outputs/zode/T25  --num_samples 32 --batch_size 16 
