"""
Evaluate generated samples: estimate class distribution using the classifier
specified in each run's config.yaml, and save JSON metrics into that run dir.

New: pass --runs-file path/to/list.txt to read run dirs from a file (one per line).
Lines starting with '#' or blank lines are ignored. Both --runs and --runs-file
may be used together; duplicates are removed.

Usage examples:
  python eval_runs.py \
    --runs eval/sampling/emsa/emsa_cls-meta_mb-*_ts-* eval/sampling/emsa/emsa_cls-coarse_* \
    --device cuda:0 --batch-size 256

  python eval_runs.py \
    --runs-file run_dirs.txt \
    --device cuda --batch-size 512 --max-samples 8192
    --fid-ref eval/fid/fid_cifar100.npz
"""

import argparse
import json
import math
import os
import re
import sys
import glob
import time
import yaml 
import datetime as dt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

from guided_diffusion.dnnlib.util import open_url
from guided_diffusion.eval_util.distributions import dist_registry
from guided_diffusion.sampling_util import (
    get_target_dist, make_deterministic_classifier,
    get_support_size, get_merge_map,
    dict_apply, dict_apply_reduce, dict_apply_split,
    merge_probs,
    compute_joint_probs, compute_joint_target_dist,
    parse_target_distribution,
)
from guided_diffusion.eval_util.metrics import (
    kl_div, tv_dist, l2_dist, js_dist, chi2_dist, 
    fairness_discrepancy,
)
from guided_diffusion.eval_util.fid import (
    calculate_inception_stats, calculate_fid_from_inception_stats
)
from guided_diffusion.classifiers import CLASSIFIER_CONFIGS
from guided_diffusion.classifiers.fairface_classifiers import (
    # get_fairface_classifier_4, get_fairface_classifier_7, 
    get_fairface_transforms, 
    infer_fairface, 
)
from guided_diffusion.classifiers.latent_classifier_resnet_enc_multihead import (
    get_pcd_classifier, 
    infer_pcd_classifier, AGE_10_TO_3_MAP,
)


# ---------------------------------------------------------------------
# Basic stats & metrics
# DATASET_MEAN = np.array([0.5070, 0.4865, 0.4409], dtype=np.float32)
# DATASET_STD  = np.array([0.2673, 0.2564, 0.2761], dtype=np.float32)

FID_CACHE_DIR = "outputs/fid"


def load_inception_model(device):
    """Load Inception-v3 model once for reuse across FID calculations."""
    import pickle
    logger.info('Loading Inception-v3 model for FID evaluation...')
    detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
    with open_url(detector_url, verbose=True) as f:
        detector_net = pickle.load(f).to(device)
    return detector_net



def calc_fid(image_path, ref_path, num_expected, seed, batch, cache=True, detector_net=None):
    """Calculate FID for a given set of images.

    Args:
        image_path (str): Path to generated images.
        ref_path (str): Path to reference statistics (.npz file).
        num_expected (int): Number of samples.
        seed (int): Random seed for statistics calculation.
        batch (int): Maximum batch size for statistics calculation.
        cache (bool): Whether to use caching for inception activations.
        detector_net: Optional pre-loaded Inception model for reuse.

    Returns:
        fid (float): Calculated FID score.
    """
    logger.info(f'Loading dataset reference statistics from "{ref_path}"...')
    with open_url(ref_path) as f:
        ref = np.load(f)
        mu_ref = ref['mu']
        sigma_ref = ref['sigma']

    cache_hit = False
    if cache:
        cache_dir = os.path.join(image_path, "fid_cache")
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
        cache_filenm = f"fid_inception_stats_n{num_expected}_s{seed}_b{batch}.npz"
        cache_path = os.path.join(cache_dir, cache_filenm)
        if os.path.exists(cache_path):
            logger.info(f'Using cached inception activations from "{cache_dir}"...')
            with open_url(cache_path) as f:
                cached = dict(np.load(f))
                mu = cached['mu']
                sigma = cached['sigma']
            cache_hit = True

    if not cache_hit:
        logger.info('Calculating inception statistics for generated images...')
        mu, sigma = calculate_inception_stats(image_path=image_path, num_expected=num_expected, seed=seed, max_batch_size=batch, num_workers=16, detector_net=detector_net)

    if cache and not cache_hit:
        logger.info(f'Saving cached inception statistics to "{cache_path}"...')
        np.savez(cache_path, mu=mu, sigma=sigma)

    fid = calculate_fid_from_inception_stats(mu, sigma, mu_ref, sigma_ref)
    logger.info(f'FID: {fid:g}')

    return fid

# ---------------------------------------------------------------------
# Config reader (robust to missing PyYAML; we only need 'classifier')
# def read_classifier_from_config(cfg_path: Path) -> str:
def retrieve_from_config(cfg_path: Path, key: str) -> str:
    if not cfg_path.exists():
        raise FileNotFoundError(f"No configs.yaml at {cfg_path}")
    txt = cfg_path.read_text(encoding="utf-8")
    # Try PyYAML first
    try:
        data = yaml.safe_load(txt)
        val = data.get(key, None)
        if val is None:
            raise ValueError(f"{key} field not found or invalid in configs.yaml")
        return val
    except Exception:
        pass
    # Fallback: parse a simple "key: value" line
    for line in txt.splitlines():
        m = re.match(rf"^\s*{key}\s*:\s*(.+?)\s*$", line)
        if m:
            return m.group(1).strip().strip('"').strip("'")
    raise ValueError(f"{key} field not found in configs.yaml")


def get_fid_ref_stats_path(target_dist:str, classifier:str, num_expected:int, seed:int, batch_size:int) -> Optional[str]:
    """Get FID reference stats path based on target distribution and classifier.
    """
    # Example: "eval/fid/gaussian/fid_inception_stats_cifar100fine_gaussian_n2048_s0_b512.npz"
    if target_dist == "uniform":
        file_nm = f"fid_inception_stats_cifar100_n{num_expected}_s{seed}_b{batch_size}.npz"
    else:
        file_nm = f"fid_inception_stats_cifar100{classifier}_{target_dist}_n{num_expected}_s{seed}_b{batch_size}.npz"
    fid_ref_path = os.path.join(FID_CACHE_DIR, target_dist, file_nm)
    if not os.path.exists(fid_ref_path):
        logger.warning(f"FID reference stats file not found: {fid_ref_path}")
        return None
    return fid_ref_path

# ---------------------------------------------------------------------
# Image loading (recursive) -> list of paths
def collect_pngs(root: Path, max_samples: Optional[int]) -> List[Path]:
    paths = sorted([Path(p) for p in root.rglob("*.png")])
    if max_samples is not None and max_samples > 0:
        paths = paths[:max_samples]
    return paths

# Load a slice of image paths into a tensor [B, C, H, W], uint8
def load_batch(paths: List[Path]) -> torch.Tensor:
    imgs = []
    for p in paths:
        with Image.open(p) as im:
            im = im.convert("RGB")
            arr = np.asarray(im, dtype=np.uint8)  # H W C, [0..255]
        chw = np.transpose(arr, (2, 0, 1))       # C H W
        imgs.append(chw)
    batch = np.stack(imgs, axis=0)               # B C H W
    return torch.from_numpy(batch)               # uint8


# ---------------------------------------------------------------------
@torch.no_grad()

def _logits_to_probs(
    logits_dict: Dict[str, torch.Tensor], 
    probs_all: Dict[str, torch.Tensor],
    prob_merge_maps: Optional[Dict[str, torch.Tensor]]={},
) -> Dict[str, torch.Tensor]:
    probs = dict_apply(logits_dict, lambda x_: torch.softmax(x_, dim=1).detach().cpu())  # Dict of probs per attribute
    for attr_key in prob_merge_maps:
        if attr_key in probs:
            probs[attr_key] = merge_probs(probs[attr_key], prob_merge_maps[attr_key])
    probs_all = dict_apply_reduce([probs_all, probs], lambda tup_: torch.cat(tup_, dim=0)) # Concatenate per attribute

    return probs_all


def infer_probs(model: torch.nn.Module,
                classifier_name: str,
                support_sizes: Dict[str, int],
                images_uint8: torch.Tensor,
                batch_size: int,
                device: torch.device,
                prob_merge_maps: Optional[Dict[str, torch.Tensor]] = {},
                num_race_cls: int = 4,
) -> np.ndarray:
    """
    Args:
        model: Classifier model 
        support_sizes: Dict of attribute key to support size
        images_uint8: torch.uint8 tensor [N, C, H, W] in [0,255]
        batch_size: Batch size for inference
        device: Device for inference

    Returns: 
        Dict[str, torch.Tensor] with shape [N, support_size] per attribute
    """
    n = images_uint8.shape[0]
    probs_all = {k: torch.empty(0, s) for k, s in support_sizes.items()}
    # mean = torch.tensor(DATASET_MEAN, device=device).view(1, 3, 1, 1)
    # std  = torch.tensor(DATASET_STD,  device=device).view(1, 3, 1, 1)

    if classifier_name.startswith("fairface"):
        trans = get_fairface_transforms(differentiable=True)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = images_uint8[start:end].to(device=device, dtype=torch.float32) / 255.0
            batch = trans(batch)
            outputs = infer_fairface(model, batch, num_race_cls) # Dict of logits per attribute
            probs_all = _logits_to_probs(outputs, probs_all, prob_merge_maps)
    elif classifier_name.startswith("pcd"):
        heads = list(support_sizes.keys())
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = images_uint8[start:end].to(device=device, dtype=torch.float32) / 255.0
            outputs = infer_pcd_classifier(model, batch, heads=heads, timesteps=None) # Dict of logits per attribute
            probs_all = _logits_to_probs(outputs, probs_all, prob_merge_maps)

    return probs_all





def get_classifier(model_name: str, device):
    model_path = CLASSIFIER_CONFIGS[model_name]["path"]
    get_func = CLASSIFIER_CONFIGS[model_name]["get_func"]
    if model_name.startswith("fairface"):
        classifier = get_func(model_path, device)
    elif model_name.startswith("pcd"):
        classifier = get_func(model_path["config"], model_path["weights"], device)
    classifier = classifier.to(torch.float32)
    # classifier = make_deterministic_classifier(classifier)

    return classifier


def get_classifier_probs_merge_maps(
    classifier_name: str,
    attribute_keys: Iterable[str],
) -> Dict[str, torch.Tensor]:
    merge_maps = {}
    if classifier_name.startswith("pcd"):
        if "age_group" in attribute_keys:
            # merge_maps["age_group"] = torch.tensor([v for k, v in AGE_10_TO_3_MAP.items()]).long()
            merge_maps["age_group"] = get_merge_map('age_group', classifier_name)
    elif classifier_name.startswith("fairface"):
        merge_maps['age_group'] = get_merge_map('age_group', classifier_name)
        if classifier_name == "fairface_7": 
            merge_maps['race'] = get_merge_map('race_7', classifier_name)
    else:
        raise ValueError(f"Unknown classifier name: {classifier_name}")
    return merge_maps

def evaluate_run(
    run_dir: Path,
    classifier_name: str,
    device_str: str,
    batch_size: int,
    save_name: str,
    max_samples: Optional[int],
    target_dist_name: Optional[str],
    fid_ref_path: str = None,
    fid_seed: int = 0,
    attribute_keys: Optional[Iterable[str]] = None,
    save_intermediate: bool = True,
    detector_net = None,
    target_distribution_cli: Optional[Dict[str, List[float]]] = None,
) -> None:
    # cfg_path = run_dir / "configs.yaml"
    cfg_path = run_dir / "args.yaml"
    if not cfg_path.exists():
        logger.warning(f"[SKIP] \'{run_dir}\': args.yaml not found.")
        return

    if attribute_keys is None:
        attribute_keys = retrieve_from_config(cfg_path, 'classifier_heads')
    support_sizes = {k: get_support_size(k) for k in attribute_keys}

    # Determine target distribution with precedence: CLI > config > uniform
    target_distribution_config = None
    target_distribution_source = "uniform"

    if target_distribution_cli is not None:
        target_distribution_config = target_distribution_cli
        target_distribution_source = "cli"
        logger.info(f"Using target distribution from CLI: {target_distribution_config}")
    else:
        try:
            cfg_data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
            if cfg_data and "target_distribution" in cfg_data and cfg_data["target_distribution"]:
                target_distribution_config = cfg_data["target_distribution"]
                target_distribution_source = "config"
                logger.info(f"Using target distribution from args.yaml: {target_distribution_config}")
        except Exception as e:
            logger.debug(f"Could not read target_distribution from config: {e}")

    target_dists = parse_target_distribution(
        target_dist_config=target_distribution_config,
        classifier_heads=list(attribute_keys),
        warn_fn=logger.warning,
    )

    # Compute joint target distribution
    joint_key, joint_target = compute_joint_target_dist(target_dists)
    target_dists[joint_key] = joint_target
    support_sizes[joint_key] = joint_target.shape[0]

    merge_maps = get_classifier_probs_merge_maps(classifier_name, attribute_keys)

    device = torch.device(device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu"))
    if not torch.cuda.is_available(): logger.warning("CUDA not available; using CPU for evaluation.")

    model = get_classifier(model_name=classifier_name, device=device)

    pngs = collect_pngs(run_dir, max_samples=max_samples)
    if len(pngs) == 0:
        logger.warning(f"[SKIP] {run_dir}: no .png files found.")
        return
    
    if not isinstance(max_samples, int):
        max_samples = len(pngs)

    # Include joint key in tracking
    all_keys = list(attribute_keys) + [joint_key]
    soft_sums = {k: torch.zeros((support_sizes[k],), dtype=torch.float64) for k in all_keys}
    hard_counts = {k: torch.zeros((support_sizes[k],), dtype=torch.int64) for k in all_keys}
    if save_intermediate:
        logger.info(f"Saving intermediate results...")
        all_probs_intermediate = {k: torch.empty((0, support_sizes[k]), dtype=torch.float64) for k in all_keys}
        all_classes_intermediate = {k: torch.empty((0,), dtype=torch.int64) for k in all_keys}

    N = 0
    load_chunk = max(batch_size * 4, 128)

    # Support sizes for classifier (excluding joint)
    classifier_support_sizes = {k: get_support_size(k) for k in attribute_keys}

    for i in range(0, len(pngs), load_chunk):
        chunk_paths = pngs[i:i+load_chunk]
        batch_uint8 = load_batch(chunk_paths)  # [B, C, H, W], uint8
        probs_dict = infer_probs(model, classifier_name, classifier_support_sizes, batch_uint8,
                                batch_size=batch_size, device=device,
                                prob_merge_maps=merge_maps,
                                num_race_cls=7 if classifier_name == "fairface_7" else 4)  # [B, K]

        # Compute joint distribution from marginals
        _, joint_probs = compute_joint_probs(probs_dict)
        probs_dict[joint_key] = joint_probs

        # soft_sum += probs.sum(axis=0)
        soft_sums = dict_apply_reduce([soft_sums, probs_dict], lambda tup_: tup_[0] + tup_[1].sum(dim=0).to(torch.float64))
        # hard_labels = probs.argmax(axis=1)
        hard_labels = dict_apply(probs_dict, lambda x_: x_.argmax(dim=1))
        hard_counts = dict_apply_reduce(
            [hard_counts, hard_labels],
            lambda tup_: tup_[0].index_add_(0, tup_[1], torch.ones_like(tup_[1], dtype=torch.int64))
        )
        N += batch_uint8.shape[0]
        if save_intermediate:
            all_probs_intermediate = dict_apply_reduce(
                [all_probs_intermediate, probs_dict],
                lambda tup_: torch.cat([tup_[0], tup_[1].to(torch.float64)], dim=0)
            )
            all_classes_intermediate = dict_apply_reduce(
                [all_classes_intermediate, hard_labels],
                lambda tup_: torch.cat([tup_[0], tup_[1].to(torch.int64)], dim=0)
            )

    soft_dists = dict_apply(soft_sums, lambda x_: (x_ / max(N, 1)))
    hard_dists = dict_apply(hard_counts, lambda x_: (x_.double() / max(N, 1)))

    kl_gen_tar_dict = dict_apply_reduce([hard_dists, target_dists], lambda tup_: kl_div(tup_[0], tup_[1]))
    kl_tar_gen_dict = dict_apply_reduce([target_dists, hard_dists], lambda tup_: kl_div(tup_[0], tup_[1]))
    tv_dict = dict_apply_reduce([hard_dists, target_dists], lambda tup_: tv_dist(tup_[0], tup_[1]))
    l2_dict = dict_apply_reduce([hard_dists, target_dists], lambda tup_: l2_dist(tup_[0], tup_[1]))
    js_dict = dict_apply_reduce([hard_dists, target_dists], lambda tup_: js_dist(tup_[0], tup_[1]))
    chi2_dict = dict_apply_reduce([hard_dists, target_dists], lambda tup_: chi2_dist(tup_[0], tup_[1]))
    fd_dict = dict_apply_reduce([soft_dists, target_dists], lambda tup_: fairness_discrepancy(tup_[0], tup_[1]))

    ## Retrieve FID reference stats path if not provided
    if fid_ref_path is None:
        # fid_ref_path = get_fid_ref_stats_path(
        #     target_dist=target_dist_name,
        #     classifier=classifier_name,
        #     num_expected=max_samples,
        #     seed=fid_seed,
        #     batch_size=batch_size,
        # )
        pass
    ## Calculate FID
    if fid_ref_path is not None:
        fid = calc_fid(
            run_dir.as_posix(), fid_ref_path, max_samples, fid_seed, batch_size,
            detector_net=detector_net,
        )
    else:
        fid = None

    metrics = {
        "kl_gen_tar": kl_gen_tar_dict,
        "kl_tar_gen": kl_tar_gen_dict,
        "tv":         tv_dict,
        "l2":         l2_dict,
        "js":         js_dict,
        "chi2":       chi2_dict,
        "fd":         fd_dict, 
        "fid": fid if fid is not None else -1.0,
    }

    out = {
        "evaluated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "target_distribution_name": target_dist_name,
        "target_distribution_source": target_distribution_source,
        "num_images": int(N),
        "classifier": classifier_name,
        "num_classes": support_sizes,
        "fid_ref_stats": fid_ref_path if fid_ref_path is not None else "",
        "fid_seed": fid_seed,
        "metrics_vs_target": metrics,
        "soft_distribution": dict_apply(soft_dists, lambda x_: x_.numpy().tolist()),
        "hard_distribution": dict_apply(hard_dists, lambda x_: x_.numpy().tolist()),
        "hard_counts": dict_apply(hard_counts, lambda x_: x_.numpy().tolist()),
        "target_distribution": dict_apply(target_dists, lambda x_: x_.numpy().tolist()),
    }
    save_name_cls = f"{save_name.rsplit('.', 1)[0]}_n{N}_{classifier_name}_{target_dist_name}.json"
    save_path = run_dir / save_name_cls
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4)
    if save_intermediate:
        # Save intermediate data as JSON
        intermediate_data = {
            "probs": dict_apply(all_probs_intermediate, lambda x_: x_.float().numpy().tolist()),
            "classes": dict_apply(all_classes_intermediate, lambda x_: x_.long().numpy().tolist()),
        }
        intermediate_path = run_dir / f"{save_name_cls.rsplit('.', 1)[0]}_intermediate.json"
        logger.info(f"Saving intermediate per-image probabilities & classes to '{intermediate_path}'")
        with open(intermediate_path, "w", encoding="utf-8") as f:
            json.dump(intermediate_data, f, indent=4)
    
    logger.info(f"[OK] {run_dir} -> {save_path} (N={N}, Attributes={attribute_keys}, cls={classifier_name})")


# ---------------------------------------------------------------------
def _expand_path(p: str) -> str:
    return os.path.expandvars(os.path.expanduser(p.strip()))

def _read_runs_file(path: Path) -> List[str]:
    lines = []
    txt = path.read_text(encoding="utf-8").splitlines()
    for line in txt:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        lines.append(_expand_path(s))
    return lines

def _collect_run_dirs(runs_cli: List[str], runs_file: Optional[str]) -> List[Path]:
    paths: List[str] = []
    # from CLI
    for pat in runs_cli or []:
        pat = _expand_path(pat)
        matches = glob.glob(pat)
        if matches:
            paths.extend(matches)
        else:
            paths.append(pat)
    # from file
    if runs_file:
        file_path = Path(_expand_path(runs_file))
        if not file_path.exists():
            raise FileNotFoundError(f"--runs-file not found: {file_path}")
        for pat in _read_runs_file(file_path):
            matches = glob.glob(pat)
            if matches:
                paths.extend(matches)
            else:
                paths.append(pat)
    # normalize, dedupe, keep order
    seen = set()
    normed: List[Path] = []
    for p in paths:
        q = str(Path(p).resolve())
        if q not in seen:
            seen.add(q)
            normed.append(Path(q))
    return normed

# ---------------------------------------------------------------------
def load_target_config(config_path: str) -> Dict[str, List[float]]:
    """
    Load target distribution config from JSON file.

    Args:
        config_path: Path to JSON config file.

    Returns:
        Dict mapping attribute names to probability lists.

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
    logger.info(f"Loading target distribution config from '{config_path}'...")
    with open(config_path, 'r') as f:
        config = json.load(f)

    if "target_distribution" not in config:
        raise ValueError(f"Config file missing 'target_distribution' key: {config_path}")

    target_dist = config["target_distribution"]
    support_sizes = config.get("support_sizes", {
        "gender": 2,
        "age_group": 3,
        "race": 4,
    })

    result = {}
    for attr, value in target_dist.items():
        if value is None:
            # Explicitly null - use uniform based on support_sizes
            size = support_sizes.get(attr)
            if size is None:
                logger.warning(f"Attribute '{attr}' has null value but no support_size specified, skipping")
                continue
            result[attr] = [1.0 / size] * size
        else:
            # Array specified - normalize
            probs = np.array(value, dtype=np.float64)
            result[attr] = (probs / probs.sum()).tolist()

    return result


def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate class distribution of generated samples for one or more run directories.")
    ap.add_argument("--runs", nargs="*", default=[],
                    help="Run directories (supports globs). Can be combined with --runs-file.")
    ap.add_argument("--runs-file", type=str, default=None,
                    help="Path to a text file listing run directories (one per line; supports ~, $VARS, and globs).")
    ap.add_argument("--classifier", dest="classifier_name", type=str, required=True,
                    help="Classifier name to use (e.g. 'fairface_4', 'fairface_7', 'pcd_pretrained').")
    ap.add_argument("--device", type=str, default="cuda",
                    help="Device to use, e.g. 'cuda', 'cuda:0', or 'cpu' (default: cuda if available).")
    ap.add_argument("--batch-size", type=int, default=128,
                    help="Classifier batch size for inference (default: 128).")
    ap.add_argument("--max-samples", type=int, default=0,
                    help="Max images per run to evaluate; 0 means no limit (default: 0).")
    ap.add_argument("--save-name", type=str, default="eval_class_dist.json",
                    help="Filename for the JSON saved under each run dir (default: eval_class_dist.json).")
    ap.add_argument("--fid-ref", type=str, default=None,
                    help="Path to the FID reference image (default: None).")
    ap.add_argument("--fid-seed", type=int, default=0,
                    help="Random seed for FID calculation (default: 0).")
    ap.add_argument("--attributes", nargs="*", default=None,
                    help="Attributes to evaluate (default: None); if None, by config.yaml.")
    ap.add_argument("--target-dist", type=str, default=None,
                    help="Target distribution name (overrides config.yaml if there are any).")
    ap.add_argument("--target-distribution", type=str, default=None,
                    help="Custom target distribution as JSON dict, e.g., "
                         "'{\"gender\": [0.6, 0.4], \"race\": [0.25, 0.25, 0.25, 0.25]}'. "
                         "If not provided, reads from run's args.yaml if available, "
                         "else falls back to uniform.")
    ap.add_argument("--target-config", type=str, default=None,
                    help="Path to JSON config file with target distributions. "
                         "Inline --target-distribution overrides this.")
    return ap.parse_args()

def main():
    # torch.multiprocessing.set_start_method('spawn')
    # dist.init()

    args = parse_args()

    # Parse target distribution with priority: inline JSON > JSON file > None
    target_distribution_cli = None

    # Load from config file if specified
    if args.target_config:
        try:
            logger.info(f"Loading target distribution from config file: {args.target_config}")
            target_distribution_cli = load_target_config(args.target_config)
        except Exception as e:
            logger.error(f"Failed to load target config from '{args.target_config}': {e}")
            sys.exit(1)

    # Inline JSON overrides config file
    if args.target_distribution:
        try:
            inline_dist = json.loads(args.target_distribution)
            if target_distribution_cli is None:
                target_distribution_cli = inline_dist
            else:
                # Merge: inline overrides file
                target_distribution_cli.update(inline_dist)
            logger.info(f"Using inline target distribution (overrides config file if any)")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in --target-distribution: {e}")
            sys.exit(1)

    run_dirs = _collect_run_dirs(args.runs, args.runs_file)
    if not run_dirs:
        logger.error("[ERROR] No run directories provided via --runs or --runs-file.")
        sys.exit(2)

    # Determine device and load Inception model once for FID evaluation
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    detector_net = None
    if args.fid_ref is not None or args.target_dist is not None:
        # FID evaluation will be performed - load model once
        detector_net = load_inception_model(device)

    max_samples = None if args.max_samples is None or args.max_samples <= 0 else int(args.max_samples)
    for d in tqdm(run_dirs, desc="Evaluating runs", unit="dir", ncols=100):
        try:
            if not d.exists():
                logger.warning(f"[WARN] Skipping missing run dir: {d}")
                continue
            
            evaluate_run(d, classifier_name=args.classifier_name,
                        device_str=args.device, batch_size=args.batch_size,
                        save_name=args.save_name,
                        max_samples=max_samples, target_dist_name=args.target_dist,
                        fid_ref_path=args.fid_ref, fid_seed=args.fid_seed,
                        attribute_keys=args.attributes,
                        detector_net=detector_net,
                        target_distribution_cli=target_distribution_cli,
            )
        except Exception as e:
            logger.error(f" \'{d}\': \'{e}\'")
            logger.error("Skipping to next run dir...")


if __name__ == "__main__":
    sys.exit(main())
