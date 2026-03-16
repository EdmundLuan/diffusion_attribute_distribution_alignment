# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Script for calculating Frechet Inception Distance (FID)."""

import os
import sys
import click
import tqdm
import pickle
import numpy as np
import scipy.linalg
import torch
from guided_diffusion.dnnlib.util import open_url
import guided_diffusion.eval_util.torch_utils as torch_utils
import guided_diffusion.eval_util.torch_utils.persistence as persistence
import guided_diffusion.eval_util.torch_utils.misc as misc

# Register modules under names expected by NVIDIA's pickled Inception model
sys.modules['torch_utils'] = torch_utils
sys.modules['torch_utils.persistence'] = persistence
sys.modules['torch_utils.misc'] = misc

from PIL import Image

#----------------------------------------------------------------------------

class ImageFolderDataset(torch.utils.data.Dataset):
    """Simple PyTorch Dataset that recursively loads PNG/JPG images from a directory."""

    def __init__(self, path, max_size=None, random_seed=0):
        """
        Args:
            path: Directory path containing images (searched recursively)
            max_size: Maximum number of images to load (None = all)
            random_seed: Random seed for shuffling when max_size is set
        """
        self.path = path
        self._image_paths = []

        # Recursively find all PNG and JPG images
        for ext in ['*.png', '*.PNG', '*.jpg', '*.JPG', '*.jpeg', '*.JPEG']:
            import glob as glob_module
            self._image_paths.extend(glob_module.glob(os.path.join(path, '**', ext), recursive=True))

        self._image_paths = sorted(self._image_paths)

        # Apply max_size with deterministic shuffling
        if max_size is not None and len(self._image_paths) > max_size:
            rng = np.random.default_rng(random_seed)
            indices = rng.permutation(len(self._image_paths))[:max_size]
            self._image_paths = [self._image_paths[i] for i in sorted(indices)]

    def __len__(self):
        return len(self._image_paths)

    def __getitem__(self, idx):
        """Returns (image_tensor, label) where image_tensor is [C, H, W] uint8 and label=0."""
        img_path = self._image_paths[idx]
        with Image.open(img_path) as img:
            img = img.convert('RGB')
            arr = np.array(img, dtype=np.uint8)  # [H, W, C]
        tensor = torch.from_numpy(arr).permute(2, 0, 1)  # [C, H, W]
        return tensor, 0  # dummy label

#----------------------------------------------------------------------------

def calculate_inception_stats(
    image_path, num_expected=None, seed=0, max_batch_size=64,
    num_workers=3, prefetch_factor=2, device=torch.device('cuda'),
    detector_net=None,
):
    # Load Inception-v3 model only if not provided.
    # This is a direct PyTorch translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    detector_kwargs = dict(return_features=True)
    feature_dim = 2048
    if detector_net is None:
        print('Loading Inception-v3 model...')
        detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
        with open_url(detector_url, verbose=True) as f:
            detector_net = pickle.load(f).to(device)

    # List images.
    print(f'Loading images from "{image_path}"...')
    dataset_obj = ImageFolderDataset(path=image_path, max_size=num_expected, random_seed=seed)
    if num_expected is not None and len(dataset_obj) < num_expected:
        raise click.ClickException(f'Found {len(dataset_obj)} images, but expected at least {num_expected}')
    if len(dataset_obj) < 2:
        raise click.ClickException(f'Found {len(dataset_obj)} images, but need at least 2 to compute statistics')

    # Divide images into batches.
    num_batches = (len(dataset_obj) - 1) // max_batch_size + 1
    all_batches = torch.arange(len(dataset_obj)).tensor_split(num_batches)
    data_loader = torch.utils.data.DataLoader(dataset_obj, batch_sampler=all_batches, num_workers=num_workers, prefetch_factor=prefetch_factor)

    # Accumulate statistics.
    print(f'Calculating statistics for {len(dataset_obj)} images...')
    mu = torch.zeros([feature_dim], dtype=torch.float64, device=device)
    sigma = torch.zeros([feature_dim, feature_dim], dtype=torch.float64, device=device)
    for images, _labels in tqdm.tqdm(data_loader, unit='batch', ncols=120):
        if images.shape[0] == 0:
            continue
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        features = detector_net(images.to(device), **detector_kwargs).to(torch.float64)
        mu += features.sum(0)
        sigma += features.T @ features

    # Calculate final statistics (local accumulation, no distributed reduce).
    mu /= len(dataset_obj)
    sigma -= mu.ger(mu) * len(dataset_obj)
    sigma /= len(dataset_obj) - 1
    return mu.cpu().numpy(), sigma.cpu().numpy()


#----------------------------------------------------------------------------

def calculate_inception_stats_from_dataset(
    dataset: torch.utils.data.Dataset,
    max_batch_size: int = 64,
    num_workers: int = 3,
    prefetch_factor: int = 2,
    device: torch.device = torch.device('cuda'),
    target_resolution: int = 256,
):
    """
    Calculate inception statistics from a PyTorch Dataset.

    This function uses NVIDIA's Inception-v3 model (direct PyTorch translation
    of TensorFlow's inception-2015-12-05) to extract features and compute
    statistics for FID calculation.

    The dataset's __getitem__ should return (image_tensor, ...) where
    image_tensor is [C, H, W] uint8.

    Args:
        dataset: PyTorch Dataset returning (image, labels_or_other)
        max_batch_size: Maximum batch size for inference
        num_workers: DataLoader workers
        prefetch_factor: DataLoader prefetch factor
        device: Device for inference
        target_resolution: Target image resolution for FID calculation (default: 256).
                          Images will be resized to (target_resolution, target_resolution).

    Returns:
        (mu, sigma): Inception feature statistics as numpy arrays
    """
    # Load NVIDIA's Inception-v3 model
    print('Loading Inception-v3 model...')
    detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
    detector_kwargs = dict(return_features=True)
    feature_dim = 2048
    with open_url(detector_url, verbose=True) as f:
        detector_net = pickle.load(f).to(device)

    if len(dataset) < 2:
        raise ValueError(f'Found {len(dataset)} images, but need at least 2')

    # Divide into batches
    num_batches = (len(dataset) - 1) // max_batch_size + 1
    all_batches = torch.arange(len(dataset)).tensor_split(num_batches)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=all_batches,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor
    )

    # Accumulate statistics
    print(f'Calculating statistics for {len(dataset)} images...')
    mu = torch.zeros([feature_dim], dtype=torch.float64, device=device)
    sigma = torch.zeros([feature_dim, feature_dim], dtype=torch.float64, device=device)

    for batch_data in tqdm.tqdm(data_loader, unit='batch', ncols=120):
        # Handle both (images,) and (images, labels) returns
        images = batch_data[0] if isinstance(batch_data, (tuple, list)) else batch_data

        if images.shape[0] == 0:
            continue
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])

        # Resize images to target resolution if needed
        if images.shape[2] != target_resolution or images.shape[3] != target_resolution:
            images = torch.nn.functional.interpolate(
                images.float(),
                size=(target_resolution, target_resolution),
                mode='bilinear',
                antialias=True,
            ).to(images.dtype)

        # Extract features using NVIDIA's Inception model
        features = detector_net(images.to(device), **detector_kwargs).to(torch.float64)
        mu += features.sum(0)
        sigma += features.T @ features

    # Calculate final statistics
    mu /= len(dataset)
    sigma -= mu.ger(mu) * len(dataset)
    sigma /= len(dataset) - 1

    return mu.cpu().numpy(), sigma.cpu().numpy()


# #----------------------------------------------------------------------------
# def _subset_by_target_distribution(
#     dataset_type: str,
#     path: str,
#     probs,
#     total: int,
#     *,
#     # dataset ctor kwargs
#     resolution=None,
#     use_pyspng=True,
#     use_labels=True,
#     cache=False,
#     coarse_to_meta_label_map=None,
#     coarse_to_meta_label_map_dict=None,
#     # sampling controls
#     seed: int = 0,
#     allow_oversample: bool = False,
# ):
#     """
#     Return a *MetaImageFolderDataset* sampled to match `probs` (sum=1) with exactly `total` items.

#     If `allow_oversample=True`, classes with insufficient items will be sampled with replacement.
#     """
#     # 1) Build a probe dataset to read META labels aligned to raw indices.
#     # probe = dataset.MetaImageFolderDataset(
#     #     path,
#     #     resolution=resolution,
#     #     use_pyspng=use_pyspng,
#     #     use_labels=True,
#     #     cache=cache,
#     #     xflip=False,           # important: don't let ctor duplicate indices
#     #     max_size=None,
#     #     # coarse_to_meta=coarse_to_meta,
#     #     # coarse_to_meta_dict=coarse_to_meta_dict,
#     # )
#     dataset_args = {
#         "path": path,
#         "resolution": resolution,
#         "use_pyspng": use_pyspng,
#         "use_labels": True,
#         "cache": cache,
#         "xflip": False,           # important: don't let ctor duplicate indices
#         "max_size": None,
#     }
#     if dataset_type == "meta":
#         dataset_args.update({
#             "coarse_to_meta": coarse_to_meta_label_map,
#             "coarse_to_meta_dict": coarse_to_meta_label_map_dict,
#         })
#         probe = dataset.MetaImageFolderDataset(**dataset_args)
#     elif dataset_type == "meta5":
#         dataset_args.update({
#             "coarse_to_meta": coarse_to_meta_label_map,
#             "coarse_to_meta_dict": coarse_to_meta_label_map_dict,
#         })
#         probe = dataset.Meta5ImageFolderDataset(**dataset_args)
#     elif dataset_type in ["coarse", "fine"]:
#         probe = dataset.ImageFolderDataset(**dataset_args)
#     else:
#         raise ValueError(f"Unsupported dataset_type: {dataset_type}")

#     labels = probe._get_raw_labels()  # int64, shape [N]
#     assert labels.dtype == np.int64 and labels.ndim == 1
#     K = int(labels.max()) + 1

#     probs = np.asarray(probs, dtype=np.float64)
#     assert probs.shape == (K,), f"`probs` must have length {K} (got {probs.shape})"
#     assert np.isclose(probs.sum(), 1.0), "`probs` must sum to 1.0"
#     if total is None:
#         total = len(probe)
#     assert total > 0

#     # 2) Convert probabilities to target counts (handle rounding)
#     target = np.floor(probs * total).astype(int)
#     rem = total - target.sum()
#     if rem > 0:
#         for i in np.argsort(-probs)[:rem]:
#             target[i] += 1

#     # 3) Sample per class
#     rng = np.random.default_rng(seed)
#     picked = []
#     for k, n in enumerate(target):
#         cls_idx = np.flatnonzero(labels == k)
#         if n > len(cls_idx):
#             if not allow_oversample:
#                 raise ValueError(f"Class {k}: need {n}, but only {len(cls_idx)} available. "
#                                  f"Set allow_oversample=True to sample with replacement.")
#             # with replacement
#             take = rng.choice(cls_idx, size=n, replace=True)
#         else:
#             # without replacement
#             # take = rng.choice(cls_idx, size=n, replace=False)
#             ## Take the first n of the already shuffled indices
#             take = cls_idx[:n].copy()
#         picked.extend(take.tolist())

#     rng.shuffle(picked)
#     picked = np.asarray(picked, dtype=np.int64)
#     assert picked.shape[0] == total

#     # 4) Create a fresh dataset and *override* its index view
#     # ds = dataset.ImageFolderDataset(path)
#     if dataset_type == "meta":
#         ds = dataset.MetaImageFolderDataset(path, use_labels=use_labels)
#     elif dataset_type == "meta5":
#         ds = dataset.Meta5ImageFolderDataset(path, use_labels=use_labels)
#     elif dataset_type in ["coarse", "fine"]:
#         ds = dataset.ImageFolderDataset(path, use_labels=use_labels)
#     else:
#         raise ValueError(f"Unsupported dataset_type: {dataset_type}")
#     ds._raw_idx = picked
#     ds._xflip = np.zeros_like(picked, dtype=np.uint8)

#     return ds


# def calc_inception_stats_customed_class_dist(
#     dataset_type: str,
#     image_path, num_expected=None, seed=0, max_batch_size=64,
#     num_workers=4, prefetch_factor=2, device=torch.device('cuda'),
#     class_dist=None,
# ):
#     # TODO: This function requires custom dataset classes (MetaImageFolderDataset, Meta5ImageFolderDataset)
#     # that are not yet implemented. The _subset_by_target_distribution function depends on them.

#     # Load Inception-v3 model.
#     # This is a direct PyTorch translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
#     print('Loading Inception-v3 model...')
#     detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
#     detector_kwargs = dict(return_features=True)
#     feature_dim = 2048
#     with open_url(detector_url, verbose=True) as f:
#         detector_net = pickle.load(f).to(device)

#     # List images.
#     print(f'Loading images from "{image_path}"...')
#     dataset_obj = _subset_by_target_distribution(
#         dataset_type=dataset_type,
#         path=image_path,
#         probs=class_dist,
#         total=num_expected,
#         seed=seed,
#         allow_oversample=False,
#     )

#     if num_expected is not None and len(dataset_obj) < num_expected:
#         raise click.ClickException(f'Found {len(dataset_obj)} images, but expected at least {num_expected}')
#     if len(dataset_obj) < 2:
#         raise click.ClickException(f'Found {len(dataset_obj)} images, but need at least 2 to compute statistics')

#     # Divide images into batches.
#     num_batches = (len(dataset_obj) - 1) // max_batch_size + 1
#     all_batches = torch.arange(len(dataset_obj)).tensor_split(num_batches)
#     data_loader = torch.utils.data.DataLoader(dataset_obj, batch_sampler=all_batches, num_workers=num_workers, prefetch_factor=prefetch_factor)

#     # Accumulate statistics.
#     print(f'Calculating statistics for {len(dataset_obj)} images...')
#     mu = torch.zeros([feature_dim], dtype=torch.float64, device=device)
#     sigma = torch.zeros([feature_dim, feature_dim], dtype=torch.float64, device=device)
#     for images, _labels in tqdm.tqdm(data_loader, unit='batch', ncols=80):
#         if images.shape[0] == 0:
#             continue
#         if images.shape[1] == 1:
#             images = images.repeat([1, 3, 1, 1])
#         features = detector_net(images.to(device), **detector_kwargs).to(torch.float64)
#         mu += features.sum(0)
#         sigma += features.T @ features

#     # Calculate final statistics (local accumulation, no distributed reduce).
#     mu /= len(dataset_obj)
#     sigma -= mu.ger(mu) * len(dataset_obj)
#     sigma /= len(dataset_obj) - 1
#     return mu.cpu().numpy(), sigma.cpu().numpy()

#----------------------------------------------------------------------------

def calculate_fid_from_inception_stats(mu, sigma, mu_ref, sigma_ref):
    m = np.square(mu - mu_ref).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma, sigma_ref), disp=False)
    fid = m + np.trace(sigma + sigma_ref - s * 2)
    return float(np.real(fid))

#----------------------------------------------------------------------------

@click.group()
def main():
    """Calculate Frechet Inception Distance (FID).

    Examples:

    \b
    # Generate 50000 images and save them as fid-tmp/*/*.png
    torchrun --standalone --nproc_per_node=1 generate.py --outdir=fid-tmp --seeds=0-49999 --subdirs \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl

    \b
    # Calculate FID
    torchrun --standalone --nproc_per_node=1 fid.py calc --images=fid-tmp \\
        --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz

    \b
    # Compute dataset reference statistics
    python fid.py ref --data=datasets/my-dataset.zip --dest=fid-refs/my-dataset.npz
    """

#----------------------------------------------------------------------------

@main.command()
@click.option('--images', 'image_path', help='Path to the images', metavar='PATH|ZIP',              type=str, required=True)
@click.option('--ref', 'ref_path',      help='Dataset reference statistics ', metavar='NPZ|URL',    type=str, required=True)
@click.option('--num', 'num_expected',  help='Number of images to use', metavar='INT',              type=click.IntRange(min=2), default=50000, show_default=True)
@click.option('--seed',                 help='Random seed for selecting the images', metavar='INT', type=int, default=0, show_default=True)
@click.option('--batch',                help='Maximum batch size', metavar='INT',                   type=click.IntRange(min=1), default=64, show_default=True)

def calc(image_path, ref_path, num_expected, seed, batch):
    """Calculate FID for a given set of images."""
    torch.multiprocessing.set_start_method('spawn')

    print(f'Loading dataset reference statistics from "{ref_path}"...')
    with open_url(ref_path) as f:
        ref = dict(np.load(f))

    mu, sigma = calculate_inception_stats(image_path=image_path, num_expected=num_expected, seed=seed, max_batch_size=batch)
    print('Calculating FID...')
    fid = calculate_fid_from_inception_stats(mu, sigma, ref['mu'], ref['sigma'])
    print(f'{fid:g}')
    return fid

#----------------------------------------------------------------------------

# @main.command()
# @click.option('--dataset', 'dataset_type',  help='Type of the dataset', metavar='STR', type=click.Choice(['meta', 'meta5', 'coarse', 'fine']), required=True)
# @click.option('--path', 'dataset_path',     help='Path to the dataset', metavar='PATH|ZIP', type=str, required=True)
# @click.option('--dest', 'dest_path',    help='Destination .npz file', metavar='NPZ',    type=str, required=True)
# @click.option('--batch',                help='Maximum batch size', metavar='INT',       type=click.IntRange(min=1), default=64, show_default=True)
# @click.option('--seed',                 help='Random seed for selecting the images', metavar='INT', type=int, default=0, show_default=True)
# @click.option('--num', 'num_expected',  help='Number of images to use', metavar='INT',              type=click.IntRange(min=2), default=None, show_default=True)
# @click.option('--class_dist',           help='Class distribution as a comma-separated list of probabilities summing to 1.0', metavar='STR', type=str, default=None, show_default=True)
# @click.option('--class_dist_type',      help='Certain types of class distribution', metavar='STR', type=click.Choice(['uniform','zigzag','gaussian']), default=None, show_default=True)
# @click.option('--class_dist_supp_size', help='Support size for certain class distributions (e.g., uniform)', metavar='INT', type=click.IntRange(min=1), default=None, show_default=True)
# @click.option('--class_dist_param',    help='Additional parameter for certain class distributions (e.g., stddev for gaussian)', metavar='FLOAT', type=float, default=1.0, show_default=True)

# def ref(dataset_type, dataset_path, dest_path, batch, seed, num_expected, class_dist, class_dist_type, class_dist_supp_size, class_dist_param):
#     """Calculate dataset reference statistics needed by 'calc'."""
#     torch.multiprocessing.set_start_method('spawn')

#     customed_class_dist = False
#     if class_dist is not None:
#         class_dist = [float(x) for x in class_dist.split(',')]
#         customed_class_dist = True
#     elif class_dist_type is not None:
#         from .distributions import dist_registry
#         dist_gen_fn = dist_registry[class_dist_type]
#         class_dist = dist_gen_fn(class_dist_supp_size, class_dist_param).cpu().numpy()
#         customed_class_dist = True

#     if customed_class_dist:
#         print(f'Using custom class distribution: {class_dist}')
#         mu, sigma = calc_inception_stats_customed_class_dist(
#             dataset_type=dataset_type,
#             image_path=dataset_path,
#             num_expected=num_expected,
#             seed=seed,
#             max_batch_size=batch,
#             class_dist=np.array(class_dist),
#         )
#     else:
#         mu, sigma = calculate_inception_stats(image_path=dataset_path, num_expected=num_expected, seed=seed, max_batch_size=batch)

#     print(f'Saving dataset reference statistics to "{dest_path}"...')
#     if os.path.dirname(dest_path):
#         os.makedirs(os.path.dirname(dest_path), exist_ok=True)
#     np.savez(dest_path, mu=mu, sigma=sigma, num_samples=num_expected, seed=seed, batch_size=batch, class_dist=class_dist)
#     print('Done.')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
