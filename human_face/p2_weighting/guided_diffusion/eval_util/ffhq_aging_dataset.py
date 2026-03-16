"""
FFHQ-Aging Dataset with multi-attribute label support and joint distribution sampling.

This module provides a PyTorch Dataset class for FFHQ-Aging images with multi-attribute
labels (gender, age_group, race) and utilities for sampling subsets that match target
joint distributions defined as product of marginals.
"""

import os
import json
import numpy as np
import torch as th
from PIL import Image
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from guided_diffusion.classifiers.latent_classifier_resnet_enc_multihead import (
    GENDER_ID2LABEL,
    AGE_GROUP_ID2LABEL,
    RACE_ID2LABEL,
    AGE_10_TO_3_MAP,
)


# Construct LABEL2ID mappings from imported ID2LABEL
GENDER_LABEL2ID = {v: k for k, v in GENDER_ID2LABEL.items()}
AGE_GROUP_LABEL2ID = {v: k for k, v in AGE_GROUP_ID2LABEL.items()}
AGE_GROUP_LABEL2ID["70+"] = 9  # Alias for 70-120
RACE_LABEL2ID = {v: k for k, v in RACE_ID2LABEL.items()}

# Add coarse age group ID2LABEL (3 classes)
AGE_GROUP_COARSE_ID2LABEL = {0: "0-19", 1: "20-49", 2: "50+"}

ATTRIBUTE_SIZES = {
    "gender": 2,
    "age_group": 3,        # Coarse (3-class) is now the default
    "age_group_fine": 10,  # Fine-grained (10-class)
    "race": 4,
}


class FFHQAgingDataset(th.utils.data.Dataset):
    """
    PyTorch Dataset for FFHQ-Aging with multi-attribute labels.

    The dataset loads images from a directory and associated labels from a JSON file.
    Each image has four attributes: gender, age_group (3-class coarse), age_group_fine (10-class), and race.

    Args:
        images_dir: Path to images directory (e.g., data/ffhq_aging/images/images512x512)
        labels_json: Path to labels JSON file
        image_ids: Optional list of image IDs to include (e.g., ["00000", "00001", ...])
                   If None, all images in the labels file are used.
        transform: Optional image transform callable

    Returns:
        __getitem__ returns (image_tensor, labels_dict) where:
            - image_tensor: [C, H, W] uint8 tensor
            - labels_dict: Dict with keys 'gender', 'age_group' (3-class coarse),
                          'age_group_fine' (10-class), 'race' (integer indices)
    """

    def __init__(
        self,
        images_dir: str,
        labels_json: str,
        image_ids: Optional[List[str]] = None,
        transform: Optional[callable] = None,
    ):
        self.images_dir = images_dir
        self.labels_json = labels_json
        self.transform = transform

        # Load labels from JSON
        with open(labels_json, 'r') as f:
            self._all_labels = json.load(f)

        # Filter to specified image IDs or use all
        if image_ids is not None:
            self._image_ids = [iid for iid in image_ids if iid in self._all_labels]
        else:
            self._image_ids = sorted(self._all_labels.keys())

        # Parse labels into integer indices
        self._parsed_labels = self._parse_labels()

    def _parse_labels(self) -> Dict[str, Dict[str, int]]:
        """Convert string labels to integer indices."""
        parsed = {}
        for iid in self._image_ids:
            raw = self._all_labels[iid]
            age_group_fine = AGE_GROUP_LABEL2ID[raw["age_group"]]
            parsed[iid] = {
                "gender": GENDER_LABEL2ID[raw["gender"]],
                "age_group": AGE_10_TO_3_MAP[age_group_fine],      # Coarse 3-class
                "age_group_fine": age_group_fine,                   # Fine 10-class
                "race": RACE_LABEL2ID[raw["race_4"]],
            }
        return parsed

    def __len__(self) -> int:
        return len(self._image_ids)

    def __getitem__(self, idx: int) -> Tuple[th.Tensor, Dict[str, int]]:
        """
        Returns:
            image: [C, H, W] uint8 tensor
            labels: Dict with keys 'gender', 'age_group' (3-class coarse),
                    'age_group_fine' (10-class), 'race' (integer indices)
        """
        iid = self._image_ids[idx]
        img_path = os.path.join(self.images_dir, f"{iid}.png")

        with Image.open(img_path) as img:
            img = img.convert('RGB')
            arr = np.array(img, dtype=np.uint8)

        tensor = th.from_numpy(arr).permute(2, 0, 1)  # [C, H, W]

        if self.transform is not None:
            tensor = self.transform(tensor)

        return tensor, self._parsed_labels[iid]

    def get_label_arrays(self) -> Dict[str, np.ndarray]:
        """
        Return label arrays for all images.

        This is useful for sampling algorithms that need to inspect all labels at once.

        Returns:
            Dict mapping attribute name to numpy array of integer labels
        """
        n = len(self._image_ids)
        labels = {
            "gender": np.zeros(n, dtype=np.int64),
            "age_group": np.zeros(n, dtype=np.int64),           # Coarse 3-class
            "age_group_fine": np.zeros(n, dtype=np.int64),      # Fine 10-class
            "race": np.zeros(n, dtype=np.int64),
        }
        for i, iid in enumerate(self._image_ids):
            for attr in labels:
                labels[attr][i] = self._parsed_labels[iid][attr]
        return labels


def subset_by_joint_marginals(
    dataset: FFHQAgingDataset,
    marginals: Dict[str, np.ndarray],
    total: int,
    seed: int = 0,
    allow_oversample: bool = False,
) -> FFHQAgingDataset:
    """
    Create a subset of the dataset matching the target joint distribution.

    The joint distribution is computed as the product of marginals:
        P(gender, age_group, race) = P(gender) * P(age_group) * P(race)

    This function samples images from the dataset to match the target joint distribution
    as closely as possible. The sampling is deterministic given a seed.

    Args:
        dataset: Source FFHQAgingDataset
        marginals: Dict mapping attribute name to probability array. Each array must sum to 1.
            Example: {
                "gender": np.array([0.5, 0.5]),
                "age_group": np.array([0.1] * 10),
                "race": np.array([0.25, 0.25, 0.25, 0.25])
            }
        total: Total number of samples to select
        seed: Random seed for reproducibility
        allow_oversample: If True, sample with replacement for bins that don't have enough images.
                         If False, raise ValueError when a bin has insufficient samples.

    Returns:
        New FFHQAgingDataset with subset of image_ids matching the target distribution

    Raises:
        ValueError: If marginals don't sum to 1.0 or if a bin has insufficient samples
                   (when allow_oversample=False)
    """
    rng = np.random.default_rng(seed)
    labels = dataset.get_label_arrays()
    n = len(dataset)

    # Validate marginals
    attr_order = ["gender", "age_group", "race"]
    for attr in attr_order:
        if attr not in marginals:
            raise ValueError(f"Missing marginal for attribute: {attr}")
        marg = np.asarray(marginals[attr], dtype=np.float64)
        if not np.isclose(marg.sum(), 1.0):
            raise ValueError(f"Marginal for {attr} does not sum to 1.0 (sum={marg.sum()})")
        marginals[attr] = marg

    # Compute joint distribution as outer product
    # P(g, a, r) = P(g) * P(a) * P(r)
    joint_probs = np.einsum(
        'g,a,r->gar',
        marginals["gender"],
        marginals["age_group"],
        marginals["race"]
    )

    # Compute target counts with rounding adjustment
    target_counts = np.floor(joint_probs * total).astype(int)
    remainder = total - target_counts.sum()

    # Assign remainder to bins with highest fractional parts
    if remainder > 0:
        fractional = (joint_probs * total) - target_counts
        flat_indices = np.argsort(fractional.ravel())[::-1][:remainder]
        for flat_idx in flat_indices:
            idx = np.unravel_index(flat_idx, target_counts.shape)
            target_counts[idx] += 1

    # Group dataset indices by (gender, age_group, race) tuple
    bins = defaultdict(list)
    for i in range(n):
        key = (labels["gender"][i], labels["age_group"][i], labels["race"][i])
        bins[key].append(i)

    # Sample from each bin
    picked_indices = []
    insufficient_bins = []

    for idx in np.ndindex(target_counts.shape):
        g, a, r = idx
        count = target_counts[idx]
        if count == 0:
            continue

        available = bins[(g, a, r)]
        if count > len(available):
            if not allow_oversample:
                insufficient_bins.append((g, a, r, count, len(available)))
                continue
            # Sample with replacement
            sampled = rng.choice(available, size=count, replace=True)
        else:
            # Sample without replacement
            sampled = rng.choice(available, size=count, replace=False)

        picked_indices.extend(sampled.tolist())

    # Report errors if any bins were insufficient
    if insufficient_bins:
        error_msg = "Insufficient samples in the following bins:\n"
        for g, a, r, needed, avail in insufficient_bins:
            error_msg += f"  (gender={GENDER_ID2LABEL[g]}, age_group={AGE_GROUP_ID2LABEL[a]}, "
            error_msg += f"race={RACE_ID2LABEL[r]}): need {needed}, have {avail}\n"
        error_msg += "Set allow_oversample=True to sample with replacement."
        raise ValueError(error_msg)

    # Shuffle the selected indices
    rng.shuffle(picked_indices)

    # Create new dataset with selected image IDs
    selected_ids = [dataset._image_ids[i] for i in picked_indices]

    return FFHQAgingDataset(
        images_dir=dataset.images_dir,
        labels_json=dataset.labels_json,
        image_ids=selected_ids,
        transform=dataset.transform,
    )
