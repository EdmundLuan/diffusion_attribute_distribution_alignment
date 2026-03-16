"""Tests for joint probability computation in sampling_util."""
import pytest
import torch
from guided_diffusion.sampling_util import compute_joint_probs, compute_joint_target_dist


def test_compute_joint_probs_two_attributes():
    """Test joint probability computation for two attributes."""
    probs_dict = {
        "gender": torch.tensor([[0.3, 0.7], [0.6, 0.4]]),  # B=2, K=2
        "race": torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.25, 0.25, 0.25, 0.25]]),  # B=2, K=4
    }

    joint_key, joint_probs = compute_joint_probs(probs_dict)

    assert joint_key == "gender-race"  # sorted alphabetically
    assert joint_probs.shape == (2, 8)  # B=2, K=2*4=8

    # First sample: outer product of [0.3, 0.7] and [0.1, 0.2, 0.3, 0.4]
    # = [0.03, 0.06, 0.09, 0.12, 0.07, 0.14, 0.21, 0.28]
    expected_0 = torch.tensor([0.03, 0.06, 0.09, 0.12, 0.07, 0.14, 0.21, 0.28])
    assert torch.allclose(joint_probs[0], expected_0, atol=1e-6)

    # Joint probs should sum to 1 for each sample
    assert torch.allclose(joint_probs.sum(dim=1), torch.ones(2), atol=1e-6)


def test_compute_joint_probs_three_attributes():
    """Test joint probability with three attributes."""
    probs_dict = {
        "gender": torch.tensor([[0.5, 0.5]]),  # B=1, K=2
        "race": torch.tensor([[0.25, 0.25, 0.25, 0.25]]),  # B=1, K=4
        "age_group": torch.tensor([[1/3, 1/3, 1/3]]),  # B=1, K=3
    }

    joint_key, joint_probs = compute_joint_probs(probs_dict)

    assert joint_key == "age_group-gender-race"  # sorted alphabetically
    assert joint_probs.shape == (1, 24)  # B=1, K=3*2*4=24

    # Uniform marginals -> uniform joint
    expected = torch.ones(24) / 24
    assert torch.allclose(joint_probs[0], expected, atol=1e-6)


def test_compute_joint_target_dist():
    """Test joint target distribution computation."""
    target_dists = {
        "gender": torch.tensor([0.5, 0.5]),
        "race": torch.tensor([0.25, 0.25, 0.25, 0.25]),
    }

    joint_key, joint_target = compute_joint_target_dist(target_dists)

    assert joint_key == "gender-race"
    assert joint_target.shape == (8,)  # K=2*4=8
    assert torch.allclose(joint_target.sum(), torch.tensor(1.0), atol=1e-6)

    # Uniform marginals -> uniform joint
    expected = torch.ones(8) / 8
    assert torch.allclose(joint_target, expected, atol=1e-6)


def test_compute_joint_target_dist_nonuniform():
    """Test joint target with non-uniform marginals."""
    target_dists = {
        "gender": torch.tensor([0.3, 0.7]),
        "race": torch.tensor([0.1, 0.2, 0.3, 0.4]),
    }

    joint_key, joint_target = compute_joint_target_dist(target_dists)

    assert joint_key == "gender-race"
    assert joint_target.shape == (8,)

    # Outer product: [0.3, 0.7] x [0.1, 0.2, 0.3, 0.4]
    # = [0.03, 0.06, 0.09, 0.12, 0.07, 0.14, 0.21, 0.28]
    expected = torch.tensor([0.03, 0.06, 0.09, 0.12, 0.07, 0.14, 0.21, 0.28])
    assert torch.allclose(joint_target, expected, atol=1e-6)
    assert torch.allclose(joint_target.sum(), torch.tensor(1.0), atol=1e-6)
