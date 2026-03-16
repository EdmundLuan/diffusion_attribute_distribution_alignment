"""Tests for custom target distribution parsing."""
import pytest
import torch
from guided_diffusion.sampling_util import parse_target_distribution


def test_parse_none_config_uses_uniform():
    """When config is None, return uniform distributions with warning."""
    warnings = []
    result = parse_target_distribution(
        target_dist_config=None,
        classifier_heads=["gender", "race"],
        warn_fn=lambda msg: warnings.append(msg),
    )
    assert torch.allclose(result["gender"], torch.tensor([0.5, 0.5]))
    assert torch.allclose(result["race"], torch.tensor([0.25, 0.25, 0.25, 0.25]))
    assert any("uniform" in w.lower() for w in warnings)


def test_parse_custom_distribution():
    """Custom distributions are correctly parsed and normalized."""
    result = parse_target_distribution(
        target_dist_config={"gender": [0.7, 0.3], "race": [0.1, 0.2, 0.3, 0.4]},
        classifier_heads=["gender", "race"],
        warn_fn=print,
    )
    assert torch.allclose(result["gender"], torch.tensor([0.7, 0.3]))
    assert torch.allclose(result["race"], torch.tensor([0.1, 0.2, 0.3, 0.4]))


def test_normalization():
    """Distributions are normalized to sum to 1."""
    result = parse_target_distribution(
        target_dist_config={"gender": [3, 1]},  # Not normalized
        classifier_heads=["gender"],
        warn_fn=print,
    )
    assert torch.allclose(result["gender"], torch.tensor([0.75, 0.25]))


def test_partial_config_uses_uniform_for_missing():
    """Missing heads use uniform with warning."""
    warnings = []
    result = parse_target_distribution(
        target_dist_config={"gender": [0.6, 0.4]},
        classifier_heads=["gender", "race"],
        warn_fn=lambda msg: warnings.append(msg),
    )
    assert torch.allclose(result["gender"], torch.tensor([0.6, 0.4]))
    assert torch.allclose(result["race"], torch.tensor([0.25, 0.25, 0.25, 0.25]))
    assert any("race" in w for w in warnings)


def test_validation_wrong_size_raises():
    """Wrong distribution size raises ValueError."""
    with pytest.raises(ValueError, match="expected 2"):
        parse_target_distribution(
            target_dist_config={"gender": [0.3, 0.3, 0.4]},  # 3 elements, gender has 2
            classifier_heads=["gender"],
            warn_fn=print,
        )
