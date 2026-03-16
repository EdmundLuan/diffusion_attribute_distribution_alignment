"""Integration test for joint distribution evaluation."""
import torch
from guided_diffusion.sampling_util import (
    compute_joint_probs,
    compute_joint_target_dist,
    get_target_dist,
    get_support_size,
)


def test_eval_joint_distribution_integration():
    """Test the full joint distribution evaluation pipeline."""
    # Simulate the evaluate_run workflow
    attribute_keys = ["gender", "race"]

    # Step 1: Get support sizes and target distributions (as in eval_runs.py)
    support_sizes = {k: get_support_size(k) for k in attribute_keys}
    target_dists = {k: get_target_dist(k) for k in attribute_keys}

    # Step 2: Compute joint target
    joint_key, joint_target = compute_joint_target_dist(target_dists)
    target_dists[joint_key] = joint_target
    support_sizes[joint_key] = joint_target.shape[0]

    # Verify joint key and size
    assert joint_key == "gender-race"
    assert support_sizes["gender"] == 2
    assert support_sizes["race"] == 4
    assert support_sizes["gender-race"] == 8

    # Step 3: Simulate batch processing
    batch_size = 10
    probs_dict = {
        "gender": torch.softmax(torch.randn(batch_size, 2), dim=1),
        "race": torch.softmax(torch.randn(batch_size, 4), dim=1),
    }

    # Step 4: Compute joint probs
    _, joint_probs = compute_joint_probs(probs_dict)
    probs_dict[joint_key] = joint_probs

    # Verify joint probs
    assert joint_probs.shape == (batch_size, 8)
    assert torch.allclose(joint_probs.sum(dim=1), torch.ones(batch_size), atol=1e-5)

    # Step 5: Verify all keys are present
    assert set(probs_dict.keys()) == {"gender", "race", "gender-race"}
    assert set(target_dists.keys()) == {"gender", "race", "gender-race"}
    assert set(support_sizes.keys()) == {"gender", "race", "gender-race"}

    print("✓ Joint distribution integration test passed")


def test_joint_distribution_three_attributes():
    """Test joint distribution with three attributes."""
    attribute_keys = ["gender", "race", "age_group"]

    support_sizes = {k: get_support_size(k) for k in attribute_keys}
    target_dists = {k: get_target_dist(k) for k in attribute_keys}

    joint_key, joint_target = compute_joint_target_dist(target_dists)

    assert joint_key == "age_group-gender-race"  # alphabetically sorted
    assert joint_target.shape[0] == 3 * 2 * 4  # 24

    # Simulate batch
    batch_size = 5
    probs_dict = {
        "gender": torch.softmax(torch.randn(batch_size, 2), dim=1),
        "race": torch.softmax(torch.randn(batch_size, 4), dim=1),
        "age_group": torch.softmax(torch.randn(batch_size, 3), dim=1),
    }

    _, joint_probs = compute_joint_probs(probs_dict)

    assert joint_probs.shape == (batch_size, 24)
    assert torch.allclose(joint_probs.sum(dim=1), torch.ones(batch_size), atol=1e-5)

    print("✓ Three-attribute joint distribution test passed")


if __name__ == "__main__":
    test_eval_joint_distribution_integration()
    test_joint_distribution_three_attributes()
    print("\n✓ All integration tests passed!")
