"""Tests for custom target distribution support in image_sample_dps.py."""
import json
import pytest
import tempfile
import os
from unittest.mock import patch, MagicMock
from scripts.image_sample_dps import get_defaults
from guided_diffusion.sampling_util import load_target_config


def test_get_defaults_includes_target_distribution():
    """Verify get_defaults() includes target_distribution=None."""
    defaults = get_defaults()
    assert 'target_distribution' in defaults
    assert defaults['target_distribution'] is None


def test_json_string_parsing():
    """Test JSON string parsing logic that batch_exec.py uses."""
    # Simulate the JSON parsing logic from main()
    target_dist_str = '{"gender": [0.7, 0.3], "race": [0.1, 0.2, 0.3, 0.4]}'
    parsed = json.loads(target_dist_str)

    assert isinstance(parsed, dict)
    assert "gender" in parsed
    assert "race" in parsed
    assert parsed["gender"] == [0.7, 0.3]
    assert parsed["race"] == [0.1, 0.2, 0.3, 0.4]


def test_json_parsing_invalid_raises():
    """Test that invalid JSON raises appropriate error."""
    invalid_json = '{"gender": [0.7, 0.3'  # Missing closing bracket and brace

    with pytest.raises(json.JSONDecodeError):
        json.loads(invalid_json)


@patch('scripts.image_sample_dps.parse_target_distribution')
def test_parse_target_distribution_called_with_custom_config(mock_parse):
    """Verify parse_target_distribution is called with custom config."""
    from scripts.image_sample_dps import create_multidim_classifier_loss_fn
    from unittest.mock import Mock
    import torch

    # Mock the classifier loading and config lookup
    mock_classifier = Mock()
    mock_classifier.eval = Mock()

    custom_target = {"gender": [0.8, 0.2], "race": [0.1, 0.2, 0.3, 0.4]}
    mock_parse.return_value = {
        "gender": torch.tensor([0.8, 0.2]),
        "race": torch.tensor([0.1, 0.2, 0.3, 0.4])
    }

    # Mock CLASSIFIER_CONFIGS
    with patch('scripts.image_sample_dps.CLASSIFIER_CONFIGS', {
        'test_classifier': {
            'path': {'config': 'test.yaml', 'weights': 'test.pt'},
            'get_func': lambda **kwargs: mock_classifier
        }
    }):
        with patch('scripts.image_sample_dps.CostFunctionKLMultiDimPCD'):
            with patch('scripts.image_sample_dps.logger'):
                # Call the function with custom target distributions
                create_multidim_classifier_loss_fn(
                    classifier_name='test_classifier',
                    classifier_heads=['gender', 'race'],
                    classifier_type='image',
                    device='cpu',
                    target_distributions=custom_target,
                )

    # Verify parse_target_distribution was called with the custom config
    mock_parse.assert_called_once()
    call_kwargs = mock_parse.call_args[1]
    assert call_kwargs['target_dist_config'] == custom_target
    assert call_kwargs['classifier_heads'] == ['gender', 'race']


@patch('scripts.image_sample_dps.parse_target_distribution')
def test_parse_target_distribution_called_with_none(mock_parse):
    """Verify parse_target_distribution is called with None when not specified."""
    from scripts.image_sample_dps import create_multidim_classifier_loss_fn
    from unittest.mock import Mock
    import torch

    # Mock the classifier loading and config lookup
    mock_classifier = Mock()
    mock_classifier.eval = Mock()

    # Return uniform distributions when None is passed
    mock_parse.return_value = {
        "gender": torch.tensor([0.5, 0.5]),
        "race": torch.tensor([0.25, 0.25, 0.25, 0.25])
    }

    # Mock CLASSIFIER_CONFIGS
    with patch('scripts.image_sample_dps.CLASSIFIER_CONFIGS', {
        'test_classifier': {
            'path': {'config': 'test.yaml', 'weights': 'test.pt'},
            'get_func': lambda **kwargs: mock_classifier
        }
    }):
        with patch('scripts.image_sample_dps.CostFunctionKLMultiDimPCD'):
            with patch('scripts.image_sample_dps.logger'):
                # Call the function WITHOUT custom target distributions (backward compatibility)
                create_multidim_classifier_loss_fn(
                    classifier_name='test_classifier',
                    classifier_heads=['gender', 'race'],
                    classifier_type='image',
                    device='cpu',
                    target_distributions=None,  # Explicitly None
                )

    # Verify parse_target_distribution was called with None
    mock_parse.assert_called_once()
    call_kwargs = mock_parse.call_args[1]
    assert call_kwargs['target_dist_config'] is None
    assert call_kwargs['classifier_heads'] == ['gender', 'race']


def test_get_defaults_includes_target_config():
    """Verify get_defaults() includes target_config=None."""
    defaults = get_defaults()
    assert 'target_config' in defaults
    assert defaults['target_config'] is None


def test_load_target_config_loads_json():
    """Test load_target_config() loads JSON correctly."""
    config_data = {
        "name": "test_config",
        "support_sizes": {"gender": 2, "race": 4},
        "target_distribution": {
            "gender": [0.3, 0.7],
            "race": [0.4, 0.3, 0.2, 0.1]
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_data, f)
        temp_path = f.name

    try:
        result = load_target_config(temp_path)

        # Check that values are loaded and normalized
        assert "gender" in result
        assert "race" in result
        assert result["gender"] == pytest.approx([0.3, 0.7])
        assert result["race"] == pytest.approx([0.4, 0.3, 0.2, 0.1])
    finally:
        os.unlink(temp_path)


def test_load_target_config_handles_null_values():
    """Test load_target_config() handles null values (uniform distribution)."""
    config_data = {
        "name": "test_config",
        "support_sizes": {"gender": 2, "age_group": 3, "race": 4},
        "target_distribution": {
            "gender": None,  # Should become uniform [0.5, 0.5]
            "age_group": [0.5, 0.2, 0.3],
            "race": None  # Should become uniform [0.25, 0.25, 0.25, 0.25]
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_data, f)
        temp_path = f.name

    try:
        result = load_target_config(temp_path)

        # Check null values became uniform
        assert result["gender"] == pytest.approx([0.5, 0.5])
        assert result["age_group"] == pytest.approx([0.5, 0.2, 0.3])
        assert result["race"] == pytest.approx([0.25, 0.25, 0.25, 0.25])
    finally:
        os.unlink(temp_path)


def test_load_target_config_normalizes():
    """Test load_target_config() normalizes distributions."""
    config_data = {
        "name": "test_config",
        "support_sizes": {"gender": 2},
        "target_distribution": {
            "gender": [3, 1]  # Not normalized, should become [0.75, 0.25]
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_data, f)
        temp_path = f.name

    try:
        result = load_target_config(temp_path)
        assert result["gender"] == pytest.approx([0.75, 0.25])
    finally:
        os.unlink(temp_path)
