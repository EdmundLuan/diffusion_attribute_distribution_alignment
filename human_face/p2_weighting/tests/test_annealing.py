"""
Unit tests for AnnealedCostFunction and temperature scheduling.
"""

import pytest
import torch as th
import torch.nn as nn
import numpy as np
from guided_diffusion.cost_functions.cost_functions import AnnealedCostFunction, CostFunction

class MockCostFunction(CostFunction):
    def __init__(self, device):
        # Initialize nn.Module to ensure hooks/buffers are setup
        nn.Module.__init__(self)
        self.device = device
        self.temperature = 1.0
    
    def forward(self, x, **kwargs):
        # Return a dummy cost based on temperature to verify propagation
        # Cost = mean(x) / temperature
        return x.mean(dim=(1, 2, 3))

@pytest.fixture
def mock_cost_fn():
    return MockCostFunction(device=th.device("cpu"))

class TestAnnealedCostFunctionSchedules:
    """Test temperature calculation logic."""

    def test_constant_schedule(self, mock_cost_fn):
        annealer = AnnealedCostFunction(
            cost_fn=mock_cost_fn,
            temp_schedule='constant',
            temp_start=2.0,
            temp_end=0.5,
            total_iters=10
        )
        # Should always return start temp
        assert annealer.cached_temps[0] == 2.0
        assert annealer.cached_temps[5] == 2.0
        assert annealer.cached_temps[9] == 2.0

    def test_linear_schedule(self, mock_cost_fn):
        annealer = AnnealedCostFunction(
            cost_fn=mock_cost_fn,
            temp_schedule='linear',
            temp_start=2.0,
            temp_end=1.0,
            total_iters=5
        )
        # Iter 0 -> start
        assert np.isclose(annealer.cached_temps[0], 2.0)
        # Iter 4 (last of 5) -> end
        assert np.isclose(annealer.cached_temps[4], 1.0)
        # Iter 2 (middle) -> 1.5
        assert np.isclose(annealer.cached_temps[2], 1.5)

    def test_cosine_schedule(self, mock_cost_fn):
        annealer = AnnealedCostFunction(
            cost_fn=mock_cost_fn,
            temp_schedule='cosine',
            temp_start=2.0,
            temp_end=1.0,
            total_iters=5
        )
        # Iter 0 -> start (cos(0)=1 -> factor=1)
        assert np.isclose(annealer.cached_temps[0], 2.0)
        # Iter 4 -> end (cos(pi)=-1 -> factor=0)
        assert np.isclose(annealer.cached_temps[4], 1.0)
        # Iter 2 -> mid (cos(pi/2)=0 -> factor=0.5) -> 1.5
        assert np.isclose(annealer.cached_temps[2], 1.5)

class TestAnnealedCostFunctionScaling:
    """Test cost scaling logic."""

    def test_no_compensation(self, mock_cost_fn):
        annealer = AnnealedCostFunction(
            cost_fn=mock_cost_fn,
            temp_schedule='constant',
            temp_start=2.0,
            cost_scale=1.5,
            compensation_strategy='none',
            total_iters=10
        )
        x = th.ones(1, 3, 4, 4)
        # Base cost = 1.0
        # Output should be base * cost_scale = 1.5
        out = annealer(x, iteration=0)
        assert np.isclose(out.item(), 1.5)

    def test_linear_compensation(self, mock_cost_fn):
        # S = T * cost_scale
        annealer = AnnealedCostFunction(
            cost_fn=mock_cost_fn,
            temp_schedule='constant',
            temp_start=2.0,
            cost_scale=1.5,
            compensation_strategy='linear',
            total_iters=10
        )
        x = th.ones(1, 3, 4, 4)
        # Effective scale = 2.0 * 1.5 = 3.0
        out = annealer(x, iteration=0)
        assert np.isclose(out.item(), 3.0)

    def test_temp_propagation(self, mock_cost_fn):
        """Verify temperature is set on the underlying cost function."""
        annealer = AnnealedCostFunction(
            cost_fn=mock_cost_fn,
            temp_schedule='constant',
            temp_start=5.0,
            total_iters=10
        )
        x = th.randn(1, 3, 4, 4)
        annealer(x, iteration=0)
        assert mock_cost_fn.temperature == 5.0

class TestAnnealedCostFunctionIntegration:
    """Integration style tests."""
    
    def test_annealing_loop(self, mock_cost_fn):
        annealer = AnnealedCostFunction(
            cost_fn=mock_cost_fn,
            temp_schedule='linear',
            temp_start=10.0,
            temp_end=1.0,
            compensation_strategy='linear',
            total_iters=2
        )
        x = th.ones(1, 3, 4, 4)
        
        # Iter 0: T=10, Scale=10. Cost=10.
        out0 = annealer(x, iteration=0)
        assert np.isclose(out0.item(), 10.0)
        assert mock_cost_fn.temperature == 10.0

        # Iter 1: T=1, Scale=1. Cost=1.
        out1 = annealer(x, iteration=1)
        assert np.isclose(out1.item(), 1.0)
        assert mock_cost_fn.temperature == 1.0
