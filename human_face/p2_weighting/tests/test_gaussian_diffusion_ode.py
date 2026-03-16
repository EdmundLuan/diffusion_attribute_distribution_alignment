"""
Unit tests for GaussianDiffusionDDIMODE and emsa_solve method.

Tests cover:
- Class initialization and inheritance
- zdot computation
- _DiffusionDynamicsAdapter functionality
- emsa_solve method (shapes, convergence, determinism)
- Edge cases
"""

import numpy as np
import pytest
import torch as th

from guided_diffusion.gaussian_diffusion import (
    GaussianDiffusion,
    ModelMeanType,
    ModelVarType,
    LossType,
)
from guided_diffusion.gaussian_diffusion_ode import GaussianDiffusionDDIMODE


# =============================================================================
# Helper Functions and Mock Classes
# =============================================================================


def get_test_betas(num_timesteps=10):
    """Create valid beta schedule for testing."""
    return np.linspace(0.0001, 0.02, num_timesteps)


class MockModel(th.nn.Module):
    """Mock diffusion model that predicts epsilon (noise)."""

    def __init__(self, channels=3):
        super().__init__()
        self.channels = channels
        self.dummy = th.nn.Parameter(th.zeros(1))  # For device placement

    def forward(self, x, t, **kwargs):
        # Return zeros - simple predictable behavior
        return th.zeros_like(x)


class MockModelLearned(th.nn.Module):
    """Mock diffusion model with learned variance."""

    def __init__(self, channels=3):
        super().__init__()
        self.channels = channels
        self.dummy = th.nn.Parameter(th.zeros(1))

    def forward(self, x, t, **kwargs):
        # Return noise prediction and variance prediction
        noise = th.zeros_like(x)
        var = th.zeros_like(x)
        return th.cat([noise, var], dim=1)


# =============================================================================
# Pytest Fixtures
# =============================================================================


@pytest.fixture
def diffusion_params():
    """Common diffusion parameters."""
    num_timesteps = 10  # Small for fast tests
    betas = get_test_betas(num_timesteps)
    return {
        "betas": betas,
        "model_mean_type": ModelMeanType.EPSILON,
        "model_var_type": ModelVarType.FIXED_SMALL,
        "loss_type": LossType.MSE,
    }


@pytest.fixture
def diffusion(diffusion_params):
    """Create a GaussianDiffusionDDIMODE instance."""
    return GaussianDiffusionDDIMODE(**diffusion_params)


@pytest.fixture
def model():
    """Create a mock model."""
    return MockModel()


@pytest.fixture
def simple_phi_fn():
    """Terminal cost: L2 distance to zeros."""

    def phi_fn(x, **kwargs):
        # Return per-sample cost [B]
        return (x**2).flatten(1).mean(1)

    return phi_fn


# =============================================================================
# Test Classes
# =============================================================================


class TestGaussianDiffusionDDIMODEInit:
    """Test initialization and class structure."""

    def test_inherits_from_gaussian_diffusion(self, diffusion):
        """Verify GaussianDiffusionDDIMODE inherits from GaussianDiffusion."""
        assert isinstance(diffusion, GaussianDiffusion)

    def test_has_emsa_solve_method(self, diffusion):
        """Verify emsa_solve method exists."""
        assert hasattr(diffusion, "emsa_solve")
        assert callable(diffusion.emsa_solve)

    def test_has_zdot_method(self, diffusion):
        """Verify zdot method exists."""
        assert hasattr(diffusion, "zdot")
        assert callable(diffusion.zdot)

    def test_has_euler_solve_z_method(self, diffusion):
        """Verify euler_solve_z method exists."""
        assert hasattr(diffusion, "euler_solve_z")
        assert callable(diffusion.euler_solve_z)

    def test_has_dynamics_adapter_class(self, diffusion):
        """Verify _DiffusionDynamicsAdapter inner class exists."""
        assert hasattr(diffusion, "_DiffusionDynamicsAdapter")

    def test_num_timesteps(self, diffusion):
        """Verify num_timesteps is set correctly."""
        assert diffusion.num_timesteps == 10


class TestZdot:
    """Test zdot computation."""

    def test_zdot_returns_correct_shape(self, diffusion, model):
        """Verify zdot returns tensor with same shape as input."""
        B, C, H, W = 2, 3, 8, 8
        z = th.randn(B, C, H, W)
        t = th.tensor([5, 5])  # Timestep indices

        epsilon = diffusion.zdot(model, z, t)

        assert epsilon.shape == z.shape

    def test_zdot_with_clip_denoised_true(self, diffusion, model):
        """Verify zdot works with clip_denoised=True."""
        z = th.randn(2, 3, 8, 8)
        t = th.tensor([5, 5])

        epsilon = diffusion.zdot(model, z, t, clip_denoised=True)

        assert epsilon.shape == z.shape

    def test_zdot_with_clip_denoised_false(self, diffusion, model):
        """Verify zdot works with clip_denoised=False."""
        z = th.randn(2, 3, 8, 8)
        t = th.tensor([5, 5])

        epsilon = diffusion.zdot(model, z, t, clip_denoised=False)

        assert epsilon.shape == z.shape

    def test_zdot_with_model_kwargs(self, diffusion, model):
        """Verify zdot passes model_kwargs correctly."""
        z = th.randn(2, 3, 8, 8)
        t = th.tensor([5, 5])

        epsilon = diffusion.zdot(
            model, z, t, model_kwargs={"extra_arg": "test"}
        )

        assert epsilon.shape == z.shape


class TestDiffusionDynamicsAdapter:
    """Test the _DiffusionDynamicsAdapter inner class."""

    def test_adapter_xdot_returns_correct_shape(self, diffusion, model):
        """Verify xdot returns correct shape."""
        B, C, H, W = 2, 3, 8, 8
        N = diffusion.num_timesteps
        timestep_indices = list(range(N))[::-1]

        adapter = diffusion._DiffusionDynamicsAdapter(
            pipeline=diffusion,
            model=model,
            timestep_indices=timestep_indices,
        )

        x = th.randn(B, C, H, W)
        u = th.zeros(B, C, H, W)
        t = th.tensor(timestep_indices[0])  # First DDIM index

        result = adapter.xdot(x, u, t)

        assert result.shape == x.shape

    def test_adapter_xdot_zero_control_equals_zdot(self, diffusion, model):
        """Verify xdot with u=0 returns same as zdot."""
        B, C, H, W = 2, 3, 8, 8
        N = diffusion.num_timesteps
        timestep_indices = list(range(N))[::-1]
        ddim_idx = timestep_indices[0]

        adapter = diffusion._DiffusionDynamicsAdapter(
            pipeline=diffusion,
            model=model,
            timestep_indices=timestep_indices,
        )

        x = th.randn(B, C, H, W)
        u = th.zeros(B, C, H, W)
        t = th.tensor(ddim_idx)

        # Get xdot with zero control
        xdot_result = adapter.xdot(x, u, t)

        # Get zdot directly
        t_tensor = th.tensor([ddim_idx] * B)
        zdot_result = diffusion.zdot(model, x, t_tensor)

        assert th.allclose(xdot_result, zdot_result, atol=1e-6)

    def test_adapter_xdot_additive_control(self, diffusion, model):
        """Verify control is added to drift."""
        B, C, H, W = 2, 3, 8, 8
        N = diffusion.num_timesteps
        timestep_indices = list(range(N))[::-1]

        adapter = diffusion._DiffusionDynamicsAdapter(
            pipeline=diffusion,
            model=model,
            timestep_indices=timestep_indices,
        )

        x = th.randn(B, C, H, W)
        u = th.ones(B, C, H, W) * 0.5  # Non-zero control
        t = th.tensor(timestep_indices[0])

        # Get result with control
        result_with_u = adapter.xdot(x, u, t)

        # Get result without control
        u_zero = th.zeros_like(u)
        result_without_u = adapter.xdot(x, u_zero, t)

        # Difference should be u
        diff = result_with_u - result_without_u
        assert th.allclose(diff, u, atol=1e-6)

    def test_adapter_juT_p_identity(self, diffusion, model):
        """Verify juT_p returns p unchanged (identity for additive control)."""
        N = diffusion.num_timesteps
        timestep_indices = list(range(N))[::-1]

        adapter = diffusion._DiffusionDynamicsAdapter(
            pipeline=diffusion,
            model=model,
            timestep_indices=timestep_indices,
        )

        p = th.randn(2, 3, 8, 8)
        t = th.tensor(5)

        result = adapter.juT_p(p, t)

        assert th.allclose(result, p)


class TestEmsaSolve:
    """Test emsa_solve method."""

    def test_returns_dict_with_expected_keys(self, diffusion, model, simple_phi_fn):
        """Verify emsa_solve returns dict with all expected keys."""
        result = diffusion.emsa_solve(
            model=model,
            shape=(2, 3, 8, 8),
            phi_fn=simple_phi_fn,
            iters=2,
            device=th.device("cpu"),
            verbose=False,
        )

        expected_keys = {"sample", "z", "U", "X", "cost_history", "time_bench"}
        assert set(result.keys()) == expected_keys

    def test_output_sample_shape(self, diffusion, model, simple_phi_fn):
        """Verify sample has correct shape [B, C, H, W]."""
        B, C, H, W = 2, 3, 8, 8

        result = diffusion.emsa_solve(
            model=model,
            shape=(B, C, H, W),
            phi_fn=simple_phi_fn,
            iters=2,
            device=th.device("cpu"),
            verbose=False,
        )

        assert result["sample"].shape == (B, C, H, W)

    def test_output_z_shape(self, diffusion, model, simple_phi_fn):
        """Verify z has correct shape [B, C, H, W]."""
        B, C, H, W = 2, 3, 8, 8

        result = diffusion.emsa_solve(
            model=model,
            shape=(B, C, H, W),
            phi_fn=simple_phi_fn,
            iters=2,
            device=th.device("cpu"),
            verbose=False,
        )

        assert result["z"].shape == (B, C, H, W)

    def test_output_U_shape(self, diffusion, model, simple_phi_fn):
        """Verify U has correct shape [B, N, C, H, W]."""
        B, C, H, W = 2, 3, 8, 8
        N = diffusion.num_timesteps

        result = diffusion.emsa_solve(
            model=model,
            shape=(B, C, H, W),
            phi_fn=simple_phi_fn,
            iters=2,
            device=th.device("cpu"),
            verbose=False,
        )

        assert result["U"].shape == (B, N, C, H, W)

    def test_output_X_shape(self, diffusion, model, simple_phi_fn):
        """Verify X has correct shape [B, N+1, C, H, W]."""
        B, C, H, W = 2, 3, 8, 8
        N = diffusion.num_timesteps

        result = diffusion.emsa_solve(
            model=model,
            shape=(B, C, H, W),
            phi_fn=simple_phi_fn,
            iters=2,
            device=th.device("cpu"),
            verbose=False,
        )

        assert result["X"].shape == (B, N + 1, C, H, W)

    def test_deterministic_with_generator(self, diffusion, model, simple_phi_fn):
        """Verify reproducibility with same generator seed."""
        shape = (2, 3, 8, 8)

        gen1 = th.Generator().manual_seed(42)
        result1 = diffusion.emsa_solve(
            model=model,
            shape=shape,
            phi_fn=simple_phi_fn,
            iters=2,
            device=th.device("cpu"),
            generator=gen1,
            verbose=False,
        )

        gen2 = th.Generator().manual_seed(42)
        result2 = diffusion.emsa_solve(
            model=model,
            shape=shape,
            phi_fn=simple_phi_fn,
            iters=2,
            device=th.device("cpu"),
            generator=gen2,
            verbose=False,
        )

        assert th.allclose(result1["sample"], result2["sample"])
        assert th.allclose(result1["U"], result2["U"])

    def test_cost_history_exists(self, diffusion, model, simple_phi_fn):
        """Verify cost_history is returned and has entries."""
        result = diffusion.emsa_solve(
            model=model,
            shape=(2, 3, 8, 8),
            phi_fn=simple_phi_fn,
            iters=3,
            device=th.device("cpu"),
            verbose=False,
        )

        # cost_history can be a list or numpy array
        cost_history = result["cost_history"]
        assert cost_history is not None
        assert len(cost_history) > 0

    def test_chunked_execution(self, diffusion, model, simple_phi_fn):
        """Verify chunked execution produces valid output."""
        B, C, H, W = 4, 3, 8, 8

        result = diffusion.emsa_solve(
            model=model,
            shape=(B, C, H, W),
            phi_fn=simple_phi_fn,
            iters=2,
            chunk=True,
            minibatch=2,
            device=th.device("cpu"),
            verbose=False,
        )

        assert result["sample"].shape == (B, C, H, W)

    def test_different_rho_u_values(self, diffusion, model, simple_phi_fn):
        """Verify different rho_u values produce different results."""
        shape = (2, 3, 8, 8)
        gen1 = th.Generator().manual_seed(42)
        gen2 = th.Generator().manual_seed(42)

        result_low_rho = diffusion.emsa_solve(
            model=model,
            shape=shape,
            phi_fn=simple_phi_fn,
            rho_u=0.1,
            iters=3,
            device=th.device("cpu"),
            generator=gen1,
            verbose=False,
        )

        result_high_rho = diffusion.emsa_solve(
            model=model,
            shape=shape,
            phi_fn=simple_phi_fn,
            rho_u=10.0,
            iters=3,
            device=th.device("cpu"),
            generator=gen2,
            verbose=False,
        )

        # Higher rho_u penalizes control more, so U should be smaller
        u_norm_low = result_low_rho["U"].norm()
        u_norm_high = result_high_rho["U"].norm()
        assert u_norm_high <= u_norm_low

    def test_verbose_output(self, diffusion, model, simple_phi_fn, capsys):
        """Verify verbose=True produces output."""
        diffusion.emsa_solve(
            model=model,
            shape=(2, 3, 8, 8),
            phi_fn=simple_phi_fn,
            iters=2,
            device=th.device("cpu"),
            verbose=True,
        )

        captured = capsys.readouterr()
        assert "EMSA" in captured.out


class TestEmsaSolveEdgeCases:
    """Test edge cases for emsa_solve."""

    def test_single_batch(self, diffusion, model, simple_phi_fn):
        """Verify emsa_solve works with batch size 1."""
        result = diffusion.emsa_solve(
            model=model,
            shape=(1, 3, 8, 8),
            phi_fn=simple_phi_fn,
            iters=2,
            device=th.device("cpu"),
            verbose=False,
        )

        assert result["sample"].shape == (1, 3, 8, 8)

    def test_small_image_size(self, diffusion, model, simple_phi_fn):
        """Verify emsa_solve works with small 4x4 images."""
        result = diffusion.emsa_solve(
            model=model,
            shape=(2, 3, 4, 4),
            phi_fn=simple_phi_fn,
            iters=2,
            device=th.device("cpu"),
            verbose=False,
        )

        assert result["sample"].shape == (2, 3, 4, 4)

    def test_single_iteration(self, diffusion, model, simple_phi_fn):
        """Verify emsa_solve works with single iteration."""
        result = diffusion.emsa_solve(
            model=model,
            shape=(2, 3, 8, 8),
            phi_fn=simple_phi_fn,
            iters=1,
            device=th.device("cpu"),
            verbose=False,
        )

        assert result["sample"].shape == (2, 3, 8, 8)

    def test_single_channel(self, diffusion_params):
        """Verify emsa_solve works with single channel images."""
        diffusion = GaussianDiffusionDDIMODE(**diffusion_params)
        model = MockModel(channels=1)

        def phi_fn(x, **kwargs):
            return (x**2).flatten(1).mean(1)

        result = diffusion.emsa_solve(
            model=model,
            shape=(2, 1, 8, 8),
            phi_fn=phi_fn,
            iters=2,
            device=th.device("cpu"),
            verbose=False,
        )

        assert result["sample"].shape == (2, 1, 8, 8)

    def test_no_nan_in_output(self, diffusion, model, simple_phi_fn):
        """Verify output contains no NaN values."""
        result = diffusion.emsa_solve(
            model=model,
            shape=(2, 3, 8, 8),
            phi_fn=simple_phi_fn,
            iters=3,
            device=th.device("cpu"),
            verbose=False,
        )

        assert not th.isnan(result["sample"]).any()
        assert not th.isnan(result["z"]).any()
        assert not th.isnan(result["U"]).any()
        assert not th.isnan(result["X"]).any()
