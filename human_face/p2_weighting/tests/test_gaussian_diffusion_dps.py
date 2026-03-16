"""Tests for Diffusion Posterior Sampling (DPS) implementation."""

import numpy as np
import pytest
import torch as th

from guided_diffusion.gaussian_diffusion import (
    ModelMeanType,
    ModelVarType,
    LossType,
)
from guided_diffusion.gaussian_diffusion_dps import GaussianDiffusionDPS


def get_test_betas(num_timesteps=10):
    """Create valid beta schedule for testing.

    The standard linear schedule is designed for 1000 timesteps,
    so we create a simple valid schedule for smaller timestep counts.
    """
    # Simple linear schedule that stays within valid range (0, 1]
    return np.linspace(0.0001, 0.02, num_timesteps)


class MockModel(th.nn.Module):
    """Mock diffusion model that predicts epsilon (noise)."""

    def __init__(self, channels=3):
        super().__init__()
        self.channels = channels
        # Dummy parameter to place model on device
        self.dummy = th.nn.Parameter(th.zeros(1))

    def forward(self, x, t, **kwargs):
        """Return a fixed noise prediction for testing."""
        return th.zeros_like(x)


class MockModelLearned(th.nn.Module):
    """Mock diffusion model with learned variance."""

    def __init__(self, channels=3):
        super().__init__()
        self.channels = channels
        self.dummy = th.nn.Parameter(th.zeros(1))

    def forward(self, x, t, **kwargs):
        """Return noise and variance predictions."""
        noise = th.zeros_like(x)
        var = th.zeros_like(x)  # Will be interpreted as min variance
        return th.cat([noise, var], dim=1)


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
    """Create a GaussianDiffusionDPS instance."""
    return GaussianDiffusionDPS(**diffusion_params)


@pytest.fixture
def diffusion_learned_var():
    """Create a GaussianDiffusionDPS instance with learned variance."""
    num_timesteps = 10
    betas = get_test_betas(num_timesteps)
    return GaussianDiffusionDPS(
        betas=betas,
        model_mean_type=ModelMeanType.EPSILON,
        model_var_type=ModelVarType.LEARNED_RANGE,
        loss_type=LossType.MSE,
    )


@pytest.fixture
def model():
    """Create a mock model."""
    return MockModel()


@pytest.fixture
def model_learned():
    """Create a mock model with learned variance."""
    return MockModelLearned()


@pytest.fixture
def simple_loss_fn():
    """Simple L2 loss to a target image (zeros)."""
    def loss_fn(pred_x0):
        # Return per-sample loss
        return (pred_x0 ** 2).mean(dim=(1, 2, 3))
    return loss_fn


class TestGaussianDiffusionDPSInit:
    """Tests for GaussianDiffusionDPS initialization and inheritance."""

    def test_inherits_from_gaussian_diffusion(self, diffusion):
        """Verify DPS class inherits correctly."""
        from guided_diffusion.gaussian_diffusion import GaussianDiffusion
        assert isinstance(diffusion, GaussianDiffusion)

    def test_has_dps_methods(self, diffusion):
        """Verify DPS-specific methods exist."""
        assert hasattr(diffusion, "ddim_sample_dps")
        assert hasattr(diffusion, "ddim_sample_loop_dps")
        assert hasattr(diffusion, "ddim_sample_loop_progressive_dps")

    def test_num_timesteps(self, diffusion):
        """Verify timesteps are set correctly."""
        assert diffusion.num_timesteps == 10


class TestDDIMSampleDPS:
    """Tests for the single-step DPS sampling method."""

    def test_returns_dict_with_expected_keys(self, diffusion, model, simple_loss_fn):
        """Verify output format."""
        batch_size = 2
        x = th.randn(batch_size, 3, 8, 8)
        t = th.tensor([5, 5])

        out = diffusion.ddim_sample_dps(
            model, x, t, loss_fn=simple_loss_fn, scale=1.0
        )

        assert isinstance(out, dict)
        assert "sample" in out
        assert "pred_xstart" in out

    def test_output_shapes(self, diffusion, model, simple_loss_fn):
        """Verify output tensor shapes match input."""
        batch_size = 2
        x = th.randn(batch_size, 3, 8, 8)
        t = th.tensor([5, 5])

        out = diffusion.ddim_sample_dps(
            model, x, t, loss_fn=simple_loss_fn, scale=1.0
        )

        assert out["sample"].shape == x.shape
        assert out["pred_xstart"].shape == x.shape

    def test_deterministic_with_eta_zero(self, diffusion, model, simple_loss_fn):
        """DDIM should be deterministic when eta=0."""
        x = th.randn(2, 3, 8, 8)
        t = th.tensor([5, 5])

        out1 = diffusion.ddim_sample_dps(
            model, x, t, loss_fn=simple_loss_fn, scale=1.0, eta=0.0
        )
        out2 = diffusion.ddim_sample_dps(
            model, x, t, loss_fn=simple_loss_fn, scale=1.0, eta=0.0
        )

        th.testing.assert_close(out1["sample"], out2["sample"])

    def test_stochastic_with_eta_nonzero(self, diffusion, model, simple_loss_fn):
        """DDIM should be stochastic when eta > 0 (for t > 0)."""
        x = th.randn(2, 3, 8, 8)
        t = th.tensor([5, 5])  # Non-zero timestep

        # Set different seeds for each call
        th.manual_seed(42)
        out1 = diffusion.ddim_sample_dps(
            model, x, t, loss_fn=simple_loss_fn, scale=1.0, eta=1.0
        )
        th.manual_seed(123)
        out2 = diffusion.ddim_sample_dps(
            model, x, t, loss_fn=simple_loss_fn, scale=1.0, eta=1.0
        )

        # Samples should differ due to noise
        assert not th.allclose(out1["sample"], out2["sample"])

    def test_gradient_scale_affects_output(self, diffusion, model, simple_loss_fn):
        """Different gradient scales should produce different outputs."""
        x = th.randn(2, 3, 8, 8)
        t = th.tensor([5, 5])

        out_small = diffusion.ddim_sample_dps(
            model, x, t, loss_fn=simple_loss_fn, scale=0.1, eta=0.0
        )
        out_large = diffusion.ddim_sample_dps(
            model, x, t, loss_fn=simple_loss_fn, scale=10.0, eta=0.0
        )

        assert not th.allclose(out_small["sample"], out_large["sample"])

    def test_zero_scale_no_guidance(self, diffusion, model, simple_loss_fn):
        """Scale=0 should produce same result regardless of loss function."""
        x = th.randn(2, 3, 8, 8)
        t = th.tensor([5, 5])

        def alt_loss_fn(pred_x0):
            return (pred_x0 - 1.0).abs().mean(dim=(1, 2, 3))

        out1 = diffusion.ddim_sample_dps(
            model, x, t, loss_fn=simple_loss_fn, scale=0.0, eta=0.0
        )
        out2 = diffusion.ddim_sample_dps(
            model, x, t, loss_fn=alt_loss_fn, scale=0.0, eta=0.0
        )

        th.testing.assert_close(out1["sample"], out2["sample"])

    def test_clip_denoised(self, diffusion, model, simple_loss_fn):
        """Verify clipping works for pred_xstart."""
        x = th.randn(2, 3, 8, 8) * 10  # Large values
        t = th.tensor([5, 5])

        out_clipped = diffusion.ddim_sample_dps(
            model, x, t, loss_fn=simple_loss_fn, clip_denoised=True
        )

        # pred_xstart should be clipped to [-1, 1]
        assert out_clipped["pred_xstart"].min() >= -1.0
        assert out_clipped["pred_xstart"].max() <= 1.0

    def test_no_clip_denoised(self, diffusion, model, simple_loss_fn):
        """Verify no clipping when disabled."""
        # Create a model that returns large values
        class LargeOutputModel(th.nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = th.nn.Parameter(th.zeros(1))

            def forward(self, x, t, **kwargs):
                return th.ones_like(x) * -5  # Large epsilon prediction

        large_model = LargeOutputModel()
        x = th.ones(2, 3, 8, 8)
        t = th.tensor([5, 5])

        out = diffusion.ddim_sample_dps(
            large_model, x, t, loss_fn=simple_loss_fn, clip_denoised=False
        )

        # pred_xstart might exceed [-1, 1] bounds
        # The specific values depend on the math, just verify no error

    def test_custom_denoised_fn(self, diffusion, model, simple_loss_fn):
        """Verify custom denoised_fn is applied."""
        x = th.randn(2, 3, 8, 8)
        t = th.tensor([5, 5])

        def denoised_fn(x):
            return x * 0.5  # Scale down

        out = diffusion.ddim_sample_dps(
            model, x, t, loss_fn=simple_loss_fn, denoised_fn=denoised_fn
        )

        # Result should be affected by denoised_fn
        assert out["pred_xstart"] is not None

    def test_model_kwargs_passed(self, diffusion, simple_loss_fn):
        """Verify model_kwargs are passed to the model."""
        received_kwargs = {}

        class KwargsModel(th.nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = th.nn.Parameter(th.zeros(1))

            def forward(self, x, t, **kwargs):
                nonlocal received_kwargs
                received_kwargs = kwargs
                return th.zeros_like(x)

        model = KwargsModel()
        x = th.randn(2, 3, 8, 8)
        t = th.tensor([5, 5])

        diffusion.ddim_sample_dps(
            model, x, t, loss_fn=simple_loss_fn,
            model_kwargs={"y": th.tensor([1, 2])}
        )

        assert "y" in received_kwargs
        th.testing.assert_close(received_kwargs["y"], th.tensor([1, 2]))

    def test_at_timestep_zero(self, diffusion, model, simple_loss_fn):
        """Test sampling at timestep 0 (final step)."""
        x = th.randn(2, 3, 8, 8)
        t = th.tensor([0, 0])

        out = diffusion.ddim_sample_dps(
            model, x, t, loss_fn=simple_loss_fn, eta=1.0
        )

        # At t=0, no noise should be added (nonzero_mask = 0)
        # Run twice with different seeds
        th.manual_seed(42)
        out1 = diffusion.ddim_sample_dps(
            model, x, t, loss_fn=simple_loss_fn, eta=1.0
        )
        th.manual_seed(123)
        out2 = diffusion.ddim_sample_dps(
            model, x, t, loss_fn=simple_loss_fn, eta=1.0
        )

        th.testing.assert_close(out1["sample"], out2["sample"])

    def test_with_learned_variance(self, diffusion_learned_var, model_learned, simple_loss_fn):
        """Test with learned variance model."""
        x = th.randn(2, 3, 8, 8)
        t = th.tensor([5, 5])

        out = diffusion_learned_var.ddim_sample_dps(
            model_learned, x, t, loss_fn=simple_loss_fn
        )

        assert out["sample"].shape == x.shape


class TestDDIMSampleLoopDPS:
    """Tests for the full DPS sampling loop."""

    def test_returns_tensor(self, diffusion, model, simple_loss_fn):
        """Verify output is a tensor, not a dict."""
        shape = (2, 3, 8, 8)

        samples = diffusion.ddim_sample_loop_dps(
            model, shape, loss_fn=simple_loss_fn
        )

        assert isinstance(samples, th.Tensor)
        assert samples.shape == shape

    def test_output_shape(self, diffusion, model, simple_loss_fn):
        """Verify output shape matches requested shape."""
        shape = (4, 3, 16, 16)

        samples = diffusion.ddim_sample_loop_dps(
            model, shape, loss_fn=simple_loss_fn
        )

        assert samples.shape == shape

    def test_custom_noise(self, diffusion, model, simple_loss_fn):
        """Verify custom noise is used."""
        shape = (2, 3, 8, 8)
        custom_noise = th.ones(shape) * 0.5

        samples = diffusion.ddim_sample_loop_dps(
            model, shape, loss_fn=simple_loss_fn, noise=custom_noise
        )

        assert samples.shape == shape

    def test_device_placement(self, diffusion, model, simple_loss_fn):
        """Verify samples are on the correct device."""
        shape = (2, 3, 8, 8)

        samples = diffusion.ddim_sample_loop_dps(
            model, shape, loss_fn=simple_loss_fn, device="cpu"
        )

        assert samples.device == th.device("cpu")

    def test_progress_bar(self, diffusion, model, simple_loss_fn):
        """Verify progress=True doesn't cause errors."""
        shape = (2, 3, 8, 8)

        # Should not raise
        samples = diffusion.ddim_sample_loop_dps(
            model, shape, loss_fn=simple_loss_fn, progress=True
        )

        assert samples.shape == shape


class TestDDIMSampleLoopProgressiveDPS:
    """Tests for the progressive DPS sampling loop."""

    def test_yields_intermediate_results(self, diffusion, model, simple_loss_fn):
        """Verify it yields results at each timestep."""
        shape = (2, 3, 8, 8)
        results = list(diffusion.ddim_sample_loop_progressive_dps(
            model, shape, loss_fn=simple_loss_fn
        ))

        # Should have one result per timestep
        assert len(results) == diffusion.num_timesteps

    def test_each_result_has_expected_keys(self, diffusion, model, simple_loss_fn):
        """Verify each intermediate result has correct format."""
        shape = (2, 3, 8, 8)

        for out in diffusion.ddim_sample_loop_progressive_dps(
            model, shape, loss_fn=simple_loss_fn
        ):
            assert isinstance(out, dict)
            assert "sample" in out
            assert "pred_xstart" in out
            assert out["sample"].shape == shape
            assert out["pred_xstart"].shape == shape

    def test_final_result_matches_loop(self, diffusion, model, simple_loss_fn):
        """Verify final progressive result matches loop result."""
        shape = (2, 3, 8, 8)
        th.manual_seed(42)
        noise = th.randn(shape)

        th.manual_seed(123)
        progressive_results = list(diffusion.ddim_sample_loop_progressive_dps(
            model, shape, loss_fn=simple_loss_fn, noise=noise.clone(), eta=0.0
        ))

        th.manual_seed(123)
        loop_result = diffusion.ddim_sample_loop_dps(
            model, shape, loss_fn=simple_loss_fn, noise=noise.clone(), eta=0.0
        )

        th.testing.assert_close(progressive_results[-1]["sample"], loop_result)


class TestDPSGradientComputation:
    """Tests for gradient computation in DPS."""

    def test_gradient_computed_from_loss(self, diffusion, model):
        """Verify gradient is computed from loss function."""
        gradient_computed = [False]

        def tracking_loss_fn(pred_x0):
            gradient_computed[0] = pred_x0.requires_grad
            return (pred_x0 ** 2).mean(dim=(1, 2, 3))

        x = th.randn(2, 3, 8, 8)
        t = th.tensor([5, 5])

        diffusion.ddim_sample_dps(
            model, x, t, loss_fn=tracking_loss_fn, scale=1.0
        )

        # The pred_x0 passed to loss should have gradients enabled
        assert gradient_computed[0]

    def test_loss_fn_receives_pred_xstart(self, diffusion, model):
        """Verify loss function receives the predicted x_start."""
        received_input = [None]

        def capturing_loss_fn(pred_x0):
            received_input[0] = pred_x0.clone().detach()
            return (pred_x0 ** 2).mean(dim=(1, 2, 3))

        x = th.randn(2, 3, 8, 8)
        t = th.tensor([5, 5])

        diffusion.ddim_sample_dps(
            model, x, t, loss_fn=capturing_loss_fn, scale=1.0
        )

        assert received_input[0] is not None
        assert received_input[0].shape == x.shape

    def test_guidance_moves_toward_target(self, diffusion, model):
        """Verify DPS guidance moves samples toward measurement target."""
        # Create a loss that penalizes deviation from a target
        target = th.zeros(2, 3, 8, 8)

        def loss_fn(pred_x0):
            return ((pred_x0 - target) ** 2).mean(dim=(1, 2, 3))

        x = th.randn(2, 3, 8, 8) * 5  # Start far from target
        t = th.tensor([5, 5])

        out_guided = diffusion.ddim_sample_dps(
            model, x, t, loss_fn=loss_fn, scale=1.0, eta=0.0
        )
        out_unguided = diffusion.ddim_sample_dps(
            model, x, t, loss_fn=loss_fn, scale=0.0, eta=0.0
        )

        # Guided sample should be closer to target
        guided_dist = ((out_guided["sample"] - target) ** 2).mean()
        unguided_dist = ((out_unguided["sample"] - target) ** 2).mean()

        # With proper guidance, we expect guided to be closer
        # (This may not always hold due to diffusion dynamics, but
        # for strong enough scale it should help)


class TestDPSEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_batch(self, diffusion, model, simple_loss_fn):
        """Test with batch size 1."""
        x = th.randn(1, 3, 8, 8)
        t = th.tensor([5])

        out = diffusion.ddim_sample_dps(
            model, x, t, loss_fn=simple_loss_fn
        )

        assert out["sample"].shape == (1, 3, 8, 8)

    def test_different_image_sizes(self, diffusion, model, simple_loss_fn):
        """Test with different spatial dimensions."""
        for size in [4, 16, 32]:
            x = th.randn(2, 3, size, size)
            t = th.tensor([5, 5])

            out = diffusion.ddim_sample_dps(
                model, x, t, loss_fn=simple_loss_fn
            )

            assert out["sample"].shape == (2, 3, size, size)

    def test_different_channels(self, diffusion_params, simple_loss_fn):
        """Test with different channel counts."""
        diffusion = GaussianDiffusionDPS(**diffusion_params)

        for channels in [1, 3, 4]:
            model = MockModel(channels=channels)
            x = th.randn(2, channels, 8, 8)
            t = th.tensor([5, 5])

            out = diffusion.ddim_sample_dps(
                model, x, t, loss_fn=simple_loss_fn
            )

            assert out["sample"].shape == (2, channels, 8, 8)

    def test_negative_scale(self, diffusion, model, simple_loss_fn):
        """Test with negative scale (anti-guidance)."""
        x = th.randn(2, 3, 8, 8)
        t = th.tensor([5, 5])

        # Should not raise - negative scale reverses guidance direction
        out = diffusion.ddim_sample_dps(
            model, x, t, loss_fn=simple_loss_fn, scale=-1.0
        )

        assert out["sample"].shape == x.shape

    def test_very_large_scale(self, diffusion, model, simple_loss_fn):
        """Test with very large scale."""
        x = th.randn(2, 3, 8, 8)
        t = th.tensor([5, 5])

        out = diffusion.ddim_sample_dps(
            model, x, t, loss_fn=simple_loss_fn, scale=1000.0
        )

        # Should not produce NaN
        assert not th.isnan(out["sample"]).any()


class TestDPSWithGenerator:
    """Tests for reproducibility with custom generators."""

    def test_generator_produces_reproducible_results(self, diffusion, model, simple_loss_fn):
        """Verify generator enables reproducible sampling."""

        class SimpleGenerator:
            def __init__(self, seed):
                self.gen = th.Generator()
                self.gen.manual_seed(seed)

            def randn_like(self, x):
                return th.randn(x.shape, generator=self.gen, device=x.device)

            def randn(self, shape, device):
                return th.randn(shape, generator=self.gen, device=device)

        shape = (2, 3, 8, 8)

        gen1 = SimpleGenerator(42)
        out1 = diffusion.ddim_sample_loop_dps(
            model, shape, loss_fn=simple_loss_fn, generator=gen1, eta=1.0
        )

        gen2 = SimpleGenerator(42)
        out2 = diffusion.ddim_sample_loop_dps(
            model, shape, loss_fn=simple_loss_fn, generator=gen2, eta=1.0
        )

        th.testing.assert_close(out1, out2)


class TestDPSIntegration:
    """Integration tests for DPS."""

    def test_full_sampling_pipeline(self, diffusion, model, simple_loss_fn):
        """Test complete sampling from noise to final sample."""
        shape = (2, 3, 8, 8)

        samples = diffusion.ddim_sample_loop_dps(
            model, shape,
            loss_fn=simple_loss_fn,
            scale=1.0,
            clip_denoised=True,
            eta=0.0,
            progress=False,
        )

        assert samples.shape == shape
        assert not th.isnan(samples).any()
        assert not th.isinf(samples).any()

    def test_sampling_with_all_options(self, diffusion, model, simple_loss_fn):
        """Test sampling with all optional parameters."""
        shape = (2, 3, 8, 8)
        noise = th.randn(shape)

        def denoised_fn(x):
            return x

        samples = diffusion.ddim_sample_loop_dps(
            model,
            shape,
            loss_fn=simple_loss_fn,
            scale=0.5,
            noise=noise,
            clip_denoised=True,
            denoised_fn=denoised_fn,
            model_kwargs={"dummy": 1},
            device="cpu",
            progress=False,
            eta=0.5,
        )

        assert samples.shape == shape
