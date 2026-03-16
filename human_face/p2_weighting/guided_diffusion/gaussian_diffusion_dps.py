"""
Diffusion Posterior Sampling (DPS) implementation.

Extends GaussianDiffusion to support DPS-based sampling for solving inverse problems.
The sampling methods take an additional loss function that guides the diffusion process
towards samples that are consistent with measurements.
"""

import torch as th

from .gaussian_diffusion import GaussianDiffusion, _extract_into_tensor


class GaussianDiffusionDPS(GaussianDiffusion):
    """
    GaussianDiffusion with Diffusion Posterior Sampling (DPS) support.

    DPS modifies the reverse diffusion sampling to incorporate measurement constraints.
    At each sampling step:
    1. Compute predicted x_0 from noisy x_t
    2. Compute gradient of the measurement loss w.r.t. x_t
    3. Apply gradient correction to the score (epsilon), following Song et al. (2020):
       eps_corrected = eps - scale * sqrt(1 - alpha_bar) * grad
    4. Recompute x_0 from corrected epsilon and proceed with DDIM step
    """

    def ddim_sample_dps(
        self,
        model,
        x,
        t,
        loss_fn,
        scale=1.0,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
        generator=None,
    ):
        """
        Sample x_{t-1} from the model using DDIM with DPS guidance.

        :param model: the model to sample from.
        :param x: the current tensor at x_t.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param loss_fn: a callable that takes pred_x0 and returns a loss tensor
                        of shape [B,]. The loss function should internally handle
                        the forward model and measurement.
        :param scale: gradient scale factor for DPS correction.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model.
        :param eta: the eta parameter for DDIM (0 = deterministic).
        :param generator: optional random generator for reproducibility.
        :return: a dict containing 'sample' and 'pred_xstart'.
        """
        if model_kwargs is None:
            model_kwargs = {}

        # Step 1: Enable gradients and get model prediction
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            out = self.p_mean_variance(
                model,
                x_in,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
            )

            # Step 2: Compute loss and gradient
            loss = loss_fn(out["pred_xstart"]).sum()
            grad = th.autograd.grad(loss, x_in)[0]

        # Step 3: Extract original epsilon and apply DPS correction to score
        # Following Song et al. (2020) score conditioning: eps = eps - sqrt(1-alpha_bar) * grad
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
        sqrt_one_minus_alpha_bar = _extract_into_tensor(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        eps = eps - scale * sqrt_one_minus_alpha_bar * grad

        # Step 4: Recompute pred_xstart from corrected epsilon
        pred_xstart = self._predict_xstart_from_eps(x, t, eps)
        if clip_denoised:
            pred_xstart = pred_xstart.clamp(-1, 1)
        if denoised_fn is not None:
            pred_xstart = denoised_fn(pred_xstart)

        # Step 5: Standard DDIM step using original x and corrected eps/pred_xstart
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        if abs(eta) < 1e-5:
            sigma = 0.0
        else:
            sigma = (
                eta
                * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                * th.sqrt(1 - alpha_bar / alpha_bar_prev)
            )

        if generator is not None:
            noise = generator.randn_like(x)
        else:
            noise = th.randn_like(x)

        mean_pred = (
            pred_xstart * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        sample = mean_pred + nonzero_mask * sigma * noise

        return {"sample": sample, "pred_xstart": pred_xstart}

    def ddim_sample_loop_dps(
        self,
        model,
        shape,
        loss_fn,
        scale=1.0,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        generator=None,
    ):
        """
        Generate samples from the model using DDIM with DPS guidance.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param loss_fn: a callable that takes pred_x0 and returns a loss tensor
                        of shape [B,].
        :param scale: gradient scale factor for DPS correction.
        :param noise: if specified, the noise from the encoder to sample.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model.
        :param device: if specified, the device to create the samples on.
        :param progress: if True, show a tqdm progress bar.
        :param eta: the eta parameter for DDIM.
        :param generator: optional random generator for reproducibility.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.ddim_sample_loop_progressive_dps(
            model,
            shape,
            loss_fn=loss_fn,
            scale=scale,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
            generator=generator,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive_dps(
        self,
        model,
        shape,
        loss_fn,
        scale=1.0,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        generator=None,
    ):
        """
        Use DDIM with DPS to sample from the model and yield intermediate samples.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param loss_fn: a callable that takes pred_x0 and returns a loss tensor
                        of shape [B,].
        :param scale: gradient scale factor for DPS correction.
        :param noise: if specified, the noise from the encoder to sample.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model.
        :param device: if specified, the device to create the samples on.
        :param progress: if True, show a tqdm progress bar.
        :param eta: the eta parameter for DDIM.
        :param generator: optional random generator for reproducibility.
        :return: a generator over dicts with 'sample' and 'pred_xstart'.
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))

        if noise is not None:
            img = noise
        elif generator is not None:
            img = generator.randn(shape, device=device)
        else:
            img = th.randn(*shape, device=device)

        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.ddim_sample_dps(
                    model,
                    img,
                    t,
                    loss_fn=loss_fn,
                    scale=scale,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                    generator=generator,
                )
                yield out
                img = out["sample"]
