import torch as th
from typing import Callable, Optional

from guided_diffusion.gaussian_diffusion import GaussianDiffusion, ModelMeanType, ModelVarType, _extract_into_tensor
from guided_diffusion.oc_solvers.emsa_quadcost import EMSAQuadCostSolver, ControlledDynamicsProtocol

class GaussianDiffusionDDIMODE(GaussianDiffusion):
    """
    Extension of GaussianDiffusion that implements sampling via an ODE.

    Following DDIM (Song et al. 2020), using cumulative alpha (alpha_bar):
    z_t = x_t / sqrt(alpha_bar_t)
    h_t = sqrt((1 - alpha_bar_t) / alpha_bar_t) = sqrt(1/alpha_bar_t - 1)

    ODE: dz/dh = epsilon_theta(z / sqrt(1 + h^2))
    """

    @staticmethod
    def _process_xstart(x, clip_denoised, denoised_fn):
        """Apply denoised_fn and/or clipping to predicted x_start."""
        if denoised_fn is not None:
            x = denoised_fn(x)
        if clip_denoised:
            return x.clamp(-1, 1)
        return x

    def zdot(self, model, z, t, clip_denoised=True, denoised_fn=None, model_kwargs=None):
        """
        Compute the derivative dz/dh at timestep t.

        :param model: the model to sample from.
        :param z: the current latent state z_t.
        :param t: the timestep tensor.
        :param clip_denoised: if True, clip the predicted x_start to [-1, 1].
        :param denoised_fn: if not None, a function to apply to the predicted x_start.
        :param model_kwargs: if not None, extra kwargs to pass to the model.
        :return: epsilon (the derivative dz/dh).
        """
        if model_kwargs is None:
            model_kwargs = {}

        # 1. Get cumulative alpha_bar_t (precomputed in base class)
        sqrt_alpha_bar = _extract_into_tensor(self.sqrt_alphas_cumprod, t, z.shape)

        # 2. Map latent z back to image space x_t using cumulative alpha
        # z_t = x_t / sqrt(alpha_bar_t) => x_t = z_t * sqrt(alpha_bar_t)
        x_t = z * sqrt_alpha_bar

        # 3. Get model prediction
        # The model still expects scaled timesteps if configured
        model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)

        # 4. Handle learned variance
        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            B, C = x_t.shape[:2]
            model_output, _ = th.split(model_output, C, dim=1)

        # 5. Convert to epsilon, applying clip_denoised/denoised_fn to pred_xstart
        if self.model_mean_type == ModelMeanType.EPSILON:
            # Derive pred_xstart, process it, then re-derive epsilon
            pred_xstart = self._predict_xstart_from_eps(x_t=x_t, t=t, eps=model_output)
            pred_xstart = self._process_xstart(pred_xstart, clip_denoised, denoised_fn)
            epsilon = self._predict_eps_from_xstart(x_t, t, pred_xstart=pred_xstart)
        elif self.model_mean_type == ModelMeanType.START_X:
            pred_xstart = self._process_xstart(model_output, clip_denoised, denoised_fn)
            epsilon = self._predict_eps_from_xstart(x_t, t, pred_xstart=pred_xstart)
        elif self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = self._predict_xstart_from_xprev(x_t=x_t, t=t, xprev=model_output)
            pred_xstart = self._process_xstart(pred_xstart, clip_denoised, denoised_fn)
            epsilon = self._predict_eps_from_xstart(x_t, t, pred_xstart=pred_xstart)
        else:
            raise NotImplementedError(self.model_mean_type)

        return epsilon

    def euler_solve_z(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples using ODE-based Euler method.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function to apply to the x_start prediction.
        :param model_kwargs: if not None, extra kwargs to pass to the model.
        :param device: if specified, the device to create the samples on.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.euler_solve_z_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample
        return final["sample"]
    
    # Helper to get cumulative alpha_bar and h params at specific t
    def _get_params(self, timestep_idx, shape, device):
        t_tensor = th.tensor([timestep_idx] * shape[0], device=device)
        # Use cumulative alpha (alpha_bar), precomputed in base class
        sqrt_alpha_bar = _extract_into_tensor(self.sqrt_alphas_cumprod, t_tensor, shape)
        # h_t = sqrt(1/alpha_bar - 1), also precomputed
        h = _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t_tensor, shape)
        return h, sqrt_alpha_bar

    def euler_solve_z_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Use ODE-based Euler method to sample from the model and yield intermediate samples.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function to apply to the x_start prediction.
        :param model_kwargs: if not None, extra kwargs to pass to the model.
        :param device: if specified, the device to create the samples on.
        :param progress: if True, show a tqdm progress bar.
        :return: a generator over dicts with 'sample', 'z', and 'pred_xstart'.
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))

        # 1. Initialize x_T
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)

        # 2. Transform Initial Condition: x_T -> z_T
        # z_T = x_T / sqrt(alpha_bar_T) using cumulative alpha
        t_start = th.tensor([self.num_timesteps - 1] * shape[0], device=device)
        sqrt_alpha_bar_T = _extract_into_tensor(self.sqrt_alphas_cumprod, t_start, img.shape)
        z = img / sqrt_alpha_bar_T

        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                h_cur, _ = self._get_params(i, shape, device)

                if i > 0:
                    h_next, sqrt_alpha_next = self._get_params(i - 1, shape, device)
                else:
                    # Terminal condition t=0
                    h_next = th.zeros_like(h_cur)
                    sqrt_alpha_next = th.ones_like(z)

                # Calculate derivative
                epsilon = self.zdot(model, z, t, clip_denoised, denoised_fn, model_kwargs)

                # Euler Step
                dt = h_next - h_cur
                z = z + dt * epsilon

                # Map back to x-space
                sample = z * sqrt_alpha_next

                yield {"sample": sample, "z": z, "pred_xstart": None}


    class _DiffusionDynamicsAdapter(ControlledDynamicsProtocol):
        """
        Wraps the pipeline to behave like a ControlledDynamics object.
        Injects the control 'u' additively: dz/dh = epsilon_theta(z,t) + u

        The adapter stores the model and diffusion kwargs at construction time
        to avoid needing to pass them through the solver's xdot calls.
        """
        def __init__(
            self,
            pipeline: "GaussianDiffusionDDIMODE",
            model,
            timestep_indices: list,
            clip_denoised: bool = True,
            denoised_fn=None,
            model_kwargs: dict = None,
        ):
            self.pipe = pipeline
            self.model = model
            self.timestep_indices = timestep_indices  # DDIM indices in solver order [T-1, T-2, ..., 0]
            self.clip_denoised = clip_denoised
            self.denoised_fn = denoised_fn
            self.model_kwargs = model_kwargs if model_kwargs is not None else {}

        def xdot(self, x: th.Tensor, u: th.Tensor, t: th.Tensor, **kwargs) -> th.Tensor:
            """
            Compute dz/dh = epsilon_theta(z, t) + u

            Args:
                x: Current state z in z-space [B, C, H, W]
                u: Control signal [B, C, H, W]
                t: DDIM timestep index (passed from solver as timesteps[k])

            Returns:
                Drift plus control [B, C, H, W]
            """
            batch_size = x.shape[0]
            # t comes as timesteps[k] which is the DDIM index (long tensor or scalar)
            ddim_t = t.long().expand(batch_size)

            # Get natural diffusion drift in z-space
            drift_z = self.pipe.zdot(
                model=self.model,
                z=x,
                t=ddim_t,
                clip_denoised=self.clip_denoised,
                denoised_fn=self.denoised_fn,
                model_kwargs=self.model_kwargs,
            )
            # Add control
            return drift_z + u

        def juT_p(self, p: th.Tensor, t: th.Tensor) -> th.Tensor:
            # For dx/dt = F(x) + u, Jacobian J_u = I.
            # J_u^T * p = I * p = p
            return p

    def emsa_solve(
        self,
        model,
        shape,
        phi_fn: Callable,
        rho_u: float = 1.0,
        iters: int = 20,
        eta: float = 0.01,
        xi: float = 0.9,
        minibatch: int = 0,
        chunk: bool = False,
        clip_denoised: bool = True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        generator: Optional[th.Generator] = None,
        verbose: bool = True,
        print_fn: Optional[Callable] = print,
        convergence_mode: str = "both",
        cost_tol_abs: float = 1e-5,
        cost_tol_rel: float = 1e-4,
    ):
        """
        Solve optimal control problem using EMSA in z-space.

        The perturbed ODE is:
            dz/dh = epsilon_theta(z, t) + u
        where h = sqrt((1 - alpha_bar) / alpha_bar) decreases from h_T to 0.

        Args:
            model: The noise prediction model.
            shape: Shape of samples (B, C, H, W).
            phi_fn: Terminal cost function Phi(x) -> [B] tensor.
            rho_u: Weight for quadratic control cost.
            iters: Number of EMSA iterations.
            eta: Step size for control update.
            xi: Damping factor (0 < xi < 1).
            minibatch: Minibatch size for chunked execution.
            chunk: Whether to use chunked execution for memory efficiency.
            clip_denoised: If True, clip predicted x_start to [-1, 1].
            denoised_fn: If not None, a function to apply to predicted x_start.
            model_kwargs: Extra kwargs for the model.
            device: Device to run on.
            generator: Optional random generator for reproducibility.
            verbose: Whether to print progress.

        Returns:
            Dictionary containing:
                - "sample": Final sample in x-space [B, C, H, W]
                - "z": Final state in z-space [B, C, H, W]
                - "U": Optimized controls [B, N, C, H, W]
                - "X": State trajectory in z-space [B, N+1, C, H, W]
                - "cost_history": Cost evolution over iterations
                - "time_bench": Timing breakdown
        """
        if device is None:
            device = next(model.parameters()).device

        N = self.num_timesteps

        # 1. Build timestep arrays
        # DDIM indices in solver order: [T-1, T-2, ..., 0]
        timestep_indices = list(range(N))[::-1]

        # Build h values for step size computation
        h_values = []
        for idx in timestep_indices:
            h_i, _ = self._get_params(idx, shape, device)
            h_values.append(h_i.flatten()[0])
        h_values.append(th.tensor(0.0, device=device, dtype=h_values[0].dtype))
        h_values = th.stack(h_values)  # [N+1]

        # Step sizes (negative because h decreases from h_T to 0)
        h_steps = h_values[1:] - h_values[:-1]  # [N]

        # Timesteps tensor contains DDIM indices (passed to xdot as t)
        timesteps = th.tensor(timestep_indices + [0], device=device, dtype=th.long)

        # 2. Create dynamics adapter with stored model and kwargs
        dynamics = self._DiffusionDynamicsAdapter(
            pipeline=self,
            model=model,
            timestep_indices=timestep_indices,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )

        # 3. Create solver
        solver = EMSAQuadCostSolver(dynamics=dynamics, rho_u=rho_u, verbose=verbose)

        # 4. Initialize z_T from noise
        if generator is not None:
            # img = th.randn(*shape, device=device, generator=generator)
            img = generator.randn(shape, device=device)
        else:
            img = th.randn(*shape, device=device)

        # Transform x_T -> z_T
        t_start = th.tensor([N - 1] * shape[0], device=device)
        sqrt_alpha_bar_T = _extract_into_tensor(self.sqrt_alphas_cumprod, t_start, img.shape)
        z0 = img / sqrt_alpha_bar_T

        # 5. Initialize control sequence (zeros)
        U_init = th.zeros(shape[0], N, *shape[1:], device=device, dtype=z0.dtype)

        # 6. Wrap terminal cost to z-space
        # At t=0, x_0 = z_0 * sqrt(alpha_bar_0)
        # Use a scalar that broadcasts to any batch size (for chunked execution)
        sqrt_alpha_bar_0_scalar = self.sqrt_alphas_cumprod[0]
        wrapped_phi = lambda z_terminal, **kw: phi_fn(z_terminal * sqrt_alpha_bar_0_scalar, **kw)

        # 7. Run solver
        result = solver.solve(
            x0=z0,
            U_init=U_init,
            timesteps=timesteps,
            h_steps=h_steps,
            phi_fn=wrapped_phi,
            iters=iters,
            eta=eta,
            xi=xi,
            chunk=chunk,
            minibatch=minibatch,
            print_fn=print_fn,
            convergence_mode=convergence_mode,
            cost_tol_abs=cost_tol_abs,
            cost_tol_rel=cost_tol_rel,
        )

        # 8. Convert final state to x-space
        z_final = result["X"][:, -1]
        x_final = z_final * sqrt_alpha_bar_0_scalar

        return {
            "sample": x_final,
            "z": z_final,
            "U": result["U"],
            "X": result["X"],
            "cost_history": result.get("cost_history"),
            "time_bench": result.get("time_bench"),
        }
