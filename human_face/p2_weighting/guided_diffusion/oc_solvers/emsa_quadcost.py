import torch
import numpy as np
from time import perf_counter
from typing import Callable, Optional, Tuple, List, Dict, Any, Union
from contextlib import nullcontext

# A simple protocol for the Dynamics to ensure type safety
class ControlledDynamicsProtocol:
    """
    Protocol for dynamics governed by dx/dt = F(x, u, t).
    """
    def xdot(self, x: torch.Tensor, u: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute the drift dx/dt."""
        raise NotImplementedError

    def juT_p(self, p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute (∂F/∂u)^T * p. 
        For additive control dx/dt = f(x,t) + u, this is simply p.
        """
        raise NotImplementedError


class EMSAQuadCostSolver:
    """
    Generic Extended Method of Successive Approximations (EMSA) solver for Optimal Control.
    
    Solves:
        min_u J(x, u) = E [ \\int (rho_u/2)||u||^2 dt + Phi(x(T)) ] 
        s.t.  dx/dt = F(x, u, t)
    
    This solver is agnostic to the specific physics (handled by `dynamics`) and 
    time schedules (handled by `timesteps`).
    """
    def __init__(
        self,
        dynamics: ControlledDynamicsProtocol,
        rho_u: float = 1.0,
        verbose: bool = True,
    ):
        """
        Args:
            dynamics: Object implementing xdot and juT_p.
            rho_u: Weight for the quadratic control cost (L2 penalty).
            verbose: Whether to print progress.
        """
        self.dyn = dynamics
        self.rho_u = rho_u
        self.verbose = verbose

    @torch.no_grad()
    def forward_sim(
        self,
        x0: torch.Tensor,
        U: torch.Tensor,
        timesteps: torch.Tensor,
        h_steps: torch.Tensor = None,
        chunk: bool = False,
        minibatch: int = 0
    ) -> torch.Tensor:
        """
        Forward Euler integration of the controlled ODE.

        Args:
            x0: Initial state [B, C, H, W]
            U: Controls [B, N, C, H, W]
            timesteps: Time grid [N+1] (e.g., [T, ..., 0])
            h_steps: Explicit step sizes [N]. If None, computed from timesteps.

        Returns:
            X: State trajectory [B, N+1, C, H, W]
        """
        B, N, *_ = U.shape
        X = torch.zeros(B, N + 1, *x0.shape[1:], device=x0.device, dtype=x0.dtype)
        X[:, 0] = x0

        # Calculate steps (dt)
        # Note: timesteps might be descending (diffusion), so h can be negative.
        h = h_steps if h_steps is not None else timesteps[1:] - timesteps[:-1]

        if not chunk:
            for k in range(N):
                xk = X[:, k]
                uk = U[:, k]
                t_k = timesteps[k]
                
                # Euler step: x_{k+1} = x_k + h * F(x_k, u_k, t_k)
                g = self.dyn.xdot(xk, uk, t_k)
                X[:, k + 1] = xk + h[k] * g
        else:
            # Memory-efficient chunked execution
            G = B if (not minibatch or minibatch <= 0) else minibatch
            for k in range(N):
                dt = h[k]
                t_k = timesteps[k]
                for b0 in range(0, B, G):
                    b1 = min(b0 + G, B)
                    xk = X[b0:b1, k]
                    uk = U[b0:b1, k]
                    g = self.dyn.xdot(xk, uk, t_k)
                    X[b0:b1, k + 1] = xk + dt * g
                    
        return X
    
    @staticmethod
    def convergence_check( 
        cost_last: float,
        cost_curr: float,
        mode: str = "rel",
        abs_tol: float = 1e-6,
        rel_tol: float = 1e-5,
    ) -> Tuple[bool, float]:
        """
        Returns (converged, metric_value).

        mode:
        - "abs": |c_k - c_{k-1}| <= abs_tol
        - "rel": |c_k - c_{k-1}| / max(1, |c_{k-1}|) <= rel_tol
        - "both": abs AND rel must be satisfied
        """
        if cost_last is None:
            return False, float("nan")

        delta = abs(cost_curr - cost_last)

        if mode == "abs":
            return delta <= abs_tol, delta

        denom = max(1.0, abs(cost_last))
        rel = delta / denom

        if mode == "rel":
            return rel <= rel_tol, rel

        if mode == "both":
            converged = (delta <= abs_tol) and (rel <= rel_tol)
            # return the more informative metric (relative is usually preferred)
            return converged, rel

        raise ValueError(f"Unknown convergence mode: {mode!r}")

    
    def solve(
        self,
        x0: torch.Tensor,
        U_init: torch.Tensor,
        timesteps: torch.Tensor,
        h_steps: torch.Tensor = None,
        phi_fn: Callable[[torch.Tensor], torch.Tensor] = None,
        iters: int = 20,
        eta: float = 0.001,  # Step size / Learning rate
        xi: float = 0.995,     # Momentum / Decay factor for MSA
        chunk: bool = False,
        minibatch: int = 0,
        print_fn: Optional[Callable[[str], None]] = print,
        return_history: bool = False,
        convergence_mode: str = "both",  # "abs", "rel", or "both"
        cost_tol_abs: float = 1e-5,
        cost_tol_rel: float = 1e-4,
    ) -> Dict[str, Any]:
        """
        Main optimization loop.

        Args:
            x0: Initial state.
            U_init: Initial guess for controls.
            timesteps: Tensor of time points (length N+1).
            h_steps: Explicit step sizes [N]. If None, computed from timesteps.
            phi_fn: Terminal cost function Phi(x_T) -> scalar or [B].
            iters: Number of EMSA iterations.
            eta: Step size for control update.
            xi: Damping factor (0 < xi < 1).
            chunk: Whether to use chunked execution for memory efficiency.
            minibatch: Size of minibatch for chunked execution (if chunk=True).
            print_fn: Function for printing logs.
            return_history: Whether to return cost history.
            convergence_mode: "abs", "rel", or "both" for convergence check.
            cost_tol_abs: Absolute tolerance for convergence.
            cost_tol_rel: Relative tolerance for convergence.

        Returns:
            Dictionary containing optimized controls 'U', trajectory 'X', and stats.
        """
        if self.verbose:
            print_fn(f"\n{'='*30} [EMSA Optimal Control Solver] {'='*30}")
            print_fn(f"Iters: {iters} | Rho_u: {self.rho_u} | Eta: {eta} | Xi: {xi}")
            if chunk:
                print_fn(f"Chunked execution enabled: batch size {minibatch}")

        # Normalize types
        compute_dtype = x0.dtype
        device = x0.device
        x0 = x0.contiguous()
        U = U_init.clone().contiguous()
        timesteps = timesteps.to(device=device, dtype=compute_dtype)

        
        # Precompute dt (h)
        h = h_steps if h_steps is not None else timesteps[1:] - timesteps[:-1]  # [N]
        w = h.abs()  # For integral approximation if needed
        B, N, *_ = U.shape

        # Prepare group/minibatch size for gradient rescale. 
        group_size = min(minibatch, B) if (minibatch and minibatch > 0) else B
        num_groups = (B + group_size - 1) // group_size
        group_ids = torch.arange(B, device=U.device) // group_size
        group_ids = torch.clamp(group_ids, max=num_groups - 1)
        membership = torch.nn.functional.one_hot(group_ids, num_classes=num_groups).to(U.dtype)
        group_counts = membership.sum(dim=0, keepdim=False).unsqueeze(1).clamp_min(1.0).long()
        print_fn(
            "group size:", group_size,
            "; num groups:", num_groups,
            # "; group counts:", group_counts.cpu().squeeze().numpy()
        )
        term_cost_grad_scls = 1.0 / group_counts[group_ids]   # [B, 1]

        
        # History containers
        cost_history = []
        forward_time_ls = []
        term_cost_time_ls = []
        backward_time_ls = []
        
        # Costate buffer (Full buffer for debugging)
        # Shape matches U's spatial dims
        P = torch.zeros(B, N + 1, *U.shape[2:], device=device, dtype=compute_dtype)

        cost_last = None

        start_time = perf_counter()

        for it in range(iters):
            iter_start = perf_counter()
            
            # 1. Forward Simulation
            #    Compute trajectory X based on current controls U
            X = self.forward_sim(x0, U, timesteps, h_steps=h, chunk=chunk, minibatch=minibatch)
            forward_time_ls.append(perf_counter() - iter_start)

            # 2. Terminal Cost & Gradient
            term_cost_start = perf_counter()
            running_cost = 0.5 * self.rho_u * (U * U).flatten(2).sum(-1)  # [B, N]
            running_cost = (running_cost * w.view(1,N).to(U)).sum(-1)  # [B, N] -> [B] 

            P.zero_()
            if not chunk:
                x_N = X[:, -1].detach().requires_grad_(True)
                if torch.any(torch.isnan(x_N)):
                    raise ValueError("NaN detected in state trajectory during forward simulation.")
                    print("NaN detected in state trajectory during forward simulation.")
                    import pdb; pdb.set_trace()
                    ## Get the index of NaN
                    nan_mask_B_C = torch.isnan(x_N).any(dim=tuple(range(2, x_N.ndim)))
                    nan_indices = torch.nonzero(nan_mask_B_C, as_tuple=False).squeeze().cpu().numpy()
                    print(f"NaN indices in final state: {nan_indices}")

                with torch.enable_grad():
                    term_cost = phi_fn(x_N, minibatch=minibatch, iteration=it, total_iters=iters, print_fn=print_fn)
                    (grad_xN, ) = torch.autograd.grad(term_cost.sum(), x_N)
                #end with
                term_val = term_cost.detach()
                ### Rescale gradient for minibatch 
                grad_xN = grad_xN * term_cost_grad_scls.view(grad_xN.shape[0], *([1] * (grad_xN.ndim - 1)))
                P[:, N] = grad_xN.detach() 
            else:
                # Chunked terminal cost
                term_val_accum = 0.0
                G = minibatch if minibatch > 0 else B
                for b0 in range(0, B, G):
                    b1 = min(b0 + G, B)
                    x_N_chunk = X[b0:b1, -1].detach().requires_grad_(True)
                    with torch.enable_grad():
                        tc = phi_fn(x_N_chunk, minibatch=minibatch, iteration=it, total_iters=iters, print_fn=print_fn)
                        g_chunk = torch.autograd.grad(tc.sum(), x_N_chunk)[0]
                    ## Rescale gradient for minibatch
                    scl = term_cost_grad_scls[b0:b1].view(b1-b0, *([1]*(g_chunk.ndim-1)))
                    P[b0:b1, N] = (g_chunk * scl).detach()
                    term_val_accum += tc.detach().sum()
                term_val = term_val_accum

            ## Note: This forces sync by calling .item() and may slow down compute
            pN_norm = P[:, N].flatten(start_dim=1).norm(dim=-1).mean().item()

            total_cost = running_cost + term_val # [B]
            cost_history.append(total_cost.detach())  # Batch size is not huge -> OK to store all 
            term_cost_time_ls.append(perf_counter() - term_cost_start)

            # 3. Backward Adjoint & Control Update
            #    Iterate k from N-1 down to 0
            
            backward_start = perf_counter()
            for k in range(N - 1, -1, -1):
                t_k = timesteps[k]
                dt = h[k]

                # --- Adjoint Step ---
                # p_k = p_{k+1} + h * (∂f/∂x)^T p_{k+1}
                
                if not chunk:
                    xk = X[:, k].detach().requires_grad_(True)
                    uk = U[:, k].detach() # Treat u as constant for adjoint eq of state
                    
                    # Re-evaluate F(x_k, u_k) to build graph
                    F_val = self.dyn.xdot(xk, uk, t_k)
                    
                    # VJP: (∂f/∂x)^T * p_{k+1}
                    # grad_outputs acts as the vector in VJP
                    (JxT_p, ) = torch.autograd.grad(
                        F_val, xk, 
                        grad_outputs=P[:, k+1], 
                        retain_graph=False,
                        allow_unused=False
                    )
                    
                    P[:, k] = P[:, k+1] + dt * JxT_p
                    
                    # --- Control Update ---
                    # Gradient of Hamiltonian w.r.t u:
                    # dH/du = -rho_u * u + (∂f/∂u)^T * p_{k+1}
                    # EMSA Update: u_{new} = xi * u_{old} + eta * (∂f/∂u)^T * p_{k+1}
                    # (The -rho_u * u term is implicitly handled by the decay xi if xi ~ 1-eta*rho)
                    
                    # Compute (∂f/∂u)^T * p_{k+1}
                    # For xdot = F + u, this is just p_{k+1}
                    JuT_p = self.dyn.juT_p(P[:, k+1], t_k)
                    
                    # Update U in place
                    U[:, k] = xi * U[:, k] + eta * JuT_p

                else:
                    # Chunked backward pass
                    G = minibatch if minibatch > 0 else B
                    for b0 in range(0, B, G):
                        b1 = min(b0 + G, B)
                        xk_c = X[b0:b1, k].detach().requires_grad_(True)
                        uk_c = U[b0:b1, k].detach()
                        p_next_c = P[b0:b1, k+1]
                        
                        F_val_c = self.dyn.xdot(xk_c, uk_c, t_k)
                        (JxT_p_c, ) = torch.autograd.grad(
                            F_val_c, xk_c, grad_outputs=p_next_c, retain_graph=False, allow_unused=False)
                        
                        P[b0:b1, k] = p_next_c + dt * JxT_p_c
                        
                        juT_p_c = self.dyn.juT_p(p_next_c, t_k)
                        U[b0:b1, k] = xi * U[b0:b1, k] + eta * juT_p_c

                # pk_norm = P[:, k].flatten(start_dim=1).norm(dim=-1).mean().item()
                # print_fn(f"    Backward Step k={k:02d} | p_k norm: {pk_norm:.5g}") 
            #end for k 
            backward_time_ls.append(perf_counter() - backward_start)

            p0_norm = P[:, 0].flatten(start_dim=1).norm(dim=-1).mean().item()

            # 4. Logging
            mean_total_cost = total_cost.mean().item()
            converged, conv_metric = self.convergence_check(
                cost_last=cost_last,
                cost_curr=mean_total_cost,
                mode=convergence_mode,
                abs_tol=cost_tol_abs,
                rel_tol=cost_tol_rel,
            )
            metric_name = (
                "absΔcost" if convergence_mode == "abs"
                else "relΔcost" if convergence_mode == "rel"
                else "relΔcost(both)"
            )
            delt_cost = abs(cost_last - mean_total_cost) if cost_last is not None else float('nan')
            if self.verbose:
                print_fn(
                    f"Iter {it:02d}: Total Cost = {mean_total_cost:.5g} (Term: {term_val.mean().item():.5g}) "\
                    f"| p0 norm: {p0_norm:.5g} | p_N norm: {pN_norm:.5g} "\
                    f"| Δcost: {delt_cost:.5g} | {metric_name}: {conv_metric:.5g} "\
                    f"| Time: {perf_counter() - iter_start:.3f}s"
                )

            if converged:
                print_fn(
                    f"Converged at iter {it:02d} "
                    f"(mode={convergence_mode}, abs_tol={cost_tol_abs}, rel_tol={cost_tol_rel})"
                )
                break
            cost_last = mean_total_cost
        #end for 

        
        # Final Simulation with optimized controls
        final_fwd_start = perf_counter()
        X_final = self.forward_sim(x0, U, timesteps, h_steps=h, chunk=chunk, minibatch=minibatch)
        forward_time_ls.append(perf_counter() - final_fwd_start)

        # Final cost (evaluate phi once; no grad needed here)
        running_final = 0.5 * self.rho_u * (U * U).flatten(2).sum(-1)  # [B, N]
        term_final = phi_fn(X_final[:, -1], iteration=iters-1, total_iters=iters).detach()
        J_final = ((running_final * w.view(1, N).to(U)).sum(dim=1) + term_final)

        total_time = perf_counter() - start_time

        print_fn(f"Final Total Cost: {J_final.mean().item():.5g} (Term: {term_final.mean().item():.5g})")

        # if return_opt_x:
        #     if term_final.sum().item() < phi_best:
        #         opt_x = X_final

        # Pack histories
        # if return_term_cost_history:
        cost_history.append(J_final.detach())
        cost_history = torch.stack(cost_history, dim=-1).cpu().float().numpy()  # [B, iters+1]
        # if return_u_history:
        #     U_hist = np.stack(U_hist, axis=1)  # [B, iters, N, *u_shapes]
        # if return_p_history:
        #     P_hist = np.stack(P_hist, axis=1)   # [B, iters, N+1, *p_shapes]

        # timings
        sum_forward_time = sum(forward_time_ls)
        sum_backward_time = sum(backward_time_ls)
        sum_eval_cost_time = sum(term_cost_time_ls)
        misc_time = total_time - (sum_forward_time + sum_backward_time + sum_eval_cost_time)
        
        time_bench = {
            "total_time": float(total_time),
            "forward_time_avg": float(sum_forward_time / len(forward_time_ls)),
            "backward_time_avg": float(sum_backward_time / len(backward_time_ls)) if len(backward_time_ls) > 0 else 0.0,
            "eval_cost_time_avg": float(sum_eval_cost_time / len(term_cost_time_ls)) if len(term_cost_time_ls) > 0 else 0.0,
            "misc_time": float(misc_time),
            "iterations": int(iters),
        }

        if self.verbose:
            print_fn(f"\n{'='*25} [EMSA Solver Time Breakdown (per iter)] {'='*25}")
            # print_fn("Time Breakdown (per iter):")
            avg_misc = time_bench["misc_time"] / max(iters, 1)
            avg_total = time_bench["total_time"] / max(iters, 1)
            print_fn(
                "|{:<10} {:>10.3f}s | {:<10} {:>10.3f}s |".format("Forward", time_bench["forward_time_avg"], "Backward", time_bench["backward_time_avg"]) +
                "\n|{:<10} {:>10.3f}s | {:<10} {:>10.3f}s |".format("Cost Eval", time_bench["eval_cost_time_avg"], "Misc", avg_misc) +
                "\n|{:<10} {:>10.3f}s |".format("Total", avg_total)
            )
            print_fn(f"\n{'='*30} [EMSA Solver Finished] {'='*30}")

        return {
            "U": U,
            "X": X_final,
            "time_bench": time_bench,
            "cost_history": cost_history,
        }
