# Copyright 2020 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ** MODIFIED FILE **
# ---------------------------------------------------------------------------
# This file is a modified version of the original file from the JAX library
# (https://github.com/google/jax).
#
# Copyright 2025 [Juan Toscano / Crunch Group]
#
# Modifications were made to support the vRBA framework, as described in:
# "A Principled Framework for Residual-Based Adaptivity..." 
# [Toscano, et al., 2025]
#
# These modifications are also licensed under the Apache License, Version 2.0.
# ---------------------------------------------------------------------------

"""The Broyden-Fletcher-Goldfarb-Shanno minimization algorithm."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax._src.scipy.optimize.line_search import line_search
from jax._src.scipy.optimize.line_search import line_search as _jax_single_wolfe_line_search
from jax._src.scipy.optimize.line_search import _LineSearchResults # Ensure correct import


@partial(jax.jit, static_argnames=("fun",))
def line_search_jax_with_fallback( # This is the 2-stage fallback discussed before
    fun: Callable, xk: jax.Array, pk: jax.Array,
    old_fval: jax.Array | None = None, old_old_fval: jax.Array | None = None, gfk: jax.Array | None = None,
    c1_try1: float = 1e-4, c2_try1: float = 0.8, maxiter_try1: int = 10,
    c1_try2: float = 1e-4, c2_try2: float = 0.5, maxiter_try2: int = 20
) -> _LineSearchResults:
    results_try1 = _jax_single_wolfe_line_search(fun, xk, pk, old_fval, old_old_fval, gfk,
                                                 c1_try1, c2_try1, maxiter_try1)
    def true_branch_fb_internal_failed(_):
        results_try2 = _jax_single_wolfe_line_search(fun, xk, pk, old_fval, old_old_fval, gfk,
                                                     c1_try2, c2_try2, maxiter_try2)
        return results_try2
    def false_branch_fb_internal_succeeded(r1):
        return r1
    return jax.lax.cond(results_try1.failed, true_branch_fb_internal_failed, false_branch_fb_internal_succeeded, results_try1)


# Also in your jax/_src/scipy/optimize/bfgs.py

@partial(jax.jit, static_argnames=("fun",))
def line_search_normal_then_fallback(
    fun: Callable,
    xk: jax.Array,
    pk: jax.Array,
    old_fval: jax.Array | None = None,
    old_old_fval: jax.Array | None = None,
    gfk: jax.Array | None = None,
    # Parameters for the "normal" initial attempt
    normal_c1: float = 1e-4,
    normal_c2: float = 0.9,
    normal_maxiter: int = 15,
    # Parameters to pass to line_search_jax_with_fallback (which has its own try1 & try2)
    fallback_c1_try1: float = 1e-4,
    fallback_c2_try1: float = 0.8,
    fallback_maxiter_try1: int = 10,
    fallback_c1_try2: float = 1e-4,
    fallback_c2_try2: float = 0.5,
    fallback_maxiter_try2: int = 20
) -> _LineSearchResults:

    # 1. Attempt "normal" line search
    results_normal = _jax_single_wolfe_line_search(
        fun, xk, pk, old_fval, old_old_fval, gfk,
        c1=normal_c1, c2=normal_c2, maxiter=normal_maxiter
    )

    # This function is called if results_normal.failed is True
    def true_branch_normal_failed(_results_normal_ignored):
        # Normal search failed, now use the line_search_jax_with_fallback
        results_full_fallback = line_search_jax_with_fallback(
            fun, xk, pk, old_fval, old_old_fval, gfk,
            c1_try1=fallback_c1_try1, c2_try1=fallback_c2_try1, maxiter_try1=fallback_maxiter_try1,
            c1_try2=fallback_c1_try2, c2_try2=fallback_c2_try2, maxiter_try2=fallback_maxiter_try2
        )
        return results_full_fallback

    # This function is called if results_normal.failed is False
    def false_branch_normal_succeeded(res_normal):
        return res_normal

    final_results = jax.lax.cond(
        results_normal.failed,
        true_branch_normal_failed,
        false_branch_normal_succeeded,
        results_normal
    )
    return final_results


class _BFGSResults(NamedTuple):
  """Results from BFGS optimization.

  Parameters:
    converged: True if minimization converged.
    failed: True if line search failed.
    k: integer the number of iterations of the BFGS update.
    nfev: integer total number of objective evaluations performed.
    ngev: integer total number of jacobian evaluations
    nhev: integer total number of hessian evaluations
    x_k: array containing the last argument value found during the search. If
      the search converged, then this value is the argmin of the objective
      function.
    f_k: array containing the value of the objective function at `x_k`. If the
      search converged, then this is the (local) minimum of the objective
      function.
    g_k: array containing the gradient of the objective function at `x_k`. If
      the search converged the l2-norm of this tensor should be below the
      tolerance.
    H_k: array containing the inverse of the estimated Hessian.
    status: int describing end state.
    line_search_status: int describing line search end state (only means
      something if line search fails).
  """
  converged: bool | jax.Array
  failed: bool | jax.Array
  k: int | jax.Array
  nfev: int | jax.Array
  ngev: int | jax.Array
  nhev: int | jax.Array
  x_k: jax.Array
  f_k: jax.Array
  g_k: jax.Array
  H_k: jax.Array
  old_old_fval: jax.Array
  status: int | jax.Array
  line_search_status: int | jax.Array


_dot = partial(jnp.dot, precision=lax.Precision.HIGHEST)
_einsum = partial(jnp.einsum, precision=lax.Precision.HIGHEST)


def minimize_bfgs(
    fun: Callable,
    x0: jax.Array,
    initial_H: jax.Array,
    maxiter: int | None = None,
    norm=jnp.inf,
    gtol: float = 1e-5,
    update_method: str = "bfgs",
    initial_scale: bool = False,
    # Parameters for the "normal" (primary) line search attempt
    ls_normal_c1: float = 1e-4,
    ls_normal_c2: float = 0.9,
    ls_normal_maxiter: int = 15,
    # Parameters for the *internal tries* of the fallback line search
    ls_fb_c1_try1: float = 1e-4,
    ls_fb_c2_try1: float = 0.8,
    ls_fb_maxiter_try1: int = 10,
    ls_fb_c1_try2: float = 1e-4,
    ls_fb_c2_try2: float = 0.5,
    ls_fb_maxiter_try2: int = 10,
) -> _BFGSResults:
  """Minimize a function using BFGS.

  Implements the BFGS algorithm from
    Algorithm 6.1 from Wright and Nocedal, 'Numerical Optimization', 1999, pg.
    136-143.

  Args:
    fun: function of the form f(x) where x is a flat ndarray and returns a real
      scalar. The function should be composed of operations with vjp defined.
    x0: initial guess.
    maxiter: maximum number of iterations.
    norm: order of norm for convergence check. Default inf.
    gtol: terminates minimization when |grad|_norm < g_tol.
    line_search_maxiter: maximum number of linesearch iterations.

  Returns:
    Optimization result.
  """

  if maxiter is None:
    maxiter = jnp.size(x0) * 200

  d = x0.shape[0]

  #initial_H = jnp.eye(d, dtype=x0.dtype)
  f_0, g_0 = jax.value_and_grad(fun)(x0)
  state = _BFGSResults(
      converged=jnp.linalg.norm(g_0, ord=norm) < gtol,
      failed=False,
      k=0,
      nfev=1,
      ngev=1,
      nhev=0,
      x_k=x0,
      f_k=f_0,
      g_k=g_0,
      H_k=initial_H,
      old_old_fval=f_0 + jnp.linalg.norm(g_0) / 2,
      status=0,
      line_search_status=0,
  )

  def cond_fun(state):
    return (jnp.logical_not(state.converged)
            & jnp.logical_not(state.failed)
            & (state.k < maxiter))    

  def body_fun(state):
    p_k = -_dot(state.H_k, state.g_k)
    
    line_search_results = line_search_normal_then_fallback(
        fun, 
        state.x_k,
        p_k,
        old_fval=state.f_k,
        old_old_fval=state.old_old_fval,
        gfk=state.g_k,
        # Normal attempt parameters
        normal_c1=ls_normal_c1,
        normal_c2=ls_normal_c2,
        normal_maxiter=ls_normal_maxiter,
        # Fallback (which contains two tries) parameters
        fallback_c1_try1=ls_fb_c1_try1,
        fallback_c2_try1=ls_fb_c2_try1,
        fallback_maxiter_try1=ls_fb_maxiter_try1,
        fallback_c1_try2=ls_fb_c1_try2,
        fallback_c2_try2=ls_fb_c2_try2,
        fallback_maxiter_try2=ls_fb_maxiter_try2
    )
    
    state = state._replace(
        nfev=state.nfev + line_search_results.nfev,
        ngev=state.ngev + line_search_results.ngev,
        failed=line_search_results.failed,
        line_search_status=line_search_results.status,
    )
    s_k = line_search_results.a_k * p_k
    x_kp1 = state.x_k + s_k
    f_kp1 = line_search_results.f_k
    g_kp1 = line_search_results.g_k
    y_k = g_kp1 - state.g_k
    # --- MODIFIED rho_k CALCULATION ---
    yk_dot_sk = _dot(y_k, s_k)
    # Define a threshold for considering the denominator effectively zero
    small_denominator_threshold = 1e-32 

    # Condition: is yk_dot_sk too close to zero?
    pred_yk_dot_sk_is_problematic = jnp.abs(yk_dot_sk) < small_denominator_threshold

    def calculate_rho_k_normally(operand_yds):
        # Normal calculation: 1.0 / (y_k^T s_k)
        # Add a very small epsilon here too, just in case yds passed the threshold
        # but is still problematic for reciprocal on some hardware/dtype,
        # or make the threshold slightly larger.
        return 1.0 / (operand_yds + jnp.sign(operand_yds) * 1e-20)


    def handle_problematic_rho_k_denominator(operand_yds):
        # Fallback value, mimicking SciPy's behavior for BFGS
        return jnp.array(1000.0, dtype=operand_yds.dtype)

    # This is the new rho_k, calculated conditionally
    rho_k = jax.lax.cond(
        pred_yk_dot_sk_is_problematic,
        handle_problematic_rho_k_denominator,
        calculate_rho_k_normally,
        yk_dot_sk # Operand passed to the branches
    )
    # --- END OF MODIFIED rho_k CALCULATION ---

    # Store current state variables for convenience
    Hk = state.H_k
    yk = y_k
    sk = s_k
    rhok = rho_k
    alpha_k_linesearch = line_search_results.a_k # Step length from line search
    gfk_old = state.g_k # Gradient at x_k (before line search)
    d = state.x_k.shape[0] # Dimension

    # --- Conditional Hessian Update ---
    if update_method.lower() == "ssbroyden2":
        # SSBroyden2 Update Logic (translated to JAX)
        # Add a small epsilon for numerical stability in divisions
        eps_div = 1e-32 

        Hkyk = _dot(Hk, yk)
        ykHkyk = _dot(yk, Hkyk)
        
        # hk and bk calculations
        hk = ykHkyk * rhok
        bk = -alpha_k_linesearch * rhok * _dot(sk, gfk_old)
        ak = bk * hk - 1.0

        # Intermediate terms for thetak
        abs_ak = jnp.abs(ak)
        # Add epsilon to denominator for sqrt_term
        sqrt_term = jnp.sqrt(abs_ak / (1.0 + abs_ak + eps_div)) 
        rhokm_val = hk * (1.0 - sqrt_term)
        # If hk is large, rhokm_val could be > 1, ensure it's capped by 1
        # Or, if hk is negative (ykHkyk and rhok different signs), rhokm_val could be negative.
        # The original `min(1, ...)` suggests rhokm should not exceed 1.
        # Let's ensure it's also non-negative if that's implied by theory, though original min(1,val) doesn't enforce it.
        # For now, directly translate:
        rhokm = jnp.minimum(1.0, rhokm_val) 

        thetakm_den = ak + jnp.sign(ak) * eps_div # Avoid division by exact zero
        thetakm = (rhokm - 1.0) / thetakm_den
        
        thetakp_den = rhokm + eps_div # Avoid division by zero if rhokm is zero
        thetakp = 1.0 / thetakp_den

        bk_safe_den = bk + jnp.sign(bk) * eps_div # Avoid division by exact zero for term_for_thetak
        term_for_thetak = (1.0 - bk) / bk_safe_den
        thetak = jnp.maximum(thetakm, jnp.minimum(thetakp, term_for_thetak))

        # Tauk calculation (depends on initial_scale and current iteration)
        is_initial_step_and_scale_cond = initial_scale & (state.k == 0) & \
                                        jnp.allclose(Hk, jnp.eye(d, dtype=Hk.dtype), atol=1e-6) # Check if Hk is identity

        tauk_A_den = (1.0 + ak * thetak + eps_div)
        tauk_A = hk / tauk_A_den # tauk for initial scaling case

        rhokk_den = bk + jnp.sign(bk) * eps_div
        rhokk = jnp.minimum(1.0, 1.0 / rhokk_den) # if bk is very small, 1/bk is large
        
        sigmak = 1.0 + thetak * ak
        
        # Handle d=1 case for sigmaknm1_exp to avoid division by zero
        sigmaknm1_exp = lax.cond(d == 1,
                                lambda _: 0.0, # Effectively makes sigmaknm1 = 1 if d=1 (abs(sigmak)**0)
                                lambda _: 1.0 / (1.0 - d),
                                None)
        sigmaknm1 = jnp.abs(sigmak) ** sigmaknm1_exp

        tauk_B_cond = thetak <= 0.0
        tauk_B_true = jnp.minimum(rhokk * sigmaknm1, sigmak)
        tauk_B_false_den = thetak + eps_div
        tauk_B_false = rhokk * jnp.minimum(sigmaknm1, 1.0 / tauk_B_false_den)
        tauk_B = jnp.where(tauk_B_cond, tauk_B_true, tauk_B_false)

        tauk = jnp.where(is_initial_step_and_scale_cond, tauk_A, tauk_B)
        tauk_safe = tauk + jnp.sign(tauk) * eps_div # Ensure tauk is not zero for division

        # vk and phik calculations
        ykHkyk_safe = ykHkyk + jnp.sign(ykHkyk) * eps_div
        vk = sk * rhok - Hkyk / ykHkyk_safe
        
        phik_den = (1.0 + ak * thetak + eps_div)
        phik = (1.0 - thetak) / phik_den

        # SSBroyden2 Hessian update
        term1_H_update = Hk
        term2_H_update_num = _einsum('i,j->ij', Hkyk, Hkyk) # Outer product: Hkyk @ Hkyk.T
        term2_H_update = term2_H_update_num / ykHkyk_safe
        
        term3_H_update_vk_outer = _einsum('i,j->ij', vk, vk) # Outer product: vk @ vk.T
        term3_H_update = phik * ykHkyk * term3_H_update_vk_outer
        
        H_numerator = term1_H_update - term2_H_update + term3_H_update
        H_scaled = H_numerator / tauk_safe # Divide by tauk
        
        term4_H_update_sk_outer = _einsum('i,j->ij', sk, sk) # Outer product: sk @ sk.T
        term4_H_update = term4_H_update_sk_outer * rhok
        
        H_kp1 = H_scaled + term4_H_update

    elif update_method.lower() == "bfgs":
        # Standard BFGS update (already in JAX's bfgs.py)
        sy_k = sk[:, jnp.newaxis] * yk[jnp.newaxis, :] # s_k y_k^T
        w = jnp.eye(d, dtype=rhok.dtype) - rhok * sy_k
        # H_kp1 = (_einsum('ij,jk,lk', w, Hk, w) + rhok * sk[:, jnp.newaxis] * sk[jnp.newaxis, :])
        # More direct Nocedal & Wright (6.17) form:
        H_s = _einsum('i,j->ij', sk, sk) * rhok # rho_k s_k s_k^T
        H_y = _einsum('i,j->ij', Hk @ yk, sk) # H_k y_k s_k^T
        H_yt = _einsum('i,j->ij', sk, yk) @ Hk # s_k y_k^T H_k (use if Hk not symm, but it should be)
                                            # For symmetric Hk, H_yt is transpose of H_y
        # Standard update: H_kp1 = Hk - (Hkyk sk^T + sk (Hkyk)^T)/sTyk + (1 + ykHkyk/sTyk) sk skT/sTyk
        # The einsum version is likely fine and tested by JAX authors. Let's use it.
        # H_kp1 = (_einsum('ij,jk,lk', w, Hk, w) + rhok * _einsum('i,j->ij', sk, sk))
        # The one from the JAX source:
        H_kp1 = (_einsum('ij,jk,lk', w, state.H_k, w)
                + rho_k * s_k[:, jnp.newaxis] * s_k[jnp.newaxis, :])


    else:
        raise ValueError(f"Unknown update_method: {update_method}")

    # Safeguard against non-finite values (already in JAX's bfgs.py)
    H_kp1 = jnp.where(jnp.isfinite(rho_k), H_kp1, state.H_k)
    H_kp1 = jnp.where(jnp.isfinite(rho_k), H_kp1, state.H_k)
    converged = jnp.linalg.norm(g_kp1, ord=norm) < gtol

    state = state._replace(
        converged=converged,
        k=state.k + 1,
        x_k=x_kp1,
        f_k=f_kp1,
        g_k=g_kp1,
        H_k=H_kp1,
        old_old_fval=state.f_k,
    )
    return state

  state = lax.while_loop(cond_fun, body_fun, state)
  status = jnp.where(
      state.converged,
      0,  # converged
      jnp.where(
          state.k == maxiter,
          1,  # max iters reached
          jnp.where(
              state.failed,
              2 + state.line_search_status, # ls failed (+ reason)
              -1,  # undefined
          )
      )
  )
  state = state._replace(status=status)
  return state
