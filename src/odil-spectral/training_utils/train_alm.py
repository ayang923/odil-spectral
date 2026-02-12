"""
General Augmented Lagrangian Method (ALM) outer loop.

The user supplies:
  - make_loss_fn(mu_k, lam) -> loss_fn(params) -> scalar
        A factory that builds the loss for a given penalty weight
        and multiplier vector.  All grid/problem-specific arguments
        are captured inside this closure.
  - eval_data_residual(params) -> jnp.ndarray (1-D)
        Evaluates the constraint residual vector.
  - inner_solver(params, loss_fn, **inner_kwargs) -> (params, history)
        Any of train_model_lbfgs, train_model_adam, or a custom solver
        with the same signature.
"""

import jax.numpy as jnp


def train_augmented_lagrangian(
    params,
    make_loss_fn,
    eval_data_residual,
    residual_metric,
    inner_solver,
    n_constraints,
    mu_k_init=10.0,
    max_outer_steps=100,
    target_res=1e-3,
    eta=0.95,
    mu_factor=10.0,
):
    """
    Augmented Lagrangian outer loop.

    Parameters
    ----------
    params : pytree
        Initial parameters (warm-started across outer iterations).
    make_loss_fn : callable
        ``make_loss_fn(mu_k, lam) -> loss_fn``
        Factory that returns a scalar loss function of ``params``.
        All problem-specific arguments (grid, data, PDE, etc.) should
        be captured in the closure.
    eval_data_residual : callable
        ``eval_data_residual(params) -> jnp.ndarray``  (1-D)
        Evaluates the equality-constraint residual vector.
    residual_metric : tuple (str, callable)
        ``(name, fn)`` where ``fn(r) -> scalar`` maps a 1-D residual
        vector to a scalar metric (e.g. RMSE, max absolute error).
        ``name`` is used as the history key and in printed diagnostics.
        Example: ``("rmse", lambda r: jnp.sqrt(jnp.mean(r**2)))``.
    inner_solver : callable
        ``inner_solver(params, loss_fn, **kwargs) -> (params, history)``
        E.g. ``train_model_lbfgs`` or ``train_model_adam``.
    n_constraints : int
        Length of the multiplier / residual vector.
    mu_k_init : float
        Initial penalty weight.
    max_outer_steps : int
        Maximum number of ALM outer iterations.
    target_res : float
        Stop when the metric drops below this value.
    eta : float
        If ``metric > eta * best_metric``, the penalty is increased.
    mu_factor : float
        Multiplicative increase applied to ``mu_k`` when the metric stagnates.

    Returns
    -------
    params : pytree
        Optimised parameters.
    history : dict
        ``{"r_{name}": [...], "mu_k": [...], "inner_histories": [...]}``
    """
    metric_name, metric_fn = residual_metric
    history_key = f"r_{metric_name}"

    mu_k = mu_k_init
    lam = jnp.zeros(n_constraints)
    best_metric = jnp.inf

    alm_history = {
        history_key: [],
        "mu_k": [],
        "inner_histories": [],
    }

    print("=" * 80)
    print("Augmented Lagrangian Method")
    print(f"max_outer_steps={max_outer_steps}, mu_k_init={mu_k_init:.2e}, "
          f"target_res={target_res:.2e}, eta={eta}, mu_factor={mu_factor}")
    print(f"Convergence metric: {history_key}")
    print("=" * 80)

    for k in range(max_outer_steps):
        # --- check convergence ---
        if best_metric <= target_res:
            print(f"\nALM converged at outer step {k}: "
                  f"best {history_key} = {best_metric:.6e}")
            break

        # --- build loss for current (mu_k, lam) ---
        loss_fn = make_loss_fn(mu_k, lam)

        # --- inner solve ---
        print(f"\n{'─'*60}")
        print(f"ALM outer step {k}  |  mu_k = {mu_k:.4e}")
        print(f"{'─'*60}")
        params, inner_hist = inner_solver(params, loss_fn)
        alm_history["inner_histories"].append(inner_hist)

        # --- evaluate constraint residual ---
        r = eval_data_residual(params)
        metric_val = float(metric_fn(r))

        # --- multiplier update ---
        lam = lam + mu_k * r

        # --- adaptive penalty ---
        if metric_val > eta * best_metric:
            mu_k *= mu_factor
            print(f"  {history_key} not improving → mu_k increased to {mu_k:.4e}")

        if metric_val < best_metric:
            best_metric = metric_val

        alm_history[history_key].append(metric_val)
        alm_history["mu_k"].append(mu_k)

        print(f"  Outer step {k}: {history_key} = {metric_val:.6e}, "
              f"mu_k = {mu_k:.4e}")

    return params, alm_history
