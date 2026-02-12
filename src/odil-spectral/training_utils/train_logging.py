"""
Helper factories that produce `log_fn` callables for use with
`train_model_lbfgs(..., log_fn=...)`.

Each factory closes over the necessary data (grid, target, etc.) and
returns a JAX-traceable function with signature

    log_fn(params) -> dict[str, scalar]
"""

import jax.numpy as jnp


def make_pde_error_log_fn(reconstruct_fn, u_target):
    """
    Return a log_fn that computes abs-max, rel-max, and rel-L2 errors
    between the current solution and a known target.

    Parameters
    ----------
    reconstruct_fn : callable
        Maps params -> u_current (a JAX array the same shape as u_target).
        Example: ``lambda params: grid.eval_function(params["cheb_coeffs"])``
    u_target : jnp.ndarray
        Ground-truth solution on the grid.

    Returns
    -------
    log_fn : callable
        ``log_fn(params) -> {"abs_max": ..., "rel_max": ..., "rel_l2": ...}``
    """
    u_target_max = jnp.max(jnp.abs(u_target))
    u_target_l2 = jnp.linalg.norm(u_target)

    def log_fn(params):
        u_current = reconstruct_fn(params)
        err = u_current - u_target
        abs_max = jnp.max(jnp.abs(err))
        rel_max = abs_max / u_target_max
        rel_l2 = jnp.linalg.norm(err) / u_target_l2
        return {"abs_max": abs_max, "rel_max": rel_max, "rel_l2": rel_l2}

    return log_fn


def make_data_rmse_log_fn(predict_fn, data_u):
    """
    Return a log_fn that computes the RMSE between predicted and observed
    data values (e.g. sparse sensor measurements).

    Parameters
    ----------
    predict_fn : callable
        Maps params -> predicted data values (1-D JAX array, same length
        as data_u).
        Example: ``lambda params: interp_matrix @ params["cheb_coeffs"].ravel()``
    data_u : jnp.ndarray
        Observed data values (1-D).

    Returns
    -------
    log_fn : callable
        ``log_fn(params) -> {"data_rmse": ...}``
    """
    n_data = data_u.shape[0] * data_u.shape[1]

    def log_fn(params):
        pred = predict_fn(params)
        rmse = jnp.sqrt(jnp.sum((pred - data_u) ** 2) / n_data)
        return {"data_rmse": rmse}

    return log_fn


def compose_log_fns(*log_fns):
    """
    Combine multiple log_fn's into one.  If keys collide, later functions
    overwrite earlier ones.

    Parameters
    ----------
    *log_fns : callables
        Each has signature ``log_fn(params) -> dict[str, scalar]``.

    Returns
    -------
    log_fn : callable
        Merged ``log_fn(params) -> dict[str, scalar]``.
    """
    def log_fn(params):
        merged = {}
        for fn in log_fns:
            merged.update(fn(params))
        return merged

    return log_fn
