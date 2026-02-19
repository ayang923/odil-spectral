import jax
import jax.numpy as jnp
import time
from jaxopt import LBFGS
import optax


# ──────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────

def _grad_norm(grad):
    """L2 norm of a flattened gradient pytree."""
    return float(jnp.linalg.norm(
        jnp.concatenate([v.flatten() for v in jax.tree_util.tree_leaves(grad)])
    ))


def _run_diagnostics(params, jit_val_grad, jit_log_fn=None):
    """Return (loss, grad_norm, user_metrics_dict)."""
    loss, grad = jit_val_grad(params)
    grad_norm = _grad_norm(grad)
    user_metrics = {}
    if jit_log_fn is not None:
        user_metrics = {k: float(v) for k, v in jit_log_fn(params).items()}
    return float(loss), grad_norm, user_metrics


def _user_metrics(params, jit_log_fn):
    """Evaluate user log_fn only (no loss/grad)."""
    if jit_log_fn is not None:
        return {k: float(v) for k, v in jit_log_fn(params).items()}
    return {}


def _init_history(init_loss, init_grad_norm, init_user):
    """Create the history dict with initial values."""
    history = {
        "loss": [init_loss],
        "grad_norm": [init_grad_norm],
    }
    for k, v in init_user.items():
        history[k] = [v]
    return history


def _append_history(history, loss, grad_norm, user_metrics):
    """Append one diagnostic snapshot to history."""
    history["loss"].append(loss)
    history["grad_norm"].append(grad_norm)
    for k, v in user_metrics.items():
        history[k].append(v)


def _append_final(history, final_loss, final_grad_norm, final_user):
    """Append final point only if it differs from the last logged one."""
    if abs(history["loss"][-1] - final_loss) > 1e-10:
        _append_history(history, final_loss, final_grad_norm, final_user)


def _print_diag(label, loss, grad_norm, user_metrics):
    """Pretty-print one diagnostic line."""
    parts = [f"Loss = {loss:.6e}", f"Grad Norm = {grad_norm:.6e}"]
    for k, v in user_metrics.items():
        parts.append(f"{k} = {v:.6e}")
    print(f"{label}: {', '.join(parts)}")


def _jit_log_fn(log_fn):
    """JIT-compile the user logging function if provided."""
    return jax.jit(log_fn) if log_fn is not None else None


# ──────────────────────────────────────────────────────────────────────
# L-BFGS
# ──────────────────────────────────────────────────────────────────────

def train_model_lbfgs(params, loss_fn, maxiter=1000, log_every=10, grad_tol=1e-5,
                      stepsize=1.0, history_size=10, log_fn=None):
    """
    Train using L-BFGS optimizer from jaxopt.

    Parameters
    ----------
    params : pytree
        Initial parameters.
    loss_fn : callable
        Scalar loss function of params.
    maxiter : int
        Maximum number of L-BFGS iterations.
    log_every : int
        Compute and store diagnostics every this many iterations.
    grad_tol : float
        Stop when the gradient norm drops below this value.
    stepsize : float
        Initial step size for L-BFGS line search.
    history_size : int
        Number of past iterates to keep for the L-BFGS approximation.
    log_fn : callable or None
        Optional user-supplied logging function with signature
            log_fn(params) -> dict[str, scalar]
        Called every `log_every` iterations.  Each key becomes an entry
        in the returned history (list of scalars).  The function is
        JIT-compiled internally, so it must be JAX-traceable.
    """
    print("=" * 80)
    print("Training Model using L-BFGS")
    print(f"Maximum iterations: {maxiter}, Gradient tolerance: {grad_tol:.2e}")
    print("=" * 80)

    jit_val_grad = jax.jit(jax.value_and_grad(loss_fn))
    jit_log = _jit_log_fn(log_fn)

    # ---------- initial diagnostics ----------
    init_loss, init_grad_norm, init_user = _run_diagnostics(params, jit_val_grad, jit_log)
    history = _init_history(init_loss, init_grad_norm, init_user)
    _print_diag("Initial", init_loss, init_grad_norm, init_user)

    # ---------- optimizer ----------
    lbfgs = LBFGS(
        fun=loss_fn,
        maxiter=maxiter,
        tol=grad_tol,
        stepsize=stepsize,
        history_size=history_size,
        jit=True,
    )

    state = lbfgs.init_state(params)
    current_params = params
    jit_update = jax.jit(lbfgs.update)

    start_time = time.time()

    for i in range(maxiter):
        current_params, state = jit_update(current_params, state)

        if i % log_every == 0 or i == maxiter - 1:
            loss, grad_norm, um = _run_diagnostics(current_params, jit_val_grad, jit_log)
            _append_history(history, loss, grad_norm, um)
            _print_diag(f"Iteration {i}", loss, grad_norm, um)

            if grad_norm < grad_tol:
                print(f"\nConverged at iteration {i} with grad norm {grad_norm:.6e}")
                break

        if hasattr(state, 'error') and state.error <= grad_tol:
            print(f"\nL-BFGS converged at iteration {i}")
            break

    total_time = time.time() - start_time

    # ---------- final diagnostics ----------
    final_loss, final_grad_norm, final_user = _run_diagnostics(current_params, jit_val_grad, jit_log)
    _append_final(history, final_loss, final_grad_norm, final_user)

    print(f"\nL-BFGS optimization completed in {total_time:.2f}s")
    _print_diag("Final", final_loss, final_grad_norm, final_user)
    print(f"Total logged steps: {len(history['loss']) - 1}")

    history["total_time"] = total_time
    history["log_every"] = log_every

    return current_params, history


# ──────────────────────────────────────────────────────────────────────
# Adam (optax)
# ──────────────────────────────────────────────────────────────────────

def train_model_adam(params, loss_fn, maxiter=50000, log_every=100, grad_tol=5e-5,
                     lr=1e-3, patience=2000, factor=0.5, min_lr=1e-8,
                     plateau_tol=0.05, log_fn=None):
    """
    Train using Adam with reduce-on-plateau LR scheduling.

    Parameters
    ----------
    params : pytree
        Initial parameters.
    loss_fn : callable
        Scalar loss function of params.
    maxiter : int
        Maximum number of training steps.
    log_every : int
        Compute and store diagnostics every this many steps.
    grad_tol : float
        Stop when the gradient norm drops below this value.
    lr : float
        Initial learning rate.
    patience : int
        Steps without improvement before reducing LR.
    factor : float
        LR multiplicative reduction factor.
    min_lr : float
        Minimum learning rate.
    plateau_tol : float
        Relative improvement threshold for plateau detection.  The loss
        must decrease by at least ``plateau_tol * |best_loss|`` to count
        as an improvement.
    log_fn : callable or None
        Optional user-supplied logging function with signature
            log_fn(params) -> dict[str, scalar]
        Called every `log_every` steps.  Each key becomes an entry
        in the returned history (list of scalars).  The function is
        JIT-compiled internally, so it must be JAX-traceable.
    """
    optimizer = optax.inject_hyperparams(optax.adam)(learning_rate=lr)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state_new = optimizer.update(grads, opt_state, params)
        params_new = optax.apply_updates(params, updates)
        return params_new, opt_state_new, loss, grads

    jit_val_grad = jax.jit(jax.value_and_grad(loss_fn))
    jit_log = _jit_log_fn(log_fn)

    print("=" * 80)
    print("Training Model using Adam")
    print(f"Maximum iterations: {maxiter}, Initial LR: {lr:.2e}, Gradient tolerance: {grad_tol:.2e}")
    print("=" * 80)

    # ---------- initial diagnostics ----------
    init_loss, init_grad_norm, init_user = _run_diagnostics(params, jit_val_grad, jit_log)
    history = _init_history(init_loss, init_grad_norm, init_user)
    _print_diag("Initial", init_loss, init_grad_norm, init_user)

    # ---------- training loop ----------
    best_loss = jnp.inf
    patience_counter = 0
    start_time = time.time()

    for step in range(maxiter):
        params, opt_state, loss, grads = train_step(params, opt_state)

        # LR Schedule: Reduce on Plateau
        # Strategy: Reset patience on ANY improvement, but only reduce LR if improvement
        # is less than plateau_tol * |best_loss| over the patience window
        if loss < best_loss:
            # Any improvement resets patience
            best_loss = loss
            patience_counter = 0
        else:
            # No improvement: increment patience
            patience_counter += 1

        # Check if we should reduce LR: loss hasn't improved by plateau_tol in patience steps
        if patience_counter >= patience:
            # Check if the improvement since last LR reduction is significant
            improvement_needed = plateau_tol * jnp.abs(best_loss)
            
            current_lr = opt_state.hyperparams['learning_rate']
            new_lr = jnp.maximum(current_lr * factor, min_lr)
            if new_lr < current_lr:
                opt_state.hyperparams['learning_rate'] = new_lr
                print(f"\nStep {step}: Loss plateaued (patience={patience_counter}, best_loss={float(best_loss):.6e}, current_loss={float(loss):.6e}, needed_improvement={float(improvement_needed):.6e}). Reducing LR to {float(new_lr):.2e}")
            patience_counter = 0  # Reset after reducing LR

        if step % log_every == 0 or step == maxiter - 1:
            loss_val = float(loss)
            grad_norm = _grad_norm(grads)
            um = _user_metrics(params, jit_log)

            _append_history(history, loss_val, grad_norm, um)

            lr_val = float(opt_state.hyperparams['learning_rate'])
            _print_diag(f"Step {step} (lr={lr_val:.2e})", loss_val, grad_norm, um)

            if step > 10 and grad_norm < grad_tol:
                print(f"\nConverged at step {step} with grad norm {grad_norm:.6e}")
                break

    total_time = time.time() - start_time

    # ---------- final diagnostics ----------
    final_loss, final_grad_norm, final_user = _run_diagnostics(params, jit_val_grad, jit_log)
    _append_final(history, final_loss, final_grad_norm, final_user)

    print(f"\nAdam optimization completed in {total_time:.2f}s")
    _print_diag("Final", final_loss, final_grad_norm, final_user)
    print(f"Total logged steps: {len(history['loss']) - 1}")

    history["total_time"] = total_time
    history["log_every"] = log_every

    return params, history
