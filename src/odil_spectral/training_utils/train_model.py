import jax
import jax.numpy as jnp
import time
from jaxopt import LBFGS
import optax

from jax.scipy.sparse.linalg import gmres

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

def train_model_gauss_newton_exact(
    params,
    residual_fn,
    maxiter=1000,
    log_every=100,
    grad_tol=1e-10,
    damping=1e-3,
    damping_factor=10.0,
    min_damping=1e-12,
    max_damping=1e12,
    tr_rho_good=0.75,
    tr_rho_bad=0.25,
    max_damping_tries=20,
    max_step_norm=None,
    fallback_step_size=1e-8,
    log_fn=None,
):
    """
    Train using damped Gauss-Newton / Levenberg-Marquardt on a residual vector r(params).

    Loss: 0.5 * ||r||^2.  Steps are full LM steps (no line search).  The damping λ is
    updated with a trust-region style rule (gain ratio ρ = actual / predicted reduction):
    accept if ρ ≥ tr_rho_bad; decrease λ if ρ > tr_rho_good; reject and increase λ if ρ < tr_rho_bad.

    Parameters
    ----------
    params : pytree
        Initial parameters.
    residual_fn : callable
        Residual function of params returning a vector; loss = 0.5 * ||r||^2.
    maxiter : int
        Maximum number of iterations.
    log_every : int
        Compute and store diagnostics every this many iterations.
    grad_tol : float
        Stop when the gradient norm drops below this value.
    damping : float
        Initial Levenberg-Marquardt damping coefficient.
    damping_factor : float
        Multiplicative factor to increase/decrease damping on reject / strong agreement.
    min_damping : float
        Minimum allowable damping.
    max_damping : float
        Maximum allowable damping.
    tr_rho_good : float
        If gain ratio ρ exceeds this after an accepted step, decrease λ (expand step).
    tr_rho_bad : float
        If ρ is below this, reject the step and increase λ (shrink trust region).
    max_damping_tries : int
        Maximum λ adjustments per outer iteration before fallback gradient step.
    max_step_norm : float or None
        If set, clip the LM step to this L2 norm for safety.
    fallback_step_size : float
        Tiny gradient step if no acceptable LM step is found.
    log_fn : callable or None
        Optional user-supplied logging function with signature
            log_fn(params) -> dict[str, scalar]
        Called every `log_every` iterations.  Each key becomes an entry
        in the returned history (list of scalars).  The function is
        JIT-compiled internally, so it must be JAX-traceable.
    """
    print("=" * 80)
    print("Training Model using Gauss-Newton (Levenberg-Marquardt)")
    print(f"Maximum iterations: {maxiter}, Gradient tolerance: {grad_tol:.2e}, Initial damping: {damping:.2e}")
    print("=" * 80)

    def flatten_params(p):
        leaves, treedef = jax.tree_util.tree_flatten(p)
        flat = jnp.concatenate([x.reshape(-1) for x in leaves]) if leaves else jnp.array([])
        return flat, treedef, leaves

    def unflatten_params(flat, treedef, template_leaves):
        sizes = [x.size for x in template_leaves]
        shapes = [x.shape for x in template_leaves]
        out_leaves = []
        idx = 0
        for size, shape in zip(sizes, shapes):
            out_leaves.append(flat[idx:idx + size].reshape(shape))
            idx += size
        return jax.tree_util.tree_unflatten(treedef, out_leaves)

    flat0, treedef, template_leaves = flatten_params(params)
    n_params = int(flat0.size)

    @jax.jit
    def residual_vector(flat_p):
        p_dict = unflatten_params(flat_p, treedef, template_leaves)
        return residual_fn(p_dict).reshape(-1)

    @jax.jit
    def loss_scalar(flat_p):
        r = residual_vector(flat_p)
        return 0.5 * jnp.sum(r * r)

    def loss_fn_dict(p):
        flat, _, _ = flatten_params(p)
        return loss_scalar(flat)

    grad_fn = jax.jit(jax.grad(loss_scalar))
    jac_fn = jax.jit(jax.jacfwd(residual_vector))

    jit_val_grad = jax.jit(jax.value_and_grad(loss_fn_dict))
    jit_log = _jit_log_fn(log_fn)

    # ---------- initial diagnostics ----------
    init_loss, init_grad_norm, init_user = _run_diagnostics(params, jit_val_grad, jit_log)
    history = _init_history(init_loss, init_grad_norm, init_user)
    history["residual_norm"] = [float(jnp.sqrt(2.0 * init_loss))]
    history["damping"] = [float(damping)]
    history["gain_ratio"] = [float("nan")]
    _print_diag("Initial", init_loss, init_grad_norm, init_user)

    current_damping = float(damping)
    start_time = time.time()

    for it in range(maxiter):
        flat_params, _, _ = flatten_params(params)

        r = residual_vector(flat_params)
        loss = loss_scalar(flat_params)
        grad = grad_fn(flat_params)
        grad_norm = float(jnp.linalg.norm(grad))

        if grad_norm < grad_tol:
            print(f"\nConverged at iteration {it} with grad norm {grad_norm:.6e}")
            break

        J = jac_fn(flat_params)
        Jtr = -grad
        JtJ = J.T @ J

        step_found = False
        last_rho = float("nan")
        new_flat = flat_params

        for _try in range(max_damping_tries):
            lam = current_damping
            H = JtJ + lam * jnp.eye(n_params, dtype=flat_params.dtype)
            step = jnp.linalg.solve(H, Jtr)

            if max_step_norm is not None:
                step_norm = jnp.linalg.norm(step)
                scale = jnp.minimum(1.0, max_step_norm / (step_norm + 1e-30))
                step = step * scale

            # Predicted reduction from the LM quadratic model: m(p) = 0.5||r + Jp||^2 + 0.5*lam||p||^2
            r_lin = r + J @ step
            pred_loss = 0.5 * jnp.sum(r_lin * r_lin) + 0.5 * lam * jnp.dot(step, step)
            predicted_reduction = loss - pred_loss

            pred_pos = jnp.isfinite(predicted_reduction) & (predicted_reduction > 1e-30 * (jnp.abs(loss) + 1.0))
            if not pred_pos:
                current_damping = min(current_damping * damping_factor, max_damping)
                continue

            trial_flat = flat_params + step
            trial_loss = loss_scalar(trial_flat)
            actual_reduction = loss - trial_loss

            if not jnp.isfinite(trial_loss):
                current_damping = min(current_damping * damping_factor, max_damping)
                continue

            rho = actual_reduction / predicted_reduction

            # Trust region: reject poor agreement, shrink λ schedule via larger damping
            if rho < tr_rho_bad:
                current_damping = min(current_damping * damping_factor, max_damping)
                continue

            # Accept full step
            new_flat = trial_flat
            step_found = True
            last_rho = float(rho)
            if rho > tr_rho_good:
                current_damping = max(current_damping / damping_factor, min_damping)
            break

        if not step_found:
            alpha = fallback_step_size
            new_flat = flat_params - alpha * grad
            last_rho = float("nan")
            print(f"Warning: no LM step accepted at iteration {it}; taking tiny gradient step alpha={alpha:g}")

        params = unflatten_params(new_flat, treedef, template_leaves)

        if it % log_every == 0 or it == maxiter - 1:
            loss_val, grad_norm_val, um = _run_diagnostics(params, jit_val_grad, jit_log)
            _append_history(history, loss_val, grad_norm_val, um)
            history["residual_norm"].append(float(jnp.sqrt(2.0 * loss_val)))
            history["damping"].append(current_damping)
            history["gain_ratio"].append(last_rho)
            rho_str = "nan" if last_rho != last_rho else f"{last_rho:.3f}"
            _print_diag(
                f"Iteration {it} (damping={current_damping:.2e}, rho={rho_str})",
                loss_val, grad_norm_val, um,
            )

    total_time = time.time() - start_time

    # ---------- final diagnostics ----------
    final_loss, final_grad_norm, final_user = _run_diagnostics(params, jit_val_grad, jit_log)
    if abs(history["loss"][-1] - final_loss) > 1e-10:
        _append_history(history, final_loss, final_grad_norm, final_user)
        history["residual_norm"].append(float(jnp.sqrt(2.0 * final_loss)))
        history["damping"].append(current_damping)
        history["gain_ratio"].append(float("nan"))

    print(f"\nGauss-Newton optimization completed in {total_time:.2f}s")
    _print_diag("Final", final_loss, final_grad_norm, final_user)
    print(f"Total logged steps: {len(history['loss']) - 1}")

    history["total_time"] = total_time
    history["log_every"] = log_every

    return params, history

def train_model_gauss_newton_approx(
    params,
    residual_fn,
    maxiter=1000,
    log_every=100,
    grad_tol=1e-10,
    damping=1e-3,
    damping_factor=2,
    min_damping=1e-12,
    max_damping=1e12,
    tr_rho_good=0.75,
    tr_rho_bad=0.25,
    max_damping_tries=20,
    cg_tol=1e-2,
    cg_maxiter=100,
    max_step_norm=None,
    fallback_step_size=1e-8,
    log_fn=None,
):
    """
    Train using matrix-free damped Gauss-Newton / Levenberg-Marquardt on a residual vector r(params).

    Loss: 0.5 * ||r||^2.  Never forms J or J^T J explicitly; uses JVP/VJP to apply
    J and J^T to vectors and solves (J^T J + lam I) p = -J^T r with GMRES.
    Full LM steps only (no line search); λ is updated via a trust-region gain ratio
    ρ = actual / predicted reduction (see ``tr_rho_good`` / ``tr_rho_bad``).

    Parameters
    ----------
    params : pytree
        Initial parameters.
    residual_fn : callable
        Residual function of params returning a vector; loss = 0.5 * ||r||^2.
    maxiter : int
        Maximum number of iterations.
    log_every : int
        Compute and store diagnostics every this many iterations.
    grad_tol : float
        Stop when the gradient norm drops below this value.
    damping : float
        Initial Levenberg-Marquardt damping coefficient.
    damping_factor : float
        Multiplicative factor to increase/decrease damping on reject / strong agreement.
    min_damping : float
        Minimum allowable damping.
    max_damping : float
        Maximum allowable damping.
    tr_rho_good : float
        If gain ratio ρ exceeds this after an accepted step, decrease λ.
    tr_rho_bad : float
        If ρ is below this, reject the step and increase λ.
    max_damping_tries : int
        Maximum λ adjustments per outer iteration before fallback gradient step.
    cg_tol : float
        Relative residual tolerance for the GMRES linear solve.
    cg_maxiter : int
        Maximum GMRES iterations per LM step.
    max_step_norm : float or None
        If set, clip the LM step to this L2 norm for safety.
    fallback_step_size : float
        Tiny gradient step if no acceptable LM step is found.
    log_fn : callable or None
        Optional user-supplied logging function with signature
            log_fn(params) -> dict[str, scalar]
        Called every `log_every` iterations.  Each key becomes an entry
        in the returned history (list of scalars).  The function is
        JIT-compiled internally, so it must be JAX-traceable.
    """
    print("=" * 80)
    print("Training Model using Gauss-Newton Approx (Matrix-Free Levenberg-Marquardt)")
    print(f"Maximum iterations: {maxiter}, Gradient tolerance: {grad_tol:.2e}, Initial damping: {damping:.2e}")
    print("=" * 80)

    def flatten_params(p):
        leaves, treedef = jax.tree_util.tree_flatten(p)
        flat = jnp.concatenate([x.reshape(-1) for x in leaves]) if leaves else jnp.array([])
        return flat, treedef, leaves

    def unflatten_params(flat, treedef, template_leaves):
        sizes = [x.size for x in template_leaves]
        shapes = [x.shape for x in template_leaves]
        out_leaves = []
        idx = 0
        for size, shape in zip(sizes, shapes):
            out_leaves.append(flat[idx:idx + size].reshape(shape))
            idx += size
        return jax.tree_util.tree_unflatten(treedef, out_leaves)

    flat0, treedef, template_leaves = flatten_params(params)

    @jax.jit
    def residual_vector(flat_p):
        p_dict = unflatten_params(flat_p, treedef, template_leaves)
        return residual_fn(p_dict).reshape(-1)

    @jax.jit
    def loss_scalar(flat_p):
        r = residual_vector(flat_p)
        return 0.5 * jnp.sum(r * r)

    def loss_fn_dict(p):
        flat, _, _ = flatten_params(p)
        return loss_scalar(flat)

    def build_ops_at(params_flat):
        """Build matrix-free J and J^T operators via JVP/VJP at the current point."""
        r = residual_vector(params_flat)
        loss = 0.5 * jnp.sum(r * r)
        _, pullback = jax.vjp(residual_vector, params_flat)

        def JT(u):
            return pullback(u)[0]

        def J(v):
            return jax.jvp(residual_vector, (params_flat,), (v,))[1]

        g = JT(r)
        return r, loss, g, J, JT

    jit_val_grad = jax.jit(jax.value_and_grad(loss_fn_dict))
    jit_log = _jit_log_fn(log_fn)

    # ---------- initial diagnostics ----------
    init_loss, init_grad_norm, init_user = _run_diagnostics(params, jit_val_grad, jit_log)
    history = _init_history(init_loss, init_grad_norm, init_user)
    history["residual_norm"] = [float(jnp.sqrt(2.0 * init_loss))]
    history["damping"] = [float(damping)]
    history["gain_ratio"] = [float("nan")]
    _print_diag("Initial", init_loss, init_grad_norm, init_user)

    current_damping = float(damping)
    start_time = time.time()

    for it in range(maxiter):
        flat_params, _, _ = flatten_params(params)

        r, loss, grad, J, JT = build_ops_at(flat_params)
        grad_norm = float(jnp.linalg.norm(grad))

        if grad_norm < grad_tol:
            print(f"\nConverged at iteration {it} with grad norm {grad_norm:.6e}")
            break

        Jtr = -grad
        step_found = False
        last_rho = float("nan")
        new_flat = flat_params

        for _try in range(max_damping_tries):
            lam = current_damping

            def H(v, lam=lam):
                return JT(J(v)) + lam * v

            step = gmres(H, Jtr, tol=cg_tol, maxiter=cg_maxiter)[0]

            if max_step_norm is not None:
                step_norm = jnp.linalg.norm(step)
                scale = jnp.minimum(1.0, max_step_norm / (step_norm + 1e-30))
                step = step * scale

            # Predicted reduction from the LM quadratic model: m(p) = 0.5||r + Jp||^2 + 0.5*lam||p||^2
            r_lin = r + J(step)
            pred_loss = 0.5 * jnp.sum(r_lin * r_lin) + 0.5 * lam * jnp.dot(step, step)
            predicted_reduction = loss - pred_loss

            pred_pos = jnp.isfinite(predicted_reduction) & (predicted_reduction > 1e-30 * (jnp.abs(loss) + 1.0))
            if not pred_pos:
                current_damping = min(current_damping * damping_factor, max_damping)
                continue

            trial_flat = flat_params + step
            trial_loss = loss_scalar(trial_flat)
            actual_reduction = loss - trial_loss

            if not jnp.isfinite(trial_loss):
                current_damping = min(current_damping * damping_factor, max_damping)
                continue

            rho = actual_reduction / predicted_reduction

            if rho < tr_rho_bad:
                current_damping = min(current_damping * damping_factor, max_damping)
                continue

            new_flat = trial_flat
            step_found = True
            last_rho = float(rho)
            if rho > tr_rho_good:
                current_damping = max(current_damping / damping_factor, min_damping)
            break

        if not step_found:
            alpha = fallback_step_size
            new_flat = flat_params - alpha * grad
            last_rho = float("nan")
            print(f"Warning: no LM step accepted at iteration {it}; taking tiny gradient step alpha={alpha:g}")

        params = unflatten_params(new_flat, treedef, template_leaves)

        if it % log_every == 0 or it == maxiter - 1:
            loss_val, grad_norm_val, um = _run_diagnostics(params, jit_val_grad, jit_log)
            _append_history(history, loss_val, grad_norm_val, um)
            history["residual_norm"].append(float(jnp.sqrt(2.0 * loss_val)))
            history["damping"].append(current_damping)
            history["gain_ratio"].append(last_rho)
            rho_str = "nan" if last_rho != last_rho else f"{last_rho:.3f}"
            _print_diag(
                f"Iteration {it} (damping={current_damping:.2e}, rho={rho_str})",
                loss_val, grad_norm_val, um,
            )

    total_time = time.time() - start_time

    # ---------- final diagnostics ----------
    final_loss, final_grad_norm, final_user = _run_diagnostics(params, jit_val_grad, jit_log)
    if abs(history["loss"][-1] - final_loss) > 1e-10:
        _append_history(history, final_loss, final_grad_norm, final_user)
        history["residual_norm"].append(float(jnp.sqrt(2.0 * final_loss)))
        history["damping"].append(current_damping)
        history["gain_ratio"].append(float("nan"))

    print(f"\nGauss-Newton Approx optimization completed in {total_time:.2f}s")
    _print_diag("Final", final_loss, final_grad_norm, final_user)
    print(f"Total logged steps: {len(history['loss']) - 1}")

    history["total_time"] = total_time
    history["log_every"] = log_every

    return params, history