import jax
import jax.numpy as jnp
import jax.scipy.fft as jsp_fft


class ChebyshevGaussGrid2D:
    """Chebyshev-Gauss grid on [x_start, x_end] x [y_start, y_end]."""

    def __init__(
        self,
        x_start: float,
        x_end: float,
        y_start: float,
        y_end: float,
        n_x: int,
        n_y: int,
        dtype=jnp.float64,
    ):
        if n_x < 1 or n_y < 1:
            raise ValueError("Chebyshev-Gauss grids need at least one node per axis.")

        self.x_start = x_start
        self.x_end = x_end
        self.y_start = y_start
        self.y_end = y_end
        self.n_x = n_x
        self.n_y = n_y
        self.dtype = dtype

        self.N_x = n_x - 1
        self.N_y = n_y - 1

        # Chebyshev-Gauss angles: roots of T_n, no endpoints.
        self.theta_x = _chebgauss_theta(n_x, dtype)
        self.theta_y = _chebgauss_theta(n_y, dtype)

        self._nodes_x_ref = jnp.cos(self.theta_x)
        self._nodes_y_ref = jnp.cos(self.theta_y)

        # Map t in [-1, 1] to the physical domain.
        self.nodes_x = (x_start + (x_end - x_start) * (1.0 - self._nodes_x_ref) / 2.0).astype(dtype)
        self.nodes_y = (y_start + (y_end - y_start) * (1.0 - self._nodes_y_ref) / 2.0).astype(dtype)

        # 2-D meshgrid (indexing='xy': shape (n_y, n_x)).
        self.X, self.Y = jnp.meshgrid(self.nodes_x, self.nodes_y)

    def make_evaluation_operator(self, x_eval, y_eval):
        x_ref = 1.0 - 2.0 * (x_eval - self.x_start) / (self.x_end - self.x_start)
        y_ref = 1.0 - 2.0 * (y_eval - self.y_start) / (self.y_end - self.y_start)

        w1 = _barycentric_weights_chebgauss(self.n_x, self.dtype)
        w2 = _barycentric_weights_chebgauss(self.n_y, self.dtype)

        def evaluation_operator(val_cheb_grid):
            """
            val_cheb_grid: (n_y, n_x)
            returns: (p,) values at paired points (x_eval[k], y_eval[k])
            """
            # Interpolate in y for each x-column, then in x at paired points.
            f_y = jax.vmap(
                lambda col: _barycentric_interpolate(self._nodes_y_ref, col, y_ref, w2),
                in_axes=1,
                out_axes=1,
            )(val_cheb_grid)

            return jax.vmap(
                lambda row_k, xk: _barycentric_interpolate(
                    self._nodes_x_ref, row_k, jnp.array([xk], dtype=self.dtype), w1
                )[0],
                in_axes=(0, 0),
            )(f_y, x_ref)

        return evaluation_operator

    def compute_coeffs(self, f_grid: jnp.ndarray) -> jnp.ndarray:
        return chebgauss_coeffs_nd(f_grid).astype(self.dtype)

    def eval_function(self, cheb_coeffs: jnp.ndarray) -> jnp.ndarray:
        return chebgauss_values_nd(cheb_coeffs).astype(self.dtype)

    def eval_gradient(self, cheb_coeffs: jnp.ndarray):
        """
        Differentiate in Chebyshev coefficient space via backward recurrence,
        then evaluate on the Chebyshev-Gauss grid.

        Returns
        -------
        du_dx, du_dy : each (n_y, n_x) arrays of physical-space derivatives
        """
        dc_dx = self.du_dx_coeff(cheb_coeffs)
        dc_dy = self.du_dy_coeff(cheb_coeffs)

        du_dx = self.eval_function(dc_dx)
        du_dy = self.eval_function(dc_dy)

        return du_dx, du_dy

    def pad_coeffs(self, cheb_coeffs, n_y_new, n_x_new):
        assert n_x_new >= self.n_x and n_y_new >= self.n_y
        return jnp.zeros((n_y_new, n_x_new), dtype=cheb_coeffs.dtype).at[: self.n_y, : self.n_x].set(cheb_coeffs)

    def pad_coeffs_rho(self, cheb_coeffs: jnp.ndarray, rho: int):
        """Pad coefficients for a Chebyshev-Gauss grid refined by a factor rho."""
        return self.pad_coeffs(cheb_coeffs, self.n_y * rho, self.n_x * rho)

    def eval_laplacian(self, cheb_coeffs: jnp.ndarray):
        """Evaluate the Laplacian on the Chebyshev-Gauss grid."""
        d2c_dx2 = self.du_dx_coeff(self.du_dx_coeff(cheb_coeffs))
        d2c_dy2 = self.du_dy_coeff(self.du_dy_coeff(cheb_coeffs))

        return self.eval_function(d2c_dx2 + d2c_dy2)

    def du_dx_coeff(self, cheb_coeffs: jnp.ndarray):
        """
        Differentiate Chebyshev coefficients along x (axis 1), including
        the reference-to-physical scaling: dt/dx = -2/(x_end - x_start).
        """
        return jax.vmap(_cheb_diff_1d)(cheb_coeffs) * (-2.0 / (self.x_end - self.x_start))

    def du_dy_coeff(self, cheb_coeffs: jnp.ndarray):
        """
        Differentiate Chebyshev coefficients along y (axis 0), including
        the reference-to-physical scaling: dt/dy = -2/(y_end - y_start).
        """
        return jax.vmap(_cheb_diff_1d)(cheb_coeffs.T).T * (-2.0 / (self.y_end - self.y_start))


class ChebyshevGaussGrid1D:
    """Chebyshev-Gauss grid on [x_start, x_end]."""

    def __init__(self, x_start: float, x_end: float, n_x: int, dtype=jnp.float64):
        if n_x < 1:
            raise ValueError("Chebyshev-Gauss grids need at least one node.")

        self.x_start = x_start
        self.x_end = x_end
        self.n_x = n_x
        self.dtype = dtype

        self.N_x = n_x - 1

        self.theta_x = _chebgauss_theta(n_x, dtype)
        self._nodes_x_ref = jnp.cos(self.theta_x)

        self.nodes_x = (x_start + (x_end - x_start) * (1.0 - self._nodes_x_ref) / 2.0).astype(dtype)

    def make_evaluation_operator(self, x_eval):
        """
        Barycentric interpolation from grid values to arbitrary physical points.

        Parameters
        ----------
        x_eval : (p,) array
            Physical x-coordinates where values are desired.

        Returns
        -------
        evaluation_operator : callable
            ``evaluation_operator(val_cheb_grid)`` with ``val_cheb_grid`` shape (n_x,)
            returns values at ``x_eval`` with shape (p,).
        """
        x_ref = 1.0 - 2.0 * (x_eval - self.x_start) / (self.x_end - self.x_start)
        w1 = _barycentric_weights_chebgauss(self.n_x, self.dtype)

        def evaluation_operator(val_cheb_grid):
            return _barycentric_interpolate(self._nodes_x_ref, val_cheb_grid, x_ref, w1)

        return evaluation_operator

    def compute_coeffs(self, f_grid: jnp.ndarray) -> jnp.ndarray:
        return chebgauss_coeffs_nd(f_grid).astype(self.dtype)

    def eval_function(self, cheb_coeffs: jnp.ndarray) -> jnp.ndarray:
        return chebgauss_values_nd(cheb_coeffs).astype(self.dtype)

    def eval_gradient(self, cheb_coeffs: jnp.ndarray):
        """
        Differentiate in Chebyshev coefficient space via backward recurrence,
        then evaluate on the Chebyshev-Gauss grid.

        Returns
        -------
        du_dx : (n_x,) array of physical-space derivatives
        """
        dc_dx = self.du_dx_coeff(cheb_coeffs)
        return self.eval_function(dc_dx)

    def pad_coeffs(self, cheb_coeffs, n_x_new):
        assert n_x_new >= self.n_x
        return jnp.zeros((n_x_new,), dtype=cheb_coeffs.dtype).at[: self.n_x].set(cheb_coeffs)

    def pad_coeffs_rho(self, cheb_coeffs: jnp.ndarray, rho: int):
        """Pad coefficients for a Chebyshev-Gauss grid refined by a factor rho."""
        return self.pad_coeffs(cheb_coeffs, self.n_x * rho)

    def eval_laplacian(self, cheb_coeffs: jnp.ndarray):
        """Evaluate the second derivative on the Chebyshev-Gauss grid."""
        d2c_dx2 = self.du_dx_coeff(self.du_dx_coeff(cheb_coeffs))
        return self.eval_function(d2c_dx2)

    def du_dx_coeff(self, cheb_coeffs: jnp.ndarray):
        """
        Differentiate Chebyshev coefficients along x, including
        the reference-to-physical scaling: dt/dx = -2/(x_end - x_start).
        """
        return _cheb_diff_1d(cheb_coeffs) * (-2.0 / (self.x_end - self.x_start))


def chebgauss_coeffs_axis(x: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
    """
    Convert values at Chebyshev-Gauss nodes to Chebyshev coefficients.

    Nodes are x_j = cos((j + 1/2) pi / n), j = 0, ..., n - 1.
    The returned coefficients satisfy f_j = sum_k a_k T_k(x_j).
    """
    x = jnp.asarray(x)
    n = x.shape[axis]

    coeffs = jsp_fft.dct(x, type=2, axis=axis, norm=None) / n
    return _scale_axis_index(coeffs, axis, 0, 0.5)


def chebgauss_values_axis(coeffs: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
    """Evaluate Chebyshev coefficients at Chebyshev-Gauss nodes along one axis."""
    coeffs = jnp.asarray(coeffs)
    n = coeffs.shape[axis]

    # Pack Chebyshev coefficients into the unnormalized DCT-II spectrum.
    dct_spectrum = coeffs * n
    dct_spectrum = _scale_axis_index(dct_spectrum, axis, 0, 2.0)
    return jsp_fft.idct(dct_spectrum, type=2, axis=axis, norm=None)


def chebgauss_coeffs_nd(x: jnp.ndarray, axes=None) -> jnp.ndarray:
    """Apply the Chebyshev-Gauss value-to-coefficient transform along each axis."""
    x = jnp.asarray(x)
    if axes is None:
        axes = tuple(range(x.ndim))
    coeffs = jsp_fft.dctn(x, type=2, axes=axes, norm=None)
    for ax in axes:
        coeffs = coeffs / x.shape[ax]
        coeffs = _scale_axis_index(coeffs, ax, 0, 0.5)
    return coeffs


def chebgauss_values_nd(coeffs: jnp.ndarray, axes=None) -> jnp.ndarray:
    """Evaluate Chebyshev coefficients on the tensor-product Chebyshev-Gauss grid."""
    coeffs = jnp.asarray(coeffs)
    if axes is None:
        axes = tuple(range(coeffs.ndim))
    dct_spectrum = coeffs
    for ax in axes:
        dct_spectrum = dct_spectrum * coeffs.shape[ax]
        dct_spectrum = _scale_axis_index(dct_spectrum, ax, 0, 2.0)
    return jsp_fft.idctn(dct_spectrum, type=2, axes=axes, norm=None)


def _chebgauss_theta(n: int, dtype) -> jnp.ndarray:
    j = jnp.arange(n, dtype=dtype)
    return (j + 0.5) * jnp.pi / n


def _barycentric_weights_chebgauss(n: int, dtype) -> jnp.ndarray:
    theta = _chebgauss_theta(n, dtype)
    signs = jnp.where(jnp.arange(n) % 2 == 0, 1.0, -1.0).astype(dtype)
    return signs * jnp.sin(theta)


def _scale_axis_index(x: jnp.ndarray, axis: int, index: int, factor: float) -> jnp.ndarray:
    idx = [slice(None)] * x.ndim
    idx[axis] = index
    idx = tuple(idx)
    return x.at[idx].set(x[idx] * factor)


def _barycentric_interpolate(x_nodes, f_nodes, x_eval, weights):
    # The exact-node branch avoids 0/0 when an evaluation point is a grid node.
    diff = x_eval[:, None] - x_nodes[None, :]
    exact = jnp.abs(diff) < 1e-12
    safe_diff = jnp.where(exact, 1.0, diff)
    w_over_d = weights[None, :] / safe_diff
    numerator = w_over_d @ f_nodes
    denominator = jnp.sum(w_over_d, axis=1)
    result = numerator / denominator
    exact_idx = jnp.argmax(exact, axis=1)
    hit_any = jnp.any(exact, axis=1)
    return jnp.where(hit_any, f_nodes[exact_idx], result)


def _cheb_diff_1d(a: jnp.ndarray) -> jnp.ndarray:
    """
    Differentiate 1D Chebyshev coefficients using the backward recurrence:
        b[N]   = 0
        b[N-1] = 2*N * a[N]
        c_k * b[k] = b[k+2] + 2*(k+1)*a[k+1]   for k = N-2, ..., 0
    where c_0 = 2, c_k = 1 for k >= 1.
    """
    n = a.shape[0]
    N = n - 1

    if N <= 0:
        return jnp.zeros_like(a)
    if N == 1:
        return jnp.array([a[1], 0.0], dtype=a.dtype)

    b_N = jnp.zeros((), dtype=a.dtype)
    b_Nm1 = 2.0 * N * a[N]

    def body(carry, k_from_top):
        b_kp2, b_kp1 = carry
        k = N - 2 - k_from_top
        c_k = jnp.where(k == 0, 2.0, 1.0)
        b_k = (b_kp2 + 2.0 * (k + 1) * a[k + 1]) / c_k
        return (b_kp1, b_k), b_k

    _, b_rest = jax.lax.scan(body, (b_N, b_Nm1), jnp.arange(N - 1))
    b = jnp.concatenate([jnp.flip(b_rest), jnp.array([b_Nm1, b_N])])
    return b


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    u = lambda x, y: jnp.exp(x) * jnp.exp(y)
    du_dx_exact = lambda x, y: jnp.exp(x) * jnp.exp(y)
    du_dy_exact = lambda x, y: jnp.exp(x) * jnp.exp(y)

    grid = ChebyshevGaussGrid2D(0.0, 1.0, 0.0, 1.0, 24, 24)
    coeffs = grid.compute_coeffs(u(grid.X, grid.Y))
    u_recovered = grid.eval_function(coeffs)
    du_dx, du_dy = grid.eval_gradient(coeffs)

    print(f"max reconstruction err = {jnp.max(jnp.abs(u_recovered - u(grid.X, grid.Y))):.2e}")
    print(f"max du/dx err          = {jnp.max(jnp.abs(du_dx - du_dx_exact(grid.X, grid.Y))):.2e}")
    print(f"max du/dy err          = {jnp.max(jnp.abs(du_dy - du_dy_exact(grid.X, grid.Y))):.2e}")
