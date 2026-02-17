import jax.numpy as jnp
import jax



class ChebyshevGrid2D:
    """Chebyshev grid on [x_start, x_end] x [y_start, y_end]."""

    def __init__(self, x_start: float, x_end: float,
                 y_start: float, y_end: float,
                 n_x: int, n_y: int, dtype=jnp.float64):
        self.x_start = x_start
        self.x_end = x_end
        self.y_start = y_start
        self.y_end = y_end
        self.n_x = n_x
        self.n_y = n_y
        self.dtype = dtype

        self.N_x = n_x - 1
        self.N_y = n_y - 1

        # Uniform theta grids (Chebyshev-Lobatto): k = 0, 1, ..., N
        self.theta_x = (jnp.pi * jnp.arange(n_x) / self.N_x).astype(dtype)
        self.theta_y = (jnp.pi * jnp.arange(n_y) / self.N_y).astype(dtype)

        # 1-D Chebyshev-Lobatto nodes on [-1, 1], endpoints included
        self._nodes_x_ref = jnp.cos(self.theta_x)  # descending
        self._nodes_y_ref = jnp.cos(self.theta_y)

        # Map to physical domain: t in [-1,1] -> a + (b-a)*(1-t)/2
        self.nodes_x = (x_start + (x_end - x_start) * (1 - self._nodes_x_ref) / 2).astype(dtype)
        self.nodes_y = (y_start + (y_end - y_start) * (1 - self._nodes_y_ref) / 2).astype(dtype)

        # 2-D meshgrid (indexing='xy': shape (n_y, n_x))
        self.X, self.Y = jnp.meshgrid(self.nodes_x, self.nodes_y)

        self._norm_x = jnp.ones(self.n_x).at[0].set(2.0).at[self.N_x].set(2.0) * self.N_x
        self._norm_y = jnp.ones(self.n_y).at[0].set(2.0).at[self.N_y].set(2.0) * self.N_y

        # Clenshaw-Curtis quadrature weights (on physical domain)
        self.quadrature_weights_x = (_clenshaw_curtis_weights(n_x) * (x_end - x_start) / 2.0).astype(dtype)
        self.quadrature_weights_y = (_clenshaw_curtis_weights(n_y) * (y_end - y_start) / 2.0).astype(dtype)

    def interpolation_matrix(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the data evaluation matrix for the given data points (x, y).

        Entry (i, j) = T_{l}(y_ref_i) * T_{k}(x_ref_i)
        where j = l * n_x + k is the flattened index over the 2D
        Chebyshev basis {T_l(y) T_k(x)}, l=0..n_y-1, k=0..n_x-1.
        This matches the (n_y, n_x) layout from meshgrid with indexing='xy'.

        Parameters
        ----------
        x : (n_data,) array — physical x-coordinates of data points
        y : (n_data,) array — physical y-coordinates of data points

        Returns
        -------
        A : (n_data, n_y * n_x) array
        """
        # Map physical coords to reference [-1, 1]
        x_ref = 1.0 - 2.0 * (x - self.x_start) / (self.x_end - self.x_start)
        y_ref = 1.0 - 2.0 * (y - self.y_start) / (self.y_end - self.y_start)

        # Evaluate all 1D Chebyshev polynomials via recurrence:
        #   T_0(x) = 1,  T_1(x) = x,  T_{k+1}(x) = 2x T_k(x) - T_{k-1}(x)
        Tx = _cheb_eval_all(x_ref, self.n_x)   # (n_data, n_x)
        Ty = _cheb_eval_all(y_ref, self.n_y)   # (n_data, n_y)

        # Tensor-product basis: A_{i, l*n_x+k} = Ty_{i,l} * Tx_{i,k}
        # Matches C-order ravel of (n_y, n_x) coefficient array
        A = (Ty[:, :, None] * Tx[:, None, :]).reshape(-1, self.n_y * self.n_x)

        return A.astype(self.dtype)

    def compute_coeffs(self, f_grid: jnp.ndarray) -> jnp.ndarray:
        return dct1_nd(f_grid) / (self._norm_y[:, None] * self._norm_x[None, :])

    def eval_function(self, cheb_coeffs: jnp.ndarray) -> jnp.ndarray:
        return idct1_nd(cheb_coeffs * self._norm_y[:, None] * self._norm_x[None, :])

    def eval_gradient(self, cheb_coeffs: jnp.ndarray):
        """
        Differentiate in Chebyshev coefficient space via backward recurrence,
        then evaluate on grid via eval_function.

        Returns
        -------
        du_dx, du_dy : each (n_y, n_x) arrays of physical-space derivatives
        """

        # Differentiate along y (axis 0): vmap over columns
        dc_dx = self.du_dx_coeff(cheb_coeffs)
        dc_dy = self.du_dy_coeff(cheb_coeffs)

        du_dx = self.eval_function(dc_dx)
        du_dy = self.eval_function(dc_dy)

        return du_dx, du_dy

    def pad_coeffs(self, cheb_coeffs: jnp.ndarray, rho: int):
        """
        Pad the Chebyshev coefficients with zeros for a prescribed refinement factor.
        The old Nyquist modes (last row/col) are doubled because they move from
        endpoint (c_k=2) to interior (c_k=1) in the finer basis.
        """
        n_y, n_x = cheb_coeffs.shape
        padded = jnp.zeros((self.N_y * rho + 1, self.N_x * rho + 1))
        return padded.at[:n_y, :n_x].set(cheb_coeffs)

    def eval_laplacian(self, cheb_coeffs: jnp.ndarray):
        """
        Evaluate the Laplacian on the grid.
        """
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


def dct1_axis(x: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
    """
    Unnormalized DCT-I along `axis`.
    Input length must be n = N+1 >= 2.
    Returns length n along that axis.
    """
    x = jnp.asarray(x)
    n = x.shape[axis]
    if n < 2:
        return x
    N = n - 1

    # Build even extension of length 2N:
    # v = [x0..xN, x_{N-1}..x1]
    x_head = x
    idx = jnp.arange(1, N)  # 1..N-1
    x_mid = jnp.take(x, idx, axis=axis)
    x_tail = jnp.flip(x_mid, axis=axis)
    v = jnp.concatenate([x_head, x_tail], axis=axis)  # length 2N

    V = jnp.fft.rfft(v, axis=axis)  # length N+1 along axis
    y = jnp.real(V)                 # DCT-I coefficients (unnormalized)

    return y

def idct1_axis(y: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
    """
    Inverse of the unnormalized DCT-I above.
    For n = N+1, inverse is dct1_axis(y)/(2N).
    """
    y = jnp.asarray(y)
    n = y.shape[axis]
    if n < 2:
        return y
    N = n - 1
    return dct1_axis(y, axis=axis) / (2.0 * N)

def dct1_nd(x: jnp.ndarray, axes=None) -> jnp.ndarray:
    """
    Apply unnormalized DCT-I along each axis in `axes` (tensor-product ND DCT-I).
    If axes is None, transform along all axes.
    """
    x = jnp.asarray(x)
    if axes is None:
        axes = tuple(range(x.ndim))
    for ax in axes:
        x = dct1_axis(x, axis=ax)
    return x

def idct1_nd(y: jnp.ndarray, axes=None) -> jnp.ndarray:
    """
    Inverse of dct1_nd for the unnormalized convention above.
    """
    y = jnp.asarray(y)
    if axes is None:
        axes = tuple(range(y.ndim))
    for ax in axes:
        y = idct1_axis(y, axis=ax)
    return y


def _cheb_eval_all(x: jnp.ndarray, n: int) -> jnp.ndarray:
    """
    Evaluate T_0(x), T_1(x), ..., T_{n-1}(x) via the three-term recurrence.

    Parameters
    ----------
    x : (m,) array — evaluation points in [-1, 1]
    n : int        — number of Chebyshev polynomials (0 through n-1)

    Returns
    -------
    T : (m, n) array — T[i, k] = T_k(x_i)
    """
    def body(carry, _):
        T_prev, T_curr = carry
        T_next = 2.0 * x * T_curr - T_prev
        return (T_curr, T_next), T_next

    T0 = jnp.ones_like(x)
    T1 = x
    if n == 1:
        return T0[:, None]
    if n == 2:
        return jnp.stack([T0, T1], axis=-1)

    # scan over k = 2, 3, ..., n-1
    _, T_rest = jax.lax.scan(body, (T0, T1), None, length=n - 2)
    # T_rest shape: (n-2, m) — stack with T0, T1
    return jnp.concatenate([T0[None, :], T1[None, :], T_rest], axis=0).T


def _clenshaw_curtis_weights(n: int) -> jnp.ndarray:
    """
    Clenshaw-Curtis quadrature weights for n Chebyshev-Lobatto points on [-1, 1].
    Points: x_j = cos(j*pi/(n-1)), j = 0, ..., n-1.

    Uses DCT-I:  w_j = DCT-I(mu)_j / (N * c_j),
    where mu_k = int_{-1}^{1} T_k(x) dx = 2/(1-k^2) for even k, 0 for odd k,
    and c_0 = c_N = 2, c_j = 1 otherwise.
    """
    N = n - 1  # n points, polynomial degree N

    # Chebyshev moments (avoid div-by-zero for odd k)
    k = jnp.arange(n)
    safe_denom = jnp.where(k % 2 == 0, 1.0 - k ** 2, 1.0)
    mu = jnp.where(k % 2 == 0, 2.0 / safe_denom, 0.0)

    # Invert the cosine sum via DCT-I
    dct_mu = dct1_axis(mu)

    # Endpoint scaling factors
    c = jnp.ones(n).at[0].set(2.0).at[N].set(2.0)

    return dct_mu / (N * c)

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
    # b_rest = [b_{N-2}, b_{N-3}, ..., b_0]
    b = jnp.concatenate([jnp.flip(b_rest), jnp.array([b_Nm1, b_N])])
    return b



if __name__ == "__main__":
    import time

    u = lambda x, y: jnp.exp(x) * jnp.exp(y)
    du_dx_exact = lambda x, y: jnp.exp(x) * jnp.exp(y)
    du_dy_exact = lambda x, y: jnp.exp(x) * jnp.exp(y)

    t0 = time.perf_counter()
    grid = ChebyshevGrid2D(0.0, 1.0, 0.0, 1.0, 16, 24)
    t1 = time.perf_counter()
    print(f"Grid construction:      {t1 - t0:.6f} s")

    t0 = time.perf_counter()
    cheb_coeffs = dct1_nd(u(grid.X, grid.Y)) / (grid._norm_y[:, None] * grid._norm_x[None, :])
    t1 = time.perf_counter()
    print(f"Forward DCT (coeffs):   {t1 - t0:.6f} s")

    t0 = time.perf_counter()
    u_recovered = grid.eval_function(cheb_coeffs)
    u_recovered.block_until_ready()
    t1 = time.perf_counter()
    print(f"eval_function:          {t1 - t0:.6f} s  | max err = {jnp.max(jnp.abs(u_recovered - u(grid.X, grid.Y))):.2e}")

    t0 = time.perf_counter()
    grid_fine = ChebyshevGrid2D(0.0, 1.0, 0.0, 1.0, 32, 64)
    t1 = time.perf_counter()
    print(f"Fine grid construction: {t1 - t0:.6f} s")

    t0 = time.perf_counter()
    A = grid.interpolation_matrix(grid_fine.X.ravel(), grid_fine.Y.ravel())
    A.block_until_ready()
    t1 = time.perf_counter()
    print(f"interpolation_matrix:   {t1 - t0:.6f} s")

    t0 = time.perf_counter()
    u_fine = A @ cheb_coeffs.ravel()
    u_fine.block_until_ready()
    t1 = time.perf_counter()
    print(f"A @ coeffs (interp):    {t1 - t0:.6f} s  | max err = {jnp.max(jnp.abs(u_fine.reshape(grid_fine.X.shape) - u(grid_fine.X, grid_fine.Y))):.2e}")

    t0 = time.perf_counter()
    du_dx_recovered, du_dy_recovered = grid.eval_gradient(cheb_coeffs)
    du_dx_recovered.block_until_ready()
    t1 = time.perf_counter()
    print(f"eval_gradient:          {t1 - t0:.6f} s  | max err dx = {jnp.max(jnp.abs(du_dx_recovered - du_dx_exact(grid.X, grid.Y))):.2e}, dy = {jnp.max(jnp.abs(du_dy_recovered - du_dy_exact(grid.X, grid.Y))):.2e}")