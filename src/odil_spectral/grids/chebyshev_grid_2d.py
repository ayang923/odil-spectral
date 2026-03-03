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

    def make_evaluation_operator(self, x_eval, y_eval):
        x_ref = 1.0 - 2.0 * (x_eval - self.x_start) / (self.x_end - self.x_start)
        y_ref = 1.0 - 2.0 * (y_eval - self.y_start) / (self.y_end - self.y_start)

        def barycentric_weights_chebyshev(n):
            w = jnp.ones(n)
            w = w.at[0].set(0.5)
            w = w.at[-1].set(0.5)
            signs = jnp.array([(-1)**i for i in range(n)])
            return w * signs

        w1 = barycentric_weights_chebyshev(len(self._nodes_x_ref))  # x-weights (n_x,)
        w2 = barycentric_weights_chebyshev(len(self._nodes_y_ref))  # y-weights (n_y,)

        def barycentric_interpolate(x_nodes, f_nodes, x_eval, weights):
            # x_eval: (p,), x_nodes: (n,), f_nodes: (n,)
            diff = x_eval[:, None] - x_nodes[None, :]
            exact = jnp.abs(diff) < 1e-12
            diff = jnp.where(exact, 1.0, diff)
            w_over_d = weights[None, :] / diff
            numerator = w_over_d @ f_nodes
            denominator = jnp.sum(w_over_d, axis=1)
            result = numerator / denominator
            exact_idx = jnp.argmax(exact, axis=1)
            hit_any = jnp.any(exact, axis=1)
            return jnp.where(hit_any, f_nodes[exact_idx], result)

        def evaluation_operator(val_cheb_grid):
            """
            val_cheb_grid: (n_y, n_x)
            returns: (p,) values at paired points (x_ref[k], y_ref[k])
            """
            # 1) Interpolate in y for each x-column: (n_y,n_x) -> (p,n_x)
            # vmap over x-columns (axis=1)
            f_y = jax.vmap(
                lambda col: barycentric_interpolate(self._nodes_y_ref, col, y_ref, w2),
                in_axes=1,   # take val_cheb_grid[:, j]
                out_axes=1
            )(val_cheb_grid)  # (p, n_x)

            # 2) For each paired point k, interpolate in x along row f_y[k, :]
            result = jax.vmap(
                lambda row_k, xk: barycentric_interpolate(self._nodes_x_ref, row_k, jnp.array([xk]), w1)[0],
                in_axes=(0, 0)
            )(f_y, x_ref)  # (p,)

            return result

        return evaluation_operator

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

    def pad_coeffs(self, cheb_coeffs, n_y_new, n_x_new):
        assert(n_x_new >= self.n_x and n_y_new >= self.n_y)
        padded = jnp.zeros((n_y_new, n_x_new)).at[:self.n_y, :self.n_x].set(cheb_coeffs)

        return padded
    
    def pad_coeffs_rho(self, cheb_coeffs: jnp.ndarray, rho: int):
        """
        Pad the Chebyshev coefficients with zeros for a prescribed refinement factor.
        The old Nyquist modes (last row/col) are doubled because they move from
        endpoint (c_k=2) to interior (c_k=1) in the finer basis.
        """
        return self.pad_coeffs(cheb_coeffs, self.N_y * rho + 1, self.N_x * rho + 1)

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

class ChebyshevGrid1D:
    """Chebyshev grid on [x_start, x_end]."""

    def __init__(self, x_start: float, x_end: float, n_x: int, dtype=jnp.float64):
        self.x_start = x_start
        self.x_end = x_end
        self.n_x = n_x
        self.dtype = dtype

        self.N_x = n_x - 1

        # Uniform theta grids (Chebyshev-Lobatto): k = 0, 1, ..., N
        self.theta_x = (jnp.pi * jnp.arange(n_x) / self.N_x).astype(dtype)

        # 1-D Chebyshev-Lobatto nodes on [-1, 1], endpoints included
        self._nodes_x_ref = jnp.cos(self.theta_x)  # descending

        # Map to physical domain: t in [-1,1] -> a + (b-a)*(1-t)/2
        self.nodes_x = (x_start + (x_end - x_start) * (1 - self._nodes_x_ref) / 2).astype(dtype)

        self._norm_x = jnp.ones(self.n_x).at[0].set(2.0).at[self.N_x].set(2.0) * self.N_x

        # Clenshaw-Curtis quadrature weights (on physical domain)
        self.quadrature_weights_x = (_clenshaw_curtis_weights(n_x) * (x_end - x_start) / 2.0).astype(dtype)

    def interpolation_matrix(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the data evaluation matrix for the given data points (x, y).

        Entry (i, j) = T_{l}(y_ref_i) * T_{k}(x_ref_i)
        where j = l * n_x + k is the flattened index over the 2D
        Chebyshev basis {T_l(y) T_k(x)}, l=0..n_y-1, k=0..n_x-1.
        This matches the (n_y, n_x) layout from meshgrid with indexing='xy'.

        Parameters
        ----------
        x : (n_data,) array — physical x-coordinates of data points

        Returns
        -------
        A : (n_data, n_x) array
        """
        # Map physical coords to reference [-1, 1]
        x_ref = 1.0 - 2.0 * (x - self.x_start) / (self.x_end - self.x_start)

        # Evaluate all 1D Chebyshev polynomials via recurrence:
        #   T_0(x) = 1,  T_1(x) = x,  T_{k+1}(x) = 2x T_k(x) - T_{k-1}(x)
        Tx = _cheb_eval_all(x_ref, self.n_x)   # (n_data, n_x)

        A = Tx

        return A.astype(self.dtype)

    def compute_coeffs(self, f_grid: jnp.ndarray) -> jnp.ndarray:
        return dct1_nd(f_grid) / self._norm_x

    def eval_function(self, cheb_coeffs: jnp.ndarray) -> jnp.ndarray:
        return idct1_nd(cheb_coeffs * self._norm_x)

    def eval_gradient(self, cheb_coeffs: jnp.ndarray):
        """
        Differentiate in Chebyshev coefficient space via backward recurrence,
        then evaluate on grid via eval_function.

        Returns
        -------
        du_dx : (n_x) array of physical-space derivatives
        """

        # Differentiate along y (axis 0): vmap over columns
        dc_dx = self.du_dx_coeff(cheb_coeffs)
        du_dx = self.eval_function(dc_dx)

        return du_dx

    def pad_coeffs(self, cheb_coeffs: jnp.ndarray, rho: int):
        """
        Pad the Chebyshev coefficients with zeros for a prescribed refinement factor.
        The old Nyquist mode (last element) is doubled because it moves from
        endpoint (c_k=2) to interior (c_k=1) in the finer basis.
        """
        n_x = cheb_coeffs.shape[0]
        padded = jnp.zeros((self.N_x * rho + 1))
        
        # Copy all coefficients first
        padded = padded.at[:n_x].set(cheb_coeffs)
        
        return padded

    def eval_laplacian(self, cheb_coeffs: jnp.ndarray):
        """
        Evaluate the Laplacian on the grid.
        """
        d2c_dx2 = self.du_dx_coeff(self.du_dx_coeff(cheb_coeffs))

        return self.eval_function(d2c_dx2)

    def du_dx_coeff(self, cheb_coeffs: jnp.ndarray):
        """
        Differentiate Chebyshev coefficients along x, including
        the reference-to-physical scaling: dt/dx = -2/(x_end - x_start).
        """
        return _cheb_diff_1d(cheb_coeffs) * (-2.0 / (self.x_end - self.x_start))

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
    jax.config.update("jax_enable_x64", True)

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

    print("\n" + "="*60)
    print("Test 1: Padding a function")
    print("="*60)
    
    # Simple polynomial function: u(x, y) = x^2 + y^2
    u_test = lambda x, y: x**2 + y**2
    
    # Coarse grid
    grid_coarse = ChebyshevGrid2D(0.0, 1.0, 0.0, 1.0, 8, 8)
    u_coarse = u_test(grid_coarse.X, grid_coarse.Y)
    coeffs_coarse = grid_coarse.compute_coeffs(u_coarse)
    
    # Fine grid
    grid_fine_test = ChebyshevGrid2D(0.0, 1.0, 0.0, 1.0, 16, 16)
    u_fine_exact = u_test(grid_fine_test.X, grid_fine_test.Y)
    
    # Pad coefficients and evaluate
    coeffs_padded = grid_coarse.pad_coeffs(coeffs_coarse, grid_fine_test.n_y, grid_fine_test.n_x)
    u_fine_padded = grid_fine_test.eval_function(coeffs_padded)
    
    error = jnp.max(jnp.abs(u_fine_padded - u_fine_exact))
    print(f"Max error after padding: {error:.2e}")
    print(f"Test {'PASSED' if error < 1e-10 else 'FAILED'}")
    
    print("\n" + "="*60)
    print("Test 2: Commutativity of padding and differentiation")
    print("="*60)
    
    # Use a simple function: u(x, y) = x^3 + y^3
    u_test2 = lambda x, y: x**3 + y**3
    du_dx_exact2 = lambda x, y: 3 * x**2
    du_dy_exact2 = lambda x, y: 3 * y**2
    
    # Coarse and fine grids
    grid_c = ChebyshevGrid2D(0.0, 1.0, 0.0, 1.0, 12, 8)
    grid_f = ChebyshevGrid2D(0.0, 1.0, 0.0, 1.0, 24, 16)
    
    # Compute coefficients on coarse grid
    u_c = u_test2(grid_c.X, grid_c.Y)
    coeffs_c = grid_c.compute_coeffs(u_c)
    
    # Method 1: Differentiate then pad
    dc_dx_c, dc_dy_c = grid_c.du_dx_coeff(coeffs_c), grid_c.du_dy_coeff(coeffs_c)
    dc_dx_padded = grid_c.pad_coeffs(dc_dx_c, grid_f.n_y, grid_f.n_x)
    dc_dy_padded = grid_c.pad_coeffs(dc_dy_c, grid_f.n_y, grid_f.n_x)
    du_dx_method1 = grid_f.eval_function(dc_dx_padded)
    du_dy_method1 = grid_f.eval_function(dc_dy_padded)
    
    # Method 2: Pad then differentiate
    coeffs_padded2 = grid_c.pad_coeffs(coeffs_c, grid_f.n_y, grid_f.n_x)
    dc_dx_padded2, dc_dy_padded2 = grid_f.du_dx_coeff(coeffs_padded2), grid_f.du_dy_coeff(coeffs_padded2)
    du_dx_method2 = grid_f.eval_function(dc_dx_padded2)
    du_dy_method2 = grid_f.eval_function(dc_dy_padded2)
    
    # Compare the two methods
    error_dx = jnp.max(jnp.abs(du_dx_method1 - du_dx_method2))
    error_dy = jnp.max(jnp.abs(du_dy_method1 - du_dy_method2))
    
    print(f"Max error in du/dx: {error_dx:.2e}")
    print(f"Max error in du/dy: {error_dy:.2e}")
    print(f"Test {'PASSED' if error_dx < 1e-10 and error_dy < 1e-10 else 'FAILED'}")
    
    # Also compare with exact solution on fine grid
    du_dx_exact_f = du_dx_exact2(grid_f.X, grid_f.Y)
    du_dy_exact_f = du_dy_exact2(grid_f.X, grid_f.Y)
    
    error_exact_dx = jnp.max(jnp.abs(du_dx_method1 - du_dx_exact_f))
    error_exact_dy = jnp.max(jnp.abs(du_dy_method1 - du_dy_exact_f))
    
    print(f"\nComparison with exact solution:")
    print(f"Max error in du/dx vs exact: {error_exact_dx:.2e}")
    print(f"Max error in du/dy vs exact: {error_exact_dy:.2e}")

    
