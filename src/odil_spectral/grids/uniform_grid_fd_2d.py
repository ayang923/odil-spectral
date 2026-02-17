import jax.numpy as jnp
import jax

class UniformGridFD2D:
    """Uniform finite-difference grid on [x_start, x_end] x [y_start, y_end]."""

    def __init__(self, x_start: float, x_end: float,
                 y_start: float, y_end: float,
                 n_x: int, n_y: int, L: int, dtype=jnp.float64):
        self.x_start = x_start
        self.x_end = x_end
        self.y_start = y_start
        self.y_end = y_end
        self.n_x = n_x
        self.n_y = n_y
        self.hx = (x_end - x_start) / n_x
        self.hy = (y_end - y_start) / n_y
        
        self.L = L
        self.dtype = dtype

        self.n_x_levels = [n_x // (2 ** (L - 1 - ell)) for ell in range(L)]
        self.n_y_levels = [n_y // (2 ** (L - 1 - ell)) for ell in range(L)]

        # Uniform grid nodes (n+1 points, including endpoints)
        self.nodes_x = jnp.linspace(x_start, x_end, n_x+1)
        self.nodes_y = jnp.linspace(y_start, y_end, n_y+1)

        # 2-D meshgrid on the full grid (n_y+1, n_x+1)
        self.X, self.Y = jnp.meshgrid(self.nodes_x, self.nodes_y)

    def reconstruct_function(self, function_levels):
        """Reconstruct full grid from multi-level corrections."""
        return _reconstruct_function(function_levels, self.n_x_levels, self.n_y_levels)

    def eval_gradient(self, f_grid: jnp.ndarray):
        """
        Gradient on the full node grid.

        Input:  f_grid of shape (n_y+1, n_x+1).
        Output: du_dx, du_dy of shape (n_y+1, n_x+1).

        Uses central differences at interior nodes. forward and backward differences at boundaries.
        """
        # du/dx
        du_dx_interior = (f_grid[:, 2:] - f_grid[:, :-2]) / (2*self.hx)
        du_dx_left = (f_grid[:, 1:2] - f_grid[:, 0:1]) / self.hx
        du_dx_right = (f_grid[:, -1:] - f_grid[:, -2:-1]) / self.hx
        du_dx = jnp.concatenate([du_dx_left, du_dx_interior, du_dx_right], axis=1)

        # du/dy
        du_dy_interior = (f_grid[2:, :] - f_grid[:-2, :]) / (2.0 * self.hy)
        du_dy_bottom = (f_grid[1:2, :] - f_grid[0:1, :]) / self.hy
        du_dy_top = (f_grid[-1:, :] - f_grid[-2:-1, :]) / self.hy
        du_dy = jnp.concatenate([du_dy_bottom, du_dy_interior, du_dy_top], axis=0)

        return du_dx, du_dy

    def make_trapezoid_quadrature_weights(self):
        """
        Trapezoid quadrature weights on the full node grid.
        """
        wx = jnp.ones(self.n_x + 1).at[0].set(0.5).at[self.n_x].set(0.5) * self.hx
        wy = jnp.ones(self.n_y + 1).at[0].set(0.5).at[self.n_y].set(0.5) * self.hy
        return wx, wy


def _prolong(u, n_x_fine, n_y_fine):
    """Linearly interpolate to a finer grid: (n_y+1, n_x+1)->(n_y_fine+1, n_x_fine+1)."""
    return jax.image.resize(u, (n_y_fine+1, n_x_fine+1), method="linear")

def _reconstruct_function(function_levels, n_x_levels, n_y_levels):
    """
    function_levels: tuple/list length L
                u_levels[i] is a flat array for level i (coarse->fine),
                shapes correspond to (n_y_levels[i]+1, n_x_levels[i]+1) when reshaped.
    """
    # Start from coarsest
    f_grid = function_levels[0].reshape((n_y_levels[0] + 1, n_x_levels[0] + 1))

    for i in range(1, len(n_x_levels)):
        f_grid = _prolong(f_grid, n_x_levels[i], n_y_levels[i])
        f_grid_i = function_levels[i].reshape((n_y_levels[i] + 1, n_x_levels[i] + 1))
        f_grid += f_grid_i
    return f_grid

if __name__ == "__main__":
    import time

    u = lambda x, y: jnp.exp(x) * jnp.exp(3*y)
    du_dx_exact = lambda x, y: jnp.exp(x) * jnp.exp(3*y)
    du_dy_exact = lambda x, y: 3*jnp.exp(x) * jnp.exp(3*y)

    for n in [16, 32, 64, 128, 256, 512]:
        grid = UniformGridFD2D(0.0, 1.0, 0.0, 1.0, n, n, 1)
        grad_x, grad_y = grid.eval_gradient(u(grid.X, grid.Y))
        err_x = jnp.max(jnp.abs(grad_x - du_dx_exact(grid.X, grid.Y)))
        err_y = jnp.max(jnp.abs(grad_y - du_dy_exact(grid.X, grid.Y)))
        print(f"n={n:4d}  |  max|du/dx err| = {err_x:.6e}  |  max|du/dy err| = {err_y:.6e}")

