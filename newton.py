import jax
import numpy as np

def optimize(start, fun):
    """
    Find a stationary point (local minimum, maximum, or saddle point)
    of a function using Newton's method.

    Starting from an initial guess, iteratively updates x according to
        x_new = x_old - fun'(x_old) / fun''(x_old)
    using finite-difference approximations of the first and second
    derivatives, until successive estimates differ by less than 1e-6.

    Parameters
    ----------
    start : float
        Initial guess for the location of the stationary point.
    fun : callable
        A function of a single variable, fun(x), to optimize.

    Returns
    -------
    float
        The x-value at which fun is (approximately) stationary,
        i.e. where fun'(x) ≈ 0.

    Notes
    -----
    - Convergence is not guaranteed; it depends on the starting point
      and the shape of fun (standard caveat of Newton's method).
    - No check is made for a zero or near-zero second derivative,
      which could cause a division by zero or numerical instability.
    """
    grad_f = jax.grad(fun)
    hess_f = jax.hessian(fun)
    x_old = start
    x_new = x_old - np.linalg.solve(hess_f(x_old), grad_f(x_old))
    while np.linalg.norm(x_old - x_new, ord=1) > 1e-6:
        x_old = x_new
        x_new = x_old - np.linalg.solve(hess_f(x_old), grad_f(x_old))
    return x_new