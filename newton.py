def derivative(x, fun):
    """
    Approximate the first derivative of a function at a point using
    the forward difference method.

    Parameters
    ----------
    x : float
        The point at which to evaluate the derivative.
    fun : callable
        A function of a single variable, fun(x), to differentiate.

    Returns
    -------
    float
        An approximation of fun'(x), computed as
        (fun(x + h) - fun(x)) / h, with h = 1e-6.
    """
    result = (fun(x + 1e-6) - fun(x))/1e-6
    return result


def second_derivative(x, fun):
    """
    Approximate the second derivative of a function at a point using
    the central difference method.

    Parameters
    ----------
    x : float
        The point at which to evaluate the second derivative.
    fun : callable
        A function of a single variable, fun(x), to differentiate.

    Returns
    -------
    float
        An approximation of fun''(x), computed as
        (fun(x + h) - 2*fun(x) + fun(x - h)) / h^2, with h = 1e-6.
    """
    result = (fun(x + 1e-6) - 2 * fun(x) + fun(x - 1e-6)) / ((1e-6) ** 2)
    return result


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
    x_old = start
    x_new = x_old - derivative(x_old, fun) / second_derivative(x_old, fun)
    while abs(x_new - x_old) > 1e-6:
        x_old = x_new
        x_new = x_old - derivative(x_old, fun) / second_derivative(x_old, fun)
    return x_new