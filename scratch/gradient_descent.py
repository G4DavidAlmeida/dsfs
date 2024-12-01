import random
from typing import TypeVar, List, Iterator, Callable
from scratch.linear_algebra import add, scalar_multiply
from scratch.linear_algebra import Vector

T = TypeVar('T')  # this allows us to type "generic" functions

def difference_quotient(f: Callable[[float], float], x: float, h: float) -> float:
    """
    Calculate the difference quotient of a function at a given point.
    This function calculates the difference quotient of a function `f` at a given
    point `x`. The difference quotient is an approximation of the derivative of
    the function at that point.
    Parameters:
    -----------
    f : Callable[[float], float]
        The function for which to calculate the difference quotient.
    x : float
        The point at which to calculate the difference quotient.
    h : float
        The small value by which to perturb `x`.
    Returns:
    --------
    float:
        The difference quotient of the function at the given point.
    """
    return (f(x + h) - f(x)) / h


def partial_difference_quotient(f: Callable[[float], float], v: List[float], i: int, h: float) -> float:
    """
    Calculate the partial difference quotient of a function at a given point.
    This function calculates the partial difference quotient of a function `f` at a given
    point `v`. The partial difference quotient is an approximation of the partial derivative
    of the function at that point with respect to the `i`-th variable.
    Parameters:
    -----------
    f : Callable[[List[float]], float]
        The function for which to calculate the partial difference quotient.
    v : List[float]
        The point at which to calculate the partial difference quotient.
    i : int
        The index of the variable with respect to which to calculate the partial derivative.
    h : float
        The small value by which to perturb the `i`-th variable.
    Returns:
    --------
    float:
        The partial difference quotient of the function at the given point.
    """
    w = [v_j + (h if j == i else 0) for j, v_j in enumerate(v)]
    return (f(w) - f(v)) / h

# %%


def estimate_gradient(f: Callable[[List[float]], float], v: List[float], h: float = 0.00001) -> List[float]:
    """
    Estimate the gradient of a function at a given point.
    This function estimates the gradient of a function `f` at a given point `v`. The gradient
    is a vector of partial derivatives of the function at that point.
    Parameters:
    -----------
    f : Callable[[List[float]], float]
        The function for which to calculate the gradient.
    v : List[float]
        The point at which to calculate the gradient.
    h : float
        The small value by which to perturb the variables.
    Returns:
    --------
    List[float]:
        The gradient of the function at the given point.
    """
    return [partial_difference_quotient(f, v, i, h) for i in range(len(v))]


# %%


def gradient_step(v: List[float], gradient: List[float], step_size: float) -> List[float]:
    """
    Take a step in the direction of the gradient.
    This function takes a step in the direction of the gradient of a function `f` at a given
    point `v`. The step size is determined by the `step_size` parameter.
    Parameters:
    -----------
    v : List[float]
        The point at which to take a step.
    gradient : List[float]
        The gradient of the function at the given point.
    step_size : float
        The size of the step to take.
    Returns:
    --------
    List[float]:
        The new point after taking a step in the direction of the gradient.
    """
    step = scalar_multiply(step_size, gradient)
    return add(v, step)


def sum_of_squares_gradient(v: List[float]) -> List[float]:
    """
    Calculate the gradient of the sum of squares function.
    This function calculates the gradient of the sum of squares function at a given point `v`.
    The sum of squares function is a simple quadratic function that has a minimum at the origin.
    Parameters:
    -----------
    v : List[float]
        The point at which to calculate the gradient.
    Returns:
    --------
    List[float]:
        The gradient of the sum of squares function at the given point.
    """
    return [2 * v_i for v_i in v]


def linear_gradient(x: float, y: float, theta: Vector) -> Vector:
    """
    Calculate the gradient of the linear regression model.
    This function calculates the gradient of the linear regression model at a given point `(x, y)`.
    The linear regression model is parameterized by the vector `theta`.
    Parameters:
    -----------
    x : float
        The input feature.
    y : float
        The observed output.
    theta : List[float]
        The parameters of the linear regression model.
    Returns:
    --------
    List[float]:
        The gradient of the linear regression model at the given point.
    """
    slope, intercept = theta
    predicted = slope * x + intercept
    error = predicted - y
    # squared_error = error ** 2
    grad = [2 * error * x, 2 * error]
    return grad

def minibatches(dataset: List[T],
                batch_size: int,
                shuffle: bool = True) -> Iterator[List[T]]:
    """Gererate `batch_size`-sized minibatches from the dataset"""
    # start indexes 0, batch_size, 2 * batch_size, ...
    batch_starts = [start for start in range(0, len(dataset), batch_size)]

    if shuffle:
        random.shuffle(batch_starts)  # shuffle the batches

    for start in batch_starts:
        end = start + batch_size
        yield dataset[start:end]
