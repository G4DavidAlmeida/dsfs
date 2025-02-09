
import tqdm
import random

from typing import TypeVar, Callable, List, Tuple

from scratch.probability import normal_cdf
from scratch.gradient_descent import gradient_step
from scratch.linear_algebra import add, vector_mean, dot, Vector
from scratch.simple_linear_regression import total_sum_of_squares


def predict(x: Vector, beta: Vector) -> float:
    return dot(x, beta)


def error(x: Vector, y: float, beta: Vector) -> float:
    return predict(x, beta) - y


def squared_error(x: Vector, y: float, beta: Vector) -> float:
    return error(x, y, beta) ** 2


def sqerror_gradient(x: Vector, y: float, beta: Vector) -> Vector:
    err = error(x, y, beta)
    return [2 * err * x_i for x_i in x]


def least_squares_fit(xs: List[Vector], ys: List[float], learning_rate: float = 0.001, num_steps: int = 1000, batch_size: int = 1) -> Vector:
    # Start with a random guess
    guess = [random.random() for _ in xs[0]]
    for _ in tqdm.trange(num_steps, desc="least squares fit"):
        for start in range(0, len(xs), batch_size):
            batch_xs = xs[start:start+batch_size]
            batch_ys = ys[start:start+batch_size]

            gradient = vector_mean([sqerror_gradient(x, y, guess)
                                   for x, y in zip(batch_xs, batch_ys)])
            guess = gradient_step(guess, gradient, -learning_rate)
    return guess


def multiple_r_squared(xs: List[Vector], ys: Vector, beta: Vector) -> float:
    sum_of_squared_errors = sum(error(x, y, beta) ** 2 for x, y in zip(xs, ys))
    return 1.0 - sum_of_squared_errors / total_sum_of_squares(ys)


X = TypeVar('X')  # generic type for data
Stat = TypeVar('Stat')  # generic type for "summary statistic"


def bootstrap_sample(data: List[X]) -> List[X]:
    """randomly samples len(data) elements with replacement"""
    return [random.choice(data) for _ in data]


def bootstrap_statistic(data: List[X], stats_fn: Callable[[List[X]], Stat], num_samples: int) -> List[Stat]:
    """evaluates stats_fn on num_samples bootstrap samples from data"""
    return [stats_fn(bootstrap_sample(data)) for _ in range(num_samples)]


def estimate_sample_beta(pairs: List[Tuple[Vector, float]], learning_rate: float = 0.001, num_samples: int = 100, batch_size: int = 25) -> Vector:
    x_sample = [x for x, _ in pairs]
    y_sample = [y for _, y in pairs]
    beta = least_squares_fit(
        x_sample, y_sample, learning_rate, num_samples, batch_size)
    return beta


def p_value(beta_hat_j: float, sigma_hat_j: float) -> float:
    if beta_hat_j > 0:
        # if the coefficient is positive, we need to compute twice the probability of seeing an even *larger* value
        return 2 * (1 - normal_cdf(beta_hat_j / sigma_hat_j))
    else:
        # otherwise twice the probability of seeing a *smaller* value
        return 2 * normal_cdf(beta_hat_j / sigma_hat_j)


assert p_value(30.58, 1.27) < 0.001  # constant term
assert p_value(0.972, 0.103) < 0.001  # num_friends
assert p_value(-1.865, 0.155) < 0.001  # work_hours
assert p_value(0.923, 1.249) > 0.4    # phd


# alpha is a *hyperparameter* controlling how harsh the penalty is.
# Sometimes it's called "lambda" but that already means something in Python
def ridge_penalty(beta: Vector, alpha: float) -> float:
    return alpha * dot(beta[1:], beta[1:])


def squared_error_ridge(x: Vector, y: float, beta: Vector, alpha: float) -> float:
    """estimate error plus ridge penalty on beta"""
    return error(x, y, beta) ** 2 + ridge_penalty(beta, alpha)


def ridge_penalty_gradient(beta: Vector, alpha: float) -> Vector:
    """gradient of just the ridge penalty"""
    return [0.] + [2 * alpha * beta_j for beta_j in beta[1:]]


def sqerror_ridge_gradient(x: Vector, y: float, beta: Vector, alpha: float) -> Vector:
    """the gradient corresponding to the ith squared error term including the ridge penalty"""
    return add(sqerror_gradient(x, y, beta), ridge_penalty_gradient(beta, alpha))


def least_squares_fit_ridget(xs: List[Vector], ys: List[float], learning_rate: float = 0.001, num_steps: int = 1000, batch_size: int = 1, alpha: float = 0.0) -> Vector:
    # Start with a random guess
    guess = [random.random() for _ in xs[0]]
    for _ in tqdm.trange(num_steps, desc="least squares fit"):
        for start in range(0, len(xs), batch_size):
            batch_xs = xs[start:start+batch_size]
            batch_ys = ys[start:start+batch_size]

            gradient = vector_mean([sqerror_ridge_gradient(
                x, y, guess, alpha) for x, y in zip(batch_xs, batch_ys)])
            guess = gradient_step(guess, gradient, -learning_rate)
    return guess


def lasso_penalty(beta, alpha):
    return alpha * sum(abs(beta_i) for beta_i in beta[1:])
