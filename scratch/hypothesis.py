
from typing import List
import random
from scratch.probability import inverse_normal_cdf
from scratch.probability import normal_cdf
import math
from typing import Tuple


def normal_approximation_to_binomimal(n: int, p: float) -> Tuple[float, float]:
    """
    Calculates the mean (mu) and standard deviation (sigma) of a binomial distribution
    approximated by a normal distribution.
    This function takes the number of trials (n) and the probability of success (p) 
    in a binomial distribution and returns the parameters of the corresponding normal 
    distribution approximation.
    Args:
        n (int): The number of trials in the binomial distribution.
        p (float): The probability of success in each trial.
    Returns:
        Tuple[float, float]: A tuple containing:
            - mu (float): The mean of the approximated normal distribution.
            - sigma (float): The standard deviation of the approximated normal distribution.
    Detailed Explanation:
        In a binomial distribution, the mean (mu) is calculated as the product of the 
        number of trials (n) and the probability of success (p). The standard deviation 
        (sigma) is calculated as the square root of the product of the number of trials (n), 
        the probability of success (p), and the probability of failure (1 - p). This function 
        uses these formulas to approximate the binomial distribution with a normal distribution.
    """

    mu = p * n
    sigma = math.sqrt(p * (1 - p) * n)
    return mu, sigma


# The normal cdf _is_ the probability the variable is below a threshold
normal_probability_below = normal_cdf

# It's above the threshold if it's not below the threshold


def normal_probability_above(lo: float, mu: float = 0, sigma: float = 1) -> float:
    """
    Calculates the probability that a value from a normal distribution is above a given threshold.
    This function computes the probability that a value drawn from a normal distribution with 
    mean `mu` and standard deviation `sigma` is greater than the specified lower bound `lo`.
    Parameters:
    -----------
    lo : float
        The lower bound threshold value.
    mu : float, optional
        The mean of the normal distribution (default is 0).
    sigma : float, optional
        The standard deviation of the normal distribution (default is 1).
    Returns:
    --------
    float
        The probability that a value from the specified normal distribution is greater than `lo`.
    Detailed Explanation:
    ---------------------
    The function uses the cumulative distribution function (CDF) of the normal distribution to 
    calculate the probability that a value is less than or equal to `lo`. Since the total probability 
    is 1, the probability that a value is greater than `lo` is given by `1 - CDF(lo)`.
    """

    return 1 - normal_cdf(lo, mu, sigma)

# It's between if it's less than hi, but not less than lo


def normal_probability_between(lo: float, hi: float, mu: float = 0, sigma: float = 1) -> float:
    """
    Calculates the probability that a normally distributed random variable falls between two values.
    This function computes the probability that a value from a normal distribution with mean `mu` and 
    standard deviation `sigma` lies between `lo` and `hi`.
    Parameters:
    lo (float): The lower bound of the interval.
    hi (float): The upper bound of the interval.
    mu (float, optional): The mean of the normal distribution. Default is 0.
    sigma (float, optional): The standard deviation of the normal distribution. Default is 1.
    Returns:
    float: The probability that a value from the specified normal distribution falls between `lo` and `hi`.
    Detailed Explanation:
    The function uses the cumulative distribution function (CDF) of the normal distribution to calculate 
    the probability. The CDF, `normal_cdf`, gives the probability that a normally distributed random variable 
    is less than or equal to a given value. By subtracting the CDF value at `lo` from the CDF value at `hi`, 
    the function determines the probability that the random variable falls within the interval `[lo, hi]`.
    """

    return normal_cdf(hi, mu, sigma) - normal_cdf(lo, mu, sigma)

# It's outside if it's not between


def normal_probability_outside(lo: float, hi: float, mu: float = 0, sigma: float = 1) -> float:
    """
    Calculate the probability that a normally distributed random variable falls outside the interval [lo, hi].
    This function computes the probability that a value drawn from a normal distribution with mean `mu` and standard deviation `sigma` lies outside the range defined by `lo` and `hi`.
    Parameters:
    lo (float): The lower bound of the interval.
    hi (float): The upper bound of the interval.
    mu (float, optional): The mean of the normal distribution. Default is 0.
    sigma (float, optional): The standard deviation of the normal distribution. Default is 1.
    Returns:
    float: The probability that a value lies outside the interval [lo, hi].
    Detailed Explanation:
    The function calculates the probability that a normally distributed random variable with specified mean (`mu`) and standard deviation (`sigma`) falls outside the interval [lo, hi]. It does this by subtracting the probability that the variable falls within the interval from 1. This is useful in statistical hypothesis testing and other applications where the likelihood of extreme values is of interest.
    """

    return 1 - normal_probability_between(lo, hi, mu, sigma)


def normal_upper_bound(probability: float, mu: float = 0, sigma: float = 1) -> float:
    """
    Calculates the upper bound of a normal distribution for a given probability.
    This function computes the value below which a given percentage (probability) 
    of the data in a normal distribution falls. It uses the inverse cumulative 
    distribution function (inverse CDF) to determine this value.
    Parameters:
    - probability (float): The probability threshold (between 0 and 1) for which 
      the upper bound is calculated. For example, a probability of 0.95 means 
      that 95% of the data falls below the calculated upper bound.
    - mu (float, optional): The mean (average) of the normal distribution. 
      Default is 0.
    - sigma (float, optional): The standard deviation of the normal distribution. 
      Default is 1.
    Returns:
    - float: The upper bound value of the normal distribution for the given 
      probability.
    Example:
    If you want to find the value below which 95% of the data in a standard 
    normal distribution (mean = 0, standard deviation = 1) falls, you would call:
    normal_upper_bound(0.95)
    """

    return inverse_normal_cdf(probability, mu, sigma)


def normal_lower_bound(probability: float, mu: float = 0, sigma: float = 1) -> float:
    """
    Calculates the lower bound of a normal distribution for a given probability.
    This function computes the value below which a given percentage (probability) 
    of the data in a normal distribution falls. It uses the inverse cumulative 
    distribution function (inverse CDF) to find this value.
    Parameters:
    -----------
    probability : float
        The probability (between 0 and 1) for which the lower bound is calculated. 
        This represents the area under the normal distribution curve to the left 
        of the returned value.
    mu : float, optional
        The mean (average) of the normal distribution. Default is 0.
    sigma : float, optional
        The standard deviation of the normal distribution. Default is 1.
    Returns:
    --------
    float
        The lower bound value of the normal distribution for the given probability.
    Example:
    --------
    If you want to find the value below which 5% of the data in a standard normal 
    distribution (mean=0, standard deviation=1) falls, you would call:
    normal_lower_bound(0.05)
    """

    return inverse_normal_cdf(1 - probability, mu, sigma)


def normal_two_sided_bounds(probability: float, mu: float = 0, sigma: float = 1) -> Tuple[float, float]:
    """
    Calculate the two-sided bounds for a normal distribution given a probability.
    This function computes the lower and upper bounds of a normal distribution
    such that the specified probability lies within these bounds. The bounds are
    symmetric around the mean (mu).
    Parameters:
    -----------
    probability : float
        The probability that the true value lies within the calculated bounds.
        This should be a value between 0 and 1.
    mu : float, optional
        The mean of the normal distribution. Default is 0.
    sigma : float, optional
        The standard deviation of the normal distribution. Default is 1.
    Returns:
    --------
    Tuple[float, float]
        A tuple containing the lower and upper bounds of the normal distribution
        for the given probability.
    Detailed Explanation:
    ---------------------
    The function calculates the tail probability as (1 - probability) / 2. This
    tail probability is then used to determine the upper and lower bounds of the
    normal distribution. The upper bound is calculated such that the tail probability
    lies above it, and the lower bound is calculated such that the tail probability
    lies below it. The function returns these bounds as a tuple.
    """

    tail_probability = (1 - probability) / 2

    # The upper bound should have tail_probability above it
    upper_bound = normal_lower_bound(tail_probability, mu, sigma)

    # The lower bound should have tail_probability below it
    lower_bound = normal_upper_bound(tail_probability, mu, sigma)

    return lower_bound, upper_bound


def two_side_p_value(x: float, mu: float = 0, sigma: float = 1) -> float:
    """
    Calculate the two-sided p-value for a given value in a normal distribution.
    This function calculates the two-sided p-value for a given value in a normal 
    distribution with the specified mean and standard deviation. The p-value is 
    the probability that a random variable from the distribution is at least as 
    extreme as the given value (in both tails).
    Parameters:
    -----------
    x : float
        The value for which the p-value is calculated.
    mu : float, optional
        The mean of the normal distribution. Default is 0.
    sigma : float, optional
        The standard deviation of the normal distribution. Default is 1.
    Returns:
    --------
    float
        The two-sided p-value for the given value in the normal distribution.
    """

    if x >= mu:
        # x is greater than the mean, so the tail is everything greater than x
        return 2 * normal_probability_above(x, mu, sigma)
    else:
        # x is less than the mean, so the tail is everything less than x
        return 2 * normal_probability_below(x, mu, sigma)


def run_experiment() -> List[bool]:
    """
    Simulate the results of a coin-flipping experiment.
    This function simulates the results of flipping a fair coin 1000 times.
    It returns a list of boolean values, where `True` represents a "heads" result
    and `False` represents a "tails" result.
    Returns:
    --------
    List[bool]: A list of boolean values representing the results of the coin flips.
    """

    return [random.random() < 0.5 for _ in range(1000)]


def reject_fairness(experiment: List[bool]) -> bool:
    """
    Determine if the coin used in the experiment is fair.
    This function performs a hypothesis test to determine if the coin used in the
    experiment is fair. It uses a two-sided p-value test to check if the number of
    "heads" in the experiment is significantly different from the expected number
    for a fair coin.
    Parameters:
    -----------
    experiment : List[bool]
        A list of boolean values representing the results of the coin flips.
    Returns:
    --------
    bool:
        `True` if the null hypothesis of fairness is rejected, `False` otherwise.
    """

    num_heads = len([flip for flip in experiment if flip])
    return num_heads < 469 or num_heads > 531



def estimated_parameters(N: int, n: int) -> Tuple[float, float]:
    """
    Calculate the estimated parameters for a binomial distribution.
    This function calculates the estimated parameters for a binomial distribution
    based on the number of trials and the number of successes.
    Parameters:
    -----------
    N : int
        The total number of trials.
    n : int
        The number of successful trials.
    Returns:
    --------
    Tuple[float, float]:
        A tuple containing the estimated parameters:
        - p: The estimated probability of success.
        - sigma: The standard deviation of the estimated probability.
    """

    p = n / N
    sigma = math.sqrt(p * (1 - p) / N)
    return p, sigma


def a_b_test_statistic(N_A: int, n_A: int, N_B: int, n_B: int) -> float:
    """
    Calculate the z-score for an A/B test.
    This function calculates the z-score for an A/B test based on the number of trials
    and successes in two groups, A and B.
    Parameters:
    -----------
    N_A : int
        The total number of trials in group A.
    n_A : int
        The number of successful trials in group A.
    N_B : int
        The total number of trials in group B.
    n_B : int
        The number of successful trials in group B.
    Returns:
    --------
    float:
        The z-score for the A/B test.
    """

    p_A, sigma_A = estimated_parameters(N_A, n_A)
    p_B, sigma_B = estimated_parameters(N_B, n_B)
    return (p_B - p_A) / math.sqrt(sigma_A**2 + sigma_B**2)



def B(alpha: float, beta: float) -> float:
    """
    Calculate the normalizing constant for a beta distribution.
    This function calculates the normalizing constant for a beta distribution
    with parameters `alpha` and `beta`. The normalizing constant ensures that
    the probability density function integrates to 1.
    Parameters:
    -----------
    alpha : float
        The alpha parameter of the beta distribution.
    beta : float
        The beta parameter of the beta distribution.
    Returns:
    --------
    float:
        The normalizing constant for the beta distribution.
    """

    return math.gamma(alpha) * math.gamma(beta) / math.gamma(alpha + beta)


def beta_pdf(x: float, alpha: float, beta: float) -> float:
    """
    Calculate the probability density function of a beta distribution.
    This function calculates the probability density function (PDF) of a beta
    distribution at a given value `x`, with parameters `alpha` and `beta`.
    Parameters:
    -----------
    x : float
        The value at which to calculate the PDF.
    alpha : float
        The alpha parameter of the beta distribution.
    beta : float
        The beta parameter of the beta distribution.
    Returns:
    --------
    float:
        The PDF of the beta distribution at the given value `x`.
    """

    if x < 0 or x > 1:
        return 0
    return x**(alpha - 1) * (1 - x)**(beta - 1) / B(alpha, beta)
