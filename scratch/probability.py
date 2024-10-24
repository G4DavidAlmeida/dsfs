
import math
import random


def uniform_pdf(x: float) -> float:
    """
    Computes the probability density function (PDF) of a uniform distribution.
    The uniform distribution is a type of probability distribution in which all outcomes are equally likely within a certain range. 
    In this case, the range is [0, 1).
    Parameters:
    x (float): The value at which the PDF is evaluated. It should be a real number.
    Returns:
    float: The probability density at the given value x. 
           - Returns 1 if 0 <= x < 1, indicating that the value x is within the range of the uniform distribution.
           - Returns 0 otherwise, indicating that the value x is outside the range of the uniform distribution.
    Detailed Explanation:
    The function checks if the input value x lies within the interval [0, 1). If it does, the function returns 1, 
    indicating that the probability density is 1 within this interval. If x is outside this interval, the function 
    returns 0, indicating that the probability density is 0 outside this interval. This is characteristic of a 
    uniform distribution over the interval [0, 1).
    
    Examples:
    >>> uniform_pdf(0.5)
    1
    >>> uniform_pdf(1.5)
    0
    >>> uniform_pdf(-0.5)
    0
    """
    return 1 if 0 <= x < 1 else 0


def uniform_cdf(x: float) -> float:
    """
    Computes the cumulative distribution function (CDF) for a uniform distribution.
    The uniform distribution is defined over the interval [0, 1]. The CDF is a function 
    that gives the probability that a uniform random variable is less than or equal to x.
    Parameters:
    x (float): The value at which to evaluate the CDF. Should be a real number.
    Returns:
    float: The probability that a uniform random variable is less than or equal to x.
           - If x < 0, the function returns 0.
           - If 0 <= x < 1, the function returns x.
           - If x >= 1, the function returns 1.
    Detailed Explanation:
    - For values of x less than 0, the probability is 0 because the uniform distribution 
      is only defined for values between 0 and 1.
    - For values of x between 0 and 1, the probability is equal to x because the uniform 
      distribution has a constant probability density over this interval.
    - For values of x greater than or equal to 1, the probability is 1 because the entire 
      distribution lies within the interval [0, 1].
    
    Examples:
    >>> uniform_cdf(0.5)
    0.5
    >>> uniform_cdf(1.5)
    1
    >>> uniform_cdf(-0.5)
    0
    """
    if x < 0:
        return 0
    elif x < 1:
        return x
    else:
        return 1


SQRT_TWO_PI = math.sqrt(2 * math.pi)


def normal_pdf(x: float, mu: float = 0, sigma: float = 1) -> float:
    """
    Calculates the value of the probability density function (PDF) for a normal distribution.
    The normal distribution, also known as the Gaussian distribution, is a continuous probability distribution 
    characterized by a bell-shaped curve. This function computes the PDF value for a given point `x` in a normal 
    distribution with mean `mu` and standard deviation `sigma`.
    Parameters:
    x (float): The point at which to evaluate the PDF.
    mu (float, optional): The mean (average) of the normal distribution. Default is 0.
    sigma (float, optional): The standard deviation of the normal distribution. Default is 1.
    Returns:
    float: The value of the PDF at point `x` for a normal distribution with mean `mu` and standard deviation `sigma`.
    Detailed Explanation:
    The PDF of a normal distribution is given by the formula:
        f(x) = (1 / (sqrt(2 * pi) * sigma)) * exp(-((x - mu)^2) / (2 * sigma^2))
    This function implements the above formula. It calculates the exponent part `exp(-((x - mu)^2) / (2 * sigma^2))` 
    and divides it by the normalization factor `(sqrt(2 * pi) * sigma)` to get the PDF value at `x`.
    
    Examples:
    >>> normal_pdf(0)
    0.3989422804014327
    >>> normal_pdf(1)
    0.24197072451914337
    >>> normal_pdf(0, mu=1, sigma=2)
    0.17603266338214976
    """
    return (math.exp(-(x-mu) ** 2 / 2 / sigma ** 2) / (SQRT_TWO_PI * sigma))


def normal_cdf(x: float, mu: float = 0, sigma: float = 1) -> float:
    """
    Computes the cumulative distribution function (CDF) for a normal distribution.
    The normal cumulative distribution function is the probability that a normally 
    distributed random variable with mean `mu` and standard deviation `sigma` is 
    less than or equal to `x`.
    Parameters:
    x (float): The value up to which the CDF is computed.
    mu (float, optional): The mean of the normal distribution. Default is 0.
    sigma (float, optional): The standard deviation of the normal distribution. Default is 1.
    Returns:
    float: The probability that a normally distributed random variable is less than or equal to `x`.
    Detailed Explanation:
    The function uses the error function `erf` to compute the CDF of the normal distribution.
    The error function is scaled and shifted to account for the mean `mu` and standard deviation `sigma`
    of the normal distribution. The result is then scaled to the range [0, 1] to represent a probability.
    
    Examples:
    >>> normal_cdf(0)
    0.5
    >>> normal_cdf(1)
    0.8413447460685429
    >>> normal_cdf(0, mu=1, sigma=2)
    0.3085375387259869
    """
    return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2


def inverse_normal_cdf(p: float, mu: float = 0, sigma: float = 1, tolerance: float = 0.00001) -> float:
    """
    Computes the inverse of the cumulative distribution function (CDF) of a normal distribution.
    This function finds the z-value such that the probability of the normal distribution 
    being less than or equal to that z-value is equal to the given probability `p`. This is 
    also known as the quantile function or the percent-point function.
    Parameters:
    -----------
    p : float
        The probability corresponding to the desired quantile. Must be between 0 and 1.
    mu : float, optional
        The mean of the normal distribution. Default is 0.
    sigma : float, optional
        The standard deviation of the normal distribution. Default is 1.
    tolerance : float, optional
        The tolerance level for the binary search algorithm. Default is 0.00001.
    Returns:
    --------
    float
        The z-value such that the cumulative distribution function of the normal distribution 
        with mean `mu` and standard deviation `sigma` is equal to `p`.
    Detailed Explanation:
    ---------------------
    The function uses a binary search algorithm to find the z-value that corresponds to the 
    given probability `p`. If the mean `mu` and standard deviation `sigma` are not the default 
    values (0 and 1, respectively), the function first normalizes the problem to the standard 
    normal distribution and then scales the result back to the specified mean and standard 
    deviation. The binary search continues until the difference between the high and low 
    bounds is less than the specified tolerance.
    
    Examples:
    >>> inverse_normal_cdf(0.5)
    0.0
    >>> inverse_normal_cdf(0.8413447460685429)
    1.0
    >>> inverse_normal_cdf(0.5, mu=1, sigma=2)
    1.0
    """
    if mu != 0 or sigma != 1:
        return mu + sigma * inverse_normal_cdf(p, tolerance=tolerance)

    low_z = -10.0
    high_z = 10.0
    while high_z - low_z > tolerance:
        mid_z = (low_z + high_z) / 2
        mid_p = normal_cdf(mid_z)
        if mid_p < p:
            low_z = mid_z
        else:
            high_z = mid_z
    return mid_z


def bernoulli_trial(p: float) -> int:
    """
    Perform a Bernoulli trial, which is a random experiment with exactly two possible outcomes: success (1) and failure (0).
    Parameters:
    p (float): The probability of success on a single trial. Must be between 0 and 1 inclusive.
    Returns:
    int: Returns 1 if the trial is a success (with probability p), otherwise returns 0 (with probability 1 - p).
    Detailed Explanation:
    A Bernoulli trial is a random experiment where there are only two possible outcomes: success and failure. 
    This function simulates such a trial by generating a random number between 0 and 1. If the generated number 
    is less than the probability of success `p`, the function returns 1 (indicating success). Otherwise, it returns 0 
    (indicating failure). This is useful in various probabilistic and statistical simulations.
    
    Examples:
    >>> bernoulli_trial(0.5) in [0, 1]
    True
    >>> bernoulli_trial(1)
    1
    >>> bernoulli_trial(0)
    0
    """
    return 1 if random.random() < p else 0


def binomial(n: int, p: float) -> int:
    """
    Perform a binomial trial.
    This function simulates a binomial trial, which is the sum of `n` independent Bernoulli trials with success probability `p`.
    Args:
        n (int): The number of Bernoulli trials to perform.
        p (float): The probability of success for each Bernoulli trial.
    Returns:
        int: The number of successful trials out of `n`.
    Detailed Explanation:
        A binomial trial is a sequence of `n` independent experiments, each of which results in a success with probability `p` and a failure with probability `1 - p`. 
        This function calculates the total number of successes in these `n` trials by summing up the results of individual Bernoulli trials, where each Bernoulli trial is a random experiment that results in a success (1) with probability `p` and a failure (0) with probability `1 - p`.
    
    Examples:
    >>> binomial(10, 0.5) <= 10
    True
    >>> binomial(10, 1)
    10
    >>> binomial(10, 0)
    0
    """
    return sum(bernoulli_trial(p) for _ in range(n))
