
import math
from typing import List
from collections import Counter

from .linear_algebra import sum_of_squares
from .linear_algebra import dot


def mean(xs: List[float]) -> float:
    """
    Calculate the mean (average) of a list of numbers.
    Args:
        xs (List[float]): A list of floating-point numbers.
    Returns:
        float: The mean (average) of the numbers in the list.
    The function computes the mean by summing all the numbers in the list
    and then dividing the sum by the number of elements in the list. If the
    list is empty, this function will raise a ZeroDivisionError.
    
    Example:
        >>> mean([1, 2, 3, 4, 5])
        3.0
    """
    return sum(xs) / len(xs)


def _median_odd(xs: List[float]) -> float:
    """
    Calculate the median of a list with an odd number of elements.
    Args:
        xs (List[float]): A list of floating-point numbers. The list must contain an odd number of elements.
    Returns:
        float: The median value of the list.
    Detailed Explanation:
        The function first sorts the input list in ascending order. Since the list has an odd number of elements,
        the median is the middle element of the sorted list. The function calculates the index of this middle element
        by performing integer division of the list length by 2, and then returns the element at this index.
    
    Example:
        >>> _median_odd([3, 1, 2])
        2
    """

    return sorted(xs)[len(xs) // 2]


def _median_even(xs: List[float]) -> float:
    """
    Calculate the median of a list of numbers when the list length is even.
    This function takes a list of floating-point numbers, sorts it, and then
    calculates the median by averaging the two middle numbers.
    Parameters:
    xs (List[float]): A list of floating-point numbers. The list must have an even number of elements.
    Returns:
    float: The median of the list.
    Detailed Explanation:
    - The function first sorts the input list `xs`.
    - It then finds the midpoint index of the list by performing integer division of the list length by 2.
    - Since the list length is even, there are two middle numbers. The function retrieves these two numbers
      from the sorted list and calculates their average.
    - The result is the median of the list.
    
    Example:
        >>> _median_even([4, 1, 3, 2])
        2.5
    """

    sorted_xs = sorted(xs)
    hi_midpoint = len(xs) // 2  # e.g. length 4 => hi_midpoint 2
    return (sorted_xs[hi_midpoint - 1] + sorted_xs[hi_midpoint]) / 2


def median(v: List[float]) -> float:
    """
    Calculate the median of a list of numbers.
    The median is the value separating the higher half from the lower half of a data sample.
    For a dataset, it may be thought of as "the middle" value. If the dataset contains an 
    even number of observations, the median is the average of the two middle values.
    Args:
        v (List[float]): A list of floating-point numbers for which the median is to be calculated.
    Returns:
        float: The median value of the list.
    The function determines whether the length of the list is even or odd. If the length is even,
    it calls the helper function `_median_even` to compute the median. If the length is odd, it 
    calls the helper function `_median_odd` to compute the median.
    
    Example:
        >>> median([1, 10, 2, 9, 5])
        5
        >>> median([1, 9, 2, 10])
        5.5
    """

    return _median_even(v) if len(v) % 2 == 0 else _median_odd(v)


def quantile(xs: List[float], p: float) -> float:
    """
    Computes the pth-percentile value in a list of numbers.
    Parameters:
    xs (List[float]): A list of numerical values.
    p (float): A float between 0 and 1 representing the desired percentile.
    Returns:
    float: The pth-percentile value from the sorted list of numbers.
    Detailed Explanation:
    The function calculates the pth-percentile of a list of numbers. The percentile is a measure used in statistics indicating the value below which a given percentage of observations in a group of observations falls. For example, the 50th percentile (median) is the value below which 50% of the observations may be found.
    The function works by first calculating the index corresponding to the desired percentile in the sorted list of numbers. This is done by multiplying the percentile `p` by the length of the list minus one, and then converting this product to an integer. The list is then sorted, and the value at the calculated index is returned as the pth-percentile value.
    
    Example:
        >>> quantile([1, 2, 3, 4, 5], 0.5)
        3
    """

    p_index = int(p * (len(xs) - 1))  # Adjusted to avoid index out of range
    return sorted(xs)[p_index]


def mode(x: List[float]) -> List[float]:
    """
    Calculate the mode(s) of a list of numbers.
    The mode is the value(s) that appear most frequently in the list. If there are multiple values with the same highest frequency, all of them are returned.
    Parameters:
    x (List[float]): A list of numbers from which to calculate the mode(s).
    Returns:
    List[float]: A list containing the mode(s) of the input list. If there are multiple modes, all of them are included in the returned list.
    Detailed Explanation:
    - The function first counts the frequency of each number in the input list using a Counter.
    - It then determines the maximum frequency count.
    - Finally, it returns a list of all numbers that have this maximum frequency count.
    
    Example:
        >>> mode([1, 2, 2, 3, 3, 3])
        [3]
    """

    counts = Counter(x)
    max_count = max(counts.values())
    return [x_i for x_i, count in counts.items() if count == max_count]


def data_range(xs: List[float]) -> float:
    """
    Calculate the range of a dataset.
    The range is the difference between the maximum and minimum values in the dataset.
    Args:
        xs (List[float]): A list of numerical values.
    Returns:
        float: The range of the dataset, calculated as the difference between the maximum and minimum values.
    Example:
        >>> data_range([10, 2, 5, 8])
        8.0
    Detailed Explanation:
        The function `data_range` takes a list of floating-point numbers as input and computes the range of the dataset.
        The range is a measure of statistical dispersion, defined as the difference between the largest and smallest values.
        This function uses the built-in `max` and `min` functions to find the maximum and minimum values in the list, respectively,
        and then subtracts the minimum value from the maximum value to obtain the range.
    """
    
    return max(xs) - min(xs)


def de_mean(xs: List[float]) -> List[float]:
    """
    De-mean the input list of numbers.
    This function takes a list of floating-point numbers and returns a new list where each element is the difference between the original element and the mean of the list. This process is known as de-meaning, and it centers the data around zero.
    Parameters:
    xs (List[float]): A list of floating-point numbers to be de-meaned.
    Returns:
    List[float]: A new list of floating-point numbers where each element is the original element minus the mean of the input list.
    Example:
        >>> de_mean([1.0, 2.0, 3.0])
        [-1.0, 0.0, 1.0]
    Detailed Explanation:
    - Calculate the mean (average) of the input list.
    - Subtract the mean from each element in the list.
    - Return the resulting list of differences.
    """

    x_bar = mean(xs)
    return [x - x_bar for x in xs]


def variance(xs: List[float]) -> float:
    """
    Calculate the variance of a list of numbers.
    The variance is a measure of how spread out the numbers in the list are. 
    It is calculated as the average of the squared deviations from the mean.
    Args:
        xs (List[float]): A list of floating-point numbers. The list must contain at least two elements.
    Returns:
        float: The variance of the numbers in the list.
    Raises:
        AssertionError: If the list contains fewer than two elements.
    Detailed Explanation:
        1. The function first checks that the list contains at least two elements.
        2. It calculates the number of elements in the list (n).
        3. It computes the deviations of each element from the mean of the list.
        4. It calculates the sum of the squared deviations.
        5. Finally, it returns the sum of the squared deviations divided by (n - 1), which is the variance.
    
    Example:
        >>> variance([1, 2, 3, 4, 5])
        2.5
    """

    assert len(xs) >= 2, "variance requires at least two elements"
    n = len(xs)
    deviations = de_mean(xs)
    return sum_of_squares(deviations) / (n - 1)


def standard_deviation(xs: List[float]) -> float:
    """
    Calculate the standard deviation of a list of numbers.
    The standard deviation is a measure of the amount of variation or dispersion 
    in a set of values. It is the square root of the variance, which is the 
    average of the squared differences from the mean.
    Args:
        xs (List[float]): A list of numerical values for which the standard 
                          deviation is to be calculated.
    Returns:
        float: The standard deviation of the list of numbers.
    Detailed Explanation:
        1. The function first calculates the variance of the list of numbers.
        2. The variance is the average of the squared differences from the mean.
        3. The standard deviation is then obtained by taking the square root of 
           the variance.
    
    Example:
        >>> standard_deviation([1, 2, 3, 4, 5])
        1.5811388300841898
    """

    return math.sqrt(variance(xs))


def interquartile_range(xs: List[float]) -> float:
    """
    Calculate the interquartile range (IQR) of a list of numbers.
    The interquartile range is a measure of statistical dispersion, or how spread out the values in a data set are. 
    It is calculated as the difference between the 75th percentile (Q3) and the 25th percentile (Q1) of the data.
    Parameters:
    xs (List[float]): A list of numerical values from which to calculate the interquartile range.
    Returns:
    float: The interquartile range of the input list.
    Detailed Explanation:
    The interquartile range (IQR) is a measure of variability, based on dividing a data set into quartiles. 
    Quartiles divide a rank-ordered data set into four equal parts. The values that divide each part are called the first, second, and third quartiles (Q1, Q2, Q3). 
    The IQR is the range between the first quartile (25th percentile) and the third quartile (75th percentile), and it effectively measures the spread of the middle 50% of the data.
    
    Example:
        >>> interquartile_range([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        5.0
    """
    
    return quantile(xs, 0.75) - quantile(xs, 0.25)


def covariance(xs: List[float], ys: List[float]) -> float:
    """
    Calculate the covariance between two lists of numbers.
    Covariance is a measure of how much two random variables vary together. 
    If the greater values of one variable mainly correspond with the greater values of the other variable, 
    and the same holds for the lesser values, the covariance is positive. 
    In the opposite case, when the greater values of one variable mainly correspond to the lesser values of the other, 
    the covariance is negative. A covariance close to zero indicates that the variables are uncorrelated.
    Args:
        xs (List[float]): A list of numerical values representing the first variable.
        ys (List[float]): A list of numerical values representing the second variable.
    Returns:
        float: The covariance between the two lists of numbers.
    Raises:
        AssertionError: If the lengths of xs and ys are not equal.
    Example:
        >>> xs = [1, 2, 3, 4, 5]
        >>> ys = [5, 4, 3, 2, 1]
        >>> covariance(xs, ys)
        -2.5
    """
    
    assert len(xs) == len(ys), "xs and ys must have same number of elements"
    return dot(de_mean(xs), de_mean(ys)) / (len(xs) - 1)


def correlation(xs: List[float], ys: List[float]) -> float:
    """
    Calculates the Pearson correlation coefficient between two lists of numbers.
    The Pearson correlation coefficient is a measure of the linear relationship between two datasets.
    It is calculated as the covariance of the two variables divided by the product of their standard deviations.
    Args:
        xs (List[float]): A list of floats representing the first dataset.
        ys (List[float]): A list of floats representing the second dataset.
    Returns:
        float: The Pearson correlation coefficient between the two datasets. 
               Returns 0 if either dataset has no variation (i.e., standard deviation is zero).
    Detailed Explanation:
        - The function first calculates the standard deviation of both input lists `xs` and `ys`.
        - If both standard deviations are greater than zero, it computes the covariance of the two lists and divides it by the product of the standard deviations.
        - If either of the standard deviations is zero, indicating no variation in one or both datasets, the function returns 0, as the correlation is undefined in such cases.
    
    Example:
        >>> xs = [1, 2, 3, 4, 5]
        >>> ys = [5, 4, 3, 2, 1]
        >>> correlation(xs, ys)
        -1.0
    """

    stdev_x = standard_deviation(xs)
    stdev_y = standard_deviation(ys)
    if stdev_x > 0 and stdev_y > 0:
        return covariance(xs, ys) / stdev_x / stdev_y
    else:
        return 0  # if no variation, correlation is zero
