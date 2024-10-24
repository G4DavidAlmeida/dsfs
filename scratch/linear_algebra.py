
import math
from typing import Callable, Tuple, List

Vector = List[float]


def add(v: Vector, w: Vector) -> Vector:
    """
    Adds two vectors element-wise.
    Args:
        v (Vector): The first vector. Must be of the same length as the second vector.
        w (Vector): The second vector. Must be of the same length as the first vector.
    Returns:
        Vector: A new vector where each element is the sum of the corresponding elements of `v` and `w`.
    Raises:
        AssertionError: If the vectors `v` and `w` are not of the same length.
    Detailed Explanation:
        This function takes two vectors `v` and `w` as input and returns a new vector. 
        Each element of the returned vector is the sum of the corresponding elements from `v` and `w`. 
        For example, if `v = [1, 2, 3]` and `w = [4, 5, 6]`, the function will return `[5, 7, 9]`.
        The function uses an assertion to ensure that the input vectors are of the same length, 
        as element-wise addition is only defined for vectors of the same length.
    """

    assert len(v) == len(w), "vectors must be the same length"
    return [v_i + w_i for v_i, w_i in zip(v, w)]


assert add([1, 2, 3], [4, 5, 6]) == [5, 7, 9]


def subtract(v: Vector, w: Vector) -> Vector:
    """
    Subtracts corresponding elements of two vectors.
    This function takes two vectors `v` and `w` of the same length and returns a new vector 
    where each element is the result of subtracting the corresponding elements of `w` from `v`.
    Parameters:
    v (Vector): The first vector.
    w (Vector): The second vector, which will be subtracted from the first vector.
    Returns:
    Vector: A new vector containing the differences of the corresponding elements of `v` and `w`.
    Raises:
    AssertionError: If the input vectors `v` and `w` are not of the same length.
    Example:
    >>> v = [1, 2, 3]
    >>> w = [4, 5, 6]
    >>> subtract(v, w)
    [-3, -3, -3]
    """

    assert len(v) == len(w), "vectors must be the same length"
    return [v_i - w_i for v_i, w_i in zip(v, w)]


assert subtract([5, 7, 9], [4, 5, 6]) == [1, 2, 3]


def vector_sum(vectors: List[Vector]) -> Vector:
    """
    Computes the element-wise sum of a list of vectors.
    Args:
        vectors (List[Vector]): A list of vectors, where each vector is a list of numbers.
                                All vectors must be of the same length.
    Returns:
        Vector: A single vector which is the element-wise sum of the input vectors.
    Raises:
        AssertionError: If the input vectors are not all of the same length.
    Example:
        >>> vector_sum([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        [12, 15, 18]
    Detailed Explanation:
        The function takes a list of vectors (each vector being a list of numbers) and computes
        the sum of each corresponding element across all vectors. It first checks that all vectors
        are of the same length. Then, it iterates over the range of the number of elements in the
        vectors and sums the corresponding elements from each vector, returning a new vector with
        these sums.
    """

    num_elements = len(vectors[0])
    assert all(len(v) == num_elements for v in vectors), "different sizes"
    return [sum(vector[i] for vector in vectors) for i in range(num_elements)]


assert vector_sum([[1, 2], [3, 4], [5, 6], [7, 8]]) == [16, 20]


def scalar_multiply(c: float, v: Vector) -> Vector:
    """
    Multiplies each element of the vector `v` by the scalar `c`.
    Args:
        c (float): The scalar value to multiply with each element of the vector.
        v (Vector): A list of numerical values representing the vector.
    Returns:
        Vector: A new vector where each element is the product of the scalar `c` and the corresponding element in the original vector `v`.
    Detailed Explanation:
        This function takes a scalar value `c` and a vector `v` (which is a list of numerical values) as inputs. It returns a new vector where each element is the result of multiplying the scalar `c` by the corresponding element in the input vector `v`. This operation is commonly used in linear algebra to scale vectors.
    """

    return [c * v_i for v_i in v]


assert scalar_multiply(2, [1, 2, 3]) == [2, 4, 6]


def vector_mean(vectors: List[Vector]) -> Vector:
    """
    Computes the mean (average) of a list of vectors.
    Args:
        vectors (List[Vector]): A list of vectors, where each vector is a list of numbers.
    Returns:
        Vector: A vector representing the mean of the input vectors.
    Detailed Explanation:
        The function `vector_mean` calculates the mean of a list of vectors. 
        It first determines the number of vectors, `n`. Then, it computes the 
        sum of all vectors using the `vector_sum` function. Finally, it scales 
        the resulting sum vector by `1/n` using the `scalar_multiply` function 
        to obtain the mean vector. The mean vector is a vector where each 
        component is the average of the corresponding components in the input 
        vectors.
    """

    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))


assert vector_mean([[1, 2], [3, 4], [5, 6]]) == [3, 4]


def dot(v: Vector, w: Vector) -> float:
    """
    Computes the dot product of two vectors.
    The dot product is the sum of the products of the corresponding entries 
    of the two sequences of numbers. It is a measure of the extent to which 
    two vectors are parallel.
    Args:
        v (Vector): The first vector, a sequence of numbers.
        w (Vector): The second vector, a sequence of numbers.
    Returns:
        float: The dot product of the two vectors.
    Raises:
        AssertionError: If the vectors are not of the same length.
    Example:
        >>> v = [1, 2, 3]
        >>> w = [4, 5, 6]
        >>> dot(v, w)
        32
    Detailed Explanation:
        The function first checks if the lengths of the two vectors are the same.
        If they are not, it raises an AssertionError. If they are the same length,
        it computes the dot product by multiplying corresponding elements from each
        vector and summing the results.
    """

    assert len(v) == len(w), "vectors must be the same length"
    return sum(v_i * w_i for v_i, w_i in zip(v, w))


assert dot([1, 2, 3], [4, 5, 6]) == 32


def sum_of_squares(v: Vector) -> float:
    """
    Computes the sum of squares of the elements in the given vector.
    Args:
        v (Vector): A vector of numerical values.
    Returns:
        float: The sum of the squares of the vector's elements.
    Detailed Explanation:
        The function calculates the sum of squares by taking the dot product of the vector with itself.
        Mathematically, this is equivalent to summing up the squares of each individual element in the vector.
        For example, if the vector is [a, b, c], the sum of squares is calculated as a^2 + b^2 + c^2.
    """

    return dot(v, v)


assert sum_of_squares([1, 2, 3]) == 14


def magnitude(v: Vector) -> float:
    """
    Calculate the magnitude (or length) of a vector.
    The magnitude of a vector is calculated as the square root of the sum of the squares of its components.
    This is also known as the Euclidean norm or 2-norm of the vector.
    Parameters:
    v (Vector): A vector represented as a list or tuple of numerical values.
    Returns:
    float: The magnitude of the vector.
    Example:
    >>> magnitude([3, 4])
    5.0
    """

    return math.sqrt(sum_of_squares(v))


assert magnitude([3, 4]) == 5


def square_distance(v: Vector, w: Vector) -> float:
    """
    Computes the squared Euclidean distance between two vectors.
    The squared Euclidean distance is the sum of the squared differences 
    between corresponding elements of the two vectors. This function 
    calculates the distance without taking the square root, which can be 
    useful for performance reasons when only the relative distances are 
    needed.
    Parameters:
    v (Vector): The first vector.
    w (Vector): The second vector.
    Returns:
    float: The squared Euclidean distance between the two vectors.
    Example:
    >>> v = [1, 2, 3]
    >>> w = [4, 5, 6]
    >>> square_distance(v, w)
    27
    Note:
    This function assumes that both vectors have the same length.
    """

    return sum_of_squares(subtract(v, w))


assert square_distance([1, 2], [2, 3]) == 2


def distance(v: Vector, w: Vector) -> float:
    """
    Calculates the Euclidean distance between two vectors.
    Args:
        v (Vector): The first vector.
        w (Vector): The second vector.
    Returns:
        float: The Euclidean distance between vectors v and w.
    Detailed Explanation:
        The function computes the Euclidean distance between two vectors v and w.
        It first subtracts vector w from vector v to get the difference vector.
        Then, it calculates the magnitude (or length) of this difference vector,
        which represents the Euclidean distance between the original vectors.
    """

    return magnitude(subtract(v, w))


assert distance([1, 2], [2, 3]) == math.sqrt(2)

Matrix = List[List[float]]


def shape(A: Matrix) -> Tuple[int, int]:
    """
    Returns the shape (dimensions) of a given matrix.
    Parameters:
    A (Matrix): A 2-dimensional list representing the matrix.
    Returns:
    Tuple[int, int]: A tuple containing two integers:
        - The number of rows in the matrix.
        - The number of columns in the matrix.
    Detailed Explanation:
    This function calculates the dimensions of a matrix by determining the number of rows and columns it contains. 
    It first counts the number of rows by using the `len` function on the matrix `A`. 
    Then, it counts the number of columns by using the `len` function on the first row of the matrix `A` (i.e., `A[0]`), 
    provided that the matrix is not empty. If the matrix is empty, it returns 0 for the number of columns.
    """

    num_rows = len(A)
    num_cols = len(A[0]) if A else 0
    return num_rows, num_cols


assert shape([[1, 2, 3], [4, 5, 6]]) == (2, 3)


def get_row(A: Matrix, i: int) -> Vector:
    """
    Retrieves a specific row from a given matrix.
    Args:
        A (Matrix): The matrix from which to retrieve the row. 
                    It is assumed to be a list of lists, where each inner list represents a row.
        i (int): The index of the row to retrieve. Indexing starts at 0.
    Returns:
        Vector: The row at the specified index as a list.
    Detailed Explanation:
        This function takes a matrix `A` and an integer `i` as inputs. 
        It returns the `i`-th row of the matrix `A`. The matrix `A` is expected to be a list of lists, 
        where each inner list represents a row of the matrix. The function uses Python's list indexing 
        to access and return the desired row.
    """

    return A[i]


def get_column(A: Matrix, j: int) -> Vector:
    """
    Extracts a column from a given matrix.
    Args:
        A (Matrix): A 2-dimensional list representing the matrix from which the column is to be extracted.
        j (int): The index of the column to be extracted.
    Returns:
        Vector: A list representing the j-th column of the matrix A.
    Detailed Explanation:
        This function takes a matrix `A` and an integer `j` as inputs. The matrix `A` is assumed to be a list of lists,
        where each inner list represents a row of the matrix. The function iterates over each row of the matrix and 
        extracts the element at the j-th index, effectively collecting all elements from the j-th column of the matrix.
        The result is a list (vector) containing all the elements from the specified column.
    """

    return [A_i[j] for A_i in A]


def make_matrix(num_rows: int, num_cols: int, entry_fn: Callable[[int, int], float]) -> Matrix:
    """
    Creates a matrix with the specified number of rows and columns, where each entry
    is generated by a provided function.
    Args:
        num_rows (int): The number of rows in the matrix.
        num_cols (int): The number of columns in the matrix.
        entry_fn (Callable[[int, int], float]): A function that takes two integers 
            (the row index and the column index) and returns a float. This function 
            is used to generate the value for each entry in the matrix.
    Returns:
        Matrix: A list of lists representing the matrix, where each inner list is a row 
        of the matrix and each element within the inner list is a float.
    Detailed Explanation:
        The function `make_matrix` constructs a matrix by iterating over the specified 
        number of rows and columns. For each position (i, j) in the matrix, it calls 
        the provided `entry_fn` function with the current row index `i` and column index `j`. 
        The value returned by `entry_fn` is placed in the corresponding position in the matrix.
        This allows for flexible matrix creation, where the values can be generated based 
        on the indices, enabling the creation of identity matrices, diagonal matrices, 
        or any other custom matrix structure.
    """
    return [
        [entry_fn(i, j) for j in range(num_cols)]
        for i in range(num_rows)
    ]


def identity_matrix(n: int) -> Matrix:
    """
    Generates an identity matrix of size n x n.
    An identity matrix is a square matrix with ones on the main diagonal and zeros elsewhere.
    It is denoted as I_n for an n x n identity matrix.
    Parameters:
    n (int): The size of the identity matrix to generate. It determines both the number of rows and columns.
    Returns:
    Matrix: An n x n identity matrix where each element (i, j) is 1 if i == j, otherwise 0.
    Example:
    >>> identity_matrix(3)
    [[1, 0, 0],
     [0, 1, 0],
     [0, 0, 1]]
    This function uses the `make_matrix` function to create the matrix, with a lambda function to set the value of each element.
    """

    return make_matrix(n, n, lambda i, j: 1 if i == j else 0)


assert identity_matrix(5) == [
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1]
]
