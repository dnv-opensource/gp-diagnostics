import numpy as np

from gp_diagnostics.utils.checks import is_lower_triang, is_numeric_np_array, is_square


def test_is_numeric_np_array():
    """
    Test the function is_numeric_np_array()
    """

    # These should be ok
    assert is_numeric_np_array(np.array([1, 2, 2.3]))
    assert is_numeric_np_array(np.array([[1, 2, 2.3], [0, 1e-6, 0.001]]))
    assert is_numeric_np_array(np.array(200))

    # These are not ok
    assert not is_numeric_np_array(np.array([[1, 2, 2.3], [0, 0.001]], dtype=object))
    assert not is_numeric_np_array(np.array(None))
    assert not is_numeric_np_array(np.array([[1, 2, 2.3], [0, 0.001, "a"]]))
    assert not is_numeric_np_array("a")
    assert not is_numeric_np_array(50)
    assert not is_numeric_np_array([1, 2, 3])


def test_is_square():
    """
    Test the function is_square()
    """
    # These should be ok
    assert is_square(np.ones(shape=(1, 1)))
    assert is_square(np.ones(shape=(14, 14)))

    # These are not ok
    assert not is_square(np.ones(shape=(13, 14)))
    assert not is_square(np.ones(shape=(3, 2)))
    assert not is_square(np.ones(shape=(3, 3, 3)))
    assert not is_square(np.array(2))
    assert not is_square(np.array([1, 2, 3]))


def test_is_lower_triang():
    """
    Test the function is_lower_triang()
    """
    # These should be ok
    arr = np.array([[0.2, 0, 0], [3, 2.2, 0], [1, 2, 4]])

    assert is_lower_triang(arr)

    arr = np.array([[1, 0], [2, 2.2]])
    assert is_lower_triang(arr)

    arr = np.array([[1]])
    assert is_lower_triang(arr)

    # These are not ok
    arr = np.array([[1, 2, 2.3], [0, 2.2, 3], [0.1, 0, 4]])

    assert not is_lower_triang(arr)
