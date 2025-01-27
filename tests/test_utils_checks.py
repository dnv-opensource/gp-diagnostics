"""
Unit tests for the checks.py utility functions (numeric array, square, lower triangular).
"""

import numpy as np

from gp_diagnostics.utils.checks import is_lower_triang, is_numeric_np_array, is_square


def test_is_numeric_np_array() -> None:
    """
    Tests is_numeric_np_array with various valid and invalid inputs.
    """
    assert is_numeric_np_array(np.array([1, 2, 2.3]))
    assert is_numeric_np_array(np.array([[1, 2, 2.3], [0, 1e-6, 0.001]]))
    assert is_numeric_np_array(np.array(200))

    assert not is_numeric_np_array(np.array([[1, 2, 2.3], [0, 0.001]], dtype=object))
    assert not is_numeric_np_array(np.array(None))
    assert not is_numeric_np_array(np.array([[1, 2, 2.3], [0, 0.001, "a"]]))
    assert not is_numeric_np_array("a")
    assert not is_numeric_np_array(50)
    assert not is_numeric_np_array([1, 2, 3])


def test_is_square() -> None:
    """
    Tests is_square with squares, non-squares, etc.
    """
    assert is_square(np.ones((1, 1)))
    assert is_square(np.ones((14, 14)))

    assert not is_square(np.ones((13, 14)))
    assert not is_square(np.ones((3, 2)))
    assert not is_square(np.ones((3, 3, 3)))
    assert not is_square(np.array(2))
    assert not is_square(np.array([1, 2, 3]))


def test_is_lower_triang() -> None:
    """
    Tests is_lower_triang with valid and invalid matrices.
    """
    arr_ok = np.array([[0.2, 0, 0], [3, 2.2, 0], [1, 2, 4]])
    assert is_lower_triang(arr_ok)

    arr_ok2 = np.array([[1, 0], [2, 2.2]])
    assert is_lower_triang(arr_ok2)

    arr_ok3 = np.array([[1]])
    assert is_lower_triang(arr_ok3)

    arr_bad = np.array([[1, 2, 2.3], [0, 2.2, 3], [0.1, 0, 4]])
    assert not is_lower_triang(arr_bad)
