import sys

import numpy as np

sys.path.append("calib")

# Paste your two helpers here (or import them):
from scoring import _align_pred
from scoring import _to_matrix


def assert_equal(a, b, msg=""):
    if a != b:
        raise AssertionError(f"{msg} | {a} != {b}")


def assert_shape(arr, shape, msg=""):
    if arr.shape != shape:
        raise AssertionError(f"{msg} | shape {arr.shape} != {shape}")


def assert_allclose(a, b, rtol=1e-9, atol=1e-12, msg=""):
    if not np.allclose(a, b, rtol=rtol, atol=atol):
        raise AssertionError(f"{msg} | arrays differ:\n{a}\n!=\n{b}")


def test_scalar():
    A, rows, cols = _to_matrix(3.5)
    assert_shape(A, (1, 1), "scalar to_matrix")
    assert_equal(rows, [None], "scalar rows")
    assert_equal(cols, [None], "scalar cols")
    assert_allclose(A, np.array([[3.5]]), msg="scalar value")

    P = _align_pred(4.2, rows, cols)
    assert_shape(P, (1, 1), "scalar align_pred")
    assert_allclose(P, np.array([[4.2]]), msg="scalar pred")

    # Non-castable pred -> 0.0 fallback
    P = _align_pred(object(), rows, cols)
    assert_allclose(P, np.array([[0.0]]), msg="scalar pred fallback")


def test_vector_list():
    vec = [1.0, 2.0, 3.0]
    A, rows, cols = _to_matrix(vec)
    assert_shape(A, (1, 3), "vector to_matrix")
    assert_equal(rows, [None], "vector rows")
    assert_equal(cols, [0, 1, 2], "vector cols")
    assert_allclose(A, np.array([[1.0, 2.0, 3.0]]), msg="vector values")

    # Align dict prediction by index keys
    pred_dict = {0: 10.0, 1: 20.0, 2: 30.0}
    P = _align_pred(pred_dict, rows, cols)
    assert_allclose(P, np.array([[10.0, 20.0, 30.0]]), msg="vector pred dict")

    # Align shorter list -> zero-pad remaining
    pred_list = [9.0, 8.0]
    P = _align_pred(pred_list, rows, cols)
    assert_allclose(P, np.array([[9.0, 8.0, 0.0]]), msg="vector pred short list")


def test_vector_1level_dict():
    d = {"b": 2.0, "a": 1.0}
    A, rows, cols = _to_matrix(d)
    # columns sorted by key -> ["a","b"]
    assert_shape(A, (1, 2), "1-level dict to_matrix")
    assert_equal(cols, ["a", "b"], "1-level dict col labels")
    assert_allclose(A, np.array([[1.0, 2.0]]), msg="1-level dict row values")

    # Align dict must match by same keys
    pred = {"a": 10.0, "b": 20.0}
    P = _align_pred(pred, rows, cols)
    assert_allclose(P, np.array([[10.0, 20.0]]), msg="1-level dict pred (dict)")

    # Align list here should NOT map (labels are strings), expect zeros
    pred_list = [10.0, 20.0]
    P = _align_pred(pred_list, rows, cols)
    assert_allclose(P, np.array([[0.0, 0.0]]), msg="1-level dict pred (list) -> zeros")


def test_matrix_2level_dict_of_dict():
    actual = {
        "row1": {"a": 1.0, "b": 2.0},
        "row0": {"b": 3.0},
    }
    A, rows, cols = _to_matrix(actual)
    # rows sorted -> ["row0","row1"], cols union/sorted -> ["a","b"]
    assert_equal(rows, ["row0", "row1"], "2-level rows order")
    assert_equal(cols, ["a", "b"], "2-level cols order")
    expected = np.array([[0.0, 3.0], [1.0, 2.0]])
    assert_allclose(A, expected, msg="2-level dict-of-dict matrix")

    # Align pred with missing entries -> zeros filled
    pred = {
        "row1": {"a": 10.0},  # missing "b"
        "row0": {"b": 30.0, "c": 999},  # "c" ignored (not in cols)
    }
    P = _align_pred(pred, rows, cols)
    expectedP = np.array([[0.0, 30.0], [10.0, 0.0]])
    assert_allclose(P, expectedP, msg="2-level dict-of-dict align")


def test_matrix_2level_dict_of_list():
    actual = {
        "r0": [1.0, 2.0],
        "r1": [3.0],  # shorter row
    }
    A, rows, cols = _to_matrix(actual)
    assert_equal(rows, ["r0", "r1"], "2-level list rows")
    assert_equal(cols, [0, 1], "2-level list cols are indices")
    expected = np.array([[1.0, 2.0], [3.0, 0.0]])
    assert_allclose(A, expected, msg="2-level dict-of-list matrix")

    # Align pred with dict-of-list, shorter row handled
    pred = {"r0": [10.0, 20.0], "r1": [30.0]}
    P = _align_pred(pred, rows, cols)
    expectedP = np.array([[10.0, 20.0], [30.0, 0.0]])
    assert_allclose(P, expectedP, msg="2-level dict-of-list align")


def test_matrix_pred_missing_row():
    actual = {"A": [1.0, 2.0], "B": [3.0, 4.0]}
    A, rows, cols = _to_matrix(actual)
    pred = {"A": [10.0, 20.0]}  # B missing -> zeros row
    P = _align_pred(pred, rows, cols)
    expectedP = np.array([[10.0, 20.0], [0.0, 0.0]])
    assert_allclose(P, expectedP, msg="2-level missing row -> zeros")


def test_empty_dict():
    A, rows, cols = _to_matrix({})
    assert_shape(A, (1, 0), "empty dict row shape")
    assert_equal(rows, [None], "empty dict rows")
    assert_equal(cols, [], "empty dict cols")
    # Align any pred returns (1,0) zeros
    P = _align_pred({}, rows, cols)
    assert_shape(P, (1, 0), "empty dict align_pred shape")


def test_mixed_inner_key_types_raises():
    # NOTE: Python cannot sort mixed types (e.g., int and str).
    # This test ensures we see/acknowledge the limitation.
    actual = {
        "r0": {"0": 1.0, "1": 2.0},  # string keys
        "r1": [3.0, 4.0],  # list -> integer indices
    }
    try:
        _to_matrix(actual)  # may raise TypeError during sorted()
        # If it doesn't raise here on your Python version, at least assert that
        # col_labels are consistent types (all str or all int).
        A, rows, cols = _to_matrix(actual)
        all_str = all(isinstance(c, str) for c in cols)
        all_int = all(isinstance(c, int) for c in cols)
        if not (all_str or all_int):
            raise AssertionError("Mixed-type col_labels produced without error.")
    except TypeError:
        # Expected behavior on most Python versions.
        pass


def test_vector_pred_extra_length_ignored():
    actual = [1.0, 2.0]
    A, rows, cols = _to_matrix(actual)  # cols [0,1]
    pred = [10.0, 20.0, 30.0]  # extra value ignored
    P = _align_pred(pred, rows, cols)
    assert_allclose(P, np.array([[10.0, 20.0]]), msg="extra vector entries ignored")


def run_all():
    test_scalar()
    test_vector_list()
    test_vector_1level_dict()
    test_matrix_2level_dict_of_dict()
    test_matrix_2level_dict_of_list()
    test_matrix_pred_missing_row()
    test_empty_dict()
    test_mixed_inner_key_types_raises()
    test_vector_pred_extra_length_ignored()
    print("All _to_matrix/_align_pred tests: OK")


if __name__ == "__main__":
    run_all()
