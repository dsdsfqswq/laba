import sys
import numpy as np
from numpy_tasks import (
    sum_prod, binarize, unique_rows, unique_columns,
    matrix_statistics, chess, draw_rectangle, draw_ellipse,
    analyze_time_series, one_hot
)

class MockInput:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

def run_tests():
    m = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    v = np.array([[[1], [2]], [[3], [4]]])
    res = sum_prod(MockInput(matrices=m, vectors=v))
    assert np.array_equal(res, np.array([[44], [74]]))

    mat = np.array([[1, 5], [3, 2]])
    res = binarize(MockInput(matrix=mat, threshold=2))
    assert np.array_equal(res, np.array([[0, 1], [1, 0]]))

    # 3. unique_rows & unique_columns
    mat = np.array([[1, 1, 2], [3, 3, 3]])
    assert unique_rows(MockInput(matrix=mat)) == [[1.0, 2.0], [3.0]]
    assert unique_columns(MockInput(matrix=mat)) == [[1.0, 3.0], [1.0, 3.0], [2.0, 3.0]]

    stats = matrix_statistics(MockInput(rows=10, columns=5, mean=0.0, std=1.0, seed=42))
    assert stats.matrix.shape == (10, 5)
    assert len(stats.row_means) == 10
    assert len(stats.col_means) == 5

    res = chess(MockInput(rows=3, columns=4, first=0, second=1))
    expected = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]])
    assert np.array_equal(res, expected)

    rect = draw_rectangle(MockInput(width=2, height=2, image_height=4, image_width=4, shape_color=[255,0,0], background_color=[0,0,0]))
    assert rect.shape == (4, 4, 3)
    
    ellipse = draw_ellipse(MockInput(semi_axis_x=2, semi_axis_y=2, image_height=6, image_width=6, shape_color=[255,0,0], background_color=[0,0,0]))
    assert ellipse.shape == (6, 6, 3)

    ts = np.array([1, 4, 2, 5, 1])
    res = analyze_time_series(MockInput(values=ts, window=3))
    assert res.local_maxima == [1, 3]  
    assert res.local_minima == [2]     
    assert np.allclose(res.moving_average, np.array([7.0/3, 11.0/3, 8.0/3]))

    res = one_hot(MockInput(labels=[0, 2, 1], class_count=3))
    expected = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    assert np.array_equal(res, expected)
    print("tests passed")

if __name__ == "__main__":
    try:
        run_tests()
    except AssertionError:
        print("tests failed")
        sys.exit(1)
