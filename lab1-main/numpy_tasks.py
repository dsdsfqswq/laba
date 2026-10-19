"""Заготовки задач на NumPy."""

import numpy as np
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics,
)


def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices, vectors = data.matrices, data.vectors
    return np.sum(matrices @ vectors, axis=0)


def binarize(data: BinarizeInput) -> np.ndarray:
    matrix, threshold = data.matrix, data.threshold
    return (matrix > threshold).astype(int)


def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    return [list(np.unique(row)) for row in matrix]


def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = data.matrix
    return [list(np.unique(col)) for col in matrix.T]


def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rows, columns, mean, std, seed = data.rows, data.columns, data.mean, data.std, data.seed
    rng = np.random.default_rng(seed)
    matrix = rng.normal(loc=mean, scale=std, size=(rows, columns))
    row_means = np.mean(matrix, axis=1)
    col_means = np.mean(matrix, axis=0)
    row_vars = np.var(matrix, axis=1)
    col_vars = np.var(matrix, axis=0)
    return MatrixStatistics(
        matrix=matrix,
        row_means=row_means,
        col_means=col_means,
        row_vars=row_vars,
        col_vars=col_vars
    )


def chess(data: ChessInput) -> np.ndarray:
    rows, columns, first, second = data.rows, data.columns, data.first, data.second
    r = np.arange(rows)[:, None]
    c = np.arange(columns)
    return np.where((r + c) % 2 == 0, first, second)


def draw_rectangle(data: RectangleInput) -> np.ndarray:
    width, height = data.width, data.height
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    
    img = np.full((image_height, image_width, 3), background_color, dtype=np.uint8)

    cy, cx = image_height // 2, image_width // 2
    y_start = max(0, cy - height // 2)
    y_end = min(image_height, cy + (height + 1) // 2)
    x_start = max(0, cx - width // 2)
    x_end = min(image_width, cx + (width + 1) // 2)
    img[y_start:y_end, x_start:x_end] = shape_color
    return img


def draw_ellipse(data: EllipseInput) -> np.ndarray:
    semi_axis_x, semi_axis_y = data.semi_axis_x, data.semi_axis_y
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    img = np.full((image_height, image_width, 3), background_color, dtype=np.uint8)
    cy, cx = image_height // 2, image_width // 2
    y, x = np.ogrid[:image_height, :image_width]
    mask = ((x - cx) ** 2) / (semi_axis_x ** 2) + ((y - cy) ** 2) / (semi_axis_y ** 2) <= 1
    img[mask] = shape_color
    return img


def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    values, window = data.values, data.window
    mean_val = np.mean(values)
    var_val = np.var(values)
    std_val = np.std(values)
    left = values[:-2]
    mid = values[1:-1]
    right = values[2:]
    local_max_indices = np.where((mid > left) & (mid > right))[0] + 1
    local_min_indices = np.where((mid < left) & (mid < right))[0] + 1
    moving_avg = np.convolve(values, np.ones(window) / window, mode='valid')
    return TimeSeriesStatistics(
        mean=mean_val,
        variance=var_val,
        std_dev=std_val,
        local_maxima=local_max_indices.tolist(),
        local_minima=local_min_indices.tolist(),
        moving_average=moving_avg
    )


def one_hot(data: OneHotInput) -> np.ndarray:
    labels, class_count = data.labels, data.class_count
    if class_count is None:
        class_count = int(np.max(labels)) + 1
    return np.eye(class_count)[labels].astype(int)