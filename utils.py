import numpy as np
from typing import Optional, Union, Tuple


def euclidean_distance(x: np.array, y: np.array) -> float:
    """
    Calculate euclidean distance between points x and y
    Args:
        x, y: two points in Euclidean n-space
    Returns:
        Length of the line segment connecting given points
    """
    dlt = (x - y) * (x - y)
    return (dlt.sum()) ** 0.5


def euclidean_similarity(x: np.array, y: np.array) -> float:
    """
    Calculate euclidean similarity between points x and y
    Args:
        x, y: two points in Euclidean n-space
    Returns:
        Similarity between points x and y
    """
    return 1.0 / (1.0 + euclidean_distance(x, y))


def pearson_similarity(x: np.array, y: np.array) -> float:
    """
    Calculate a Pearson correlation coefficient given 1-D data arrays x and y
    Args:
        x, y: two points in n-space
    Returns:
        Pearson correlation between x and y
    """
    x_mean, y_mean = x.mean(), y.mean()
    dx, dy = (x - x_mean), (y - y_mean)
    return (dx * dy).sum() / (((dx * dx).sum() * (dy * dy).sum()) ** 0.5)


def apk(actual: np.array, predicted: np.array, k: int = 10) -> float:
    """
    Compute the average precision at k
    Args:
        actual: a list of elements that are to be predicted (order doesn't matter)
        predicted: a list of predicted elements (order does matter)
        k: the maximum number of predicted elements
    Returns:
        The average precision at k over the input lists
    """
    pak_sum = 0.0
    for m in range(1, k + 1):
        if predicted[m - 1] in actual:
            correct_num = 0
            for el in predicted[0:m]:
                correct_num += 1 if el in actual else 0
            pak_sum += correct_num / m
    return pak_sum / k


def mapk(actual: np.array, predicted: np.array, k: int = 10) -> float:
    """
    Compute the mean average precision at k
    Args:
        actual: a list of lists of elements that are to be predicted
        predicted: a list of lists of predicted elements
        k: the maximum number of predicted elements
    Returns:
        The mean average precision at k over the input lists
    """
    n = len(predicted)
    apk_sum = sum(list(map(lambda el: apk(el[0], el[-1], k), zip(actual, predicted))))
    return float(apk_sum) / n
