import numpy as np


def histogram(image, normalized=True):
    L = 2 ** 8
    row, col = image.shape
    if normalized:
        return np.bincount(image.flat, minlength=L) / (row * col)
    else:
        return np.bincount(image.flat, minlength=L)
