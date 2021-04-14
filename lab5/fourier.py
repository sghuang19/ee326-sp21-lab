from numpy.fft import fft2, fftshift, ifft2
from math import pi

import numpy as np
from PIL import Image

# %% read images
img_raw = np.asarray(Image.open('filename'))
row, col = img_raw.shape

img_shift = img_raw * np.fromfunction(lambda x, y: (-1) ** (x + y), (row, col))
img_fourier = np.asarray([(img_raw * np.fromfunction(lambda x, y: (-1) ** (x + y), (row, col)) * np.fromfunction(lambda x, y: np.exp(-1j * 2 * pi * (mu * x / row + nu * y / col)), (row, col))).sum() for mu in range(row) for nu in range(col)]).reshape(row, col)
