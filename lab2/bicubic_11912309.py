import numpy as np
import cv2 as cv
from scipy.interpolate import interp2d
from matplotlib import pyplot as plt


def bicubic_11912309(input_file: str, dim, output_file: str = 'bicubic_test.tif') -> np.ndarray:
    """
    Use Python function “interp2” from packet “scipy” or your own written algorithm to interpolate a grey scale image
    by using bicubic interpolation.

    :param input_file: the file name that to be interpolated
    :param dim: a 1 by 2 vector specifying the row and column numbers of the interpolated image
    :param output_file: the file name that is interpolated
    :return: processed image
    """

    target_row, target_col = dim
    raw_pic = cv.imread(input_file, cv.IMREAD_GRAYSCALE)
    raw_row, raw_col = raw_pic.shape

    target_x = np.linspace(0, raw_row - 1, num=target_row)
    target_y = np.linspace(0, raw_col - 1, num=target_col)

    f = interp2d(range(raw_row), range(raw_col), raw_pic, kind='cubic')
    target_pic = f(target_x, target_y)

    cv.imwrite(output_file, target_pic.astype(np.uint8))
    plt.imshow(target_pic, cmap='gray')
    plt.show()
    return target_pic


# %% bicubic interpolation
dim_enlarged = [round(256 * (1 + 9 / 10))] * 2
dim_shrank = [round(256 * 9 / 10)] * 2
raw_file = "rice.tif"

bicubic_11912309(raw_file, dim_enlarged, "enlarged_bicubic_11912309.png")
bicubic_11912309(raw_file, dim_enlarged, "enlarged_bicubic_11912309.tif")
bicubic_11912309(raw_file, dim_shrank, "shrank_bicubic_11912309.png")
bicubic_11912309(raw_file, dim_shrank, "shrank_bicubic_11912309.tif")
