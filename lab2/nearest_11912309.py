# %%
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def nearest_11912309(input_file: str, dim, output_file: str = 'test.tif') -> np.ndarray:
    """
    Use nearest neighbor interpolation to interpolate a grey scale image.
    The result is stored in a 8-bit grey .tif file

    :param input_file: the file name that to be interpolated
    :param dim: a 1 by 2 vector specifying the row and column numbers of the interpolated image
    :param output_file: the file name that is interpolated
    :return: processed image
    """

    target_row, target_col = dim
    raw_pic = cv.imread(input_file, cv.IMREAD_GRAYSCALE)
    raw_row, raw_col = raw_pic.shape

    # row_scale = raw_row / target_row
    # col_scale = raw_col / target_col

    # target_pic = np.zeros((target_row, target_col), dtype=np.uint8)
    # for target_x in range(target_row):
    #     raw_x = min(round(target_x * row_scale), raw_row - 1)
    #     for target_y in range(target_col):
    #         raw_y = min(round(target_y * col_scale), raw_col - 1)
    #         target_pic[target_x, target_y] = raw_pic[raw_x, raw_y]

    target_x = np.linspace(0, raw_row - 1, num=target_row)
    target_y = np.linspace(0, raw_col - 1, num=target_col)
    target_pic = np.zeros((target_row, target_col), dtype=np.uint8)

    for x in range(target_row):
        for y in range(target_col):
            # target_pic[x, y] = raw_pic[round(target_x[x]),
            #                            raw_pic[round(target_y[y])]]
            target_pic[x, y] = raw_pic[round(target_x[x]), round(target_y[y])]

    cv.imwrite(output_file, target_pic)
    plt.imshow(target_pic, cmap='gray')
    plt.show()
    return target_pic


# %%
dim_enlarged = [round(256 * (1 + 9 / 10))] * 2
dim_shrunk = [round(256 * 9 / 10)] * 2
raw_file = "rice.tif"


# %% nearest interpolation
nearest_11912309(raw_file, dim_enlarged, 'enlarged_nearest_11912309.png')
nearest_11912309(raw_file, dim_enlarged, 'enlarged_nearest_11912309.tif')
nearest_11912309(raw_file, dim_shrunk, 'shrunk_nearest_11912309.png')
nearest_11912309(raw_file, dim_shrunk, 'shrunk_nearest_11912309.tif')
