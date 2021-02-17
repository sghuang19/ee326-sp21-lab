import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def bilinear_11912309(input_file: str, dim, output_file: str = 'bilinear_test.tif') -> np.ndarray:
    """
    Use bilinear interpolation to interpolate a grey scale image.
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
    target_pic = np.zeros((target_row, target_col), dtype=np.uint8)

    # for x in range(target_row):
    #     if target_x[x] == int(target_x[x]):
    #         for y in range(target_col):
    #             if target_y[y] == int(target_y[y]):
    #                 if target_y[y] == 256 or target_x[x] == 256:
    #                     print('test')
    #                     pass
    #                 target_pic[x, y] = raw_pic[int(
    #                     target_x[x]), int(target_y[y])]
    #             else:
    #                 yf = int(target_y[y])
    #                 yc = yf + 1
    #                 target_pic[x, y] = (target_y[y] - yf) * \
    #                                    (int(raw_pic[int(target_x[x]), yc]) -
    #                                     int(raw_pic[int(target_x[x]), yf]))
    #     else:
    #         xf = int(target_x[x])
    #         xc = xf + 1
    #         for y in range(target_col):
    #             if target_y[y] == int(target_y[y]):
    #                 target_pic[x, y] = (target_x[x] - xf) * \
    #                                    (int(raw_pic[xc, int(target_y[y])]) -
    #                                     int(raw_pic[xf, int(target_y[y])]))
    #             else:
    #                 yf = int(y)
    #                 yc = yf + 1

    #                 xl = (target_x[x] - xf) * \
    #                     (int(raw_pic[xc, yf]) - int(raw_pic[xf, yf]))
    #                 xr = (target_x[x] - xf) * \
    #                     (int(raw_pic[xc, yc]) - int(raw_pic[xf, yc]))

    #                 target_pic[x, y] = (int(target_y[y]) - yf) * (xr - xl)

    for x in range(target_row):
        xf = int(target_x[x])
        xc = min(xf + 1, raw_row - 1)
        for y in range(target_col):
            yf = int(target_y[y])
            yc = min(yf + 1, raw_row - 1)

            vl = raw_pic[xf, yf] + (target_x[x] - xf) * \
                (int(raw_pic[xc, yf]) - int(raw_pic[xf, yf]))
            vr = raw_pic[xf, yc] + (target_x[x] - xf) * \
                (int(raw_pic[xc, yc]) - int(raw_pic[xf, yc]))

            target_pic[x, y] = vl + (target_y[y] - yf) * (vr - vl)

    cv.imwrite(output_file, target_pic)
    plt.imshow(target_pic, cmap='gray')
    plt.show()
    return target_pic


# %%
dim_enlarged = [round(256 * (1 + 9 / 10))] * 2
dim_shrunk = [round(256 * 9 / 10)] * 2

# %% bilinear interpolation
raw_file = "rice.tif"
bilinear_11912309(raw_file, dim_enlarged, "enlarged_bilinear_11912309.png")
bilinear_11912309(raw_file, dim_enlarged, "enlarged_bilinear_11912309.tif")
bilinear_11912309(raw_file, dim_shrunk, "shrunk_bilinear_11912309.png")
bilinear_11912309(raw_file, dim_shrunk, "shrunk_bilinear_11912309.tif")
