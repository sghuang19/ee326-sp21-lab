import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize


def gradient(raw_img: np.ndarray, operator: str = 'sobel'):
    row, col = raw_img.shape
    i = j = 0
    gradient = np.zeros((row, col))

    def gamma_trans(img, gamma):
        gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
        gamma_table = np.round(np.array(gamma_table)).astype(int)
        return cv.LUT(img, gamma_table)

    if operator == 'sobel':
        filter = (np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]]),
                  np.array([[-1, 0, 1],
                            [-2, 0, 0],
                            [1, 2, 1]]))

        temp_img = np.pad(raw_img, 1)
        while i < row:
            while j < col:
                local_img = temp_img[i:i + 3, j:j + 3]
                gradient[i][j] = abs((local_img * filter[0]).sum()) + abs((local_img * filter[1]).sum())
                j += 1
            j = 0
            i += 1
    elif operator == 'roberts':
        filter = (np.array([[-1, 0],
                            [0, 1]]),
                  np.array([[0, -1],
                            [1, 0]]))

        temp_img = np.pad(raw_img, [0, 1])
        while i < row:
            while j < col:
                local_img = temp_img[i:i + 2, j:j + 2]
                gradient[i][j] = abs((local_img * filter[0]).sum()) + \
                                 abs((local_img * filter[1]).sum())
                j += 1
            i += 1
            j = 0
    raw_mean = raw_img.mean()
    # gradient = gradient * (raw_mean / gradient.mean())
    gradient = gradient * (255. / gradient.max())
    out_img = raw_img + gradient
    # out_img = out_img.astype(np.uint8)

    if operator == 'sobel':
        out_img = out_img * (255. / out_img.max())
        out_img = out_img * (raw_mean / out_img.mean())
        # out_img = gamma_trans(out_img.astype(np.uint8), 0.75 )
    if operator == 'roberts':
        out_img = np.clip(out_img, a_min=0, a_max=255)
    #     out_img = gamma_trans(out_img, 0.4)

    norm = Normalize(vmin=0, vmax=255)
    plt.figure(figsize=(6.4 * 3, 6.4))
    plt.subplot(131)
    plt.title('Raw image')
    plt.imshow(raw_img, cmap='gray', norm=norm)
    plt.subplot(132)
    plt.title('Enhanced image')
    plt.imshow(out_img, cmap='gray', norm=norm)
    plt.subplot(133)
    plt.title(operator + ' gradient')
    plt.imshow(gradient, cmap='gray', norm=norm)
    plt.show()
    return out_img.astype(int), gradient


# %%
raw_img_1 = cv.imread('Q4_1.tif', cv.IMREAD_GRAYSCALE)
out_img_1, gradient_1 = gradient(raw_img_1, operator='sobel')
out_img_2, gradient_2 = gradient(raw_img_1, operator='roberts')

# %%
raw_img_2 = cv.imread('Q4_2.tif', cv.IMREAD_GRAYSCALE)
out_img_3, gradient_3 = gradient(raw_img_2, operator='sobel')
out_img_4, gradient_4 = gradient(raw_img_2, operator='roberts')

# %%
img = cv.fastNlMeansDenoising(raw_img_2, h=10, templateWindowSize=7, searchWindowSize=21)
out_img_5, gradient_5 = gradient(img, operator='sobel')
out_img_6, gradient_6 = gradient(img, operator='roberts')
