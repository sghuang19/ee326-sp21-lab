import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize


def highboost(raw_img: np.ndarray, k=1):
    blur_img = cv.GaussianBlur(raw_img, (3, 3), 0)
    mask = raw_img - blur_img
    out_img = raw_img + k * mask

    # norm = Normalize(vmin=0, vmax=255)
    plt.figure(figsize=(6.4 * 3, 6.4))

    plt.subplot(131)
    plt.title('Raw image')
    # plt.imshow(raw_img, cmap='gray', norm=norm)
    plt.imshow(raw_img, cmap='gray')

    plt.subplot(132)
    plt.title('Highboosted image')
    # plt.imshow(out_img, cmap='gray', norm=norm)
    plt.imshow(out_img, cmap='gray')

    plt.subplot(133)
    plt.title('Unsharp Mask')
    # plt.imshow(out_img, cmap='gray', norm=norm)
    plt.imshow(out_img, cmap='gray')

    plt.show()

    return out_img, mask


# %%
raw_img_1 = cv.imread("Q4_1.tif", cv.IMREAD_GRAYSCALE)
raw_img_2 = cv.imread("Q4_2.tif", cv.IMREAD_GRAYSCALE)

out_img_1, _ = highboost(raw_img_1)
out_img_2, _ = highboost(raw_img_1, k=2)
out_img_3, _ = highboost(raw_img_1, k=5)

# %%
out_img_4, _ = highboost(raw_img_2)
img = cv.fastNlMeansDenoising(raw_img_2, h=10, templateWindowSize=7, searchWindowSize=21)
out_img_5, _ = highboost(img)
out_img_6, _ = highboost(img, 5)
