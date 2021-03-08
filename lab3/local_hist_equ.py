import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize


def local_hist_equ(in_img, m_size):
    """
    Implement the local histogram equalization to the input images Q3_3.tif
    """

    row, col = in_img.shape
    out_img = np.zeros(in_img.shape, int)

    L = 2 ** 8
    bins = range(L + 1)
    for i in list(range(0, row - m_size, m_size)) + [row - m_size - 1]:
        # i = min(i, row - m_size)
        for j in list(range(0, col - m_size, m_size)) + [col - m_size - 1]:
            # j = min(j, col - m_size)
            local_img = in_img[i:i + m_size, j:j + m_size]
            local_hist, _ = np.histogram(local_img.flat, bins=bins, density=True)
            s = np.array([(L - 1) * sum(local_hist[:k + 1]) for k in range(L)])
            out_img[i:i + m_size, j:j + m_size] = np.array([s[r] for r in local_img], int).reshape((m_size, m_size))

    norm = Normalize(vmin=0, vmax=255)
    plt.subplot(121)
    plt.imshow(raw_img, cmap='gray', norm=norm)
    plt.title("Raw")

    plt.subplot(122)
    plt.imshow(out_img, cmap='gray', norm=norm)
    plt.title("Equalized")
    # plt.savefig(file_name + "_comparison.png")
    plt.show()

    return out_img


raw_img = cv.imread("Q3_3.tif", cv.IMREAD_GRAYSCALE)
local_hist_equ(raw_img, 3)
