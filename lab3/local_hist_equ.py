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
        for j in list(range(0, col - m_size, m_size)) + [col - m_size - 1]:
            local_img = in_img[i:i + m_size + 1, j:j + m_size + 1]
            local_hist, _ = np.histogram(local_img.flat, bins=bins, density=True)
            # s = np.array([(L - 1) * sum(local_hist[:k + 1]) for k in range(L)])

            s = np.array([(L - 1) * local_hist[:k + 1].sum() for k in range(L)])
            # s = np.zeros((1, L))
            # for k in range(L):
            #     s[0, k] = (L - 1) * local_hist[:k + 1].sum()

            out_img[i:i + m_size, j:j + m_size] = np.array([s[r] for r in local_img], int).reshape([m_size] * 2)

            # tmp = np.zeros([1, m_size ** 2])
            # i = 0
            # for r in local_img:
            #     tmp = s[0, r]
            #     i += 1
            # out_img[i:i + m_size, j:j + m_size] = tmp

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
result_3 = local_hist_equ(raw_img, 3)
# result_5 = local_hist_equ(raw_img, 5)
# result_10 = local_hist_equ(raw_img, 10)
# result_20 = local_hist_equ(raw_img, 20)
#
# norm = Normalize(vmin=0, vmax=255)
#
# plt.subplot(221)
# plt.imshow(result_3, cmap='gray', norm=norm)
# plt.title("m_size = 3")
# plt.xticks([])
#
# plt.subplot(222)
# plt.imshow(result_5, cmap='gray', norm=norm)
# plt.title("m_size = 5")
# plt.xticks([])
#
# plt.subplot(223)
# plt.imshow(result_10, cmap='gray', norm=norm)
# plt.title("m_size = 10")
#
# plt.subplot(224)
# plt.imshow(result_20, cmap='gray', norm=norm)
# plt.title("m_size = 20")
# plt.show()
