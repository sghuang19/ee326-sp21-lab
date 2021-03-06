import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from histogram import histogram
from matplotlib.colors import Normalize


def hist_equ(input_image: str):
    """
    Implement the histogram equalization to the input images Q3_1_1.tif and Q3_1_2.tif
    """

    norm = Normalize(vmin=0, vmax=255)

    raw_img = cv.imread(input_image, cv.IMREAD_GRAYSCALE)

    L = 2 ** 8
    r_min = raw_img.min()
    r_max = raw_img.max()
    row, col = raw_img.shape

    # input_hist = np.zeros(L, int)
    # for i in raw_img.flat:
    #     input_hist[i] += 1

    input_hist = histogram(raw_img)
    print(input_image, 'raw', np.count_nonzero(input_hist))

    s = np.zeros(L, int)
    for k in range(L):
        s[k] = (L - 1) * sum(input_hist[:k + 1])

    out_img = np.array([s[r] for r in raw_img], int).reshape(row, col)

    output_hist = histogram(out_img)
    print(input_image, 'equalized', np.count_nonzero(output_hist))

    # %% plots
    plt.subplot(121)
    plt.imshow(raw_img, cmap='gray', norm=norm)
    plt.title("Raw " + input_image)

    plt.subplot(122)
    plt.imshow(out_img, cmap='gray', norm=norm)
    plt.title("Equalized " + input_image)
    plt.savefig(input_image + "_comparison.png")
    plt.show()

    plt.title("Histogram of " + input_image)
    plt.bar(range(L), input_hist)
    plt.bar(range(L), output_hist)
    plt.legend(('raw image', 'equalized image'))
    plt.savefig(input_image + "_histogram.png")
    plt.show()

    plt.plot(range(L), s)
    plt.title("Histogram equalization transformation for " + input_image)
    plt.xlabel('$r_k$')
    plt.ylabel('$s_k$')
    plt.show()

    return out_img, output_hist, input_hist, s


# %%

*_, trans_1 = hist_equ("Q3_1_1.tif")
*_, trans_2 = hist_equ("Q3_1_2.tif")

plt.plot(range(2 ** 8), trans_1)
plt.plot(range(2 ** 8), trans_2)
plt.title("Histogram equalization transformation")
plt.xlabel('$r_k$')
plt.ylabel('$s_k$')
plt.legend(('Q3_1_1.tif', 'Q3_1_2.tif'))
plt.savefig("Q3_trans.png")
plt.show()
