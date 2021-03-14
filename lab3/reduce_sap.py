import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize


def reduce_sap(input_image, n_size):
    """
    Implement an algorithm to reduce the salt-and-pepper noise of an image. The input image is Q3_4.tif
    """

    m = (n_size - 1) // 2
    row, col = input_image.shape
    L = 2 ** 8
    bins = range(L + 1)
    input_hist, _ = np.histogram(input_image.flat, bins=bins, density=True)

    temp_image = np.pad(input_image, [m, m], 'constant', constant_values=[0] * 2)
    output_image = np.zeros((row, col))

    for i in range(m, row):
        for j in range(m, col):
            # output_image[i, j] = int(np.median(output_image[i - m:i + m, j - m:j + m]))
            output_image[i - m, j - m] = np.median(temp_image[i - m:i + m, j - m:j + m])

    # output_image = output_image[m:row, m: col]
    output_hist, _ = np.histogram(output_image.flat, bins=bins, density=True)

    return output_image, output_hist, input_hist


raw_img = cv.imread("Q3_4.tif", cv.IMREAD_GRAYSCALE)
out_img, *_ = reduce_sap(raw_img, 3)

norm = Normalize(vmin=0, vmax=255)
plt.subplot(121)
plt.imshow(raw_img, cmap='gray', norm=norm)
plt.title('Raw image')
plt.subplot(122)
plt.title('Median filtering with filter size 3')
plt.imshow(out_img, cmap='gray', norm=norm)
plt.show()

out_img_2, *_ = reduce_sap(raw_img, 5)
plt.title('Median filtering with filter size 5')
plt.imshow(out_img_2, cmap='gray', norm=norm)
plt.show()
