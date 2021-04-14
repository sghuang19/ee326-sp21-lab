import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize


def laplacian(raw_img: np.ndarray, operator: np.ndarray, c):
    m = (operator.shape[0] - 1) // 2
    row, col = raw_img.shape
    L = 2 ** 8
    bins = range(L + 1)

    input_hist, _ = np.histogram(raw_img.flat, bins=bins, density=True)

    tmp_img = np.pad(raw_img, m)
    edge = np.zeros((row, col), int)

    for i in range(row):
        for j in range(col):
            edge[i, j] = (tmp_img[i:i + 2 * m + 1, j:j + 2 * m + 1] * operator).sum()

    edge_scaled = cv.equalizeHist((edge - edge.min(initial=0)).astype(np.uint8))
    img_enhanced = raw_img + c * edge

    norm = Normalize(vmin=0, vmax=255)

    plt.figure(figsize=(6.4, 6.4))
    plt.subplot(221)
    plt.title('Raw image')
    plt.imshow(raw_img, cmap='gray', norm=norm)

    plt.subplot(222)
    plt.title('Enhanced image')
    plt.imshow(img_enhanced, cmap='gray', norm=norm)

    plt.subplot(223)
    plt.title('Laplacian')
    plt.imshow(edge, cmap='gray', norm=norm)

    plt.subplot(224)
    plt.title('Scaled Laplacian')
    plt.imshow(edge_scaled, cmap='gray', norm=norm)

    plt.show()
    return edge, edge_scaled, img_enhanced


# %%
filter_1 = np.array([[0, 1, 0],
                     [1, -4, 1],
                     [0, 1, 0]])
filter_2 = np.array([[1, 1, 1],
                     [1, -8, 1],
                     [1, 1, 1]])

raw_img_1 = cv.imread("Q4_1.tif", cv.IMREAD_GRAYSCALE)
raw_img_2 = cv.imread("Q4_2.tif", cv.IMREAD_GRAYSCALE)

# %%
out_img_1, *_ = laplacian(raw_img_1, filter_1, -2)
out_img_2, *_ = laplacian(raw_img_1, filter_2, -2)

# %%

plt.hist(raw_img_2.ravel(), bins=255, range=(0, 255))
plt.show()

out_img_3, *_ = laplacian(raw_img_2, filter_1, -2)

# %% median blur
out_img_4, *_ = laplacian(cv.medianBlur(raw_img_2, 3), filter_2, -2)
# %% median blur
out_img_5, *_ = laplacian(cv.GaussianBlur(raw_img_2, (5, 5), 0), filter_2, -2)

# %%
img = raw_img_2
img = cv.fastNlMeansDenoising(img, h=10, templateWindowSize=7, searchWindowSize=21)
# img = cv.medianBlur(img, 11)
# img = cv.GaussianBlur(img, (3, 3), 0)
# img = cv.GaussianBlur(img, (3, 3), 0)
laplacian(img, filter_2, -2)

# %%
img = raw_img_2
img = cv.medianBlur(img, 3)
img = cv.GaussianBlur(img, (3, 3), 0)
# img = cv.GaussianBlur(img, (3, 3), 0)
laplacian(img, filter_2, -2)
