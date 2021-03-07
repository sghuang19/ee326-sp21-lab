import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize


def hist_match(input_image, spec_hist=None):
    """
    Specify a histogram for image Q3_2.tif, such that by matching the histogram of Q3_2.tif to the specified one,
    the image is enhanced. Implement the specified histogram matching to the input image Q3_2.tif. You may refer to
    the histogram given in the Lecture Notes 3 page 49, but not necessary to use the same one. in your report.
    """

    raw_img = cv.imread(input_image, cv.IMREAD_GRAYSCALE)
    L = 2 ** 8
    if spec_hist is None:
        spec_hist = np.array([1 / L] * L)
    # spec_hist = spec_hist / sum(spec_hist)

    bins = range(L + 1)

    input_hist, _ = np.histogram(raw_img.flat, bins=bins, density=True)
    s = np.array([(L - 1) * sum(input_hist[:k + 1]) for k in range(L)], int)
    g = np.array([(L - 1) * sum(spec_hist[:q + 1]) for q in range(L)], int)
    # the index of g is implicitly z
    z = np.array([np.where(g == e)[0][0] if len(np.where(g == e)[0]) != 0 else 0 for e in s], int)

    out_img = np.array([z[r] for r in raw_img], int).reshape(raw_img.shape)
    output_hist, _ = np.histogram(out_img.flat, bins=bins, density=True)

    # %% plots
    norm = Normalize(vmin=0, vmax=255)

    plt.subplot(121)
    plt.imshow(raw_img, cmap='gray', norm=norm)
    plt.title("Raw " + input_image)

    plt.subplot(122)
    plt.imshow(out_img, cmap='gray', norm=norm)
    plt.title("Enhanced " + input_image)
    # plt.savefig(input_image + "_comparison.png")
    plt.show()

    plt.figure(figsize=(9.6, 4.8))
    plt.subplot(131)
    plt.title("Histogram of raw " + input_image)
    plt.bar(range(L), input_hist)
    # plt.xticks([])

    plt.subplot(132)
    plt.title("Specified histogram")
    plt.bar(range(L), spec_hist)
    # plt.xticks([])

    plt.subplot(133)
    plt.title("Histogram of enhanced " + input_image)
    plt.bar(range(L), output_hist)
    # plt.legend(('raw image', 'specified histogram', 'matched image'))
    # plt.savefig(input_image + "_histogram.png")
    plt.show()

    plt.plot(z)
    plt.title("Histogram matching transformation for " + input_image)
    plt.xlabel('$r_k$')
    plt.ylabel('$z_k$')
    plt.show()

    return out_img, output_hist, input_hist  # , z


# %%
out_img_1, out_hist_1, *_ = hist_match('Q3_2.tif')
# out_img_1, out_hist_1, _, z1 = hist_match('Q3_2.tif')

# %%
hist = np.concatenate((np.linspace(0, 7, 8 - 0),
                       np.linspace(7, 0.75, 16 - 8),
                       np.linspace(0.75, 0, 184 - 16),
                       np.linspace(0, 0.5, 200 - 184),
                       np.linspace(0.5, 0, 256 - 200)), axis=0)
hist = hist / sum(hist)
plt.plot(hist)
plt.title("Histogram specified")
plt.xlabel('intensity')
plt.ylabel('percentage')
plt.show()

out_img_2, *_ = hist_match('Q3_2.tif', hist)
# out_img_2, *_, z2 = hist_match('Q3_2.tif', hist)
norm = Normalize(vmin=0, vmax=255)
plt.subplot(121)
plt.imshow(out_img_1, cmap='gray', norm=norm)
plt.title("Histogram equalized")

plt.subplot(122)
plt.imshow(out_img_2, cmap='gray', norm=norm)
plt.title("Enhanced with a specified histogram")
# plt.savefig(input_image + "_comparison.png")
plt.show()

# plt.plot(z1)
# plt.plot(z2)
# plt.title("Histogram matching transformation")
# plt.legend(('equalized histogram', 'specified histogram'))
# plt.show()
