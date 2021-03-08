import numpy as np
import cv2 as cv


def hist_equ(input_image: str):
    raw_img = cv.imread(input_image, cv.IMREAD_GRAYSCALE)
    L = 2 ** 8
    bins = range(L + 1)
    input_hist, _ = np.histogram(raw_img.flat, bins=bins, density=True)
    s = np.array([(L - 1) * sum(input_hist[:k + 1]) for k in range(L)])
    out_img = np.array([s[r] for r in raw_img], int).reshape(raw_img.shape)
    output_hist, _ = np.histogram(out_img.flat, bins=bins, density=True)
    return out_img, output_hist, input_hist
