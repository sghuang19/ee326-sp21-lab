import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from numpy.fft import fft2, fftshift, ifft2

np.seterr(divide='ignore', invalid='ignore')

# %% read images and fft
img_raw = np.asarray(Image.open('Q5_3.tif'))
row, col = img_raw.shape

img_fourier = fftshift(fft2(np.pad(img_raw, ((0, row), (0, col)))))
img_view = np.log10(np.abs(img_fourier) + 1).astype(np.uint8)

plt.subplot(121)
plt.imshow(img_raw, cmap='gray')
plt.title('Fourier transform of Q5_3.tif\n(Shifted and log scaled)')

plt.subplot(122)
plt.imshow(img_view, cmap='gray')
plt.title('Fourier transform of Q5_3.tif\n(Shifted and log scaled)')
plt.savefig('Q5_3_1.png')
plt.show()


# %% filter gen


def butterworth(n: int = 4, d0: int = 20, center=(row / 2, col / 2)):
    def d(mu: int, nu: int, center=center):
        center_row, center_col = center

        return np.sqrt((mu - row / 2 - center_row) ** 2 + (nu - col / 2 - center_col) ** 2)

    notch_filter = np.fromfunction(lambda mu, nu: 1 / (1 + (d0 / d(mu, nu)) ** (2 * n)),
                                   (2 * row, 2 * col)) * \
                   np.fromfunction(lambda mu, nu: 1 / (1 + (d0 / d(-mu, -nu)) ** (2 * n)),
                                   (2 * row, 2 * col))
    plt.imshow(notch_filter)
    plt.show()

    return notch_filter


filter0 = butterworth()
plt.imshow(filter0)
plt.show()

filter1 = butterworth(center=(-0.15 * row, 0.1 * col))
filter2 = butterworth(center=(-0.15 * row, 0.8 * col))

filter3 = butterworth(center=(0.15 * row, 0.1 * col))
filter4 = butterworth(center=(0.15 * row, 0.8 * col))

filter5 = butterworth(center=(0.85 * row, 0.2 * col))
filter6 = butterworth(center=(0.8 * row, 0.85 * col))

filter7 = butterworth(center=(1.2 * row, 0.1 * col))
filter8 = butterworth(center=(1.2 * row, 0.8 * col))

filter_total = filter1 * filter2 * filter3 * filter4 * filter5 * filter6 * filter7 * filter8
plt.imshow(filter_total)
plt.title('Butterworth filter with 8 notch pairs')
plt.savefig('Q5_3_2.png')
plt.show()

# %% filtering

img_filtered_fourier = img_fourier * filter_total
img_filtered_fourier_view = np.log10(np.abs(img_filtered_fourier) + 1).astype(np.uint8)
img_filtered = np.real(ifft2(fftshift(img_filtered_fourier)))[0:row, 0:col]

plt.subplot(121)
plt.imshow(img_filtered_fourier_view, cmap='gray')
plt.title('Fourier transform of Q5_3.tif\n(Shifted and log scaled)')

plt.subplot(122)
plt.imshow(img_filtered, cmap='gray')
plt.title('Fourier transform of Q5_3.tif\n(Shifted and log scaled)')
plt.savefig('Q5_3_3.png')
plt.show()
