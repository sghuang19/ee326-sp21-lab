import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from numpy.fft import fft2, fftshift, ifft2

# %% read images
img_raw = np.asarray(Image.open('Q5_2.tif'))
row, col = img_raw.shape

# %% fft
img_fourier = fft2(np.pad(img_raw, ((0, row), (0, col))))
img_view = np.log10(fftshift(np.abs(img_fourier)) + 1).astype(np.uint8)

plt.subplot(121)
plt.imshow(img_raw, cmap='gray')
plt.title('Raw Q5_2.tif')

plt.subplot(122)
plt.imshow(img_view, cmap='gray')
plt.title('Fourier Transformed Q5_2.tif')
# plt.savefig('Q5_2.png')
plt.show()


# %% low pass filtering
def gaussian(d0: int):
    def d(mu: int, nu: int):
        return np.sqrt((mu - row) ** 2 + (nu - col) ** 2)

    gaussian_lpf = np.fromfunction(lambda mu, nu: np.exp(-d(mu, nu) ** 2 / (2 * d0 ** 2)),
                                   (2 * row, 2 * col))
    img_freq_filtered = img_fourier * fftshift(gaussian_lpf)
    img_freq_filtered_view = np.log10(fftshift(np.abs(img_freq_filtered)) + 1).astype(np.uint8)
    img_filtered = np.real(ifft2(img_freq_filtered))[0:row, 0:col].astype(np.uint8)

    # gaussian_lpf = np.ones((row * 2, col * 2)) - gaussian_lpf

    return img_filtered, gaussian_lpf


filtered10, filter10 = gaussian(10)
filtered30, filter30 = gaussian(30)
filtered60, filter60 = gaussian(60)
filtered160, filter160 = gaussian(160)

plt.subplot(121)
plt.imshow(img_raw, cmap='gray')
plt.title('Raw Q5_2.tif')

plt.subplot(122)
plt.imshow(filtered10, cmap='gray')
# plt.title('Gaussian low pass filtered Q5_2.tif\n$D_0=30$')
plt.title('Gaussian high-pass filtered Q5_2.tif\n$D_0=10$')
# plt.savefig('Q5_2_10_high.png')
plt.show()

# %%
plt.figure(figsize=(12.8, 4.8 * 2))
plt.subplot(231)
plt.imshow(filtered30, cmap='gray')
plt.title('Gaussian low-pass filtered image\n$D_0=30$')
# plt.title('Gaussian high-pass filtered image\n$D_0=30$')

plt.subplot(232)
plt.imshow(filtered60, cmap='gray')
plt.title('Gaussian low-pass filtered image\n$D_0=60$')
# plt.title('Gaussian high-pass filtered image\n$D_0=60$')

plt.subplot(233)
plt.imshow(filtered160, cmap='gray')
plt.title('Gaussian low-pass filtered image, $D_0=160$')
# plt.title('Gaussian high-pass filtered image\n$D_0=160$')

plt.subplot(234)
plt.imshow(filter30, cmap='gray')
plt.title('Gaussian low-pass filter, $D_0=30$')
# plt.title('Gaussian high-pass filter\n$D_0=30$')

plt.subplot(235)
plt.imshow(filter60, cmap='gray')
plt.title('Gaussian low-pass filter, $D_0=60$')
# plt.title('Gaussian high-pass filter\n$D_0=60$')

plt.subplot(236)
plt.imshow(filter160, cmap='gray')
plt.title('Gaussian low-pass filter, $D_0=160$')
# plt.title('Gaussian high-pass filter\n$D_0=160$')
plt.savefig('Q5_2_filtered_low.png')
# plt.savefig('Q5_2_filtered_high.png')

plt.show()

# # %%
# plt.subplot(121)
# plt.imshow(img_raw, cmap='gray')
# plt.title('Raw Q5_2.tif')
#
# plt.subplot(122)
# plt.imshow(filtered30, cmap='gray')
# plt.title('Gaussian low pass filtered Q5_2.tif\n$D_0=30$')
# plt.savefig('Q5_2_30.png')
# plt.show()
#
# plt.subplot(121)
# plt.imshow(img_raw, cmap='gray')
# plt.title('Raw Q5_2.tif')
#
# plt.subplot(122)
# plt.imshow(filtered60, cmap='gray')
# plt.title('Gaussian low pass filtered Q5_2.tif\n$D_0=60$')
# plt.savefig('Q5_2_60.png')
# plt.show()
#
# plt.subplot(121)
# plt.imshow(img_raw, cmap='gray')
# plt.title('Raw Q5_2.tif')
#
# plt.subplot(122)
# plt.imshow(filtered30, cmap='gray')
# plt.title('Gaussian low pass filtered Q5_2.tif\n$D_0=160$')
# plt.savefig('Q5_2_160.png')
# plt.show()
