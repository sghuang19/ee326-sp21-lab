import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from numpy.fft import fft2, fftshift, ifft2

# %% read images
img_raw = np.asarray(Image.open('Q5_1.tif'))
row, col = img_raw.shape

# %% spatial filtering
sobel_filter = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])

img_pad = np.pad(img_raw, 1)
img_spatial = np.asarray([(img_pad[i:i + 3, j:j + 3] * sobel_filter).sum()
                          for i in range(row)
                          for j in range(col)]).reshape(row, col)

plt.subplot(121)
plt.imshow(img_raw, cmap='gray')
plt.title('Raw Q5_1.tif')

plt.subplot(122)
plt.imshow(img_spatial, cmap='gray')
plt.title('Sobel filtered Q5_1.tif\n(in spatial domain)')
# plt.savefig('Q5_1_1.png')
plt.show()

# %% freq domain filtering
img_fourier = fftshift(fft2(np.pad(img_raw, ((0, row), (0, col)))))
filter_fourier = fftshift(fft2(np.pad(sobel_filter, ((0, 2 * row - 3), (0, 2 * col - 3)))))

img_view = np.log10(np.abs(img_fourier) + 1).astype(np.uint8)
filter_view = np.abs(filter_fourier).astype(np.uint8)

plt.subplot(121)
plt.imshow(img_view, cmap='gray')
plt.title('Fourier transform of Q5_1.tif\n(Shifted and log transformed)')

plt.subplot(122)
plt.imshow(filter_view, cmap='gray')
plt.title('Fourier transform of Sobel Filter\n(Shifted and log transformed)')
plt.savefig('Q5_1_2.png')
plt.show()

img_freq = np.real(ifft2(fftshift(img_fourier * filter_fourier)))[0:row, 0:col]

plt.subplot(121)
plt.imshow(img_raw, cmap='gray')
plt.title('Raw Q5_1.tif')

plt.subplot(122)
plt.imshow(img_freq, cmap='gray')
plt.title('Sobel Filtered Q5_1.tif\n(in frequency domain)')
plt.savefig('Q5_1_3.png')
plt.show()

# %% no fft shift
img_fourier_no_shift = fft2(np.pad(img_raw, ((0, row), (0, col))))
img_view_no_shift = np.log10(np.abs(img_fourier_no_shift) + 1).astype(np.uint8)
img_freq_no_shift = np.real(ifft2(img_fourier * filter_fourier))[0:row, 0:col]

plt.subplot(121)
plt.imshow(img_view_no_shift, cmap='gray')
plt.title('Fourier transform of Q5_1.tif\n(Not shifted, log transformed)')

plt.subplot(122)
plt.imshow(img_freq_no_shift, cmap='gray')
plt.title('Sobel Filtered Q5_1.tif\n(in frequency domain, not shifted)')
plt.savefig('Q5_1_4.png')
plt.show()
