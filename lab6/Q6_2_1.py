import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift

# %% read in the image and do transform

Q6_2 = np.array(Image.open("Q6_2.tif"))
row, col = Q6_2.shape

img_fourier = fftshift(fft2(Q6_2))
img_view = np.log10(np.abs(img_fourier) + 1)

plt.subplot(121)
plt.imshow(Q6_2, cmap='gray')
plt.title('Raw Q6_2.tif')
plt.subplot(122)
plt.imshow(img_view, cmap='gray')
plt.title('Fourier Transformed Q6_2.tif\n(log transformed)')
plt.savefig('Q6_2.png')
plt.show()

# %% generate degradation function
k = 0.0025
func = np.fromfunction(lambda mu, nu:
                       np.exp(-k * ((mu - row / 2) ** 2 + (nu - col / 2) ** 2) ** (5 / 6)),
                       (row, col))

plt.imshow(func, cmap='gray')
plt.title('Degradation function')
plt.savefig('Q6_2_h.png')
plt.show()

# %% restore the image

img_out_fourier = img_fourier / func
img_out_view = np.log10(np.abs(img_out_fourier) + 1)
img_out = np.abs(ifft2(ifftshift(img_out_fourier)))

plt.subplot(121)
plt.imshow(img_out, cmap='gray')
plt.title('Restored Q6_2.tif')
plt.subplot(122)
plt.imshow(img_out_view, cmap='gray')
plt.title('Fourier Transform of\nthe restored image\n(log transformed)')
plt.savefig('Q6_2_1.png')
plt.show()
