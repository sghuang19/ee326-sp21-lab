import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from numpy.fft import fft2, ifft2, fftshift, ifftshift

# %% read in the image and do transform
Q6_2 = np.array(Image.open("Q6_2.tif"))
row, col = Q6_2.shape
img_fourier = fftshift(fft2(Q6_2))

# %% generate degradation function
k = 0.0025
func = np.fromfunction(lambda mu, nu:
                       np.exp(-k * ((mu - row / 2) ** 2 + (nu - col / 2) ** 2) ** (5 / 6)),
                       (row, col))


# %% restore the image
def wiener(fourier=img_fourier, K=0.1):
    img_out_fourier = func ** 2 / func / (func ** 2 + K) * fourier
    img_out_view = np.log10(np.abs(fourier) + 1)
    img_out = np.abs(ifft2(ifftshift(img_out_fourier)))
    return img_out_fourier, img_out_view, img_out


_, img_out_view1, img_out1 = wiener(K=0.1)
_, img_out_view2, img_out2 = wiener(K=0.01)
_, img_out_view3, img_out3 = wiener(K=0.001)

plt.figure(figsize=(9.6, 6.4))
plt.subplot(231)
plt.imshow(img_out_view1, cmap='gray')
plt.title(
    'Fourier transform of\n'
    'the restored image\n'
    '(log transformed, $K = 0.1$)')
plt.subplot(234)
plt.imshow(img_out1, cmap='gray')
plt.title('Restored image')

plt.subplot(232)
plt.imshow(img_out_view2, cmap='gray')
plt.title(
    'Fourier transform of\n'
    'the restored image\n'
    '(log transformed, $K = 0.01$)')
plt.subplot(235)
plt.imshow(img_out2, cmap='gray')
plt.title('Restored image')

plt.subplot(233)
plt.imshow(img_out_view2, cmap='gray')
plt.title(
    'Fourier transform of\n'
    'the restored image\n'
    '(log transformed, $K = 0.001$)')
plt.subplot(236)
plt.imshow(img_out3, cmap='gray')
plt.title('Restored image')
plt.savefig('Q6_2_3.png')
plt.show()
