import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from numpy.fft import fft2, ifft2, fftshift, ifftshift

# %% read in the image and do transform
Q6_2 = np.array(Image.open("Q6_2.tif"))
row, col = Q6_2.shape
img_fourier = fftshift(fft2(Q6_2))


# %% butterworth low pass filter
def butterworth(d: int = 40):
    n = 10

    def _func(mu, nu):
        return 1 / (1 + ((mu - row / 2) ** 2 + (nu - col / 2) ** 2) ** n /
                    d ** (2 * n))

    return np.fromfunction(_func, (row, col)).astype(float)


butter40 = butterworth(d=40)
butter70 = butterworth(d=70)
butter85 = butterworth(d=85)

plt.figure(figsize=(9.6, 4.8))
plt.subplot(131)
plt.imshow(butter40, cmap='gray')
plt.title('Butterworth low pass filter\n$d_0=40$')
plt.subplot(132)
plt.imshow(butter70, cmap='gray')
plt.title('Butterworth low pass filter\n$d_0=70$')
plt.subplot(133)
plt.imshow(butter85, cmap='gray')
plt.title('Butterworth low pass filter\n$d_0=85$')
plt.savefig("Q6_2_b.png")
plt.show()

# %% generate degradation function
k = 0.0025
func = np.fromfunction(lambda mu, nu:
                       np.exp(-k * ((mu - row / 2) ** 2 + (nu - col / 2) ** 2) ** (5 / 6)),
                       (row, col))

# %% restore the image
img_out_fourier = img_fourier / func


def restore(butter, fourier=img_out_fourier):
    fourier *= butter
    img_out_view = np.log10(np.abs(fourier) + 1)
    img_out = np.abs(ifft2(ifftshift(img_out_fourier)))
    return fourier, img_out_view, img_out


_, img_out_view40, img_out40 = restore(butter40)
_, img_out_view70, img_out70 = restore(butter70)
_, img_out_view85, img_out85 = restore(butter85)

plt.figure(figsize=(9.6, 6.4))
plt.subplot(231)
plt.imshow(img_out_view40, cmap='gray')
plt.title(
    'Fourier transform of\n'
    'the restored image\n'
    '(log transformed, $d_0 = 40$)')
plt.subplot(234)
plt.imshow(img_out40, cmap='gray')
plt.title('Restored image')

plt.subplot(232)
plt.imshow(img_out_view70, cmap='gray')
plt.title(
    'Fourier transform of\n'
    'the restored image\n'
    '(log transformed, $d_0 = 70$)')
plt.subplot(235)
plt.imshow(img_out70, cmap='gray')
plt.title('Restored image')

plt.subplot(233)
plt.imshow(img_out_view85, cmap='gray')
plt.title(
    'Fourier transform of\n'
    'the restored image\n'
    '(log transformed, $d_0 = 85$)')
plt.subplot(236)
plt.imshow(img_out85, cmap='gray')
plt.title('Restored image')
plt.savefig('Q6_2_2.png')
plt.show()
