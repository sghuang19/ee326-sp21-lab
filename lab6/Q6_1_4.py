import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

# %% Q6_1_4 alpha-trimmed mean filter

Q6_1_4 = np.asarray(Image.open("Q6_1_4.tiff"))
plt.show()


def alpha_filter(img_raw, n: int = 3, d=0.1):
    m = (n - 1) // 2
    trimmed = max(int(d * n ** 2), 1)
    row, col = img_raw.shape
    img_pad = np.pad(img_raw, m)

    img_out = np.array([np.mean(np.sort(img_pad[i:i + n, j:j + n].flat)
                                [trimmed:-trimmed])
                        for i in range(row)
                        for j in range(col)])

    return img_out.astype(np.uint8).reshape(row, col)


# %% test different filter size
Q6_1_4_n3 = alpha_filter(Q6_1_4, 3)
Q6_1_4_n5 = alpha_filter(Q6_1_4, 5)
Q6_1_4_n7 = alpha_filter(Q6_1_4, 7)

plt.figure(figsize=(6.4, 7.2))
plt.subplot(221)
plt.title('Raw Q6_1_4.tiff')
plt.imshow(Q6_1_4, cmap='gray')
plt.subplot(222)
plt.title('Alpha-trimmed mean filtered\nwith filter size $n=3$, $d=2$')
plt.imshow(Q6_1_4_n3, cmap='gray')
plt.subplot(223)
plt.title('Alpha-trimmed mean filtered\nwith filter size $n=5$, $d=2$')
plt.imshow(Q6_1_4_n5, cmap='gray')
plt.subplot(224)
plt.title('Alpha-trimmed mean filtered\nwith filter size $n=7$, $d=2$')
plt.imshow(Q6_1_4_n7, cmap='gray')
plt.savefig('Q6_1_4_n.png')
plt.show()

# %% test the d param in alpha_filter
Q6_1_4_d1 = alpha_filter(Q6_1_4, d=1 / 9)
Q6_1_4_d2 = alpha_filter(Q6_1_4, d=2 / 9)
Q6_1_4_d3 = alpha_filter(Q6_1_4, d=3 / 9)

plt.figure(figsize=(6.4, 7.2))
plt.subplot(221)
plt.title('Raw Q6_1_4.tiff')
plt.imshow(Q6_1_4, cmap='gray')
plt.subplot(222)
plt.title('Alpha-trimmed mean filtered\nwith filter size $n=3$, $d=1/9$')
plt.imshow(Q6_1_4_d1, cmap='gray')
plt.subplot(223)
plt.title('Alpha-trimmed mean filtered\nwith filter size $n=3$, $d=2/9$')
plt.imshow(Q6_1_4_d2, cmap='gray')
plt.subplot(224)
plt.title('Alpha-trimmed mean filtered\nwith filter size $n=3$, $d=3/9$')
plt.imshow(Q6_1_4_d3, cmap='gray')
plt.savefig('Q6_1_4_d.png')
plt.show()

# %% test for multiple median filter
Q6_1_4_1 = alpha_filter(Q6_1_4, d=4 / 9)
Q6_1_4_2 = alpha_filter(Q6_1_4_1, d=4 / 9)
Q6_1_4_3 = alpha_filter(Q6_1_4_2, d=4 / 9)
Q6_1_4_4 = alpha_filter(Q6_1_4_3, d=4 / 9)

plt.figure(figsize=(6.4, 7.2))
plt.subplot(221)
plt.title('Median filtered once\nwith filter size $n=3$')
plt.imshow(Q6_1_4_1, cmap='gray')
plt.subplot(222)
plt.title('Median filtered twice\nwith filter size $n=3$')
plt.imshow(Q6_1_4_2, cmap='gray')
plt.subplot(223)
plt.title('Median filtered thrice\nwith filter size $n=3$')
plt.imshow(Q6_1_4_3, cmap='gray')
plt.subplot(224)
plt.title('Median filtered quadrice\nwith filter size $n=3$')
plt.imshow(Q6_1_4_4, cmap='gray')
plt.savefig('Q6_1_4_m.png')
plt.show()
