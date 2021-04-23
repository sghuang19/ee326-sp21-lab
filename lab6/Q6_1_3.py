import numpy as np
from PIL import Image
from matplotlib import pyplot as plt



# %% Q6_1_3 median filter

Q6_1_3 = np.asarray(Image.open("Q6_1_3.tiff"))
plt.show()


def median_filter(img_raw, n: int = 3):
    m = (n - 1) // 2
    row, col = img_raw.shape
    img_pad = np.pad(img_raw, m)
    img_out = np.array([np.median(img_pad[i:i + n, j:j + n])
                        for i in range(row)
                        for j in range(col)])
    return img_out.astype(np.uint8).reshape(row, col)


Q6_1_3_n3 = median_filter(Q6_1_3, 3)
Q6_1_3_n5 = median_filter(Q6_1_3, 5)
Q6_1_3_n7 = median_filter(Q6_1_3, 7)

plt.figure(figsize=(7.2, 6.4))
plt.subplot(221)
plt.title('Raw Q6_1_3.tiff')
plt.imshow(Q6_1_3, cmap='gray')
plt.subplot(222)
plt.title('Median filtered with filter size $n=3$')
plt.imshow(Q6_1_3_n3, cmap='gray')
plt.subplot(223)
plt.title('Median filtered with filter size $n=5$')
plt.imshow(Q6_1_3_n5, cmap='gray')
plt.subplot(224)
plt.title('Median filtered with filter size $n=7$')
plt.imshow(Q6_1_3_n7, cmap='gray')
plt.savefig('Q6_1_3.png')
plt.show()
