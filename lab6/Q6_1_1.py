import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

# %% Q6_1_1 max filter
Q6_1_1 = np.asarray(Image.open("Q6_1_1.tiff"))


def max_filter(img_raw, n: int):
    m = (n - 1) // 2
    row, col = img_raw.shape
    img_pad = np.pad(img_raw, m)
    img_out = np.array([img_pad[i:i + n, j:j + n].max()
                        for i in range(row)
                        for j in range(col)])
    return img_out.astype(np.uint8).reshape(row, col)


Q6_1_1_n3 = max_filter(Q6_1_1, 3)
Q6_1_1_n5 = max_filter(Q6_1_1, 5)
Q6_1_1_n7 = max_filter(Q6_1_1, 7)

plt.figure(figsize=(6.4, 6.4))
plt.subplot(221)
plt.title('Raw Q6_1_1.tiff')
plt.imshow(Q6_1_1, cmap='gray')
plt.subplot(222)
plt.title('Max filtered with filter size $n=3$')
plt.imshow(Q6_1_1_n3, cmap='gray')
plt.subplot(223)
plt.title('Max filtered with filter size $n=5$')
plt.imshow(Q6_1_1_n5, cmap='gray')
plt.subplot(224)
plt.title('Max filtered with filter size $n=7$')
plt.imshow(Q6_1_1_n7, cmap='gray')
plt.savefig('Q6_1_1.png')
plt.show()
