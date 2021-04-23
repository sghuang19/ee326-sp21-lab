import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


# %% Q6_1_2 min filter
Q6_1_2 = np.asarray(Image.open("Q6_1_2.tiff"))
plt.show()


def min_filter(img_raw, n: int):
    m = (n - 1) // 2
    row, col = img_raw.shape
    img_pad = np.pad(img_raw, m)
    img_out = np.array([img_pad[i:i + n, j:j + n].min()
                        for i in range(row)
                        for j in range(col)])
    return img_out.astype(np.uint8).reshape(row, col)


Q6_1_2_n3 = min_filter(Q6_1_2, 3)
Q6_1_2_n5 = min_filter(Q6_1_2, 5)
Q6_1_2_n7 = min_filter(Q6_1_2, 7)

plt.figure(figsize=(6.4, 6.4))
plt.subplot(221)
plt.title('Raw Q6_1_2.tiff')
plt.imshow(Q6_1_2, cmap='gray')
plt.subplot(222)
plt.title('Min filtered with filter size $n=3$')
plt.imshow(Q6_1_2_n3, cmap='gray')
plt.subplot(223)
plt.title('Min filtered with filter size $n=5$')
plt.imshow(Q6_1_2_n5, cmap='gray')
plt.subplot(224)
plt.title('Min filtered with filter size $n=7$')
plt.imshow(Q6_1_2_n7, cmap='gray')
plt.savefig('Q6_1_2.png')
plt.show()
