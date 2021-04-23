import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

Q6_1_4 = np.array(Image.open("Q6_1_4.tiff"))


# %% adaptive median filtering method
def adaptive_median(img_raw, s: int = 7):
    # start from window size of one, that is the pixel itself
    n = 1
    m = (n - 1) // 2
    row, col = img_raw.shape

    def window():
        """
        :return: if out of bound or reaching the , return None, if not return the window
        """
        if i - m >= 0 and i + m <= row and j - m >= 0 and j + m <= col and n <= s:
            return img_raw[i - m:i + m + 1, j - m:j + m + 1]

    img_out = np.zeros((row, col))

    for i in range(row):
        for j in range(col):

            # initialize the window and the values in stage A, by initialization the window is single pixel
            
            # reload default window size
            n = 1
            m = (n - 1) // 2

            img_tmp = window()
            a1 = np.median(img_tmp) - np.min(img_tmp)
            a2 = np.median(img_tmp) - np.max(img_tmp)

            # while the window is valid and the requirement tof going stage B is not satisfied, keep looping in stage A
            while img_tmp is not None and not (a1 > 0 and a2 < 0):
                n += 2
                m = (n - 1) // 2
                img_tmp = window()

            # when out of bound, or reaching the maximum windows size, or stage B is available
            # recover the window size to the last one
            n -= 2
            m = (n - 1) // 2
            img_tmp = window()  # get the window

            if a1 > 0 and a2 < 0:  # when stage B is available
                b1 = img_raw[i, j] - np.min(img_tmp)
                b2 = img_raw[i, j] - np.max(img_tmp)
                if b1 > 0 and b2 < 0:
                    img_out[i, j] = img_raw[i, j]
            #     another possible case is combined below
            else:  # stage B is not available, the exit of while loop is due to reaching the max possible window size
                img_out[i, j] = np.median(img_tmp)


    return img_out.astype(np.uint8)


# %% adaptive filtering test
Q6_1_4_a3 = adaptive_median(Q6_1_4, s=3)
Q6_1_4_a5 = adaptive_median(Q6_1_4, s=5)
Q6_1_4_a7 = adaptive_median(Q6_1_4, s=7)

plt.figure(figsize=(6.4, 7.2))
plt.subplot(221)
plt.title('Raw Q6_1_4.tiff')
plt.imshow(Q6_1_4, cmap='gray')
plt.subplot(222)
plt.title('Adaptive median filtered\nwith maximum window size $s=3$')
plt.imshow(Q6_1_4_a3, cmap='gray')
plt.subplot(223)
plt.title('Adaptive median filtered\nwith maximum window size $s=5$')
plt.imshow(Q6_1_4_a5, cmap='gray')
plt.subplot(224)
plt.title('Adaptive median filtered\nwith maximum window size $s=7$')
plt.imshow(Q6_1_4_a7, cmap='gray')
plt.savefig('Q6_1_4_a.png')
plt.show()
