import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread("images/1247.jpg")

mask = np.zeros(img.shape[:2], np.uint8)
mask[106:156, 296:342] = 255
masked_img = cv2.bitwise_and(img, img, mask=mask)

plt.subplot(221), plt.imshow(img)
plt.subplot(222), plt.imshow(mask)
plt.subplot(223), plt.imshow(masked_img)

plt.subplot(224)
color = ('b','g','r')
for i,col in enumerate(color):
    hist = cv2.calcHist([img], [i], , [256], [0, 256])
    plt.plot(hist, color=col)
    plt.xlim([0,256])

plt.show()
