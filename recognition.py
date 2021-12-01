import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb

# smoothening images and reducing noise
# https://www.geeksforgeeks.org/python-bilateral-filtering/
img2 = cv2.imread('2.png')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img = cv2.medianBlur(img2, 15)
imgb = cv2.bilateralFilter(img2, 15, 75, 75)

low_green = (29, 28, 20)
high_green = (68, 234, 255)

hsv_imgb = cv2.cvtColor(imgb, cv2.COLOR_RGB2HSV)
#hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

mask1 = cv2.inRange(hsv_imgb, low_green, high_green)




result =  cv2.bitwise_and(img, img, mask=mask1)
img=imgb
cv2.imwrite("mask.jpg", mask1)
#plt.subplot(1, 2, 2)
plt.imshow(result)
plt.show()