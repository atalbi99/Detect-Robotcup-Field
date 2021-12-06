import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb

def erase_component(mask):
    #The method receives an binary image
    #find all your connected components
    #connectedComponentswithStats yields every seperated component with information on each of them, such as size
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    #the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    # minimum size of particles we want to keep (number of pixels)
    min_size = 2000

    #your answer image
    img2 = np.zeros((output.shape))
    #for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255

    #kernel = np.ones((12,12),np.uint8)
    #dilation = cv2.dilate(img2,kernel,iterations = 1)
    cv2.imshow('mask', img2)


# smoothening images and reducing noise
img2 = cv2.imread('2.png')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img = cv2.medianBlur(img2, 15)
imgb = cv2.bilateralFilter(img2, 15, 75, 75)

light_green = (29, 28, 20)
dark_green = (68, 234, 255)
light_green1 = (29, 28, 20)
dark_green1 = (68, 234, 255)

hsv_imgb = cv2.cvtColor(imgb, cv2.COLOR_RGB2HSV)
#hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

mask1 = cv2.inRange(hsv_imgb, light_green, dark_green)
cv2.imshow('mask1', mask1)
#mask2 = cv2.inRange(hsv_img, light_green, dark_green)
mask2 = cv2.inRange(hsv_imgb, light_green1, dark_green1)
cv2.imshow('mask2', mask2)
mask = mask1 + mask2
cv2.imshow('mask', mask)

result =  cv2.bitwise_and(img, img, mask=mask1)
img=imgb

cv2.imwrite("mask.jpg", mask)
#connected_component_label(mask)
erase_component(mask)
cv2.waitKey(0)  
cv2.destroyAllWindows()
#Contour detection
