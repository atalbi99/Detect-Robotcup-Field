import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb

def erase_component(mask):
    # The method receives an binary image, find all your connected components
    # ConnectedComponentswithStats yields every seperated component with information on each of them, such as size
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    # The following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    # Minimum size of particles we want to keep (number of pixels)
    min_size = 20000

    img2 = np.zeros((output.shape))
    # Fo every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255

    return img2

def FillHole(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    len_contour = len (contours)
    contour_list = []
    for i in range(len_contour):
        drawing = np.zeros_like(mask, np.uint8)  # create a black image
        img_contour = cv2.drawContours(drawing, contours, i, (255, 255, 255), -1)
        contour_list.append(img_contour)

    out = sum(contour_list)

    for i in range(len(contours)):
        hull = cv2.convexHull(contours[i])
        cv2.drawContours(out, [hull], -1, (255, 0, 0), 2)
    return out

def contour(image):

    img = cv2.imread(image)
    t = 10

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    (t, binary) = cv2.threshold(blur, t, 255, cv2.THRESH_BINARY)

    contours,hierachy = cv2.findContours(binary, cv2.RETR_CCOMP,
                                        cv2.CHAIN_APPROX_SIMPLE)
    # contours if for all the contours of the image
    print("Found %d objects." % len(contours))
    for (i, c) in enumerate(contours):
        print("\tSize of contour %d: %d" % (i, len(c)))
    # contours length - length of all contours
    contours_length = [len(c) for i, c in enumerate(contours)]
    contours_length.sort()
    # contours temp - length of contours of top 30 percent contours
    contours_temp = [contours_length[i] for i in range(int(len(contours_length) * 0.70), len(contours_length))]
    contours_image = []
    count = 0
    for i in contours:
        if len(i) in contours_temp:
            contours_image.append(i)
            count += 1
    contours_image_len = [len(i) for i in contours_image]
    print('\ncsd\n', contours_image_len)
    cv2.drawContours(img, contours, -1, (0, 0, 0), 15)
    return img

def dice_coef(img, img2):
        if img.shape != img2.shape:
            raise ValueError("Shape mismatch: img and img2 must have to be of the same shape.")
        else:
            
            lenIntersection=0
            
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    if ( np.array_equal(img[i][j],img2[i][j]) ):
                        lenIntersection+=1
             
            lenimg=img.shape[0]*img.shape[1]
            lenimg2=img2.shape[0]*img2.shape[1]  
            value = (2. * lenIntersection  / (lenimg + lenimg2))
        return value

L1 = [10,20,25,53,62,68,79,86,95,98,103,105,109,271,279,330]
L2 = [18,33,37,43,53,55,89,154,193,198,201,236]
L3 = [1,11,15,72,96,150,163,167,208,217]
L4 = [22,24,31,35]
dice_score =[]

for img_number in L4 :

    cadena = '{0:02d}'.format(img_number) # Use the format 0:03d for L1,L2,L3 
    path = 'data/log4/'+str(cadena)+'-rgb.png' 
    img2 = cv2.imread(path)
    print(path)
    #cv2.imshow('Image initiale',img2)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img = cv2.medianBlur(img2, 15)

    light_green = (29, 28, 20)
    dark_green = (68, 234, 255)
    light_green1 = (29, 28, 20)
    dark_green1 = (68, 234, 255)

    
    hsv_imgb = cv2.cvtColor(img2, cv2.COLOR_RGB2HSV)

    mask1 = cv2.inRange(hsv_imgb, light_green, dark_green)
    mask2 = cv2.inRange(hsv_imgb, light_green1, dark_green1)
    mask = mask1 + mask2
    result =  cv2.bitwise_and(img, img, mask=mask1)
    img=img2

    # File hole
    mask_out = FillHole(mask)
    mask_out1 = FillHole(mask_out)
    # Erase component
    erased = erase_component(mask_out1)
    cv2.imwrite('erased.png',erased)

    filename = 'erased.png'
    t = 10
    image1 = contour(filename)
    cv2.imwrite("out2.png",image1)

    imageb = cv2.imread('out2.png')
    imageb = cv2.cvtColor(imageb, cv2.COLOR_BGR2GRAY)

    medianb = cv2.medianBlur(imageb, 15)

    cv2.imwrite("afterblur.jpg", medianb)


    image_gray = cv2.imread("afterblur.jpg", 0)
    image_gray = np.where(image_gray > 30, 255, image_gray)
    image_gray = np.where(image_gray <= 30, 0, image_gray)

    # Draw the result
    image = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)


    path_result = 'data_result/log4/'+str(cadena)+'-rgb_result.png'
    path_test = 'mask-field/log4/'+str(cadena)+'-rgb.png'
    cv2.imwrite(path_result,image)
    image_test = cv2.imread(path_test)

    value = dice_coef(image, image_test)
    dice_score.append(value)
    img_number += 1
    cv2.waitKey()
    cv2.destroyAllWindows()

print(dice_score)