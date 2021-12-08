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
    min_size = 20000

    #your answer image
    img2 = np.zeros((output.shape))
    #for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255

    #kernel = np.ones((12,12),np.uint8)
    #dilation = cv2.dilate(img2,kernel,iterations = 1)
    #cv2.imshow('masked', img2)
    return img2
""""
def FillHole(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    len_contour = len (contours)
    contour_list = []
    for i in range(len_contour):
        drawing = np.zeros_like(mask, np.uint8)  # create a black image
        img_contour = cv2.drawContours(drawing, contours, i, (255, 255, 255), -1)
        contour_list.append(img_contour)

    out = sum(contour_list)
    return out
""" 
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
    # Display the final convex hull image
    #cv2.imshow('ConvexHull', out)
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


img_number = 64

while(img_number < 265) :

    cadena = '{0:03d}'.format(img_number)
    path = 'data/log1/'+str(cadena)+'-rgb.png'
    img2 = cv2.imread(path)
    print(path)
    #cv2.imshow('Image initiale',img2)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img = cv2.medianBlur(img2, 15)
    #imgb = cv2.bilateralFilter(img2, 15, 75, 75)

    light_green = (29, 28, 20)
    dark_green = (68, 234, 255)
    light_green1 = (29, 28, 20)
    dark_green1 = (68, 234, 255)

    #hsv_imgb = cv2.cvtColor(imgb, cv2.COLOR_RGB2HSV)
    hsv_imgb = cv2.cvtColor(img2, cv2.COLOR_RGB2HSV)

    mask1 = cv2.inRange(hsv_imgb, light_green, dark_green)
    #print(mask1.dtype)
    #cv2.imshow('mask1', mask1)
    #mask2 = cv2.inRange(hsv_img, light_green, dark_green)
    mask2 = cv2.inRange(hsv_imgb, light_green1, dark_green1)
    #cv2.imshow('mask2', mask2)
    mask = mask1 + mask2
    #cv2.imshow('mask', mask)

    #result = np zeros ..
    #result[mask2]=img[mask2]
    result =  cv2.bitwise_and(img, img, mask=mask1)
    #cv2.imshow('result',result)
    #img=imgb
    img=img2

    # File hole
    mask_out = FillHole(mask)
    mask_out1 = FillHole(mask_out)
    #cv2.imshow('maskb', mask_out1)
    # Erase component
    erased = erase_component(mask_out1)
    # cv2.imshow('erased.png',erased)
    cv2.imwrite('erased.png',erased)
    #cv2.waitKey(0)  
    #cv2.destroyAllWindows()

    # Detection de contour

    #edit the filename based on the inpput name of the file
    filename = 'erased.png'
    t = 10
    image1 = contour(filename)
    cv2.imwrite("out2.png",image1)

    imageb = cv2.imread('out2.png')
    imageb = cv2.cvtColor(imageb, cv2.COLOR_BGR2GRAY)

    #identify rect

    medianb = cv2.medianBlur(imageb, 15)

    cv2.imwrite("afterblur.jpg", medianb)


    image_gray = cv2.imread("afterblur.jpg", 0)
    image_gray = np.where(image_gray > 30, 255, image_gray)
    image_gray = np.where(image_gray <= 30, 0, image_gray)

    contours,hierachy = cv2.findContours(image_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rect_cnts = []
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)

        if len(approx) == 4: # shape filtering condition
            rect_cnts.append(cnt)

    max_area = 0
    football_square = None
    for cnt in rect_cnts:
        (x, y, w, h) = cv2.boundingRect(cnt)
        if max_area < w*h:
            max_area = w*h
            football_square = cnt

    # Draw the result
    image = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)
   # cv2.drawContours(image, [football_square], -1, (0, 0,255), 5)

    #cv2.imshow('Result Preview.png', image)
    #cv2.imwrite('result.png', image)
    path_result = 'data_result/log1/'+str(cadena)+'-rgb_result.png'
    cv2.imwrite(path_result,image)
    img_number += 1
    cv2.waitKey()
    cv2.destroyAllWindows()
