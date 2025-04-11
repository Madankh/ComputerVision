import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os


def template_matching(img, shortimage):
    # Load the image and template
    root = os.getcwd()
    imgPath = os.path.join(root, img)
    img = cv.imread(imgPath)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    sumanPath = os.path.join(root, shortimage)
    sumanimg = cv.imread(sumanPath)
    sumanimg = cv.cvtColor(sumanimg, cv.COLOR_BGR2RGB)
    height, width, _ = sumanimg.shape

    # plt.figure()
    # plt.subplot(121)
    # plt.imshow(img)
    # plt.subplot(122)
    # plt.imshow(sumanimg)
    # plt.show()

    methods = [
               cv.TM_CCOEFF, 
               cv.TM_CCOEFF_NORMED, 
               cv.TM_CCORR, 
               cv.TM_CCORR_NORMED,
               cv.TM_SQDIFF,
               cv.TM_SQDIFF_NORMED]
    
    title =['cv.TM_CCOEFF', 
            'cv.TM_CCOEFF_NORMED', 
            'cv.TM_CCORR', 
            'cv.TM_CCORR_NORMED',
            'cv.TM_SQDIFF',
            'cv.TM_SQDIFF_NORMED']
    
    for i in range(len(methods)):
        method = methods[i]
        img2 = img.copy()
        templatemap = cv.matchTemplate(img2, sumanimg, method)
        _,_,minLoc,maxLoc = cv.minMaxLoc(templatemap)

        if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
            topLeft = minLoc
        else:
            topLeft = maxLoc

        bottomRight = (topLeft[0] + width, topLeft[1] + height)

        cv.rectangle(img2, topLeft, bottomRight, 255, 2)
        plt.figure()

        plt.subplot(121)
        plt.imshow(templatemap)
        plt.title(title[i])
        plt.subplot(122)
        plt.imshow(img2)
        plt.show()
if __name__ == "__main__":
    template_matching('111.jpg', '111short.jpg')

