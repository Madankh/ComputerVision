import cv2 as cv
import numpy as np
import sys

imagePath = "image.jpeg"
src_img = cv.imread(imagePath, cv.IMREAD_GRAYSCALE)

edges = cv.Canny(src_img, 100, 200)
cv.imshow("canny", edges)
cv.waitKey(0)
cv.destroyAllWindows()