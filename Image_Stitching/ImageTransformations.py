import numpy as np
import cv2 as cv

img = cv.imread("1.png")
img = cv.imread("1.png", cv.IMREAD_GRAYSCALE)
assert img is not None, "File could not be read, check with os.path.exists()"

# res = cv.resize(img,None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)
# cv.imshow('Resized Image', res)
# cv.waitKey(0)  # Wait for a key press
# cv.destroyAllWindows()  # Close the window when a key is pressed

# # another approch
# height, width = img.shape[:2]
# res = cv.resize(img, (2*width, 2*height), interpolation=cv.INTER_CUBIC)
# cv.imshow('Resized Image', res)
# cv.waitKey(0)  # Wait for a key press
# cv.destroyAllWindows()  # Close the window when a key is pressed

rows,cols = img.shape
M = np.float32([[1,0,100], [0,1,50]])
dst = cv.warpAffine(img,M,(cols, rows))

cv.imshow('img',dst)
cv.waitKey(0)
cv.destroyAllWindows()
