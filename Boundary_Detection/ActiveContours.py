import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("dog.jpeg")

# Convert the image from BGR to RGB for displaying with matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# apply gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5,5),0)

# Apply Gaussian Blur to reduce noise
# The threshold value (150) may need adjectment based on your image's lighting and contrast
ret, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

largest_contour = max(contours, key=cv2.contourArea)

# Create copies of the original and threadholded images for drawing contours
image_with_contour = image_rgb.copy()
thresh_with_contour = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

cv2.drawContours(image_with_contour, [largest_contour], -1, (14,21,239),5)
cv2.drawContours(thresh_with_contour, [largest_contour], -1, (14,21,239),5)

plt.figure(figsize=(12,8))

# Original Image
plt.subplot(2, 2, 1)
plt.imshow(image_rgb)
plt.title('Original Image')
plt.axis('off')

# Thresholded Image
plt.subplot(2, 2, 2)
plt.imshow(thresh, cmap='gray')
plt.title('Thresholded Image')
plt.axis('off')

# Contour on Original Image
plt.subplot(2, 2, 3)
plt.imshow(image_with_contour)
plt.title('Original Image with Contour')
plt.axis('off')

# Contour on Thresholded Image
plt.subplot(2, 2, 4)
plt.imshow(thresh_with_contour)
plt.title('Thresholded Image with Contour')
plt.axis('off')

plt.tight_layout()
plt.show()