import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("dog.jpeg")

# Convert the image from BGR to RGB for displaying with matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# apply gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5,5),0)

# Apply Gaussian Blur to reduce noise
# The threshold value (150) may need adjectment based on your image's lighting and contrast
