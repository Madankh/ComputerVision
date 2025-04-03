import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.morphology import area_closing, area_opening
from skimage.measure import regionprops

# Load and Binarize the source image
image_path = "image.jpeg"
image = rgb2gray(imread(image_path))
threshold = 0.85
binary_image = image < threshold

# Use Matplotlib to visualize
# plt.imshow(binary_image, cmap="gray")
# plt.axis("off")  # Hide axis for better visualization
# plt.show()
plt.imshow(binary_image, cmap="gray")
plt.axis("off")
plt.savefig("binary_image.png")