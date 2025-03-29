import sys
import cv2 as cv
import numpy as np
# Load image in grayscale
imagePath = "image.jpeg"
src_gray = cv.imread(imagePath, cv.IMREAD_GRAYSCALE)

# Check if image loaded successfully
if src_gray is None:
    sys.exit("Error: Could not read the image. Check the file path.")

# Laplacian edge detection
ddepth = cv.CV_16S  # Output depth (16-bit signed to handle negative edges)
# kernel_size = 3  # Size of the Laplacian kernel (typically 1, 3, or 5)
kernel_size = np.array([
    [0,  1, 0],
    [1, -4, 1],
    [0,  1, 0]
], dtype=np.float32)

def apply_lapacian(image, kernel):
    h,w = image.shape
    k_h, k_w = kernel.shape
    pad = k_h//2

    # pad the image (to handle borders)
    padded = np.pad(image, ((pad, pad), (pad, pad)), mode='constant', constant_values=0)
    output = np.zeros_like(image, dtype=np.float32)

    # apply convolution
    for y in range(pad, h+pad):
        for x in range(pad, w+pad):
            roi = padded[y-pad:y+pad+1, x-pad:x+pad+1]
            output[y-pad, x-pad] = np.sum(roi * kernel)

    return output


# dst = cv.Laplacian(src_gray, ddepth, kernel_size)
dst = apply_lapacian(src_gray , kernel_size)

# Convert back to 8-bit for display
abs_dst = cv.convertScaleAbs(dst)

# Display results
cv.imshow("Original", src_gray)
cv.imshow("Laplacian Edges", abs_dst)
cv.waitKey(0)
cv.destroyAllWindows()