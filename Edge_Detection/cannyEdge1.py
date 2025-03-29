import cv2 as cv
import numpy as np
import sys

def gaussian_kernel1(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2))) * normal
    return g / np.sum(g)

def gaussian_kernel(img, kernel_size=5, sigma=1):
    kernel = gaussian_kernel1(kernel_size, sigma)
    return cv.filter2D(img, -1, kernel)

def non_max_suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.float32)  # Changed to float32
    angle = D * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, M-1):
        for j in range(1, N-1):
            try:
                q, r = 255, 255

                #angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = img[i, j + 1]
                    r = img[i, j - 1]
                #angle 45
                elif (22.5 <= angle[i, j] < 67.5):
                    q = img[i + 1, j - 1]
                    r = img[i - 1, j + 1]
                #angle 90
                elif (67.5 <= angle[i, j] < 112.5):
                    q = img[i + 1, j]
                    r = img[i - 1, j]
                #angle 135
                elif (112.5 <= angle[i, j] < 157.5):
                    q = img[i - 1, j - 1]
                    r = img[i + 1, j + 1]

                if (img[i, j] >= q) and (img[i, j] >= r):
                    Z[i, j] = img[i, j]
                else:
                    Z[i, j] = 0

            except IndexError:
                pass

    return Z

def sobel_gradients(img):
    S_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float32)
    S_y = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]], dtype=np.float32)

    G_x = cv.filter2D(img, -1, S_x)
    G_y = cv.filter2D(img, -1, S_y)

    G = np.hypot(G_x, G_y)
    G = G / G.max() * 255  # Normalize
    theta = np.arctan2(G_y, G_x)
    return G, theta

def threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.15):  # Adjusted ratios
    highThreshold = img.max() * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio

    M, N = img.shape
    res = np.zeros((M, N), dtype=np.uint8)  # Changed to uint8 for image output

    strong = np.uint8(255)
    weak = np.uint8(75)  # Changed value for better visualization

    strong_i, strong_j = np.where(img >= highThreshold)
    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return res, weak, strong

def hysteresis(img, weak, strong=255):
    M, N = img.shape
    for i in range(1, M-1):
        for j in range(1, N-1):
            if img[i, j] == weak:
                if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or 
                    (img[i+1, j+1] == strong) or (img[i, j-1] == strong) or 
                    (img[i, j+1] == strong) or (img[i-1, j-1] == strong) or 
                    (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                    img[i, j] = strong
                else:
                    img[i, j] = 0
    return img

def canny_edge_detector(img):
    img_blurred = gaussian_kernel(img)
    G, theta = sobel_gradients(img_blurred)
    non_max = non_max_suppression(G, theta)
    edges, weak, strong = threshold(non_max)
    edges = hysteresis(edges, weak, strong)
    return edges

# Load image in grayscale
imagePath = "image.jpeg"
src_img = cv.imread(imagePath, cv.IMREAD_GRAYSCALE)

# Check if image loaded properly
if src_img is None:
    print("Error: Could not load image")
    sys.exit()

# Run Canny edge detection
edges = canny_edge_detector(src_img)

# Save the result
cv.imwrite('edges.jpg', edges)
cv.waitKey(0)
cv.destroyAllWindows()
