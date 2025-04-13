import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load images
img1 = cv2.imread('2.png', 0)  # Single book cover
img2 = cv2.imread('1.png', 0)  # Multiple books image

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Find keypoints and descriptors
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
print(f"Keypoints in image 1: {len(kp1)}")
print(f"Keypoints in image 2: {len(kp2)}")

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)
MIN_MATCH_COUNT = 10
# Ratio test
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)

print(f"Good matches found: {len(good)}")

print(f"Good matches found: {len(good)}")

if len(good) > MIN_MATCH_COUNT:
    # Estimate homography between template and scene
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    # Draw detected template in scene image
    h, w = img1.shape
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)
    
    img2_with_box = cv2.polylines(img2.copy(), [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    
    # Create side-by-side image
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    nWidth = w1 + w2
    nHeight = max(h1, h2)
    hdif = int((h2 - h1) / 2)
    newimg = np.zeros((nHeight, nWidth, 3), np.uint8)
    
    for i in range(3):
        newimg[hdif:hdif + h1, :w1, i] = img1
        newimg[:h2, w1:w1 + w2, i] = img2_with_box
    
    # Draw SIFT keypoint matches
    for m in good:
        pt1 = (int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1] + hdif))
        pt2 = (int(kp2[m.trainIdx].pt[0] + w1), int(kp2[m.trainIdx].pt[1]))
        cv2.line(newimg, pt1, pt2, (255, 0, 0))
    
    plt.figure(figsize=(15, 8))
    plt.imshow(newimg)
    plt.title(f"SIFT Matching - {len(good)} Good Matches")
    plt.axis('off')
    plt.show()
    
    # Also show the homography result separately
    plt.figure(figsize=(8, 8))
    plt.imshow(img2_with_box, cmap='gray')
    plt.title("Detected Book in Scene")
    plt.axis('off')
    plt.show()
else:
    print(good)