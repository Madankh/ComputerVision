import numpy as np
import cv2 as cv
import json
import matplotlib.pyplot as plt

# # load the image
img = cv.imread('./1.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# # initialize SIFT detector
# sift = cv.SIFT_create()

# # Detect keypoints
# kp = sift.detect(gray, None)

# # Draw keypoints on the image
# img_with_kp = cv.drawKeypoints(gray, kp, img)
# img_with_kp_rgb = cv.cvtColor(img_with_kp, cv.COLOR_BGR2RGB)

# # display the image with keypoints 
# plt.figure(figsize=(10,10))
# plt.imshow(img_with_kp_rgb)
# plt.axis('off')
# plt.show()


# Dectecting keypoints in ROI
# mask = np.zeros(gray.shape, dtype=np.uint8)

# # Define ROI coordinates
# x_start, y_start = 160, 160
# x_end, y_end = 450, 368

# mask[y_start:y_end, x_start:x_end] = 255
# sift = cv.SIFT_create()
# # draw keypoints
# kp = sift.detect(gray, mask)

# img_with_kp = cv.drawKeypoints(img, kp, None)

# # Hightlist the roi with a rectangle 
# cv.rectangle(img_with_kp, (x_start, y_start), (x_end, y_end), (0, 0, 255), 2) 
# # Convert the image from BGR to RGB for displaying with matplotlib
# img_with_kp_rgb = cv.cvtColor(img_with_kp, cv.COLOR_BGR2RGB)

# # Display the image with keypoints and highlighted ROI
# plt.figure(figsize=(10, 10))
# plt.imshow(img_with_kp_rgb)
# plt.axis('off')
# plt.show()



# 1. Load object and scene image
# 2. Detect SIFT keypoints and compute descriptors
# 3. Configure FLANN with KD-TREE
# 4. Match descriptors with knnMatch (k=2)
# 5. Apply Lowe’s ratio test
# 6. Count good matches
# 7. Decide if object found (based on min_keypoints)

img1 = cv.imread('2.png', 0)  # Single book cover
img2 = cv.imread('1.png', 0)  # Multiple books image

def find_object(img, scene_image, min_keypoints=100):
    # read the object 
    img1 = cv.imread('2.png', 0)  # Single book cover
    img2 = cv.imread('1.png', 0)  # Multiple books image

    sift = cv.SIFT_create()

    # find keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    # FLANN-based matcher
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    
    # prepare result
    num_keypoints = len(good_matches)

    result = {
        "status": "PASS" if num_keypoints >= min_keypoints else "FAIL",
        "keypoints_detected": num_keypoints
    }

    img_matches = cv.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.figure(figsize=(15, 5))
    plt.imshow(cv.cvtColor(img_matches, cv.COLOR_BGR2RGB))
    plt.title(f"Result: {result['status']} ({result['keypoints_detected']} keypoints)")
    plt.axis('off')
    plt.tight_layout()

    return result, 'output_visualization.png'

result, output_image = find_object(img1, img2, 10)
print(json.dumps(result, indent=2))
