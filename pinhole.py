import cv2
import numpy as np

# Create a blank image with a black background
width, height = 640, 480
image = np.zeros((height, width, 3), np.uint8)

# Define the camera matrix
focal_length = 1
center = np.array([width/2, height/2])
camera_matrix = np.array([[focal_length, 0, center[0]],
                        [0, focal_length, center[1]],
                        [0, 0, 1]], dtype = "double")

# Create a 3D point in the world space
world_points = np.array([[0, 0, 0]], dtype='double')

# Project the 3D point onto the image plane
projected_points, _ = cv2.projectPoints(world_points, np.zeros((3,1)), np.zeros((3,1)), camera_matrix, None)

# Draw the projected point on the image
cv2.circle(image, tuple(np.squeeze(projected_points[0]).astype(int)), 5, (0, 255, 0), -1)

cv2.imshow("Pinhole Camera", image)
cv2.waitKey(0)
cv2.destroyAllWindows()