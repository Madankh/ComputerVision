import cv2
import numpy as np
import glob
import os

# image_paths = glob.glob(os.path.join("images", "*.jpg"))

folder_path = os.path.join(os.getcwd(), "Image_Stitching/images")
print("Looking in folder:", folder_path)

image_paths = glob.glob(os.path.join(folder_path, "*.jpg"))
print("Image paths found:", image_paths)

images = []

for image_path in image_paths:
    img = cv2.imread(image_path)
    if img is None:
        print("Failed to read:", image_path)
        continue
    images.append(img)

print("Number of valid images:", len(images))

if len(images) < 2:
    print("Need at least two valid images to stitch.")
else:
    stitcher = cv2.Stitcher_create()
    error, stitched_img = stitcher.stitch(images)

    if error == cv2.Stitcher_OK:
        cv2.imwrite("stitchedOutput.png", stitched_img)
        cv2.imshow("Stitched Image", stitched_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Stitching failed with error code:", error)
