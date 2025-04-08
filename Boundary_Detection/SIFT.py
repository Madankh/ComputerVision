import cv2
import pysift
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread("image.jpeg", 0)
# keypoints, descriptors = pysift.computeKeypointsAndDescriptors(image)
# print("Keypoints: ", keypoints)

def displayDoGImages(dog_images):
    """Display the Difference of Gaussian images for visualization"""
    for octave_index, dog_images_in_octave in enumerate(dog_images):
        for image_index, dog_image in enumerate(dog_images_in_octave):
            plt.figure(figsize=(8, 8))
            plt.title(f'DoG Octave {octave_index+1}, Image {image_index+1}')
            plt.imshow(dog_image, cmap='gray')
            plt.show()


test_image = pysift.computeKeypointsAndDescriptors(image)
displayDoGImages(test_image)

