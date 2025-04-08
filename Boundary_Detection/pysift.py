from numpy import all, array, arctan2, cos, sin, exp, dot,log, logical_and, roll, sqrt, trace, unravel_index, pi, deg2rad, where, zeros, floor, full, nan, isnan, round, float32
from numpy.linalg import det, lstsq, norm
from cv2 import resize, GaussianBlur, subtract, KeyPoint, INTER_LINEAR, INTER_NEAREST
from functools import cmp_to_key
import logging

###########################
# Global Variables
###########################
logger = logging.getLogger(__name__)
float_tolerance = 1e-7

def computeKeypointsAndDescriptors(image, sigma=1.6, num_intervals=3, assume_blur=0.5, image_border_width=5):
    image = image.astype(float32)
    base_image = generateBaseImage(image, sigma, assume_blur)
    num_octaves = computeNumberOfOctaves(base_image.shape)
    gaussian_kernels = generateGaussianKernels(sigma, num_intervals)
    gaussian_images = generateGaussianImages(base_image, num_octaves, gaussian_kernels)
    dog_images = generateDoGImages(gaussian_images)
    keypoints = findScaleSpaceExtrema(gaussian_images, dog_images, num_intervals, sigma, image_border_width)
    # keypoints = removeDuplicateKeypoints(keypoints)
    # keypoints = convertKeypointsToInputImageSize(keypoints)
    # descriptors = generateDescriptors(keypoints, gaussian_images)
    # return keypoints , descriptors
    return gaussian_images


def generateBaseImage(image, sigma, assume_blur):
    logger.debug("Generating base image")
    image = resize(image, (0,0), fx=2, fy=2, interpolation=INTER_LINEAR)
    sigma_diff = sqrt(max((sigma**2) - ((2 * assume_blur)**2), 0.01))
    return GaussianBlur(image, (0,0), sigmaX=sigma_diff, sigmaY=sigma_diff)

def computeNumberOfOctaves(image_shape):
    logger.debug("Computing number of octaves")
    return int(round(log(min(image_shape))/log(2) - 1))

def generateGaussianKernels(sigma, num_intervals):
    logger.debug("Generating Gaussian kernels")
    num_images_per_octave = num_intervals + 3
    k = 2**(1/num_intervals)
    gaussian_kernels = zeros(num_images_per_octave)
    gaussian_kernels[0] = sigma

    for image_idex in range(1, num_images_per_octave):
        sigma_previos = (k**(image_idex - 1)) * sigma
        sigma_total = k * sigma_previos
        gaussian_kernels[image_idex] = sqrt(sigma_total**2 - sigma_previos**2)
    print(gaussian_kernels, "gaussian_kernels")
    return gaussian_kernels


def generateGaussianImages(base_image, num_octaves, gaussian_kernels):
    """ Generate scale-space pyramid of Gaussian Images"""
    logger.debug("Generating Gaussian Images")
    gaussian_images = []
    for octave_index in range(num_octaves):
        gaussian_images_in_octaves = []
        gaussian_images_in_octaves.append(base_image)
        for gaussian_kernel in gaussian_kernels[1:]:
            image = GaussianBlur(base_image, (0,0), sigmaX=gaussian_kernel, sigmaY=gaussian_kernel)
            gaussian_images_in_octaves.append(image)
        gaussian_images.append(gaussian_images_in_octaves)
        octave_base = gaussian_images_in_octaves[-3]
        image = resize(octave_base, (int(octave_base.shape[1]/2), int(octave_base.shape[0]/2)), interpolation=INTER_NEAREST)
    return array(gaussian_images)

def generateDoGImages(gaussian_images):
    """Generate Difference-of-Gaussian image pyramid"""
    logger.debug("Generating difference-of-Gaussian images...")
    dog_images = []

    for gaussian_images_in_octave in gaussian_images:
        dog_images_in_octave = []
        for first_image, second_image in zip(gaussian_images_in_octave, gaussian_images_in_octave[1:]):
            dog_images_in_octave.append(subtract(second_image, first_image)) # # ordinary subtraction will not work because the images are unsigned integers
        dog_images.append(dog_images_in_octave)
    return array(dog_images)


    

def findScaleSpaceExtrema(gaussian_images, dog_images, num_intervals, sigma, image_border_width, contrast_threshold=0.04):
    """
    Find pixel positions of all scale-space extrema in the image pyramid
    """
    logger.debug('Finding scale-space extrema')
    threshold = floor(0.5 * contr)
    

def removeDuplicateKeypoints(keypoints):
    print("Removing duplicate keypoints")


def convertKeypointsToInputImageSize(keypoints):
    print("Converting keypoints to input iamge size")
    

def generateDescriptors(keypoints, gaussian_images):
    print("Generating descriptors")


