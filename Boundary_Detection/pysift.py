from numpy import all, any, array, arctan2, cos, sin, exp, dot, log, logical_and, roll, sqrt, stack, trace, unravel_index, pi, deg2rad, rad2deg, where, zeros, floor, full, nan, isnan, round, float32
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
    threshold = floor(0.5 * contrast_threshold / num_intervals * 255)
    keypoints = []

    for octave_index, dog_images_in_octave in enumerate(dog_images):
        for image_index, (first_image, second_image, third_image) in enumerate(zip(dog_images_in_octave, dog_images_in_octave[1:], dog_images_in_octave[2:])):
            # (i, j) is the center of the 3x3 array
            for i in range(image_border_width, first_image.shape[0] - image_border_width):
                for j in range(image_border_width, first_image.shape[1] - image_border_width):
                    if isPixelAnExtremum(first_image[i-1:i+2, j-1:j+2], second_image[i-1:i+2, j-1:j+2], third_image[i-1:i+2, j-1:j+2], threshold):
                        localization_result = localizeExtremumViaQuadraticFit(i,j,image_index+1, octave_index, num_intervals, dog_images_in_octave, sigma, contrast_threshold, image_border_width)
                        if localization_result is not None:
                            keypoint, localized_image_index = localization_result
                            keypoints_with_orientations = computeKeypointsWithOrientations(keypoint, octave_index, gaussian_images[octave_index][localized_image_index])
                            for keypoint_with_orientation in keypoints_with_orientations:
                                keypoints.append(keypoint_with_orientation)
    return keypoints
    
def isPixelAnExtremum(first_subimage, second_subimage, third_subimage, threshold):
    """Return True if the center element of the 3x3x3 input array is strictly greater than or less than all its neighbors, false otherwise"""
    center_pixel_value = second_subimage[1,1]
    if abs(center_pixel_value) > threshold:
        if center_pixel_value > 0:
            return all(center_pixel_value >= first_subimage) and \
            all(center_pixel_value >= third_subimage) and \
            all(center_pixel_value >= second_subimage[0,:]) and \
            all(center_pixel_value >= second_subimage[2,:]) and \
            center_pixel_value >= second_subimage[1,0] and \
            center_pixel_value >= second_subimage[1,2]
        elif center_pixel_value < 0:
            return all(center_pixel_value <= first_subimage) and \
            all(center_pixel_value <= third_subimage) and \
            all(center_pixel_value <= second_subimage[0,:]) and \
            all(center_pixel_value <= second_subimage[2,:]) and \
            center_pixel_value <= second_subimage[1,0] and \
            center_pixel_value <= second_subimage[1,2]

    return False


def computeGradientAtCenterPixel(pixel_array):
    # dx = 0.5 * (right_value - left_value)
    dx = 0.5 * (pixel_array[1 , 1 , 2] - pixel_array[1 , 1 , 0])
    # dy = 0.5 * (below_value - above_value)
    dy = 0.5 * (pixel_array[1 , 2 , 1] - pixel_array[1 , 0 , 1])
    # ds = 0.5 * (more_blurred_value - less_blurred_value)
    ds = 0.5 * (pixel_array[2 , 1 , 1] - pixel_array[0 , 1 , 1])
    return array([dx,dy,ds])

def computeHessianAtCenterPixel(pixel_array):
    center_pixel_value = pixel_array[1,1,1]
    dxx = pixel_array[1,1,2] - 2 * center_pixel_value + pixel_array[1,1,0]
    dyy = pixel_array[1,2,1] - 2 * center_pixel_value + pixel_array[1,0,1]
    dss = pixel_array[2,1,1] - 2 * center_pixel_value + pixel_array[0,1,1]
    dxy = 0.25 * (pixel_array[1, 2, 2] - pixel_array[1, 2, 0] - pixel_array[1, 0, 2] + pixel_array[1, 0, 0])
    dxs = 0.25 * (pixel_array[2, 1, 2] - pixel_array[2, 1, 0] - pixel_array[0, 1, 2] + pixel_array[0, 1, 0])
    dys = 0.25 * (pixel_array[2, 2, 1] - pixel_array[2, 0, 1] - pixel_array[0, 2, 1] + pixel_array[0, 0, 1])
    return array([[dxx, dxy, dxs], 
                  [dxy, dyy, dys],
                  [dxs, dys, dss]])



# def localizeExtremumViaQuadraticFit(i,j,image_index, octave_index, num_intervals, dog_images_in_octave, sigma, contrast_threshold, image_border_width, eigenvalue_ratio=10, num_attempts_until_convergence=5):
#     """Iteratively refine pixel positions of scale-space extrema via quadratic fit around each extremum's neighbors
#     """
#     logger.debug('Localizing scale-space extrema...')
#     extremum_is_outside_image = False
#     image_shape = dog_images_in_octave[0].shape
#     for attempt_index in (num_attempts_until_convergence):
#         # Need to convert from uint8 to float32 to compute derivatives and need to rescale pixel
#         first_image, second_image, third_image = dog_images_in_octave[image_index-1:image_index+2]
#         pixel_cube = stack([first_image[i-1:i+2, j-1:j+2],
#                             second_image[i-1:i+2, j-1:j+2],
#                             third_image[i-1:i+2, j-1:j+2]]).astype('float32')/255.
#         gradient = computeGradientAtCenterPixel(pixel_cube)
#         hessian =  computeHessianAtCenterPixel(pixel_cube)
#         extremum_update = -lstsq(hessian, gradient, rcond=None)[0]
#         if abs(extremum_update[0]) < 0.5 and abs(extremum_update[1]) < 0.5 and abs(extremum_update[2])<0.5:
#             break
#         j+=int(round(extremum_update[0]))
#         i+=int(round(extremum_update[1]))
#         image_index += int(round(extremum_update[2]))

#         # Make sure the new pixel_cube will lie entirely within the image
#         if i < image_border_width or i>=image_shape[0] - image_border_width or j < image_border_width or j >= image_shape[1] - image_border_width or image_index < 1 or image_index > num_intervals:
#             extremum_is_outside_image = True
#             break

#     if extremum_is_outside_image:
#         return None
#     if attempt_index >= num_attempts_until_convergence - 1:
#         return None
    
#     functionValueAtUpdatedExtremum = pixel_cube[1,1,1] + 0.5 * dot(gradient, extremum_update)
#     if abs(functionValueAtUpdatedExtremum) * num_intervals >=  contrast_threshold:
#         xy_hessian = hessian[:2,:2]
#         xy_hessian_trace = trace(xy_hessian)
#         xy_hessian_det = det(xy_hessian)

#         if xy_hessian_det > 0 and eigenvalue_ratio * (xy_hessian_trace ** 2) < ((eigenvalue_ratio + 1)**2) * xy_hessian_det:
#             # Contrast check passed 
#             keypoint = KeyPoint()
#             keypoint.pt = ((j + extremum_update[0]) * (2 ** octave_index), (i + extremum_update[1]) * (2 ** octave_index))
#             keypoint.octave = octave_index + image_index * (2 ** 8) + int(round((extremum_update[2] + 0.5) * 255)) * (2 ** 16)
#             keypoint.size = sigma * (2 ** ((image_index + extremum_update[2])/float32(num_intervals))) * (2 ** (octave_index + 1))
#             keypoint.response = abs(functionValueAtUpdatedExtremum)
#             return keypoint, image_index
#     return None

def localizeExtremumViaQuadraticFit(i, j, image_index, octave_index, num_intervals, dog_images_in_octave, sigma, contrast_threshold, image_border_width, eigenvalue_ratio=10, num_attempts_until_convergence=5):
    """Iteratively refine pixel positions of scale-space extrema via quadratic fit around each extremum's neighbors
    """
    logger.debug('Localizing scale-space extrema...')
    extremum_is_outside_image = False
    image_shape = dog_images_in_octave[0].shape
    for attempt_index in range(num_attempts_until_convergence):
        # need to convert from uint8 to float32 to compute derivatives and need to rescale pixel values to [0, 1] to apply Lowe's thresholds
        first_image, second_image, third_image = dog_images_in_octave[image_index-1:image_index+2]
        pixel_cube = stack([first_image[i-1:i+2, j-1:j+2],
                            second_image[i-1:i+2, j-1:j+2],
                            third_image[i-1:i+2, j-1:j+2]]).astype('float32') / 255.
        gradient = computeGradientAtCenterPixel(pixel_cube)
        hessian = computeHessianAtCenterPixel(pixel_cube)
        extremum_update = -lstsq(hessian, gradient, rcond=None)[0]
        if abs(extremum_update[0]) < 0.5 and abs(extremum_update[1]) < 0.5 and abs(extremum_update[2]) < 0.5:
            break
        j += int(round(extremum_update[0]))
        i += int(round(extremum_update[1]))
        image_index += int(round(extremum_update[2]))
        # make sure the new pixel_cube will lie entirely within the image
        if i < image_border_width or i >= image_shape[0] - image_border_width or j < image_border_width or j >= image_shape[1] - image_border_width or image_index < 1 or image_index > num_intervals:
            extremum_is_outside_image = True
            break
    if extremum_is_outside_image:
        logger.debug('Updated extremum moved outside of image before reaching convergence. Skipping...')
        return None
    if attempt_index >= num_attempts_until_convergence - 1:
        logger.debug('Exceeded maximum number of attempts without reaching convergence for this extremum. Skipping...')
        return None
    functionValueAtUpdatedExtremum = pixel_cube[1, 1, 1] + 0.5 * dot(gradient, extremum_update)
    if abs(functionValueAtUpdatedExtremum) * num_intervals >= contrast_threshold:
        xy_hessian = hessian[:2, :2]
        xy_hessian_trace = trace(xy_hessian)
        xy_hessian_det = det(xy_hessian)
        if xy_hessian_det > 0 and eigenvalue_ratio * (xy_hessian_trace ** 2) < ((eigenvalue_ratio + 1) ** 2) * xy_hessian_det:
            # Contrast check passed -- construct and return OpenCV KeyPoint object
            keypoint = KeyPoint()
            keypoint.pt = ((j + extremum_update[0]) * (2 ** octave_index), (i + extremum_update[1]) * (2 ** octave_index))
            keypoint.octave = octave_index + image_index * (2 ** 8) + int(round((extremum_update[2] + 0.5) * 255)) * (2 ** 16)
            keypoint.size = sigma * (2 ** ((image_index + extremum_update[2]) / float32(num_intervals))) * (2 ** (octave_index + 1))  # octave_index + 1 because the input image was doubled
            keypoint.response = abs(functionValueAtUpdatedExtremum)
            return keypoint, image_index
    return None


def computeKeypointsWithOrientations():
    pass

def removeDuplicateKeypoints(keypoints):
    print("Removing duplicate keypoints")


def convertKeypointsToInputImageSize(keypoints):
    print("Converting keypoints to input iamge size")
    

def generateDescriptors(keypoints, gaussian_images):
    print("Generating descriptors")


