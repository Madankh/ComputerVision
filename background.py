import cv2
import mediapipe as mp
import numpy as np
import copy

def stackImages(_imgList, cols, scale):
    """
    Stack Images together to display in a single window
    :param _imgList: list of images to stack
    :param cols: the num of img in a row
    :param scale: bigger~1+ ans smaller~1-
    :return: Stacked Image
    """
    imgList = copy.deepcopy(_imgList)

    # Get dimensions of the first image
    width1, height1 = imgList[0].shape[1], imgList[0].shape[0]

    # make the array full by adding blank img, otherwise the openCV can't work
    totalImages = len(imgList)
    rows = totalImages // cols if totalImages // cols * cols == totalImages else totalImages // cols + 1
    blankImages = cols * rows - totalImages

    # Create a blank image with dimensions of the first image
    imgBlank = np.zeros((height1, width1, 3), np.uint8)
    imgList.extend([imgBlank] * blankImages)

    # resize the images to be the same as the first image and apply scaling
    for i in range(cols * rows):
        imgList[i] = cv2.resize(imgList[i], (width1, height1), interpolation=cv2.INTER_AREA)
        imgList[i] = cv2.resize(imgList[i], (0, 0), None, scale, scale)

        if len(imgList[i].shape) == 2:  # Convert grayscale to color if necessary
            imgList[i] = cv2.cvtColor(imgList[i], cv2.COLOR_GRAY2BGR)

    # put the images in a board
    hor = [imgBlank] * rows
    for y in range(rows):
        line = []
        for x in range(cols):
            line.append(imgList[y * cols + x])
        hor[y] = np.hstack(line)
    ver = np.vstack(hor)
    return ver


class SelfiSegmentation():

    def __init__(self, model=1):
        """
        :param model: model type 0 or 1. 0 is general 1 is landscape(faster)
        """
        self.model = model
        self.mpDraw = mp.solutions.drawing_utils
        self.mpSelfieSegmentation = mp.solutions.selfie_segmentation
        self.selfieSegmentation = self.mpSelfieSegmentation.SelfieSegmentation(model_selection=self.model)

    def removeBG(self, img, imgBg=(255, 255, 255), cutThreshold=0.1):
        """

        :param img: image to remove background from
        :param imgBg: Background Image. can be a color (255,0,255) or an image . must be same size
        :param cutThreshold: higher = more cut, lower = less cut
        :return:
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.selfieSegmentation.process(imgRGB)
        condition = np.stack(
            (results.segmentation_mask,) * 3, axis=-1) > cutThreshold
        if isinstance(imgBg, tuple):
            _imgBg = np.zeros(img.shape, dtype=np.uint8)
            _imgBg[:] = imgBg
            imgOut = np.where(condition, img, _imgBg)
        else:
            imgOut = np.where(condition, img, imgBg)
        return imgOut


def main():
    # Initialize the webcam. '2' indicates the third camera connected to the computer.
    # '0' usually refers to the built-in camera.
    cap = cv2.VideoCapture(0)

    # Set the frame width to 640 pixels
    cap.set(3, 640)
    # Set the frame height to 480 pixels
    cap.set(4, 480)

    # Initialize the SelfiSegmentation class. It will be used for background removal.
    # model is 0 or 1 - 0 is general 1 is landscape(faster)
    segmentor = SelfiSegmentation(model=0)

    # Infinite loop to keep capturing frames from the webcam
    while True:
        # Capture a single frame
        success, img = cap.read()

        # Use the SelfiSegmentation class to remove the background
        # Replace it with a magenta background (255, 0, 255)
        # imgBG can be a color or an image as well. must be same size as the original if image
        # 'cutThreshold' is the sensitivity of the segmentation.
        imgOut = segmentor.removeBG(img, imgBg=(255, 0, 255), cutThreshold=0.1)

        # Stack the original image and the image with background removed side by side
        imgStacked = stackImages([img, imgOut], cols=2, scale=1)

        # Display the stacked images
        cv2.imshow("Image", imgStacked)

        # Check for 'q' key press to break the loop and close the window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

