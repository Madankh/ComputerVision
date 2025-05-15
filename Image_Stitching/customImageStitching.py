import os
import math
import numpy as np
import cv2

def ReadImage(ImageFolderPath):
    Images = []

    if os.path.isdir(ImageFolderPath):
        ImageNames = os.listdir(ImageFolderPath)
        ImageNames_Split = [[int(os.path.splitext(os.path.basename(ImageNames))[0]), ImageNames] for ImageName in ImageNames]
        ImageNames_Split = sorted(ImageNames_Split, key=lambda x:x[0])
        ImageNames_Sorted = [ImageNames_Split[i][1] for i in range(len(ImageNames_Split))]

        for i in range(len(ImageNames_Sorted)):
            ImageName = ImageNames_Sorted[i]
            InputImage = cv2.imread(ImageFolderPath + "/" + ImageName)

            if InputImage is None:
                print("Not able to read image : {}".format(ImageName))
                exit(0)
            Images.append(InputImage)
    else:
        print("\nEnter valid Image folder path. \n")
    
    if len(Images) < 2:
        print("\n Not enough images found, please provide 2 or more images \n")
    return Images

def findMatches(BaseImage, SecImage):
    Sift = cv2.SIFT_create()

