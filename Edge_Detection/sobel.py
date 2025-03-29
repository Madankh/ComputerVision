import numpy as np
import cv2
import matplotlib.pyplot as plt

def display(img):
    """
    Display the image
    :param img: image to display
    :return: None
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    ax.axis('off')


# function for creating image

def createImage():
    blank_img = np.zeros((200, 200))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(blank_img, text='H', org=(50,150), fontFace=font, fontScale=5, color=(255,255,255), thickness=25, lineType=cv2.LINE_AA)
    return blank_img

image_path = "image.jpeg"
i = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    

# sobel edge detection
sobel_x = np.array([[-1,-2,-1],
                    [0,0,0],
                   [1,2,1]])

sobel_y = np.array([[-1,0,1],
                    [-2,0,2],
                    [-1,0,1]])


# input image
# i = createImage()
display(i)

edge_x = cv2.filter2D(i, -1, sobel_x)
display(edge_x)
edge_y = cv2.filter2D(i, -1, sobel_y)
display(edge_y)
plt.show()