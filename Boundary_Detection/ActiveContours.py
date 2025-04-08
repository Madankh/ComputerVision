import cv2 as cv2
import numpy as np

def gaussian_filter(edge , sigma):
    kernel = np.fromfunction(
        lambda x, y : (1/(2**np.pi * sigma**2)) * 
        np.exp(-((x - (edge-1)/2)**2 + (y - (edge - 1 )/2)**2)/ (2 * sigma**2)),
        (edge, edge)
    )
    return kernel / np.sum(kernel)

def active_contour_from_scratch(image, initial_contour, alpha=0.1, beta=0.1, gamma=0.1,
                                max_iterations=100, convergence_threshold=0.5):
    image = image.astype(float)
    sobel_x = filter.sobel_h(image)
    # sobel edge detection
    sobel_x = np.array([[-1,-2,-1],
                        [0,0,0],
                       [1,2,1]])
    
    sobel_y = np.array([[-1,0,1],
                        [-2,0,2],
                        [-1,0,1]])
    
    edge_x = cv2.filter2D(image, -1, sobel_x)
    edge_y = cv2.filter2D(image, -1, sobel_y)
    edge_map = np.sqrt(sobel_x**2 + sobel_y**2)

    # Blur the edge map to extend the capture range
    edge_map = gaussian_filter(edge_map, sigma=2)

    external_energy = -edge_map

    fx = edge_x(external_energy)
    fy = edge_y(external_energy)

    # Current contour points
    contour = initial_contour.copy()
    num_points = len(contour)

    all_contours = [contour.copy()]

    A = np.zeros(num_points):
    