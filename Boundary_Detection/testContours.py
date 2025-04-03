import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage import filters

def active_contour_from_scratch(image, initial_contour, alpha=0.1, beta=0.1, gamma=0.1, 
                               max_iterations=100, convergence_threshold=0.5):
    """
    Implement active contour (snake) algorithm from scratch.
    
    Parameters:
    - image: Grayscale input image
    - initial_contour: Initial contour points as Nx2 array of (x,y) coordinates
    - alpha: Weight of elasticity energy
    - beta: Weight of smoothness energy
    - gamma: Weight of external energy (image force)
    - max_iterations: Maximum number of iterations
    - convergence_threshold: Threshold for convergence check
    
    Returns:
    - contour: Final contour points
    - all_contours: List of contours at each iteration
    """
    # Convert image to float
    image = image.astype(float)
    
    # Create edge map for external energy
    # Using gradient magnitude as external energy source
    sobel_x = filters.sobel_h(image)
    sobel_y = filters.sobel_v(image)
    edge_map = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Blur the edge map to extend the capture range
    edge_map = gaussian_filter(edge_map, sigma=2)
    
    # Negate the edge map to create a force that attracts to edges
    external_energy = -edge_map
    
    # Precompute external force field (gradient of external energy)
    fx = filters.sobel_h(external_energy)
    fy = filters.sobel_v(external_energy)
    
    # Current contour points
    contour = initial_contour.copy()
    num_points = len(contour)
    
    # Store all contours for visualization
    all_contours = [contour.copy()]
    
    # Create matrix A for internal energy (elasticity and smoothness)
    # This implements the pentadiagonal matrix for the internal energy derivatives
    A = np.zeros((num_points, num_points))
    
    # Fill the pentadiagonal matrix
    for i in range(num_points):
        A[i, i] = 2*alpha + 6*beta
        A[i, (i+1) % num_points] = -alpha - 4*beta
        A[i, (i-1) % num_points] = -alpha - 4*beta
        A[i, (i+2) % num_points] = beta
        A[i, (i-2) % num_points] = beta
    
    # Add identity matrix scaled by gamma to make matrix invertible
    A = A + gamma * np.eye(num_points)
    
    # Precompute inverse of A for efficiency
    A_inv = np.linalg.inv(A)
    
    # Main loop
    for iteration in range(max_iterations):
        # Store previous contour for convergence check
        previous_contour = contour.copy()
        
        # Compute external forces at each contour point
        fx_interp = np.array([fx[int(y), int(x)] for x, y in contour])
        fy_interp = np.array([fy[int(y), int(x)] for x, y in contour])
        
        # Update x coordinates
        b_x = gamma * contour[:, 0] + fx_interp
        contour[:, 0] = A_inv @ b_x
        
        # Update y coordinates
        b_y = gamma * contour[:, 1] + fy_interp
        contour[:, 1] = A_inv @ b_y
        
        # Ensure contour points stay within image boundaries
        contour[:, 0] = np.clip(contour[:, 0], 0, image.shape[1] - 1)
        contour[:, 1] = np.clip(contour[:, 1], 0, image.shape[0] - 1)
        
        # Store current contour
        all_contours.append(contour.copy())
        
        # Check convergence: calculate total movement
        total_movement = np.sum(np.sqrt(np.sum((contour - previous_contour)**2, axis=1)))
        print(f"Iteration {iteration+1}, Movement: {total_movement:.4f}")
        
        if total_movement < convergence_threshold:
            print(f"Converged after {iteration+1} iterations")
            break
    
    return contour, all_contours

def visualize_contour_evolution(image, all_contours, frequency=5):
    """
    Visualize the evolution of the contour during the snake algorithm.
    
    Parameters:
    - image: Input image
    - all_contours: List of contours at each iteration
    - frequency: Frequency of contours to display
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap='gray')
    
    # Plot initial contour in blue
    plt.plot(all_contours[0][:, 0], all_contours[0][:, 1], 'b--', lw=1, label='Initial')
    
    # Plot intermediate contours in green with decreasing opacity
    num_contours = len(all_contours)
    for i in range(1, num_contours-1, frequency):
        alpha = 0.2 + 0.6 * i / (num_contours - 1)
        plt.plot(all_contours[i][:, 0], all_contours[i][:, 1], 'g-', lw=1, alpha=alpha)
    
    # Plot final contour in red
    plt.plot(all_contours[-1][:, 0], all_contours[-1][:, 1], 'r-', lw=2, label='Final')
    
    plt.legend()
    plt.title('Active Contour Evolution')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Example usage with a sample image
def run_example():
    from skimage import data
    from skimage.color import rgb2gray
    
    # Load sample image
    image = rgb2gray(data.astronaut())
    
    # Create an initial contour (circle)
    center = np.array([220, 100])
    radius = 100
    theta = np.linspace(0, 2*np.pi, 100)
    initial_contour = np.array([center[0] + radius*np.cos(theta), 
                               center[1] + radius*np.sin(theta)]).T
    
    # Run active contour algorithm
    final_contour, all_contours = active_contour_from_scratch(
        image, initial_contour, alpha=0.1, beta=0.1, gamma=0.2, 
        max_iterations=250, convergence_threshold=0.5
    )
    
    # Visualize results
    visualize_contour_evolution(image, all_contours, frequency=5)
    
    # Final result
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap='gray')
    plt.plot(final_contour[:, 0], final_contour[:, 1], 'r-', lw=2)
    plt.title('Final Active Contour Result')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_example()