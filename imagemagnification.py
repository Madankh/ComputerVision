import numpy as np
import matplotlib.pyplot as plt

def calculate_magnification(focal_length, object_distance):
    """
    Calculate the magnification of an optical system
    
    Args:
    focal_length (float): Focal length of the lens
    object_distance (float): Distance of object from the lens
    
    Returns:
    float: Magnification value
    float: Image distance
    bool: Whether image is inverted
    """
    # Calculate image distance using the lens equation: 1/f = 1/do + 1/di
    image_distance = (focal_length * object_distance) / (object_distance - focal_length)
    
    # Calculate magnification: m = -di/do = f/(zo)
    magnification = -image_distance / object_distance
    
    # Determine if image is inverted (negative magnification)
    is_inverted = magnification < 0
    
    return magnification, image_distance, is_inverted

def plot_optical_system(focal_length, object_distance, object_height=1):
    """
    Plot the optical system showing object and image formation
    """
    magnification, image_distance, is_inverted = calculate_magnification(focal_length, object_distance)
    image_height = magnification * object_height
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Draw optical axis
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    # Draw lens
    ax.axvline(x=0, color='b', label='Lens', linewidth=2)
    
    # Draw object
    ax.arrow(-object_distance, 0, 0, object_height, color='g', label='Object')
    
    # Draw image
    ax.arrow(image_distance, 0, 0, image_height, color='r', label='Image')
    
    # Add focal points
    ax.plot(focal_length, 0, 'ko', label='Focal point')
    ax.plot(-focal_length, 0, 'ko')
    
    # Set plot properties
    ax.set_xlim([-object_distance*1.5, max(image_distance*1.5, focal_length*2)])
    ax.set_ylim([-max(abs(object_height), abs(image_height))*1.5, 
                 max(abs(object_height), abs(image_height))*1.5])
    ax.grid(True)
    ax.legend()
    ax.set_title(f'Magnification: {magnification:.2f}x {"(Inverted)" if is_inverted else ""}')
    ax.set_xlabel('Distance from lens (units)')
    ax.set_ylabel('Height (units)')
    
    plt.show()

# Example usage
focal_length = 10  # units
object_distances = [15, 20, 30]  # units

# Show effect of different object distances
for do in object_distances:
    m, di, inv = calculate_magnification(focal_length, do)
    print(f"\nObject distance: {do} units")
    print(f"Magnification: {m:.2f}x")
    print(f"Image distance: {di:.2f} units")
    print(f"Image is {'inverted' if inv else 'upright'}")
    
    plot_optical_system(focal_length, do)