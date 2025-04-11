import cv2
import numpy as np

def calculate_dof_range(focal_length, f_number, coc, focus_distance):
    """
    Calculate depth of field range using basic lens equations.
    Returns near and far points of acceptable sharpness.
    """
    # Avoid division by zero and negative values
    if any(x <= 0 for x in [focal_length, f_number, coc, focus_distance]):
        raise ValueError("All parameters must be positive")
        
    hyperfocal = (focal_length ** 2) / (f_number * coc) + focal_length
    
    # Calculate near point
    near = (focus_distance * (hyperfocal - focal_length)) / \
          (hyperfocal + focus_distance - 2 * focal_length)
          
    # Calculate far point
    if hyperfocal <= focus_distance:
        far = float('inf')
    else:
        far = (focus_distance * (hyperfocal - focal_length)) / \
             (hyperfocal - focus_distance)
    
    return max(focal_length, near), far

def apply_depth_of_field(image_path, depth_map=None, focal_length=0.05, 
                        focus_distance=2.0, f_number=2.8, blur_strength=30):
    """
    Apply depth of field effect using OpenCV.
    
    Parameters:
    - image_path: path to input image
    - depth_map: optional depth map (if None, creates synthetic one)
    - focal_length: lens focal length in meters
    - focus_distance: focus distance in meters
    - f_number: aperture f-number
    - blur_strength: strength of the blur effect
    """
    # Read and check image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    
    height, width = image.shape[:2]
    
    # Create synthetic depth map if none provided
    if depth_map is None:
        depth_map = np.linspace(1, 10, width, dtype=np.float32)
        depth_map = np.tile(depth_map, (height, 1))
    
    # Calculate depth of field range
    near, far = calculate_dof_range(
        focal_length, f_number, 0.00002, focus_distance
    )
    print(f"Depth of field range: {near:.2f}m to {far:.2f}m")
    
    # Create masks for different depth regions
    in_focus_mask = cv2.inRange(
        depth_map, 
        near, 
        float('inf') if far == float('inf') else far
    )
    
    # Create blur masks based on distance from focus plane
    blur_amount = np.abs(depth_map - focus_distance)
    blur_amount = cv2.normalize(blur_amount, None, 0, 1, cv2.NORM_MINMAX)
    blur_amount = (blur_amount * blur_strength).astype(np.uint8)
    
    # Initialize output image
    output = image.copy()
    
    # Apply varying blur based on depth
    max_blur = blur_strength
    for blur_size in range(1, max_blur + 1, 2):
        # Create mask for current blur level
        level_mask = cv2.inRange(blur_amount, blur_size - 1, blur_size + 1)
        if not cv2.countNonZero(level_mask):
            continue
            
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(
            image, 
            (blur_size * 2 + 1, blur_size * 2 + 1), 
            0
        )
        
        # Combine blurred regions
        output = cv2.bitwise_and(output, output, mask=cv2.bitwise_not(level_mask))
        output += cv2.bitwise_and(blurred, blurred, mask=level_mask)
    
    # Keep in-focus regions sharp
    output = cv2.bitwise_and(output, output, mask=cv2.bitwise_not(in_focus_mask))
    output += cv2.bitwise_and(image, image, mask=in_focus_mask)
    
    return output

def main():
    # Example usage
    image_path = "111.jpg"
    
    # Camera parameters
    params = {
        'focal_length': 0.05,    # 50mm lens
        'focus_distance': 2.0,   # Focus at 2 meters
        'f_number': 2.8,         # f/2.8 aperture
        'blur_strength': 30      # Maximum blur radius in pixels
    }
    
    try:
        # Apply depth of field effect
        result = apply_depth_of_field(image_path, **params)
        
        # Save result
        cv2.imwrite("depth_of_field_opencv.jpg", result)
        print("Successfully saved depth_of_field_opencv.jpg")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()