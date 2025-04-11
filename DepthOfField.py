import numpy as np
import cv2

def calculate_depth_of_field(focal_length, f_number, circle_of_confusion, distance_to_subject):
    # Calculate the hyperfocal distance
    hyperfocal_distance = (focal_length ** 2) / (f_number * circle_of_confusion)
    # Calculate the near point
    o1 = distance_to_subject * (hyperfocal_distance - focal_length) / (
        hyperfocal_distance + distance_to_subject - 2 * focal_length
    )
    # Calculate the far point
    if hyperfocal_distance - distance_to_subject == 0:  # Avoid division by zero
        o2 = float('inf')
    else:
        o2 = distance_to_subject * (hyperfocal_distance - focal_length) / (
            hyperfocal_distance - distance_to_subject
        )
    return (max(focal_length, o1), o2)  # Ensure o1 > focal_length

def depth_of_field(image, depth, f, s, N, c, k):
    """
    Apply depth of field effect using the depth of field range.
    
    Parameters:
    - image: numpy array of shape (H, W, 3), RGB image with values in [0, 255]
    - depth: numpy array of shape (H, W), depth map in same units as f and s
    - f: focal length (e.g., meters)
    - s: focus distance (e.g., meters)
    - N: f-number (e.g., 2.8)
    - c: circle of confusion (e.g., 0.00002 meters)
    - k: scale factor to convert blur to pixel radius
    
    Returns:
    - output: blurred image with depth of field effect
    """
    H, W, _ = image.shape  # Correctly unpack 3D shape
    output = np.zeros_like(image, dtype=np.float32)

    o1, o2 = calculate_depth_of_field(f, N, c, s)
    print(f"Depth of field range: {o1} to {o2}")  # For debugging

    for i in range(H):
        for j in range(W):
            u = depth[i, j]

            if o1 <= u <= o2:
                output[i, j] = image[i, j]
            else:
                # Corrected circle of confusion formula
                c_u = (f ** 2 / (N * (s - f))) * abs(u - s) / u
                r = k * c_u
                r_int = int(np.ceil(r))

                # Define neighborhood bounds, clipped to image edges
                p_min = max(0, i - r_int)
                p_max = min(H, i + r_int + 1)
                q_min = max(0, j - r_int)
                q_max = min(W, j + r_int + 1)

                # Extract sub-image
                sub_image = image[p_min:p_max, q_min:q_max]

                # Create coordinate grid with 'ij' indexing
                P, Q = np.meshgrid(
                    np.arange(p_min, p_max), np.arange(q_min, q_max), indexing='ij'
                )

                # Compute mask for pixels within the blur disk
                mask = ((P - i) ** 2 + (Q - j) ** 2) <= r ** 2

                # Average colors within the disk
                if np.any(mask):
                    blurred_pixel = np.mean(sub_image[mask], axis=0)
                else:
                    blurred_pixel = image[i, j]
                output[i, j] = blurred_pixel

    output = np.clip(output, 0, 255).astype(np.uint8)
    return output

# Example usage
if __name__ == "__main__":
    # Load an example image and create a synthetic depth map
    image = cv2.imread("111.jpg")
    if image is None:
        print("Error: Unable to load image '111.jpg'. Check the file path.")
        exit()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    H, W, _ = image.shape

    # Synthetic depth map
    depth = np.linspace(1, 10, W).astype(np.float32)
    depth = np.tile(depth, (H, 1))

    # Parameters
    f = 0.05  # focal length in meters
    s = 2.0   # focus distance in meters
    N = 2.8   # f-number
    c = 0.00002  # circle of confusion in meters
    k = 1000  # scale factor

    # Apply depth of field
    result = depth_of_field(image, depth, f, s, N, c, k)

    # Convert back to BGR for saving with OpenCV
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.imwrite("depth_of_field.jpg", result_bgr)
    print("Image saved as 'depth_of_field.jpg'.")