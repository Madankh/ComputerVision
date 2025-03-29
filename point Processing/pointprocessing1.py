from PIL import Image

#load the image using PIL
def load_image(file_path):
    image = Image.open(file_path)
    image = image.convert('RGB')
    return image

# Display the image 
def show_image(image, title):
    image.show(title=title)

# Function to darken an image
def darken(image, value=128):
    # Point processing: decrease brightness of each pixel
    darkened_image = image.point(lambda p:max(0, p - value))
    return darkened_image

def lighten(image, value=128):
    # Point processing: decrease brightness of each pixel
    lightened_image  = image.point(lambda p:min(255, p + value))
    return lightened_image

# Function to invert an image
def invert(image):
    # Point processing: invert each pixel (255 - pixel value)
    inverted_image = image.point(lambda p: 255 - p)
    return inverted_image

# Example usage:
image_path = 'point Processing/blurred_image.jpg'  # Replace with the path to your image

# Load the image
original_image = load_image(image_path)

# Perform operations
darkened_image = darken(original_image)
lightened_image = lighten(original_image)
inverted_image = invert(original_image)

# Show the results (if using an environment where image display is possible)
show_image(darkened_image, title="Darkened Image")
show_image(lightened_image, title="Lightened Image")
show_image(inverted_image, title="Inverted Image")

# Save the results if needed
darkened_image.save('darkened_image.jpg')
lightened_image.save('lightened_image.jpg')
inverted_image.save('inverted_image.jpg')