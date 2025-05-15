import cv2
import numpy as np
import glob
import os
import imutils

def stitch_images():
    # Use a more reliable path construction
    folder_path = os.path.join(os.getcwd(), "Image_Stitching/images")
    print(f"Looking in folder: {folder_path}")
    
    # Find all jpg images in the folder
    image_paths = glob.glob(os.path.join(folder_path, "*.jpg"))
    print(f"Image paths found: {image_paths}")
    
    # Load all images
    images = []
    for image_path in image_paths:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to read: {image_path}")
            continue
        images.append(img)
    
    print(f"Number of valid images: {len(images)}")
    
    # Check if we have enough images
    if len(images) < 2:
        print("Need at least two valid images to stitch.")
        return None
    
    # Create a stitcher and stitch the images
    stitcher = cv2.Stitcher_create()
    status, stitched_img = stitcher.stitch(images)
    
    # Check if stitching was successful
    if status != cv2.Stitcher_OK:
        print(f"Stitching failed with error code: {status}")
        return None
    
    # Display the initial stitched image
    cv2.imshow("Original Stitched Image", stitched_img)
    cv2.waitKey(0)
    
    # Process the stitched image to remove unwanted borders
    stitched_processed = process_stitched_image(stitched_img)
    
    if stitched_processed is not None:
        # Save and display the final result
        cv2.imwrite("stitchedOutput.png", stitched_processed)
        cv2.imshow("Final Stitched Image", stitched_processed)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return stitched_processed
    
    return stitched_img

def process_stitched_image(stitched_img):
    try:
        # Add a border to help with finding contours
        stitched_img = cv2.copyMakeBorder(stitched_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0))
        
        # Convert to grayscale and threshold
        gray = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2GRAY)
        _, thresh_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
        
        # Show the threshold image
        cv2.imshow("Threshold Image", thresh_img)
        cv2.waitKey(0)
        
        # Find contours
        contours = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        
        if not contours:
            print("No contours found")
            return stitched_img
        
        # Find the largest contour
        areaOI = max(contours, key=cv2.contourArea)
        
        # Create mask and find minimum rectangle
        mask = np.zeros(thresh_img.shape, dtype="uint8")
        x, y, w, h = cv2.boundingRect(areaOI)
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
        
        # Show the mask with rectangle
        cv2.imshow("Initial Mask", mask)
        cv2.waitKey(0)
        
        minRectangle = mask.copy()
        sub = mask.copy()
        
        # Erode until we find the minimum rectangle
        while cv2.countNonZero(sub) > 0:
            minRectangle = cv2.erode(minRectangle, None)
            sub = cv2.subtract(minRectangle, thresh_img)
        
        # Find contours in the minimum rectangle
        contours = cv2.findContours(minRectangle.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        
        if not contours:
            print("No contours found in minimum rectangle")
            return stitched_img
        
        # Find the largest contour
        areaOI = max(contours, key=cv2.contourArea)
        
        # Show the minimum rectangle
        cv2.imshow("Minimum Rectangle", minRectangle)
        cv2.waitKey(0)
        
        # Create a bounding rectangle and crop the image
        x, y, w, h = cv2.boundingRect(areaOI)
        stitched_processed = stitched_img[y:y+h, x:x+w]
        
        # Show the cropped result
        cv2.imshow("Cropped Result", stitched_processed)
        cv2.waitKey(0)
        
        return stitched_processed
    
    except Exception as e:
        print(f"Error in processing stitched image: {e}")
        return stitched_img

if __name__ == "__main__":
    stitch_images()